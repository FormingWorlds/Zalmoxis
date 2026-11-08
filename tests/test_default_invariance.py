"""A default-configuration solve is bit-identical to the one-layer core before the two-layer
core: every array and scalar of the result, against a reference that
``tools/default_solve_reference.py`` wrote with the one-layer code, one per platform and CPU
model (macOS arm64 locally, Linux x86_64 on an AMD EPYC 7763 runner). The converged radius
depends on the CPU, within the mass tolerance of the outer loop. On a CPU with no reference
the solve meets the nearest stored one within that tolerance instead, and the message says so.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from tools.default_solve_reference import cpu_key, solve_arrays

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'tests' / 'data'
TOLERANCE_OUTER = 1e-3  # the mass tolerance of the outer loop in the default solve


def off_tolerance(got, ref, tol=TOLERANCE_OUTER):
    """Entries of a converged solve outside the band that the mass tolerance `tol` allows.

    Two converged solves have masses up to 2 tol apart; the radius and the central pressure
    scale with the mass by powers below 1, so each differs by less than 2 tol in relative terms.
    """
    off = [k for k in ('converged', 'converged_mass') if not bool(got[k])]
    if bool(got['stopped_on_wall_clock']):
        off.append('stopped_on_wall_clock')
    if not abs(got['mass_enclosed'][-1] / ref['mass_enclosed'][-1] - 1) < 2 * tol:
        off.append('mass_enclosed')
    if not abs(got['mass_error_at_accept']) < tol:
        off.append('mass_error_at_accept')
    for k, x, y in (
        ('radii', got['radii'][-1], ref['radii'][-1]),
        ('p_center', got['p_center'], ref['p_center']),
    ):
        if not abs(x / y - 1) < 2 * tol:
            off.append(k)
    return off


@pytest.mark.integration
@pytest.mark.default_invariance
@pytest.mark.timeout(1800)
def test_a_default_solve_is_bit_identical_to_main():
    key = cpu_key()
    ref_file = DATA / f'default_solve_reference_{key}.npz'
    got = solve_arrays()
    assert not bool(got['stopped_on_wall_clock']), f'{key}: the solve stopped on its wall clock'
    if ref_file.is_file():
        ref = np.load(ref_file)
        assert set(got) == set(ref.files)
        differ = [k for k in ref.files if got[k].tobytes() != ref[k].tobytes()]
        assert not differ, f'{key}: entries not bit-identical to main: {differ}'
        return
    stored = sorted(DATA.glob('default_solve_reference_*.npz'))
    near = min(stored, key=lambda f: abs(np.load(f)['radii'][-1] / got['radii'][-1] - 1))
    off = off_tolerance(got, np.load(near))
    assert not off, (
        f'{key} has no stored reference, so bit-identity was not checked; against '
        f'{near.name} these entries lie outside the outer-loop tolerance: {off}'
    )
    warnings.warn(
        f'{key}: no stored reference, bit-identity not checked; {near.name} within tolerance'
    )


@pytest.mark.unit
def test_the_reference_key_names_the_cpu_model():
    epyc = 'processor\t: 0\nmodel name\t: AMD EPYC 7763 64-Core Processor\nflags\t: x\n'
    assert cpu_key('linux', 'x86_64', epyc) == 'linux-x86_64-AMD-EPYC-7763'
    xeon = 'model name\t: Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz\n'
    assert (
        cpu_key('linux', 'x86_64', xeon)
        == 'linux-x86_64-Intel-R-Xeon-R-Platinum-8370C-CPU-2-80GHz'
    )
    assert cpu_key('linux', 'aarch64', 'processor : 0\n') == 'linux-aarch64-unknown'
    assert cpu_key('darwin', 'arm64', 'model name : ignored') == 'darwin-arm64'
    assert (DATA / f'default_solve_reference_{cpu_key("linux", "x86_64", epyc)}.npz').is_file()


@pytest.mark.unit
def test_the_tolerance_band_holds_the_2_linux_cpus_and_refuses_a_wider_gap():
    """The EPYC 7763 result meets the macOS one inside the band; 2.1e-3 off the macOS one fails."""
    epyc = dict(np.load(DATA / 'default_solve_reference_linux-x86_64-AMD-EPYC-7763.npz'))
    mac = np.load(DATA / 'default_solve_reference_darwin-arm64.npz')
    assert off_tolerance(epyc, mac) == []
    for k, scale, want in (
        ('radii', 1.0021, ['radii']),
        ('p_center', 0.9979, ['p_center']),
        ('mass_enclosed', 1.0021, ['mass_enclosed']),
    ):
        bad = dict(epyc, **{k: mac[k] * scale})
        assert off_tolerance(bad, mac) == want, k
    assert off_tolerance(dict(epyc, mass_error_at_accept=np.float64(1.1e-3)), mac) == [
        'mass_error_at_accept'
    ]
    assert off_tolerance(dict(epyc, converged=np.bool_(False)), mac) == ['converged']
    cut = dict(epyc, stopped_on_wall_clock=np.bool_(True))
    assert off_tolerance(cut, mac) == ['stopped_on_wall_clock']
    assert off_tolerance(dict(epyc, p_center=np.float64(np.nan)), mac) == ['p_center']
