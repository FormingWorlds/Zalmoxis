"""A default-configuration solve is bit-identical to its stored reference: every array and
scalar of the result, against a reference that ``tools/default_solve_reference.py`` wrote, one
per platform and CPU model (macOS arm64 is stored; a CI runner has none, see below). The
references pin the adiabat with its T(P) table read by layer, the strict accept on a table of
a converged pass, the end-slope continuation and the CMB split of the node segment. The
converged radius depends on the CPU, within the mass tolerance of the outer loop. On a CPU
with no reference, such as the Linux runners of the nightly job, the solve meets the nearest
stored one within that tolerance instead, and a warning says that bit-identity was not
checked.
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
    A key in only one of the 2 results is reported alone, before any value is compared.
    """
    if set(got) != set(ref):
        return [f'keys {sorted(set(got) ^ set(ref))}']
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
def test_a_default_solve_is_bit_identical_to_its_reference():
    key = cpu_key()
    ref_file = DATA / f'default_solve_reference_{key}.npz'
    got = solve_arrays()
    stored = sorted(DATA.glob('default_solve_reference_*.npz'))
    keys = set(np.load(ref_file if ref_file.is_file() else stored[0]).files)
    assert set(got) == keys, f'{key}: keys not in both results: {sorted(set(got) ^ keys)}'
    assert not bool(got['stopped_on_wall_clock']), f'{key}: the solve stopped on its wall clock'
    if ref_file.is_file():
        ref = np.load(ref_file)
        differ = [k for k in ref.files if got[k].tobytes() != ref[k].tobytes()]
        assert not differ, f'{key}: entries not bit-identical to the reference: {differ}'
        return
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
    assert (DATA / f'default_solve_reference_{cpu_key("darwin", "arm64", "")}.npz').is_file()


@pytest.mark.unit
def test_the_tolerance_band_holds_the_reference_and_refuses_a_wider_gap():
    """A copy 1.9e-3 off the stored macOS result lies inside the band; 2.1e-3 off it fails."""
    mac = np.load(DATA / 'default_solve_reference_darwin-arm64.npz')
    got = dict(mac)  # the base of the changed copies below
    for k, scale, want in (
        ('radii', 1.0021, ['radii']),
        ('radii', 1.0019, []),
        ('p_center', 0.9979, ['p_center']),
        ('p_center', 0.9981, []),
        ('mass_enclosed', 1.0021, ['mass_enclosed']),
        ('mass_enclosed', 1.0019, []),
    ):
        bad = dict(got, **{k: mac[k] * scale})
        assert off_tolerance(bad, mac) == want, k
    assert off_tolerance(dict(got, mass_error_at_accept=np.float64(1.1e-3)), mac) == [
        'mass_error_at_accept'
    ]
    assert off_tolerance(dict(got, converged=np.bool_(False)), mac) == ['converged']
    cut = dict(got, stopped_on_wall_clock=np.bool_(True))
    assert off_tolerance(cut, mac) == ['stopped_on_wall_clock']
    assert off_tolerance(dict(got, p_center=np.float64(np.nan)), mac) == ['p_center']


@pytest.mark.unit
def test_a_result_key_missing_from_the_reference_or_the_solve_is_off():
    """A new or a lost result key fails the comparison on a host without a stored reference."""
    mac = np.load(DATA / 'default_solve_reference_darwin-arm64.npz')
    got = dict(mac)
    assert off_tolerance(dict(got, new_key=np.float64(1.0)), mac) == ["keys ['new_key']"]
    lost = {k: v for k, v in got.items() if k != 'density'}
    assert off_tolerance(lost, mac) == ["keys ['density']"]
    no_radii = {k: v for k, v in got.items() if k != 'radii'}  # a key the band reads
    assert off_tolerance(no_radii, mac) == ["keys ['radii']"]


@pytest.mark.unit
def test_a_lost_key_fails_the_default_solve_test_by_name(monkeypatch):
    """On a host without a stored reference a solve that lost a key the test reads ('radii',
    'stopped_on_wall_clock') fails on the key report, not on a KeyError."""
    import tests.test_default_invariance as mod

    mac = dict(np.load(DATA / 'default_solve_reference_darwin-arm64.npz'))
    monkeypatch.setattr(mod, 'cpu_key', lambda: 'linux-x86_64-no-reference')
    for lost in ('radii', 'stopped_on_wall_clock'):
        monkeypatch.setattr(
            mod, 'solve_arrays', lambda: {k: v for k, v in mac.items() if k != lost}
        )
        with pytest.raises(AssertionError, match=lost):
            mod.test_a_default_solve_is_bit_identical_to_its_reference()
