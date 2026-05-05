"""Tests for the r-indexed temperature_arrays kwarg in the JAX path.

Covers:
  * JAX+Anderson converges at CHILI-like coupled params when the caller
    provides an r-indexed temperature profile (regression against the
    pre-fix failure where the P-indexed tabulation collapsed to a
    constant for P-ignoring callables; see
    ``tools/benchmarks/bench_coupled_tempfunc.py``).
  * A P-ignoring ``temperature_function`` and the equivalent
    ``temperature_arrays`` produce structure profiles that agree on
    R_planet, P_center, mass_enclosed, gravity, and pressure to within
    tight relative tolerance.
  * Providing both ``temperature_function`` AND ``temperature_arrays``
    silently lets arrays win (the inner solver loop always builds its
    own ``_temperature_func`` for density updates, so rejecting the
    pair would force callers to monkey-patch).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zalmoxis.config import (  # noqa: E402
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
)
from zalmoxis.solver import main  # noqa: E402

_CONFIG = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'data',
        'bench_performance.toml',
    )
)
_INPUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'input')
)


def _chili_like_profile(n=80):
    """Return (r_arr, T_arr) resembling a mid-evolution PROTEUS mantle.

    CMB-anchored linear T(r) from T_cmb=5500 K to T_surf=3830 K, matching
    the CHILI Earth default in ``chili_dry_coupled_stage1b_postrevert_baseline.toml``.
    """
    r_cmb, r_surf = 3.46e6, 7.06e6
    T_cmb, T_surf = 5500.0, 3830.0
    r = np.linspace(r_cmb, r_surf, n)
    T = T_cmb + (T_surf - T_cmb) * (r - r_cmb) / (r_surf - r_cmb)
    return r, T, r_cmb, T_cmb


def _load_fixtures():
    cp = load_zalmoxis_config(_CONFIG)
    cp['wall_timeout'] = 3600.0
    mat = load_material_dictionaries()
    mf = load_solidus_liquidus_functions(
        cp['layer_eos_config'],
        cp.get('rock_solidus', 'Stixrude14-solidus'),
        cp.get('rock_liquidus', 'Stixrude14-liquidus'),
    )
    return cp, mat, mf


@pytest.mark.unit
@pytest.mark.slow
def test_jax_anderson_converges_with_temperature_arrays():
    """JAX+Anderson must converge at CHILI params using temperature_arrays.

    Before this fix, the JAX wrapper's P-indexed tabulation collapsed
    for P-ignoring callables (T sampled at r_mid becomes constant), so
    the ODE saw a flat 4665 K column and the brentq outer loop never
    bracketed. ``temperature_arrays`` swaps the RHS to an r-indexed
    axis and restores convergence.
    """
    cp, mat, mf = _load_fixtures()
    cp['use_jax'] = True
    cp['use_anderson'] = True

    r_arr, T_arr, _, _ = _chili_like_profile()
    result = main(
        cp,
        mat,
        mf,
        _INPUT_DIR,
        temperature_arrays=(r_arr, T_arr),
    )

    assert result['converged'], (
        f'JAX+Anderson failed to converge with temperature_arrays at CHILI '
        f'params (mass={result["converged_mass"]}, '
        f'density={result["converged_density"]}, '
        f'pressure={result["converged_pressure"]})'
    )
    # Sanity: the returned planet radius must be Earth-like.
    R = float(result['radii'][-1])
    assert 6.0e6 < R < 8.0e6, f'unphysical planet radius {R:.3e} m'


@pytest.mark.unit
@pytest.mark.slow
@pytest.mark.timeout(600)
def test_parity_numpy_function_vs_jax_arrays():
    """Numpy+temperature_function vs JAX+Anderson+temperature_arrays agree.

    Runs the same physical setup through two independent paths and
    requires the structure-defining quantities (R, P_center, total mass,
    surface gravity) to agree to within 1e-3 relative tolerance. The
    numpy path uses a P-ignoring closure; the JAX path uses the
    equivalent (r, T) arrays. The stored ``temperature`` profile is
    intentionally not compared because the two paths populate it via
    different internal helpers (post-hoc label, not a physics output).
    """
    cp, mat, mf = _load_fixtures()

    r_arr, T_arr, r_cmb, T_cmb = _chili_like_profile()

    def tf(r, P):
        if r <= r_cmb:
            return float(T_cmb)
        return float(np.interp(r, r_arr, T_arr))

    # Numpy reference
    cp_np = dict(cp)
    cp_np.pop('use_jax', None)
    cp_np.pop('use_anderson', None)
    r_np = main(cp_np, mat, mf, _INPUT_DIR, temperature_function=tf)
    assert r_np['converged']

    # JAX + Anderson + temperature_arrays
    cp_jx = dict(cp)
    cp_jx['use_jax'] = True
    cp_jx['use_anderson'] = True
    r_jx = main(cp_jx, mat, mf, _INPUT_DIR, temperature_arrays=(r_arr, T_arr))
    assert r_jx['converged']

    # Physics-critical quantities.
    R_np = float(r_np['radii'][-1])
    R_jx = float(r_jx['radii'][-1])
    assert R_np == pytest.approx(R_jx, rel=1e-3)

    P_np = float(r_np['pressure'][0])
    P_jx = float(r_jx['pressure'][0])
    assert P_np == pytest.approx(P_jx, rel=1e-3)

    M_np = float(r_np['mass_enclosed'][-1])
    M_jx = float(r_jx['mass_enclosed'][-1])
    assert M_np == pytest.approx(M_jx, rel=1e-3)

    g_np = float(r_np['gravity'][-1])
    g_jx = float(r_jx['gravity'][-1])
    assert g_np == pytest.approx(g_jx, rel=1e-3)


@pytest.mark.unit
@pytest.mark.slow
def test_arrays_override_function_when_both_passed():
    """When both kwargs are provided, temperature_arrays wins silently.

    Regression: an earlier draft raised ValueError when both were
    present, but Zalmoxis' inner solver always constructs a
    ``_temperature_func`` for density updates (linear guess / adiabat
    blend) independent of the caller's T source. That made the strict
    check fire on every call for callers that ONLY want to use arrays.
    The wrapper now lets arrays take precedence.
    """
    cp, mat, mf = _load_fixtures()
    cp['use_jax'] = True
    cp['use_anderson'] = True

    r_arr, T_arr, r_cmb, T_cmb = _chili_like_profile()

    def tf(r, P):
        # Deliberately wrong: constant 99 K. If the wrapper accidentally
        # picked this path the solution would diverge wildly.
        return 99.0

    result = main(
        cp,
        mat,
        mf,
        _INPUT_DIR,
        temperature_function=tf,
        temperature_arrays=(r_arr, T_arr),
    )

    assert result['converged'], (
        'JAX path must accept both kwargs with arrays taking precedence; '
        'convergence failure here means temperature_function was used.'
    )
    R = float(result['radii'][-1])
    # If the 99 K closure had been used, the structure would be
    # ludicrously dense; demand an Earth-like radius instead.
    assert 6.0e6 < R < 8.0e6
