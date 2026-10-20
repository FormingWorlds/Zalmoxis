"""Parity test: get_tdep_density_jax vs numpy reference.

Uses the real Stage-1b PALEOS-2phase:MgSiO3 mantle EOS (two sub-tables:
solid_mantle and melted_mantle) and the Stixrude14 solidus/liquidus
curves. Queries both implementations at random (P, T) points covering
below-solidus, in-mushy, and above-liquidus regions.

Requires max_rel <= 1e-8 — FP-rounding precision on float64.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def _load_stage1b_mantle():
    """Load Stage-1b PALEOS-2phase mantle (solid + melted sub-tables).

    The mantle's sub-tables have ``format='paleos'``; they're loaded
    lazily via ``get_tabulated_eos`` into the shared interp_cache. We
    trigger both loads by querying once from each phase.
    """
    from zalmoxis.config import (
        load_material_dictionaries,
        load_solidus_liquidus_functions,
        load_zalmoxis_config,
    )
    from zalmoxis.eos.seager import get_tabulated_eos

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = os.path.join(here, 'data', 'bench_performance.toml')
    config_params = load_zalmoxis_config(cfg)
    mat_dicts = load_material_dictionaries()
    mantle_name = config_params['layer_eos_config']['mantle']
    mat = mat_dicts.get(mantle_name)
    if mat is None or 'melted_mantle' not in mat or 'solid_mantle' not in mat:
        pytest.skip(f'expected PALEOS-2phase layout in mantle {mantle_name}')

    # Resolve the sub-table file paths via the PALEOS-API shim if needed
    from zalmoxis.eos.paleos_api_cache import resolve_registry_entry

    resolve_registry_entry(mat)

    sol_file = mat['solid_mantle']['eos_file']
    liq_file = mat['melted_mantle']['eos_file']
    if not (os.path.isfile(sol_file) and os.path.isfile(liq_file)):
        pytest.skip(f'sub-table file(s) not found: {sol_file}, {liq_file}')

    interp_cache = {}
    # Trigger the paleos-format loader for both sub-tables
    _ = get_tabulated_eos(1e10, mat, 'solid_mantle', 2500.0, interp_cache)
    _ = get_tabulated_eos(1e10, mat, 'melted_mantle', 5000.0, interp_cache)
    sol_cached = interp_cache[sol_file]
    liq_cached = interp_cache[liq_file]

    sol_func, liq_func = load_solidus_liquidus_functions(
        config_params['layer_eos_config'],
        config_params.get('rock_solidus', 'Stixrude14-solidus'),
        config_params.get('rock_liquidus', 'Stixrude14-liquidus'),
    )

    return mat, sol_cached, liq_cached, sol_func, liq_func, interp_cache


def _extract_sub_args(cached, prefix):
    """Flatten a sub-table cache dict into jax kwargs with given prefix."""
    return {
        f'{prefix}_density_grid': np.asarray(cached['density_grid'], dtype=float),
        f'{prefix}_unique_log_p': np.asarray(cached['unique_log_p'], dtype=float),
        f'{prefix}_unique_log_t': np.asarray(cached['unique_log_t'], dtype=float),
        f'{prefix}_logp_min': float(cached['logp_min']),
        f'{prefix}_logt_min': float(cached['logt_min']),
        f'{prefix}_dlog_p': float(cached['dlog_p']),
        f'{prefix}_dlog_t': float(cached['dlog_t']),
        f'{prefix}_n_p': int(cached['n_p']),
        f'{prefix}_n_t': int(cached['n_t']),
        f'{prefix}_p_min': float(cached['p_min']),
        f'{prefix}_p_max': float(cached['p_max']),
        f'{prefix}_lt_min_per_p': np.asarray(cached['logt_valid_min'], dtype=float),
        f'{prefix}_lt_max_per_p': np.asarray(cached['logt_valid_max'], dtype=float),
    }


@pytest.mark.unit
def test_get_tdep_density_parity_vs_numpy():
    """JAX PALEOS-2phase density matches numpy across (P, T) query points."""
    from zalmoxis.eos.tdep import get_Tdep_density
    from zalmoxis.jax_eos.tdep import get_tdep_density_jax

    mat, sol_cached, liq_cached, sol_func, liq_func, interp_cache = _load_stage1b_mantle()

    jax_args = {}
    jax_args.update(_extract_sub_args(sol_cached, 'sol'))
    jax_args.update(_extract_sub_args(liq_cached, 'liq'))

    rng = np.random.default_rng(23)

    # Sample query points inside both tables' P range
    p_min = max(sol_cached['p_min'], liq_cached['p_min'])
    p_max = min(sol_cached['p_max'], liq_cached['p_max'])
    q_p = 10.0 ** rng.uniform(np.log10(p_min) + 0.2, np.log10(p_max) - 0.2, 400)
    q_t = rng.uniform(1800.0, 8000.0, 400)

    numpy_vals = []
    jax_vals = []
    for i in range(400):
        T_sol = float(sol_func(q_p[i]))
        T_liq = float(liq_func(q_p[i]))
        nv = get_Tdep_density(q_p[i], q_t[i], mat, sol_func, liq_func, interp_cache)
        numpy_vals.append(nv if nv is not None else np.nan)
        jv = float(
            get_tdep_density_jax(
                q_p[i],
                q_t[i],
                T_sol,
                T_liq,
                **jax_args,
            )
        )
        jax_vals.append(jv)
    numpy_vals = np.asarray(numpy_vals)
    jax_vals = np.asarray(jax_vals)

    both_finite = np.isfinite(numpy_vals) & np.isfinite(jax_vals)
    nan_mismatch = np.isfinite(numpy_vals) ^ np.isfinite(jax_vals)
    nan_mismatch_count = int(np.sum(nan_mismatch))

    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.abs(numpy_vals[both_finite] - jax_vals[both_finite]) / np.maximum(
            np.abs(numpy_vals[both_finite]),
            1e-30,
        )
    max_rel = float(rel.max()) if rel.size > 0 else 0.0
    print(
        f'n_pts_finite={int(both_finite.sum())}/400, '
        f'nan_mismatch={nan_mismatch_count}, max_rel={max_rel:.3e}'
    )

    assert max_rel <= 1e-8, f'Tdep parity failed: max_rel={max_rel:.3e} (want <=1e-8)'


@pytest.mark.unit
def test_get_tdep_density_parity_at_collapsed_boundary():
    """JAX and numpy agree at T_sol == T_liq (mzf = 1.0), and equality is solid.

    A mushy_zone_factor of 1.0 collapses the mushy band to zero width, so
    T_sol == T_liq at every pressure. Exactly on that boundary numpy routes
    T == T_liq to the solid table (get_Tdep_density: temperature <= T_sol).
    The JAX path must match: a strict ``temperature > T_liq`` is required, an
    inclusive ``>=`` selects the liquid table and breaks parity at equality.

    The test brackets the boundary at +/- 1e-9 (relative) and asserts:
    - numpy and JAX agree to float64 rounding at all three points;
    - on and just below the boundary both take the solid density, which
      differs from the just-above liquid density by the finite melting jump.
    """
    from zalmoxis.eos.tdep import get_Tdep_density
    from zalmoxis.jax_eos.tdep import get_tdep_density_jax

    mat, sol_cached, liq_cached, sol_func, liq_func, interp_cache = _load_stage1b_mantle()

    jax_args = {}
    jax_args.update(_extract_sub_args(sol_cached, 'sol'))
    jax_args.update(_extract_sub_args(liq_cached, 'liq'))

    p_min = max(sol_cached['p_min'], liq_cached['p_min'])
    p_max = min(sol_cached['p_max'], liq_cached['p_max'])
    pressures = 10.0 ** np.linspace(np.log10(p_min) + 0.3, np.log10(p_max) - 0.3, 12)

    for pressure in pressures:
        # Collapse the melting curve: solidus == liquidus at this pressure.
        t_star = float(liq_func(pressure))
        const_sol = lambda p, _t=t_star: _t  # noqa: E731
        const_liq = lambda p, _t=t_star: _t  # noqa: E731

        for frac in (1.0 - 1e-9, 1.0, 1.0 + 1e-9):
            temperature = frac * t_star
            nv = get_Tdep_density(pressure, temperature, mat, const_sol, const_liq, interp_cache)
            jv = float(get_tdep_density_jax(pressure, temperature, t_star, t_star, **jax_args))
            assert nv is not None and np.isfinite(jv)
            rel = abs(nv - jv) / max(abs(nv), 1e-30)
            assert rel <= 1e-8, (
                f'collapsed-boundary parity failed at P={pressure:.2e} Pa, '
                f'frac={frac}: numpy={nv:.6e}, jax={jv:.6e}, rel={rel:.3e}'
            )

        # The melting jump must be resolvable, else the equality branch is
        # not discriminating. Solid (on/below) differs from liquid (above).
        rho_on = float(get_tdep_density_jax(pressure, t_star, t_star, t_star, **jax_args))
        rho_below = float(
            get_tdep_density_jax(pressure, t_star * (1.0 - 1e-9), t_star, t_star, **jax_args)
        )
        rho_above = float(
            get_tdep_density_jax(pressure, t_star * (1.0 + 1e-9), t_star, t_star, **jax_args)
        )
        assert abs(rho_on - rho_below) / rho_below <= 1e-6, (
            f'equality did not take the solid branch at P={pressure:.2e} Pa'
        )
        assert abs(rho_on - rho_above) / rho_above > 1e-3, (
            f'melting jump not resolvable at P={pressure:.2e} Pa; test is not '
            f'discriminating (rho_on={rho_on:.6e}, rho_above={rho_above:.6e})'
        )
