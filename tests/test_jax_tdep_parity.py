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
    cfg = os.path.join(here, '..', 'input', 'bench_performance.toml')
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
