"""Parity test: coupled_odes_jax vs numpy coupled_odes on Stage-1b config.

Assemble the args the JAX RHS expects (cache contents for both core
and mantle sub-tables, pre-tabulated T(P) adiabat and melting curves),
query both implementations at random (r, y) state vectors spanning
core and mantle cells, and compare dy/dr at rtol <= 1e-6. The RHS
output depends on bilinear interps plus pure arithmetic, so FP-
rounding precision should hold.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def _stage1b_setup():
    """Load mantle + core caches + melting curves + adiabat; build jax_args."""
    from zalmoxis.config import (
        load_material_dictionaries,
        load_solidus_liquidus_functions,
        load_zalmoxis_config,
    )
    from zalmoxis.eos.interpolation import _ensure_unified_cache
    from zalmoxis.eos.paleos_api_cache import resolve_registry_entry
    from zalmoxis.eos.seager import get_tabulated_eos

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = os.path.join(here, 'data', 'bench_performance.toml')
    config_params = load_zalmoxis_config(cfg)
    mat_dicts = load_material_dictionaries()
    layer_cfg = config_params['layer_eos_config']

    # Core (PALEOS:iron unified)
    core_mat = mat_dicts[layer_cfg['core']]
    resolve_registry_entry(core_mat)
    if core_mat.get('format') != 'paleos_unified':
        pytest.skip(f'expected paleos_unified core, got {core_mat.get("format")}')
    core_file = core_mat['eos_file']
    if not os.path.isfile(core_file):
        pytest.skip(f'core EOS file missing: {core_file}')

    interp_cache = {}
    core_cached = _ensure_unified_cache(core_file, interp_cache)

    # Mantle (PALEOS-2phase:MgSiO3 = solid + melted sub-tables)
    mantle_mat = mat_dicts[layer_cfg['mantle']]
    resolve_registry_entry(mantle_mat)
    sol_file = mantle_mat['solid_mantle']['eos_file']
    liq_file = mantle_mat['melted_mantle']['eos_file']
    if not (os.path.isfile(sol_file) and os.path.isfile(liq_file)):
        pytest.skip('mantle sub-table files missing')
    _ = get_tabulated_eos(1e10, mantle_mat, 'solid_mantle', 2500.0, interp_cache)
    _ = get_tabulated_eos(1e10, mantle_mat, 'melted_mantle', 5000.0, interp_cache)
    sol_cached = interp_cache[sol_file]
    liq_cached = interp_cache[liq_file]

    sol_func, liq_func = load_solidus_liquidus_functions(
        layer_cfg,
        config_params.get('rock_solidus', 'Stixrude14-solidus'),
        config_params.get('rock_liquidus', 'Stixrude14-liquidus'),
    )

    # Tabulated mantle melting curves on a shared log-P axis, sampled
    # in log-T. The JAX RHS uses jnp.interp(log10(P), axis,
    # log_T_*_table) and exponentiates to recover T. Linear interp on
    # (log_P, log_T) is bit-exact for any single Simon-Glatzel power
    # law, so 256 samples reproduce Stix14 solidus/liquidus to machine
    # precision.
    _melt_log_p_axis = np.linspace(np.log10(1e8), np.log10(5e12), 256)
    _melt_p_axis = 10.0**_melt_log_p_axis
    _log_T_liq_table = np.log10(
        np.array([float(liq_func(P)) for P in _melt_p_axis], dtype=float)
    )
    _log_T_sol_table = np.log10(
        np.array([float(sol_func(P)) for P in _melt_p_axis], dtype=float)
    )

    # Adiabatic T(P) for this parity test: simple linear adiabat in log-P
    # matching a typical Stage-1b profile. Resolution fine enough that
    # jnp.interp errors are below 1e-8.
    T_logP_grid = np.linspace(5.0, 12.5, 2000)
    # Pick a plausible adiabat: surface=3000 K, center~8000 K monotone.
    T_values = 3000.0 + 5000.0 * (T_logP_grid - 5.0) / (12.5 - 5.0)

    cmb_mass = 0.325 * 5.972e24  # Earth-like CMF

    def extract_args(cached, prefix):
        out = {
            f'{prefix}_density_grid': np.asarray(cached['density_grid']),
            f'{prefix}_unique_log_p': np.asarray(cached['unique_log_p']),
            f'{prefix}_unique_log_t': np.asarray(cached['unique_log_t']),
            f'{prefix}_logp_min': float(cached['logp_min']),
            f'{prefix}_logt_min': float(cached['logt_min']),
            f'{prefix}_dlog_p': float(cached['dlog_p']),
            f'{prefix}_dlog_t': float(cached['dlog_t']),
            f'{prefix}_n_p': int(cached['n_p']),
            f'{prefix}_n_t': int(cached['n_t']),
            f'{prefix}_p_min': float(cached['p_min']),
            f'{prefix}_p_max': float(cached['p_max']),
            f'{prefix}_lt_min_per_p': np.asarray(cached['logt_valid_min']),
            f'{prefix}_lt_max_per_p': np.asarray(cached['logt_valid_max']),
        }
        return out

    jax_args = {
        'cmb_mass': float(cmb_mass),
        'T_axis_grid': T_logP_grid,
        'T_values': T_values,
        'T_surface': 3000.0,
        'mushy_zone_factor_core': 1.0,  # PALEOS:iron: no mushy (uses table's own phase)
    }
    # Core args
    jax_args.update(extract_args(core_cached, 'core'))
    # Core liquidus (may be present for paleos_unified)
    core_liq_lp = np.asarray(core_cached.get('liquidus_log_p', []), dtype=float)
    core_liq_lt = np.asarray(core_cached.get('liquidus_log_t', []), dtype=float)
    jax_args['core_liquidus_log_p'] = core_liq_lp
    jax_args['core_liquidus_log_t'] = core_liq_lt
    jax_args['core_liquidus_min_log_p'] = float(core_liq_lp[0]) if len(core_liq_lp) else 0.0
    jax_args['core_liquidus_max_log_p'] = float(core_liq_lp[-1]) if len(core_liq_lp) else 0.0
    jax_args['core_has_liquidus_f'] = 1.0 if len(core_liq_lp) else 0.0

    # Mantle sub-table args
    jax_args.update(extract_args(sol_cached, 'sol'))
    jax_args.update(extract_args(liq_cached, 'liq'))
    jax_args['melt_log_p_min'] = float(_melt_log_p_axis[0])
    jax_args['melt_dlog_p'] = float(_melt_log_p_axis[1] - _melt_log_p_axis[0])
    jax_args['melt_n'] = int(len(_melt_log_p_axis))
    jax_args['log_T_liq_table'] = _log_T_liq_table
    jax_args['log_T_sol_table'] = _log_T_sol_table
    # Zalmoxis's G constant (6.67428e-11) differs from CODATA 2018 (6.6743e-11)
    # at ~3e-6 relative; that mismatch propagates into dg/dr which has a
    # (4 pi G rho - 2 g/r) cancellation, amplifying the G error by ~100x.
    from zalmoxis.constants import G

    jax_args['G'] = G

    return {
        'config_params': config_params,
        'core_mat': core_mat,
        'mantle_mat': mantle_mat,
        'interp_cache': interp_cache,
        'mushy_zone_factors': {'PALEOS:iron': 1.0, 'PALEOS:MgSiO3': 1.0, 'PALEOS:H2O': 1.0},
        'sol_func': sol_func,
        'liq_func': liq_func,
        'cmb_mass': cmb_mass,
        'jax_args': jax_args,
        'T_axis_grid': T_logP_grid,
        'T_values': T_values,
    }


@pytest.mark.unit
def test_coupled_odes_jax_parity():
    """JAX coupled_odes matches numpy on 2-layer Stage-1b config."""
    from zalmoxis.jax_eos.rhs import coupled_odes_jax
    from zalmoxis.mixing import LayerMixture
    from zalmoxis.structure_model import coupled_odes

    setup = _stage1b_setup()
    cfg = setup['config_params']
    T_logP_grid = setup['T_axis_grid']
    T_values = setup['T_values']

    # Numpy side uses LayerMixture + material_dictionaries
    layer_mixtures = {
        'core': LayerMixture([cfg['layer_eos_config']['core']], [1.0]),
        'mantle': LayerMixture([cfg['layer_eos_config']['mantle']], [1.0]),
    }
    mat_dicts = {
        cfg['layer_eos_config']['core']: setup['core_mat'],
        cfg['layer_eos_config']['mantle']: setup['mantle_mat'],
    }

    def numpy_temp(r, P):
        if P <= 0:
            return 3000.0
        return float(np.interp(np.log10(max(P, 1.0)), T_logP_grid, T_values))

    rng = np.random.default_rng(113)

    # Query state vectors spanning core (small M) and mantle (M > cmb)
    M_planet = 5.972e24
    cmb_mass = setup['cmb_mass']
    R_planet_approx = 6.4e6

    numpy_dydr = []
    jax_dydr = []
    query_info = []
    for _ in range(200):
        r = rng.uniform(1e5, R_planet_approx)
        # Pick M either below or above cmb_mass so we cover both layers
        M = rng.uniform(0.0, M_planet)
        g = rng.uniform(0.5, 25.0)
        # Pressure: rough scaling from a 1 M_E planet structure
        P = rng.uniform(1e6, 3e11)

        y = np.array([M, g, P])
        # Numpy
        temperature = numpy_temp(r, P)
        nv = coupled_odes(
            r,
            y,
            cmb_mass,  # cmb_mass
            cmb_mass + 0.675 * M_planet,  # core_mantle_mass (no ice layer)
            layer_mixtures,
            setup['interp_cache'],
            mat_dicts,
            temperature,
            setup['sol_func'],
            setup['liq_func'],
            setup['mushy_zone_factors'],
        )
        numpy_dydr.append(nv)
        # JAX
        jv = coupled_odes_jax(r, y, **setup['jax_args'])
        jax_dydr.append(np.asarray(jv))
        query_info.append((r, M, g, P, temperature, M < cmb_mass))

    numpy_dydr = np.asarray(numpy_dydr, dtype=float)
    jax_dydr = np.asarray(jax_dydr, dtype=float)

    # Filter rows where numpy returned zeros (pressure_zero / NaN branch)
    # OR jax returned zeros; compare only where both produced non-zero
    both_nonzero = (np.abs(numpy_dydr).max(axis=1) > 0) & (np.abs(jax_dydr).max(axis=1) > 0)
    n_both = int(both_nonzero.sum())

    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.abs(numpy_dydr[both_nonzero] - jax_dydr[both_nonzero]) / np.maximum(
            np.abs(numpy_dydr[both_nonzero]),
            1e-30,
        )
    max_rel = float(rel.max()) if rel.size > 0 else 0.0
    print(f'n_both={n_both}/200, max_rel={max_rel:.3e}')

    # Diagnostic for bad points
    if max_rel > 1e-6:
        row_max = rel.max(axis=1)
        worst_idx = int(np.argmax(row_max))
        # Re-map back to the original 200 queries (both_nonzero mask selection)
        orig_indices = np.where(both_nonzero)[0]
        orig_i = int(orig_indices[worst_idx])
        r_, M_, g_, P_, T_, is_core = query_info[orig_i]
        print(f'  worst original idx={orig_i}')
        print(
            f'    r={r_:.3e}, M={M_:.3e}, g={g_:.3e}, P={P_:.3e}, T={T_:.3f}, is_core={is_core}'
        )
        print(f'    numpy_dydr={numpy_dydr[orig_i]}')
        print(f'    jax_dydr={jax_dydr[orig_i]}')
        print(f'    rel={rel[worst_idx]}')

    assert max_rel <= 1e-6, f'RHS parity failed: max_rel={max_rel:.3e} (want <=1e-6)'
