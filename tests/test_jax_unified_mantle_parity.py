"""Parity tests: JAX unified-mantle RHS vs numpy coupled_odes (issue #75).

The production-default mantle (PALEOS:MgSiO3) is a single unified table
with the solidus derived internally from the table's own liquidus and
the mushy zone factor. Before the mantle_is_unified port, the JAX
wrapper rejected it and every default-config solve fell back to numpy.
These tests compare the JAX RHS against the numpy coupled_odes on that
representation, dry and wet.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def _unified_setup(mushy_zone_factor=0.8):
    """Load unified iron core + unified MgSiO3 mantle caches; build jax_args."""
    from zalmoxis.config import load_material_dictionaries
    from zalmoxis.eos.interpolation import _ensure_unified_cache
    from zalmoxis.eos.paleos_api_cache import resolve_registry_entry
    from zalmoxis.melting_curves import get_solidus_liquidus_functions

    mat_dicts = load_material_dictionaries()

    interp_cache = {}
    cached = {}
    for name, key in (('PALEOS:iron', 'core'), ('PALEOS:MgSiO3', 'mun')):
        mat = mat_dicts[name]
        resolve_registry_entry(mat)
        if mat.get('format') != 'paleos_unified':
            pytest.skip(f'expected paleos_unified {name}, got {mat.get("format")}')
        if not os.path.isfile(mat['eos_file']):
            pytest.skip(f'EOS file missing: {mat["eos_file"]}')
        cached[key] = _ensure_unified_cache(mat['eos_file'], interp_cache)

    # External melting curves are only consumed by the wet blend's phi
    # (the unified density derives its solidus internally); any
    # (solidus, liquidus) pair works for parity since both sides get
    # the same functions.
    sol_func, liq_func = get_solidus_liquidus_functions()

    def extract_args(c, prefix):
        return {
            f'{prefix}_density_grid': np.asarray(c['density_grid']),
            f'{prefix}_unique_log_p': np.asarray(c['unique_log_p']),
            f'{prefix}_unique_log_t': np.asarray(c['unique_log_t']),
            f'{prefix}_logp_min': float(c['logp_min']),
            f'{prefix}_logt_min': float(c['logt_min']),
            f'{prefix}_dlog_p': float(c['dlog_p']),
            f'{prefix}_dlog_t': float(c['dlog_t']),
            f'{prefix}_n_p': int(c['n_p']),
            f'{prefix}_n_t': int(c['n_t']),
            f'{prefix}_p_min': float(c['p_min']),
            f'{prefix}_p_max': float(c['p_max']),
            f'{prefix}_lt_min_per_p': np.asarray(c['logt_valid_min']),
            f'{prefix}_lt_max_per_p': np.asarray(c['logt_valid_max']),
        }

    def extract_liquidus(c, prefix):
        lp = np.asarray(c.get('liquidus_log_p', []), dtype=float)
        lt = np.asarray(c.get('liquidus_log_t', []), dtype=float)
        return {
            f'{prefix}_liquidus_log_p': lp,
            f'{prefix}_liquidus_log_t': lt,
            f'{prefix}_liquidus_min_log_p': float(lp[0]) if len(lp) else 0.0,
            f'{prefix}_liquidus_max_log_p': float(lp[-1]) if len(lp) else 0.0,
            f'{prefix}_has_liquidus_f': 1.0 if len(lp) else 0.0,
        }

    # External melting curves on the shared log-P axis (wet blend only;
    # the unified density derives its own solidus internally).
    melt_lp = np.linspace(np.log10(1e8), np.log10(5e12), 256)
    melt_p = 10.0**melt_lp
    log_T_liq = np.log10(np.array([float(liq_func(P)) for P in melt_p]))
    log_T_sol = np.log10(np.array([float(sol_func(P)) for P in melt_p]))

    T_logP_grid = np.linspace(5.0, 12.5, 2000)
    T_values = 3000.0 + 5000.0 * (T_logP_grid - 5.0) / (12.5 - 5.0)

    from zalmoxis.constants import G

    cmb_mass = 0.325 * 5.972e24
    jax_args = {
        'cmb_mass': float(cmb_mass),
        'T_axis_grid': T_logP_grid,
        'T_values': T_values,
        'T_surface': 3000.0,
        'mushy_zone_factor_core': 1.0,
        'mushy_zone_factor_mantle': float(mushy_zone_factor),
        'melt_log_p_min': float(melt_lp[0]),
        'melt_dlog_p': float(melt_lp[1] - melt_lp[0]),
        'melt_n': int(len(melt_lp)),
        'log_T_liq_table': np.ascontiguousarray(log_T_liq),
        'log_T_sol_table': np.ascontiguousarray(log_T_sol),
        'G': G,
    }
    jax_args.update(extract_args(cached['core'], 'core'))
    jax_args.update(extract_liquidus(cached['core'], 'core'))
    jax_args.update(extract_args(cached['mun'], 'mun'))
    jax_args.update(extract_liquidus(cached['mun'], 'mun'))

    return {
        'mat_dicts': mat_dicts,
        'interp_cache': interp_cache,
        'sol_func': sol_func,
        'liq_func': liq_func,
        'mushy_zone_factors': {
            'PALEOS:iron': 1.0,
            'PALEOS:MgSiO3': mushy_zone_factor,
            'PALEOS:H2O': 1.0,
        },
        'cmb_mass': cmb_mass,
        'jax_args': jax_args,
        'T_axis_grid': T_logP_grid,
        'T_values': T_values,
    }


def _compare_rhs(setup, layer_mixtures, mat_dicts, profile, jax_extra, seed, tol):
    from zalmoxis.jax_eos.rhs import coupled_odes_jax
    from zalmoxis.structure_model import coupled_odes

    T_logP_grid = setup['T_axis_grid']
    T_values = setup['T_values']

    def numpy_temp(P):
        if P <= 0:
            return 3000.0
        return float(np.interp(np.log10(max(P, 1.0)), T_logP_grid, T_values))

    rng = np.random.default_rng(seed)
    M_planet = 5.972e24
    cmb_mass = setup['cmb_mass']

    numpy_dydr, jax_dydr = [], []
    for _ in range(200):
        r = rng.uniform(1e5, 6.4e6)
        y = np.array(
            [rng.uniform(0.0, M_planet), rng.uniform(0.5, 25.0), rng.uniform(1e6, 3e11)]
        )
        nv = coupled_odes(
            r,
            y,
            cmb_mass,
            cmb_mass + 0.675 * M_planet,
            layer_mixtures,
            setup['interp_cache'],
            mat_dicts,
            numpy_temp(y[2]),
            setup['sol_func'],
            setup['liq_func'],
            setup['mushy_zone_factors'],
            volatile_profile=profile,
        )
        numpy_dydr.append(nv)
        jv = coupled_odes_jax(r, y, mantle_is_unified=True, **jax_extra, **setup['jax_args'])
        jax_dydr.append(np.asarray(jv))

    numpy_dydr = np.asarray(numpy_dydr, dtype=float)
    jax_dydr = np.asarray(jax_dydr, dtype=float)
    both = (np.abs(numpy_dydr).max(axis=1) > 0) & (np.abs(jax_dydr).max(axis=1) > 0)
    n_both = int(both.sum())
    assert n_both > 100, f'too few comparable points: {n_both}/200'
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.abs(numpy_dydr[both] - jax_dydr[both]) / np.maximum(
            np.abs(numpy_dydr[both]), 1e-30
        )
    max_rel = float(rel.max()) if rel.size else 0.0
    print(f'n_both={n_both}/200, max_rel={max_rel:.3e}')
    assert max_rel <= tol, f'unified-mantle parity failed: max_rel={max_rel:.3e}'


@pytest.mark.integration
def test_unified_mantle_dry_parity():
    """Dry unified mantle: JAX RHS matches numpy, mushy zone included."""
    from zalmoxis.mixing import LayerMixture

    setup = _unified_setup(mushy_zone_factor=0.8)
    layer_mixtures = {
        'core': LayerMixture(['PALEOS:iron'], [1.0]),
        'mantle': LayerMixture(['PALEOS:MgSiO3'], [1.0]),
    }
    _compare_rhs(setup, layer_mixtures, setup['mat_dicts'], None, {}, seed=317, tol=1e-6)


@pytest.mark.integration
def test_unified_mantle_wet_parity():
    """Wet unified mantle: the volatile blend composes with the unified kernel."""
    from zalmoxis.eos.interpolation import _ensure_unified_cache
    from zalmoxis.eos.paleos_api_cache import resolve_registry_entry
    from zalmoxis.mixing import (
        _COMPONENT_RHO_MIN,
        _COMPONENT_RHO_SCALE,
        CONDENSED_RHO_MIN_DEFAULT,
        CONDENSED_RHO_SCALE_DEFAULT,
        LayerMixture,
        VolatileProfile,
    )

    setup = _unified_setup(mushy_zone_factor=0.8)
    vol_mat = setup['mat_dicts']['PALEOS:H2O']
    resolve_registry_entry(vol_mat)
    if not os.path.isfile(vol_mat.get('eos_file', '')):
        pytest.skip('H2O EOS file missing')
    vol_cached = _ensure_unified_cache(vol_mat['eos_file'], setup['interp_cache'])

    jax_extra = {
        'has_volatile': True,
        'vol_w_liquid': 0.083,
        'vol_w_solid': 0.0,
        'vol_mushy_zone_factor': 1.0,
        'sil_rho_min': _COMPONENT_RHO_MIN.get('PALEOS:MgSiO3', CONDENSED_RHO_MIN_DEFAULT),
        'sil_rho_scale': _COMPONENT_RHO_SCALE.get('PALEOS:MgSiO3', CONDENSED_RHO_SCALE_DEFAULT),
        'vol_rho_min': _COMPONENT_RHO_MIN.get('PALEOS:H2O', CONDENSED_RHO_MIN_DEFAULT),
        'vol_rho_scale': _COMPONENT_RHO_SCALE.get('PALEOS:H2O', CONDENSED_RHO_SCALE_DEFAULT),
        'vol_density_grid': np.asarray(vol_cached['density_grid']),
        'vol_unique_log_p': np.asarray(vol_cached['unique_log_p']),
        'vol_unique_log_t': np.asarray(vol_cached['unique_log_t']),
        'vol_logp_min': float(vol_cached['logp_min']),
        'vol_logt_min': float(vol_cached['logt_min']),
        'vol_dlog_p': float(vol_cached['dlog_p']),
        'vol_dlog_t': float(vol_cached['dlog_t']),
        'vol_n_p': int(vol_cached['n_p']),
        'vol_n_t': int(vol_cached['n_t']),
        'vol_p_min': float(vol_cached['p_min']),
        'vol_p_max': float(vol_cached['p_max']),
        'vol_lt_min_per_p': np.asarray(vol_cached['logt_valid_min']),
        'vol_lt_max_per_p': np.asarray(vol_cached['logt_valid_max']),
    }
    vlp = np.asarray(vol_cached.get('liquidus_log_p', []), dtype=float)
    vlt = np.asarray(vol_cached.get('liquidus_log_t', []), dtype=float)
    jax_extra['vol_liquidus_log_p'] = vlp
    jax_extra['vol_liquidus_log_t'] = vlt
    jax_extra['vol_liquidus_min_log_p'] = float(vlp[0]) if len(vlp) else 0.0
    jax_extra['vol_liquidus_max_log_p'] = float(vlp[-1]) if len(vlp) else 0.0
    jax_extra['vol_has_liquidus_f'] = 1.0 if len(vlp) else 0.0

    layer_mixtures = {
        'core': LayerMixture(['PALEOS:iron'], [1.0]),
        'mantle': LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.99, 0.01]),
    }
    profile = VolatileProfile(
        w_liquid={'PALEOS:H2O': 0.083},
        w_solid={'PALEOS:H2O': 0.0},
        primary_component='PALEOS:MgSiO3',
    )
    _compare_rhs(
        setup, layer_mixtures, setup['mat_dicts'], profile, jax_extra, seed=419, tol=1e-5
    )
