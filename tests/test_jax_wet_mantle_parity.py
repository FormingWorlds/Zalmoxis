"""Parity test: JAX wet-mantle RHS vs numpy coupled_odes with a VolatileProfile.

Extends the Stage-1b parity setup with a PALEOS:H2O volatile blended into
the mantle through a strong-partition VolatileProfile (w_solid = 0). The
numpy side goes through calculate_mixed_density's suppressed harmonic
mean with the profile's phi blend; the JAX side through the
has_volatile=True variant of coupled_odes_jax. Both consume the same
bilinear kernels, so agreement is expected at the melt-table tabulation
level (<= 1e-5 relative).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_jax_rhs_parity import _stage1b_setup  # noqa: E402


def _wet_setup():
    setup = _stage1b_setup()

    from zalmoxis.config import load_material_dictionaries
    from zalmoxis.eos.interpolation import _ensure_unified_cache
    from zalmoxis.eos.paleos_api_cache import resolve_registry_entry

    mat_dicts = load_material_dictionaries()
    vol_mat = mat_dicts['PALEOS:H2O']
    resolve_registry_entry(vol_mat)
    if vol_mat.get('format') != 'paleos_unified':
        pytest.skip(f'expected paleos_unified H2O, got {vol_mat.get("format")}')
    vol_file = vol_mat['eos_file']
    if not os.path.isfile(vol_file):
        pytest.skip(f'H2O EOS file missing: {vol_file}')
    vol_cached = _ensure_unified_cache(vol_file, setup['interp_cache'])
    setup['vol_mat'] = vol_mat

    # Volatile table args (prefix vol_), same extraction as the core.
    jax_args = setup['jax_args']
    jax_args.update(
        {
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
    )
    vol_liq_lp = np.asarray(vol_cached.get('liquidus_log_p', []), dtype=float)
    vol_liq_lt = np.asarray(vol_cached.get('liquidus_log_t', []), dtype=float)
    jax_args['vol_liquidus_log_p'] = vol_liq_lp
    jax_args['vol_liquidus_log_t'] = vol_liq_lt
    jax_args['vol_liquidus_min_log_p'] = float(vol_liq_lp[0]) if len(vol_liq_lp) else 0.0
    jax_args['vol_liquidus_max_log_p'] = float(vol_liq_lp[-1]) if len(vol_liq_lp) else 0.0
    jax_args['vol_has_liquidus_f'] = 1.0 if len(vol_liq_lp) else 0.0

    # Profile scalars and sigmoid constants, resolved exactly as
    # mixing.calculate_mixed_density does.
    from zalmoxis.mixing import (
        _COMPONENT_RHO_MIN,
        _COMPONENT_RHO_SCALE,
        CONDENSED_RHO_MIN_DEFAULT,
        CONDENSED_RHO_SCALE_DEFAULT,
    )

    w_liq, w_sol = 0.083, 0.0
    jax_args.update(
        {
            'vol_w_liquid': w_liq,
            'vol_w_solid': w_sol,
            'vol_mushy_zone_factor': 1.0,
            'sil_rho_min': _COMPONENT_RHO_MIN.get('PALEOS:MgSiO3', CONDENSED_RHO_MIN_DEFAULT),
            'sil_rho_scale': _COMPONENT_RHO_SCALE.get(
                'PALEOS:MgSiO3', CONDENSED_RHO_SCALE_DEFAULT
            ),
            'vol_rho_min': _COMPONENT_RHO_MIN.get('PALEOS:H2O', CONDENSED_RHO_MIN_DEFAULT),
            'vol_rho_scale': _COMPONENT_RHO_SCALE.get(
                'PALEOS:H2O', CONDENSED_RHO_SCALE_DEFAULT
            ),
        }
    )
    setup['w_liq'] = w_liq
    setup['w_sol'] = w_sol
    return setup


@pytest.mark.integration
def test_coupled_odes_jax_wet_parity():
    """JAX wet-mantle RHS matches numpy coupled_odes + VolatileProfile."""
    from zalmoxis.jax_eos.rhs import coupled_odes_jax
    from zalmoxis.mixing import LayerMixture, VolatileProfile
    from zalmoxis.structure_model import coupled_odes

    setup = _wet_setup()
    cfg = setup['config_params']
    T_logP_grid = setup['T_axis_grid']
    T_values = setup['T_values']

    core_eos = cfg['layer_eos_config']['core']
    mantle_eos = cfg['layer_eos_config']['mantle']
    layer_mixtures = {
        'core': LayerMixture([core_eos], [1.0]),
        # Extended wet mixture: placeholder fractions, overridden per
        # shell by the profile (mirrors extend_mantle_eos_with_volatiles).
        'mantle': LayerMixture([mantle_eos, 'PALEOS:H2O'], [0.99, 0.01]),
    }
    mat_dicts = {
        core_eos: setup['core_mat'],
        mantle_eos: setup['mantle_mat'],
        'PALEOS:H2O': setup['vol_mat'],
    }
    profile = VolatileProfile(
        w_liquid={'PALEOS:H2O': setup['w_liq']},
        w_solid={'PALEOS:H2O': setup['w_sol']},
        primary_component=mantle_eos,
    )

    def numpy_temp(r, P):
        if P <= 0:
            return 3000.0
        return float(np.interp(np.log10(max(P, 1.0)), T_logP_grid, T_values))

    rng = np.random.default_rng(211)
    M_planet = 5.972e24
    cmb_mass = setup['cmb_mass']

    numpy_dydr = []
    jax_dydr = []
    query_info = []
    for _ in range(200):
        r = rng.uniform(1e5, 6.4e6)
        M = rng.uniform(0.0, M_planet)
        g = rng.uniform(0.5, 25.0)
        P = rng.uniform(1e6, 3e11)
        y = np.array([M, g, P])
        temperature = numpy_temp(r, P)
        nv = coupled_odes(
            r,
            y,
            cmb_mass,
            cmb_mass + 0.675 * M_planet,
            layer_mixtures,
            setup['interp_cache'],
            mat_dicts,
            temperature,
            setup['sol_func'],
            setup['liq_func'],
            setup['mushy_zone_factors'],
            volatile_profile=profile,
        )
        numpy_dydr.append(nv)
        jv = coupled_odes_jax(r, y, has_volatile=True, **setup['jax_args'])
        jax_dydr.append(np.asarray(jv))
        query_info.append((r, M, g, P, temperature, M < cmb_mass))

    numpy_dydr = np.asarray(numpy_dydr, dtype=float)
    jax_dydr = np.asarray(jax_dydr, dtype=float)

    both_nonzero = (np.abs(numpy_dydr).max(axis=1) > 0) & (np.abs(jax_dydr).max(axis=1) > 0)
    n_both = int(both_nonzero.sum())
    assert n_both > 100, f'too few comparable points: {n_both}/200'

    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.abs(numpy_dydr[both_nonzero] - jax_dydr[both_nonzero]) / np.maximum(
            np.abs(numpy_dydr[both_nonzero]), 1e-30
        )
    max_rel = float(rel.max()) if rel.size > 0 else 0.0
    print(f'n_both={n_both}/200, max_rel={max_rel:.3e}')

    if max_rel > 1e-5:
        row_max = rel.max(axis=1)
        worst_idx = int(np.argmax(row_max))
        orig_i = int(np.where(both_nonzero)[0][worst_idx])
        r_, M_, g_, P_, T_, is_core = query_info[orig_i]
        print(
            f'  worst idx={orig_i}: r={r_:.3e}, M={M_:.3e}, P={P_:.3e}, T={T_:.1f}, core={is_core}'
        )
        print(f'    numpy={numpy_dydr[orig_i]}')
        print(f'    jax  ={jax_dydr[orig_i]}')

    assert max_rel <= 1e-5, f'wet RHS parity failed: max_rel={max_rel:.3e} (want <=1e-5)'


@pytest.mark.integration
def test_wet_profile_gates_fall_back():
    """Unsupported profiles raise ValueError (numpy fallback) in the wrapper."""
    from zalmoxis.jax_eos.wrapper import _validate_wet_mantle
    from zalmoxis.mixing import LayerMixture, VolatileProfile

    lm = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O', 'Chabrier:H'], [0.98, 0.01, 0.01])
    mats = {'PALEOS:H2O': {}, 'Chabrier:H': {}}

    # Two active volatiles: unsupported.
    p2 = VolatileProfile(
        w_liquid={'PALEOS:H2O': 0.05, 'Chabrier:H': 0.01},
        w_solid={},
        primary_component='PALEOS:MgSiO3',
    )
    with pytest.raises(ValueError, match='exactly one active volatile'):
        _validate_wet_mantle(p2, lm, mats)

    # H2 alone: binodal suppression not ported. The mixture must match
    # the profile so the Chabrier-specific gate fires rather than the
    # unmanaged-component check.
    ph2 = VolatileProfile(
        w_liquid={'Chabrier:H': 0.01},
        w_solid={},
        primary_component='PALEOS:MgSiO3',
    )
    lm_h2 = LayerMixture(['PALEOS:MgSiO3', 'Chabrier:H'], [0.99, 0.01])
    with pytest.raises(ValueError, match='Chabrier:H'):
        _validate_wet_mantle(ph2, lm_h2, mats)

    # Miscibility profiles stay on numpy.
    pm = VolatileProfile(
        w_liquid={'PALEOS:H2O': 0.05},
        w_solid={},
        primary_component='PALEOS:MgSiO3',
    )
    pm.global_miscibility = True
    with pytest.raises(ValueError, match='global_miscibility'):
        _validate_wet_mantle(pm, lm, mats)

    # The supported envelope: one active paleos-table volatile, and the
    # mixture carries nothing outside the profile (the 3-component lm
    # above has an unmanaged Chabrier:H, which is itself rejected; see
    # test_wet_mantle_with_unmanaged_component_falls_back).
    pm.global_miscibility = False
    lm_ok = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.99, 0.01])
    vol_eos, (w_l, w_s) = _validate_wet_mantle(pm, lm_ok, mats)
    assert vol_eos == 'PALEOS:H2O'
    assert w_l == pytest.approx(0.05)
    assert w_s == 0.0


@pytest.mark.integration
def test_wet_mantle_with_unmanaged_component_falls_back():
    """A mixture component outside the profile keeps its fraction on the
    numpy path (apply_to_mixture), so the 2-component JAX blend would
    silently drop it; the wrapper must reject it to the numpy fallback.
    Profile-managed components with zero weight in both phases contribute
    nothing in numpy and stay allowed."""
    from zalmoxis.jax_eos.wrapper import _validate_wet_mantle
    from zalmoxis.mixing import LayerMixture, VolatileProfile

    mats = {'PALEOS:H2O': {}}
    profile = VolatileProfile(
        w_liquid={'PALEOS:H2O': 0.05},
        w_solid={'PALEOS:H2O': 0.0},
        primary_component='PALEOS:MgSiO3',
    )

    lm_extra = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O', 'PALEOS:iron'], [0.90, 0.05, 0.05])
    with pytest.raises(ValueError, match='outside the volatile profile'):
        _validate_wet_mantle(profile, lm_extra, mats)

    # Managed-but-inactive component: zero weight in both phases blends
    # to zero in numpy, so the JAX envelope keeps it.
    profile_managed = VolatileProfile(
        w_liquid={'PALEOS:H2O': 0.05, 'Chabrier:H': 0.0},
        w_solid={'PALEOS:H2O': 0.0, 'Chabrier:H': 0.0},
        primary_component='PALEOS:MgSiO3',
    )
    lm_managed = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O', 'Chabrier:H'], [0.90, 0.05, 0.05])
    vol_eos, (w_l, w_s) = _validate_wet_mantle(profile_managed, lm_managed, mats)
    assert vol_eos == 'PALEOS:H2O'
    assert w_l == pytest.approx(0.05)
    assert w_s == 0.0
