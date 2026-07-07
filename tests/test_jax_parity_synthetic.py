"""Per-PR numerical parity on synthetic unified tables (no EOS files).

The table-based parity tests need the PALEOS tables, which are not in
the repository, so on a PR runner they skip and the JAX-vs-numpy
cross-check effectively only runs nightly. These tests fabricate small
unified tables in memory and inject them into the interpolation cache
(``_ensure_unified_cache`` early-returns on pre-populated entries), so
the same cross-implementation check runs on every PR with no data
dependency: the numpy reference (``eos.paleos.get_paleos_unified_density``
through ``structure_model.coupled_odes``) against the JAX kernels, on
the unified-mantle and wet-blend paths this PR adds.

The synthetic density surface is smooth and fully covers the query
range, so the numpy KDTree NN fallback (not ported to JAX) never
engages and agreement is expected at rounding level.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.unit

# Synthetic table axes: 1e9 to 1e12 Pa, 1e3 to 1e4 K.
_N_P, _N_T = 8, 8
_LOG_P = np.linspace(9.0, 12.0, _N_P)
_LOG_T = np.linspace(3.0, 4.0, _N_T)


def _synthetic_unified_cache(rho0, drho_dlogp, drho_dlogt, liq_lo, liq_hi):
    """A smooth fabricated unified-table cache entry.

    Density is affine in (log P, log T) so bilinear interpolation is
    exact and any numpy-vs-JAX discrepancy is implementation, not
    resolution. The liquidus runs from ``liq_lo`` to ``liq_hi`` K across
    the pressure range, inside the table's T coverage, so the internal
    mushy branch (mushy_zone_factor < 1) is exercised.
    """
    lp, lt = np.meshgrid(_LOG_P, _LOG_T, indexing='ij')
    grid = rho0 + drho_dlogp * (lp - 9.0) + drho_dlogt * (lt - 3.0)
    return {
        'type': 'paleos_unified',
        'density_grid': grid,
        'unique_log_p': _LOG_P.copy(),
        'unique_log_t': _LOG_T.copy(),
        'logp_min': float(_LOG_P[0]),
        'logt_min': float(_LOG_T[0]),
        'dlog_p': float(_LOG_P[1] - _LOG_P[0]),
        'dlog_t': float(_LOG_T[1] - _LOG_T[0]),
        'n_p': _N_P,
        'n_t': _N_T,
        'p_min': float(10.0 ** _LOG_P[0]),
        'p_max': float(10.0 ** _LOG_P[-1]),
        'logt_valid_min': np.full(_N_P, _LOG_T[0]),
        'logt_valid_max': np.full(_N_P, _LOG_T[-1]),
        'liquidus_log_p': _LOG_P.copy(),
        'liquidus_log_t': np.log10(np.linspace(liq_lo, liq_hi, _N_P)),
        # Never reached (the affine surface has no NaN cells); present
        # so an unexpected engagement fails loudly instead of KeyError.
        'density_nn': lambda pt: (_ for _ in ()).throw(
            AssertionError('NN fallback engaged on a synthetic table')
        ),
    }


def _synthetic_world():
    """Materials, caches, curves, and jax_args for a 2-layer synthetic
    planet: unified core + unified mantle + one unified volatile."""
    # Real (whitelisted) component names over synthetic tables: numpy's
    # _get_mushy_zone_factor returns 1.0 for any name outside
    # _PALEOS_UNIFIED_NAMES regardless of the config dict, and the JAX
    # wrapper resolves through the same function, so synthetic names
    # would silently disable the mushy branch on both sides (and a
    # hand-fed factor on one side would break parity by construction).
    core_eos, sil_eos, vol_eos = 'PALEOS:iron', 'PALEOS:MgSiO3', 'PALEOS:H2O'
    caches = {
        '/synthetic/core.dat': _synthetic_unified_cache(9000.0, 2500.0, -400.0, 3500.0, 8000.0),
        '/synthetic/sil.dat': _synthetic_unified_cache(4000.0, 1200.0, -300.0, 2000.0, 6000.0),
        '/synthetic/vol.dat': _synthetic_unified_cache(1200.0, 900.0, -200.0, 1500.0, 4000.0),
    }
    mats = {
        core_eos: {'eos_file': '/synthetic/core.dat', 'format': 'paleos_unified'},
        sil_eos: {'eos_file': '/synthetic/sil.dat', 'format': 'paleos_unified'},
        vol_eos: {'eos_file': '/synthetic/vol.dat', 'format': 'paleos_unified'},
    }
    interp_cache = dict(caches)  # pre-populated: _ensure_unified_cache early-returns

    # Analytic external melting curves (wet phi only), inside the T
    # range. Power-law form on purpose: the RHS consumes these through a
    # log-log tabulation that is bit-exact for T = A * P^B, so the
    # tabulation contributes nothing to the parity budget (a curve
    # linear in P would leave ~1e-2 interpolation residue and mask real
    # discrepancies).
    def liq_func(P):
        return 2200.0 * (P / 1e9) ** 0.12

    def sol_func(P):
        return 0.8 * liq_func(P)

    melt_lp = np.linspace(9.0, 12.0, 256)
    melt_p = 10.0**melt_lp
    log_T_liq = np.log10(np.array([liq_func(P) for P in melt_p]))
    log_T_sol = np.log10(np.array([sol_func(P) for P in melt_p]))

    # P-indexed synthetic adiabat spanning the table's T range.
    T_grid = np.linspace(9.0, 12.0, 500)
    T_vals = 1500.0 + 6000.0 * (T_grid - 9.0) / 3.0

    from zalmoxis.constants import G
    from zalmoxis.mixing import (
        _COMPONENT_RHO_MIN,
        _COMPONENT_RHO_SCALE,
        CONDENSED_RHO_MIN_DEFAULT,
        CONDENSED_RHO_SCALE_DEFAULT,
        _get_mushy_zone_factor,
    )

    def extract(cache, prefix):
        out = {
            f'{prefix}_density_grid': cache['density_grid'],
            f'{prefix}_unique_log_p': cache['unique_log_p'],
            f'{prefix}_unique_log_t': cache['unique_log_t'],
            f'{prefix}_logp_min': cache['logp_min'],
            f'{prefix}_logt_min': cache['logt_min'],
            f'{prefix}_dlog_p': cache['dlog_p'],
            f'{prefix}_dlog_t': cache['dlog_t'],
            f'{prefix}_n_p': cache['n_p'],
            f'{prefix}_n_t': cache['n_t'],
            f'{prefix}_p_min': cache['p_min'],
            f'{prefix}_p_max': cache['p_max'],
            f'{prefix}_lt_min_per_p': cache['logt_valid_min'],
            f'{prefix}_lt_max_per_p': cache['logt_valid_max'],
            f'{prefix}_liquidus_log_p': cache['liquidus_log_p'],
            f'{prefix}_liquidus_log_t': cache['liquidus_log_t'],
            f'{prefix}_liquidus_min_log_p': float(cache['liquidus_log_p'][0]),
            f'{prefix}_liquidus_max_log_p': float(cache['liquidus_log_p'][-1]),
            f'{prefix}_has_liquidus_f': 1.0,
        }
        return out

    cmb_mass = 2.0e24
    jax_args = {
        'cmb_mass': cmb_mass,
        'T_axis_grid': T_grid,
        'T_values': T_vals,
        'T_surface': 1500.0,
        'mushy_zone_factor_core': 1.0,
        'mushy_zone_factor_mantle': _get_mushy_zone_factor(sil_eos, {sil_eos: 0.8}),
        'melt_log_p_min': float(melt_lp[0]),
        'melt_dlog_p': float(melt_lp[1] - melt_lp[0]),
        'melt_n': 256,
        'log_T_liq_table': np.ascontiguousarray(log_T_liq),
        'log_T_sol_table': np.ascontiguousarray(log_T_sol),
        'G': G,
        'vol_w_liquid': 0.083,
        'vol_w_solid': 0.0,
        'vol_mushy_zone_factor': 1.0,
        'sil_rho_min': _COMPONENT_RHO_MIN.get(sil_eos, CONDENSED_RHO_MIN_DEFAULT),
        'sil_rho_scale': _COMPONENT_RHO_SCALE.get(sil_eos, CONDENSED_RHO_SCALE_DEFAULT),
        'vol_rho_min': _COMPONENT_RHO_MIN.get(vol_eos, CONDENSED_RHO_MIN_DEFAULT),
        'vol_rho_scale': _COMPONENT_RHO_SCALE.get(vol_eos, CONDENSED_RHO_SCALE_DEFAULT),
    }
    jax_args.update(extract(caches['/synthetic/core.dat'], 'core'))
    # The core kernel takes the liquidus set through the same names but
    # without the sub-table split; drop the two keys the extract helper
    # adds that the core call signature does not carry.
    jax_args.update(extract(caches['/synthetic/sil.dat'], 'mun'))
    jax_args.update(extract(caches['/synthetic/vol.dat'], 'vol'))

    return {
        'eos': (core_eos, sil_eos, vol_eos),
        'mats': mats,
        'interp_cache': interp_cache,
        'sol_func': sol_func,
        'liq_func': liq_func,
        'mushy_zone_factors': {core_eos: 1.0, sil_eos: 0.8, vol_eos: 1.0},
        'cmb_mass': cmb_mass,
        'jax_args': jax_args,
        'T_grid': T_grid,
        'T_vals': T_vals,
    }


@pytest.mark.reference_pinned
def test_unified_kernel_parity_synthetic():
    """JAX unified kernel matches eos.paleos.get_paleos_unified_density
    (the numpy reference) on a fabricated table, direct and mushy
    branches, at rounding level."""
    from zalmoxis.eos.paleos import get_paleos_unified_density
    from zalmoxis.jax_eos.paleos import get_paleos_unified_density_jax

    world = _synthetic_world()
    cache = world['interp_cache']['/synthetic/sil.dat']
    mat = world['mats'][world['eos'][1]]

    kernel_args = {
        k.replace('mun_', ''): v for k, v in world['jax_args'].items() if k.startswith('mun_')
    }

    P_q = np.logspace(9.1, 11.9, 12)
    T_q = np.linspace(1200.0, 9500.0, 12)
    n_mushy = 0
    for P in P_q:
        for T in T_q:
            rho_np = get_paleos_unified_density(P, T, mat, 0.8, world['interp_cache'])
            rho_jx = float(get_paleos_unified_density_jax(P, T, 0.8, **kernel_args))
            assert rho_np is not None
            assert rho_jx == pytest.approx(rho_np, rel=1e-12), (P, T)
            # Count mushy-branch hits so branch coverage is asserted,
            # not assumed.
            T_melt = 10.0 ** float(
                np.interp(np.log10(P), cache['liquidus_log_p'], cache['liquidus_log_t'])
            )
            if 0.8 * T_melt < T < T_melt:
                n_mushy += 1
    assert n_mushy > 5, f'mushy branch barely exercised: {n_mushy} points'


@pytest.mark.reference_pinned
def test_rhs_parity_synthetic_unified_wet_and_dry():
    """JAX RHS matches structure_model.coupled_odes (the numpy
    reference) on the synthetic planet: unified mantle, dry and with
    the wet volatile blend."""
    from zalmoxis.jax_eos.rhs import coupled_odes_jax
    from zalmoxis.mixing import LayerMixture, VolatileProfile
    from zalmoxis.structure_model import coupled_odes

    world = _synthetic_world()
    core_eos, sil_eos, vol_eos = world['eos']
    T_grid, T_vals = world['T_grid'], world['T_vals']

    def numpy_temp(P):
        return float(np.interp(np.log10(max(P, 1.0)), T_grid, T_vals))

    cases = [
        (
            False,
            None,
            {'core': LayerMixture([core_eos], [1.0]), 'mantle': LayerMixture([sil_eos], [1.0])},
        ),
        (
            True,
            VolatileProfile(
                w_liquid={vol_eos: 0.083},
                w_solid={vol_eos: 0.0},
                primary_component=sil_eos,
            ),
            {
                'core': LayerMixture([core_eos], [1.0]),
                'mantle': LayerMixture([sil_eos, vol_eos], [0.99, 0.01]),
            },
        ),
    ]

    rng = np.random.default_rng(823)
    M_planet = 6.0e24
    for has_volatile, profile, layer_mixtures in cases:
        n_compared = 0
        for _ in range(60):
            r = rng.uniform(1e5, 7e6)
            P = rng.uniform(2e9, 8e11)  # inside the synthetic table
            y = np.array([rng.uniform(0.0, M_planet), rng.uniform(0.5, 25.0), P])
            nv = np.asarray(
                coupled_odes(
                    r,
                    y,
                    world['cmb_mass'],
                    world['cmb_mass'] + 3.0e24,
                    layer_mixtures,
                    world['interp_cache'],
                    world['mats'],
                    numpy_temp(P),
                    world['sol_func'],
                    world['liq_func'],
                    world['mushy_zone_factors'],
                    volatile_profile=profile,
                ),
                dtype=float,
            )
            jv = np.asarray(
                coupled_odes_jax(
                    r,
                    y,
                    has_volatile=has_volatile,
                    mantle_is_unified=True,
                    **world['jax_args'],
                )
            )
            # The synthetic tables fully cover the query range: neither
            # side may zero a shell here.
            assert np.abs(nv).max() > 0, (r, P)
            assert np.abs(jv).max() > 0, (r, P)
            np.testing.assert_allclose(jv, nv, rtol=1e-10, atol=0.0)
            n_compared += 1
        assert n_compared == 60
