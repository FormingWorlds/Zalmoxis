"""End-to-end wrapper: numpy-signature solve_structure → JAX path.

``solve_structure_via_jax`` takes the same arguments as
``zalmoxis.structure_model.solve_structure`` and produces the same
(mass, gravity, pressure) output arrays, but routes through the
JIT-compiled JAX path.

Scope: Stage-1b 2-layer config (single-component core + mantle,
PALEOS:iron + PALEOS-2phase:MgSiO3). Anything outside this envelope
(3-layer ice, multi-component mixing, non-Stixrude14 melting) falls
back to the numpy path at the caller (solver._solve).

The wrapper does three things per call:
 1. Extract cache contents into flat arrays the JIT function needs.
 2. Tabulate temperature_function(r, P) on a log-P grid so the RHS
    can do jnp.interp without calling back into Python.
 3. Call solve_structure_jax and return numpy arrays in the same
    layout as the numpy solve_structure.

Caches are stored in the numpy interpolation_cache dict with a key
'_jax_extracted' so repeated calls within one _solve() reuse the
flat-array extraction.
"""
from __future__ import annotations

import numpy as np

from .solver import solve_structure_jax


def _extract_sub_args(cached, prefix):
    """Flatten a paleos/paleos_unified cache entry into jax kwargs."""
    return {
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


def _extract_liquidus(cached, prefix='core'):
    """Flatten liquidus curve data for paleos_unified tables."""
    liq_lp = np.asarray(cached.get('liquidus_log_p', []), dtype=float)
    liq_lt = np.asarray(cached.get('liquidus_log_t', []), dtype=float)
    return {
        f'{prefix}_liquidus_log_p': liq_lp,
        f'{prefix}_liquidus_log_t': liq_lt,
        f'{prefix}_liquidus_min_log_p': float(liq_lp[0]) if len(liq_lp) else 0.0,
        f'{prefix}_liquidus_max_log_p': float(liq_lp[-1]) if len(liq_lp) else 0.0,
        f'{prefix}_has_liquidus_f': 1.0 if len(liq_lp) else 0.0,
    }


def _tabulate_adiabat(radii, temperature_function, n_pts=4000):
    """Build a log_P → T lookup covering the structure's pressure range.

    Samples ``temperature_function`` at ``n_pts`` pressures spanning
    the expected physical range. Uses a log-uniform grid.
    """
    # Cover 1 bar to 10 TPa, wider than any current PROTEUS config
    log_p_min = 5.0   # log10(1 bar)
    log_p_max = 13.0  # log10(10 TPa)
    log_p_grid = np.linspace(log_p_min, log_p_max, n_pts)
    P_grid = 10.0 ** log_p_grid
    # temperature_function is (r, P) -> T. For adiabatic mode T is
    # ~independent of r, so we sample at the median radius. For the
    # degenerate linear-T fallback this is fine too.
    r_mid = 0.5 * (float(radii[0]) + float(radii[-1]))
    T_values = np.array(
        [float(temperature_function(r_mid, P)) for P in P_grid], dtype=float
    )
    return log_p_grid, T_values


def solve_structure_via_jax(
    layer_mixtures,
    cmb_mass,
    core_mantle_mass,
    radii,
    adaptive_radial_fraction,   # unused in JAX path (Tsit5 adaptive internal)
    relative_tolerance,
    absolute_tolerance,
    maximum_step,               # unused in JAX path for now
    material_dictionaries,
    interpolation_cache,
    y0,
    solidus_func,               # used to extract Stixrude14 params
    liquidus_func,              # used to extract Stixrude14 params
    temperature_function=None,
    mushy_zone_factors=None,
    condensed_rho_min=None,     # ignored (JAX path assumes no multi-component mixing)
    condensed_rho_scale=None,
    binodal_T_scale=None,
):
    """Drop-in replacement for ``solve_structure`` using the JAX path.

    Returns (mass_enclosed, gravity, pressure) numpy arrays on the
    ``radii`` grid, matching numpy's solve_structure output contract.
    """
    from .. import melting_curves as mc
    from ..eos.interpolation import _ensure_unified_cache
    from ..eos.seager import get_tabulated_eos

    radii_arr = np.asarray(radii, dtype=float)

    # Core cache: paleos_unified (load via _ensure_unified_cache)
    core_lm = layer_mixtures['core']
    core_eos = core_lm.components[0]  # single-component assumption
    core_mat = material_dictionaries[core_eos]
    # Resolve PALEOS-API lazily if needed (matches numpy dispatch)
    from ..eos.dispatch import _is_paleos_api
    if '_api_resolved' not in core_mat:
        if _is_paleos_api(core_mat):
            from ..eos.paleos_api_cache import resolve_registry_entry
            resolve_registry_entry(core_mat)
        core_mat['_api_resolved'] = True
    if core_mat.get('format') != 'paleos_unified':
        raise ValueError(
            f"JAX path requires paleos_unified core, got format "
            f"{core_mat.get('format')!r}"
        )
    core_cached = _ensure_unified_cache(core_mat['eos_file'], interpolation_cache)

    # Mantle cache: PALEOS-2phase (solid + melted sub-tables)
    mantle_lm = layer_mixtures['mantle']
    mantle_eos = mantle_lm.components[0]
    mantle_mat = material_dictionaries[mantle_eos]
    if '_api_resolved' not in mantle_mat:
        if _is_paleos_api(mantle_mat):
            from ..eos.paleos_api_cache import resolve_registry_entry
            resolve_registry_entry(mantle_mat)
        mantle_mat['_api_resolved'] = True
    if 'melted_mantle' not in mantle_mat or 'solid_mantle' not in mantle_mat:
        raise ValueError(
            "JAX path requires PALEOS-2phase mantle with solid_mantle+melted_mantle"
        )
    # Lazy-load both sub-tables via numpy's get_tabulated_eos
    _ = get_tabulated_eos(1e10, mantle_mat, 'solid_mantle', 3000.0, interpolation_cache)
    _ = get_tabulated_eos(1e10, mantle_mat, 'melted_mantle', 5000.0, interpolation_cache)
    sol_file = mantle_mat['solid_mantle']['eos_file']
    liq_file = mantle_mat['melted_mantle']['eos_file']
    sol_cached = interpolation_cache[sol_file]
    liq_cached = interpolation_cache[liq_file]

    # mushy_zone_factor for the CORE (paleos_unified takes one; mantle uses Tdep
    # which handles its own solid/liquid separately).
    core_mzf = 1.0
    if mushy_zone_factors is not None:
        if isinstance(mushy_zone_factors, dict):
            core_mzf = float(mushy_zone_factors.get(core_eos, 1.0))
        else:
            core_mzf = float(mushy_zone_factors)

    # Adiabat tabulation. If no temperature_function, use constant 3000 K.
    if temperature_function is None:
        T_logP_grid = np.linspace(5.0, 13.0, 4)
        T_values = np.full(4, 3000.0)
        T_surface = 3000.0
    else:
        T_logP_grid, T_values = _tabulate_adiabat(radii_arr, temperature_function)
        T_surface = float(temperature_function(float(radii_arr[-1]), 1e5))

    # Stixrude14 params (hardcoded to module constants; matches what the
    # numpy path uses for this config).
    stix = {
        'stix_T_ref': mc._STIX14_T_REF,
        'stix_P_ref': mc._STIX14_P_REF,
        'stix_exponent': mc._STIX14_EXPONENT,
        'stix_cryo_factor': mc._STIX14_CRYO_FACTOR,
    }

    # Physical constant G matching numpy path
    from ..constants import G

    jax_args = {
        'cmb_mass': float(cmb_mass),
        'T_logP_grid': T_logP_grid,
        'T_values': T_values,
        'T_surface': T_surface,
        'mushy_zone_factor_core': core_mzf,
        'G': float(G),
    }
    jax_args.update(_extract_sub_args(core_cached, 'core'))
    jax_args.update(_extract_liquidus(core_cached, 'core'))
    jax_args.update(_extract_sub_args(sol_cached, 'sol'))
    jax_args.update(_extract_sub_args(liq_cached, 'liq'))
    jax_args.update(stix)

    ys = solve_structure_jax(
        radii_arr, np.asarray(y0, dtype=float),
        rtol=float(relative_tolerance),
        atol=float(absolute_tolerance),
        **jax_args,
    )
    ys = np.asarray(ys)
    mass_enclosed = ys[:, 0]
    gravity = ys[:, 1]
    pressure = ys[:, 2]
    return mass_enclosed, gravity, pressure
