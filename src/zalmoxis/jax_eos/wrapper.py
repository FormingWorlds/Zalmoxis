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

import os as _os
import time as _time

import numpy as np

from .solver import solve_structure_jax

_CALL_COUNT = 0
_TOTAL_WALL = 0.0
_DEBUG = bool(_os.environ.get('ZALMOXIS_JAX_DEBUG'))
# Phase timing buckets for deep profiling. Populated when
# ZALMOXIS_JAX_PROFILE=1 is set.
_PROFILE = bool(_os.environ.get('ZALMOXIS_JAX_PROFILE'))
_PHASE_TIMES = {'cache_extract': 0.0, 'adiabat_tab': 0.0, 'jit_solve': 0.0, 'other': 0.0}

# Cache the mantle melting-curve tabulation on a shared log-P axis,
# keyed by (id(solidus_func), id(liquidus_func)). See the rebuild
# branch in solve_structure_via_jax for the rationale and cost numbers.
_MELT_TABLE_CACHE: dict = {}


def _extract_sub_args(cached, prefix):
    """Flatten a paleos/paleos_unified cache entry into jax kwargs.

    Caches the extracted dict on the cache entry itself so we only build
    it once per (cached_id, prefix) pair. The hot-path inner Picard loop
    calls solve_structure_via_jax thousands of times per main(); without
    this cache the np.asarray + dict-allocation overhead alone hit 16.7 s
    of a 23.5 s no-Anderson real-CHILI bench (cProfile 2026-04-25).
    """
    cache_key = f'_jax_sub_args::{prefix}'
    cached_args = cached.get(cache_key)
    if cached_args is not None:
        return cached_args
    # _ensure_unified_cache and seager.get_tabulated_eos already store
    # numpy arrays for these fields; np.asarray here is redundant.
    out = {
        f'{prefix}_density_grid': cached['density_grid'],
        f'{prefix}_unique_log_p': cached['unique_log_p'],
        f'{prefix}_unique_log_t': cached['unique_log_t'],
        f'{prefix}_logp_min': float(cached['logp_min']),
        f'{prefix}_logt_min': float(cached['logt_min']),
        f'{prefix}_dlog_p': float(cached['dlog_p']),
        f'{prefix}_dlog_t': float(cached['dlog_t']),
        f'{prefix}_n_p': int(cached['n_p']),
        f'{prefix}_n_t': int(cached['n_t']),
        f'{prefix}_p_min': float(cached['p_min']),
        f'{prefix}_p_max': float(cached['p_max']),
        f'{prefix}_lt_min_per_p': cached['logt_valid_min'],
        f'{prefix}_lt_max_per_p': cached['logt_valid_max'],
    }
    cached[cache_key] = out
    return out


def _extract_liquidus(cached, prefix='core'):
    """Flatten liquidus curve data for paleos_unified tables.

    Cached on the cache entry itself for the same reason as _extract_sub_args.
    """
    cache_key = f'_jax_liquidus::{prefix}'
    cached_args = cached.get(cache_key)
    if cached_args is not None:
        return cached_args
    liq_lp = np.asarray(cached.get('liquidus_log_p', []), dtype=float)
    liq_lt = np.asarray(cached.get('liquidus_log_t', []), dtype=float)
    out = {
        f'{prefix}_liquidus_log_p': liq_lp,
        f'{prefix}_liquidus_log_t': liq_lt,
        f'{prefix}_liquidus_min_log_p': float(liq_lp[0]) if len(liq_lp) else 0.0,
        f'{prefix}_liquidus_max_log_p': float(liq_lp[-1]) if len(liq_lp) else 0.0,
        f'{prefix}_has_liquidus_f': 1.0 if len(liq_lp) else 0.0,
    }
    cached[cache_key] = out
    return out


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
    temperature_arrays=None,    # (r_arr, T_arr): r-indexed T profile
    mushy_zone_factors=None,
    condensed_rho_min=None,     # ignored (JAX path assumes no multi-component mixing)
    condensed_rho_scale=None,
    binodal_T_scale=None,
):
    """Drop-in replacement for ``solve_structure`` using the JAX path.

    Returns (mass_enclosed, gravity, pressure) numpy arrays on the
    ``radii`` grid, matching numpy's solve_structure output contract.

    Temperature input
    -----------------
    One (and only one) of ``temperature_arrays`` and
    ``temperature_function`` should carry the T profile:

    * ``temperature_arrays = (r_arr, T_arr)`` — explicit r-indexed T
      profile. The JAX RHS interpolates T directly on radius
      (``T_axis_is_radius=True``). Use this when the caller's T is
      naturally a function of r alone (e.g. PROTEUS'
      ``update_structure_from_interior``, which hands Zalmoxis a
      closure over SPIDER/Aragog's T(r) staggered grid — that closure
      ignores P and cannot be meaningfully tabulated on a log-P axis;
      see ``tools/benchmarks/bench_coupled_tempfunc.py`` for the
      reproducer).

    * ``temperature_function(r, P) -> T`` — a Python callable. The
      wrapper samples it on a 4000-point log-P axis at
      ``r_mid = 0.5 * (radii[0] + radii[-1])``, caches the result by
      ``id(temperature_function)``, and the RHS interpolates on
      ``log10(P)``. This matches Zalmoxis' internal adiabat path where
      T along a column tracks P strongly and weakly depends on r.

    If neither is provided, the RHS uses a constant 3000 K fallback
    (bench idealisation only; not physically meaningful for real
    structure solves).

    Providing both is rejected to avoid ambiguity.
    """
    from ..eos.interpolation import _ensure_unified_cache
    from ..eos.seager import get_tabulated_eos

    _p_t0 = _time.perf_counter() if _PROFILE else 0.0

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

    if _PROFILE:
        _PHASE_TIMES['cache_extract'] += _time.perf_counter() - _p_t0
        _p_t0 = _time.perf_counter()

    # Temperature-profile setup. Two paths (see function docstring for
    # when to use which):
    #   1. temperature_arrays=(r_arr, T_arr): r-indexed. Used by PROTEUS.
    #   2. temperature_function callable:      P-indexed. Used by bench.
    # When both are provided, temperature_arrays wins. Zalmoxis' inner
    # solver loop always constructs an internal `_temperature_func`
    # (linear guess / adiabat blend) even when the caller hands in
    # r-indexed arrays, so rejecting the pair would force every caller
    # to monkey-patch solver.py; we just let arrays take precedence.
    if temperature_arrays is not None:
        _r_arr, _T_arr = temperature_arrays
        T_axis_grid = np.asarray(_r_arr, dtype=float)
        T_values = np.asarray(_T_arr, dtype=float)
        if T_axis_grid.shape != T_values.shape or T_axis_grid.ndim != 1:
            raise ValueError(
                'temperature_arrays must be two 1-D arrays of equal length, '
                f'got shapes {T_axis_grid.shape} and {T_values.shape}.'
            )
        # T_surface is unused in the r-indexed RHS branch (jnp.interp
        # clamps at the endpoint T), but passed through for signature
        # compatibility with the P-indexed branch.
        T_surface = float(T_values[-1])
        T_axis_is_radius = True
    elif temperature_function is None:
        T_axis_grid = np.linspace(5.0, 13.0, 4)
        T_values = np.full(4, 3000.0)
        T_surface = 3000.0
        T_axis_is_radius = False
    else:
        # P-indexed path. The temperature_function changes between outer
        # Picard iterations (blend, converged density update) but is
        # CONSTANT across brentq evals within one inner iteration. Cache
        # the tabulation by id(temperature_function) in
        # interpolation_cache so the 4000-point np.interp sweep only
        # runs once per outer iter, not per brentq call. Empirical:
        # removes ~5 ms/call * 5000 calls ~ 25 s of the JAX total
        # (cProfile 2026-04-23).
        _adia_cache = interpolation_cache.setdefault('_jax_adiabat_cache', {})
        _key = id(temperature_function)
        _entry = _adia_cache.get(_key)
        if _entry is None:
            _T_logP_grid, _T_values = _tabulate_adiabat(radii_arr, temperature_function)
            _T_surface = float(temperature_function(float(radii_arr[-1]), 1e5))
            _entry = (_T_logP_grid, _T_values, _T_surface)
            # Cap cache size so stale closures don't accumulate across
            # many outer iters (rare in practice; the _solve() control
            # flow creates ~10-20 distinct _temperature_func objects).
            if len(_adia_cache) > 64:
                _adia_cache.pop(next(iter(_adia_cache)))
            _adia_cache[_key] = _entry
        T_axis_grid, T_values, T_surface = _entry
        T_axis_is_radius = False

    if _PROFILE:
        _PHASE_TIMES['adiabat_tab'] += _time.perf_counter() - _p_t0
        _p_t0 = _time.perf_counter()

    # Mantle melting-curve tabulation on a shared log-P axis, sampled
    # in log-T. Numpy calls liquidus_func(P) and solidus_func(P)
    # directly inside calculate_mixed_density; previously the JAX RHS
    # used a hardcoded Stix14 power law that disagreed with
    # PALEOS-liquidus by 14-50 % (root cause of the +2.3 % R_outer
    # parity gap, see scripts/single_shot_ode_parity.py commit 9d275c7c).
    # Sampling log10(liquidus_func) and log10(solidus_func) on a
    # log-P axis (rather than T directly) makes linear interp bit-exact
    # for any Simon-Glatzel power law T = A*P^B, since log T is linear
    # in log P; piecewise power laws (PALEOS-liquidus) are also exact
    # except at the kink where the residual is ≤1e-7. This keeps N
    # small (256) so the JIT compile and per-call cost stay short.
    # Tabulating BOTH curves (rather than ``T_sol = T_liq * mzf``)
    # preserves generality: any (solidus, liquidus) pair the caller
    # passes via ``melting_curves_functions`` works, including
    # independently-tabulated pairs (e.g. Monteux600 solidus + liquidus)
    # where T_sol/T_liq varies with P. PROTEUS' usual convention of
    # ``solidus_func = liquidus_func * mushy_zone_factor`` flows through
    # unchanged because the wrapper just samples whatever solidus_func
    # returns.
    # Cache the melting-curve tabulation by (id(solidus_func), id(liquidus_func)).
    # The samples depend ONLY on the curve functions, so they are constant
    # across all solve_structure_via_jax calls within a single main() (and
    # across all main() calls that re-use the same closure pair). The
    # solver's outer Picard loop calls this wrapper ~5500 times per main()
    # on real CHILI T(r); without this cache we re-tabulate 256 × 2 = 512
    # melting-curve evaluations per call (~2.9M paleos_liquidus calls per
    # main()) and re-allocate / re-log10 / re-ascontiguousarray them every
    # time. cProfile 2026-04-25 attributed ~16 s of a 22.7 s real-CHILI
    # solve to this rebuild path. Cache key uses object id; the dict cap
    # prevents unbounded growth from unique-per-call closures (rare).
    _melt_cache = _MELT_TABLE_CACHE
    _key = (id(solidus_func), id(liquidus_func))
    _entry = _melt_cache.get(_key)
    if _entry is None:
        n_melt = 256
        log_p_axis = np.linspace(np.log10(1e8), np.log10(5e12), n_melt)
        p_axis = 10.0 ** log_p_axis
        T_liq_samples = np.array(
            [float(liquidus_func(P)) for P in p_axis], dtype=float
        )
        T_sol_samples = np.array(
            [float(solidus_func(P)) for P in p_axis], dtype=float
        )
        _entry = {
            'melt_log_p_min': float(log_p_axis[0]),
            'melt_dlog_p': float(log_p_axis[1] - log_p_axis[0]),
            'melt_n': int(n_melt),
            'log_T_liq_table': np.ascontiguousarray(
                np.log10(T_liq_samples), dtype=np.float64,
            ),
            'log_T_sol_table': np.ascontiguousarray(
                np.log10(T_sol_samples), dtype=np.float64,
            ),
        }
        if len(_melt_cache) > 64:
            _melt_cache.pop(next(iter(_melt_cache)))
        _melt_cache[_key] = _entry
    melt_curves = _entry

    # Physical constant G matching numpy path
    from ..constants import G

    jax_args = {
        'cmb_mass': float(cmb_mass),
        'T_axis_grid': T_axis_grid,
        'T_values': T_values,
        'T_surface': T_surface,
        'mushy_zone_factor_core': core_mzf,
        'G': float(G),
    }
    jax_args.update(_extract_sub_args(core_cached, 'core'))
    jax_args.update(_extract_liquidus(core_cached, 'core'))
    jax_args.update(_extract_sub_args(sol_cached, 'sol'))
    jax_args.update(_extract_sub_args(liq_cached, 'liq'))
    jax_args.update(melt_curves)

    global _CALL_COUNT, _TOTAL_WALL
    _CALL_COUNT += 1
    _t0 = _time.perf_counter()
    ys = solve_structure_jax(
        radii_arr, np.asarray(y0, dtype=float),
        rtol=float(relative_tolerance),
        atol=float(absolute_tolerance),
        T_axis_is_radius=T_axis_is_radius,
        **jax_args,
    )
    # np.asarray on a jnp.ndarray yields a read-only view of the JAX
    # buffer; np.array would copy (writable) but adds a ~1 ms host-device
    # sync per call, which dominates in coupled PROTEUS (75k calls per
    # main() -> ~88 s of pure sync, 57 % of wall, see
    # session_2026_04_24_scaffolding_gap_investigation.md). Keep this
    # cheap; callers that need to write to the return must take their
    # own writable copy at the write site. The one known write site
    # (solver._solve wall-timeout handler, line 614-622) now does so.
    ys = np.asarray(ys)
    _dt = _time.perf_counter() - _t0

    if _PROFILE:
        _PHASE_TIMES['jit_solve'] += _dt
        # Report cumulative every 200 calls
        if _CALL_COUNT % 200 == 0:
            total = sum(_PHASE_TIMES.values())
            print(
                f'[jax_profile] {_CALL_COUNT} calls | '
                f'cache_extract={_PHASE_TIMES["cache_extract"]:.2f}s '
                f'({100*_PHASE_TIMES["cache_extract"]/max(total,1e-9):.1f}%) | '
                f'adiabat_tab={_PHASE_TIMES["adiabat_tab"]:.2f}s '
                f'({100*_PHASE_TIMES["adiabat_tab"]/max(total,1e-9):.1f}%) | '
                f'jit_solve={_PHASE_TIMES["jit_solve"]:.2f}s '
                f'({100*_PHASE_TIMES["jit_solve"]/max(total,1e-9):.1f}%) | '
                f'total={total:.2f}s',
                flush=True,
            )
    _TOTAL_WALL += _dt
    if _DEBUG and (_CALL_COUNT <= 5 or _CALL_COUNT % 100 == 0):
        print(f'[jax_wrapper] call {_CALL_COUNT}: {_dt*1000:.1f} ms, cumulative '
              f'{_TOTAL_WALL:.2f} s', flush=True)
    mass_enclosed = ys[:, 0]
    gravity = ys[:, 1]
    pressure = ys[:, 2]

    # Pressure-zero terminal event post-processing. When the event fires
    # mid-grid, diffrax returns `inf` for all saveat entries past the
    # crossing (verified 2026-04-23). We replace those with the numpy
    # contract: mass/gravity carry the last valid value, pressure is
    # padded to 0. Matches structure_model.solve_structure's final pad.
    post_event = ~np.isfinite(pressure)
    if np.any(post_event):
        valid_idx = np.flatnonzero(~post_event)
        if valid_idx.size > 0:
            last_M = mass_enclosed[valid_idx[-1]]
            last_g = gravity[valid_idx[-1]]
            mass_enclosed = np.where(post_event, last_M, mass_enclosed)
            gravity = np.where(post_event, last_g, gravity)
            pressure = np.where(post_event, 0.0, pressure)
    return mass_enclosed, gravity, pressure
