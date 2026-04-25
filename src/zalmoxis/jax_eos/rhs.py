"""JAX-native coupled_odes RHS for the 2-layer (core + mantle) planet.

Port of ``zalmoxis.structure_model.coupled_odes`` for the Stage-1b
configuration: a two-layer planet with PALEOS:iron (paleos_unified) core
and PALEOS-2phase:MgSiO3 (Tdep) mantle. Returns [dM/dr, dg/dr, dP/dr].

Layer dispatch is the only data-dependent branch. In numpy:

    if mass < cmb_mass:
        rho = paleos_unified(P, T, core_table, mzf_core)
    else:
        rho = tdep(P, T, T_sol, T_liq, solid_table, liquid_table)

In JAX this becomes: evaluate BOTH density functions, select the right
one with jnp.where. Both evaluations are cheap compiled kernels, so the
wasted work is small and the branching is trace-friendly.

Temperature lookup supports two axis conventions, selected by the
``T_axis_is_radius`` static flag:

  * ``False`` (default) — ``T_axis_grid`` is ``log10(P)`` and T comes
    from ``jnp.interp(log10(pressure), T_axis_grid, T_values)``. This
    matches Zalmoxis' internal adiabat path, where T along a column
    tracks P strongly and weakly depends on r. The wrapper builds the
    table via ``_tabulate_adiabat`` at a fixed ``r_mid`` across a
    log-uniform P grid.
  * ``True`` — ``T_axis_grid`` is a radial grid and T comes from
    ``jnp.interp(radius, T_axis_grid, T_values)``. Needed for callers
    whose temperature profile is genuinely r-indexed and P-ignored
    (e.g. PROTEUS' ``update_structure_from_interior``, where the
    closure interpolates SPIDER/Aragog's T(r) staggered grid). The
    P-indexed tabulation collapses to a constant for such callers,
    which wrecks the ODE (see
    ``tools/benchmarks/bench_coupled_tempfunc.py`` for the
    reproducer).

The two modes live in the same jitted function; JAX emits a separate
compiled variant per flag value.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .paleos import get_paleos_unified_density_jax
from .tdep import get_tdep_density_jax


@partial(jax.jit, static_argnames=('T_axis_is_radius',))
def coupled_odes_jax(
    radius: jnp.ndarray,
    y: jnp.ndarray,          # shape (3,): [mass, gravity, pressure]
    cmb_mass: float,
    # --- temperature lookup (see module docstring for axis conventions) ---
    T_axis_grid: jnp.ndarray,  # monotone increasing: log10(P) OR radius
    T_values: jnp.ndarray,     # matching T values on the axis grid
    T_surface: float,          # fallback when P <= 0 (P-indexed mode only)
    # --- core (paleos_unified) table ---
    mushy_zone_factor_core: jnp.ndarray,
    core_density_grid: jnp.ndarray,
    core_unique_log_p: jnp.ndarray,
    core_unique_log_t: jnp.ndarray,
    core_logp_min: float,
    core_logt_min: float,
    core_dlog_p: float,
    core_dlog_t: float,
    core_n_p: int,
    core_n_t: int,
    core_p_min: float,
    core_p_max: float,
    core_lt_min_per_p: jnp.ndarray,
    core_lt_max_per_p: jnp.ndarray,
    core_liquidus_log_p: jnp.ndarray,
    core_liquidus_log_t: jnp.ndarray,
    core_liquidus_min_log_p: float,
    core_liquidus_max_log_p: float,
    core_has_liquidus_f: jnp.ndarray,
    # --- mantle Tdep (solid + liquid sub-tables + melting curves) ---
    sol_density_grid: jnp.ndarray,
    sol_unique_log_p: jnp.ndarray,
    sol_unique_log_t: jnp.ndarray,
    sol_logp_min: float,
    sol_logt_min: float,
    sol_dlog_p: float,
    sol_dlog_t: float,
    sol_n_p: int,
    sol_n_t: int,
    sol_p_min: float,
    sol_p_max: float,
    sol_lt_min_per_p: jnp.ndarray,
    sol_lt_max_per_p: jnp.ndarray,
    liq_density_grid: jnp.ndarray,
    liq_unique_log_p: jnp.ndarray,
    liq_unique_log_t: jnp.ndarray,
    liq_logp_min: float,
    liq_logt_min: float,
    liq_dlog_p: float,
    liq_dlog_t: float,
    liq_n_p: int,
    liq_n_t: int,
    liq_p_min: float,
    liq_p_max: float,
    liq_lt_min_per_p: jnp.ndarray,
    liq_lt_max_per_p: jnp.ndarray,
    # Mantle liquidus and solidus tabulated on a UNIFORM log10(P [Pa])
    # axis as ``log10(T)`` values. The wrapper samples
    # ``log10(liquidus_func(P))`` and ``log10(solidus_func(P))`` on the
    # same regular grid (log_p_min + i * dlog_p, i = 0..N-1) and passes
    # the index parameters here so the RHS does an O(1) regular-grid
    # lookup (no binary search) — matches the pre-fix Stix14 inlined
    # power law in cost while keeping full liquidus generality. The
    # pattern mirrors paleos.get_paleos_unified_density_jax which uses
    # the same (log_p_min, dlog_p, n) trick. Linear interp on
    # (log_P, log_T) is bit-exact for any single power law T = A*P^B
    # and ≤1e-7 off the piecewise PALEOS-liquidus at the kink.
    # Tabulating BOTH curves (vs T_sol = T_liq * mzf) supports
    # independently-defined solidus/liquidus pairs.
    melt_log_p_min: float = 8.0,    # log10(P_min)
    melt_dlog_p: float = 0.0,       # log10(P) sample spacing
    melt_n: int = 0,                # number of samples
    log_T_liq_table: jnp.ndarray = None,   # type: ignore[assignment]
    log_T_sol_table: jnp.ndarray = None,   # type: ignore[assignment]
    # physical constant G (SI)
    G: float = 6.6743e-11,
    # Temperature-axis convention (static, selects P- vs r-indexed branch).
    # Kept at the end of the signature so older callers that rely on the
    # positional order of the EOS-cache args are not affected.
    T_axis_is_radius: bool = False,
):
    """Return dy/dr = [dM/dr, dg/dr, dP/dr] at (radius, y)."""
    mass, gravity, pressure = y[0], y[1], y[2]

    # Temperature at this (r, P). Two axis conventions — see module docstring.
    # P-indexed: interp on log10(max(P, 1)), fall back to T_surface for P<=0.
    # R-indexed: interp directly on radius. The P<=0 fallback is not needed
    # because the r-grid covers the full column from CMB to surface; any
    # overshoot beyond r[-1] is clamped by ``jnp.interp`` at the endpoint T.
    if T_axis_is_radius:
        temperature = jnp.interp(radius, T_axis_grid, T_values)
    else:
        log_p_for_T = jnp.log10(jnp.maximum(pressure, 1.0))
        T_interp = jnp.interp(log_p_for_T, T_axis_grid, T_values)
        temperature = jnp.where(pressure > 0, T_interp, T_surface)

    # Melting curves at this pressure: O(1) regular-grid lookup on
    # log_T tables (see arg comments above for the rationale). Compute
    # the float index, clip into [0, N-2], and linearly interp the two
    # bracketing samples. This avoids the O(log N) binary search
    # jnp.interp would do, matching the cost of the pre-fix Stix14
    # inlined power law.
    log_p_for_melt = jnp.log10(jnp.maximum(pressure, 1.0))
    melt_idx_f = (log_p_for_melt - melt_log_p_min) / melt_dlog_p
    melt_i = jnp.clip(jnp.floor(melt_idx_f).astype(jnp.int32), 0, melt_n - 2)
    melt_frac = melt_idx_f - melt_i.astype(log_p_for_melt.dtype)
    log_T_liq_lo = log_T_liq_table[melt_i]
    log_T_liq_hi = log_T_liq_table[melt_i + 1]
    log_T_sol_lo = log_T_sol_table[melt_i]
    log_T_sol_hi = log_T_sol_table[melt_i + 1]
    log_T_liq = (1.0 - melt_frac) * log_T_liq_lo + melt_frac * log_T_liq_hi
    log_T_sol = (1.0 - melt_frac) * log_T_sol_lo + melt_frac * log_T_sol_hi
    T_liq_interp = 10.0 ** log_T_liq
    T_sol_interp = 10.0 ** log_T_sol
    T_liq = jnp.where(pressure > 0, T_liq_interp, 0.0)
    T_sol = jnp.where(pressure > 0, T_sol_interp, 0.0)

    # Core density (paleos_unified)
    rho_core = get_paleos_unified_density_jax(
        pressure, temperature, mushy_zone_factor_core,
        core_density_grid, core_unique_log_p, core_unique_log_t,
        core_logp_min, core_logt_min, core_dlog_p, core_dlog_t,
        core_n_p, core_n_t, core_p_min, core_p_max,
        core_lt_min_per_p, core_lt_max_per_p,
        core_liquidus_log_p, core_liquidus_log_t,
        core_liquidus_min_log_p, core_liquidus_max_log_p,
        core_has_liquidus_f,
    )

    # Mantle density (Tdep 2-phase)
    rho_mantle = get_tdep_density_jax(
        pressure, temperature, T_sol, T_liq,
        sol_density_grid, sol_unique_log_p, sol_unique_log_t,
        sol_logp_min, sol_logt_min, sol_dlog_p, sol_dlog_t,
        sol_n_p, sol_n_t, sol_p_min, sol_p_max,
        sol_lt_min_per_p, sol_lt_max_per_p,
        liq_density_grid, liq_unique_log_p, liq_unique_log_t,
        liq_logp_min, liq_logt_min, liq_dlog_p, liq_dlog_t,
        liq_n_p, liq_n_t, liq_p_min, liq_p_max,
        liq_lt_min_per_p, liq_lt_max_per_p,
    )

    # Layer selection
    rho = jnp.where(mass < cmb_mass, rho_core, rho_mantle)

    # Match numpy's coupled_odes: return zeros when the EOS produces
    # a non-finite density (out-of-table T clamp, PALEOS edge case).
    # This mirrors structure_model.coupled_odes line 148 and is what
    # the test_jax_rhs_parity test filters out (both_nonzero mask).
    # Note: unlike the prior implementation, we DO NOT freeze on
    # pressure <= 0. That freeze prevented diffrax.Event from seeing
    # the pressure-zero downcrossing. Event-based termination now
    # handles P<=0 instead.
    rho_finite = jnp.isfinite(rho)
    rho_safe = jnp.where(rho_finite, rho, 1.0)

    # Standard structure ODEs
    dMdr = 4.0 * jnp.pi * radius ** 2 * rho_safe
    # At r=0, dgdr singular; use analytic limit dg/dr = 4 pi G rho / 3
    dgdr_gen = 4.0 * jnp.pi * G * rho_safe - 2.0 * gravity / jnp.where(radius > 0, radius, 1.0)
    dgdr_r0 = (4.0 / 3.0) * jnp.pi * G * rho_safe
    dgdr = jnp.where(radius > 0, dgdr_gen, dgdr_r0)
    dPdr = -rho_safe * gravity

    zero3 = jnp.zeros(3, dtype=dMdr.dtype)
    return jnp.where(rho_finite, jnp.stack([dMdr, dgdr, dPdr]), zero3)
