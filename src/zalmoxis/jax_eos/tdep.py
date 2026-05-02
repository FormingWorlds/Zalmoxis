"""JAX-native T-dependent EOS density with solid+liquid sub-tables.

Port of ``zalmoxis.eos.tdep.get_Tdep_density`` for PALEOS-2phase (the
Stage-1b mantle). Uses two PALEOS-format tables (solid-side and melt-
side); in the mushy zone, density is a volume-average of per-table
bilinear lookups.

Each sub-table query is a simple bilinear (no mushy-zone-within-table
logic — the inter-table mix IS the mushy logic). We reuse
``fast_bilinear_jax`` + ``paleos_clamp_temperature_jax`` from the
bilinear kernel module.

NaN-fallback (numpy's density_nn KDTree) is NOT ported. On the Stage-1b
P-T envelope the fallback is not exercised (verified via Step 2 parity
tests).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .bilinear import fast_bilinear_jax, paleos_clamp_temperature_jax


@jax.jit
def _bilinear_with_clamp_jax(
    log_p,
    log_t,
    density_grid,
    unique_log_p,
    unique_log_t,
    logp_min,
    logt_min,
    dlog_p,
    dlog_t,
    n_p,
    n_t,
    lt_min_per_p,
    lt_max_per_p,
):
    """Per-cell-clamped bilinear, shared between solid/melt sub-tables."""
    log_t_c = paleos_clamp_temperature_jax(
        log_p,
        log_t,
        lt_min_per_p,
        lt_max_per_p,
        logp_min,
        dlog_p,
        n_p,
    )
    return fast_bilinear_jax(
        log_p,
        log_t_c,
        density_grid,
        unique_log_p,
        unique_log_t,
        logp_min,
        logt_min,
        dlog_p,
        dlog_t,
        n_p,
        n_t,
    )


@jax.jit
def get_tdep_density_jax(
    pressure: jnp.ndarray,
    temperature: jnp.ndarray,
    T_sol: jnp.ndarray,  # solidus_func(P) — from external melting curve
    T_liq: jnp.ndarray,  # liquidus_func(P)
    # --- solid-side table (indexed with _s suffix) ---
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
    # --- liquid-side table (indexed with _l suffix) ---
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
):
    """Return PALEOS-2phase density at (pressure, temperature) in kg/m^3.

    Mirrors ``zalmoxis.eos.tdep.get_Tdep_density``. T_sol and T_liq are
    evaluated by the caller via external melting-curve functions; here
    we receive them as scalar inputs so the branch structure stays pure
    jnp math.

    NaN-coverage fallback (T_sol or T_liq being NaN from the melting
    curve) is mapped to the solid-phase branch, matching numpy
    behaviour at line 137-147 of tdep.py.
    """
    # Clamp pressures per-sub-table then log10
    log_p_s = jnp.log10(jnp.clip(pressure, sol_p_min, sol_p_max))
    log_p_l = jnp.log10(jnp.clip(pressure, liq_p_min, liq_p_max))
    log_t = jnp.log10(jnp.where(temperature > 1.0, temperature, 1.0))

    rho_solid = _bilinear_with_clamp_jax(
        log_p_s,
        log_t,
        sol_density_grid,
        sol_unique_log_p,
        sol_unique_log_t,
        sol_logp_min,
        sol_logt_min,
        sol_dlog_p,
        sol_dlog_t,
        sol_n_p,
        sol_n_t,
        sol_lt_min_per_p,
        sol_lt_max_per_p,
    )
    rho_liquid = _bilinear_with_clamp_jax(
        log_p_l,
        log_t,
        liq_density_grid,
        liq_unique_log_p,
        liq_unique_log_t,
        liq_logp_min,
        liq_logt_min,
        liq_dlog_p,
        liq_dlog_t,
        liq_n_p,
        liq_n_t,
        liq_lt_min_per_p,
        liq_lt_max_per_p,
    )

    # Volume-averaged density in mushy zone.
    # Guard against degenerate melting curves where T_liq <= T_sol: then
    # numpy falls back to pure-liquid. We preserve that behaviour.
    safe_dT = jnp.where(T_liq > T_sol, T_liq - T_sol, 1.0)
    frac_melt_raw = (temperature - T_sol) / safe_dT
    # Smoothstep ramp on the mushy-zone melt fraction. Linear frac_melt
    # gives a slope discontinuity in 1/rho at T=T_sol and T=T_liq because
    # d(1/rho)/dT inside the lever rule is (1/rho_liq - 1/rho_sol)/(T_liq -
    # T_sol) (constant) while outside it is the small thermal-expansion
    # slope of the pure-phase EOS. The kink causes inner Picard plateau
    # at diff~0.1 on hot mushy profiles. Smoothstep s = x*x*(3 - 2*x)
    # has ds/dx = 0 at x=0 and x=1, so the lever rule's slope vanishes
    # at the boundaries, eliminating the kink while preserving the
    # midpoint value (s(0.5) = 0.5 = linear). frac_melt_raw is clipped
    # to [0,1] before the polynomial; outside the mushy zone the
    # selection below picks pure-phase rho so the value here is unused.
    x = jnp.clip(frac_melt_raw, 0.0, 1.0)
    frac_melt = x * x * (3.0 - 2.0 * x)
    specific_volume = frac_melt * (1.0 / rho_liquid) + (1.0 - frac_melt) * (1.0 / rho_solid)
    rho_mixed = 1.0 / specific_volume

    # Branch selection.
    # If melting curves return NaN, default to solid phase (line 137-147 numpy).
    melt_curve_valid = jnp.isfinite(T_sol) & jnp.isfinite(T_liq)
    # If T_liq <= T_sol (degenerate), use liquid (line 174-181 numpy).
    liq_above_sol = T_liq > T_sol

    is_below_sol = temperature <= T_sol
    is_above_liq = temperature >= T_liq
    # Mushy zone is the implicit fall-through of the jnp.where below: not
    # is_below_sol AND not is_above_liq AND liq_above_sol → rho_mixed.

    # Selection:
    #   if not melt_curve_valid: rho_solid
    #   elif is_above_liq OR (not liq_above_sol AND not is_below_sol): rho_liquid
    #   elif is_below_sol: rho_solid
    #   else: rho_mixed (the implicit mushy branch)
    rho = jnp.where(
        ~melt_curve_valid,
        rho_solid,
        jnp.where(
            is_above_liq | (~liq_above_sol & ~is_below_sol),
            rho_liquid,
            jnp.where(is_below_sol, rho_solid, rho_mixed),
        ),
    )
    return rho
