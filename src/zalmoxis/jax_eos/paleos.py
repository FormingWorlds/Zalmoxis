"""JAX-native PALEOS unified density lookup with mushy-zone blending.

Port of ``zalmoxis.eos.paleos.get_paleos_unified_density`` to jax.numpy.
Must produce densities matching the numpy version to within FP-rounding
precision on identical (P, T) queries.

The numpy version has five data-dependent branches:
    1. No mushy zone (mushy_zone_factor >= 1 or no liquidus data)
       → direct bilinear lookup at (P, T_clamped)
    2. Query pressure outside liquidus coverage
       → direct bilinear lookup
    3. T >= T_liq (above liquidus)
       → direct bilinear lookup
    4. T <= T_sol (below solidus)
       → direct bilinear lookup
    5. T_sol < T < T_liq (in mushy zone)
       → volume-average between rho(P, T_sol) and rho(P, T_liq)

In JAX, we compute the result of ALL applicable branches and use
``jnp.where`` to select the right one. Branches 1-4 all evaluate to the
same bilinear at (P, T_clamped), so we collapse them into a "direct"
branch. The mushy-zone branch is the only distinct path.

NaN-on-lookup fallback (numpy's ``density_nn`` KDTree nearest-neighbour)
is NOT ported — the JAX path returns NaN, which the caller must check
and fall back to numpy for. On Stage-1b-equivalent configs with
well-covered PALEOS tables, NaN returns are not expected.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .bilinear import fast_bilinear_jax, paleos_clamp_temperature_jax

# Phase-boundary guard offset (K); matches _DT_PHASE_GUARD in numpy paleos.py.
_DT_PHASE_GUARD = 1.0


@jax.jit
def get_paleos_unified_density_jax(
    pressure: jnp.ndarray,
    temperature: jnp.ndarray,
    mushy_zone_factor: jnp.ndarray,
    # --- cached table arrays (extracted once from the numpy cache dict) ---
    density_grid: jnp.ndarray,  # (n_p, n_t)
    unique_log_p: jnp.ndarray,  # (n_p,)
    unique_log_t: jnp.ndarray,  # (n_t,)
    logp_min: float,
    logt_min: float,
    dlog_p: float,
    dlog_t: float,
    n_p: int,
    n_t: int,
    p_min: float,
    p_max: float,
    lt_min_per_p: jnp.ndarray,  # (n_p,) per-P valid log_t min
    lt_max_per_p: jnp.ndarray,  # (n_p,) per-P valid log_t max
    liquidus_log_p: jnp.ndarray,  # (n_liq,) — the liquidus curve in log_p
    liquidus_log_t: jnp.ndarray,  # (n_liq,) — matching log_t
    liquidus_min_log_p: float,  # liquidus_log_p[0] as scalar
    liquidus_max_log_p: float,  # liquidus_log_p[-1] as scalar
    has_liquidus_f: jnp.ndarray,  # 1.0 if liquidus data present, else 0.0
):
    """Return PALEOS-unified density at (pressure, temperature) in kg/m^3.

    Mirrors ``zalmoxis.eos.paleos.get_paleos_unified_density``. Inputs
    are scalars or 0-d arrays; returns a scalar. NaN on lookup failure
    (caller must fall back to numpy path for those rare points).
    """
    # Clamp pressure to table bounds, then log10
    pressure_c = jnp.clip(pressure, p_min, p_max)
    log_p = jnp.log10(pressure_c)
    log_t = jnp.log10(jnp.where(temperature > 1.0, temperature, 1.0))

    # Initial per-cell clamp of query log_t
    log_t_clamped = paleos_clamp_temperature_jax(
        log_p,
        log_t,
        lt_min_per_p,
        lt_max_per_p,
        logp_min,
        dlog_p,
        n_p,
    )

    # Direct lookup at (log_p, log_t_clamped): used when no mushy zone,
    # or P outside liquidus coverage, or T above/below the mushy range.
    rho_direct = fast_bilinear_jax(
        log_p,
        log_t_clamped,
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

    # Mushy zone applies only if factor < 1 AND table has liquidus data.
    # has_liquidus_f is a float (1.0 or 0.0) so this condition can be
    # expressed without Python booleans (JIT requires jnp.where).
    use_mushy = (mushy_zone_factor < 1.0) & (has_liquidus_f > 0.5)

    # Liquidus T at this pressure (from the stored liquidus curve; 1D interp)
    log_t_melt = jnp.interp(log_p, liquidus_log_p, liquidus_log_t)
    T_melt = 10.0**log_t_melt
    T_liq = jnp.maximum(T_melt, T_melt + _DT_PHASE_GUARD)
    T_sol_raw = T_melt * mushy_zone_factor
    T_sol = jnp.minimum(T_sol_raw, T_melt - _DT_PHASE_GUARD)
    log_t_sol = jnp.log10(jnp.maximum(T_sol, 1.0))
    log_t_liq = jnp.log10(T_liq)

    # Inline per-P clamp bounds shared between T_sol and T_liq (W3 pattern).
    fp = (log_p - logp_min) / dlog_p
    ip = jnp.clip(jnp.floor(fp).astype(jnp.int32), 0, n_p - 2)
    frac = jnp.clip(fp - ip, 0.0, 1.0)
    local_tmin = lt_min_per_p[ip] + frac * (lt_min_per_p[ip + 1] - lt_min_per_p[ip])
    local_tmax = lt_max_per_p[ip] + frac * (lt_max_per_p[ip + 1] - lt_max_per_p[ip])
    bounds_ok = jnp.isfinite(local_tmin) & jnp.isfinite(local_tmax)

    log_t_sol_c = jnp.where(bounds_ok, jnp.clip(log_t_sol, local_tmin, local_tmax), log_t_sol)
    log_t_liq_c = jnp.where(bounds_ok, jnp.clip(log_t_liq, local_tmin, local_tmax), log_t_liq)

    # Bilinear at solidus and liquidus edges
    rho_sol = fast_bilinear_jax(
        log_p,
        log_t_sol_c,
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
    rho_liq = fast_bilinear_jax(
        log_p,
        log_t_liq_c,
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

    # Volume-averaged density in mushy zone: 1 / (phi/rho_liq + (1-phi)/rho_sol)
    phi = (temperature - T_sol) / (T_liq - T_sol)
    specific_volume = phi * (1.0 / rho_liq) + (1.0 - phi) * (1.0 / rho_sol)
    rho_mushy = 1.0 / specific_volume

    # Branch selection: above liquidus OR below solidus → direct lookup
    # (numpy uses `temperature >= T_liq` and `temperature <= T_sol`).
    # Inside mushy zone → volume-averaged.
    # Outside liquidus P coverage → direct lookup.
    in_liquidus_coverage = (log_p >= liquidus_min_log_p) & (log_p <= liquidus_max_log_p)
    in_mushy_zone = (temperature > T_sol) & (temperature < T_liq)

    use_mushy_at_point = use_mushy & in_liquidus_coverage & in_mushy_zone

    return jnp.where(use_mushy_at_point, rho_mushy, rho_direct)
