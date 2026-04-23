"""JAX-native bilinear interpolation and PALEOS per-cell clamping.

These are the scalar inner kernels that ``get_paleos_unified_density``
calls millions of times per Zalmoxis solve. The numpy versions live in
``zalmoxis.eos.interpolation``. The JAX versions here use ``jax.numpy``
exclusively so they can be ``jax.jit``-compiled into a diffrax ODE solve
where the entire RHS chain runs inside one XLA kernel.

Functional design: both functions are pure (no side effects, no dict
access, no Python branching on runtime values). The cached table data
is passed explicitly as flat arrays / floats rather than via a dict
lookup, because ``jax.jit`` cannot trace through Python dict access.

Parity: these must produce outputs matching their numpy counterparts to
within FP-rounding precision (rtol ~ 1e-15) on the same inputs — JIT
may reorder some operations but the math is identical.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


@jax.jit
def fast_bilinear_jax(
    log_p: jnp.ndarray,
    log_t: jnp.ndarray,
    grid: jnp.ndarray,
    unique_log_p: jnp.ndarray,
    unique_log_t: jnp.ndarray,
    logp_min: float,
    logt_min: float,
    dlog_p: float,
    dlog_t: float,
    n_p: int,
    n_t: int,
):
    """Log-uniform-grid O(1) bilinear interpolation, JAX scalar form.

    Mirrors ``zalmoxis.eos.interpolation._fast_bilinear`` for the fast
    log-uniform path. Takes the cache contents as explicit arrays so the
    function can be jit-compiled; callers should pre-extract these from
    the cached dict once.

    All inputs must be JAX tracers or Python scalars; grid/unique_* must
    be JAX arrays of the right shape.

    Returns a scalar density value (possibly NaN if the four corners
    contain NaN — JAX math propagates NaN faithfully so callers check
    with ``jnp.isnan`` afterwards, matching numpy's NaN handling).
    """
    # O(1) index computation for log-uniform grids
    fp = (log_p - logp_min) / dlog_p
    ft = (log_t - logt_min) / dlog_t

    # Lower bounding indices (clamp to valid range)
    ip = jnp.clip(jnp.floor(fp).astype(jnp.int32), 0, n_p - 2)
    it = jnp.clip(jnp.floor(ft).astype(jnp.int32), 0, n_t - 2)

    # Fractional positions within cell (using per-cell spans from unique_*)
    span_p = unique_log_p[ip + 1] - unique_log_p[ip]
    span_t = unique_log_t[it + 1] - unique_log_t[it]

    # Guard against zero spans (degenerate cells)
    dp = jnp.where(span_p > 0, (log_p - unique_log_p[ip]) / span_p, 0.0)
    dt = jnp.where(span_t > 0, (log_t - unique_log_t[it]) / span_t, 0.0)
    dp = jnp.clip(dp, 0.0, 1.0)
    dt = jnp.clip(dt, 0.0, 1.0)

    # Four corners
    v00 = grid[ip, it]
    v01 = grid[ip, it + 1]
    v10 = grid[ip + 1, it]
    v11 = grid[ip + 1, it + 1]

    return v00 * (1 - dp) * (1 - dt) + v01 * (1 - dp) * dt + v10 * dp * (1 - dt) + v11 * dp * dt


@jax.jit
def paleos_clamp_temperature_jax(
    log_p: jnp.ndarray,
    log_t: jnp.ndarray,
    lt_min: jnp.ndarray,
    lt_max: jnp.ndarray,
    logp_min: float,
    dlog_p: float,
    n_p: int,
):
    """Clamp log10(T) to the per-pressure valid range of a PALEOS table.

    Mirrors ``zalmoxis.eos.interpolation._paleos_clamp_temperature``'s
    fast log-uniform path. Returns the clamped log_t; the was_clamped
    flag returned by the numpy version is dropped here because the only
    user of that flag was a one-shot logging warning that doesn't need
    tracing through JIT.

    When bounds are NaN near table edges, returns the input log_t
    unclamped (matching numpy behaviour).
    """
    # O(1) index on log-uniform P grid
    fp = (log_p - logp_min) / dlog_p
    ip = jnp.clip(jnp.floor(fp).astype(jnp.int32), 0, n_p - 2)
    frac = jnp.clip(fp - ip, 0.0, 1.0)
    local_tmin = lt_min[ip] + frac * (lt_min[ip + 1] - lt_min[ip])
    local_tmax = lt_max[ip] + frac * (lt_max[ip + 1] - lt_max[ip])

    # NaN-bound fallback: if bounds are NaN, return input log_t unchanged.
    # jnp.where is evaluated on both branches, so we can't short-circuit;
    # but both branches are cheap scalar math.
    bounds_ok = jnp.isfinite(local_tmin) & jnp.isfinite(local_tmax)

    clamped = jnp.clip(log_t, local_tmin, local_tmax)
    return jnp.where(bounds_ok, clamped, log_t)
