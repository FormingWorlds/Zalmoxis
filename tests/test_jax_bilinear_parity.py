"""Parity tests for JAX ports of the PALEOS bilinear + clamp kernels.

The JAX kernels must match their numpy reference implementations to
within FP-rounding precision on the same inputs. This test builds a
synthetic cached table dict with realistic shape, queries both
implementations at random (log_p, log_t) points, and asserts element-
wise relative error <= 1e-12 (a few ULPs of float64).
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
def test_fast_bilinear_parity_vs_numpy():
    """fast_bilinear_jax matches _fast_bilinear to FP precision."""
    from zalmoxis.eos.interpolation import _fast_bilinear
    from zalmoxis.jax_eos.bilinear import fast_bilinear_jax

    rng = np.random.default_rng(42)

    n_p = 64
    n_t = 48
    logp_min, logp_max = 5.0, 13.0
    logt_min, logt_max = 2.0, 5.0
    unique_log_p = np.linspace(logp_min, logp_max, n_p)
    unique_log_t = np.linspace(logt_min, logt_max, n_t)
    # Smooth synthetic density surface: rho(P, T) = A * P^alpha * T^-beta
    grid = (
        4000.0
        * (10.0 ** unique_log_p[:, None]) ** 0.15
        * (10.0 ** unique_log_t[None, :]) ** (-0.05)
    )

    dlog_p = (logp_max - logp_min) / (n_p - 1)
    dlog_t = (logt_max - logt_min) / (n_t - 1)

    cached = {
        'unique_log_p': unique_log_p,
        'unique_log_t': unique_log_t,
        'logp_min': logp_min,
        'logt_min': logt_min,
        'dlog_p': dlog_p,
        'dlog_t': dlog_t,
        'n_p': n_p,
        'n_t': n_t,
    }

    # Sample random query points inside the grid
    q_lp = rng.uniform(logp_min, logp_max, 500)
    q_lt = rng.uniform(logt_min, logt_max, 500)

    numpy_vals = np.array([_fast_bilinear(q_lp[i], q_lt[i], grid, cached) for i in range(500)])
    jax_vals = np.array(
        [
            float(
                fast_bilinear_jax(
                    q_lp[i],
                    q_lt[i],
                    grid,
                    unique_log_p,
                    unique_log_t,
                    logp_min,
                    logt_min,
                    dlog_p,
                    dlog_t,
                    n_p,
                    n_t,
                )
            )
            for i in range(500)
        ]
    )

    max_rel = np.max(np.abs(numpy_vals - jax_vals) / np.maximum(np.abs(numpy_vals), 1e-30))
    # JIT reorders add/mul associativity; ULP-scale differences are expected
    assert max_rel <= 1e-12, f'bilinear parity failed: max_rel={max_rel:.3e}'


@pytest.mark.unit
def test_paleos_clamp_temperature_parity_vs_numpy():
    """paleos_clamp_temperature_jax matches _paleos_clamp_temperature."""
    from zalmoxis.eos.interpolation import _paleos_clamp_temperature
    from zalmoxis.jax_eos.bilinear import paleos_clamp_temperature_jax

    rng = np.random.default_rng(7)

    n_p = 64
    logp_min, logp_max = 5.0, 13.0
    unique_log_p = np.linspace(logp_min, logp_max, n_p)
    dlog_p = (logp_max - logp_min) / (n_p - 1)

    # Synthetic per-P valid bounds: T range expands with P (common for PALEOS)
    lt_min = 2.0 + 0.1 * np.arange(n_p) / (n_p - 1)
    lt_max = 4.5 + 0.2 * np.arange(n_p) / (n_p - 1)

    cached = {
        'unique_log_p': unique_log_p,
        'logp_min': logp_min,
        'dlog_p': dlog_p,
        'n_p': n_p,
        'logt_valid_min': lt_min,
        'logt_valid_max': lt_max,
    }

    # Mix of in-range, below-range, above-range query points
    q_lp = rng.uniform(logp_min, logp_max, 500)
    q_lt = np.concatenate(
        [
            rng.uniform(1.5, 5.0, 250),  # some out-of-range
            rng.uniform(2.2, 4.4, 250),  # most in-range
        ]
    )

    numpy_clamped = np.array(
        [_paleos_clamp_temperature(q_lp[i], q_lt[i], cached)[0] for i in range(500)]
    )
    jax_clamped = np.array(
        [
            float(
                paleos_clamp_temperature_jax(
                    q_lp[i],
                    q_lt[i],
                    lt_min,
                    lt_max,
                    logp_min,
                    dlog_p,
                    n_p,
                )
            )
            for i in range(500)
        ]
    )

    max_rel = np.max(
        np.abs(numpy_clamped - jax_clamped) / np.maximum(np.abs(numpy_clamped), 1e-30)
    )
    assert max_rel <= 1e-12, f'clamp parity failed: max_rel={max_rel:.3e}'
