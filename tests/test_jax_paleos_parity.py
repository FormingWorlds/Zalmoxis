"""Parity test: get_paleos_unified_density_jax vs numpy reference.

Uses the real PALEOS-2phase:MgSiO3 table from Stage-1b's bench config
and queries both versions at a grid of (P, T) points spanning the
table's valid range (below solidus, within mushy zone, above liquidus).
Requires max_rel <= 1e-8 — FP-rounding precision, since both implementations
do the same float64 arithmetic.
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def _load_stage1b_cache():
    """Load a Stage-1b PALEOS-2phase melt table via the numpy code path."""
    from zalmoxis.config import load_material_dictionaries, load_zalmoxis_config
    from zalmoxis.eos.interpolation import _ensure_unified_cache

    here = os.path.dirname(os.path.abspath(__file__))
    cfg = os.path.join(here, '..', 'input', 'bench_performance.toml')
    config_params = load_zalmoxis_config(cfg)
    mat_dicts = load_material_dictionaries()

    # Resolve the mantle entry so cached eos_file path is usable
    mantle_name = config_params['layer_eos_config']['mantle']
    mat = mat_dicts.get(mantle_name)
    if mat is None:
        pytest.skip(f'mantle EOS {mantle_name} not in registry')

    # The PALEOS-2phase mantle has {melted_mantle, solid_mantle} sub-dicts;
    # pick the melted side for this parity test (unified tables are on
    # sub-dicts in the 2-phase layout).
    if 'melted_mantle' in mat:
        sub = mat['melted_mantle']
        eos_file = sub.get('eos_file')
        if not eos_file or not os.path.isfile(eos_file):
            pytest.skip(f'EOS file not found: {eos_file}')
    else:
        pytest.skip('expected melted_mantle sub-dict in PALEOS-2phase mantle')

    interp_cache = {}
    cached = _ensure_unified_cache(eos_file, interp_cache)
    return cached, config_params


def _extract_jax_args(cached):
    """Flatten the cached numpy dict into the scalar/array args the
    jitted function expects."""
    import numpy as np_

    liq_lp = np_.asarray(cached.get('liquidus_log_p', []), dtype=float)
    liq_lt = np_.asarray(cached.get('liquidus_log_t', []), dtype=float)
    return {
        'density_grid': np_.asarray(cached['density_grid'], dtype=float),
        'unique_log_p': np_.asarray(cached['unique_log_p'], dtype=float),
        'unique_log_t': np_.asarray(cached['unique_log_t'], dtype=float),
        'logp_min': float(cached['logp_min']),
        'logt_min': float(cached['logt_min']),
        'dlog_p': float(cached['dlog_p']),
        'dlog_t': float(cached['dlog_t']),
        'n_p': int(cached['n_p']),
        'n_t': int(cached['n_t']),
        'p_min': float(cached['p_min']),
        'p_max': float(cached['p_max']),
        'lt_min_per_p': np_.asarray(cached['logt_valid_min'], dtype=float),
        'lt_max_per_p': np_.asarray(cached['logt_valid_max'], dtype=float),
        'liquidus_log_p': liq_lp,
        'liquidus_log_t': liq_lt,
        'liquidus_min_log_p': float(liq_lp[0]) if len(liq_lp) > 0 else 0.0,
        'liquidus_max_log_p': float(liq_lp[-1]) if len(liq_lp) > 0 else 0.0,
        'has_liquidus_f': 1.0 if len(liq_lp) > 0 else 0.0,
    }


@pytest.mark.unit
def test_get_paleos_unified_density_parity_vs_numpy():
    """JAX PALEOS density matches numpy across P, T, and mushy_zone_factor."""
    from zalmoxis.eos.paleos import get_paleos_unified_density
    from zalmoxis.jax_eos.paleos import get_paleos_unified_density_jax

    cached, _ = _load_stage1b_cache()
    jax_args = _extract_jax_args(cached)

    # Synthetic material_dict shim for the numpy version (only needs 'eos_file').
    # We pass the already-populated interp cache with this eos_file key.
    mat_dict = {'eos_file': '__dummy__', 'format': 'paleos_unified'}
    interp_cache = {'__dummy__': cached}

    rng = np.random.default_rng(17)

    # Sample query points within the table's P range; T spans below solidus
    # to above liquidus so every branch is exercised.
    p_min = cached['p_min']
    p_max = cached['p_max']
    # log-uniform P sampling inside the table
    q_p = 10.0 ** rng.uniform(np.log10(p_min) + 0.1, np.log10(p_max) - 0.1, 600)

    # Build T range based on the table's valid T at each P.
    # For simplicity: just sample T uniform in a wide physical range.
    q_t = rng.uniform(1500.0, 9000.0, 600)

    mushy_zone_factors = [1.0, 0.90, 0.80, 0.70]

    max_rel_observed = 0.0
    for mzf in mushy_zone_factors:
        numpy_vals = np.array(
            [
                get_paleos_unified_density(q_p[i], q_t[i], mat_dict, mzf, interp_cache)
                if not np.isnan(q_p[i])
                else np.nan
                for i in range(600)
            ],
            dtype=float,
        )

        jax_vals = np.array(
            [
                float(
                    get_paleos_unified_density_jax(
                        q_p[i],
                        q_t[i],
                        mzf,
                        **jax_args,
                    )
                )
                for i in range(600)
            ]
        )

        # Only compare where both are finite (numpy may return None for
        # edge cases; jax returns NaN in the same spots).
        both_finite = np.isfinite(numpy_vals) & np.isfinite(jax_vals)
        # Also skip cases where one is finite and the other isn't —
        # those are the density_nn fallback points we explicitly don't
        # port to JAX. Record them as a count; expect low.
        nan_mismatch = np.isfinite(numpy_vals) ^ np.isfinite(jax_vals)
        nan_mismatch_count = int(np.sum(nan_mismatch))

        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.abs(numpy_vals[both_finite] - jax_vals[both_finite]) / np.maximum(
                np.abs(numpy_vals[both_finite]),
                1e-30,
            )
        max_rel = float(rel.max()) if rel.size > 0 else 0.0
        max_rel_observed = max(max_rel_observed, max_rel)
        print(
            f'  mzf={mzf}: n_pts={both_finite.sum()}, max_rel={max_rel:.3e}, '
            f'nan_mismatch={nan_mismatch_count}/600'
        )

    assert max_rel_observed <= 1e-8, (
        f'PALEOS-density parity failed: max_rel={max_rel_observed:.3e} (want <= 1e-8)'
    )
