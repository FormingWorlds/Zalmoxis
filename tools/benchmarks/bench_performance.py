#!/usr/bin/env python
"""Performance-validation bench for the Zalmoxis speedup plan.

For every commit on `tl/interior-refactor-performance`, run this to:
  1. Time 3 standalone `zalmoxis.solver.main()` calls on the reference config.
  2. Save output profiles (radii, mass_enclosed, gravity, pressure, density,
     temperature) to a .npz.
  3. Compare to an existing baseline .npz (if provided) and report element-wise
     relative error; fail if it exceeds the commit-type tolerance.

Usage
-----
Save baseline after commit 0:
    python -m tools.benchmarks.bench_performance \
        --config=input/bench_performance.toml \
        --out=tests/data/bench_performance_baseline.npz \
        --save-baseline

Validate after a refactor commit:
    python -m tools.benchmarks.bench_performance \
        --config=input/bench_performance.toml \
        --baseline=tests/data/bench_performance_baseline.npz \
        --out=/tmp/bench_performance_latest.npz \
        --tolerance=pure-refactor

    # pure-refactor: requires bit-identical match (np.array_equal)
    # table-prebake: requires rtol<=1e-10 on all profiles

The script exits non-zero on validation failure, suitable for pre-commit gating.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

# Resolve zalmoxis imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

import zalmoxis as _zal
from zalmoxis.config import (
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
)
from zalmoxis.solver import main

PROFILE_KEYS = (
    'radii',
    'mass_enclosed',
    'gravity',
    'pressure',
    'temperature',
    'density',
)

SCALAR_KEYS = (
    'cmb_mass',
    'core_mantle_mass',
    'converged',
    'converged_pressure',
    'converged_density',
    'converged_mass',
)
# total_time is the solver's own wall-clock measurement; not a physics output.
# Excluded from pure-refactor comparison (varies run-to-run by ~100 ms).


def run_once(config_params, mat_dicts, melt_funcs, input_dir):
    """Run one zalmoxis.solver.main() call and return (wall, result)."""
    t0 = time.perf_counter()
    result = main(
        config_params,
        material_dictionaries=mat_dicts,
        melting_curves_functions=melt_funcs,
        input_dir=input_dir,
    )
    wall = time.perf_counter() - t0
    return wall, result


def extract_output(result):
    """Pull the reproducible profiles + scalars from a main() result dict."""
    data = {}
    for k in PROFILE_KEYS:
        v = result.get(k)
        data[k] = np.asarray(v) if v is not None else np.array([])
    for k in SCALAR_KEYS:
        v = result.get(k)
        if isinstance(v, (bool, np.bool_)):
            data[k] = np.array([int(v)])
        elif v is None:
            data[k] = np.array([np.nan])
        else:
            data[k] = np.array([float(v)])
    return data


def compare(baseline, current, tolerance):
    """Compare per-key; return (ok, report dict)."""
    report = {}
    all_ok = True
    for k in PROFILE_KEYS + SCALAR_KEYS:
        b = baseline[k]
        c = current[k]
        if b.shape != c.shape:
            report[k] = f"shape mismatch: baseline {b.shape} vs current {c.shape}"
            all_ok = False
            continue

        if tolerance == 'pure-refactor':
            ok = np.array_equal(b, c)
            if not ok:
                diff = np.abs(b - c)
                report[k] = (
                    f"FAIL pure-refactor: max|delta|={float(diff.max()):.3e} "
                    f"(want bit-identical)"
                )
                all_ok = False
            else:
                report[k] = "PASS (bit-identical)"

        elif tolerance == 'table-prebake':
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.abs(b - c) / np.maximum(np.abs(b), 1e-30)
                max_rel = float(np.nan_to_num(rel, nan=0.0, posinf=0.0).max())
            ok = max_rel <= 1e-10
            if not ok:
                report[k] = f"FAIL table-prebake: max_rel={max_rel:.3e} (want <=1e-10)"
                all_ok = False
            else:
                report[k] = f"PASS (max_rel={max_rel:.3e})"

        elif tolerance == 'solver-tolerance':
            # For JIT / solver-method swaps where FP ordering legitimately
            # differs. Tolerance matches Zalmoxis's own solver rtol (1e-5)
            # scaled by ~10x to allow Picard-loop FP accumulation; any
            # larger drift is a real physics difference.
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.abs(b - c) / np.maximum(np.abs(b), 1e-30)
                max_rel = float(np.nan_to_num(rel, nan=0.0, posinf=0.0).max())
            threshold = 1e-4
            ok = max_rel <= threshold
            if not ok:
                report[k] = (
                    f"FAIL solver-tolerance: max_rel={max_rel:.3e} "
                    f"(want <={threshold:.0e})"
                )
                all_ok = False
            else:
                report[k] = f"PASS (max_rel={max_rel:.3e})"

        else:
            raise ValueError(f"Unknown tolerance policy: {tolerance}")

    return all_ok, report


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--baseline', default=None,
                        help='Path to baseline .npz (compare to it).')
    parser.add_argument('--out', default=None,
                        help='Path to write the current-run .npz.')
    parser.add_argument('--save-baseline', action='store_true',
                        help='Write --out as new baseline and skip compare.')
    parser.add_argument('--tolerance',
                        choices=['pure-refactor', 'table-prebake', 'solver-tolerance'],
                        default='pure-refactor')
    parser.add_argument('--n-runs', type=int, default=3,
                        help='Number of timing runs (output captured on last run).')
    parser.add_argument('--use-jax', action='store_true',
                        help='Route the structure ODE through the JAX+diffrax path '
                             '(config_params["use_jax"]=True).')
    parser.add_argument('--use-anderson', action='store_true',
                        help='Enable Anderson acceleration on the density Picard '
                             'loop (config_params["use_anderson"]=True).')
    parser.add_argument('--anderson-m-max', type=int, default=5,
                        help='Anderson history window (default: 5).')
    args = parser.parse_args()

    config_params = load_zalmoxis_config(args.config)
    # Override wall_timeout: the solver's 300 s default is a sanity cap,
    # not a performance target. Under that cap PALEOS-2phase Stage 1b runs
    # bail out with a "best solution" rather than true convergence, which
    # makes bench results non-comparable between commits. Set 1 h here so
    # runs actually converge.
    config_params['wall_timeout'] = 3600.0
    if args.use_jax:
        config_params['use_jax'] = True
        print("[bench] use_jax=True (routing structure ODE via diffrax)")
    if args.use_anderson:
        config_params['use_anderson'] = True
        config_params['anderson_m_max'] = args.anderson_m_max
        print(f"[bench] use_anderson=True (m_max={args.anderson_m_max})")
    layer_eos_config = config_params['layer_eos_config']
    mat_dicts = load_material_dictionaries()
    melt_funcs = load_solidus_liquidus_functions(
        layer_eos_config,
        config_params.get('rock_solidus', 'Stixrude14-solidus'),
        config_params.get('rock_liquidus', 'Stixrude14-liquidus'),
    )
    _input_dir = os.path.normpath(
        os.path.join(os.path.dirname(_zal.__file__), '..', '..', 'input')
    )

    walls = []
    result = None
    for i in range(args.n_runs):
        wall, result = run_once(config_params, mat_dicts, melt_funcs, _input_dir)
        walls.append(wall)
        print(f"run {i + 1}/{args.n_runs}: wall = {wall:.2f} s "
              f"(converged={result.get('converged', False)})")

    walls_arr = np.array(walls)
    print(f"\nwall mean = {walls_arr.mean():.2f} s, "
          f"min = {walls_arr.min():.2f} s, "
          f"max = {walls_arr.max():.2f} s, "
          f"std = {walls_arr.std():.2f} s")

    current = extract_output(result)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        np.savez(args.out, **current)
        print(f"wrote current output: {args.out}")

    if args.save_baseline:
        if args.out is None:
            print("error: --save-baseline requires --out", file=sys.stderr)
            sys.exit(2)
        print("SAVED as baseline; no comparison performed.")
        sys.exit(0)

    if args.baseline:
        baseline = dict(np.load(args.baseline))
        ok, report = compare(baseline, current, args.tolerance)
        print("\n=== per-key comparison ({}) ===".format(args.tolerance))
        for k, v in report.items():
            print(f"  {k:25s}  {v}")
        if not ok:
            print("\nFAIL: results drifted beyond tolerance.", file=sys.stderr)
            sys.exit(1)
        print("\nPASS: all keys within tolerance.")

    sys.exit(0)


if __name__ == '__main__':
    main_cli()
