#!/usr/bin/env python
"""PROTEUS-mimic bench: new ``temperature_function`` closure per ``main()`` call.

Why this bench exists
---------------------
``bench_performance.py`` runs ``main()`` three times without a
``temperature_function``, so the JAX wrapper exercises the 3000-K fallback at
``jax_eos/wrapper.py:_tabulate_adiabat`` and never touches the adiabat cache.
In the PROTEUS coupled loop, ``update_structure_from_interior`` hands a fresh
closure to Zalmoxis on every structure update, with a slightly different
``T(r)`` each time. That stresses a path the existing benches never touch.

This script reproduces that stress standalone, so we can decide where the
per-call wall time in PROTEUS actually goes before touching code. Use the
``ZALMOXIS_JAX_PROFILE=1`` env var to see the phase breakdown
(``cache_extract`` / ``adiabat_tab`` / ``jit_solve`` / ``other``) that
``jax_eos/wrapper.py`` emits every 200 calls; in this bench that emission
fires once per structure solve (3-5 k solve_structure_via_jax calls per
``main()``).

Modes
-----
* ``none``    -> no ``temperature_function`` (matches ``bench_performance``).
* ``static``  -> one fixed ``temperature_function`` reused across the N runs
                 (bench idealisation: cache hits across main() calls).
* ``rotating``-> a *new* closure with a slightly perturbed ``T_asc`` array
                 on every run (PROTEUS-like: closure id changes every call).

``--use-temperature-arrays`` routes the T profile through the explicit
r-indexed kwarg ``(r_arr, T_arr)`` instead of via a Python callable. This
exercises the JAX RHS' r-indexed branch (``T_axis_is_radius=True`` in
``jax_eos/rhs.py``), which is the convergence-correct path for PROTEUS-style
r-only T profiles. See the ``solve_structure_via_jax`` docstring for when
to prefer this form.

Usage
-----
    ZALMOXIS_JAX_PROFILE=1 python -m tools.benchmarks.bench_coupled_tempfunc \
        --config=input/bench_performance.toml --use-jax --use-anderson \
        --n-runs=3 --mode=rotating --use-temperature-arrays
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src'))

import zalmoxis as _zal
from zalmoxis.config import (
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
)
from zalmoxis.solver import main


def build_r_T_arrays(
    r_cmb: float,
    r_surface: float,
    t_surface: float,
    t_cmb: float,
    n_pts: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a CMB-to-surface ``(r, T)`` profile resembling a mid-evolution
    PROTEUS mantle: CMB-anchored linear-ish cooling.

    80 points is close to Aragog's default staggered-node grid (79 cells).
    """
    r = np.linspace(r_cmb, r_surface, n_pts)
    T = t_cmb + (t_surface - t_cmb) * (r - r_cmb) / (r_surface - r_cmb)
    return r, T


def make_temperature_function(r_arr: np.ndarray, T_arr: np.ndarray, r_cmb: float, t_cmb: float):
    """Return a fresh P-independent callable matching PROTEUS's shape.

    Mirrors ``src/proteus/interior_energetics/wrapper.py:1246-1249``:
    below the CMB it pins ``T_cmb``; above, it linearly interpolates T(r).
    Ignoring P is the same simplification PROTEUS's wrapper uses.
    """
    _r = np.asarray(r_arr, dtype=float)
    _T = np.asarray(T_arr, dtype=float)
    _R_cmb = float(r_cmb)
    _T_cmb = float(t_cmb)

    def _temperature_function(r, P):
        if r <= _R_cmb:
            return _T_cmb
        return float(np.interp(r, _r, _T))

    return _temperature_function


def run_once(
    config_params,
    mat_dicts,
    melt_funcs,
    input_dir,
    *,
    temperature_function=None,
    temperature_arrays=None,
):
    t0 = time.perf_counter()
    result = main(
        config_params,
        material_dictionaries=mat_dicts,
        melting_curves_functions=melt_funcs,
        input_dir=input_dir,
        temperature_function=temperature_function,
        temperature_arrays=temperature_arrays,
    )
    wall = time.perf_counter() - t0
    return wall, result


def main_cli():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--n-runs', type=int, default=3)
    parser.add_argument('--mode', choices=('none', 'static', 'rotating'),
                        default='rotating',
                        help='Closure pattern across runs. See module docstring.')
    parser.add_argument('--use-jax', action='store_true')
    parser.add_argument('--use-anderson', action='store_true')
    parser.add_argument('--anderson-m-max', type=int, default=5)
    parser.add_argument('--t-surface', type=float, default=3830.0,
                        help='Surface T for constructed temperature profile (K). '
                             'Default matches CHILI Earth config.')
    parser.add_argument('--t-cmb', type=float, default=5500.0,
                        help='CMB-anchor T (K). Default is a mid-evolution value.')
    parser.add_argument('--use-temperature-arrays', action='store_true',
                        help='Route T profile through the r-indexed '
                             '(r_arr, T_arr) kwarg instead of a callable. '
                             'Exercises the JAX RHS r-indexed branch.')
    args = parser.parse_args()

    config_params = load_zalmoxis_config(args.config)
    config_params['wall_timeout'] = 3600.0
    if args.use_jax:
        config_params['use_jax'] = True
    if args.use_anderson:
        config_params['use_anderson'] = True
        config_params['anderson_m_max'] = args.anderson_m_max

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

    # Nominal mantle geometry for the temperature profile. These values don't
    # need to match the structure solution; the solver only uses
    # temperature_function to look up T along its own radial grid.
    r_cmb_nom = 3.46e6
    r_surf_nom = 7.06e6

    print(f"[bench-coupled] mode={args.mode} n_runs={args.n_runs} "
          f"use_jax={args.use_jax} use_anderson={args.use_anderson}")
    print(f"[bench-coupled] nominal r_cmb={r_cmb_nom:.2e} m, r_surf={r_surf_nom:.2e} m, "
          f"T_surf={args.t_surface} K, T_cmb={args.t_cmb} K")

    # Pre-build the static (r, T) arrays once (used by mode=static).
    _r0, _T0 = build_r_T_arrays(r_cmb_nom, r_surf_nom, args.t_surface, args.t_cmb)
    static_fn = make_temperature_function(_r0, _T0, r_cmb_nom, args.t_cmb)

    walls = []
    for i in range(args.n_runs):
        tf = None
        tarr = None
        if args.mode == 'none':
            pass  # both None
        elif args.mode == 'static':
            if args.use_temperature_arrays:
                tarr = (_r0, _T0)  # same arrays every call
            else:
                tf = static_fn  # Same closure id every call
        elif args.mode == 'rotating':
            # Perturb T_surf slightly per run: mimic PROTEUS cooling between
            # structure updates (a few K change per call).
            dT = -20.0 * i
            _r, _T = build_r_T_arrays(
                r_cmb_nom, r_surf_nom, args.t_surface + dT, args.t_cmb + dT,
            )
            if args.use_temperature_arrays:
                tarr = (_r, _T)
            else:
                tf = make_temperature_function(_r, _T, r_cmb_nom, args.t_cmb + dT)
        else:
            raise AssertionError(args.mode)

        wall, result = run_once(
            config_params, mat_dicts, melt_funcs, _input_dir,
            temperature_function=tf, temperature_arrays=tarr,
        )
        walls.append(wall)
        print(f"run {i + 1}/{args.n_runs}: wall = {wall:.2f} s "
              f"(converged={result.get('converged', False)})")

    walls_arr = np.array(walls)
    print(
        f"\n[bench-coupled] wall mean = {walls_arr.mean():.2f} s, "
        f"min = {walls_arr.min():.2f} s, "
        f"max = {walls_arr.max():.2f} s, "
        f"std = {walls_arr.std():.2f} s"
    )


if __name__ == '__main__':
    main_cli()
