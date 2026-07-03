#!/usr/bin/env python
"""Wet-mantle JAX-vs-numpy bench: wall time and structure agreement.

Runs `zalmoxis.solver.main()` on the reference config with a
strong-partition VolatileProfile (PALEOS:H2O in the melt, w_solid = 0),
once on the numpy path and once on the JAX path, and reports wall time,
planet radius, and the relative radius difference. The JAX wet path is
the `has_volatile=True` variant of `coupled_odes_jax`; before it existed
every wet solve forced numpy.

Usage
-----
    python -m tools.benchmarks.bench_wet_mantle \
        --config=tests/data/bench_performance.toml \
        --w-liquid=0.083 [--num-levels=150]
"""

from __future__ import annotations

import argparse
import sys
import time

from zalmoxis.config import (
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
)
from zalmoxis.mixing import VolatileProfile
from zalmoxis.solver import main


def run(config_path, w_liquid, num_levels, use_jax):
    config_params = load_zalmoxis_config(config_path)
    config_params['use_jax'] = use_jax
    if num_levels:
        config_params['num_levels'] = int(num_levels)
    # Long wall so both paths converge rather than accept timeout solutions.
    config_params.setdefault('wall_timeout', 3600.0)

    mantle_eos = config_params['layer_eos_config']['mantle']
    if '+' in mantle_eos:
        raise SystemExit(f'expected a single-component mantle in the config, got {mantle_eos}')
    profile = VolatileProfile(
        w_liquid={'PALEOS:H2O': float(w_liquid)},
        w_solid={'PALEOS:H2O': 0.0},
        primary_component=mantle_eos,
    )
    # Extend the mantle mixture with the volatile placeholder, as the
    # PROTEUS wrapper does; the profile overrides fractions per shell.
    config_params['layer_eos_config']['mantle'] = f'{mantle_eos}:0.9900+PALEOS:H2O:0.0100'

    mat_dicts = load_material_dictionaries()
    melt_funcs = load_solidus_liquidus_functions(
        config_params['layer_eos_config'],
        config_params.get('rock_solidus', 'Stixrude14-solidus'),
        config_params.get('rock_liquidus', 'Stixrude14-liquidus'),
    )
    if melt_funcs is None:
        # All-unified configs need no external curves for density, but
        # the wet blend's phi does; build the defaults, as PROTEUS does.
        from zalmoxis.melting_curves import get_solidus_liquidus_functions

        melt_funcs = get_solidus_liquidus_functions(
            config_params.get('rock_solidus', 'Stixrude14-solidus'),
            config_params.get('rock_liquidus', 'Stixrude14-liquidus'),
        )

    t0 = time.perf_counter()
    results = main(
        config_params,
        mat_dicts,
        melt_funcs,
        input_dir='.',
        volatile_profile=profile,
    )
    wall = time.perf_counter() - t0
    return results, wall


def cli(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', default='tests/data/bench_performance.toml')
    ap.add_argument('--w-liquid', type=float, default=0.083)
    ap.add_argument('--num-levels', type=int, default=0)
    args = ap.parse_args(argv)

    rows = []
    for label, use_jax in (('numpy', False), ('jax', True)):
        results, wall = run(args.config, args.w_liquid, args.num_levels, use_jax)
        R = float(results['radii'][-1])
        rows.append((label, wall, R, bool(results['converged'])))
        print(
            f'[{label:5s}] wall={wall:8.1f} s  R={R:.6e} m  converged={results["converged"]}',
            flush=True,
        )

    (_, wall_np, R_np, ok_np), (_, wall_jx, R_jx, ok_jx) = rows
    dR = abs(R_jx / R_np - 1.0)
    print(f'\nspeedup (numpy/jax wall) = {wall_np / max(wall_jx, 1e-9):.2f}x')
    print(f'|R_jax/R_numpy - 1|      = {dR:.3e}')
    if not (ok_np and ok_jx):
        print('WARNING: at least one path did not fully converge; times not comparable.')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(cli())
