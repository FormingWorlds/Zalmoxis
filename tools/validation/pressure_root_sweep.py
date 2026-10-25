"""Surface pressure of the accepted profile over 26 solver setups.

Each setup is one ``zalmoxis.main`` run built from ``input/default.toml``: EOS, planet mass,
number of layers, temperature mode, target surface pressure and outer solver vary. A JAX
setup takes T(r) from a numpy Picard solve at rtol 1e-5, as PROTEUS hands it the interior
profile. The plot shows |P(R) - P_t| / P_t of each accepted profile and marks the padded
ones.

Usage::

    python -m tools.validation.pressure_root_sweep run <out dir> [-j 2]
    python -m tools.validation.pressure_root_sweep plot <out dir> [--assets docs/assets/validation]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

import numpy as np

E = 5.972e24  # Earth mass [kg]
PAL = {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'}
P2 = {'core': 'Seager2007:iron', 'mantle': 'PALEOS-2phase:MgSiO3'}
WB = {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'}
AN = {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'}
NEWTON = {'outer_solver': 'newton', 'relative_tolerance': 1e-9, 'absolute_tolerance': 1e-10}
SETUPS = (
    ('Analytic 0.1', {'eos': AN, 'm': 0.1}),
    ('Analytic 1', {'eos': AN, 'm': 1}),
    ('Analytic 10', {'eos': AN, 'm': 10}),
    ('Analytic 1, 2000 layers', {'eos': AN, 'm': 1, 'n': 2000}),
    ('Analytic 1, Newton', {'eos': AN, 'm': 1, **NEWTON}),
    (
        'WB isothermal 3000 K',
        {'eos': WB, 'm': 1, 'temperature_mode': 'isothermal', 'surface_temperature': 3000},
    ),
    (
        'WB isothermal 4000 K',
        {'eos': WB, 'm': 1, 'temperature_mode': 'isothermal', 'surface_temperature': 4000},
    ),
    ('WB linear', {'eos': WB, 'm': 1, 'temperature_mode': 'linear'}),
    ('PALEOS 1 (default.toml)', {'eos': PAL, 'm': 1}),
    ('PALEOS 1, Newton', {'eos': PAL, 'm': 1, **NEWTON}),
    ('PALEOS 1, JAX Newton', {'eos': PAL, 'm': 1, 'jax': True, **NEWTON}),
    ('PALEOS 0.1', {'eos': PAL, 'm': 0.1}),
    ('PALEOS 3', {'eos': PAL, 'm': 3}),
    ('PALEOS 3, JAX Newton', {'eos': PAL, 'm': 3, 'jax': True, **NEWTON}),
    ('PALEOS 10, Newton', {'eos': PAL, 'm': 10, **NEWTON}),
    ('PALEOS linear 1', {'eos': PAL, 'm': 1, 'temperature_mode': 'linear'}),
    ('PALEOS isothermal 1', {'eos': PAL, 'm': 1, 'temperature_mode': 'isothermal'}),
    ('PALEOS 1, 2000 layers', {'eos': PAL, 'm': 1, 'n': 2000}),
    ('PALEOS 1, P_t 1e7 Pa', {'eos': PAL, 'm': 1, 'target': 1e7}),
    ('PALEOS 1, P_t 1e3 Pa', {'eos': PAL, 'm': 1, 'target': 1e3}),
    ('PALEOS-2phase linear 1', {'eos': P2, 'm': 1, 'temperature_mode': 'linear'}),
    ('PALEOS-2phase 1', {'eos': P2, 'm': 1}),
    ('PALEOS-2phase 3', {'eos': P2, 'm': 3}),
    (
        'PALEOS + 3 % H 1',
        {'eos': dict(PAL, mantle='PALEOS:MgSiO3:0.97+Chabrier:H:0.03'), 'm': 1},
    ),
    (
        'PALEOS 3-layer ice 1',
        {'eos': dict(PAL, ice_layer='PALEOS:H2O'), 'm': 1, 'mantle_mass_fraction': 0.4},
    ),
    ('PALEOS 5, Newton', {'eos': PAL, 'm': 5, **NEWTON}),
)


def run_setup(index: int) -> dict:
    """Run setup ``index`` and return its outcome: accept label, P(R) [Pa], padded nodes, time [s]."""
    import zalmoxis
    import zalmoxis.solver as zs
    from zalmoxis.config import (
        load_material_dictionaries,
        load_solidus_liquidus_functions,
        load_zalmoxis_config,
    )

    root = zalmoxis.get_zalmoxis_root()
    name, spec = SETUPS[index]
    spec = dict(spec)
    eos, jax = spec.pop('eos'), spec.pop('jax', False)
    target = spec.pop('target', 101325.0)
    out = {'index': index, 'name': name, 'target': target, 'zalmoxis': zs.__file__}
    t0 = time.time()
    try:
        cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
        cfg['layer_eos_config'] = dict(eos)
        cfg.update(
            planet_mass=spec.pop('m') * E,
            num_layers=spec.pop('n', 150),
            target_surface_pressure=target,
            wall_timeout=1800.0,
            data_output_enabled=False,
            plotting_enabled=False,
            **spec,
        )
        melt = load_solidus_liquidus_functions(
            dict(eos), cfg['rock_solidus'], cfg['rock_liquidus'], cfg['mushy_zone_factor']
        )
        mats, indir, tarr = load_material_dictionaries(), os.path.join(root, 'input'), None
        if jax:
            pre = dict(
                cfg, outer_solver='picard', relative_tolerance=1e-5, absolute_tolerance=1e-6
            )
            r0 = zs.main(pre, mats, melt, indir)
            tarr = (np.asarray(r0['radii'], float), np.asarray(r0['temperature'], float))
            cfg['use_jax'] = True
        res = zs.main(cfg, mats, melt, indir, temperature_arrays=tarr)
        P = np.asarray(res['pressure'], float)
        out.update(
            converged=bool(res['converged']),
            accepted_on=res.get('accepted_on'),
            P_R=float(P[-1]),
            padded=int(np.sum(P == 0.0)),
            error=None,
        )
    except Exception as err:  # every outcome is recorded
        out.update(
            converged=False, error=f'{type(err).__name__}: {err}', tb=traceback.format_exc()
        )
    out['time_s'] = round(time.time() - t0, 1)
    return out


def run_all(outdir: str, workers: int) -> None:
    """Run every setup in its own process, ``workers`` at a time, writing setup_<i>.json."""
    os.makedirs(outdir, exist_ok=True)

    def one(i):
        cmd = [
            sys.executable,
            '-m',
            'tools.validation.pressure_root_sweep',
            'case',
            str(i),
            outdir,
        ]
        return subprocess.run(cmd, check=False).returncode

    with ThreadPoolExecutor(workers) as pool:
        codes = list(pool.map(one, range(len(SETUPS))))
    print('return codes', codes)


def plot(outdir: str, assets: str) -> None:
    """Write the light and dark plot of |P(R) - P_t| / P_t per setup; a failed setup stops it."""
    from proteus_mpl.validation import save

    rows, failed = [], []
    for i, (name, _) in enumerate(SETUPS):
        path = os.path.join(outdir, f'setup_{i}.json')
        if not os.path.exists(path):
            failed.append(f'{i} {name}: no record (the setup process died)')
            continue
        with open(path) as f:
            rows.append(json.load(f))
        if rows[-1].get('error') or not rows[-1].get('converged'):
            failed.append(f'{i} {name}: {rows[-1].get("error") or "not converged"}')
    if failed:
        raise SystemExit('setups without an accepted profile:\n' + '\n'.join(failed))
    miss = np.array([abs(r['P_R'] - r['target']) / r['target'] for r in rows])
    padded = np.array([r['padded'] > 0 for r in rows])

    def figure():
        import matplotlib.pyplot as plt

        x = np.arange(len(rows))
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.semilogy(x[~padded], miss[~padded], 'o', ms=6, label='Reached')
        ax.semilogy(x[padded], miss[padded], 's', ms=6, label='Padded')
        ax.axhline(1.0, ls='--', label='Cap')
        ax.set_xticks(x[::5], [str(i) for i in x[::5]])
        ax.set_xlabel('Setup')
        ax.set_yticks([1e-4, 1e-2, 1.0])
        ax.set_ylabel(r'$|P(R) - P_t| / P_t$')
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    mod = 'python -m tools.validation.pressure_root_sweep'
    cmd = f'{mod} run OUTDIR && {mod} plot OUTDIR'
    for path in save(figure, 'pressure_root_sweep', assets, command=cmd):
        print(path, os.path.getsize(path))


def main(argv=None) -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_run = sub.add_parser('run')
    p_run.add_argument('outdir')
    p_run.add_argument('-j', type=int, default=2)
    p_case = sub.add_parser('case')
    p_case.add_argument('index', type=int)
    p_case.add_argument('outdir')
    p_plot = sub.add_parser('plot')
    p_plot.add_argument('outdir')
    p_plot.add_argument('--assets', default='docs/assets/validation')
    args = parser.parse_args(argv)
    if args.cmd == 'run':
        run_all(args.outdir, args.j)
    elif args.cmd == 'case':
        rec = run_setup(args.index)
        with open(os.path.join(args.outdir, f'setup_{args.index}.json'), 'w') as f:
            json.dump(rec, f, indent=1)
        print(
            json.dumps(
                {k: rec.get(k) for k in ('index', 'name', 'accepted_on', 'padded', 'time_s')}
            )
        )
    else:
        plot(args.outdir, args.assets)


if __name__ == '__main__':
    main()
