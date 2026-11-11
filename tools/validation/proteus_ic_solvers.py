"""Newton against Picard on the PROTEUS initial structure solve.

The PROTEUS initial structure solve (``adiabatic_from_cmb``, PALEOS, 1 Earth mass, the PROTEUS
solver settings of ``input/all_options.toml`` and ``input/dummy.toml``) gives the same mantle
temperature at 39 mantle mass points within 5 K and the same CMB pressure within 0.1 GPa with
the Newton and the Picard outer solver.

Usage::

    python -m tools.validation.proteus_ic_solvers run <out dir> [case ...]
    python -m tools.validation.proteus_ic_solvers plot <out dir> [--assets <assets dir>]
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

CASES = ('all_options', 'dummy')

# config_params of the PROTEUS initial structure solve (Time 0, liquidus_super), from
# input/all_options.toml and input/dummy.toml of PROTEUS
PROTEUS_IC = {
    'common': {
        'absolute_tolerance': 1e-10,
        'center_temperature': 6000.0,
        'layer_eos_config': {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
        'mantle_mass_fraction': 0.0,
        'max_iterations_inner': 100,
        'max_iterations_outer': 100,
        'mushy_zone_factor': 0.8,
        'mushy_zone_factors': {'PALEOS:MgSiO3': 0.8, 'PALEOS:iron': 0.8},
        'newton_max_iter': 30,
        'newton_tol': 1e-4,
        'num_layers': 150,
        'planet_mass': 5.972e24,
        'relative_tolerance': 1e-9,
        'temp_profile_file': None,
        'temperature_mode': 'adiabatic_from_cmb',
        'tolerance_inner': 1e-4,
        'tolerance_outer': 3e-3,
        'use_anderson': True,
        'use_jax': True,
        'wall_timeout': 1e6,
    },
    'all_options': {
        'cmb_temperature': 6121.257519593061,
        'core_mass_fraction': 0.325,
        'surface_temperature': 3936.1734971812284,
        'target_surface_pressure': 5.0e6,
    },
    'dummy': {
        'cmb_temperature': 6000.0,
        'core_mass_fraction': 0.55,
        'surface_temperature': 4000.0,
        'target_surface_pressure': 101325.0,
    },
}

LIMIT_T, LIMIT_P = 5.0, 0.1e9  # K, Pa


def record(result, cfg):
    """Mantle temperature at 39 mass points, CMB pressure and accept state of a result."""
    m = np.asarray(result['mass_enclosed'], float)
    P, T = np.asarray(result['pressure'], float), np.asarray(result['temperature'], float)
    cmb, M = float(result['cmb_mass']), float(cfg['planet_mass'])
    x = np.linspace(0.0, 1.0, 41)[1:-1]
    m_q = cmb + x * (M - cmb)
    return {
        'x_q': x.tolist(),
        'm_q': m_q.tolist(),
        'T_q': np.interp(m_q, m, T).tolist(),
        'P_cmb': float(np.interp(cmb, m, P)),
        'converged': bool(result['converged']),
        'adiabat_computed': bool(result.get('adiabat_computed')),
        'blend': result.get('adiabat_blend'),
        'table_blend': result.get('adiabat_table_blend'),
        'self_consistency': result.get('adiabat_self_consistency'),
        'accepted_on': result.get('accepted_on'),
    }


def solve(case, solver, **overrides):
    """The ``zalmoxis.solver.main`` result of one setup with the outer solver ``solver`` and
    the config values of ``overrides``."""
    import zalmoxis.solver as zs
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import load_material_dictionaries
    from zalmoxis.melting_curves import (
        derive_solidus_from_liquidus,
        get_solidus_liquidus_functions,
    )

    _, liquidus = get_solidus_liquidus_functions('Stixrude14-solidus', 'PALEOS-liquidus')
    melt = (derive_solidus_from_liquidus(liquidus, 0.8), liquidus)
    cfg = {**PROTEUS_IC['common'], **PROTEUS_IC[case], 'outer_solver': solver, **overrides}
    md, inp = load_material_dictionaries(), os.path.join(get_zalmoxis_root(), 'input')
    return zs.main(cfg, md, melt, inp)


def run_case(case):
    """Picard and Newton solve of one setup; returns their records keyed by solver."""
    cfg = {**PROTEUS_IC['common'], **PROTEUS_IC[case]}
    return {s: {**record(solve(case, s), cfg), 'case': case} for s in ('picard', 'newton')}


def compare(picard, newton):
    """Largest mantle T difference (K) and CMB pressure difference (Pa) of 2 records."""
    if not np.array_equal(picard['m_q'], newton['m_q']):
        raise ValueError('the records hold different mass points')
    return {
        'dT': float(np.max(np.abs(np.subtract(newton['T_q'], picard['T_q'])))),
        'dP': abs(newton['P_cmb'] - picard['P_cmb']),
    }


def plot(outdir, assets):
    """Write the light and dark plot of |T_Newton - T_Picard| per setup with the 5 K limit."""
    from proteus_mpl.validation import save

    recs, missing = {}, []
    for case in CASES:
        path = os.path.join(outdir, f'case_{case}.json')
        if not os.path.exists(path):
            missing.append(f'{case}: no record (the solve died)')
            continue
        with open(path) as f:
            recs[case] = json.load(f)
    if missing:
        raise SystemExit('cases without a record:\n' + '\n'.join(missing))
    comp = {case: compare(r['picard'], r['newton']) for case, r in recs.items()}
    for case, c in comp.items():
        print(case, json.dumps(c))

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5.4))
        for case in CASES:
            r = recs[case]
            d = np.maximum(np.abs(np.subtract(r['newton']['T_q'], r['picard']['T_q'])), 1e-3)
            ax.semilogy(r['picard']['x_q'], d, label=case)
        kw = {'ls': '--', 'color': plt.rcParams['text.color'], 'alpha': 0.5}
        ax.axhline(LIMIT_T, label=f'{LIMIT_T:g} K limit', **kw)
        ax.set_ylim(5e-4, 100.0)
        ax.minorticks_off()
        ax.set_yticks([1e-3, 1e-1, 1e1])
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xlabel('Mantle mass coordinate')
        ax.set_ylabel('|T difference| [K]')
        fig.legend(loc='lower center', fontsize=plt.rcParams['axes.labelsize'])
        fig.tight_layout(rect=(0, 0.22, 1, 1))
        return fig

    mod = 'python -m tools.validation.proteus_ic_solvers'
    for path in save(
        figure, 'proteus_ic_solvers', assets, command=f'{mod} run OUTDIR && {mod} plot OUTDIR'
    ):
        print(path, os.path.getsize(path))


def main(argv=None):
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest='cmd', required=True)
    p_run = sub.add_parser('run')
    p_run.add_argument('outdir')
    p_run.add_argument('cases', nargs='*', default=list(CASES))
    p_plot = sub.add_parser('plot')
    p_plot.add_argument('outdir')
    p_plot.add_argument('--assets', default='docs/assets/validation')
    args = parser.parse_args(argv)
    if args.cmd == 'plot':
        return plot(args.outdir, args.assets)
    os.makedirs(args.outdir, exist_ok=True)
    for case in args.cases:
        recs = run_case(case)
        with open(os.path.join(args.outdir, f'case_{case}.json'), 'w') as f:
            json.dump(recs, f)
        print(case, json.dumps(compare(recs['picard'], recs['newton'])))


if __name__ == '__main__':
    main()
