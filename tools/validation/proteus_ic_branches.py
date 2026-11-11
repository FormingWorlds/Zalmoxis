"""Newton against Picard on the PROTEUS initial structure solve, with the branch of the last core node.

The PROTEUS initial structure solve (``adiabatic_from_cmb``, PALEOS, 1 Earth mass, the PROTEUS
solver settings of ``input/all_options.toml`` and ``input/dummy.toml``) carries T_surf on the core
nodes, and the last core node takes one of 2 values: about T_surf (the ``T_surf`` branch) or about a
third of the way from T_surf to T_cmb (the ``raised`` branch); which one a solve gives depends on
the outer solver and its tolerance. The mantle temperature at 39 mantle mass points and the CMB pressure of the 2 solves
must agree within 5 K and 0.1 GPa on the same branch, and within 15 K and 0.2 GPa across branches.

Usage::

    python -m tools.validation.proteus_ic_branches run <out dir> [case ...]
    python -m tools.validation.proteus_ic_branches plot <out dir> [--assets <assets dir>]
"""

from __future__ import annotations

import argparse
import json
import math
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

# (T_last - T_surf) / (T_cmb - T_surf) of each branch; measured 4e-4 to 4e-3 and 0.34 to 0.35
BRANCH_WINDOWS = {'T_surf': (-0.05, 0.05), 'raised': (0.2, 0.5)}
LIMITS = {True: (5.0, 0.1e9), False: (15.0, 0.2e9)}  # same branch: (K, Pa)


def last_core_branch(T_last, T_surf, T_cmb):
    """Name the branch of the last core node.

    Parameters
    ----------
    T_last : float
        Temperature of the last node below the CMB mass, in K.
    T_surf, T_cmb : float
        Surface and CMB anchor temperatures of the solve, in K; T_cmb must exceed T_surf.

    Returns
    -------
    tuple of (str, float)
        The branch name and its fraction (T_last - T_surf) / (T_cmb - T_surf).

    Raises
    ------
    ValueError
        A temperature is not finite, T_cmb is not above T_surf, or the fraction lies in
        neither branch window.
    """
    if not all(math.isfinite(t) for t in (T_last, T_surf, T_cmb)) or not T_cmb > T_surf:
        raise ValueError(f'no branch for T_last {T_last}, T_surf {T_surf}, T_cmb {T_cmb}')
    frac = (T_last - T_surf) / (T_cmb - T_surf)
    for name, (lo, hi) in BRANCH_WINDOWS.items():
        if lo <= frac < hi:
            return name, frac
    raise ValueError(f'last core node at fraction {frac:.4f} is on neither branch')


def record(result, cfg):
    """Mantle temperature at 39 mass points, CMB pressure and last-core-node branch of a result."""
    m = np.asarray(result['mass_enclosed'], float)
    P, T = np.asarray(result['pressure'], float), np.asarray(result['temperature'], float)
    cmb, M = float(result['cmb_mass']), float(cfg['planet_mass'])
    x = np.linspace(0.0, 1.0, 41)[1:-1]
    m_q = cmb + x * (M - cmb)
    i = int(np.searchsorted(m, cmb)) - 1  # the core/mantle split of the adiabat
    if i < 0:
        raise ValueError(f'no node below the CMB mass {cmb:.6e} kg')
    branch, frac = last_core_branch(T[i], cfg['surface_temperature'], cfg['cmb_temperature'])
    return {
        'branch': branch,
        'frac': frac,
        'T_last': float(T[i]),
        'x_q': x.tolist(),
        'm_q': m_q.tolist(),
        'T_q': np.interp(m_q, m, T).tolist(),
        'P_cmb': float(np.interp(cmb, m, P)),
        'converged': bool(result['converged']),
        'adiabat_computed': bool(result.get('adiabat_computed')),
        'blend': result.get('adiabat_blend'),
        'accepted_on': result.get('accepted_on'),
    }


def solve(case, solver):
    """The ``zalmoxis.solver.main`` result of one setup with the outer solver ``solver``."""
    import zalmoxis.solver as zs
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import load_material_dictionaries
    from zalmoxis.melting_curves import (
        derive_solidus_from_liquidus,
        get_solidus_liquidus_functions,
    )

    _, liquidus = get_solidus_liquidus_functions('Stixrude14-solidus', 'PALEOS-liquidus')
    melt = (derive_solidus_from_liquidus(liquidus, 0.8), liquidus)
    cfg = {**PROTEUS_IC['common'], **PROTEUS_IC[case], 'outer_solver': solver}
    md, inp = load_material_dictionaries(), os.path.join(get_zalmoxis_root(), 'input')
    return zs.main(cfg, md, melt, inp)


def run_case(case):
    """Picard and Newton solve of one setup; returns their records keyed by solver."""
    cfg = {**PROTEUS_IC['common'], **PROTEUS_IC[case]}
    return {s: {**record(solve(case, s), cfg), 'case': case} for s in ('picard', 'newton')}


def compare(picard, newton):
    """Largest mantle T and CMB pressure difference of 2 records and the limits of their branches."""
    if not np.array_equal(picard['m_q'], newton['m_q']):
        raise ValueError('the records hold different mass points')
    same = picard['branch'] == newton['branch']
    limit_T, limit_P = LIMITS[same]
    return {
        'branches': (picard['branch'], newton['branch']),
        'same': same,
        'dT': float(np.max(np.abs(np.subtract(newton['T_q'], picard['T_q'])))),
        'dP': abs(newton['P_cmb'] - picard['P_cmb']),
        'limit_T': limit_T,
        'limit_P': limit_P,
    }


def failures(c):
    """One line per exceeded limit of a comparison, naming the branch of each solve."""
    tag = 'Picard {}, Newton {}'.format(*c['branches'])
    out = []
    if not c['dT'] < c['limit_T']:
        out.append(f'mantle T: differs by {c["dT"]:.3f} K, limit {c["limit_T"]} K ({tag})')
    if not c['dP'] < c['limit_P']:
        out.append(
            f'P_cmb: differs by {c["dP"] / 1e9:.4f} GPa, limit {c["limit_P"] / 1e9} GPa ({tag})'
        )
    return out


def plot(outdir, assets):
    """Write the light and dark plot of |T_Newton - T_Picard| per setup with its branch limit."""
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
        print(
            case, json.dumps({k: c[k] for k in ('branches', 'dT', 'dP', 'limit_T', 'limit_P')})
        )

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5.4))
        for case, ls in zip(CASES, ('--', ':')):
            r, c = recs[case], comp[case]
            d = np.maximum(np.abs(np.subtract(r['newton']['T_q'], r['picard']['T_q'])), 1e-3)
            tag = f'{case}: Picard {c["branches"][0]}, Newton {c["branches"][1]}'
            (ln,) = ax.semilogy(r['picard']['x_q'], d, label=tag)
            ax.axhline(
                c['limit_T'],
                ls=ls,
                color=ln.get_color(),
                label=f'{case}: {c["limit_T"]:g} K limit',
            )
        ax.set_ylim(5e-4, 100.0)
        ax.minorticks_off()
        ax.set_yticks([1e-3, 1e-1, 1e1])
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.set_xlabel('Mantle mass coordinate')
        ax.set_ylabel('|T difference| [K]')
        fig.legend(loc='lower center', fontsize=plt.rcParams['axes.labelsize'])
        fig.tight_layout(rect=(0, 0.3, 1, 1))
        return fig

    mod = 'python -m tools.validation.proteus_ic_branches'
    for path in save(
        figure, 'proteus_ic_branches', assets, command=f'{mod} run OUTDIR && {mod} plot OUTDIR'
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
