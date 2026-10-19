#!/usr/bin/env python3
"""Write a low-temperature extension of a PALEOS MgSiO3 solid table.

The extension evaluates the PALEOS phase potentials (the same ``MgSiO3EoS`` that made the
table) below the table's lowest temperature. Its temperatures continue the table's log-T
grid downward from that temperature, which is the seam node, to the first node at or below
``t_min``; its pressures are the table rows below ``p_max``. The output has the 10-column
PALEOS format, written at ``%.8e``, and is merged under the table by
``zalmoxis.eos_export.load_paleos_with_lowt_extension``.

Usage::

    python -m tools.converters.paleos_lowt_extension --paleos-src <PALEOS checkout> \\
        --table <paleos_mgsio3_tables_pt_proteus_solid.dat> --out <extension.dat>

The PALEOS commit of ``--paleos-src`` goes into the header; the extension is valid only for
a table made by the same commit, which the seam check of the merge tests on every row.
"""

from __future__ import annotations

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

import numpy as np

_METHODS = (
    'density',
    'specific_internal_energy',
    'specific_entropy',
    'isobaric_heat_capacity',
    'isochoric_heat_capacity',
    'thermal_expansion',
    'adiabatic_gradient',
)


def extension_grid(table_file, t_min=50.0, p_max=12.0e9):
    """Pressure rows and extension temperatures for a PALEOS table.

    Parameters
    ----------
    table_file : str or Path
        PALEOS table on a log-uniform temperature axis.
    t_min : float
        The temperatures stop at the first grid node at or below this value [K].
    p_max : float
        Only rows with 0 < P < p_max are extended [Pa].

    Returns
    -------
    P : ndarray
        Pressure rows [Pa].
    T : ndarray
        Temperatures from the table minimum (the seam node) downward [K].
    """
    data = np.loadtxt(table_file, usecols=(0, 1), comments='#')
    P, T_tab = np.unique(data[:, 0]), np.unique(data[:, 1])
    log_t = np.log10(T_tab)
    step = np.diff(log_t).mean()
    n = int(np.ceil((log_t[0] - np.log10(t_min)) / step - 1.0e-9))
    T = 10.0 ** (log_t[0] - step * np.arange(n + 1))
    T[0] = T_tab[0]
    return P[(P > 0.0) & (P < p_max)], T


def write_extension(eos, table_file, out_file, header, t_min=50.0, p_max=12.0e9):
    """Evaluate ``eos`` on the extension grid and write the table.

    Parameters
    ----------
    eos : object
        Has the 7 PALEOS property methods ``(P, T) -> float`` and ``phase(P, T) -> str``.
    table_file, out_file : str or Path
        PALEOS table to extend, and the output file.
    header : list of str
        Lines written as ``#`` comments before the column header.
    t_min, p_max : float
        See :func:`extension_grid`.

    Returns
    -------
    int
        Number of (P, T) rows written; a point where a property is not finite is left out,
        as PALEOS leaves out points that do not converge.
    """
    P_rows, T_nodes = extension_grid(table_file, t_min, p_max)
    lines = [f'# {h}\n' for h in header]
    lines.append('# P T rho u s cp cv alpha nabla_ad phase\n')
    n = 0
    for P in P_rows:
        for T in T_nodes:
            vals = [float(getattr(eos, m)(P, T)) for m in _METHODS]
            if not np.all(np.isfinite(vals)):
                continue
            body = ' '.join(f'{v:.8e}' for v in vals)
            lines.append(f'{P:.8e} {T:.8e} {body} {eos.phase(P, T)}\n')
            n += 1
    Path(out_file).write_text(''.join(lines))
    return n


def main(argv=None):
    """Command line: import PALEOS from ``--paleos-src`` and write the extension."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--paleos-src', required=True, type=Path)
    ap.add_argument('--table', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--t-min', type=float, default=50.0)
    ap.add_argument('--p-max', type=float, default=12.0e9)
    ap.add_argument(
        '--paleos-commit', help='commit of --paleos-src when it is not a git checkout'
    )
    args = ap.parse_args(argv)

    sys.path.insert(0, str(args.paleos_src))
    from paleos.mgsio3_eos import MgSiO3EoS

    commit = (
        args.paleos_commit
        or subprocess.run(
            ['git', '-C', str(args.paleos_src), 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    if not commit:
        ap.error('--paleos-src is not a git checkout; give its commit with --paleos-commit')
    header = [
        'PALEOS MgSiO3 solid low-temperature extension',
        f'PALEOS commit: {commit}',
        f'Extends: {args.table.name}, rows 0 < P < {args.p_max:.6e} Pa, '
        f'T from the table minimum down to the first node <= {args.t_min:g} K',
        f'Generated: {datetime.datetime.now(datetime.UTC):%Y-%m-%d %H:%M:%S} UTC',
    ]
    n = write_extension(MgSiO3EoS(), args.table, args.out, header, args.t_min, args.p_max)
    print(f'wrote {n} rows to {args.out}')


if __name__ == '__main__':
    main()
