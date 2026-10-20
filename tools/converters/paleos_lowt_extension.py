#!/usr/bin/env python3
"""Write a low-temperature extension of a PALEOS MgSiO3 solid table.

The extension evaluates the PALEOS phase potentials (the same ``MgSiO3EoS`` that made the
table) below the table's lowest temperature. Its temperatures continue the table's log-T
grid downward from that temperature, which is the seam node, to the first node at or below
``t_min``; its pressures are the table rows below ``p_max``. The output has the 10-column
PALEOS format, written at ``%.8e``, and is merged under the table by
``zalmoxis.eos_export.load_paleos_with_lowt_extension``; the command line checks that merge
before it reports success, and removes an output that does not merge.

Usage::

    python -m tools.converters.paleos_lowt_extension --paleos-src <PALEOS checkout> \\
        --table <paleos_mgsio3_tables_pt_proteus_solid.dat> --out <extension.dat>

The header names the PALEOS commit of ``--paleos-src`` (a commit given with
``--paleos-commit`` is written as not verified), sha256 of the data lines of the extended
table and of ``paleos/mgsio3_eos.py``, and the numpy and scipy versions. The merge checks on
every extended row that the extension and the table agree to 1e-6 at the seam node, the
table minimum; it does not test that both come from the same PALEOS commit.

``MgSiO3EoS`` builds every PALEOS MgSiO3 phase, the liquid Wolf18 model (RTpress, compiled
with sympy) too, so its dependencies must be installed although no extension point is liquid.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import re
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
    data = np.loadtxt(table_file, usecols=(0, 1), comments='#', ndmin=2)
    P, T_tab = np.unique(data[:, 0]), np.unique(data[:, 1])
    if T_tab.size < 2:
        raise ValueError(f'{table_file}: the table needs at least two temperatures')
    if not 0.0 < t_min < T_tab[0]:
        raise ValueError(
            f't_min = {t_min} K must lie between 0 K and the table minimum {T_tab[0]} K'
        )
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
        Number of (P, T) rows written; a point where a property is not finite, or where
        PALEOS raises RuntimeError, ArithmeticError or ValueError (no convergence), is left
        out, as PALEOS leaves out points that do not converge.
    """
    P_rows, T_nodes = extension_grid(table_file, t_min, p_max)
    lines = [f'# {h}\n' for h in header]
    lines.append('# P T rho u s cp cv alpha nabla_ad phase\n')
    n = 0
    for P in P_rows:
        for T in T_nodes:
            try:
                vals = [float(getattr(eos, m)(P, T)) for m in _METHODS]
                phase = eos.phase(P, T)
            except (RuntimeError, ArithmeticError, ValueError):
                continue
            if not np.all(np.isfinite(vals)):
                continue
            body = ' '.join(f'{v:.8e}' for v in vals)
            lines.append(f'{P:.8e} {T:.8e} {body} {phase}\n')
            n += 1
    Path(out_file).write_text(''.join(lines))
    return n


def _git(path, *args):
    """Output of ``git -C path args``, or None when git fails or is not installed."""
    try:
        res = subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return res.stdout.strip() if res.returncode == 0 else None


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
    if not args.out.parent.is_dir():
        ap.error(f'the output folder {args.out.parent} does not exist')

    src = args.paleos_src.resolve()
    commit = args.paleos_commit
    if commit is not None:
        if not re.fullmatch(r'[0-9a-fA-F]{7,64}', commit):
            ap.error(f'--paleos-commit {commit!r} is not a commit hash')
        commit = commit.lower() + ' (given, not verified)'
    else:
        top = _git(src, 'rev-parse', '--show-toplevel')
        if top is None or Path(top).resolve() != src:
            ap.error('--paleos-src is not the top of a git checkout; give --paleos-commit')
        commit, status = _git(src, 'rev-parse', 'HEAD'), _git(src, 'status', '--porcelain')
        if not commit:
            ap.error('--paleos-src has no commit; give --paleos-commit')
        if status is None:
            ap.error('git status failed in --paleos-src')
        if status:
            commit += ' (uncommitted changes)'
    sys.path.insert(0, str(src))
    from paleos.mgsio3_eos import MgSiO3EoS

    origin = getattr(sys.modules['paleos.mgsio3_eos'], '__file__', None)
    if origin is None or not Path(origin).resolve().is_relative_to(src):
        ap.error(f'paleos was imported from {origin}, not from --paleos-src')
    import scipy

    data = ''.join(x for x in args.table.read_text().splitlines(True) if not x.startswith('#'))
    header = [
        'PALEOS MgSiO3 solid low-temperature extension',
        f'PALEOS commit: {commit}',
        f'Table data sha256: {hashlib.sha256(data.encode()).hexdigest()}',
        f'paleos/mgsio3_eos.py sha256: {hashlib.sha256(Path(origin).read_bytes()).hexdigest()}',
        f'numpy {np.__version__}, scipy {scipy.__version__}',
        f'Extends: {args.table.name}, rows 0 < P < {args.p_max:.6e} Pa, '
        f'T from the table minimum down to the first node <= {args.t_min:g} K',
        f'Generated: {datetime.datetime.now(datetime.UTC):%Y-%m-%d %H:%M:%S} UTC',
    ]
    n = write_extension(MgSiO3EoS(), args.table, args.out, header, args.t_min, args.p_max)
    if n == 0:
        args.out.unlink()
        ap.error(f'no extension rows: no table row has 0 < P < {args.p_max:g} Pa')
    from zalmoxis.eos_export import load_paleos_with_lowt_extension

    try:
        load_paleos_with_lowt_extension(args.table, args.out)
    except ValueError as exc:
        args.out.unlink()
        ap.error(f'the extension does not merge under the table: {exc}')
    print(f'wrote {n} rows to {args.out}')


if __name__ == '__main__':
    main()
