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

A ``--table`` under a registry file name that is not its pinned release is fetched again when
it sits in its version folder under FWL_DATA; otherwise StaleTableError stops the tool before it
imports PALEOS. A missing ``--table``, an ``--out`` that is a folder or the ``--table`` file,
and a ``zalmoxis`` package other than ``src/zalmoxis`` of the converter's checkout (install
that checkout editable or set PYTHONPATH to its ``src``) are refused.
A ``--paleos-src`` with uncommitted or untracked files, one whose HEAD does not contain
``PALEOS_TESTED_SHA`` (``zalmoxis.eos.paleos_api.check_paleos_commit``; a descendant runs with a
warning), or a commit given with ``--paleos-commit`` that the tool cannot check, is refused
unless ``--allow-unverified-paleos`` is set; the header then records that state. Package
changes that git status does not show (an assume-unchanged edit, an ignored package file) are
refused too; with the flag they mark the commit as uncommitted, with the result dirty, when the
checkout contains the tested commit, and an older or unrelated checkout gets not verified with
the reason and no such mark. The header names the PALEOS commit and the result of the check
against the tested commit (tested, newer, dirty, or not verified with the reason), sha256 of the
data lines of the extended table, of ``paleos/mgsio3_eos.py`` and of the converter, the numpy,
scipy, sympy and Python versions, the Zalmoxis commit of the converter and the version of the
Zalmoxis package that checks the merge. The merge checks on every extended row that the
extension and the table agree to 1e-6 at the seam node, the table minimum; it does not test
that both come from the same PALEOS commit.

A point where PALEOS does not converge or gives a non-finite property is left out; the header
and the output give the number of left-out points and each (P, T). The merge accepts a point
left out at the lowest node of a row (the row then starts one node higher) and refuses one
inside a row; the command line then stops with an error that names the left-out points. The
tool writes a temporary file of its own next to the file ``--out`` names (a link at ``--out``
is followed and kept), checks the merge on it and only then moves it onto that file: a Python
exception of any type removes the temporary file and keeps an earlier ``--out``. The new file
gets the mode 0666 less the umask; the mode and owner of an earlier ``--out`` are not kept.

``MgSiO3EoS`` builds every PALEOS MgSiO3 phase, the liquid Wolf18 model (RTpress, compiled
with sympy) too, so its dependencies must be installed although no extension point is liquid.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import importlib.metadata
import os
import platform
import re
import secrets
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
    n : int
        Number of (P, T) rows written.
    left : list of tuple
        The (P, T) points left out, also named in the header: a point where a property is
        not finite, or where PALEOS raises RuntimeError, ArithmeticError or ValueError (no
        convergence), as PALEOS leaves out points that do not converge.
    """
    P_rows, T_nodes = extension_grid(table_file, t_min, p_max)
    rows, left = [], []
    for P in P_rows:
        for T in T_nodes:
            try:
                vals = [float(getattr(eos, m)(P, T)) for m in _METHODS]
                phase = eos.phase(P, T)
            except (RuntimeError, ArithmeticError, ValueError):
                vals = [np.nan]
            if not np.all(np.isfinite(vals)):
                left.append((P, T))
                continue
            body = ' '.join(f'{v:.8e}' for v in vals)
            rows.append(f'{P:.8e} {T:.8e} {body} {phase}\n')
    k = len(left)
    head = [
        *header,
        f'Left out: {k} point{"" if k == 1 else "s"} (no convergence or not finite)',
    ]
    head += [f'Left out at P T: {P:.8e} {T:.8e}' for P, T in left]
    lines = [f'# {h}\n' for h in head] + ['# P T rho u s cp cv alpha nabla_ad phase\n'] + rows
    Path(out_file).write_text(''.join(lines))
    return len(rows), left


def _git(path, *args):
    """Output of ``git -C path args``, or None when git fails or is not installed."""
    try:
        res = subprocess.run(['git', '-C', str(path), *args], capture_output=True, text=True)
    except FileNotFoundError:
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def _version(dist):
    """Installed version of ``dist``, or 'not installed'."""
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return 'not installed'


def _checkout(path):
    """HEAD and ``git status --porcelain`` (each None when git fails) of the checkout whose top
    is ``path``, or None when ``path`` is not the top of a git checkout."""
    top = _git(path, 'rev-parse', '--show-toplevel')
    if top is None or Path(top).resolve() != Path(path).resolve():
        return None
    return _git(path, 'rev-parse', 'HEAD'), _git(path, 'status', '--porcelain')


def _zalmoxis_commit(root):
    """HEAD of the checkout whose top is ``root``, marked when it has uncommitted or untracked
    files; 'unknown (not a git checkout)' otherwise."""
    head, status = _checkout(root) or (None, None)
    if not head or status is None:
        return 'unknown (not a git checkout)'
    return head + (' (uncommitted changes)' if status else '')


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
    ap.add_argument(
        '--allow-unverified-paleos',
        action='store_true',
        help='accept PALEOS code that cannot be shown to contain the tested commit (uncommitted '
        'changes, an older or unrelated commit, or a --paleos-commit); the header records why',
    )
    args = ap.parse_args(argv)
    out = args.out.resolve()
    if not out.parent.is_dir():
        ap.error(f'the output folder {out.parent} does not exist')
    if args.out.is_dir():
        ap.error(f'--out {args.out} is a folder')
    if not args.table.is_file():
        ap.error(f'--table {args.table} is not a file')
    if out == args.table.resolve() or (out.exists() and out.samefile(args.table)):
        ap.error('--out must not be the --table file')
    import zalmoxis

    root = Path(__file__).resolve().parents[2]
    pkg, want = Path(zalmoxis.__file__).resolve().parent, root / 'src' / 'zalmoxis'
    if not (pkg.is_dir() and want.is_dir() and pkg.samefile(want)):
        ap.error(
            f'zalmoxis was imported from {pkg}, not from {want} of the '
            'converter checkout; install that checkout editable or set PYTHONPATH to its src'
        )
    from zalmoxis.eos import paleos_api
    from zalmoxis.eos.registry_guard import verify

    verify(str(args.table))

    src = args.paleos_src.resolve()
    commit = args.paleos_commit
    if commit is not None:
        if not re.fullmatch(r'[0-9a-fA-F]{7,64}', commit):
            ap.error(f'--paleos-commit {commit!r} is not a commit hash')
        if not args.allow_unverified_paleos:
            ap.error('--paleos-commit cannot be checked; add --allow-unverified-paleos')
        commit = commit.lower() + ' (given, not verified)'
        pin = 'not verified (given commit)'
    else:
        state = _checkout(src)
        if state is None:
            ap.error(
                '--paleos-src is not the top of a git checkout; '
                'give --paleos-commit with --allow-unverified-paleos'
            )
        commit, status = state
        if not commit:
            ap.error(
                '--paleos-src has no commit; '
                'give --paleos-commit with --allow-unverified-paleos'
            )
        if status is None:
            ap.error('git status failed in --paleos-src')
        if status and not args.allow_unverified_paleos:
            ap.error(
                '--paleos-src has uncommitted or untracked files; '
                'commit them or add --allow-unverified-paleos'
            )
        # With the PALEOS-API opt-in set, changed package files come back as 'dirty' (also ones
        # git status does not show), whatever the caller set; only the flag accepts them here.
        caller = os.environ.get(paleos_api.ALLOW_DIRTY_ENV)
        os.environ[paleos_api.ALLOW_DIRTY_ENV] = '1'
        try:
            pin = paleos_api.check_paleos_commit(src)
        except paleos_api.PaleosVersionError as exc:
            if not args.allow_unverified_paleos:
                ap.error(f'{exc}; add --allow-unverified-paleos to write the extension anyway')
            pin = 'not verified: ' + ' '.join(str(exc).split())
        finally:
            if caller is None:
                del os.environ[paleos_api.ALLOW_DIRTY_ENV]
            else:
                os.environ[paleos_api.ALLOW_DIRTY_ENV] = caller
        if pin == 'dirty' and not args.allow_unverified_paleos:
            ap.error(
                f'the paleos package files in {src} differ from its commit; '
                'commit them or add --allow-unverified-paleos'
            )
        if status or pin == 'dirty':
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
        f'PALEOS check against the tested commit {paleos_api.PALEOS_TESTED_SHA}: {pin}',
        f'Table data sha256: {hashlib.sha256(data.encode()).hexdigest()}',
        f'paleos/mgsio3_eos.py sha256: {hashlib.sha256(Path(origin).read_bytes()).hexdigest()}',
        f'numpy {np.__version__}, scipy {scipy.__version__}, sympy {_version("sympy")}, '
        f'Python {platform.python_version()}',
        f'Converter sha256: {hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}',
        f'Zalmoxis commit of the converter: {_zalmoxis_commit(root)}',
        f'Zalmoxis package of the merge check: {zalmoxis.__version__}, from that checkout',
        f'Extends: {args.table.name}, rows 0 < P < {args.p_max:.6e} Pa, '
        f'T from the table minimum down to the first node <= {args.t_min:g} K',
        f'Generated: {datetime.datetime.now(datetime.UTC):%Y-%m-%d %H:%M:%S} UTC',
    ]
    tmp, made = out.with_name(f'.{out.name}.{secrets.token_hex(8)}.tmp'), False
    try:
        fd = os.open(tmp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o666)
        made = True
        os.close(fd)
        n, left = write_extension(MgSiO3EoS(), args.table, tmp, header, args.t_min, args.p_max)
        where = ', '.join(f'P = {P:.6e} Pa, T = {T:.6e} K' for P, T in left)
        if n == 0:
            ap.error(f'no extension rows: no table row has 0 < P < {args.p_max:g} Pa')
        from zalmoxis import eos_export

        try:
            eos_export.load_paleos_with_lowt_extension(args.table, tmp)
        except ValueError as exc:
            ap.error(
                'the extension does not merge under the table: '
                + str(exc).replace(str(tmp), str(args.out))
                + (f'; left-out points: {where}' if left else '')
            )
        os.replace(tmp, out)
    except BaseException:
        if made:
            tmp.unlink(missing_ok=True)
        raise
    print(f'wrote {n} rows to {args.out}')
    if left:
        print(f'left out {len(left)} point{"" if len(left) == 1 else "s"}: {where}')


if __name__ == '__main__':
    main()
