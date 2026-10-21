"""Write every cell of a PALEOS-format table that fails a consistency check to a CSV file.

Usage::

    python tools/validation/eos_table_consistency_report.py TABLE.dat OUT.csv [--digits N]

``--digits`` is the number of significant digits the table stores (default 9); the check uses
the floor of ``zalmoxis.eos.consistency.EPS_FLOOR`` at least. One row per failing cell and
condition: the condition, P [Pa], T [K], phase, the tested residual and the bound (empty for a
sign condition). A summary line per condition goes to stdout; a skipped identity says why, and
an identity with an input column that is 0 or inf everywhere names that column.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from zalmoxis.eos import consistency as c
from zalmoxis.eos_export import load_paleos_all_properties


def notes(table, res):
    """One note per identity of ``res``.

    Parameters
    ----------
    table : dict
        The table, as from ``load_paleos_all_properties``.
    res : dict
        The result of ``check`` on ``table``.

    Returns
    -------
    dict
        Identity name -> why it is skipped (the input columns that hold no value, or no cell
        with all inputs on a full stencil of one phase), the input columns that are 0 or inf
        in every cell that holds a value, or ''.
    """
    out = {}
    for name, v in res.items():
        cols = c.INPUTS[name]
        if v['skipped']:
            empty = [col for col in cols if np.isnan(np.asarray(table[col], float)).all()]
            if empty:
                out[name] = (
                    f'skipped: {", ".join(empty)} {"holds" if len(empty) == 1 else "hold"} no value'
                )
            elif name == 'nabla_ad':
                out[name] = f'skipped: no cell with {", ".join(cols)} present'
            else:
                out[name] = (
                    f'skipped: no cell with {", ".join(cols)} present on 4 cells each side in one phase'
                )
            continue
        bad = []
        for col in cols:
            x = np.asarray(table[col], float)
            x = x[~np.isnan(x)]
            if x.size and ((x == 0) | np.isinf(x)).all():
                bad.append(f'{col} is 0 or inf in every cell that holds a value')
        out[name] = '; '.join(bad)
    return out


def report(path, out, digits=9):
    """Check the table at ``path`` and write the failing cells to ``out``.

    Parameters
    ----------
    path : str
        PALEOS-format table.
    out : str
        CSV file to write.
    digits : int
        Significant digits stored in the table.

    Returns
    -------
    dict
        Condition name -> (cells tested, cells failing, note); the note is from ``notes`` for
        an identity and '' for a sign condition.

    Raises
    ------
    ValueError
        If digits is not an integer from 1 to 17, or every identity is skipped (the message
        names the columns without a value).
    """
    if (
        isinstance(digits, bool)
        or not isinstance(digits, (int, np.integer))
        or not 1 <= digits <= 17
    ):
        raise ValueError(f'digits must be an integer from 1 to 17, got {digits!r}')
    digits = int(digits)
    t = load_paleos_all_properties(path)
    eps = 0.5 * 10.0 ** (1 - digits)
    P = np.broadcast_to((10.0 ** t['unique_log_p'])[:, None], t['rho'].shape)
    T = np.broadcast_to((10.0 ** t['unique_log_t'])[None, :], t['rho'].shape)
    rows, counts = [], {}
    try:
        res = c.check(t, eps)
    except ValueError as err:
        empty = [col for col in c.COLUMNS if np.isnan(np.asarray(t[col], float)).all()]
        raise ValueError(
            f'{err}; columns without a value: {", ".join(empty) or "none"}'
        ) from err
    note = notes(t, res)
    for name, v in res.items():
        counts[name] = (int(v['tested'].sum()), int(v['fail'].sum()), note[name])
        for i, j in np.argwhere(v['fail']):
            rows.append(
                (name, P[i, j], T[i, j], t['phase'][i, j], v['rx'][i, j], v['bound'][i, j])
            )
    for name, v in c.signs(t, eps).items():
        counts[f'sign_{name}'] = (int(v['tested'].sum()), int(v['fail'].sum()), '')
        for i, j in np.argwhere(v['fail']):
            rows.append((f'sign_{name}', P[i, j], T[i, j], t['phase'][i, j], '', ''))
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(('condition', 'P_Pa', 'T_K', 'phase', 'residual', 'bound'))
        w.writerows(rows)
    return counts


def main():
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('table')
    ap.add_argument('out')
    ap.add_argument('--digits', type=int, default=9)
    args = ap.parse_args()
    for name, (tested, failing, note) in report(args.table, args.out, args.digits).items():
        if note.startswith('skipped'):
            print(f'{name}: {note}')
        else:
            print(
                f'{name}: tested {tested}, failing {failing}' + (f' ({note})' if note else '')
            )


if __name__ == '__main__':
    main()
