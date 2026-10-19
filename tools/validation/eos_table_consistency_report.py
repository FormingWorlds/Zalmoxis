"""Write every cell of a PALEOS-format table that fails a consistency check to a CSV file.

Usage::

    python tools/validation/eos_table_consistency_report.py TABLE.dat OUT.csv [--digits N]

``--digits`` is the number of significant digits the table stores (default 9); the check uses
the floor of ``zalmoxis.eos.consistency.EPS_FLOOR`` at least. One row per failing cell and
condition: the condition, P [Pa], T [K], phase, the tested residual and the bound (empty for a
sign condition). A summary line per condition goes to stdout.
"""

from __future__ import annotations

import argparse
import csv

import numpy as np

from zalmoxis.eos import consistency as c
from zalmoxis.eos_export import load_paleos_all_properties


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
        Condition name -> (cells tested, cells failing).
    Raises
    ------
    ValueError
        If digits is not an integer from 1 to 17.
    """
    if isinstance(digits, bool) or not isinstance(digits, int) or not 1 <= digits <= 17:
        raise ValueError(f'digits must be an integer from 1 to 17, got {digits!r}')
    t = load_paleos_all_properties(path)
    eps = 0.5 * 10.0 ** (1 - digits)
    P = np.broadcast_to((10.0 ** t['unique_log_p'])[:, None], t['rho'].shape)
    T = np.broadcast_to((10.0 ** t['unique_log_t'])[None, :], t['rho'].shape)
    rows, counts = [], {}
    for name, v in c.check(t, eps).items():
        counts[name] = (int(v['tested'].sum()), int(v['fail'].sum()))
        for i, j in np.argwhere(v['fail']):
            rows.append(
                (name, P[i, j], T[i, j], t['phase'][i, j], v['rx'][i, j], v['bound'][i, j])
            )
    for name, flags in c.signs(t, eps).items():
        counts[f'sign_{name}'] = (int(np.isfinite(t['rho']).sum()), int(flags.sum()))
        for i, j in np.argwhere(flags):
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
    for name, (tested, failing) in report(args.table, args.out, args.digits).items():
        print(f'{name}: tested {tested}, failing {failing}')


if __name__ == '__main__':
    main()
