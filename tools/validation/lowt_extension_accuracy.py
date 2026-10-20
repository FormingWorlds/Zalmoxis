"""Accuracy of the lowest solid entropy interval of the P-S tables with the low-T extension.

The tables of each planet mass are built as PROTEUS builds them by default: the PALEOS
two-phase MgSiO3 tables with the extension, the PALEOS liquidus, a solidus of 0.8 times the
liquidus, 1350 pressures from 1e5 Pa to min(10 TPa, (150 M + 200) GPa) for M in Earth masses,
and 280 entropy nodes. On each pressure column whose two lowest solid entropy nodes S0 and S1
are inverted cells, the deviation is the largest |T_line / T_direct - 1| between them: T_line
is linear in S between the direct temperatures of the two nodes, which is how a lookup that is
linear in S reads the table, and T_direct inverts the merged PALEOS table at that pressure. A
crossing row is a merged PALEOS row whose phase changes between S0 and S1; its band lies
between the PALEOS rows next to it, where a column interpolates with the crossing row.

Usage::

    python -m tools.validation.lowt_extension_accuracy --ext <extension.dat> --out <table.md>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from zalmoxis import eos_export as ee
from zalmoxis import get_zalmoxis_root
from zalmoxis.melting_curves import derive_solidus_from_liquidus, paleos_liquidus

MASSES = (0.5, 1.0, 2.25, 2.75, 3.0, 3.75, 4.75, 5.0, 10.0)
N_P, N_S, MZF = 1350, 280, 0.8


def p_max(mass):
    """Upper pressure of the PROTEUS P-S grid for a planet of ``mass`` Earth masses [Pa]."""
    return min(1.0e13, 150e9 * mass + 200e9)


def interval_deviation(log_t, s, S0, S1, n=401):
    """Largest |T_line / T_direct - 1| between the entropies S0 and S1 on one pressure.

    Parameters
    ----------
    log_t, s : ndarray
        log10 T and the entropy at those temperatures; s must increase with T from the last
        sample at or below S0 to the first one at or above S1.
    S0, S1 : float
        The two node entropies [J/kg/K].
    n : int
        Number of entropies sampled between S0 and S1.

    Returns
    -------
    dev : float
        The deviation, NaN when s does not span [S0, S1].
    S : float
        Entropy at which it is largest.
    T0, T1 : float
        Direct temperatures of the two nodes [K].
    """
    lo, hi = np.flatnonzero(s <= S0), np.flatnonzero(s >= S1)
    if lo.size == 0 or hi.size == 0 or hi[0] < lo[-1]:
        return np.nan, np.nan, np.nan, np.nan
    log_t, s = log_t[lo[-1] : hi[0] + 1], s[lo[-1] : hi[0] + 1]
    if not np.all(np.diff(s) > 0):
        return np.nan, np.nan, np.nan, np.nan
    S = np.linspace(S0, S1, n)
    T_dir = 10.0 ** np.interp(S, s, log_t)
    T_line = T_dir[0] + (S - S0) / (S1 - S0) * (T_dir[-1] - T_dir[0])
    rel = np.abs(T_line / T_dir - 1.0)
    k = int(np.argmax(rel))
    return float(rel[k]), float(S[k]), float(T_dir[0]), float(T_dir[-1])


def crossing_rows(s, phase, S0, S1):
    """Indices of the rows whose phase changes between S0 and S1: the cells from the last one
    with s <= S0 to the first one with s >= S1 hold more than one phase."""
    rows = []
    for i in range(s.shape[0]):
        k = np.flatnonzero(np.isfinite(s[i]))
        if k.size == 0 or s[i, k[0]] > S1 or s[i, k[-1]] < S0:
            continue
        lo = k[s[i, k] <= S0].max(initial=k[0])
        hi = k[s[i, k] >= S1].min(initial=k[-1])
        if len(set(phase[i][lo : hi + 1])) > 1:
            rows.append(i)
    return rows


def _column(sint, log_p, lo_t, hi_t, n=2001):
    """log10 T and s of the merged table along one pressure, between two temperatures."""
    log_t = np.linspace(np.log10(lo_t), np.log10(hi_t), n)
    s = sint(np.column_stack([np.full(n, log_p), log_t]))
    keep = np.isfinite(s)
    return log_t[keep], s[keep]


def mass_row(mass, files, merged, sint):
    """Build the tables of one mass and measure its lowest solid entropy interval."""
    out = ee.generate_spider_eos_tables(
        files['unified'],
        derive_solidus_from_liquidus(paleos_liquidus, MZF),
        paleos_liquidus,
        P_range=(1e5, p_max(mass)),
        n_P=N_P,
        n_S=N_S,
        solid_eos_file=files['solid'],
        liquid_eos_file=files['liquid'],
        solid_lowt_file=files['ext'],
    )
    S, T, valid, P = (
        out['S_solid'],
        out['solid']['temperature'],
        out['valid']['solid'],
        out['P_Pa'],
    )
    S0, S1 = float(S[0]), float(S[1])
    rows_p = 10.0 ** merged['unique_log_p']
    cross = crossing_rows(merged['s'], merged['phase'], S0, S1)
    bands = [(rows_p[max(i - 1, 0)], rows_p[min(i + 1, rows_p.size - 1)]) for i in cross]
    t_lo = 10.0 ** merged['unique_log_t'][0]
    cols, node_err, missed = [], 0.0, []
    for j in np.flatnonzero(valid[0] & valid[1]):
        lt, s = _column(sint, np.log10(P[j]), t_lo, 1.5 * T[1, j])
        dev, _, T0, T1 = interval_deviation(lt, s, S0, S1)
        if np.isfinite(dev):
            node_err = max(node_err, abs(T[0, j] / T0 - 1.0), abs(T[1, j] / T1 - 1.0))
        else:
            missed.append((P[j], T[0, j], s.min(initial=np.inf) - S0))
        cols.append((P[j], dev, any(a < P[j] < b for a, b in bands)))
    cross_dev = []
    for i in cross:
        lt, s = _column(sint, merged['unique_log_p'][i], t_lo, 3000.0)
        cross_dev.append((rows_p[i], interval_deviation(lt, s, S0, S1)[0]))
    return dict(
        mass=mass,
        p_max=p_max(mass),
        added=S.size - N_S,
        S0=S0,
        S1=S1,
        cols=cols,
        node_err=node_err,
        missed=missed,
        cross=cross_dev,
        bands=bands,
    )


def markdown(results):
    """The per-mass table as Markdown."""
    head = (
        '| Mass ($M_\\oplus$) | Nodes added | $S_0$, $S_1$ (J kg$^{-1}$ K$^{-1}$) '
        '| Columns with both nodes inverted | Largest deviation outside a band '
        '| Columns in a band | Crossing rows |\n'
        '|---|---|---|---|---|---|---|\n'
    )
    lines = []
    for r in results:
        out = [(p, d) for p, d, b in r['cols'] if not b and np.isfinite(d)]
        band = [(p, d) for p, d, b in r['cols'] if b]
        worst = max(out, key=lambda x: x[1], default=None)
        n_cols, nan_p = len(r['cols']), [p for p, d, _ in r['cols'] if not np.isfinite(d)]
        top = f', up to {max(p for p, _, _ in r["cols"]) / 1e9:.4g} GPa' if n_cols else ''
        top += ''.join(f' (not measured at {p / 1e9:.4g} GPa)' for p in nan_p)
        lines.append(
            f'| {r["mass"]:g} | {r["added"]} | {r["S0"]:.1f}, {r["S1"]:.1f} | {n_cols}{top} | '
            + (f'{100 * worst[1]:.2f} % at {worst[0] / 1e9:.3g} GPa' if worst else 'none')
            + ' | '
            + (', '.join(f'{100 * d:.2f} % at {p / 1e9:.4g} GPa' for p, d in band) or 'none')
            + ' | '
            + (
                ', '.join(f'{100 * d:.1f} % at {p / 1e9:.4g} GPa' for p, d in r['cross'])
                or 'none'
            )
            + ' |'
        )
    return head + '\n'.join(lines) + '\n'


def main(argv=None):
    """Command line: measure the given masses and write the Markdown table."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        '--ext', type=Path, help='the extension table (default: next to the solid table)'
    )
    ap.add_argument('--masses', type=float, nargs='+', default=MASSES)
    ap.add_argument(
        '--out', type=Path, help='Markdown file to write; the table is printed in any case'
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)
    d = Path(get_zalmoxis_root()) / 'data'
    solid = d / 'EOS_PALEOS_MgSiO3' / 'paleos_mgsio3_tables_pt_proteus_solid.dat'
    files = dict(
        solid=solid,
        liquid=solid.with_name('paleos_mgsio3_tables_pt_proteus_liquid.dat'),
        unified=d / 'EOS_PALEOS_MgSiO3_unified' / 'paleos_mgsio3_eos_table_pt.dat',
        ext=args.ext or solid.with_name('paleos_mgsio3_tables_pt_proteus_solid_lowT.dat'),
    )
    merged = ee.load_paleos_with_lowt_extension(files['solid'], files['ext'])
    sint = ee._build_interpolator(merged['unique_log_p'], merged['unique_log_t'], merged['s'])
    results = []
    for m in args.masses:
        r = mass_row(m, files, merged, sint)
        print(
            f'M = {m:g}: {r["added"]} nodes added, {len(r["cols"])} columns with both nodes '
            f'inverted, largest node temperature error {r["node_err"]:.2e}; not measured: '
            + (
                ', '.join(
                    f'{p / 1e9:.4g} GPa (node T {t:.3f} K, lowest direct s - S0 = {g:.3g})'
                    for p, t, g in r['missed']
                )
                or 'none'
            ),
            flush=True,
        )
        results.append(r)
    table = markdown(results)
    if args.out:
        args.out.write_text(table)
    print(table, end='')


if __name__ == '__main__':
    main()
