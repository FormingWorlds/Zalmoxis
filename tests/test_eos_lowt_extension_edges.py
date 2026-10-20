"""Edge cases and boundary tests for the low-temperature PALEOS solid extension."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.test_eos_lowt_extension import (
    _DLOGT,
    _P,
    _T,
    _TG,
    _ext,
    _gen_tables,
    _generate_from,
    _paleos,
    _props,
    _write,
)
from tools.converters import paleos_lowt_extension as tool
from zalmoxis import eos_export
from zalmoxis.melting_curves import derive_solidus_from_liquidus, paleos_liquidus

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ('prop_name', 'prop_idx'),
    [
        ('s', 2),
        ('rho', 0),
        ('alpha', 5),
        ('nabla_ad', 6),
    ],
)
def test_seam_offsets_just_inside_and_outside_the_tolerance(tmp_path, prop_name, prop_idx):
    """A relative seam offset of 0.9e-6 is accepted and one of 1.1e-6 refused, for s, rho, alpha and
    nabla_ad: the 1e-6 tolerance is pinned from both sides to 10 %."""
    pal = _paleos(tmp_path, [6, 6, 6, 6], name=f'pal_{prop_name}.dat')
    cells = [(P, T) for P in _P for T in _T[:7]]

    def ext_with(factor):
        def edit(P, T, vals):
            if np.isclose(T, _T[6]):
                vals[prop_idx] *= factor

        return _write(tmp_path / f'ext_{prop_name}_{factor}.dat', cells, edit=edit)

    try:
        merged = eos_export.load_paleos_with_lowt_extension(pal, ext_with(1.0 + 0.9e-6))
    except ValueError as exc:
        pytest.fail(f'refused a seam offset inside the tolerance: {exc}')
    assert np.isfinite(merged['s']).all()
    with pytest.raises(ValueError, match='seam mismatch in'):
        eos_export.load_paleos_with_lowt_extension(pal, ext_with(1.0 + 1.1e-6))


def test_continuation_offsets_just_inside_and_outside_the_tolerance(tmp_path):
    """A first extension node off the PALEOS log-T grid by 0.9e-4 of a step is accepted and one off
    by 1.1e-4 refused: the grid tolerance of 1e-4 of a step is pinned from both sides to 10 %."""
    pal = _paleos(tmp_path, [6, 6, 6, 6], name='pal_cont.dat')

    def ext_shifted(frac):
        cold = 10.0 ** (np.log10(_T[0]) + frac * _DLOGT)
        cells = [(P, T) for P in _P for T in (cold, *_T[1:7])]
        return _write(tmp_path / f'ext_cont_{frac}.dat', cells)

    try:
        merged = eos_export.load_paleos_with_lowt_extension(pal, ext_shifted(0.9e-4))
    except ValueError as exc:
        pytest.fail(f'refused a grid offset inside the tolerance: {exc}')
    assert np.isfinite(merged['s']).all()
    with pytest.raises(ValueError, match='continue the PALEOS log-T grid'):
        eos_export.load_paleos_with_lowt_extension(pal, ext_shifted(1.1e-4))


def test_generate_spider_tables_with_p_out_beyond_extension_rows(tmp_path):
    """Columns above the last extension row have invalid added cells filled from lowest inverted cell."""
    solid, liquid, _ = _gen_tables(tmp_path)
    # Extension only covers rows 0 and 1, stopping below _P[2].
    ext = _write(tmp_path / 'ext_partial.dat', [(P, T) for P in _P[:2] for T in _TG[:7]])

    base = _generate_from(solid, liquid, None, p_range=(_P[0], _P[-1]))
    ext_out = _generate_from(solid, liquid, ext, p_range=(_P[0], _P[-1]))

    n_add = ext_out['S_solid'].size - base['S_solid'].size
    assert n_add > 0
    p_out = ext_out['P_Pa']
    t_solid = ext_out['solid']['temperature']
    valid_solid = ext_out['valid']['solid']

    # For columns above the last extension row pressure (_P[1]), added rows must be invalid.
    high_p_cols = np.flatnonzero(p_out > _P[1])
    assert high_p_cols.size > 0
    for ip in high_p_cols:
        assert not np.any(valid_solid[:n_add, ip])
        # Added rows are finite and carry the flat fill from the lowest inverted cell of the column.
        assert np.isfinite(t_solid[:n_add, ip]).all()
        first_valid_idx = np.flatnonzero(valid_solid[:, ip])[0]
        np.testing.assert_allclose(t_solid[:n_add, ip], t_solid[first_valid_idx, ip], rtol=1e-8)


def test_generate_spider_tables_with_output_dir_unequal_solid_melt_axes(tmp_path):
    """Writers give the file headers and the written masks of each phase when the solid and melt
    S axes differ."""
    tables = _gen_tables(tmp_path)
    outdir = tmp_path / 'spider_tables_out'
    res = eos_export.generate_spider_eos_tables(
        eos_file=tables[0],
        solidus_func=lambda P: 1500.0,
        liquidus_func=lambda P: 1800.0,
        P_range=(1.0e6, 1.0e10),
        n_P=5,
        n_S=30,
        output_dir=outdir,
        solid_eos_file=tables[0],
        liquid_eos_file=tables[1],
        solid_lowt_file=tables[2],
    )

    n_p = res['P_Pa'].size
    n_s_solid = res['S_solid'].size
    n_s_melt = res['S_melt'].size
    assert n_s_solid > n_s_melt

    t_sol_path = outdir / 'temperature_solid.dat'
    header_sol = t_sol_path.read_text().splitlines()[0].strip()
    assert header_sol == f'# 5 {n_p} {n_s_solid}'

    t_melt_path = outdir / 'temperature_melt.dat'
    header_melt = t_melt_path.read_text().splitlines()[0].strip()
    assert header_melt == f'# 5 {n_p} {n_s_melt}'

    for phase, n_s in (('solid', n_s_solid), ('melt', n_s_melt)):
        mask = np.loadtxt(outdir / f'valid_mask_{phase}.dat')
        assert mask.shape == (n_s, n_p)
        np.testing.assert_array_equal(mask, res['valid'][phase].astype(float))


@pytest.mark.slow
def test_real_paleos_table_lowt_extension_numbers():
    """The real PALEOS tables with the extension give the documented S_min, n_add, coldest T and
    conductive gradient bias of the two lowest entropy intervals."""
    fwl_dir = os.environ.get('FWL_DATA')
    if not fwl_dir:
        pytest.skip('FWL_DATA environment variable is not set')

    base_dir = Path(fwl_dir) / 'zalmoxis_eos'
    mgsio3_dir = base_dir / 'EOS_PALEOS_MgSiO3'
    solid = mgsio3_dir / 'paleos_mgsio3_tables_pt_proteus_solid.dat'
    ext = mgsio3_dir / 'paleos_mgsio3_tables_pt_proteus_solid_lowT.dat'
    liquid = mgsio3_dir / 'paleos_mgsio3_tables_pt_proteus_liquid.dat'
    uni = base_dir / 'EOS_PALEOS_MgSiO3_unified' / 'paleos_mgsio3_eos_table_pt.dat'

    for fpath in (solid, ext, liquid, uni):
        if not fpath.is_file():
            pytest.skip(f'Required real PALEOS table missing: {fpath}')

    kw = dict(
        eos_file=str(uni),
        solidus_func=derive_solidus_from_liquidus(paleos_liquidus, 0.8),
        liquidus_func=paleos_liquidus,
        P_range=(1.0e5, 3.5e11),
        n_P=1350,
        n_S=280,
        solid_eos_file=str(solid),
        liquid_eos_file=str(liquid),
    )

    base = eos_export.generate_spider_eos_tables(**kw)
    ext_out = eos_export.generate_spider_eos_tables(**kw, solid_lowt_file=str(ext))

    n_add = ext_out['S_solid'].size - base['S_solid'].size
    s_min = float(ext_out['S_solid'][0])
    coldest_t = float(ext_out['solid']['temperature'][ext_out['valid']['solid']].min())

    assert n_add == 1
    assert abs(s_min - 25.99) < 0.01
    assert abs(coldest_t - 67.08) < 0.01

    # Bias of (T/cp) against the T(S) slope of an interval, in %, at its nodes (linear in S)
    S, T, cp = ext_out['S_solid'], ext_out['solid']['temperature'], ext_out['solid']['cp']
    valid, P = ext_out['valid']['solid'], ext_out['P_Pa']

    def bias(k, cols):
        slope = (T[k + 1, cols] - T[k, cols]) / (S[k + 1] - S[k])
        ends = np.stack([T[j, cols] / cp[j, cols] for j in (k, k + 1)]) / slope - 1.0
        return 100 * ends.min(), 100 * ends.max()

    both, nxt = valid[0] & valid[1], valid[1] & valid[2]
    assert P[both].max() < 6.75e9 < 7.0e9 < P[~valid[0]].min()
    assert bias(0, both) == pytest.approx((-16.01, 39.71), abs=0.05)
    assert bias(1, nxt & both) == pytest.approx((-3.43, 5.44), abs=0.05)
    assert bias(1, nxt & ~valid[0]) == pytest.approx((-8.02, 13.01), abs=0.05)


def test_an_entropy_jump_in_an_extension_row_keeps_t_monotone_and_close_to_inversion(tmp_path):
    """An extension row whose s jumps by 45 J/kg/K between two nodes, like the clinoenstatite
    boundary, gives a column on that row with T rising in S, finite properties, node temperatures
    equal to a direct inversion of the row, and midpoint temperatures within 0.5 % of it (0.37 %
    here, against 0.06 % without the jump)."""
    solid, liquid, _ = _gen_tables(tmp_path)

    def _jump_edit(P, T, vals):
        if np.isclose(P, _P[1]) and T <= _TG[2]:
            vals[2] -= 45.0

    ext_jump = _write(
        tmp_path / 'ext_jump.dat', [(P, T) for P in _P for T in _TG[:7]], edit=_jump_edit
    )
    out = _generate_from(solid, liquid, ext_jump, p_range=(_P[1], _P[-1]))
    assert out['P_Pa'][0] == pytest.approx(_P[1], rel=1e-12)
    merged = eos_export.load_paleos_with_lowt_extension(solid, ext_jump)
    s_row, lt_row = np.asarray(merged['s'])[1], np.asarray(merged['unique_log_t'])
    keep = np.isfinite(s_row)
    s_row, lt_row = s_row[keep], lt_row[keep]
    assert np.diff(s_row).max() - np.diff(s_row).min() == pytest.approx(45.0, rel=1e-6)

    S, T = out['S_solid'], out['solid']['temperature'][:, 0]
    valid = np.flatnonzero(out['valid']['solid'][:, 0])
    cold = valid[T[valid] < 300.0]
    assert cold.size > 2 and np.all(np.diff(T[valid]) > 0)
    for prop in ('rho', 'cp', 'alpha', 'nabla_ad', 'temperature'):
        assert np.isfinite(out['solid'][prop][valid, 0]).all()
    direct = 10.0 ** np.interp(S[cold], s_row, lt_row)
    np.testing.assert_allclose(T[cold], direct, rtol=1e-4)
    mid_s = 0.5 * (S[cold[:-1]] + S[cold[1:]])
    mid_t = 0.5 * (T[cold[:-1]] + T[cold[1:]])
    assert np.all(np.abs(mid_t / 10.0 ** np.interp(mid_s, s_row, lt_row) - 1.0) < 0.005)


def test_merged_bounds_pmin_pmax_tmin_tmax_full_and_partial_coverage(tmp_path):
    """p_min, p_max, t_min and t_max match base PALEOS bounds across coverage configurations."""
    # Full coverage: PALEOS covers all temperatures down to index 0.
    pal_full = _paleos(tmp_path, [0, 0, 0, 0], name='pal_full.dat')
    ext = _ext(tmp_path, ds=0.0)
    base_full = eos_export.load_paleos_all_properties(pal_full)
    merged_full = eos_export.load_paleos_with_lowt_extension(pal_full, ext)

    assert merged_full['p_min'] == base_full['p_min']
    assert merged_full['p_max'] == base_full['p_max']
    assert merged_full['t_min'] == base_full['t_min']
    assert merged_full['t_max'] == base_full['t_max']

    # Partial coverage: PALEOS stops at index 6 while extension covers rows 1..3 down to index 0.
    pal_part = _paleos(tmp_path, [6, 6, 6, 6], name='pal_part.dat')
    ext_part = _ext(tmp_path, rows=[1, 2, 3], ds=0.0, name='ext_part.dat')
    base_part = eos_export.load_paleos_all_properties(pal_part)
    merged_part = eos_export.load_paleos_with_lowt_extension(pal_part, ext_part)

    assert merged_part['p_min'] == base_part['p_min']
    assert merged_part['p_max'] == base_part['p_max']
    assert merged_part['t_max'] == base_part['t_max']
    assert merged_part['t_min'] == pytest.approx(_T[0], rel=1e-8)


def test_all_nan_paleos_row_is_skipped_and_extension_ignored(tmp_path):
    """An all-NaN PALEOS row is skipped by the merge loop and its extension cells are ignored."""
    lines = ['# table with all-nan row\n']
    for i, P in enumerate(_P):
        for T in _T[6:]:
            if i == 1:
                # Row 1 holds all NaNs.
                body = ' '.join('nan' for _ in range(7))
            else:
                body = ' '.join(f'{v:.8e}' for v in _props(P, T))
            lines.append(f'{P:.8e} {T:.8e} {body} solid\n')
    pal = tmp_path / 'pal_nan.dat'
    pal.write_text(''.join(lines))

    ext = _ext(tmp_path)
    try:
        merged = eos_export.load_paleos_with_lowt_extension(pal, ext)
    except Exception as exc:
        pytest.fail(f'Merge raised unexpected exception on all-NaN row: {exc}')

    assert np.isnan(merged['s'][1]).all()
    assert np.isfinite(merged['s'][0, :6]).all()
    assert np.isfinite(merged['s'][2, :6]).all()


def test_extension_grid_with_t_min_on_node(tmp_path):
    """t_min lying exactly on a grid node terminates at that node with exact expected count."""
    table = tmp_path / 'pal_node.dat'
    t_nodes = [300.0 * (10.0 ** (0.05 * i)) for i in range(5)]
    lines = ['# P T rho u s cp cv alpha nabla_ad phase\n']
    for t_val in t_nodes:
        lines.append(f'1.0e6 {t_val:.12e} 3000 1e5 2000 1000 800 1e-5 0.2 solid\n')
    table.write_text(''.join(lines))

    data = np.loadtxt(table, usecols=(0, 1), comments='#', ndmin=2)
    t_tab = np.unique(data[:, 1])
    log_t = np.log10(t_tab)
    step = np.diff(log_t).mean()
    target_t_min = 10.0 ** (log_t[0] - 2 * step)

    _, T = tool.extension_grid(table, t_min=target_t_min, p_max=1.0e11)

    # Starting at table minimum and descending 2 steps gives 3 nodes (indices 0, 1, 2).
    assert len(T) == 3
    assert T[-1] == pytest.approx(target_t_min, rel=1e-8)


def test_extension_grid_excludes_row_exactly_at_p_max(tmp_path):
    """A pressure row exactly equal to p_max is strictly excluded by extension_grid."""
    table = _paleos(tmp_path, [6, 6, 6, 6])
    data = np.loadtxt(table, usecols=0, comments='#', ndmin=1)
    target_p_max = float(np.unique(data)[2])
    P, _ = tool.extension_grid(table, t_min=180.0, p_max=target_p_max)

    assert target_p_max not in P
    assert np.all(P < target_p_max)


@pytest.mark.parametrize(
    ('solid_file', 'liquid_file'),
    [
        ('solid.dat', None),
        (None, 'liquid.dat'),
        (None, None),
    ],
)
def test_extension_two_phase_gate_parametrized(tmp_path, solid_file, liquid_file):
    """Missing either or both of the two-phase tables refuses low-T extension with ValueError."""
    solid, liquid, ext = _gen_tables(tmp_path)
    with pytest.raises(ValueError, match='needs the two-phase tables'):
        eos_export.generate_spider_eos_tables(
            eos_file=solid,
            solidus_func=lambda P: 1500.0,
            liquidus_func=lambda P: 1800.0,
            P_range=(1.0e6, 1.0e10),
            n_P=6,
            n_S=40,
            solid_eos_file=solid if solid_file else None,
            liquid_eos_file=liquid if liquid_file else None,
            solid_lowt_file=ext,
        )
