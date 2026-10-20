"""Tests for the low-temperature extension of a PALEOS solid table.

The extension table continues the PALEOS log-T grid below the lowest PALEOS temperature.
PALEOS values take priority wherever a PALEOS row holds them; an extension cell is used
only below the lowest finite PALEOS temperature of its pressure row, and the seam between
the two is checked on every row that uses the extension.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from zalmoxis import eos_export

pytestmark = pytest.mark.unit

_P = np.logspace(6.0, 10.0, 4)
_DLOGT = 0.05
_K = np.arange(-6, 5)
_T = 10.0 ** (np.log10(300.0) + _K * _DLOGT)
_COLS = ('rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad')


def _props(P, T, ds=0.0, drho=0.0):
    """Synthetic solid properties; ``ds`` and ``drho`` offset s and rho."""
    lt = np.log10(T)
    return (
        4000.0 + 1.0e-7 * P - 20.0 * lt + drho,
        1.0e3 * T,
        1000.0 * np.log(T / 300.0) + 2.0 * np.log10(P) + 2000.0 + ds,
        900.0 + 100.0 * lt,
        800.0 + 100.0 * lt,
        1.0e-5 * (1.0 + lt),
        0.2 + 0.01 * lt,
    )


def _write(path: Path, cells, ds=0.0, drho=0.0, edit=None, phase='solid-lpcen'):
    """Write a 10-column table from (P, T) cells; ``edit(P, T, vals)`` may change a cell's
    values in place, or leave the cell out by returning False."""
    lines = ['# synthetic table\n']
    for P, T in cells:
        vals = list(_props(P, T, ds, drho))
        if edit is not None and edit(P, T, vals) is False:
            continue
        lines.append(f'{P:.8e} {T:.8e} ' + ' '.join(f'{v:.8e}' for v in vals) + f' {phase}\n')
    path.write_text(''.join(lines))
    return path


def _paleos(tmp_path, kmin_per_row, name='paleos.dat'):
    """PALEOS table whose row ``i`` starts at grid index ``kmin_per_row[i]``."""
    cells = [(P, T) for P, k0 in zip(_P, kmin_per_row) for T in _T[k0:]]
    return _write(tmp_path / name, cells)


def _ext(tmp_path, k_top=6, rows=None, ds=0.0, drho=0.0, name='ext.dat'):
    """Extension table over grid indices 0..k_top on the given rows (all by default)."""
    rows = range(len(_P)) if rows is None else rows
    cells = [(_P[i], T) for i in rows for T in _T[: k_top + 1]]
    return _write(tmp_path / name, cells, ds, drho)


@pytest.fixture(autouse=True)
def _fresh_table_cache():
    eos_export._parse_paleos_table.cache_clear()
    yield
    eos_export._parse_paleos_table.cache_clear()


class TestMergeLowTExtension:
    """``load_paleos_with_lowt_extension`` merges an extension under a PALEOS table."""

    def test_extension_fills_only_below_each_row_minimum(self, tmp_path):
        """Rows starting at 300 K take extension cells below 300 K; a row starting lower keeps its own."""
        pal = _paleos(tmp_path, [6, 6, 6, 4])
        out = eos_export.load_paleos_with_lowt_extension(pal, _ext(tmp_path, ds=0.0))
        np.testing.assert_allclose(10.0 ** out['unique_log_t'], _T, rtol=1e-8)
        for i, P in enumerate(_P):
            for k, T in enumerate(_T):
                np.testing.assert_allclose(out['s'][i, k], _props(P, T)[2], rtol=1e-8)
        assert out['t_min'] == pytest.approx(_T[0], rel=1e-8)

    def test_paleos_value_wins_where_both_hold_one(self, tmp_path):
        """A row PALEOS covers down to index 4 keeps PALEOS at 4 and 5, not the extension."""
        pal = _paleos(tmp_path, [6, 6, 6, 4])
        ext = _ext(tmp_path, ds=5.0e-4)
        out = eos_export.load_paleos_with_lowt_extension(pal, ext)
        np.testing.assert_allclose(out['s'][3, 4:7], [_props(_P[3], T)[2] for T in _T[4:7]])
        np.testing.assert_allclose(
            out['s'][3, :4], [_props(_P[3], T)[2] + 5.0e-4 for T in _T[:4]]
        )

    def test_full_coverage_uses_no_extension_cell(self, tmp_path):
        """PALEOS down to the lowest extension node on every row: output equals PALEOS alone."""
        pal = _paleos(tmp_path, [0, 0, 0, 0])
        ext = _ext(tmp_path, ds=50.0, drho=-30.0)
        alone = {k: np.array(v) for k, v in eos_export.load_paleos_all_properties(pal).items()}
        eos_export._parse_paleos_table.cache_clear()
        try:
            out = eos_export.load_paleos_with_lowt_extension(pal, ext)
        except ValueError as exc:
            pytest.fail(f'full PALEOS coverage was refused: {exc}')
        for key in ('unique_log_p', 'unique_log_t', *_COLS):
            assert np.array_equal(out[key], alone[key], equal_nan=True), key
        assert np.array_equal(out['phase'], alone['phase'])
        assert out['t_min'] == alone['t_min']

    @pytest.mark.parametrize(('ds', 'drho'), [(1.0e-2, 0.0), (0.0, 1.0e-2), (-1.0e-2, 0.0)])
    def test_seam_guard_raises_on_a_mismatch(self, tmp_path, ds, drho):
        """s or rho at the seam differing by more than 1e-6 relative stops the load."""
        pal = _paleos(tmp_path, [6, 6, 6, 6])
        with pytest.raises(ValueError, match='seam mismatch in'):
            eos_export.load_paleos_with_lowt_extension(pal, _ext(tmp_path, ds=ds, drho=drho))

    def test_seam_guard_accepts_a_round_off_difference(self, tmp_path):
        """A 1e-8 relative entropy difference at the seam is written-precision noise."""
        pal = _paleos(tmp_path, [6, 6, 6, 6])
        ext = _ext(tmp_path, ds=2.0e-5)
        out = eos_export.load_paleos_with_lowt_extension(pal, ext)
        assert np.isfinite(out['s'][:, :6]).all()

    def test_extension_without_the_seam_node_is_an_error(self, tmp_path):
        """The seam can only be checked when the extension holds the PALEOS row minimum."""
        pal = _paleos(tmp_path, [6, 6, 6, 6])
        with pytest.raises(ValueError, match='no common seam node') as err:
            eos_export.load_paleos_with_lowt_extension(pal, _ext(tmp_path, k_top=5))
        assert str(pal) in str(err.value) and 'ext.dat ends at T = 2.673753e+02 K' in str(
            err.value
        )

    def test_row_without_extension_values_keeps_paleos(self, tmp_path):
        """A pressure row the extension leaves out stays as PALEOS has it, NaN below."""
        pal = _paleos(tmp_path, [6, 6, 6, 6])
        out = eos_export.load_paleos_with_lowt_extension(pal, _ext(tmp_path, rows=[0, 1]))
        assert np.isnan(out['s'][2:, :6]).all()
        assert np.isfinite(out['s'][:2, :6]).all()

    def test_different_pressure_axis_is_an_error(self, tmp_path):
        """The extension must sit on the PALEOS pressure rows."""
        pal = _paleos(tmp_path, [6, 6, 6, 6])
        cells = [(P * 1.01, T) for P in _P for T in _T[:7]]
        with pytest.raises(ValueError, match='is not one of those'):
            eos_export.load_paleos_with_lowt_extension(pal, _write(tmp_path / 'e.dat', cells))

    def test_temperature_off_the_paleos_grid_is_an_error(self, tmp_path):
        """An extension node that does not continue the PALEOS log-T step is refused."""
        pal = _paleos(tmp_path, [6, 6, 6, 6])
        T_off = 10.0 ** (np.log10(300.0) - 0.03 * np.arange(0, 7))
        cells = [(P, T) for P in _P for T in T_off]
        with pytest.raises(ValueError, match='do not continue the PALEOS log-T grid'):
            eos_export.load_paleos_with_lowt_extension(pal, _write(tmp_path / 'e.dat', cells))

    def test_merged_arrays_are_read_only(self, tmp_path):
        out = eos_export.load_paleos_with_lowt_extension(
            _paleos(tmp_path, [6, 6, 6, 6]), _ext(tmp_path)
        )
        with pytest.raises(ValueError):
            out['s'][0, 0] = 0.0

    @settings(
        max_examples=60,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(kmin=st.lists(st.integers(0, 6), min_size=len(_P), max_size=len(_P)))
    def test_each_cell_comes_from_paleos_where_paleos_covers(self, tmp_path, kmin):
        """Any per-row PALEOS coverage: PALEOS cells unchanged, extension exactly below them."""
        eos_export._parse_paleos_table.cache_clear()
        pal = _paleos(tmp_path, kmin, name='p.dat')
        out = eos_export.load_paleos_with_lowt_extension(
            pal, _ext(tmp_path, ds=3.0e-4, name='x.dat')
        )
        for i, k0 in enumerate(kmin):
            for k, T in enumerate(_T):
                expect = _props(_P[i], T)[2] + (3.0e-4 if k < k0 else 0.0)
                assert out['s'][i, k] == pytest.approx(expect, rel=0.0, abs=2.0e-5)


_TG = 10.0 ** (np.log10(300.0) + np.arange(-6, 21) * _DLOGT)


def _gen_tables(tmp_path):
    """Solid and liquid PALEOS tables from 300 K, and a solid extension down to 150 K."""
    solid = _write(tmp_path / 'solid.dat', [(P, T) for P in _P for T in _TG[6:]])

    def denser_hotter(P, T, vals):
        vals[0] *= 0.95
        vals[2] *= 1.08

    cells = [(P, T) for P in _P for T in _TG[6:]]
    liquid = _write(tmp_path / 'liquid.dat', cells, edit=denser_hotter, phase='liquid')
    ext = _write(tmp_path / 'ext.dat', [(P, T) for P in _P for T in _TG[:7]])
    return solid, liquid, ext


def _generate(tables, lowt=None):
    solid, liquid, ext = tables
    return eos_export.generate_spider_eos_tables(
        eos_file=solid,
        solidus_func=lambda P: 1500.0,
        liquidus_func=lambda P: 1800.0,
        P_range=(1.0e6, 1.0e10),
        n_P=6,
        n_S=40,
        solid_eos_file=solid,
        liquid_eos_file=liquid,
        solid_lowt_file=ext if lowt else None,
    )


def _generate_from(solid, liquid, ext, p_range=(1.0e6, 1.0e10), n_S=40):
    """Generator call of _generate with a given extension file, pressure range and node count."""
    return eos_export.generate_spider_eos_tables(
        eos_file=solid,
        solidus_func=lambda P: 1500.0,
        liquidus_func=lambda P: 1800.0,
        P_range=p_range,
        n_P=6,
        n_S=n_S,
        solid_eos_file=solid,
        liquid_eos_file=liquid,
        solid_lowt_file=ext,
    )


class TestLowTExtensionThatAddsNothing:
    """An extension that adds no solid entropy node says so instead of passing silently, and
    states how many cells it still fills."""

    def _run(self, tmp_path, caplog, ext, n_S=40):
        solid, liquid, _ = _gen_tables(tmp_path)
        base = _generate_from(solid, liquid, None, n_S=n_S)
        eos_export._parse_paleos_table.cache_clear()
        with caplog.at_level(logging.INFO, logger='zalmoxis.eos_export'):
            out = _generate_from(solid, liquid, ext, n_S=n_S)
        assert np.array_equal(out['S_solid'], base['S_solid'])
        changed = int(np.sum(out['solid']['temperature'] != base['solid']['temperature']))
        return [r for r in caplog.records if r.levelno == logging.WARNING], changed

    def test_an_extension_with_only_the_seam_temperature_warns(self, tmp_path, caplog):
        """An extension whose rows hold only the 300 K seam node reaches no lower temperature."""
        ext = _write(tmp_path / 'seam.dat', [(P, _TG[6]) for P in _P])
        warns, changed = self._run(tmp_path, caplog, ext)
        assert any('reaches no temperature below' in r.getMessage() for r in warns)
        assert changed == 0

    def test_an_extension_with_no_finite_entropy_warns(self, tmp_path, caplog, monkeypatch):
        """An extended entropy that is NaN everywhere adds no node and changes no cell."""
        real_load = eos_export.load_paleos_with_lowt_extension

        def load(solid, ext):
            merged = real_load(solid, ext)
            return merged | {'s': np.full_like(merged['s'], np.nan)}

        monkeypatch.setattr(eos_export, 'load_paleos_with_lowt_extension', load)
        warns, changed = self._run(tmp_path, caplog, _gen_tables(tmp_path)[2])
        assert any('gives no finite entropy at the output' in r.getMessage() for r in warns)
        assert changed == 0

    def test_an_extension_too_shallow_for_one_node_warns(self, tmp_path, caplog):
        """One temperature step below 300 K lowers the entropy by less than one node step: the
        axis keeps its nodes, but the extension still fills cells that PALEOS cannot invert, and
        the log gives their number."""
        ext = _write(tmp_path / 'one.dat', [(P, T) for P in _P for T in _TG[5:7]])
        warns, changed = self._run(tmp_path, caplog, ext, n_S=4)
        assert changed > 0
        assert any(
            'no solid entropy node fits below' in r.getMessage()
            and 'can still fill' in r.getMessage()
            for r in warns
        )
        msgs = [r.getMessage() for r in caplog.records]
        assert any(f'Low-T extension: {changed} solid cells' in m for m in msgs), msgs

    def test_an_extension_that_adds_nodes_logs_the_new_minimum(self, tmp_path, caplog):
        """The added nodes, the new lowest solid entropy and the number of filled cells (the gain in
        valid solid cells) are logged at INFO, with no warning."""
        solid, liquid, ext = _gen_tables(tmp_path)
        base = _generate_from(solid, liquid, None)
        eos_export._parse_paleos_table.cache_clear()
        with caplog.at_level(logging.INFO, logger='zalmoxis.eos_export'):
            out = _generate_from(solid, liquid, ext)
        msgs = [r.getMessage() for r in caplog.records]
        gain = int(out['valid']['solid'].sum() - base['valid']['solid'].sum())
        assert gain > 0 and any(f'Low-T extension: {gain} solid cells' in m for m in msgs)
        assert any(f'down to {out["S_solid"][0]:.0f}' in m for m in msgs)
        assert any(
            m.startswith(f'Entropy ranges: solid [{out["S_solid"][0]:.0f},') for m in msgs
        )
        assert not [
            r
            for r in caplog.records
            if 'Low-T extension' in r.getMessage() and r.levelno == logging.WARNING
        ]


class TestGenerateSpiderTablesLowT:
    """``generate_spider_eos_tables(..., solid_lowt_file=...)`` extends the solid P-S tables."""

    def test_present_solid_cells_are_bit_identical(self, tmp_path):
        """Every solid cell valid without the extension keeps its value; the S nodes are kept."""
        tables = _gen_tables(tmp_path)
        base = _generate(tables)
        eos_export._parse_paleos_table.cache_clear()
        ext = _generate(tables, lowt=True)
        n_add = ext['S_solid'].size - base['S_solid'].size
        assert n_add > 0
        assert np.array_equal(ext['S_solid'][n_add:], base['S_solid'])
        step = np.diff(base['S_solid']).mean()
        np.testing.assert_allclose(np.diff(ext['S_solid'][: n_add + 1]), step, rtol=1e-9)
        valid = base['valid']['solid']
        for prop in ('rho', 'temperature', 'cp', 'alpha', 'nabla_ad'):
            assert np.array_equal(ext['solid'][prop][n_add:][valid], base['solid'][prop][valid])
        for prop in ('rho', 'temperature', 'cp', 'alpha', 'nabla_ad'):
            assert np.array_equal(ext['melt'][prop], base['melt'][prop], equal_nan=True)

    def test_cells_below_300_k_invert_the_extension(self, tmp_path):
        """The cells that read the flat fill now hold T below 300 K, rising with S."""
        tables = _gen_tables(tmp_path)
        out = _generate(tables, lowt=True)
        T = out['solid']['temperature']
        cold = out['valid']['solid'] & (T < 299.0)
        assert cold.sum() > 0
        assert T[cold].min() < 200.0
        for ip in range(T.shape[1]):
            col = T[out['valid']['solid'][:, ip], ip]
            assert np.all(np.diff(col) > 0)
        S = out['S_solid'][:, None] * np.ones(T.shape)
        s_back = 1000.0 * np.log(T / 300.0) + 2.0 * np.log10(out['P_Pa'])[None, :] + 2000.0
        np.testing.assert_allclose(s_back[cold], S[cold], atol=0.05)

    def test_cells_below_300_k_carry_the_extension_properties(self, tmp_path):
        """rho, cp, alpha and nabla_ad of the new cells are those of the extension at the inverted T."""
        out = _generate(_gen_tables(tmp_path), lowt=True)
        T, v = out['solid']['temperature'], out['valid']['solid']
        checked = 0
        for j in (0, T.shape[1] - 1):
            P = out['P_Pa'][j]
            cold = v[:, j] & (T[:, j] < 299.0)
            expect = np.array([_props(P, t) for t in T[cold, j]])
            np.testing.assert_allclose(out['solid']['rho'][cold, j], expect[:, 0], rtol=1e-6)
            np.testing.assert_allclose(out['solid']['cp'][cold, j], expect[:, 3], rtol=1e-6)
            np.testing.assert_allclose(out['solid']['alpha'][cold, j], expect[:, 5], rtol=1e-6)
            np.testing.assert_allclose(
                out['solid']['nabla_ad'][cold, j], expect[:, 6] * T[cold, j] / P, rtol=1e-6
            )
            checked += cold.sum()
        assert checked >= 4

    def test_column_without_a_solidus_gets_no_low_temperature_cells(self, tmp_path):
        """Where the solidus is NaN the PALEOS pass skips the column, and so does the extension pass."""
        solid, liquid, ext = _gen_tables(tmp_path)
        out = eos_export.generate_spider_eos_tables(
            eos_file=solid,
            solidus_func=lambda P: np.nan if P > 5.0e9 else 1500.0,
            liquidus_func=lambda P: 1800.0,
            P_range=(1.0e6, 1.0e10),
            n_P=6,
            n_S=40,
            solid_eos_file=solid,
            liquid_eos_file=liquid,
            solid_lowt_file=ext,
        )
        high = out['P_Pa'] > 5.0e9
        assert high.any() and not out['valid']['solid'][:, high].any()
        assert out['valid']['solid'][:, ~high].any()

    def test_extension_needs_the_two_phase_tables(self, tmp_path):
        solid, _, ext = _gen_tables(tmp_path)
        with pytest.raises(ValueError, match='needs the two-phase tables'):
            eos_export.generate_spider_eos_tables(
                eos_file=solid,
                solidus_func=lambda P: 1500.0,
                liquidus_func=lambda P: 1800.0,
                P_range=(1.0e6, 1.0e10),
                n_P=6,
                n_S=40,
                solid_lowt_file=ext,
            )


class TestLowTInvariants:
    """The merge refuses an extension that the S -> T inversion cannot use."""

    def _ext_with(self, tmp_path, edit):
        cells = [(P, T) for P in _P for T in _T[:7]]
        return _write(tmp_path / 'ext_edit.dat', cells, edit=edit)

    def test_entropy_that_falls_with_temperature_is_an_error(self, tmp_path):
        def edit(P, T, vals):
            if P == _P[1] and T == _T[2]:
                vals[2] = _props(P, _T[0])[2] - 1.0

        pal = _paleos(tmp_path, [6, 6, 6, 6])
        with pytest.raises(ValueError, match='s does not increase with T'):
            eos_export.load_paleos_with_lowt_extension(pal, self._ext_with(tmp_path, edit))

    def test_equal_entropy_on_two_nodes_is_an_error(self, tmp_path):
        def edit(P, T, vals):
            if P == _P[2] and T == _T[3]:
                vals[2] = _props(P, _T[2])[2]

        pal = _paleos(tmp_path, [6, 6, 6, 6])
        with pytest.raises(ValueError, match='s does not increase with T'):
            eos_export.load_paleos_with_lowt_extension(pal, self._ext_with(tmp_path, edit))

    @pytest.mark.parametrize('cp', [0.0, -5.0])
    def test_heat_capacity_not_positive_is_an_error(self, tmp_path, cp):
        def edit(P, T, vals):
            if P == _P[0] and T == _T[1]:
                vals[3] = cp

        pal = _paleos(tmp_path, [6, 6, 6, 6])
        with pytest.raises(ValueError, match='rho, cp or cv is not positive'):
            eos_export.load_paleos_with_lowt_extension(pal, self._ext_with(tmp_path, edit))

    def test_a_bad_cell_that_paleos_covers_is_ignored(self, tmp_path):
        """An extension cell PALEOS covers is never used, so it is not checked."""

        def edit(P, T, vals):
            if P == _P[0] and T == _T[6]:
                vals[3] = -1.0

        pal = _paleos(tmp_path, [5, 6, 6, 6])
        out = eos_export.load_paleos_with_lowt_extension(pal, self._ext_with(tmp_path, edit))
        assert out['cp'][0, 6] == pytest.approx(_props(_P[0], _T[6])[3], rel=1e-8)


def test_no_solid_node_below_the_lowest_extension_entropy(tmp_path):
    """Each added entropy node holds an inverted cell on some column, so none is only a fill."""
    tables = _gen_tables(tmp_path)
    out = _generate(tables, lowt=True)
    base_nodes = _generate(tables)['S_solid'].size
    n_add = out['S_solid'].size - base_nodes
    assert n_add > 0
    assert out['valid']['solid'][:n_add].any(axis=1).all()


def _ext_cells(tmp_path, edit):
    """Extension over grid indices 0..6 with ``edit(i, k, vals)`` per cell (row i, node k)."""
    index = {(P, T): (i, k) for i, P in enumerate(_P) for k, T in enumerate(_T[:7])}
    return _write(
        tmp_path / 'ext_cells.dat', index, edit=lambda P, T, v: edit(*index[(P, T)], v)
    )


class TestLowTMergeGuards:
    """A used extension row must be complete and finite from its lowest used node up to the seam."""

    def test_hole_inside_a_used_row_is_an_error(self, tmp_path):
        """A missing node between the lowest used node and the seam stops the load."""
        pal = _paleos(tmp_path, [6, 6, 6, 6])
        ext = _ext_cells(tmp_path, lambda i, k, v: not (i == 1 and k == 3))
        with pytest.raises(ValueError, match='a missing or non-finite extension cell'):
            eos_export.load_paleos_with_lowt_extension(pal, ext)
        assert np.isfinite(
            eos_export.load_paleos_with_lowt_extension(pal, _ext(tmp_path))['s']
        ).all()

    @pytest.mark.parametrize('col', [0, 1, 2, 3, 4, 5, 6])
    def test_nan_seam_value_is_an_error(self, tmp_path, col):
        """A NaN in any property at the seam node is refused, not compared as 'no mismatch'."""

        def edit(i, k, v):
            if i == 2 and k == 6:
                v[col] = float('nan')

        pal = _paleos(tmp_path, [6, 6, 6, 6])
        with pytest.raises(ValueError, match='no finite seam value'):
            eos_export.load_paleos_with_lowt_extension(pal, _ext_cells(tmp_path, edit))

    @pytest.mark.parametrize(
        ('col', 'value'),
        [
            (0, float('nan')),
            (1, float('nan')),
            (2, float('nan')),
            (3, float('nan')),
            (4, float('nan')),
            (5, float('nan')),
            (5, float('inf')),
            (6, float('nan')),
        ],
    )
    def test_used_cell_that_is_not_finite_is_an_error(self, tmp_path, col, value):
        """Every property of a used cell must be finite."""

        def edit(i, k, v):
            if i == 0 and k == 2:
                v[col] = value

        pal = _paleos(tmp_path, [6, 6, 6, 6])
        with pytest.raises(ValueError, match='a missing or non-finite extension cell'):
            eos_export.load_paleos_with_lowt_extension(pal, _ext_cells(tmp_path, edit))
        assert np.isfinite(
            eos_export.load_paleos_with_lowt_extension(pal, _ext(tmp_path))['s']
        ).all()

    @pytest.mark.parametrize(
        ('col', 'value'), [(0, 0.0), (0, -1.0), (3, 0.0), (4, 0.0), (4, -1.0)]
    )
    def test_used_cell_with_rho_cp_or_cv_not_positive_is_an_error(self, tmp_path, col, value):
        """rho, cp and cv of a used cell must be above 0."""

        def edit(i, k, v):
            if i == 0 and k == 2:
                v[col] = value

        pal = _paleos(tmp_path, [6, 6, 6, 6])
        with pytest.raises(ValueError, match='rho, cp or cv is not positive'):
            eos_export.load_paleos_with_lowt_extension(pal, _ext_cells(tmp_path, edit))
        assert np.isfinite(
            eos_export.load_paleos_with_lowt_extension(pal, _ext(tmp_path))['s']
        ).all()

    @pytest.mark.parametrize(('col', 'value'), [(5, -1.0e-6), (5, 0.0), (6, -0.1), (6, 0.0)])
    def test_used_cell_may_hold_alpha_or_nabla_ad_of_either_sign(self, tmp_path, col, value):
        """alpha and nabla_ad may be zero or negative in a used cell; only finiteness is required."""

        def edit(i, k, v):
            if i == 0 and k == 2:
                v[col] = value

        pal = _paleos(tmp_path, [6, 6, 6, 6])
        try:
            out = eos_export.load_paleos_with_lowt_extension(pal, _ext_cells(tmp_path, edit))
        except ValueError as exc:
            pytest.fail(f'the extension was refused: {exc}')
        key = ('alpha', 'nabla_ad')[col - 5]
        assert out[key][0, 2] == value
        assert np.isfinite(out['s']).all()

    def test_entropy_not_rising_into_the_seam_is_an_error(self, tmp_path):
        """The step from the highest extension node into the seam must rise in s as well."""

        def edit(i, k, v):
            if i == 1 and k == 5:
                v[2] = _props(_P[1], _T[6])[2]

        pal = _paleos(tmp_path, [6, 6, 6, 6])
        with pytest.raises(ValueError, match='s does not increase with T'):
            eos_export.load_paleos_with_lowt_extension(pal, _ext_cells(tmp_path, edit))
        assert np.isfinite(
            eos_export.load_paleos_with_lowt_extension(pal, _ext(tmp_path))['s']
        ).all()

    def test_extension_phase_labels_reach_the_merged_cells(self, tmp_path):
        """Below the seam the merged table takes the phase label of the extension."""
        pal = _paleos(tmp_path, [6, 6, 6, 6])
        ext = _write(
            tmp_path / 'ext_hp.dat', [(P, T) for P in _P for T in _T[:7]], phase='solid-hpcen'
        )
        out = eos_export.load_paleos_with_lowt_extension(pal, ext)
        assert set(out['phase'][:, :6].ravel()) == {'solid-hpcen'}
        assert set(out['phase'][:, 6:].ravel()) == {'solid-lpcen'}

    def test_non_finite_paleos_value_at_the_seam_is_named(self, tmp_path):
        """A PALEOS seam cell with a finite s but another non-finite property is named as such."""

        def nan_cp(P, T, v):
            if P == _P[2] and T == _T[6]:
                v[3] = float('nan')

        pal = _write(
            tmp_path / 'pal_nan.dat', [(P, T) for P in _P for T in _T[6:]], edit=nan_cp
        )
        with pytest.raises(ValueError, match='PALEOS seam value is not finite'):
            eos_export.load_paleos_with_lowt_extension(pal, _ext(tmp_path))
        assert np.isfinite(
            eos_export.load_paleos_with_lowt_extension(
                _paleos(tmp_path, [6, 6, 6, 6]), _ext(tmp_path)
            )['s']
        ).all()

    @pytest.mark.parametrize(
        ('k', 'frac', 'ok'), [(6, 5.0e-4, True), (6, 5.0e-3, False), (3, 5.0e-3, False)]
    )
    def test_node_offsets_against_the_grid_tolerances(self, tmp_path, k, frac, ok):
        """A node off the PALEOS grid by 5e-4 of a step maps onto it; 5e-3 of a step is refused,
        at the seam node (node matching) and below it (grid continuation)."""
        pal = _paleos(tmp_path, [6, 6, 6, 6])
        T_k = 10.0 ** (np.log10(_T[k]) + frac * _DLOGT)
        lines = ['# extension with one node off the grid\n']
        for P in _P:
            for kk, T in enumerate(_T[:7]):
                vals = ' '.join(f'{v:.8e}' for v in _props(P, T))
                lines.append(f'{P:.8e} {T_k if kk == k else T:.8e} {vals} solid-lpcen\n')
        ext = tmp_path / 'ext_off.dat'
        ext.write_text(''.join(lines))
        if ok:
            out = eos_export.load_paleos_with_lowt_extension(pal, ext)
            assert out['unique_log_t'].size == _T.size
            assert np.isfinite(out['s']).all()
        else:
            with pytest.raises(ValueError, match='do not continue the PALEOS log-T grid'):
                eos_export.load_paleos_with_lowt_extension(pal, ext)
            assert np.isfinite(
                eos_export.load_paleos_with_lowt_extension(pal, _ext(tmp_path))['s']
            ).all()

    @pytest.mark.parametrize('col', [1, 3, 4, 5, 6])
    def test_seam_mismatch_in_any_property_is_an_error(self, tmp_path, col):
        """Every property, not only s and rho, must join PALEOS at the seam to 1e-6."""

        def edit(i, k, v):
            if k == 6:
                v[col] *= 1.0 + 1.0e-3

        pal = _paleos(tmp_path, [6, 6, 6, 6])
        with pytest.raises(ValueError, match='seam mismatch in'):
            eos_export.load_paleos_with_lowt_extension(pal, _ext_cells(tmp_path, edit))
        assert np.isfinite(
            eos_export.load_paleos_with_lowt_extension(pal, _ext(tmp_path))['s']
        ).all()

    def test_node_missing_on_every_extension_row_is_a_hole(self, tmp_path):
        """A node that PALEOS holds on one row but the extension leaves out leaves a gap below the
        seam of the other rows."""
        pal = _paleos(tmp_path, [6, 6, 6, 4])
        ext = _ext_cells(tmp_path, lambda i, k, v: k not in (4, 5))
        with pytest.raises(ValueError, match='a missing or non-finite extension cell'):
            eos_export.load_paleos_with_lowt_extension(pal, ext)
        assert np.isfinite(
            eos_export.load_paleos_with_lowt_extension(pal, _ext(tmp_path))['s']
        ).all()

    def test_row_starting_above_the_extension_top_is_an_error(self, tmp_path):
        """A PALEOS row whose lowest temperature lies above the extension has no seam node."""
        pal = _paleos(tmp_path, [8, 6, 6, 6])
        with pytest.raises(ValueError, match='no common seam node') as err:
            eos_export.load_paleos_with_lowt_extension(pal, _ext(tmp_path))
        assert f'{pal}: the PALEOS row at P = 1.000000e+06 Pa starts at T = 3.776' in str(
            err.value
        )
        assert 'ext.dat ends at T = 3.000000e+02 K' in str(err.value)
        assert np.isfinite(
            eos_export.load_paleos_with_lowt_extension(
                _paleos(tmp_path, [6, 6, 6, 6], name='p6.dat'), _ext(tmp_path)
            )['s']
        ).all()

    def test_paleos_table_with_one_temperature_is_an_error(self, tmp_path):
        """One PALEOS temperature gives no log-T step for the extension to continue."""
        pal = _write(tmp_path / 'one_t.dat', [(P, _T[6]) for P in _P])
        with pytest.raises(ValueError, match='needs at least two temperatures'):
            eos_export.load_paleos_with_lowt_extension(pal, _ext(tmp_path))
        two = _write(tmp_path / 'two_t.dat', [(P, T) for P in _P for T in _T[6:8]])
        assert np.isfinite(
            eos_export.load_paleos_with_lowt_extension(two, _ext(tmp_path))['s']
        ).all()

    def test_missing_nodes_below_the_used_run_are_not_a_hole(self, tmp_path):
        """Nodes absent below the lowest used node only shorten the row."""
        pal = _paleos(tmp_path, [6, 6, 6, 6])
        out = eos_export.load_paleos_with_lowt_extension(
            pal, _ext_cells(tmp_path, lambda i, k, v: not (i == 3 and k < 2))
        )
        assert np.isnan(out['s'][3, :2]).all()
        assert np.isfinite(out['s'][3, 2:]).all()

    def test_nodes_off_by_written_precision_still_match(self, tmp_path):
        """A node that differs from the PALEOS node by 2e-7 relative maps onto it."""
        pal = _paleos(tmp_path, [6, 6, 6, 6])
        cells = [
            (P, T * (1.0 + 2.0e-7) if k == 6 else T) for P in _P for k, T in enumerate(_T[:7])
        ]
        ext = _write(tmp_path / 'ext_off.dat', cells)
        out = eos_export.load_paleos_with_lowt_extension(pal, ext)
        assert out['unique_log_t'].size == _T.size
        np.testing.assert_allclose(10.0 ** out['unique_log_t'], _T, rtol=1e-8)

    def test_leading_column_without_a_finite_value_is_dropped(self, tmp_path):
        """A lowest node that holds NaN on every row leaves no empty column in the table."""

        def edit(i, k, v):
            if k == 0:
                v[2] = float('nan')

        pal = _paleos(tmp_path, [6, 6, 6, 6])
        out = eos_export.load_paleos_with_lowt_extension(pal, _ext_cells(tmp_path, edit))
        assert out['t_min'] == pytest.approx(_T[1], rel=1e-8)
        assert out['unique_log_t'].size == _T.size - 1


@pytest.mark.filterwarnings('ignore:loadtxt. input contained no data')
def test_extension_without_data_rows_is_an_error(tmp_path):
    """An extension file that holds only comments is refused with a named error."""
    ext = tmp_path / 'empty.dat'
    ext.write_text('# only a header\n')
    with pytest.raises(Exception) as exc:
        eos_export.load_paleos_with_lowt_extension(_paleos(tmp_path, [6, 6, 6, 6]), ext)
    assert exc.type is ValueError, repr(exc.value)
    assert 'holds no data rows' in str(exc.value)
    assert np.isfinite(
        eos_export.load_paleos_with_lowt_extension(
            _paleos(tmp_path, [6, 6, 6, 6]), _ext(tmp_path)
        )['s']
    ).all()


def test_extension_needs_two_entropy_nodes(tmp_path):
    """With one entropy node the axis has no step, so the extension cannot add nodes."""
    solid, liquid, ext = _gen_tables(tmp_path)
    kw = dict(
        eos_file=solid,
        solidus_func=lambda P: 1500.0,
        liquidus_func=lambda P: 1800.0,
        P_range=(1.0e6, 1.0e10),
        n_P=6,
        solid_eos_file=solid,
        liquid_eos_file=liquid,
    )
    with pytest.raises(ValueError, match='needs n_S >= 2'):
        eos_export.generate_spider_eos_tables(**kw, n_S=1, solid_lowt_file=ext)
    assert (
        eos_export.generate_spider_eos_tables(**kw, n_S=2, solid_lowt_file=ext)['S_solid'].size
        >= 2
    )


def test_missing_two_phase_tables_stop_before_the_extension_is_read(tmp_path, monkeypatch):
    """The two-phase check comes before the extension load: the loader is never reached."""
    solid, _, ext = _gen_tables(tmp_path)

    calls = []

    def loaded(*args):
        calls.append(args)
        raise ValueError('extension loaded')

    monkeypatch.setattr(eos_export, 'load_paleos_with_lowt_extension', loaded)
    with pytest.raises(ValueError, match='needs the two-phase tables'):
        eos_export.generate_spider_eos_tables(
            eos_file=solid,
            solidus_func=lambda P: 1500.0,
            liquidus_func=lambda P: 1800.0,
            P_range=(1.0e6, 1.0e10),
            n_P=6,
            n_S=40,
            solid_lowt_file=ext,
        )
    assert calls == []


def test_a_stale_paleos_table_under_the_extension_raises(tmp_path, monkeypatch):
    """The PALEOS table under the extension is checked against its registry pin: the pinned
    release merges, a stale one raises StaleTableError; the extension file, which no registry
    lists, is not checked."""
    import hashlib

    from zalmoxis.eos import registry_guard
    from zalmoxis.eos.registry_guard import StaleTableError

    monkeypatch.setenv('FWL_IO_OFFLINE', '1')
    table = _paleos(tmp_path, [6, 6, 6, 6], name='paleos_mgsio3_tables_pt_proteus_solid.dat')
    ext = _ext(tmp_path)
    pins = {
        table.name: {'interior.eos.paleos_mgsio3': hashlib.md5(table.read_bytes()).hexdigest()}
    }
    monkeypatch.setattr(registry_guard, '_pins', lambda: pins)
    registry_guard._HASHES.clear()
    assert np.isfinite(eos_export.load_paleos_with_lowt_extension(table, ext)['s']).all()
    pins[table.name] = {'interior.eos.paleos_mgsio3': '0' * 32}
    registry_guard._HASHES.clear()
    eos_export._parse_paleos_table.cache_clear()
    with pytest.raises(StaleTableError, match='paleos_mgsio3_tables_pt_proteus_solid.dat'):
        eos_export.load_paleos_with_lowt_extension(table, ext)
