"""Tests for the low-temperature extension of a PALEOS solid table.

The extension table continues the PALEOS log-T grid below the lowest PALEOS temperature.
PALEOS values take priority wherever a PALEOS row holds them; an extension cell is used
only below the lowest finite PALEOS temperature of its pressure row, and the seam between
the two is checked on every row that uses the extension.
"""

from __future__ import annotations

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
    return (
        4000.0 + 1.0e-7 * P - 0.05 * T + drho,
        1.0e3 * T,
        1000.0 * np.log(T / 300.0) + 1.0e-9 * P + 2000.0 + ds,
        1200.0,
        1100.0,
        1.0e-5,
        0.3,
    )


def _write(path: Path, cells, ds=0.0, drho=0.0):
    """Write a 10-column table from an iterable of (P, T) cells."""
    lines = ['# synthetic solid table\n']
    for P, T in cells:
        vals = ' '.join(f'{v:.8e}' for v in _props(P, T, ds, drho))
        lines.append(f'{P:.8e} {T:.8e} {vals} solid-lpcen\n')
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
        out = eos_export.load_paleos_with_lowt_extension(pal, ext)
        for key in ('unique_log_p', 'unique_log_t', *_COLS):
            assert np.array_equal(out[key], alone[key], equal_nan=True), key
        assert np.array_equal(out['phase'], alone['phase'])
        assert out['t_min'] == alone['t_min']

    @pytest.mark.parametrize(('ds', 'drho'), [(1.0e-2, 0.0), (0.0, 1.0e-2), (-1.0e-2, 0.0)])
    def test_seam_guard_raises_on_a_mismatch(self, tmp_path, ds, drho):
        """s or rho at the seam differing by more than 1e-6 relative stops the load."""
        pal = _paleos(tmp_path, [6, 6, 6, 6])
        with pytest.raises(ValueError, match='seam'):
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
        with pytest.raises(ValueError, match='seam'):
            eos_export.load_paleos_with_lowt_extension(pal, _ext(tmp_path, k_top=5))

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
        with pytest.raises(ValueError, match='pressure'):
            eos_export.load_paleos_with_lowt_extension(pal, _write(tmp_path / 'e.dat', cells))

    def test_temperature_off_the_paleos_grid_is_an_error(self, tmp_path):
        """An extension node that does not continue the PALEOS log-T step is refused."""
        pal = _paleos(tmp_path, [6, 6, 6, 6])
        T_off = 10.0 ** (np.log10(300.0) - 0.03 * np.arange(0, 7))
        cells = [(P, T) for P in _P for T in T_off]
        with pytest.raises(ValueError, match='grid'):
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
                assert out['s'][i, k] == pytest.approx(expect, rel=1e-9, abs=1e-6)


_TG = 10.0 ** (np.log10(300.0) + np.arange(-6, 21) * _DLOGT)


def _gen_tables(tmp_path):
    """Solid and liquid PALEOS tables from 300 K, and a solid extension down to 150 K."""
    solid = _write(tmp_path / 'solid.dat', [(P, T) for P in _P for T in _TG[6:]])
    lines = ['# synthetic liquid table\n']
    for P in _P:
        for T in _TG[6:]:
            vals = list(_props(P, T))
            vals[0] *= 0.95
            vals[2] *= 1.08
            lines.append(f'{P:.8e} {T:.8e} ' + ' '.join(f'{v:.8e}' for v in vals) + ' liquid\n')
    liquid = tmp_path / 'liquid.dat'
    liquid.write_text(''.join(lines))
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
        s_back = 1000.0 * np.log(T / 300.0) + 1.0e-9 * out['P_Pa'][None, :] + 2000.0
        np.testing.assert_allclose(s_back[cold], S[cold], atol=0.05)

    def test_extension_needs_the_two_phase_tables(self, tmp_path):
        solid, _, ext = _gen_tables(tmp_path)
        with pytest.raises(ValueError, match='two-phase'):
            eos_export.generate_spider_eos_tables(
                eos_file=solid,
                solidus_func=lambda P: 1500.0,
                liquidus_func=lambda P: 1800.0,
                P_range=(1.0e6, 1.0e10),
                n_P=6,
                n_S=40,
                solid_lowt_file=ext,
            )
