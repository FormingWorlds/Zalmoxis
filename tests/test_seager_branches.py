"""Synthetic-cache branch tests for ``zalmoxis.eos.seager.get_tabulated_eos``.

Targets the edge-case branches that the data-file-driven Seager+2007
integration tests do not reach: irregular-grid path for melted/solid
mantle tables, pressure clamping inside the melted/solid branch,
out-of-bounds temperature raise, NaN/None density raise, and the
generic-exception catchall. The function only loads from disk when
``eos_file not in interpolation_functions``; pre-populating the cache
with a synthetic dict bypasses file I/O entirely.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, interp1d

from zalmoxis.eos import seager
from zalmoxis.eos.seager import _fill_invalid_cells, get_tabulated_eos

pytestmark = pytest.mark.unit


def _regular_cache():
    """8x8 regular P-T grid, density = 5000 + 30*P_GPa + 0.5*T."""
    pressures = np.linspace(1e9, 1e12, 8)
    temps = np.linspace(1000.0, 6000.0, 8)
    P, T = np.meshgrid(pressures, temps, indexing='ij')
    rho = 5000.0 + 30.0 * (P / 1e9) + 0.5 * T
    return {
        'type': 'regular',
        'interp': RegularGridInterpolator(
            (pressures, temps), rho, bounds_error=False, fill_value=None
        ),
        'p_min': pressures[0],
        'p_max': pressures[-1],
        't_min': temps[0],
        't_max': temps[-1],
        'unique_pressures': pressures,
        **dict(zip(('t_lo', 't_hi'), _fill_invalid_cells(rho, temps)[1:])),
    }


def _irregular_cache():
    """Two pressures with overlapping but different T-ranges, scattered grid."""
    # P1: T in [1000, 4000], P2: T in [1000, 5000] (ragged top)
    pressures = np.array([1e10, 1e10, 1e10, 1e11, 1e11, 1e11, 1e11])
    temps = np.array([1000.0, 2500.0, 4000.0, 1000.0, 2500.0, 4000.0, 5000.0])
    densities = np.array([4500.0, 4400.0, 4300.0, 6000.0, 5900.0, 5800.0, 5750.0])
    log_p = np.log10(pressures)
    interp = LinearNDInterpolator(np.column_stack([log_p, temps]), densities)
    unique_pressures = np.unique(pressures)
    return {
        'type': 'irregular',
        'interp': interp,
        'p_min': unique_pressures[0],
        'p_max': unique_pressures[-1],
        't_min': 1000.0,
        't_max': 5000.0,
        'p_tmax': {p: temps[pressures == p].max() for p in unique_pressures},
        'unique_pressures': unique_pressures,
    }


def _1d_cache():
    """1D rho(P) interpolator, mimicking Seager2007 core/ice tables."""
    pressures = np.linspace(0.0, 1e13, 100)
    densities = 8000.0 + 5e-9 * pressures  # 8000-58000 kg/m^3
    return {
        'type': '1d',
        'interp': interp1d(pressures, densities, bounds_error=False, fill_value='extrapolate'),
    }


@pytest.fixture
def regular_cache():
    eos_file = '/synthetic/regular.tsv'
    return {
        'eos_file': eos_file,
        'cache': {eos_file: _regular_cache()},
        'mat': {'mantle': {'eos_file': eos_file}},
    }


@pytest.fixture
def irregular_cache():
    eos_file = '/synthetic/irregular.tsv'
    return {
        'eos_file': eos_file,
        'cache': {eos_file: _irregular_cache()},
        'mat': {'mantle': {'eos_file': eos_file}},
    }


@pytest.fixture
def cache_1d():
    eos_file = '/synthetic/iron.csv'
    return {
        'eos_file': eos_file,
        'cache': {eos_file: _1d_cache()},
        'mat': {'core': {'eos_file': eos_file}},
    }


class TestRegularMeltedMantle:
    """Regular-grid path through the ``melted_mantle`` / ``solid_mantle`` branch."""

    def test_in_bounds_returns_finite_density(self, regular_cache):
        rho = get_tabulated_eos(
            5e10,
            {'melted_mantle': {'eos_file': regular_cache['eos_file']}},
            'melted_mantle',
            temperature=3000.0,
            interpolation_functions=regular_cache['cache'],
        )
        assert rho is not None and np.isfinite(rho)
        assert 4000 < rho < 10000

    def test_pressure_below_clamps_up(self, regular_cache):
        """Pressure below ``p_min`` is clamped, not failed."""
        rho_below = get_tabulated_eos(
            1e6,
            {'melted_mantle': {'eos_file': regular_cache['eos_file']}},
            'melted_mantle',
            temperature=3000.0,
            interpolation_functions=regular_cache['cache'],
        )
        rho_at = get_tabulated_eos(
            1e9,
            {'melted_mantle': {'eos_file': regular_cache['eos_file']}},
            'melted_mantle',
            temperature=3000.0,
            interpolation_functions=regular_cache['cache'],
        )
        # Both come back finite; clamped value matches the boundary value.
        assert rho_below == pytest.approx(rho_at, rel=1e-9)

    def test_pressure_above_clamps_down(self, regular_cache):
        rho_above = get_tabulated_eos(
            1e15,
            {'melted_mantle': {'eos_file': regular_cache['eos_file']}},
            'melted_mantle',
            temperature=3000.0,
            interpolation_functions=regular_cache['cache'],
        )
        rho_at = get_tabulated_eos(
            1e12,
            {'melted_mantle': {'eos_file': regular_cache['eos_file']}},
            'melted_mantle',
            temperature=3000.0,
            interpolation_functions=regular_cache['cache'],
        )
        assert rho_above == pytest.approx(rho_at, rel=1e-9)

    def test_temperature_below_t_min_returns_none(self, regular_cache):
        """T below ``t_min`` triggers the explicit ValueError; the outer
        try/except logs and returns None."""
        rho = get_tabulated_eos(
            5e10,
            {'melted_mantle': {'eos_file': regular_cache['eos_file']}},
            'melted_mantle',
            temperature=100.0,  # well below t_min=1000
            interpolation_functions=regular_cache['cache'],
        )
        assert rho is None

    def test_temperature_above_t_max_returns_none(self, regular_cache):
        rho = get_tabulated_eos(
            5e10,
            {'melted_mantle': {'eos_file': regular_cache['eos_file']}},
            'melted_mantle',
            temperature=20000.0,  # above t_max=6000
            interpolation_functions=regular_cache['cache'],
        )
        assert rho is None

    def test_missing_temperature_raises(self, regular_cache):
        """``temperature=None`` for a T-dependent material is a usage error
        and surfaces as a None return (caught by outer try/except)."""
        rho = get_tabulated_eos(
            5e10,
            {'melted_mantle': {'eos_file': regular_cache['eos_file']}},
            'melted_mantle',
            temperature=None,
            interpolation_functions=regular_cache['cache'],
        )
        assert rho is None


class TestIrregularGrid:
    """Irregular-grid path uses the per-pressure local T_max for clamping."""

    def test_inside_local_tmax(self, irregular_cache):
        """A query inside the per-pressure local-T_max returns a finite density."""
        rho = get_tabulated_eos(
            5e10,  # midway between 1e10 and 1e11; local T_max interpolates
            {'melted_mantle': {'eos_file': irregular_cache['eos_file']}},
            'melted_mantle',
            temperature=2000.0,
            interpolation_functions=irregular_cache['cache'],
        )
        assert rho is not None and np.isfinite(rho)
        assert 4000 < rho < 7000

    def test_temperature_above_local_tmax_clamps(self, irregular_cache):
        """T above per-pressure local T_max is clamped to that boundary."""
        rho = get_tabulated_eos(
            1e10,  # T_max here is 4000
            {'melted_mantle': {'eos_file': irregular_cache['eos_file']}},
            'melted_mantle',
            temperature=4500.0,  # above local T_max 4000 at P=1e10
            interpolation_functions=irregular_cache['cache'],
        )
        # Clamping should produce a finite density (not None and not NaN)
        assert rho is not None and np.isfinite(rho)

    def test_irregular_pressure_clamping(self, irregular_cache):
        """Pressure clamping path in the irregular-grid branch."""
        rho = get_tabulated_eos(
            1e6,  # below the table's p_min=1e10
            {'melted_mantle': {'eos_file': irregular_cache['eos_file']}},
            'melted_mantle',
            temperature=2500.0,
            interpolation_functions=irregular_cache['cache'],
        )
        assert rho is not None and np.isfinite(rho)


class TestOneDPath:
    """Pure 1D rho(P) lookup for tabulated core / ice-layer materials."""

    def test_1d_returns_density(self, cache_1d):
        rho = get_tabulated_eos(
            1e11,
            {'core': {'eos_file': cache_1d['eos_file']}},
            'core',
            temperature=300.0,
            interpolation_functions=cache_1d['cache'],
        )
        assert rho is not None
        assert 8000 < rho < 100000

    def test_1d_no_temperature_required(self, cache_1d):
        """Seager2007 1D tables do not require a temperature argument."""
        rho_with_t = get_tabulated_eos(
            5e10,
            {'core': {'eos_file': cache_1d['eos_file']}},
            'core',
            temperature=300.0,
            interpolation_functions=cache_1d['cache'],
        )
        rho_no_t = get_tabulated_eos(
            5e10,
            {'core': {'eos_file': cache_1d['eos_file']}},
            'core',
            temperature=None,
            interpolation_functions=cache_1d['cache'],
        )
        assert rho_with_t == pytest.approx(rho_no_t, rel=1e-12)


class TestUnexpectedExceptionPath:
    """Generic ``except Exception`` catch returns None and logs."""

    def test_torn_cache_returns_none(self, regular_cache):
        """A cache with a missing required field surfaces as None, not as
        an unhandled exception. This exercises the outer ``except Exception``."""
        # Delete a required key so RegularGridInterpolator's call raises a TypeError
        regular_cache['cache'][regular_cache['eos_file']]['interp'] = None
        rho = get_tabulated_eos(
            5e10,
            {'melted_mantle': {'eos_file': regular_cache['eos_file']}},
            'melted_mantle',
            temperature=3000.0,
            interpolation_functions=regular_cache['cache'],
        )
        assert rho is None


FILL = 1e-15
P_AX = np.array([0.0, 1e9, 2e9, 3e9])
T_AX = np.array([500.0, 1000.0, 1500.0, 2000.0, 2500.0, 3000.0, 3500.0])
# Valid T band per pressure node; fill cells below and above it, as in the Wolf & Bower tables.
BAND = {0: (1000.0, 2500.0), 1: (1000.0, 3000.0), 2: (1500.0, 3000.0), 3: (1500.0, 3500.0)}


def _fill_table():
    """Density grid rho = 3000 + 1e-7 P - 0.2 T inside BAND, FILL outside."""
    P, T = np.meshgrid(P_AX, T_AX, indexing='ij')
    rho = 3000.0 + 1e-7 * P - 0.2 * T
    for i, (lo, hi) in BAND.items():
        rho[i, (T_AX < lo) | (T_AX > hi)] = FILL
    return rho


def _write_table(path, rho):
    """Write ``rho`` on (P_AX, T_AX) in the tab-separated P-T-rho format of the mantle tables."""
    rows = [
        f'{p:.10e}\t{t:.10e}\t{rho[i, j]:.10e}'
        for i, p in enumerate(P_AX)
        for j, t in enumerate(T_AX)
    ]
    path.write_text('#pressure temperature density\n' + '\n'.join(rows) + '\n')
    seager._tdep_edge_warned.discard(str(path))
    return {'melted_mantle': {'eos_file': str(path)}}


def _plane(p, t):
    """The valid density of the synthetic table."""
    return 3000.0 + 1e-7 * p - 0.2 * t


@pytest.fixture
def fill_table(tmp_path):
    """The fill table on disk."""
    return _write_table(tmp_path / 'density_melt.dat', _fill_table())


class TestFillCells:
    """Fill cells of a regular mantle table never enter a density."""

    def test_rows_take_their_own_edge_values(self):
        """Below and above its band each row holds its own edge value; no fill is left."""
        rho = _fill_table()
        filled, t_lo, t_hi = _fill_invalid_cells(rho, T_AX)
        np.testing.assert_allclose(t_lo, [1000.0, 1000.0, 1500.0, 1500.0])
        np.testing.assert_allclose(t_hi, [2500.0, 3000.0, 3000.0, 3500.0])
        assert filled.min() > 2000.0
        np.testing.assert_allclose(filled[0], rho[0, [1, 1, 2, 3, 4, 4, 4]])  # band edges
        np.testing.assert_allclose(filled[3, 2:], rho[3, 2:])  # valid cells are unchanged

    def test_gap_takes_the_nearest_valid_node(self):
        """A gap inside a band takes its nearest valid node, the lower one on a tie."""
        rho = _fill_table()
        rho[0, 3] = FILL  # 2000 K: 1500 and 2500 K are equally near
        rho[1, 2:4] = FILL  # 1500 and 2000 K: nearest 1000 K and 2500 K
        filled = _fill_invalid_cells(rho, T_AX)[0]
        assert filled[0, 3] == pytest.approx(rho[0, 2])
        assert (filled[1, 2], filled[1, 3]) == pytest.approx((rho[1, 1], rho[1, 4]))

    def test_nearest_is_measured_in_temperature(self):
        """On a non-uniform T axis a gap takes the node nearest in T, not in index."""
        temps = np.array([0.0, 10.0, 11.0, 12.0, 13.0])
        rho = np.array([[5.0e3, FILL, FILL, FILL, 9.0e3]])
        filled = _fill_invalid_cells(rho, temps)[0]
        np.testing.assert_allclose(filled[0], [5.0e3, 9.0e3, 9.0e3, 9.0e3, 9.0e3])

    @pytest.mark.parametrize(
        ('p', 't', 'band'),
        [
            (5e8, 1000.0, None),  # both rows hold 1000 K
            (5e8, 2500.0, None),  # the upper edge of row 0, inside row 1
            (0.0, 2600.0, '[1000, 2500] K of the pressure row at 0.00e+00'),
            (1e9, 1200.0, None),  # row-1 node: inside row 1, row 2 (from 1500 K) has no weight
            (2e9, 1200.0, '[1500, 3000] K of the pressure row at 2.00e+09'),
            (3e9, 3200.0, None),  # p_max node: inside row 3, row 2 (to 3000 K) has no weight
            (3e9, 1000.0, '[1500, 3500] K of the pressure row at 3.00e+09'),
            (1.5e9, 1200.0, '[1500, 3000] K of the pressure row at 2.00e+09'),
            (1.5e9, 3200.0, '[1000, 3000] K of the pressure row at 1.00e+09'),
        ],
    )
    def test_warning_names_the_band_of_a_used_row(self, fill_table, caplog, p, t, band):
        """The warning fires only outside the band of a row with non-zero weight and names
        that row; a pressure node uses only its own row."""
        with caplog.at_level('WARNING', logger='zalmoxis.eos.seager'):
            get_tabulated_eos(p, fill_table, 'melted_mantle', t, {})
        msgs = [r.getMessage() for r in caplog.records if 'valid band' in r.getMessage()]
        assert len(msgs) == (band is not None)
        if band:
            assert band in msgs[0]

    def test_nan_temperature_does_not_spend_the_warning(self, fill_table, caplog):
        """A NaN T fails its lookup without a band warning; a later real excursion still warns."""
        cache = {}
        with caplog.at_level('WARNING', logger='zalmoxis.eos.seager'):
            assert (
                get_tabulated_eos(2.5e9, fill_table, 'melted_mantle', float('nan'), cache)
                is None
            )
            assert fill_table['melted_mantle']['eos_file'] not in seager._tdep_edge_warned
            get_tabulated_eos(2.5e9, fill_table, 'melted_mantle', 3400.0, cache)
        msgs = [r.getMessage() for r in caplog.records if 'valid band' in r.getMessage()]
        assert len(msgs) == 1 and 'T=3400 K' in msgs[0]

    def test_row_without_valid_cell_gives_none(self, tmp_path, caplog):
        """A row of fill stays NaN; a query next to it fails as a lookup, with no band warning."""
        rho = _fill_table()
        rho[1, :] = FILL
        filled, t_lo, _ = _fill_invalid_cells(rho, T_AX)
        assert np.isnan(filled[1]).all() and np.isnan(t_lo[1])
        mat = _write_table(tmp_path / 'density_melt.dat', rho)
        cache = {}
        with caplog.at_level('WARNING', logger='zalmoxis.eos.seager'):
            assert get_tabulated_eos(5e8, mat, 'melted_mantle', 2000.0, cache) is None
        assert not [r for r in caplog.records if 'valid band' in r.getMessage()]
        assert get_tabulated_eos(2.5e9, mat, 'melted_mantle', 2000.0, cache) == pytest.approx(
            _plane(2.5e9, 2000.0)
        )

    def test_gap_through_the_loader_is_cached(self, tmp_path):
        """A table with a gap loads once; a query at the gap takes the nearest valid node."""
        rho = _fill_table()
        rho[:2, 3] = FILL  # 2000 K in rows 0 and 1
        mat = _write_table(tmp_path / 'density_melt.dat', rho)
        cache = {}
        got = get_tabulated_eos(5e8, mat, 'melted_mantle', 2000.0, cache)
        assert mat['melted_mantle']['eos_file'] in cache
        assert got == pytest.approx(0.5 * (_plane(0.0, 1500.0) + _plane(1e9, 1500.0)))
        assert get_tabulated_eos(5e8, mat, 'melted_mantle', 2000.0, cache) == got

    def test_inside_band_equals_plain_bilinear(self, fill_table):
        """A query in a valid cell is unchanged and matches the analytic plane."""
        cache = {}
        for p, t in ((5e8, 1700.0), (2.5e9, 2900.0), (1.2e9, 1500.0), (0.0, 1000.0)):
            rho = get_tabulated_eos(p, fill_table, 'melted_mantle', t, cache)
            assert rho == pytest.approx(_plane(p, t), rel=1e-12)

    def test_beyond_the_edge_combines_valid_values_and_warns_once(self, fill_table, caplog):
        """Hot and cold queries combine each row's edge value; one warning names the table."""
        cache = {}
        with caplog.at_level('WARNING', logger='zalmoxis.eos.seager'):
            hot = get_tabulated_eos(2.5e9, fill_table, 'melted_mantle', 3400.0, cache)
            cold = get_tabulated_eos(2.5e9, fill_table, 'melted_mantle', 600.0, cache)
        assert hot == pytest.approx(
            0.5 * (_plane(2e9, 3000.0) + _plane(3e9, 3400.0)), rel=1e-12
        )
        assert cold == pytest.approx(
            0.5 * (_plane(2e9, 1500.0) + _plane(3e9, 1500.0)), rel=1e-12
        )
        plain = RegularGridInterpolator((P_AX, T_AX), _fill_table())((2.5e9, 3400.0))
        assert plain < hot - 500.0  # the raw grid mixes a fill corner in
        msgs = [r.getMessage() for r in caplog.records if 'valid band' in r.getMessage()]
        assert len(msgs) == 1
        assert (
            'density_melt.dat' in msgs[0]
            and 'T=3400 K' in msgs[0]
            and 'P=2.50e+09 Pa' in msgs[0]
        )

    def test_density_does_not_fall_with_pressure_at_fixed_hot_t(self, fill_table):
        """At a temperature above every band the density never falls with pressure."""
        cache = {}
        for t in (3400.0, 3500.0):
            rho = [
                get_tabulated_eos(p, fill_table, 'melted_mantle', t, cache)
                for p in np.linspace(0.0, 3e9, 301)
            ]
            assert np.min(np.diff(rho)) >= -1e-9

    def test_sweep_is_finite_and_positive(self, fill_table):
        """Every density over a P-T sweep of the table is finite and above the fill."""
        cache = {}
        vals = np.array(
            [
                get_tabulated_eos(p, fill_table, 'melted_mantle', t, cache)
                for p in np.linspace(0.0, 3e9, 13)
                for t in np.linspace(500.0, 3500.0, 25)
            ]
        )
        assert np.all(np.isfinite(vals))
        assert vals.min() > 2000.0
        assert vals.max() < 3400.0
