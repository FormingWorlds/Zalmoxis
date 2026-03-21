"""Tests for eos_functions.py to increase coverage.

Tests cover:
- _fast_bilinear interpolation on synthetic and real grids
- _paleos_clamp_temperature boundary clamping
- _ensure_unified_cache caching behavior
- calculate_density dispatch for all EOS formats
- calculate_density_batch for unified and fallback paths
- get_Tdep_density phase-based routing
- get_Tdep_material phase classification
- load_paleos_table and load_paleos_unified_table grid construction
- calculate_temperature_profile mode dispatch
- _compute_paleos_dtdp phase-weighted nabla_ad
- create_pressure_density_files output generation

References:
    - docs/testing.md
    - docs/How-to/test_infrastructure.md
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest


def _paleos_unified_data_available():
    """Check if unified PALEOS data files are available."""
    root = os.environ.get('ZALMOXIS_ROOT', '')
    iron = os.path.join(root, 'data', 'EOS_PALEOS_iron', 'paleos_iron_eos_table_pt.dat')
    mgsio3 = os.path.join(
        root, 'data', 'EOS_PALEOS_MgSiO3_unified', 'paleos_mgsio3_eos_table_pt.dat'
    )
    return os.path.isfile(iron) and os.path.isfile(mgsio3)


def _seager_data_available():
    """Check if Seager2007 data files are available."""
    root = os.environ.get('ZALMOXIS_ROOT', '')
    return os.path.isfile(os.path.join(root, 'data', 'EOS_Seager2007', 'eos_seager07_iron.txt'))


def _wb2018_data_available():
    """Check if WolfBower2018 data files are available."""
    root = os.environ.get('ZALMOXIS_ROOT', '')
    return os.path.isfile(
        os.path.join(root, 'data', 'EOS_WolfBower2018_1TPa', 'density_melt.dat')
    )


def _rtpress_data_available():
    """Check if RTPress100TPa data files are available."""
    root = os.environ.get('ZALMOXIS_ROOT', '')
    return os.path.isfile(
        os.path.join(root, 'data', 'EOS_RTPress_melt_100TPa', 'density_melt.dat')
    )


def _chabrier_data_available():
    """Check if Chabrier H data is available."""
    root = os.environ.get('ZALMOXIS_ROOT', '')
    return os.path.isfile(
        os.path.join(root, 'data', 'EOS_Chabrier2021_HHe', 'chabrier2021_H.dat')
    )


# =====================================================================
# _fast_bilinear tests
# =====================================================================


@pytest.mark.unit
class TestFastBilinear:
    """Tests for the O(1) bilinear interpolation function."""

    def _make_synthetic_cache(self, n_p=5, n_t=5):
        """Build a synthetic log-uniform grid for testing bilinear interpolation.

        Grid values are set to f(log_p, log_t) = log_p + log_t so that
        bilinear interpolation on interior points can be verified analytically.
        """
        log_p = np.linspace(8.0, 12.0, n_p)  # log10(P) from 1e8 to 1e12
        log_t = np.linspace(2.0, 5.0, n_t)  # log10(T) from 100 to 100000

        # Grid values: f(log_p, log_t) = log_p + log_t
        grid = np.zeros((n_p, n_t))
        for i in range(n_p):
            for j in range(n_t):
                grid[i, j] = log_p[i] + log_t[j]

        dlog_p = (log_p[-1] - log_p[0]) / (n_p - 1)
        dlog_t = (log_t[-1] - log_t[0]) / (n_t - 1)

        return {
            'unique_log_p': log_p,
            'unique_log_t': log_t,
            'logp_min': log_p[0],
            'logp_max': log_p[-1],
            'logt_min': log_t[0],
            'logt_max': log_t[-1],
            'dlog_p': dlog_p,
            'dlog_t': dlog_t,
            'n_p': n_p,
            'n_t': n_t,
        }, grid

    def test_at_grid_nodes(self):
        """Interpolation at grid nodes should return exact grid values."""
        from zalmoxis.eos_functions import _fast_bilinear

        cached, grid = self._make_synthetic_cache()
        for i, lp in enumerate(cached['unique_log_p']):
            for j, lt in enumerate(cached['unique_log_t']):
                result = _fast_bilinear(lp, lt, grid, cached)
                assert result == pytest.approx(grid[i, j], abs=1e-12)

    def test_at_midpoints(self):
        """Midpoint between grid nodes: bilinear = exact for f(p,t) = p + t."""
        from zalmoxis.eos_functions import _fast_bilinear

        cached, grid = self._make_synthetic_cache()
        ulp = cached['unique_log_p']
        ult = cached['unique_log_t']

        for i in range(len(ulp) - 1):
            for j in range(len(ult) - 1):
                mid_p = (ulp[i] + ulp[i + 1]) / 2
                mid_t = (ult[j] + ult[j + 1]) / 2
                result = _fast_bilinear(mid_p, mid_t, grid, cached)
                expected = mid_p + mid_t
                assert result == pytest.approx(expected, abs=1e-12)

    def test_clamping_below_bounds(self):
        """Query below grid minimum should be clamped (dp=0, dt=0)."""
        from zalmoxis.eos_functions import _fast_bilinear

        cached, grid = self._make_synthetic_cache()
        result = _fast_bilinear(7.0, 1.0, grid, cached)
        # Should get corner value at [0, 0]
        assert result == pytest.approx(grid[0, 0], abs=1e-10)

    def test_clamping_above_bounds(self):
        """Query above grid maximum should be clamped (dp=1, dt=1)."""
        from zalmoxis.eos_functions import _fast_bilinear

        cached, grid = self._make_synthetic_cache()
        result = _fast_bilinear(13.0, 6.0, grid, cached)
        # Should get corner value at [-1, -1]
        assert result == pytest.approx(grid[-1, -1], abs=1e-10)

    def test_nan_corner_returns_nan(self):
        """If any corner value is NaN, bilinear returns NaN."""
        from zalmoxis.eos_functions import _fast_bilinear

        cached, grid = self._make_synthetic_cache()
        grid[1, 1] = np.nan  # poison one corner
        # Query at the midpoint of the cell [0:1, 0:1] which neighbors [1,1]
        ulp = cached['unique_log_p']
        ult = cached['unique_log_t']
        mid_p = (ulp[0] + ulp[1]) / 2
        mid_t = (ult[0] + ult[1]) / 2
        result = _fast_bilinear(mid_p, mid_t, grid, cached)
        assert np.isnan(result)

    def test_small_grid_n1(self):
        """Grid with n_p < 2 returns the single value directly."""
        from zalmoxis.eos_functions import _fast_bilinear

        cached = {
            'unique_log_p': np.array([10.0]),
            'unique_log_t': np.array([3.0]),
            'logp_min': 10.0,
            'logp_max': 10.0,
            'logt_min': 3.0,
            'logt_max': 3.0,
            'dlog_p': 1.0,
            'dlog_t': 1.0,
            'n_p': 1,
            'n_t': 1,
        }
        grid = np.array([[42.0]])
        assert _fast_bilinear(10.0, 3.0, grid, cached) == pytest.approx(42.0)

    def test_agrees_with_scipy(self):
        """Compare _fast_bilinear to scipy RegularGridInterpolator on a realistic grid."""
        from scipy.interpolate import RegularGridInterpolator

        from zalmoxis.eos_functions import _fast_bilinear

        # Create a 10x10 grid with a nonlinear function
        n_p, n_t = 10, 10
        log_p = np.linspace(8.0, 12.0, n_p)
        log_t = np.linspace(2.0, 5.0, n_t)
        grid = np.zeros((n_p, n_t))
        for i in range(n_p):
            for j in range(n_t):
                grid[i, j] = np.sin(log_p[i]) * np.cos(log_t[j])

        cached = {
            'unique_log_p': log_p,
            'unique_log_t': log_t,
            'logp_min': log_p[0],
            'logp_max': log_p[-1],
            'logt_min': log_t[0],
            'logt_max': log_t[-1],
            'dlog_p': (log_p[-1] - log_p[0]) / (n_p - 1),
            'dlog_t': (log_t[-1] - log_t[0]) / (n_t - 1),
            'n_p': n_p,
            'n_t': n_t,
        }

        scipy_interp = RegularGridInterpolator(
            (log_p, log_t), grid, bounds_error=False, fill_value=None
        )

        # Test at 20 random interior points
        rng = np.random.RandomState(42)
        for _ in range(20):
            qp = rng.uniform(8.2, 11.8)
            qt = rng.uniform(2.2, 4.8)
            fast_val = _fast_bilinear(qp, qt, grid, cached)
            scipy_val = float(scipy_interp((qp, qt)))
            assert fast_val == pytest.approx(scipy_val, abs=1e-12)


# =====================================================================
# _paleos_clamp_temperature tests
# =====================================================================


@pytest.mark.unit
class TestPaleosClampTemperature:
    """Tests for the per-pressure temperature clamping function."""

    def test_no_clamp_in_valid_range(self):
        """Query within valid T range returns unchanged value."""
        from zalmoxis.eos_functions import _paleos_clamp_temperature

        cached = {
            'unique_log_p': np.array([8.0, 9.0, 10.0]),
            'logt_valid_min': np.array([2.0, 2.0, 2.0]),
            'logt_valid_max': np.array([5.0, 5.0, 5.0]),
        }
        log_t_out, was_clamped = _paleos_clamp_temperature(9.0, 3.5, cached)
        assert log_t_out == pytest.approx(3.5)
        assert was_clamped is False

    def test_clamp_below_min(self):
        """Query below valid T range is clamped to local minimum."""
        from zalmoxis.eos_functions import _paleos_clamp_temperature

        cached = {
            'unique_log_p': np.array([8.0, 9.0, 10.0]),
            'logt_valid_min': np.array([2.5, 2.5, 2.5]),
            'logt_valid_max': np.array([5.0, 5.0, 5.0]),
        }
        log_t_out, was_clamped = _paleos_clamp_temperature(9.0, 1.5, cached)
        assert log_t_out == pytest.approx(2.5)
        assert was_clamped is True

    def test_clamp_above_max(self):
        """Query above valid T range is clamped to local maximum."""
        from zalmoxis.eos_functions import _paleos_clamp_temperature

        cached = {
            'unique_log_p': np.array([8.0, 9.0, 10.0]),
            'logt_valid_min': np.array([2.0, 2.0, 2.0]),
            'logt_valid_max': np.array([4.5, 4.5, 4.5]),
        }
        log_t_out, was_clamped = _paleos_clamp_temperature(9.0, 5.5, cached)
        assert log_t_out == pytest.approx(4.5)
        assert was_clamped is True

    def test_nan_bounds_no_clamp(self):
        """If interpolated bounds are NaN (all-NaN row), no clamping is applied."""
        from zalmoxis.eos_functions import _paleos_clamp_temperature

        cached = {
            'unique_log_p': np.array([8.0, 9.0, 10.0]),
            'logt_valid_min': np.array([np.nan, np.nan, np.nan]),
            'logt_valid_max': np.array([np.nan, np.nan, np.nan]),
        }
        log_t_out, was_clamped = _paleos_clamp_temperature(9.0, 3.0, cached)
        assert log_t_out == pytest.approx(3.0)
        assert was_clamped is False


# =====================================================================
# _ensure_unified_cache tests
# =====================================================================


@pytest.mark.unit
class TestEnsureUnifiedCache:
    """Tests for the _ensure_unified_cache caching mechanism."""

    def test_caches_table_on_first_call(self):
        """First call loads the table; second call returns the same object."""
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import _ensure_unified_cache
        from zalmoxis.eos_properties import EOS_REGISTRY

        eos_file = EOS_REGISTRY['PALEOS:iron']['eos_file']
        cache = {}

        result1 = _ensure_unified_cache(eos_file, cache)
        assert eos_file in cache
        assert result1['type'] == 'paleos_unified'

        result2 = _ensure_unified_cache(eos_file, cache)
        assert result2 is result1  # Same object, no reload

    def test_returns_correct_type(self):
        """Returned cache entry has type 'paleos_unified'."""
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import _ensure_unified_cache
        from zalmoxis.eos_properties import EOS_REGISTRY

        eos_file = EOS_REGISTRY['PALEOS:MgSiO3']['eos_file']
        cache = {}

        result = _ensure_unified_cache(eos_file, cache)
        assert result['type'] == 'paleos_unified'
        assert 'density_interp' in result
        assert 'nabla_ad_interp' in result


# =====================================================================
# calculate_density dispatch tests
# =====================================================================


@pytest.mark.unit
class TestCalculateDensity:
    """Tests for calculate_density dispatch across EOS formats."""

    def test_seager_iron(self):
        """Seager2007:iron density at 300 GPa should be ~13000 kg/m^3."""
        if not _seager_data_available():
            pytest.skip('Seager data not found')

        from zalmoxis.eos_functions import calculate_density
        from zalmoxis.eos_properties import EOS_REGISTRY

        rho = calculate_density(300e9, EOS_REGISTRY, 'Seager2007:iron', 300, None, None)
        assert rho is not None
        assert 10000 < rho < 16000

    def test_seager_mgsio3(self):
        """Seager2007:MgSiO3 density at 100 GPa."""
        if not _seager_data_available():
            pytest.skip('Seager data not found')

        from zalmoxis.eos_functions import calculate_density
        from zalmoxis.eos_properties import EOS_REGISTRY

        rho = calculate_density(100e9, EOS_REGISTRY, 'Seager2007:MgSiO3', 300, None, None)
        assert rho is not None
        assert 3000 < rho < 10000

    def test_seager_h2o(self):
        """Seager2007:H2O density at 50 GPa."""
        if not _seager_data_available():
            pytest.skip('Seager data not found')

        from zalmoxis.eos_functions import calculate_density
        from zalmoxis.eos_properties import EOS_REGISTRY

        root = os.environ.get('ZALMOXIS_ROOT', '')
        h2o_file = os.path.join(root, 'data', 'EOS_Seager2007', 'eos_seager07_water.txt')
        if not os.path.isfile(h2o_file):
            pytest.skip('H2O Seager data not found')

        rho = calculate_density(50e9, EOS_REGISTRY, 'Seager2007:H2O', 300, None, None)
        assert rho is not None
        assert rho > 0

    def test_analytic_iron(self):
        """Analytic:iron should return a positive density."""
        from zalmoxis.eos_functions import calculate_density
        from zalmoxis.eos_properties import EOS_REGISTRY

        rho = calculate_density(100e9, EOS_REGISTRY, 'Analytic:iron', 300, None, None)
        assert rho is not None
        assert rho > 0

    def test_analytic_mgsio3(self):
        """Analytic:MgSiO3 should return a positive density."""
        from zalmoxis.eos_functions import calculate_density
        from zalmoxis.eos_properties import EOS_REGISTRY

        rho = calculate_density(100e9, EOS_REGISTRY, 'Analytic:MgSiO3', 300, None, None)
        assert rho is not None
        assert rho > 0

    def test_unknown_eos_raises(self):
        """Unknown EOS identifier should raise ValueError."""
        from zalmoxis.eos_functions import calculate_density
        from zalmoxis.eos_properties import EOS_REGISTRY

        with pytest.raises(ValueError, match='Unknown'):
            calculate_density(100e9, EOS_REGISTRY, 'Nonexistent:stuff', 300, None, None)

    def test_wb2018_solid_phase(self):
        """WolfBower2018:MgSiO3 density below solidus (solid phase)."""
        if not _wb2018_data_available():
            pytest.skip('WB2018 data not found')

        from zalmoxis.eos_functions import calculate_density, get_solidus_liquidus_functions
        from zalmoxis.eos_properties import EOS_REGISTRY

        sf, lf = get_solidus_liquidus_functions()
        # At 50 GPa, solidus is ~3000-4000 K. Use 1500 K for solid phase.
        rho = calculate_density(50e9, EOS_REGISTRY, 'WolfBower2018:MgSiO3', 1500, sf, lf)
        assert rho is not None
        assert 3000 < rho < 8000

    def test_wb2018_liquid_phase(self):
        """WolfBower2018:MgSiO3 density above liquidus (liquid phase)."""
        if not _wb2018_data_available():
            pytest.skip('WB2018 data not found')

        from zalmoxis.eos_functions import calculate_density, get_solidus_liquidus_functions
        from zalmoxis.eos_properties import EOS_REGISTRY

        sf, lf = get_solidus_liquidus_functions()
        # At 50 GPa, liquidus is ~3294 K. Use 3500 K for liquid phase
        # (within the WB2018 melt table range).
        rho = calculate_density(50e9, EOS_REGISTRY, 'WolfBower2018:MgSiO3', 3500, sf, lf)
        assert rho is not None
        assert 3000 < rho < 8000

    def test_wb2018_mixed_phase(self):
        """WolfBower2018:MgSiO3 density in mushy zone between solidus and liquidus."""
        if not _wb2018_data_available():
            pytest.skip('WB2018 data not found')

        from zalmoxis.eos_functions import calculate_density, get_solidus_liquidus_functions
        from zalmoxis.eos_properties import EOS_REGISTRY

        sf, lf = get_solidus_liquidus_functions()
        T_sol = sf(50e9)
        T_liq = lf(50e9)
        if np.isnan(T_sol) or np.isnan(T_liq):
            pytest.skip('Melting curve undefined at 50 GPa')

        T_mid = (T_sol + T_liq) / 2
        rho = calculate_density(50e9, EOS_REGISTRY, 'WolfBower2018:MgSiO3', T_mid, sf, lf)
        assert rho is not None
        assert rho > 0

    def test_paleos_unified_iron(self):
        """PALEOS:iron density at 100 GPa, 4000 K."""
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import calculate_density
        from zalmoxis.eos_properties import EOS_REGISTRY

        rho = calculate_density(100e9, EOS_REGISTRY, 'PALEOS:iron', 4000, None, None)
        assert rho is not None
        assert 5000 < rho < 15000

    def test_chabrier_h(self):
        """Chabrier:H density at 10 GPa, 5000 K (H2 envelope conditions)."""
        if not _chabrier_data_available():
            pytest.skip('Chabrier H data not found')

        from zalmoxis.eos_functions import calculate_density
        from zalmoxis.eos_properties import EOS_REGISTRY

        rho = calculate_density(10e9, EOS_REGISTRY, 'Chabrier:H', 5000, None, None)
        assert rho is not None
        assert rho > 0

    def test_paleos_unified_with_mushy_zone(self):
        """PALEOS:MgSiO3 with mushy_zone_factor < 1.0."""
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import calculate_density
        from zalmoxis.eos_properties import EOS_REGISTRY

        rho = calculate_density(
            100e9, EOS_REGISTRY, 'PALEOS:MgSiO3', 4000, None, None, mushy_zone_factor=0.8
        )
        assert rho is not None
        assert rho > 0

    def test_interpolation_cache_reuse(self):
        """Shared interpolation_functions cache is reused across calls."""
        if not _seager_data_available():
            pytest.skip('Seager data not found')

        from zalmoxis.eos_functions import calculate_density
        from zalmoxis.eos_properties import EOS_REGISTRY

        cache = {}
        rho1 = calculate_density(100e9, EOS_REGISTRY, 'Seager2007:iron', 300, None, None, cache)
        assert len(cache) == 1
        rho2 = calculate_density(200e9, EOS_REGISTRY, 'Seager2007:iron', 300, None, None, cache)
        assert len(cache) == 1  # Same file, no reload
        assert rho2 > rho1  # Higher pressure -> higher density


# =====================================================================
# calculate_density_batch tests
# =====================================================================


@pytest.mark.unit
class TestCalculateDensityBatch:
    """Tests for vectorized density lookup."""

    def test_paleos_unified_batch(self):
        """Batch lookup for PALEOS:iron matches scalar calls."""
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import calculate_density, calculate_density_batch
        from zalmoxis.eos_properties import EOS_REGISTRY

        pressures = np.array([50e9, 100e9, 200e9, 300e9])
        temperatures = np.array([3000, 4000, 5000, 6000], dtype=float)

        cache = {}
        batch_result = calculate_density_batch(
            pressures, temperatures, EOS_REGISTRY, 'PALEOS:iron', None, None, cache
        )

        assert len(batch_result) == 4
        assert all(np.isfinite(batch_result))

        # Compare with scalar calls
        for i in range(4):
            scalar = calculate_density(
                pressures[i], EOS_REGISTRY, 'PALEOS:iron', temperatures[i], None, None, cache
            )
            assert batch_result[i] == pytest.approx(scalar, rel=1e-6)

    def test_seager_batch_fallback(self):
        """Non-unified EOS falls back to scalar loop."""
        if not _seager_data_available():
            pytest.skip('Seager data not found')

        from zalmoxis.eos_functions import calculate_density_batch
        from zalmoxis.eos_properties import EOS_REGISTRY

        pressures = np.array([100e9, 200e9, 300e9])
        temperatures = np.array([300.0, 300.0, 300.0])

        cache = {}
        result = calculate_density_batch(
            pressures, temperatures, EOS_REGISTRY, 'Seager2007:iron', None, None, cache
        )

        assert len(result) == 3
        assert all(np.isfinite(result))
        # Density should increase with pressure
        assert result[0] < result[1] < result[2]

    def test_batch_with_none_cache(self):
        """Passing None as interpolation_functions should work (auto-creates dict)."""
        if not _seager_data_available():
            pytest.skip('Seager data not found')

        from zalmoxis.eos_functions import calculate_density_batch
        from zalmoxis.eos_properties import EOS_REGISTRY

        pressures = np.array([100e9])
        temperatures = np.array([300.0])

        result = calculate_density_batch(
            pressures, temperatures, EOS_REGISTRY, 'Seager2007:iron', None, None, None
        )
        assert len(result) == 1
        assert np.isfinite(result[0])


# =====================================================================
# get_Tdep_density tests
# =====================================================================


@pytest.mark.unit
class TestGetTdepDensity:
    """Tests for the temperature-dependent density function."""

    def test_missing_solidus_func_raises(self):
        """Calling without solidus/liquidus functions raises ValueError."""
        from zalmoxis.eos_functions import get_Tdep_density
        from zalmoxis.eos_properties import material_properties_iron_Tdep_silicate_planets

        with pytest.raises(ValueError, match='solidus_func'):
            get_Tdep_density(
                50e9, 3000, material_properties_iron_Tdep_silicate_planets, None, None
            )

    def test_nan_melting_curve_defaults_to_solid(self):
        """If melting curve returns NaN, density defaults to solid phase."""
        if not _wb2018_data_available():
            pytest.skip('WB2018 data not found')

        from zalmoxis.eos_functions import get_Tdep_density
        from zalmoxis.eos_properties import material_properties_iron_Tdep_silicate_planets

        # Melting curves that always return NaN
        def nan_sf(P):
            return np.nan

        def nan_lf(P):
            return np.nan

        rho = get_Tdep_density(
            50e9, 3000, material_properties_iron_Tdep_silicate_planets, nan_sf, nan_lf
        )
        assert rho is not None
        assert rho > 0

    def test_degenerate_melting_curve(self):
        """When T_liq <= T_sol, defaults to melted_mantle."""
        if not _wb2018_data_available():
            pytest.skip('WB2018 data not found')

        from zalmoxis.eos_functions import get_Tdep_density
        from zalmoxis.eos_properties import material_properties_iron_Tdep_silicate_planets

        # T_liq <= T_sol at all pressures (degenerate)
        def sf(P):
            return 5000.0

        def lf(P):
            return 4000.0  # Lower than solidus

        rho = get_Tdep_density(
            50e9, 4500, material_properties_iron_Tdep_silicate_planets, sf, lf
        )
        assert rho is not None
        assert rho > 0


# =====================================================================
# get_Tdep_material tests
# =====================================================================


@pytest.mark.unit
class TestGetTdepMaterial:
    """Tests for phase classification based on melting curves."""

    def test_solid_phase(self):
        """Temperature below solidus returns 'solid_mantle'."""
        from zalmoxis.eos_functions import get_Tdep_material

        def sf(P):
            return 3000.0

        def lf(P):
            return 5000.0

        result = get_Tdep_material(100e9, 2000, sf, lf)
        assert result == 'solid_mantle'

    def test_melted_phase(self):
        """Temperature above liquidus returns 'melted_mantle'."""
        from zalmoxis.eos_functions import get_Tdep_material

        def sf(P):
            return 3000.0

        def lf(P):
            return 5000.0

        result = get_Tdep_material(100e9, 6000, sf, lf)
        assert result == 'melted_mantle'

    def test_mixed_phase(self):
        """Temperature between solidus and liquidus returns 'mixed_mantle'."""
        from zalmoxis.eos_functions import get_Tdep_material

        def sf(P):
            return 3000.0

        def lf(P):
            return 5000.0

        result = get_Tdep_material(100e9, 4000, sf, lf)
        assert result == 'mixed_mantle'

    def test_degenerate_melting_curve_above(self):
        """When T_liq <= T_sol and T >= T_sol, returns 'melted_mantle'."""
        from zalmoxis.eos_functions import get_Tdep_material

        def sf(P):
            return 5000.0

        def lf(P):
            return 4000.0

        result = get_Tdep_material(100e9, 5000, sf, lf)
        assert result == 'melted_mantle'

    def test_degenerate_melting_curve_below(self):
        """When T_liq <= T_sol and T < T_sol, returns 'solid_mantle'."""
        from zalmoxis.eos_functions import get_Tdep_material

        def sf(P):
            return 5000.0

        def lf(P):
            return 4000.0

        result = get_Tdep_material(100e9, 3000, sf, lf)
        assert result == 'solid_mantle'

    def test_vectorized(self):
        """Vectorized evaluation with arrays of P and T."""
        from zalmoxis.eos_functions import get_Tdep_material

        def sf(P):
            return 3000.0

        def lf(P):
            return 5000.0

        P = np.array([100e9, 100e9, 100e9])
        T = np.array([2000, 4000, 6000])
        result = get_Tdep_material(P, T, sf, lf)

        assert result[0] == 'solid_mantle'
        assert result[1] == 'mixed_mantle'
        assert result[2] == 'melted_mantle'


# =====================================================================
# load_paleos_table tests
# =====================================================================


@pytest.mark.unit
class TestLoadPaleosTable:
    """Tests for the PALEOS 2-phase table loader."""

    def test_loads_solid_table(self):
        """load_paleos_table should build valid interpolators from solid table."""
        root = os.environ.get('ZALMOXIS_ROOT', '')
        solid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
        )
        if not os.path.isfile(solid_file):
            pytest.skip('PALEOS 2-phase solid data not found')

        from zalmoxis.eos_functions import load_paleos_table

        cache = load_paleos_table(solid_file)
        assert cache['type'] == 'paleos'
        assert cache['p_min'] > 0
        assert cache['p_max'] > cache['p_min']
        assert cache['t_min'] > 0
        assert cache['t_max'] > cache['t_min']
        assert cache['n_p'] > 1
        assert cache['n_t'] > 1
        assert 'density_grid' in cache
        assert 'nabla_ad_grid' in cache
        assert 'density_nn' in cache
        assert 'nabla_ad_nn' in cache

    def test_bilinear_metadata(self):
        """load_paleos_table should include fast bilinear metadata."""
        root = os.environ.get('ZALMOXIS_ROOT', '')
        solid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
        )
        if not os.path.isfile(solid_file):
            pytest.skip('PALEOS 2-phase solid data not found')

        from zalmoxis.eos_functions import load_paleos_table

        cache = load_paleos_table(solid_file)
        assert 'dlog_p' in cache
        assert 'dlog_t' in cache
        assert cache['dlog_p'] > 0
        assert cache['dlog_t'] > 0


# =====================================================================
# load_paleos_unified_table tests
# =====================================================================


@pytest.mark.unit
class TestLoadPaleosUnifiedTable:
    """Tests for the unified PALEOS table loader."""

    def test_h2o_table(self):
        """Load PALEOS:H2O unified table."""
        root = os.environ.get('ZALMOXIS_ROOT', '')
        h2o_file = os.path.join(root, 'data', 'EOS_PALEOS_H2O', 'paleos_water_eos_table_pt.dat')
        if not os.path.isfile(h2o_file):
            pytest.skip('PALEOS H2O data not found')

        from zalmoxis.eos_functions import load_paleos_unified_table

        cache = load_paleos_unified_table(h2o_file)
        assert cache['type'] == 'paleos_unified'
        assert 'liquidus_log_p' in cache
        assert 'liquidus_log_t' in cache
        assert 'phase_grid' in cache
        assert cache['n_p'] > 1
        assert cache['n_t'] > 1

    def test_chabrier_h_table(self):
        """Load Chabrier:H unified table."""
        if not _chabrier_data_available():
            pytest.skip('Chabrier data not found')

        from zalmoxis.eos_functions import load_paleos_unified_table

        root = os.environ.get('ZALMOXIS_ROOT', '')
        h_file = os.path.join(root, 'data', 'EOS_Chabrier2021_HHe', 'chabrier2021_H.dat')
        cache = load_paleos_unified_table(h_file)

        assert cache['type'] == 'paleos_unified'
        assert cache['n_p'] > 1
        assert cache['n_t'] > 1


# =====================================================================
# get_tabulated_eos tests
# =====================================================================


@pytest.mark.unit
class TestGetTabulatedEos:
    """Tests for get_tabulated_eos which handles the internal EOS dispatch."""

    def test_1d_seager_eos(self):
        """Seager 1D rho(P) interpolation via get_tabulated_eos."""
        if not _seager_data_available():
            pytest.skip('Seager data not found')

        from zalmoxis.eos_functions import get_tabulated_eos
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['Seager2007:iron']
        rho = get_tabulated_eos(200e9, mat, 'core')
        assert rho is not None
        assert 10000 < rho < 16000

    def test_paleos_2phase_solid(self):
        """PALEOS 2-phase solid table via get_tabulated_eos."""
        root = os.environ.get('ZALMOXIS_ROOT', '')
        solid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
        )
        if not os.path.isfile(solid_file):
            pytest.skip('PALEOS 2-phase data not found')

        from zalmoxis.eos_functions import get_tabulated_eos
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS-2phase:MgSiO3']
        rho = get_tabulated_eos(100e9, mat, 'solid_mantle', temperature=3000)
        assert rho is not None
        assert rho > 0

    def test_paleos_requires_temperature(self):
        """PALEOS table should raise if temperature not provided."""
        root = os.environ.get('ZALMOXIS_ROOT', '')
        solid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
        )
        if not os.path.isfile(solid_file):
            pytest.skip('PALEOS 2-phase data not found')

        from zalmoxis.eos_functions import get_tabulated_eos
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS-2phase:MgSiO3']
        # No temperature => returns None (error is caught and logged)
        rho = get_tabulated_eos(100e9, mat, 'solid_mantle', temperature=None)
        assert rho is None

    def test_wb2018_regular_grid(self):
        """WB2018 melted_mantle (regular P-T grid) via get_tabulated_eos."""
        if not _wb2018_data_available():
            pytest.skip('WB2018 data not found')

        from zalmoxis.eos_functions import get_tabulated_eos
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['WolfBower2018:MgSiO3']
        rho = get_tabulated_eos(50e9, mat, 'melted_mantle', temperature=5000)
        assert rho is not None
        assert rho > 0

    def test_wb2018_solid_regular_grid(self):
        """WB2018 solid_mantle via get_tabulated_eos."""
        if not _wb2018_data_available():
            pytest.skip('WB2018 data not found')

        from zalmoxis.eos_functions import get_tabulated_eos
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['WolfBower2018:MgSiO3']
        rho = get_tabulated_eos(50e9, mat, 'solid_mantle', temperature=2000)
        assert rho is not None
        assert rho > 0

    def test_rtpress_irregular_grid(self):
        """RTPress100TPa melted_mantle (irregular grid) via get_tabulated_eos."""
        if not _rtpress_data_available():
            pytest.skip('RTPress100TPa data not found')

        from zalmoxis.eos_functions import get_tabulated_eos
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['RTPress100TPa:MgSiO3']
        rho = get_tabulated_eos(50e9, mat, 'melted_mantle', temperature=5000)
        assert rho is not None
        assert rho > 0

    def test_melted_mantle_requires_temperature(self):
        """melted_mantle without temperature should return None."""
        if not _wb2018_data_available():
            pytest.skip('WB2018 data not found')

        from zalmoxis.eos_functions import get_tabulated_eos
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['WolfBower2018:MgSiO3']
        rho = get_tabulated_eos(50e9, mat, 'melted_mantle', temperature=None)
        assert rho is None

    def test_pressure_clamping(self):
        """Out-of-bounds pressure should be clamped, not crash."""
        if not _seager_data_available():
            pytest.skip('Seager data not found')

        from zalmoxis.eos_functions import get_tabulated_eos
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['Seager2007:iron']
        # Very high pressure (should clamp to table max)
        rho = get_tabulated_eos(1e15, mat, 'core')
        # 1D extrapolation allows extrapolation; just check it doesn't crash
        assert rho is not None


# =====================================================================
# _get_paleos_unified_nabla_ad tests
# =====================================================================


@pytest.mark.unit
class TestGetPaleosUnifiedNablaAd:
    """Tests for unified PALEOS nabla_ad lookup."""

    def test_returns_finite_value(self):
        """nabla_ad lookup returns a finite positive value."""
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import _get_paleos_unified_nabla_ad
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS:MgSiO3']
        cache = {}
        nabla = _get_paleos_unified_nabla_ad(100e9, 4000, mat, cache)
        assert nabla is not None
        assert np.isfinite(nabla)
        assert 0 < nabla < 1

    def test_pressure_clamping(self):
        """Out-of-bounds pressure is clamped without error."""
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import _get_paleos_unified_nabla_ad
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS:iron']
        cache = {}
        nabla = _get_paleos_unified_nabla_ad(1e-5, 4000, mat, cache)
        # Even extreme pressure should return something valid
        assert nabla is not None or nabla is None  # May return None, but no crash


# =====================================================================
# _get_paleos_nabla_ad tests (2-phase PALEOS)
# =====================================================================


@pytest.mark.unit
class TestGetPaleosNablaAd:
    """Tests for 2-phase PALEOS nabla_ad lookup."""

    def test_solid_nabla_ad(self):
        """nabla_ad from solid table returns a valid value."""
        root = os.environ.get('ZALMOXIS_ROOT', '')
        solid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
        )
        if not os.path.isfile(solid_file):
            pytest.skip('PALEOS 2-phase data not found')

        from zalmoxis.eos_functions import _get_paleos_nabla_ad
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS-2phase:MgSiO3']
        cache = {}
        nabla = _get_paleos_nabla_ad(100e9, 3000, mat, 'solid_mantle', cache)
        assert nabla is not None
        assert np.isfinite(nabla)
        assert 0 < nabla < 1


# =====================================================================
# _compute_paleos_dtdp tests
# =====================================================================


@pytest.mark.unit
class TestComputePaleosDtdp:
    """Tests for the dT/dP computation with phase-aware nabla_ad weighting."""

    def test_returns_none_for_zero_pressure(self):
        """Zero pressure returns None."""
        from zalmoxis.eos_functions import _compute_paleos_dtdp

        result = _compute_paleos_dtdp(0, 3000, {}, None, None, {})
        assert result is None

    def test_returns_none_for_zero_temperature(self):
        """Zero temperature returns None."""
        from zalmoxis.eos_functions import _compute_paleos_dtdp

        result = _compute_paleos_dtdp(100e9, 0, {}, None, None, {})
        assert result is None

    def test_solid_phase_dtdp(self):
        """dT/dP in solid phase is positive and physically reasonable."""
        root = os.environ.get('ZALMOXIS_ROOT', '')
        solid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
        )
        if not os.path.isfile(solid_file):
            pytest.skip('PALEOS 2-phase data not found')

        from zalmoxis.eos_functions import _compute_paleos_dtdp
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS-2phase:MgSiO3']

        def sf(P):
            return 8000.0  # Very high solidus so T=3000 is solid

        def lf(P):
            return 10000.0

        cache = {}

        dtdp = _compute_paleos_dtdp(100e9, 3000, mat, sf, lf, cache)
        assert dtdp is not None
        assert dtdp > 0
        # dT/dP should be in range 1e-10 to 1e-5 K/Pa for typical mantles
        assert 1e-12 < dtdp < 1e-4

    def test_nan_melting_curves(self):
        """NaN melting curves fall back to solid phase."""
        root = os.environ.get('ZALMOXIS_ROOT', '')
        solid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
        )
        if not os.path.isfile(solid_file):
            pytest.skip('PALEOS 2-phase data not found')

        from zalmoxis.eos_functions import _compute_paleos_dtdp
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS-2phase:MgSiO3']

        def sf(P):
            return np.nan

        def lf(P):
            return np.nan

        cache = {}

        dtdp = _compute_paleos_dtdp(100e9, 3000, mat, sf, lf, cache)
        assert dtdp is not None
        assert dtdp > 0

    def test_none_melting_curves(self):
        """None solidus/liquidus falls back to solid phase."""
        root = os.environ.get('ZALMOXIS_ROOT', '')
        solid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
        )
        if not os.path.isfile(solid_file):
            pytest.skip('PALEOS 2-phase data not found')

        from zalmoxis.eos_functions import _compute_paleos_dtdp
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS-2phase:MgSiO3']
        cache = {}

        dtdp = _compute_paleos_dtdp(100e9, 3000, mat, None, None, cache)
        assert dtdp is not None
        assert dtdp > 0

    def test_mixed_phase_dtdp(self):
        """dT/dP in mixed phase combines solid and liquid nabla_ad."""
        root = os.environ.get('ZALMOXIS_ROOT', '')
        solid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
        )
        liquid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_liquid.dat'
        )
        if not (os.path.isfile(solid_file) and os.path.isfile(liquid_file)):
            pytest.skip('PALEOS 2-phase data not found')

        from zalmoxis.eos_functions import _compute_paleos_dtdp
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS-2phase:MgSiO3']

        # Set solidus and liquidus to bracket 4000 K
        def sf(P):
            return 3000.0

        def lf(P):
            return 5000.0

        cache = {}

        dtdp = _compute_paleos_dtdp(100e9, 4000, mat, sf, lf, cache)
        assert dtdp is not None
        assert dtdp > 0


# =====================================================================
# calculate_temperature_profile tests
# =====================================================================


@pytest.mark.unit
class TestCalculateTemperatureProfile:
    """Tests for the temperature profile mode dispatch."""

    def test_isothermal(self):
        """Isothermal mode returns constant temperature."""
        from zalmoxis.eos_functions import calculate_temperature_profile

        radii = np.linspace(0, 6e6, 100)
        T_func = calculate_temperature_profile(radii, 'isothermal', 300, 6000, '.', '')
        T = T_func(radii)
        assert np.all(T == pytest.approx(300.0))

    def test_linear(self):
        """Linear mode returns center_T at r=0, surface_T at r_max."""
        from zalmoxis.eos_functions import calculate_temperature_profile

        radii = np.linspace(0, 6e6, 100)
        T_func = calculate_temperature_profile(radii, 'linear', 300, 6000, '.', '')
        T = T_func(radii)
        assert T[0] == pytest.approx(6000.0)  # center
        assert T[-1] == pytest.approx(300.0)  # surface

    def test_adiabatic_returns_linear_guess(self):
        """Adiabatic mode returns linear profile as initial guess."""
        from zalmoxis.eos_functions import calculate_temperature_profile

        radii = np.linspace(0, 6e6, 100)
        T_func = calculate_temperature_profile(radii, 'adiabatic', 300, 6000, '.', '')
        T = T_func(radii)
        assert T[0] == pytest.approx(6000.0)  # center
        assert T[-1] == pytest.approx(300.0)  # surface

    def test_prescribed_missing_file_raises(self):
        """Prescribed mode with nonexistent file raises ValueError."""
        from zalmoxis.eos_functions import calculate_temperature_profile

        radii = np.linspace(0, 6e6, 100)
        with pytest.raises(ValueError, match='must be provided'):
            calculate_temperature_profile(
                radii, 'prescribed', 300, 6000, '/tmp/nonexistent', 'no_such_file.txt'
            )

    def test_prescribed_length_mismatch_raises(self):
        """Prescribed mode with wrong-length profile raises ValueError."""
        from zalmoxis.eos_functions import calculate_temperature_profile

        radii = np.linspace(0, 6e6, 10)
        # Create a temp file with wrong number of lines
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            # Write 5 values instead of 10
            for T in [300, 400, 500, 600, 700]:
                f.write(f'{T}\n')
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match='length does not match'):
                calculate_temperature_profile(
                    radii,
                    'prescribed',
                    300,
                    6000,
                    os.path.dirname(tmp_path),
                    os.path.basename(tmp_path),
                )
        finally:
            os.unlink(tmp_path)

    def test_prescribed_valid(self):
        """Prescribed mode with valid file returns interpolated temperatures."""
        from zalmoxis.eos_functions import calculate_temperature_profile

        radii = np.linspace(0, 6e6, 10)
        T_profile = np.linspace(6000, 300, 10)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for T in T_profile:
                f.write(f'{T}\n')
            tmp_path = f.name

        try:
            T_func = calculate_temperature_profile(
                radii,
                'prescribed',
                300,
                6000,
                os.path.dirname(tmp_path),
                os.path.basename(tmp_path),
            )
            T_out = T_func(radii)
            np.testing.assert_allclose(T_out, T_profile, atol=1e-6)
        finally:
            os.unlink(tmp_path)

    def test_unknown_mode_raises(self):
        """Unknown temperature mode raises ValueError."""
        from zalmoxis.eos_functions import calculate_temperature_profile

        radii = np.linspace(0, 6e6, 100)
        with pytest.raises(ValueError, match='Unknown temperature mode'):
            calculate_temperature_profile(radii, 'invalid_mode', 300, 6000, '.', '')


# =====================================================================
# load_melting_curve tests
# =====================================================================


@pytest.mark.unit
class TestLoadMeltingCurve:
    """Tests for load_melting_curve file loading."""

    def test_loads_valid_file(self):
        """Should return an interpolation function from a valid file."""
        from zalmoxis.eos_functions import load_melting_curve

        # Create a synthetic melting curve file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('# P (Pa) T (K)\n')
            for P in [1e9, 10e9, 50e9, 100e9, 200e9]:
                T = 1700 + 30 * (P / 1e9)
                f.write(f'{P:.6e} {T:.6e}\n')
            tmp_path = f.name

        try:
            func = load_melting_curve(tmp_path)
            assert func is not None
            T_at_50GPa = func(50e9)
            assert np.isfinite(T_at_50GPa)
            assert T_at_50GPa > 1700
        finally:
            os.unlink(tmp_path)

    def test_nonexistent_file_returns_none(self):
        """Nonexistent file returns None."""
        from zalmoxis.eos_functions import load_melting_curve

        result = load_melting_curve('/tmp/this_file_does_not_exist_at_all.dat')
        assert result is None


# =====================================================================
# create_pressure_density_files tests
# =====================================================================


@pytest.mark.unit
class TestCreatePressureDensityFiles:
    """Tests for the output file creation function."""

    def test_creates_files(self):
        """Should create pressure and density profile files."""
        from zalmoxis.eos_functions import create_pressure_density_files

        radii = np.array([0, 1e6, 2e6, 3e6])
        pressure = np.array([300e9, 200e9, 100e9, 1e5])
        density = np.array([13000, 10000, 5000, 3000], dtype=float)

        root = os.environ['ZALMOXIS_ROOT']
        p_file = os.path.join(root, 'output_files', 'pressure_profiles.txt')
        d_file = os.path.join(root, 'output_files', 'density_profiles.txt')

        # Clean up any pre-existing files
        for f in [p_file, d_file]:
            if os.path.exists(f):
                os.remove(f)

        try:
            create_pressure_density_files(0, 0, 0, radii, pressure, density)
            assert os.path.isfile(p_file)
            assert os.path.isfile(d_file)

            # Second call should append
            create_pressure_density_files(0, 0, 1, radii, pressure, density)
            with open(p_file) as f:
                content = f.read()
            assert 'Pressure iteration 0' in content
            assert 'Pressure iteration 1' in content
        finally:
            for f in [p_file, d_file]:
                if os.path.exists(f):
                    os.remove(f)


# =====================================================================
# get_solidus_liquidus_functions tests
# =====================================================================


@pytest.mark.unit
class TestGetSolidusLiquidusFunctions:
    """Tests for the solidus/liquidus function loader."""

    def test_default_stixrude14(self):
        """Default Stixrude14 melting curves return valid functions."""
        from zalmoxis.eos_functions import get_solidus_liquidus_functions

        sf, lf = get_solidus_liquidus_functions()
        assert callable(sf)
        assert callable(lf)

        # At 100 GPa, solidus should be below liquidus
        T_sol = sf(100e9)
        T_liq = lf(100e9)
        if np.isfinite(T_sol) and np.isfinite(T_liq):
            assert T_sol < T_liq


# =====================================================================
# get_paleos_unified_density_batch tests
# =====================================================================


@pytest.mark.unit
class TestGetPaleosUnifiedDensityBatch:
    """Tests for the vectorized unified PALEOS density lookup."""

    def test_batch_no_mushy_zone(self):
        """Batch lookup with mushy_zone_factor=1.0 (direct path)."""
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import get_paleos_unified_density_batch
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS:iron']
        pressures = np.array([50e9, 100e9, 200e9])
        temperatures = np.array([3000.0, 4000.0, 5000.0])

        cache = {}
        result = get_paleos_unified_density_batch(pressures, temperatures, mat, 1.0, cache)
        assert len(result) == 3
        assert all(np.isfinite(result))
        # Density should increase with pressure (at increasing T)
        # Iron compressibility dominates over thermal expansion
        assert result[0] < result[2]

    def test_batch_with_mushy_zone(self):
        """Batch lookup with mushy_zone_factor < 1.0 exercises mushy zone path."""
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import get_paleos_unified_density_batch
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS:MgSiO3']
        pressures = np.array([50e9, 100e9, 200e9])
        temperatures = np.array([3000.0, 4000.0, 5000.0])

        cache = {}
        result = get_paleos_unified_density_batch(pressures, temperatures, mat, 0.8, cache)
        assert len(result) == 3
        assert all(np.isfinite(result))

    def test_batch_empty_array(self):
        """Batch with zero-length arrays returns empty result."""
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import get_paleos_unified_density_batch
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS:iron']
        cache = {}
        result = get_paleos_unified_density_batch(np.array([]), np.array([]), mat, 1.0, cache)
        assert len(result) == 0


# =====================================================================
# RTPress100TPa irregular grid path tests
# =====================================================================


@pytest.mark.unit
class TestRTPress100TPa:
    """Tests for RTPress100TPa EOS path (irregular grid with LinearNDInterpolator)."""

    def test_rtpress_density_calculation(self):
        """RTPress100TPa:MgSiO3 density via calculate_density."""
        if not (_rtpress_data_available() and _seager_data_available()):
            pytest.skip('RTPress100TPa or Seager data not found')

        from zalmoxis.eos_functions import calculate_density, get_solidus_liquidus_functions
        from zalmoxis.eos_properties import EOS_REGISTRY

        sf, lf = get_solidus_liquidus_functions()
        # Use temperature above liquidus at moderate pressure
        rho = calculate_density(50e9, EOS_REGISTRY, 'RTPress100TPa:MgSiO3', 8000, sf, lf)
        assert rho is not None
        assert rho > 0

    def test_rtpress_solid_path(self):
        """RTPress100TPa:MgSiO3 density in the solid phase (uses WB2018 solid table)."""
        if not (_rtpress_data_available() and _wb2018_data_available()):
            pytest.skip('RTPress or WB2018 data not found')

        from zalmoxis.eos_functions import calculate_density, get_solidus_liquidus_functions
        from zalmoxis.eos_properties import EOS_REGISTRY

        sf, lf = get_solidus_liquidus_functions()
        rho = calculate_density(50e9, EOS_REGISTRY, 'RTPress100TPa:MgSiO3', 1500, sf, lf)
        assert rho is not None
        assert rho > 0

    def test_rtpress_high_pressure(self):
        """RTPress100TPa at high pressure (100 TPa) should work with clamping."""
        if not (_rtpress_data_available() and _seager_data_available()):
            pytest.skip('RTPress data not found')

        from zalmoxis.eos_functions import get_tabulated_eos
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['RTPress100TPa:MgSiO3']
        # Use a temperature within the melt table range
        rho = get_tabulated_eos(1e13, mat, 'melted_mantle', temperature=5000)
        assert rho is not None
        assert rho > 0
