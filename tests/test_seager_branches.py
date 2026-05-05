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

from zalmoxis.eos.seager import get_tabulated_eos

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
