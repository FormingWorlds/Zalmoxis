"""
Unit tests for the Seager et al. (2007) analytic modified polytropic EOS.

Tests all 6 materials from Table 3 for correctness, physical plausibility,
and consistency with the tabulated EOS data.

References:
    - Seager et al. (2007), ApJ 669:1279, Table 3, Eq. 11
    - docs/test_infrastructure.md
    - docs/test_categorization.md
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from zalmoxis.eos_analytic import (
    SEAGER2007_MATERIALS,
    VALID_MATERIAL_KEYS,
    get_analytic_density,
)

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')


ALL_MATERIALS = sorted(VALID_MATERIAL_KEYS)


class TestAnalyticDensityBasic:
    """Basic correctness tests for get_analytic_density()."""

    @pytest.mark.parametrize('material', ALL_MATERIALS)
    def test_zero_pressure_limit(self, material):
        """At very low pressure, density should approach rho_0."""
        rho_0 = SEAGER2007_MATERIALS[material][0]
        # Use a very small but positive pressure (1 Pa)
        density = get_analytic_density(1.0, material)
        assert density == pytest.approx(rho_0, rel=0.01), (
            f'At P~0, density for {material} should be ~rho_0={rho_0}'
        )

    @pytest.mark.parametrize('material', ALL_MATERIALS)
    def test_monotonicity(self, material):
        """Density must increase monotonically with pressure."""
        pressures = np.logspace(6, 15, 50)  # 1 MPa to 1 PPa
        densities = [get_analytic_density(p, material) for p in pressures]
        for i in range(1, len(densities)):
            assert densities[i] > densities[i - 1], (
                f'Density not monotonically increasing for {material} at P={pressures[i]:.2e}'
            )

    def test_iron_at_earth_center(self):
        """Iron at 365 GPa (Earth center pressure) should give ~13,000 kg/m^3."""
        # Earth center density is ~13,000 kg/m^3
        density = get_analytic_density(365e9, 'iron')
        assert 11000 < density < 15000, (
            f'Iron density at 365 GPa = {density:.0f} kg/m^3, expected ~13,000 (Earth center)'
        )

    def test_cross_material_ordering_low_pressure(self):
        """At low pressure, densities should follow the rho_0 ordering."""
        # At P = 1 GPa, ordering should roughly follow rho_0
        p = 1e9
        rho_iron = get_analytic_density(p, 'iron')
        rho_MgFeSiO3 = get_analytic_density(p, 'MgFeSiO3')
        rho_MgSiO3 = get_analytic_density(p, 'MgSiO3')
        rho_SiC = get_analytic_density(p, 'SiC')
        rho_graphite = get_analytic_density(p, 'graphite')
        rho_H2O = get_analytic_density(p, 'H2O')

        assert rho_iron > rho_MgFeSiO3 > rho_MgSiO3
        assert rho_SiC > rho_graphite
        assert rho_graphite > rho_H2O


class TestAnalyticDensityEdgeCases:
    """Edge case and error handling tests."""

    def test_invalid_material_raises(self):
        """Unknown material key should raise ValueError."""
        with pytest.raises(ValueError, match='Unknown material key'):
            get_analytic_density(1e9, 'unobtainium')

    def test_negative_pressure_returns_rho0(self):
        """Negative pressure should return rho_0 (zero-pressure density)."""
        result = get_analytic_density(-1e9, 'iron')
        assert result == pytest.approx(8300.0)

    def test_zero_pressure_returns_rho0(self):
        """Zero pressure should return rho_0 (zero-pressure density)."""
        result = get_analytic_density(0.0, 'iron')
        assert result == pytest.approx(8300.0)

    def test_nan_pressure_returns_none(self):
        """NaN pressure should return None."""
        result = get_analytic_density(float('nan'), 'iron')
        assert result is None

    @pytest.mark.parametrize('material', ALL_MATERIALS)
    def test_high_pressure_warning(self, material, caplog):
        """Pressure above P_MAX should log a warning but still return a value."""
        import logging

        with caplog.at_level(logging.WARNING):
            density = get_analytic_density(2e16, material)
        assert density is not None
        assert 'exceeds validity limit' in caplog.text


class TestAnalyticVsTabulated:
    """Compare analytic EOS against tabulated data for iron, silicate, water."""

    @pytest.mark.parametrize(
        'material_key,eos_file,col_density,col_pressure',
        [
            ('iron', 'eos_seager07_iron.txt', 0, 1),
            ('MgSiO3', 'eos_seager07_silicate.txt', 0, 1),
            ('H2O', 'eos_seager07_water.txt', 0, 1),
        ],
    )
    def test_analytic_vs_tabulated(self, material_key, eos_file, col_density, col_pressure):
        """Analytic EOS should agree with tabulated data within ~15%."""
        data_path = os.path.join(ZALMOXIS_ROOT, 'data', 'EOS_Seager2007', eos_file)

        data = np.loadtxt(data_path, delimiter=',', skiprows=1)
        densities_tab = data[:, col_density] * 1e3  # g/cm^3 -> kg/m^3
        pressures_tab = data[:, col_pressure] * 1e9  # GPa -> Pa

        # Compare at pressures from 1 GPa to 100 TPa
        mask = (pressures_tab >= 1e9) & (pressures_tab <= 1e14)
        pressures_test = pressures_tab[mask]
        densities_test = densities_tab[mask]

        for p, rho_tab in zip(pressures_test, densities_test):
            rho_analytic = get_analytic_density(p, material_key)
            rel_diff = abs(rho_analytic - rho_tab) / rho_tab
            assert rel_diff < 0.15, (
                f'Analytic vs tabulated {material_key} at P={p:.2e} Pa: '
                f'rho_analytic={rho_analytic:.0f}, rho_tab={rho_tab:.0f}, '
                f'rel_diff={rel_diff:.2%}'
            )
