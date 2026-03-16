"""Tests for the PALEOS-2phase:MgSiO3 EOS integration.

Tests cover:
- PALEOS table loading (grid reconstruction, interpolator creation)
- Density interpolation via PALEOS tables
- nabla_ad interpolation and dT/dP conversion
- Registration in VALID_TABULATED_EOS and TDEP_EOS_NAMES
- Mass limit enforcement (>50 M_earth raises ValueError)

References:
    - docs/testing.md
"""

from __future__ import annotations

import os

import numpy as np
import pytest


def _paleos_data_available():
    """Check if PALEOS data files are available."""
    root = os.environ.get('ZALMOXIS_ROOT', '')
    solid = os.path.join(
        root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
    )
    liquid = os.path.join(
        root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_liquid.dat'
    )
    return os.path.isfile(solid) and os.path.isfile(liquid)


@pytest.mark.unit
class TestPALEOSRegistration:
    """Verify PALEOS-2phase:MgSiO3 is registered in the EOS lookup tables."""

    def test_in_valid_tabulated_eos(self):
        """PALEOS-2phase:MgSiO3 should be in VALID_TABULATED_EOS."""
        from zalmoxis.zalmoxis import VALID_TABULATED_EOS

        assert 'PALEOS-2phase:MgSiO3' in VALID_TABULATED_EOS

    def test_in_tdep_eos_names(self):
        """PALEOS-2phase:MgSiO3 should be in TDEP_EOS_NAMES."""
        from zalmoxis.constants import TDEP_EOS_NAMES

        assert 'PALEOS-2phase:MgSiO3' in TDEP_EOS_NAMES


@pytest.mark.unit
class TestPALEOSMassLimit:
    """Mass limit enforcement for PALEOS-2phase:MgSiO3."""

    def test_mass_limit_raises(self):
        """Requesting > 50 M_earth with PALEOS-2phase:MgSiO3 must raise ValueError."""
        from zalmoxis.constants import earth_mass
        from zalmoxis.zalmoxis import (
            load_material_dictionaries,
            load_solidus_liquidus_functions,
            main,
        )

        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'PALEOS-2phase:MgSiO3'}
        config_params = {
            'planet_mass': 51.0 * earth_mass,
            'core_mass_fraction': 0.325,
            'mantle_mass_fraction': 0,
            'temperature_mode': 'linear',
            'surface_temperature': 3500,
            'center_temperature': 6000,
            'temp_profile_file': '',
            'layer_eos_config': layer_eos_config,
            'num_layers': 50,
            'max_iterations_outer': 10,
            'tolerance_outer': 3e-3,
            'max_iterations_inner': 10,
            'tolerance_inner': 1e-4,
            'relative_tolerance': 1e-5,
            'absolute_tolerance': 1e-6,
            'maximum_step': 250000,
            'adaptive_radial_fraction': 0.98,
            'max_center_pressure_guess': 10e12,
            'target_surface_pressure': 101325,
            'pressure_tolerance': 1e9,
            'max_iterations_pressure': 50,
            'data_output_enabled': False,
            'plotting_enabled': False,
            'verbose': False,
            'iteration_profiles_enabled': False,
        }
        with pytest.raises(ValueError, match='PALEOS'):
            main(
                config_params,
                material_dictionaries=load_material_dictionaries(),
                melting_curves_functions=load_solidus_liquidus_functions(layer_eos_config),
                input_dir='.',
            )

    def test_mass_limit_passes_at_50(self):
        """50 M_earth should not raise a mass limit error.

        We only test that the mass limit check passes, not full convergence.
        """
        from zalmoxis.zalmoxis import PALEOS_MAX_MASS_EARTH

        assert PALEOS_MAX_MASS_EARTH >= 50.0


@pytest.mark.unit
class TestLoadPALEOSTable:
    """Tests for load_paleos_table() grid reconstruction and interpolator creation."""

    def test_load_solid_table_structure(self):
        """PALEOS solid table should produce a cache entry with expected keys."""
        if not _paleos_data_available():
            pytest.skip('PALEOS data files not found')

        from zalmoxis.eos_functions import load_paleos_table

        root = os.environ['ZALMOXIS_ROOT']
        solid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
        )
        cache = load_paleos_table(solid_file)

        assert cache['type'] == 'paleos'
        assert 'density_interp' in cache
        assert 'nabla_ad_interp' in cache
        assert cache['p_min'] > 0
        assert cache['p_max'] > cache['p_min']
        assert cache['t_min'] > 0
        assert cache['t_max'] > cache['t_min']

    def test_load_liquid_table_structure(self):
        """PALEOS liquid table should produce a cache entry with expected keys."""
        if not _paleos_data_available():
            pytest.skip('PALEOS data files not found')

        from zalmoxis.eos_functions import load_paleos_table

        root = os.environ['ZALMOXIS_ROOT']
        liquid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_liquid.dat'
        )
        cache = load_paleos_table(liquid_file)

        assert cache['type'] == 'paleos'
        assert cache['p_max'] >= 1e13  # Should extend to ~100 TPa


@pytest.mark.unit
class TestPALEOSDensityInterpolation:
    """Tests for density interpolation via PALEOS tables."""

    def test_density_physically_reasonable(self):
        """Density at moderate P,T should be in a physical range for MgSiO3."""
        if not _paleos_data_available():
            pytest.skip('PALEOS data files not found')

        from zalmoxis.eos_functions import load_paleos_table

        root = os.environ['ZALMOXIS_ROOT']
        liquid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_liquid.dat'
        )
        cache = load_paleos_table(liquid_file)

        # Test at 100 GPa, 4000 K (deep mantle conditions)
        P = 100e9
        T = 4000.0
        rho = float(cache['density_interp']((np.log10(P), np.log10(T))))

        assert np.isfinite(rho), f'Density is not finite: {rho}'
        # MgSiO3 melt at 100 GPa should be ~5000-8000 kg/m3
        assert 3000 < rho < 12000, f'Density {rho:.0f} kg/m3 out of physical range'

    def test_density_increases_with_pressure(self):
        """At fixed T, density should increase with increasing pressure."""
        if not _paleos_data_available():
            pytest.skip('PALEOS data files not found')

        from zalmoxis.eos_functions import load_paleos_table

        root = os.environ['ZALMOXIS_ROOT']
        solid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
        )
        cache = load_paleos_table(solid_file)

        T = 2000.0
        pressures = [10e9, 50e9, 100e9, 200e9]
        densities = []
        for P in pressures:
            rho = float(cache['density_interp']((np.log10(P), np.log10(T))))
            if np.isfinite(rho):
                densities.append(rho)

        assert len(densities) >= 2, 'Not enough valid density values to compare'
        # Density should be monotonically increasing with pressure
        for i in range(1, len(densities)):
            assert densities[i] > densities[i - 1], (
                f'Density not increasing: rho({pressures[i]:.0e}) = {densities[i]:.0f} '
                f'<= rho({pressures[i - 1]:.0e}) = {densities[i - 1]:.0f}'
            )


@pytest.mark.unit
class TestPALEOSNablaAdInterpolation:
    """Tests for nabla_ad interpolation and dT/dP conversion."""

    def test_nabla_ad_physically_reasonable(self):
        """nabla_ad at moderate conditions should be in a physical range."""
        if not _paleos_data_available():
            pytest.skip('PALEOS data files not found')

        from zalmoxis.eos_functions import load_paleos_table

        root = os.environ['ZALMOXIS_ROOT']
        liquid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_liquid.dat'
        )
        cache = load_paleos_table(liquid_file)

        # Test at 100 GPa, 4000 K
        P = 100e9
        T = 4000.0
        nabla = float(cache['nabla_ad_interp']((np.log10(P), np.log10(T))))

        assert np.isfinite(nabla), f'nabla_ad is not finite: {nabla}'
        # nabla_ad for silicates is typically 0.1-0.4
        assert 0.01 < nabla < 1.0, f'nabla_ad {nabla:.4f} out of expected range [0.01, 1.0]'

    def test_nabla_ad_to_dtdp_conversion(self):
        """dT/dP = nabla_ad * T/P should produce physically reasonable values."""
        if not _paleos_data_available():
            pytest.skip('PALEOS data files not found')

        from zalmoxis.eos_functions import load_paleos_table

        root = os.environ['ZALMOXIS_ROOT']
        liquid_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_liquid.dat'
        )
        cache = load_paleos_table(liquid_file)

        P = 100e9  # 100 GPa
        T = 4000.0  # K
        nabla = float(cache['nabla_ad_interp']((np.log10(P), np.log10(T))))

        dtdp = nabla * T / P  # K/Pa

        assert dtdp > 0, f'dT/dP should be positive, got {dtdp}'
        # At 100 GPa, 4000 K, dT/dP ~ 1e-8 to 1e-7 K/Pa
        assert 1e-10 < dtdp < 1e-5, f'dT/dP = {dtdp:.2e} K/Pa out of expected range'
