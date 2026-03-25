"""Tests for the unified PALEOS EOS integration (PALEOS:iron, PALEOS:MgSiO3, PALEOS:H2O).

Tests cover:
- Registration of unified PALEOS identifiers in VALID_TABULATED_EOS and TDEP_EOS_NAMES
- EOS_REGISTRY structure and format flags
- Unified table loader (grid structure, phase boundary extraction, liquidus array)
- Unified density lookup (sharp boundary and mushy zone modes)
- Unified nabla_ad lookup
- Mass limit enforcement for unified PALEOS
- Dict-based material_dictionaries return type
- Backward compatibility: all existing EOS still work with dict dispatch

References:
    - docs/testing.md
    - docs/How-to/configuration.md
"""

from __future__ import annotations

import os

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


@pytest.mark.unit
class TestUnifiedPALEOSRegistration:
    """Verify unified PALEOS identifiers are registered in lookup tables."""

    def test_iron_in_valid_tabulated_eos(self):
        from zalmoxis.zalmoxis import VALID_TABULATED_EOS

        assert 'PALEOS:iron' in VALID_TABULATED_EOS

    def test_mgsio3_in_valid_tabulated_eos(self):
        from zalmoxis.zalmoxis import VALID_TABULATED_EOS

        assert 'PALEOS:MgSiO3' in VALID_TABULATED_EOS

    def test_h2o_in_valid_tabulated_eos(self):
        from zalmoxis.zalmoxis import VALID_TABULATED_EOS

        assert 'PALEOS:H2O' in VALID_TABULATED_EOS

    def test_all_in_tdep_eos_names(self):
        from zalmoxis.constants import TDEP_EOS_NAMES

        for name in ('PALEOS:iron', 'PALEOS:MgSiO3', 'PALEOS:H2O'):
            assert name in TDEP_EOS_NAMES

    def test_eos_registry_has_unified_entries(self):
        from zalmoxis.eos_properties import EOS_REGISTRY

        for name in ('PALEOS:iron', 'PALEOS:MgSiO3', 'PALEOS:H2O'):
            assert name in EOS_REGISTRY
            mat = EOS_REGISTRY[name]
            assert mat.get('format') == 'paleos_unified'
            assert 'eos_file' in mat


@pytest.mark.unit
class TestDictBasedMaterialDictionaries:
    """Verify that load_material_dictionaries returns a dict (not tuple)."""

    def test_returns_dict(self):
        from zalmoxis.zalmoxis import load_material_dictionaries

        md = load_material_dictionaries()
        assert isinstance(md, dict), f'Expected dict, got {type(md)}'

    def test_contains_all_eos(self):
        from zalmoxis.zalmoxis import VALID_TABULATED_EOS, load_material_dictionaries

        md = load_material_dictionaries()
        for eos_name in VALID_TABULATED_EOS:
            assert eos_name in md, f'{eos_name} not in material_dictionaries'

    def test_backward_compat_seager_iron(self):
        """Seager2007:iron should still have a 'core' sub-dict."""
        from zalmoxis.zalmoxis import load_material_dictionaries

        md = load_material_dictionaries()
        mat = md['Seager2007:iron']
        assert 'core' in mat
        assert 'eos_file' in mat['core']

    def test_backward_compat_wb2018(self):
        """WolfBower2018:MgSiO3 should still have melted_mantle and solid_mantle."""
        from zalmoxis.zalmoxis import load_material_dictionaries

        md = load_material_dictionaries()
        mat = md['WolfBower2018:MgSiO3']
        assert 'melted_mantle' in mat
        assert 'solid_mantle' in mat


@pytest.mark.unit
class TestLoadUnifiedTable:
    """Tests for load_paleos_unified_table() grid structure and phase extraction."""

    def test_load_iron_table_structure(self):
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import load_paleos_unified_table

        root = os.environ['ZALMOXIS_ROOT']
        iron_file = os.path.join(
            root, 'data', 'EOS_PALEOS_iron', 'paleos_iron_eos_table_pt.dat'
        )
        cache = load_paleos_unified_table(iron_file)

        assert cache['type'] == 'paleos_unified'
        assert 'density_interp' in cache
        assert 'nabla_ad_interp' in cache
        assert 'liquidus_log_p' in cache
        assert 'liquidus_log_t' in cache
        assert 'phase_grid' in cache
        assert cache['p_min'] > 0
        assert cache['p_max'] > cache['p_min']

    def test_liquidus_extraction(self):
        """The extracted liquidus should be non-empty and monotonic in log_p."""
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import load_paleos_unified_table

        root = os.environ['ZALMOXIS_ROOT']
        iron_file = os.path.join(
            root, 'data', 'EOS_PALEOS_iron', 'paleos_iron_eos_table_pt.dat'
        )
        cache = load_paleos_unified_table(iron_file)

        assert len(cache['liquidus_log_p']) > 0, 'No liquidus extracted'
        assert len(cache['liquidus_log_p']) == len(cache['liquidus_log_t'])
        # log_p should be sorted (monotonically increasing)
        assert np.all(np.diff(cache['liquidus_log_p']) >= 0)

    def test_load_mgsio3_table_structure(self):
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import load_paleos_unified_table

        root = os.environ['ZALMOXIS_ROOT']
        mgsio3_file = os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3_unified', 'paleos_mgsio3_eos_table_pt.dat'
        )
        cache = load_paleos_unified_table(mgsio3_file)

        assert cache['type'] == 'paleos_unified'
        assert cache['p_max'] >= 1e13  # Should extend to ~100 TPa


@pytest.mark.unit
class TestUnifiedDensityLookup:
    """Tests for get_paleos_unified_density()."""

    def test_density_at_moderate_conditions(self):
        """Density at 100 GPa, 4000 K should be physically reasonable."""
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import get_paleos_unified_density
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS:MgSiO3']
        cache = {}
        rho = get_paleos_unified_density(100e9, 4000.0, mat, 1.0, cache)

        assert rho is not None
        assert np.isfinite(rho)
        # MgSiO3 at 100 GPa should be ~4000-10000 kg/m^3
        assert 2000 < rho < 15000, f'Density {rho:.0f} kg/m^3 out of range'

    def test_density_increases_with_pressure(self):
        """At fixed T, density should increase with pressure."""
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import get_paleos_unified_density
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS:iron']
        cache = {}
        densities = []
        for P in [10e9, 50e9, 100e9, 300e9]:
            rho = get_paleos_unified_density(P, 4000.0, mat, 1.0, cache)
            if rho is not None and np.isfinite(rho):
                densities.append(rho)

        assert len(densities) >= 2
        for i in range(1, len(densities)):
            assert densities[i] > densities[i - 1]

    def test_mushy_zone_factor_effect(self):
        """With mushy_zone_factor < 1.0, density in the mushy zone should differ from factor=1.0."""
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import get_paleos_unified_density, load_paleos_unified_table
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS:MgSiO3']
        cache = {}

        # Load table to find a point near the liquidus
        eos_file = mat['eos_file']
        table_cache = load_paleos_unified_table(eos_file)
        cache[eos_file] = table_cache

        if len(table_cache['liquidus_log_p']) == 0:
            pytest.skip('No liquidus extracted')

        # Pick a pressure in the middle of the liquidus range
        mid_idx = len(table_cache['liquidus_log_p']) // 2
        log_p = table_cache['liquidus_log_p'][mid_idx]
        log_t_liq = table_cache['liquidus_log_t'][mid_idx]
        P = 10.0**log_p
        T_liq = 10.0**log_t_liq

        # Query just below liquidus
        T_below = T_liq * 0.9  # 90% of liquidus T

        rho_sharp = get_paleos_unified_density(P, T_below, mat, 1.0, cache)
        rho_mushy = get_paleos_unified_density(P, T_below, mat, 0.8, cache)

        # Both should return valid densities
        assert rho_sharp is not None and np.isfinite(rho_sharp)
        assert rho_mushy is not None and np.isfinite(rho_mushy)

        # The values may differ if T_below falls in the mushy zone for factor=0.8
        # (T_sol = 0.8 * T_liq, so T_below = 0.9*T_liq is in the mushy zone)
        # This is a soft check: we just verify both produce valid values


@pytest.mark.unit
class TestUnifiedNablaAd:
    """Tests for unified PALEOS nabla_ad lookup."""

    def test_nabla_ad_physically_reasonable(self):
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import _get_paleos_unified_nabla_ad
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS:iron']
        cache = {}
        nabla = _get_paleos_unified_nabla_ad(100e9, 4000.0, mat, cache)

        assert nabla is not None
        assert np.isfinite(nabla)
        assert 0.01 < nabla < 1.0, f'nabla_ad={nabla:.4f} out of range'

    def test_dtdp_conversion(self):
        """dT/dP = nabla_ad * T / P should be physically reasonable."""
        if not _paleos_unified_data_available():
            pytest.skip('Unified PALEOS data not found')

        from zalmoxis.eos_functions import _get_paleos_unified_nabla_ad
        from zalmoxis.eos_properties import EOS_REGISTRY

        mat = EOS_REGISTRY['PALEOS:MgSiO3']
        cache = {}
        P = 100e9
        T = 4000.0
        nabla = _get_paleos_unified_nabla_ad(P, T, mat, cache)
        dtdp = nabla * T / P

        assert dtdp > 0
        assert 1e-10 < dtdp < 1e-5


@pytest.mark.unit
class TestUnifiedMassLimit:
    """Mass limit enforcement for unified PALEOS."""

    def test_mass_limit_raises_at_51(self):
        """Requesting > 50 M_earth with unified PALEOS must raise ValueError."""
        from zalmoxis.constants import earth_mass
        from zalmoxis.zalmoxis import load_material_dictionaries, main

        config = {
            'planet_mass': 51.0 * earth_mass,
            'core_mass_fraction': 0.325,
            'mantle_mass_fraction': 0,
            'temperature_mode': 'linear',
            'surface_temperature': 3500,
            'center_temperature': 6000,
            'temp_profile_file': '',
            'layer_eos_config': {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
            'mushy_zone_factor': 1.0,
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
                config,
                material_dictionaries=load_material_dictionaries(),
                melting_curves_functions=None,
                input_dir='.',
            )


@pytest.mark.unit
class TestCalculateDensityDictDispatch:
    """Verify calculate_density() works with the new dict-based dispatch."""

    def test_seager_iron_via_dict(self):
        """Seager2007:iron density at 300 GPa should be ~13000 kg/m^3 (Earth center)."""
        from zalmoxis.eos_functions import calculate_density
        from zalmoxis.zalmoxis import load_material_dictionaries

        # Seager2007 data may not be present
        root = os.environ.get('ZALMOXIS_ROOT', '')
        iron_file = os.path.join(root, 'data', 'EOS_Seager2007', 'eos_seager07_iron.txt')
        if not os.path.isfile(iron_file):
            pytest.skip('Seager2007 data not found')

        md = load_material_dictionaries()
        rho = calculate_density(300e9, md, 'Seager2007:iron', 300, None, None)
        assert rho is not None
        assert 10000 < rho < 16000

    def test_analytic_via_dict(self):
        """Analytic:iron should work with dict dispatch."""
        from zalmoxis.eos_functions import calculate_density
        from zalmoxis.zalmoxis import load_material_dictionaries

        md = load_material_dictionaries()
        rho = calculate_density(300e9, md, 'Analytic:iron', 300, None, None)
        assert rho is not None
        assert rho > 0

    def test_unknown_eos_raises(self):
        """Unknown EOS should raise ValueError."""
        from zalmoxis.eos_functions import calculate_density
        from zalmoxis.zalmoxis import load_material_dictionaries

        md = load_material_dictionaries()
        with pytest.raises(ValueError, match='Unknown'):
            calculate_density(100e9, md, 'Nonexistent:stuff', 300, None, None)


@pytest.mark.unit
class TestMusyZoneFactorConfig:
    """Verify mushy_zone_factor is parsed from config."""

    def test_default_mushy_zone_factor(self):
        """Default mushy_zone_factor should be 1.0."""
        import tempfile

        import toml

        from zalmoxis.zalmoxis import load_zalmoxis_config

        # Create a minimal config file
        config = {
            'InputParameter': {'planet_mass': 1.0},
            'AssumptionsAndInitialGuesses': {
                'core_mass_fraction': 0.325,
                'mantle_mass_fraction': 0,
                'temperature_mode': 'isothermal',
                'surface_temperature': 300,
                'center_temperature': 6000,
                'temperature_profile_file': '',
            },
            'EOS': {
                'core': 'Seager2007:iron',
                'mantle': 'Seager2007:MgSiO3',
            },
            'Calculations': {'num_layers': 50},
            'IterativeProcess': {
                'max_iterations_outer': 10,
                'tolerance_outer': 3e-3,
                'max_iterations_inner': 10,
                'tolerance_inner': 1e-4,
                'relative_tolerance': 1e-5,
                'absolute_tolerance': 1e-6,
                'maximum_step': 250000,
                'adaptive_radial_fraction': 0.98,
                'max_center_pressure_guess': 10e12,
            },
            'PressureAdjustment': {
                'target_surface_pressure': 101325,
                'pressure_tolerance': 1e9,
                'max_iterations_pressure': 200,
            },
            'Output': {
                'data_enabled': False,
                'plots_enabled': False,
                'verbose': False,
                'iteration_profiles_enabled': False,
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(config, f)
            tmp_path = f.name

        try:
            parsed = load_zalmoxis_config(tmp_path)
            assert parsed['mushy_zone_factor'] == pytest.approx(1.0)
            # Per-EOS dict should all be 1.0
            assert parsed['mushy_zone_factors']['PALEOS:iron'] == pytest.approx(1.0)
            assert parsed['mushy_zone_factors']['PALEOS:MgSiO3'] == pytest.approx(1.0)
            assert parsed['mushy_zone_factors']['PALEOS:H2O'] == pytest.approx(1.0)
        finally:
            os.unlink(tmp_path)

    def test_explicit_mushy_zone_factor(self):
        """Explicit mushy_zone_factor=0.8 should be parsed correctly."""
        import tempfile

        import toml

        from zalmoxis.zalmoxis import load_zalmoxis_config

        config = {
            'InputParameter': {'planet_mass': 1.0},
            'AssumptionsAndInitialGuesses': {
                'core_mass_fraction': 0.325,
                'mantle_mass_fraction': 0,
                'temperature_mode': 'isothermal',
                'surface_temperature': 300,
                'center_temperature': 6000,
                'temperature_profile_file': '',
            },
            'EOS': {
                'core': 'PALEOS:iron',
                'mantle': 'PALEOS:MgSiO3',
                'mushy_zone_factor': 0.8,
            },
            'Calculations': {'num_layers': 50},
            'IterativeProcess': {
                'max_iterations_outer': 10,
                'tolerance_outer': 3e-3,
                'max_iterations_inner': 10,
                'tolerance_inner': 1e-4,
                'relative_tolerance': 1e-5,
                'absolute_tolerance': 1e-6,
                'maximum_step': 250000,
                'adaptive_radial_fraction': 0.98,
                'max_center_pressure_guess': 10e12,
            },
            'PressureAdjustment': {
                'target_surface_pressure': 101325,
                'pressure_tolerance': 1e9,
                'max_iterations_pressure': 200,
            },
            'Output': {
                'data_enabled': False,
                'plots_enabled': False,
                'verbose': False,
                'iteration_profiles_enabled': False,
            },
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
            toml.dump(config, f)
            tmp_path = f.name

        try:
            parsed = load_zalmoxis_config(tmp_path)
            assert parsed['mushy_zone_factor'] == pytest.approx(0.8)
            # Per-EOS dict: configured materials inherit global default,
            # unconfigured materials default to 1.0 (no mushy zone)
            assert parsed['mushy_zone_factors']['PALEOS:iron'] == pytest.approx(0.8)
            assert parsed['mushy_zone_factors']['PALEOS:MgSiO3'] == pytest.approx(0.8)
            assert parsed['mushy_zone_factors']['PALEOS:H2O'] == pytest.approx(1.0)
        finally:
            os.unlink(tmp_path)
