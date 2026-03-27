"""Tests for configuration parameter validation.

Tests cover:
- Planet mass validation (positive)
- Mass fraction validation (ranges, cross-constraints)
- 3-layer model constraints (ice_layer requires mantle_mass_fraction > 0)
- Temperature parameter validation (positive, mode checks)
- Adiabatic mode requires T-dependent EOS
- Mushy zone factor validation (range, EOS compatibility)
- Numerical parameter validation (num_layers, tolerances)
- Pressure solver parameter validation
- EOS-specific cross-checks (melting curves)

References:
    - docs/How-to/configuration.md
"""

from __future__ import annotations

import pytest


def _make_config(**overrides):
    """Build a minimal valid config dict, with overrides."""
    from zalmoxis.constants import earth_mass

    config = {
        'planet_mass': 1.0 * earth_mass,
        'core_mass_fraction': 0.325,
        'mantle_mass_fraction': 0,
        'temperature_mode': 'adiabatic',
        'surface_temperature': 3000,
        'center_temperature': 6000,
        'temp_profile_file': '',
        'layer_eos_config': {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
        'rock_solidus': 'Monteux16-solidus',
        'rock_liquidus': 'Monteux16-liquidus-A-chondritic',
        'mushy_zone_factor': 1.0,
        'num_layers': 150,
        'max_iterations_outer': 100,
        'tolerance_outer': 3e-3,
        'max_iterations_inner': 100,
        'tolerance_inner': 1e-4,
        'relative_tolerance': 1e-5,
        'absolute_tolerance': 1e-6,
        'maximum_step': 250000,
        'adaptive_radial_fraction': 0.98,
        'max_center_pressure_guess': 10e12,
        'target_surface_pressure': 101325,
        'pressure_tolerance': 1e9,
        'max_iterations_pressure': 200,
        'data_output_enabled': False,
        'plotting_enabled': False,
        'verbose': False,
        'iteration_profiles_enabled': False,
    }
    config.update(overrides)
    return config


@pytest.mark.unit
class TestPlanetMassValidation:
    def test_negative_mass_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='planet_mass must be positive'):
            validate_config(_make_config(planet_mass=-1e24))

    def test_zero_mass_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='planet_mass must be positive'):
            validate_config(_make_config(planet_mass=0))


@pytest.mark.unit
class TestMassFractionValidation:
    def test_cmf_zero_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='core_mass_fraction must be in'):
            validate_config(_make_config(core_mass_fraction=0))

    def test_cmf_negative_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='core_mass_fraction must be in'):
            validate_config(_make_config(core_mass_fraction=-0.1))

    def test_cmf_over_one_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='core_mass_fraction must be in'):
            validate_config(_make_config(core_mass_fraction=1.5))

    def test_mmf_negative_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='mantle_mass_fraction must be in'):
            validate_config(_make_config(mantle_mass_fraction=-0.1))

    def test_sum_exceeds_one_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='cannot exceed 1'):
            validate_config(
                _make_config(
                    core_mass_fraction=0.6,
                    mantle_mass_fraction=0.5,
                    layer_eos_config={
                        'core': 'PALEOS:iron',
                        'mantle': 'PALEOS:MgSiO3',
                        'ice_layer': 'PALEOS:H2O',
                    },
                )
            )


@pytest.mark.unit
class TestThreeLayerValidation:
    def test_ice_layer_without_mmf_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='mantle_mass_fraction > 0'):
            validate_config(
                _make_config(
                    mantle_mass_fraction=0,
                    layer_eos_config={
                        'core': 'PALEOS:iron',
                        'mantle': 'PALEOS:MgSiO3',
                        'ice_layer': 'PALEOS:H2O',
                    },
                )
            )

    def test_mmf_without_ice_layer_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='no ice_layer EOS is set'):
            validate_config(_make_config(mantle_mass_fraction=0.5))


@pytest.mark.unit
class TestTemperatureValidation:
    def test_invalid_mode_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='Unknown temperature_mode'):
            validate_config(_make_config(temperature_mode='invalid'))

    def test_negative_surface_temp_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='surface_temperature must be positive'):
            validate_config(_make_config(surface_temperature=-100))

    def test_adiabatic_without_tdep_eos_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='T-dependent EOS'):
            validate_config(
                _make_config(
                    temperature_mode='adiabatic',
                    layer_eos_config={
                        'core': 'Seager2007:iron',
                        'mantle': 'Seager2007:MgSiO3',
                    },
                )
            )


@pytest.mark.unit
class TestMushyZoneValidation:
    def test_factor_above_one_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='mushy_zone_factor must be in'):
            validate_config(_make_config(mushy_zone_factor=1.5))

    def test_factor_negative_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='mushy_zone_factor must be in'):
            validate_config(_make_config(mushy_zone_factor=-0.1))

    def test_factor_below_minimum_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='below the minimum of 0.7'):
            validate_config(_make_config(mushy_zone_factor=0.5))

    def test_factor_with_non_unified_eos_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='only applies to unified PALEOS'):
            validate_config(
                _make_config(
                    mushy_zone_factor=0.8,
                    temperature_mode='linear',
                    layer_eos_config={
                        'core': 'Seager2007:iron',
                        'mantle': 'WolfBower2018:MgSiO3',
                    },
                    rock_solidus='Monteux16-solidus',
                    rock_liquidus='Monteux16-liquidus-A-chondritic',
                )
            )

    def test_factor_0_7_passes(self):
        """Minimum allowed value should not raise."""
        from zalmoxis.config import validate_config

        validate_config(_make_config(mushy_zone_factor=0.7))


@pytest.mark.unit
class TestPerEosMushyZoneValidation:
    def test_per_eos_factor_above_one_raises(self):
        """Per-material factor > 1.0 should raise."""
        from zalmoxis.config import validate_config

        # mushy_zone_factor=0.9 (global) so per-material 1.5 differs and is validated
        with pytest.raises(ValueError, match='mushy_zone_factor_iron must be in'):
            validate_config(
                _make_config(
                    mushy_zone_factor=0.9,
                    mushy_zone_factors={
                        'PALEOS:iron': 1.5,
                        'PALEOS:MgSiO3': 0.9,
                        'PALEOS:H2O': 0.9,
                    },
                )
            )

    def test_per_eos_factor_below_minimum_raises(self):
        """Per-material factor below 0.7 should raise."""
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='mushy_zone_factor_MgSiO3 = 0.5 is below'):
            validate_config(
                _make_config(
                    mushy_zone_factors={
                        'PALEOS:iron': 1.0,
                        'PALEOS:MgSiO3': 0.5,
                        'PALEOS:H2O': 1.0,
                    },
                )
            )

    def test_per_eos_factor_unused_material_raises(self):
        """Override for unconfigured material should raise."""
        from zalmoxis.config import validate_config

        # Default config has PALEOS:iron + PALEOS:MgSiO3, not PALEOS:H2O.
        # Global default is 1.0, so PALEOS:H2O=0.8 is an explicit override.
        with pytest.raises(ValueError, match='PALEOS:H2O is not configured'):
            validate_config(
                _make_config(
                    mushy_zone_factors={
                        'PALEOS:iron': 1.0,
                        'PALEOS:MgSiO3': 1.0,
                        'PALEOS:H2O': 0.8,
                    },
                )
            )

    def test_per_eos_valid_passes(self):
        """Different valid factors for configured materials should pass."""
        from zalmoxis.config import validate_config

        validate_config(
            _make_config(
                mushy_zone_factors={
                    'PALEOS:iron': 0.9,
                    'PALEOS:MgSiO3': 0.8,
                    'PALEOS:H2O': 1.0,
                },
            )
        )


@pytest.mark.unit
class TestNumericalParameterValidation:
    def test_num_layers_too_small_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='num_layers must be >= 10'):
            validate_config(_make_config(num_layers=5))

    def test_num_layers_too_large_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='excessively large'):
            validate_config(_make_config(num_layers=50000))

    def test_adaptive_fraction_out_of_range_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='adaptive_radial_fraction must be in'):
            validate_config(_make_config(adaptive_radial_fraction=0))

    def test_negative_tolerance_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='tolerance_outer must be positive'):
            validate_config(_make_config(tolerance_outer=-1e-3))

    def test_negative_max_step_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='maximum_step must be positive'):
            validate_config(_make_config(maximum_step=-100))

    def test_zero_max_iterations_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='max_iterations_outer must be >= 1'):
            validate_config(_make_config(max_iterations_outer=0))


@pytest.mark.unit
class TestValidConfigPasses:
    def test_default_config_passes(self):
        """The default config should pass validation."""
        from zalmoxis.config import validate_config

        validate_config(_make_config())

    def test_seager_isothermal_passes(self):
        from zalmoxis.config import validate_config

        validate_config(
            _make_config(
                temperature_mode='isothermal',
                layer_eos_config={
                    'core': 'Seager2007:iron',
                    'mantle': 'Seager2007:MgSiO3',
                },
            )
        )

    def test_three_layer_passes(self):
        """3-layer model with H2O ice requires T_surf < 647 K."""
        from zalmoxis.config import validate_config

        validate_config(
            _make_config(
                core_mass_fraction=0.25,
                mantle_mass_fraction=0.50,
                temperature_mode='isothermal',
                surface_temperature=300.0,
                layer_eos_config={
                    'core': 'PALEOS:iron',
                    'mantle': 'PALEOS:MgSiO3',
                    'ice_layer': 'PALEOS:H2O',
                },
            )
        )


@pytest.mark.unit
class TestThreeLayerIceTemperature:
    """Three-layer H2O ice models must have T_surf < 647 K."""

    def test_ice_layer_at_high_t_raises(self):
        """H2O ice at T >= 647 K (critical point) should raise ValueError."""
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='H2O critical point'):
            validate_config(
                _make_config(
                    core_mass_fraction=0.25,
                    mantle_mass_fraction=0.50,
                    temperature_mode='adiabatic',
                    surface_temperature=1000.0,
                    layer_eos_config={
                        'core': 'PALEOS:iron',
                        'mantle': 'PALEOS:MgSiO3',
                        'ice_layer': 'PALEOS:H2O',
                    },
                )
            )

    def test_ice_layer_isothermal_cold_passes(self):
        """H2O ice at isothermal 300 K should pass."""
        from zalmoxis.config import validate_config

        validate_config(
            _make_config(
                core_mass_fraction=0.25,
                mantle_mass_fraction=0.50,
                temperature_mode='isothermal',
                surface_temperature=300.0,
                layer_eos_config={
                    'core': 'PALEOS:iron',
                    'mantle': 'PALEOS:MgSiO3',
                    'ice_layer': 'PALEOS:H2O',
                },
            )
        )

    def test_non_h2o_ice_at_high_t_passes(self):
        """Non-H2O ice layer at high T should not trigger the H2O check."""
        from zalmoxis.config import validate_config

        validate_config(
            _make_config(
                core_mass_fraction=0.25,
                mantle_mass_fraction=0.50,
                layer_eos_config={
                    'core': 'PALEOS:iron',
                    'mantle': 'PALEOS:MgSiO3',
                    'ice_layer': 'Seager2007:MgSiO3',
                },
            )
        )
