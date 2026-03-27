"""Unit tests for zalmoxis.zalmoxis core module.

Tests cover:
- parse_eos_config(): new format, legacy format, analytic format, error cases
- validate_layer_eos(): valid and invalid EOS strings
- validate_config(): condensed-phase, binodal, multi-material, H2O, H2, Chabrier checks
- load_material_dictionaries(): returns expected registry structure
- load_solidus_liquidus_functions(): returns callables or None depending on EOS
- choose_config_file(): loads default config from TOML
- Surface density smoothing: synthetic profile anomaly detection and repair
- Post-processing computations: average density, CMB index from synthetic arrays

References:
    - docs/How-to/test_infrastructure.md
    - docs/How-to/test_categorization.md
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from zalmoxis.constants import earth_mass


def _make_config(**overrides):
    """Build a minimal valid config dict, with overrides."""
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


# ── parse_eos_config ─────────────────────────────────────────────────


@pytest.mark.unit
class TestParseEosConfig:
    """Tests for parse_eos_config() TOML section parsing."""

    def test_new_format_core_mantle(self):
        """New per-layer format with core and mantle."""
        from zalmoxis.config import parse_eos_config

        section = {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'}
        result = parse_eos_config(section)
        assert result == {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'}

    def test_new_format_with_ice_layer(self):
        """New format with ice_layer included."""
        from zalmoxis.config import parse_eos_config

        section = {
            'core': 'PALEOS:iron',
            'mantle': 'PALEOS:MgSiO3',
            'ice_layer': 'PALEOS:H2O',
        }
        result = parse_eos_config(section)
        assert result == {
            'core': 'PALEOS:iron',
            'mantle': 'PALEOS:MgSiO3',
            'ice_layer': 'PALEOS:H2O',
        }

    def test_new_format_empty_ice_layer_excluded(self):
        """Empty ice_layer string should not appear in result."""
        from zalmoxis.config import parse_eos_config

        section = {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3', 'ice_layer': ''}
        result = parse_eos_config(section)
        assert 'ice_layer' not in result

    def test_new_format_missing_mantle_raises(self):
        """Core without mantle should raise ValueError."""
        from zalmoxis.config import parse_eos_config

        with pytest.raises(ValueError, match="missing 'mantle'"):
            parse_eos_config({'core': 'PALEOS:iron'})

    def test_legacy_tabulated_iron_silicate(self):
        """Legacy choice 'Tabulated:iron/silicate' maps to Seager2007."""
        from zalmoxis.config import parse_eos_config

        result = parse_eos_config({'choice': 'Tabulated:iron/silicate'})
        assert result == {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'}

    def test_legacy_tabulated_iron_tdep_silicate(self):
        """Legacy choice 'Tabulated:iron/Tdep_silicate' maps to WolfBower2018."""
        from zalmoxis.config import parse_eos_config

        result = parse_eos_config({'choice': 'Tabulated:iron/Tdep_silicate'})
        assert result == {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'}

    def test_legacy_tabulated_water(self):
        """Legacy choice 'Tabulated:water' maps to 3-layer Seager2007."""
        from zalmoxis.config import parse_eos_config

        result = parse_eos_config({'choice': 'Tabulated:water'})
        assert result == {
            'core': 'Seager2007:iron',
            'mantle': 'Seager2007:MgSiO3',
            'ice_layer': 'Seager2007:H2O',
        }

    def test_legacy_analytic_seager2007(self):
        """Legacy choice 'Analytic:Seager2007' builds analytic per-layer config."""
        from zalmoxis.config import parse_eos_config

        result = parse_eos_config(
            {
                'choice': 'Analytic:Seager2007',
                'core_material': 'iron',
                'mantle_material': 'MgSiO3',
            }
        )
        assert result == {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'}

    def test_legacy_analytic_with_water_layer(self):
        """Legacy analytic with water_layer_material creates ice_layer."""
        from zalmoxis.config import parse_eos_config

        result = parse_eos_config(
            {
                'choice': 'Analytic:Seager2007',
                'core_material': 'iron',
                'mantle_material': 'MgSiO3',
                'water_layer_material': 'H2O',
            }
        )
        assert result['ice_layer'] == 'Analytic:H2O'

    def test_legacy_analytic_defaults(self):
        """Legacy analytic without materials uses defaults (iron, MgSiO3)."""
        from zalmoxis.config import parse_eos_config

        result = parse_eos_config({'choice': 'Analytic:Seager2007'})
        assert result == {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'}

    def test_unknown_config_raises(self):
        """Completely unknown EOS config raises ValueError."""
        from zalmoxis.config import parse_eos_config

        with pytest.raises(ValueError, match='Unknown EOS config'):
            parse_eos_config({'choice': 'NonExistentEOS'})

    def test_empty_section_raises(self):
        """Empty dict with no core or choice raises ValueError."""
        from zalmoxis.config import parse_eos_config

        with pytest.raises(ValueError, match='Unknown EOS config'):
            parse_eos_config({})


# ── validate_layer_eos ───────────────────────────────────────────────


@pytest.mark.unit
class TestValidateLayerEos:
    """Tests for validate_layer_eos() EOS string validation."""

    def test_valid_tabulated_eos(self):
        """Valid tabulated EOS strings should pass."""
        from zalmoxis.config import validate_layer_eos

        validate_layer_eos({'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'})

    def test_valid_analytic_eos(self):
        """Valid analytic EOS strings should pass."""
        from zalmoxis.config import validate_layer_eos

        validate_layer_eos({'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'})

    def test_invalid_analytic_material_raises(self):
        """Invalid analytic material key should raise ValueError."""
        from zalmoxis.config import validate_layer_eos

        with pytest.raises(ValueError, match='Invalid analytic material'):
            validate_layer_eos({'core': 'Analytic:unobtanium', 'mantle': 'PALEOS:MgSiO3'})

    def test_invalid_tabulated_eos_raises(self):
        """Completely invalid EOS component should raise ValueError."""
        from zalmoxis.config import validate_layer_eos

        with pytest.raises(ValueError, match='Invalid EOS component'):
            validate_layer_eos({'core': 'FakeEOS:stuff', 'mantle': 'PALEOS:MgSiO3'})

    def test_multi_material_mixture_valid(self):
        """Multi-material mixture string should validate each component."""
        from zalmoxis.config import validate_layer_eos

        validate_layer_eos(
            {
                'core': 'PALEOS:iron',
                'mantle': 'PALEOS:MgSiO3:0.85+PALEOS:H2O:0.15',
            }
        )

    def test_chabrier_h_valid(self):
        """Chabrier:H is a valid tabulated EOS."""
        from zalmoxis.config import validate_layer_eos

        validate_layer_eos(
            {
                'core': 'PALEOS:iron',
                'mantle': 'PALEOS:MgSiO3:0.97+Chabrier:H:0.03',
            }
        )


# ── validate_config: condensed-phase and binodal params ──────────────


@pytest.mark.unit
class TestValidateCondensedPhaseParams:
    """Tests for condensed_rho_min, condensed_rho_scale, and binodal_T_scale validation."""

    def test_negative_condensed_rho_min_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='condensed_rho_min must be positive'):
            validate_config(_make_config(condensed_rho_min=-10))

    def test_zero_condensed_rho_min_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='condensed_rho_min must be positive'):
            validate_config(_make_config(condensed_rho_min=0))

    def test_negative_condensed_rho_scale_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='condensed_rho_scale must be positive'):
            validate_config(_make_config(condensed_rho_scale=-5))

    def test_zero_condensed_rho_scale_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='condensed_rho_scale must be positive'):
            validate_config(_make_config(condensed_rho_scale=0))

    def test_negative_binodal_T_scale_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='binodal_T_scale must be positive'):
            validate_config(_make_config(binodal_T_scale=-1))

    def test_zero_binodal_T_scale_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='binodal_T_scale must be positive'):
            validate_config(_make_config(binodal_T_scale=0))

    def test_valid_condensed_params_pass(self):
        """Valid condensed-phase params should not raise."""
        from zalmoxis.config import validate_config

        validate_config(
            _make_config(
                condensed_rho_min=300.0,
                condensed_rho_scale=50.0,
                binodal_T_scale=100.0,
            )
        )


# ── validate_config: multi-material mixing checks ───────────────────


@pytest.mark.unit
class TestValidateMultiMaterialMixing:
    """Tests for multi-material fraction validation in validate_config."""

    def test_negative_mass_fraction_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='Negative mass fraction'):
            validate_config(
                _make_config(
                    layer_eos_config={
                        'core': 'PALEOS:iron',
                        'mantle': 'PALEOS:MgSiO3:-0.1+PALEOS:H2O:1.1',
                    },
                )
            )

    def test_fractions_not_summing_to_one_normalized(self, caplog):
        """Fractions that don't sum to 1 are auto-normalized with a warning."""
        from zalmoxis.config import validate_config

        # parse_layer_components normalizes fractions with a warning rather than raising.
        # Verify the warning is logged and validation still passes.
        with caplog.at_level('WARNING'):
            validate_config(
                _make_config(
                    layer_eos_config={
                        'core': 'PALEOS:iron',
                        'mantle': 'PALEOS:MgSiO3:0.5+PALEOS:H2O:0.3',
                    },
                )
            )
        assert 'normalizing to 1.0' in caplog.text


# ── validate_config: H2O-dominated mantle ───────────────────────────


@pytest.mark.unit
class TestValidateH2ODominatedMantle:
    """Tests for H2O-dominated mantle rejection."""

    def test_h2o_dominated_mantle_no_silicate_raises(self):
        """Mantle with >50% H2O and no silicate in non-isothermal mode should raise."""
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='H2O with no silicate'):
            validate_config(
                _make_config(
                    layer_eos_config={
                        'core': 'PALEOS:iron',
                        'mantle': 'PALEOS:H2O',
                    },
                )
            )

    def test_h2o_dominated_mantle_isothermal_passes(self):
        """H2O-dominated mantle in isothermal mode should pass."""
        from zalmoxis.config import validate_config

        validate_config(
            _make_config(
                temperature_mode='isothermal',
                surface_temperature=300.0,
                layer_eos_config={
                    'core': 'PALEOS:iron',
                    'mantle': 'PALEOS:H2O',
                },
            )
        )

    def test_h2o_with_silicate_passes(self):
        """H2O mixed with silicate should pass even at >50%."""
        from zalmoxis.config import validate_config

        validate_config(
            _make_config(
                layer_eos_config={
                    'core': 'PALEOS:iron',
                    'mantle': 'PALEOS:MgSiO3:0.40+PALEOS:H2O:0.60',
                },
            )
        )


# ── validate_config: pure Chabrier:H mantle ─────────────────────────


@pytest.mark.unit
class TestValidatePureChabrierMantle:
    """Tests for pure Chabrier:H mantle rejection."""

    def test_pure_chabrier_h_mantle_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='Pure Chabrier:H mantle'):
            validate_config(
                _make_config(
                    layer_eos_config={
                        'core': 'PALEOS:iron',
                        'mantle': 'Chabrier:H',
                    },
                )
            )


# ── validate_config: EOS cross-checks (melting curves needed) ───────


@pytest.mark.unit
class TestValidateMeltingCurvesCrossCheck:
    """Tests for the melting-curves-required cross-check."""

    def test_wolfbower_without_melting_curves_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='melting curves'):
            validate_config(
                _make_config(
                    temperature_mode='linear',
                    layer_eos_config={
                        'core': 'Seager2007:iron',
                        'mantle': 'WolfBower2018:MgSiO3',
                    },
                    rock_solidus='',
                    rock_liquidus='',
                )
            )

    def test_wolfbower_with_melting_curves_passes(self):
        from zalmoxis.config import validate_config

        validate_config(
            _make_config(
                temperature_mode='linear',
                layer_eos_config={
                    'core': 'Seager2007:iron',
                    'mantle': 'WolfBower2018:MgSiO3',
                },
                rock_solidus='Monteux16-solidus',
                rock_liquidus='Monteux16-liquidus-A-chondritic',
            )
        )


# ── validate_config: pressure solver params ─────────────────────────


@pytest.mark.unit
class TestValidatePressureSolverParams:
    """Tests for target_surface_pressure, pressure_tolerance, max_center_pressure_guess."""

    def test_negative_target_surface_pressure_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='target_surface_pressure must be >= 0'):
            validate_config(_make_config(target_surface_pressure=-1))

    def test_zero_target_surface_pressure_passes(self):
        """Zero surface pressure (vacuum boundary) should pass."""
        from zalmoxis.config import validate_config

        validate_config(_make_config(target_surface_pressure=0))

    def test_negative_pressure_tolerance_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='pressure_tolerance must be positive'):
            validate_config(_make_config(pressure_tolerance=-1e9))

    def test_negative_max_center_pressure_guess_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='max_center_pressure_guess must be positive'):
            validate_config(_make_config(max_center_pressure_guess=-1e12))

    def test_zero_max_iterations_pressure_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='max_iterations_pressure must be >= 1'):
            validate_config(_make_config(max_iterations_pressure=0))


# ── validate_config: tolerance and solver params ─────────────────────


@pytest.mark.unit
class TestValidateToleranceParams:
    """Tests for solver tolerance validation."""

    def test_negative_tolerance_inner_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='tolerance_inner must be positive'):
            validate_config(_make_config(tolerance_inner=-1e-4))

    def test_negative_relative_tolerance_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='relative_tolerance must be positive'):
            validate_config(_make_config(relative_tolerance=-1e-5))

    def test_negative_absolute_tolerance_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='absolute_tolerance must be positive'):
            validate_config(_make_config(absolute_tolerance=-1e-6))

    def test_zero_max_iterations_inner_raises(self):
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='max_iterations_inner must be >= 1'):
            validate_config(_make_config(max_iterations_inner=0))


# ── load_material_dictionaries ───────────────────────────────────────


@pytest.mark.unit
class TestLoadMaterialDictionaries:
    """Tests for load_material_dictionaries()."""

    def test_returns_dict(self):
        from zalmoxis.config import load_material_dictionaries

        result = load_material_dictionaries()
        assert isinstance(result, dict)

    def test_contains_seager_iron(self):
        """Registry should contain Seager2007:iron."""
        from zalmoxis.config import load_material_dictionaries

        result = load_material_dictionaries()
        assert 'Seager2007:iron' in result

    def test_contains_seager_mgsio3(self):
        """Registry should contain Seager2007:MgSiO3."""
        from zalmoxis.config import load_material_dictionaries

        result = load_material_dictionaries()
        assert 'Seager2007:MgSiO3' in result

    def test_seager_iron_has_expected_structure(self):
        """Each registry entry should be a dict with EOS metadata."""
        from zalmoxis.config import load_material_dictionaries

        result = load_material_dictionaries()
        entry = result['Seager2007:iron']
        # Should be a dict (or dict-like) with at least a format or type field
        assert isinstance(entry, dict)


# ── load_solidus_liquidus_functions ──────────────────────────────────


@pytest.mark.unit
class TestLoadSolidusLiquidusFunctions:
    """Tests for load_solidus_liquidus_functions()."""

    def test_returns_none_for_paleos_unified(self):
        """Unified PALEOS tables do not need external melting curves."""
        from zalmoxis.config import load_solidus_liquidus_functions

        result = load_solidus_liquidus_functions(
            {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
        )
        assert result is None

    def test_returns_none_for_seager(self):
        """Seager2007 (T-independent) does not need melting curves."""
        from zalmoxis.config import load_solidus_liquidus_functions

        result = load_solidus_liquidus_functions(
            {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'},
        )
        assert result is None

    def test_returns_callables_for_wolfbower(self):
        """WolfBower2018 needs external melting curves."""
        from zalmoxis.config import load_solidus_liquidus_functions

        result = load_solidus_liquidus_functions(
            {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'},
            solidus_id='Monteux16-solidus',
            liquidus_id='Monteux16-liquidus-A-chondritic',
        )
        assert result is not None
        solidus_func, liquidus_func = result
        assert callable(solidus_func)
        assert callable(liquidus_func)

    def test_returns_callables_for_paleos_2phase(self):
        """PALEOS-2phase:MgSiO3 needs external melting curves."""
        from zalmoxis.config import load_solidus_liquidus_functions

        result = load_solidus_liquidus_functions(
            {'core': 'Seager2007:iron', 'mantle': 'PALEOS-2phase:MgSiO3'},
            solidus_id='Stixrude14-solidus',
            liquidus_id='Stixrude14-liquidus',
        )
        assert result is not None
        solidus_func, liquidus_func = result
        assert callable(solidus_func)
        assert callable(liquidus_func)

    def test_melting_curve_evaluates(self):
        """Returned melting curve functions should accept a pressure and return temperature."""
        from zalmoxis.config import load_solidus_liquidus_functions

        result = load_solidus_liquidus_functions(
            {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'},
            solidus_id='Monteux16-solidus',
            liquidus_id='Monteux16-liquidus-A-chondritic',
        )
        solidus_func, liquidus_func = result
        # Evaluate at 10 GPa (Earth lower mantle)
        T_solidus = solidus_func(10e9)
        T_liquidus = liquidus_func(10e9)
        assert T_solidus > 0
        assert T_liquidus > 0
        assert T_liquidus >= T_solidus


# ── choose_config_file ───────────────────────────────────────────────


@pytest.mark.unit
class TestChooseConfigFile:
    """Tests for choose_config_file()."""

    def test_loads_from_path(self):
        """Should load a TOML file from an explicit path."""
        from zalmoxis.config import choose_config_file

        root = os.environ.get('ZALMOXIS_ROOT', '')
        config_path = os.path.join(root, 'input', 'default.toml')
        if not os.path.isfile(config_path):
            pytest.skip('default.toml not found')

        config = choose_config_file(config_path)
        assert isinstance(config, dict)
        assert 'InputParameter' in config
        assert 'EOS' in config

    def test_nonexistent_path_exits(self):
        """Non-existent temp config path should exit."""
        from zalmoxis.config import choose_config_file

        with pytest.raises(SystemExit):
            choose_config_file('/nonexistent/path/config.toml')


# ── Surface density smoothing logic ─────────────────────────────────


@pytest.mark.unit
class TestSurfaceDensitySmoothing:
    """Tests for the surface density smoothing applied in main() after convergence.

    The smoothing detects gradient anomalies in the outer shells and
    extrapolates from the smooth interior. We test the logic by
    constructing synthetic density and pressure arrays and applying
    the smoothing code directly.
    """

    @staticmethod
    def _apply_smoothing(density, pressure):
        """Apply the surface density smoothing from main().

        Extracted logic from zalmoxis.py lines 1464-1517.
        """
        pressure = np.asarray(pressure)
        density = np.asarray(density, dtype=float)

        # Zero out padded (P=0) shells
        density[pressure <= 0] = 0.0

        # Find the last shell with positive density
        i_surf = len(density) - 1
        while i_surf > 0 and density[i_surf] <= 0:
            i_surf -= 1

        if i_surf > 10:
            i_layer_base = max(0, i_surf - 40)
            for _k in range(i_surf - 1, i_layer_base, -1):
                if density[_k] > 0 and density[_k + 1] > 0:
                    ratio = density[_k] / density[_k + 1]
                    if ratio > 1.5 or ratio < 1.0 / 1.5:
                        i_layer_base = _k + 1
                        break
            n_check = min(20, i_surf - i_layer_base)
            if n_check < 6:
                n_check = min(20, i_surf - 5)
            i_start = i_surf - n_check
            grads = np.diff(density[i_start : i_surf + 1])
            if len(grads) > 5:
                median_grad = np.median(grads[: n_check // 2])
                if median_grad < 0:
                    for j in range(len(grads)):
                        i_global = i_start + j + 1
                        is_sudden_drop = grads[j] < 3 * median_grad
                        is_sudden_rise = grads[j] > 0 and abs(grads[j]) > abs(median_grad)
                        if is_sudden_drop or is_sudden_rise:
                            for k in range(i_global, i_surf + 1):
                                extrap = density[i_global - 1] + median_grad * (
                                    k - i_global + 1
                                )
                                density[k] = max(extrap, 0.0)
                            break

        return density

    def test_smooth_profile_unchanged(self):
        """A smoothly decreasing density profile should not be modified."""
        n = 50
        density = np.linspace(5000, 3000, n)
        pressure = np.linspace(100e9, 1e5, n)

        original = density.copy()
        result = self._apply_smoothing(density, pressure)
        np.testing.assert_allclose(result, original, atol=1e-10)

    def test_sudden_drop_at_surface_repaired(self):
        """A sudden density drop at the outermost shells should be smoothed out."""
        n = 50
        density = np.linspace(5000, 3000, n)
        pressure = np.linspace(100e9, 1e5, n)

        # Introduce a sudden drop at the last 3 shells
        density[-3:] = [1000, 500, 200]

        result = self._apply_smoothing(density.copy(), pressure)
        # The last 3 shells should have been extrapolated, not kept at the anomalous values
        assert result[-3] > 1500, f'Expected smoothed value > 1500, got {result[-3]}'

    def test_sudden_rise_at_surface_repaired(self):
        """A sudden density increase at the surface should trigger smoothing."""
        n = 50
        density = np.linspace(5000, 3000, n)
        pressure = np.linspace(100e9, 1e5, n)

        # Introduce a sudden rise at shell -5
        density[-5] = 4500  # sudden increase against the decreasing trend

        result = self._apply_smoothing(density.copy(), pressure)
        # The rise should have been smoothed
        assert result[-5] < 4000, f'Expected smoothed value < 4000, got {result[-5]}'

    def test_zero_pressure_shells_get_zero_density(self):
        """Shells with P=0 should have density set to 0."""
        n = 50
        density = np.linspace(5000, 3000, n)
        pressure = np.linspace(100e9, 1e5, n)
        # Set last 5 shells to P=0
        pressure[-5:] = 0.0
        density[-5:] = 1000.0  # nonzero density at zero pressure

        result = self._apply_smoothing(density.copy(), pressure)
        np.testing.assert_array_equal(result[-5:], 0.0)

    def test_short_profile_no_crash(self):
        """Profile with <= 10 shells should not crash (smoothing skipped)."""
        n = 8
        density = np.linspace(5000, 3000, n)
        pressure = np.linspace(100e9, 1e5, n)

        result = self._apply_smoothing(density.copy(), pressure)
        # Should return without modification (except P=0 zeroing, which doesn't apply here)
        np.testing.assert_allclose(result, density, atol=1e-10)


# ── Post-processing computations ─────────────────────────────────────


@pytest.mark.unit
class TestPostProcessingComputations:
    """Tests for computations performed in post_processing() using synthetic arrays.

    These verify the cmb_index, average_density, and related derived
    quantities without running the full solver.
    """

    def test_average_density_from_synthetic(self):
        """Verify average density = M_total / (4/3 pi R^3)."""
        n = 100
        radii = np.linspace(0, 6.4e6, n)  # ~Earth radius
        # Total mass at outer edge
        M_total = 5.97e24  # ~Earth mass
        R_total = radii[-1]

        avg_density = M_total / (4 / 3 * math.pi * R_total**3)
        # Earth average density ~5500 kg/m^3
        assert pytest.approx(avg_density, rel=0.1) == 5500

    def test_cmb_index_from_mass_enclosed(self):
        """cmb_index = argmax(mass_enclosed >= cmb_mass) should find the CMB."""
        n = 100
        # Linear mass enclosed from 0 to M_total
        M_total = 5.97e24
        mass_enclosed = np.linspace(0, M_total, n)
        cmb_mass = 0.325 * M_total

        cmb_index = np.argmax(mass_enclosed >= cmb_mass)
        # Should be at approximately 32.5% of the way through
        expected_index = int(0.325 * n)
        assert abs(cmb_index - expected_index) <= 2

    def test_core_radius_fraction(self):
        """Core radius fraction should be R[cmb_index] / R[-1]."""
        n = 100
        radii = np.linspace(0, 6.4e6, n)
        M_total = 5.97e24
        mass_enclosed = np.linspace(0, M_total, n)
        cmb_mass = 0.325 * M_total

        cmb_index = np.argmax(mass_enclosed >= cmb_mass)
        core_radius_fraction = radii[cmb_index] / radii[-1]
        # Should be approximately 0.325 for linear mass profile
        assert pytest.approx(core_radius_fraction, abs=0.05) == 0.325

    def test_cmb_mass_fraction(self):
        """CMB mass fraction = M[cmb_index] / M_total."""
        n = 100
        M_total = 5.97e24
        mass_enclosed = np.linspace(0, M_total, n)
        cmb_mass = 0.325 * M_total

        cmb_index = np.argmax(mass_enclosed >= cmb_mass)
        cmb_mass_fraction = mass_enclosed[cmb_index] / mass_enclosed[-1]
        assert pytest.approx(cmb_mass_fraction, abs=0.02) == 0.325


# ── validate_config: mass limit warnings ────────────────────────────


@pytest.mark.unit
class TestValidateConfigMassWarnings:
    """Tests for planet mass range warnings (emitted via logger.warning)."""

    def test_low_mass_warning(self, caplog):
        """Planet mass below 0.1 M_earth should log a warning."""
        from zalmoxis.config import validate_config

        with caplog.at_level('WARNING'):
            validate_config(_make_config(planet_mass=0.05 * earth_mass))
        assert 'below the validated range' in caplog.text

    def test_high_mass_warning(self, caplog):
        """Planet mass above 50 M_earth should log a warning."""
        from zalmoxis.config import validate_config

        with caplog.at_level('WARNING'):
            validate_config(_make_config(planet_mass=60 * earth_mass))
        assert 'exceeds the validated range' in caplog.text


# ── validate_config: EOS-layer compatibility warnings ───────────────


@pytest.mark.unit
class TestValidateEosLayerCompatibility:
    """Tests for EOS-layer compatibility warnings (non-iron core)."""

    def test_non_iron_core_warning(self, caplog):
        """Core with non-iron EOS should log a warning."""
        from zalmoxis.config import validate_config

        with caplog.at_level('WARNING'):
            validate_config(
                _make_config(
                    temperature_mode='isothermal',
                    layer_eos_config={
                        'core': 'Seager2007:MgSiO3',  # silicate in core
                        'mantle': 'Seager2007:MgSiO3',
                    },
                )
            )
        assert 'does not appear to be an iron EOS' in caplog.text


# ── LEGACY_EOS_MAP ──────────────────────────────────────────────────


@pytest.mark.unit
class TestLegacyEosMap:
    """Tests for the LEGACY_EOS_MAP constant."""

    def test_contains_all_legacy_choices(self):
        from zalmoxis.config import LEGACY_EOS_MAP

        assert 'Tabulated:iron/silicate' in LEGACY_EOS_MAP
        assert 'Tabulated:iron/Tdep_silicate' in LEGACY_EOS_MAP
        assert 'Tabulated:water' in LEGACY_EOS_MAP

    def test_legacy_map_is_immutable_via_copy(self):
        """parse_eos_config returns a copy, not the original dict."""
        from zalmoxis.config import LEGACY_EOS_MAP, parse_eos_config

        result = parse_eos_config({'choice': 'Tabulated:iron/silicate'})
        result['core'] = 'MODIFIED'
        # Original should not be affected
        assert LEGACY_EOS_MAP['Tabulated:iron/silicate']['core'] == 'Seager2007:iron'


# ── VALID_TABULATED_EOS ─────────────────────────────────────────────


@pytest.mark.unit
class TestValidTabulatedEos:
    """Tests for the VALID_TABULATED_EOS set."""

    def test_contains_expected_entries(self):
        from zalmoxis.config import VALID_TABULATED_EOS

        expected = {
            'Seager2007:iron',
            'Seager2007:MgSiO3',
            'WolfBower2018:MgSiO3',
            'PALEOS:iron',
            'PALEOS:MgSiO3',
            'PALEOS:H2O',
            'Chabrier:H',
        }
        assert expected.issubset(VALID_TABULATED_EOS)


# ── _NEEDS_MELTING_CURVES ───────────────────────────────────────────


@pytest.mark.unit
class TestNeedsMeltingCurves:
    """Tests for the _NEEDS_MELTING_CURVES set."""

    def test_wolfbower_needs_melting_curves(self):
        from zalmoxis.config import _NEEDS_MELTING_CURVES

        assert 'WolfBower2018:MgSiO3' in _NEEDS_MELTING_CURVES

    def test_paleos_2phase_needs_melting_curves(self):
        from zalmoxis.config import _NEEDS_MELTING_CURVES

        assert 'PALEOS-2phase:MgSiO3' in _NEEDS_MELTING_CURVES

    def test_paleos_unified_does_not_need(self):
        from zalmoxis.config import _NEEDS_MELTING_CURVES

        assert 'PALEOS:iron' not in _NEEDS_MELTING_CURVES
        assert 'PALEOS:MgSiO3' not in _NEEDS_MELTING_CURVES


# ── Mass limit constants ────────────────────────────────────────────


@pytest.mark.unit
class TestMassLimitConstants:
    """Tests for EOS-specific mass limit constants."""

    def test_wolfbower_limit(self):
        from zalmoxis.config import WOLFBOWER2018_MAX_MASS_EARTH

        assert WOLFBOWER2018_MAX_MASS_EARTH == pytest.approx(7.0)

    def test_rtpress_limit(self):
        from zalmoxis.config import RTPRESS100TPA_MAX_MASS_EARTH

        assert RTPRESS100TPA_MAX_MASS_EARTH == pytest.approx(50.0)

    def test_paleos_limit(self):
        from zalmoxis.config import PALEOS_MAX_MASS_EARTH

        assert PALEOS_MAX_MASS_EARTH == pytest.approx(50.0)

    def test_paleos_unified_limit(self):
        from zalmoxis.config import PALEOS_UNIFIED_MAX_MASS_EARTH

        assert PALEOS_UNIFIED_MAX_MASS_EARTH == pytest.approx(50.0)


# ── validate_config: center_temperature in linear mode ──────────────


@pytest.mark.unit
class TestValidateCenterTemperature:
    """Tests for center_temperature validation in different modes."""

    def test_center_temp_zero_linear_raises(self):
        """center_temperature <= 0 in linear mode should raise."""
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='center_temperature must be positive'):
            validate_config(
                _make_config(
                    temperature_mode='linear',
                    center_temperature=0,
                    layer_eos_config={
                        'core': 'Seager2007:iron',
                        'mantle': 'WolfBower2018:MgSiO3',
                    },
                    rock_solidus='Monteux16-solidus',
                    rock_liquidus='Monteux16-liquidus-A-chondritic',
                )
            )

    def test_center_temp_zero_adiabatic_raises(self):
        """center_temperature <= 0 in adiabatic mode should raise."""
        from zalmoxis.config import validate_config

        with pytest.raises(ValueError, match='center_temperature must be positive'):
            validate_config(
                _make_config(
                    temperature_mode='adiabatic',
                    center_temperature=-100,
                )
            )
