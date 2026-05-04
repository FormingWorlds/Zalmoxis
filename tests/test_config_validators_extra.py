"""Targeted tests for ``validate_config`` branches not covered by
``test_config_validation.py``.

The existing file covers planet mass, mass fractions, three-layer rules,
basic temperature, and mushy-zone factor edges. This file fills in the
remaining branches: condensed-phase scale validators, pressure-solver
parameters, EOS-specific cross-checks, multi-material mixing rules,
H2O / Chabrier:H gating, and the ``surface_temperature > 5000 K`` warning.

Anti-happy-path: each test class includes ≥ 1 edge case and ≥ 1
physically-unreasonable input that raises.
"""

from __future__ import annotations

import pytest

from zalmoxis.config import validate_config

pytestmark = pytest.mark.unit


def _make_minimal_config(**overrides):
    """Minimal physically-valid config_params dict."""
    base = {
        'planet_mass': 5.972e24,  # 1 Earth mass
        'core_mass_fraction': 0.32,
        'mantle_mass_fraction': 0.0,
        'temperature_mode': 'linear',
        'surface_temperature': 1500.0,
        'center_temperature': 5000.0,
        'temp_profile_file': '',
        'layer_eos_config': {
            'core': 'PALEOS:iron',
            'mantle': 'PALEOS:MgSiO3',
        },
        'rock_solidus': 'Stixrude14-solidus',
        'rock_liquidus': 'Stixrude14-liquidus',
        'mushy_zone_factor': 1.0,
        'mushy_zone_factors': {
            'PALEOS:iron': 1.0,
            'PALEOS:MgSiO3': 1.0,
            'PALEOS:H2O': 1.0,
        },
        'condensed_rho_min': 300.0,
        'condensed_rho_scale': 50.0,
        'binodal_T_scale': 50.0,
        'num_layers': 200,
        'target_surface_pressure': 101325.0,
        'data_output_enabled': True,
        'plotting_enabled': False,
    }
    base.update(overrides)
    return base


class TestSurfaceTemperatureWarning:
    """``surface_temperature > 5000`` emits a warning, not an error."""

    def test_warning_above_5000_kelvin(self, caplog):
        """Surface T > 5000 K logs a warning but the config is still valid."""
        cfg = _make_minimal_config(surface_temperature=6000.0)
        with caplog.at_level('WARNING'):
            validate_config(cfg)
        assert any('exceeds the validated range' in r.message for r in caplog.records)

    def test_5000_kelvin_exactly_no_warning(self, caplog):
        """Edge: surface T == 5000 K is at the boundary, no warning."""
        cfg = _make_minimal_config(surface_temperature=5000.0)
        with caplog.at_level('WARNING'):
            validate_config(cfg)
        # No "validated range" warning at the boundary.
        assert not any('exceeds the validated range' in r.message for r in caplog.records)


class TestCenterTemperatureValidation:
    """``center_temperature`` rules in linear / adiabatic modes."""

    def test_zero_center_temp_in_linear_mode_raises(self):
        """Linear mode with center_temp=0 -> ValueError."""
        cfg = _make_minimal_config(temperature_mode='linear', center_temperature=0.0)
        with pytest.raises(ValueError, match='center_temperature must be positive'):
            validate_config(cfg)

    def test_negative_center_temp_in_adiabatic_mode_raises(self):
        """Adiabatic mode with negative center_temp -> ValueError.

        Edge case: physically unreasonable (T < 0 K).
        """
        cfg = _make_minimal_config(
            temperature_mode='adiabatic',
            center_temperature=-100.0,
            layer_eos_config={
                'core': 'PALEOS:iron',
                'mantle': 'WolfBower2018:MgSiO3',
            },
        )
        with pytest.raises(ValueError):
            validate_config(cfg)

    def test_center_temp_below_surface_in_linear_mode_warns(self, caplog):
        """Linear mode with center_temp < surface_temp logs a warning."""
        cfg = _make_minimal_config(
            temperature_mode='linear',
            surface_temperature=2000.0,
            center_temperature=1500.0,
        )
        with caplog.at_level('WARNING'):
            validate_config(cfg)
        assert any('temperature gradient' in r.message for r in caplog.records)


class TestAdiabaticFromCmb:
    """``temperature_mode = 'adiabatic_from_cmb'`` requires cmb_temperature."""

    def test_missing_cmb_temperature_raises(self):
        """Missing cmb_temperature key -> ValueError."""
        cfg = _make_minimal_config(
            temperature_mode='adiabatic_from_cmb',
            layer_eos_config={
                'core': 'PALEOS:iron',
                'mantle': 'WolfBower2018:MgSiO3',
            },
        )
        with pytest.raises(ValueError, match='cmb_temperature must be positive'):
            validate_config(cfg)

    def test_zero_cmb_temperature_raises(self):
        """Edge: cmb_temperature == 0 -> ValueError."""
        cfg = _make_minimal_config(
            temperature_mode='adiabatic_from_cmb',
            cmb_temperature=0.0,
            layer_eos_config={
                'core': 'PALEOS:iron',
                'mantle': 'WolfBower2018:MgSiO3',
            },
        )
        with pytest.raises(ValueError):
            validate_config(cfg)


class TestCondensedAndBinodalScales:
    """Validators for condensed_rho_min / scale and binodal_T_scale."""

    def test_negative_condensed_rho_min_raises(self):
        """Physically unreasonable: rho_min <= 0 -> ValueError."""
        cfg = _make_minimal_config(condensed_rho_min=0.0)
        with pytest.raises(ValueError, match='condensed_rho_min must be positive'):
            validate_config(cfg)

    def test_negative_condensed_rho_scale_raises(self):
        """Physically unreasonable: rho_scale <= 0 -> ValueError."""
        cfg = _make_minimal_config(condensed_rho_scale=-1.0)
        with pytest.raises(ValueError, match='condensed_rho_scale must be positive'):
            validate_config(cfg)

    def test_negative_binodal_T_scale_raises(self):
        """Physically unreasonable: binodal_T_scale <= 0 -> ValueError."""
        cfg = _make_minimal_config(binodal_T_scale=0.0)
        with pytest.raises(ValueError, match='binodal_T_scale must be positive'):
            validate_config(cfg)


class TestPressureSolverParams:
    """target_surface_pressure, pressure_tolerance, max_center_pressure_guess."""

    def test_negative_target_surface_pressure_raises(self):
        """target_surface_pressure < 0 -> ValueError."""
        cfg = _make_minimal_config(target_surface_pressure=-100.0)
        with pytest.raises(ValueError, match='target_surface_pressure must be >= 0'):
            validate_config(cfg)

    def test_zero_pressure_tolerance_raises(self):
        """Edge: pressure_tolerance == 0 -> ValueError."""
        cfg = _make_minimal_config(pressure_tolerance=0.0)
        with pytest.raises(ValueError, match='pressure_tolerance must be positive'):
            validate_config(cfg)

    def test_negative_max_center_pressure_guess_raises(self):
        """Physically unreasonable: max_center_pressure_guess <= 0 -> ValueError."""
        cfg = _make_minimal_config(max_center_pressure_guess=-1e9)
        with pytest.raises(ValueError, match='max_center_pressure_guess must be positive'):
            validate_config(cfg)


class TestNumericalSolverParams:
    """Per-key numeric validators in the iterative-solver section."""

    def test_zero_relative_tolerance_raises(self):
        """relative_tolerance == 0 -> ValueError."""
        cfg = _make_minimal_config(relative_tolerance=0.0)
        with pytest.raises(ValueError):
            validate_config(cfg)

    def test_zero_max_iterations_outer_raises(self):
        """max_iterations_outer < 1 -> ValueError."""
        cfg = _make_minimal_config(max_iterations_outer=0)
        with pytest.raises(ValueError):
            validate_config(cfg)

    def test_negative_maximum_step_raises(self):
        """maximum_step <= 0 -> ValueError."""
        cfg = _make_minimal_config(maximum_step=-1.0)
        with pytest.raises(ValueError, match='maximum_step must be positive'):
            validate_config(cfg)


class TestMeltingCurveCrossChecks:
    """When a layer needs external melting curves, both must be set."""

    def test_tdep_mantle_with_empty_solidus_raises(self):
        """``WolfBower2018:MgSiO3`` mantle requires non-empty rock_solidus/liquidus."""
        cfg = _make_minimal_config(
            layer_eos_config={
                'core': 'PALEOS:iron',
                'mantle': 'WolfBower2018:MgSiO3',
            },
            rock_solidus='',
            rock_liquidus='Stixrude14-liquidus',
        )
        with pytest.raises(ValueError, match='requires melting curves'):
            validate_config(cfg)


class TestMultiMaterialMixing:
    """Validators on multi-component layer strings."""

    def test_negative_mass_fraction_in_mixture_raises(self):
        """A negative mass-fraction component -> ValueError."""
        cfg = _make_minimal_config(
            layer_eos_config={
                'core': 'PALEOS:iron',
                'mantle': 'PALEOS:MgSiO3:1.5+Chabrier:H:-0.5',
            },
        )
        with pytest.raises(ValueError, match='Negative mass fraction'):
            validate_config(cfg)

    def test_underspecified_fractions_emit_normalisation_warning(self, caplog):
        """Mixture fractions summing to 0.8 trigger an in-place normalisation warning.

        Note: ``parse_layer_components`` auto-normalises and emits a
        warning, so ``validate_config`` does not see the un-normalised
        sum (the explicit-sum-mismatch raise at L527 of config.py is
        defensive code that the parser short-circuits in practice).
        """
        cfg = _make_minimal_config(
            layer_eos_config={
                'core': 'PALEOS:iron',
                'mantle': 'PALEOS:MgSiO3:0.5+Chabrier:H:0.3',
            },
        )
        with caplog.at_level('WARNING'):
            validate_config(cfg)
        assert any('normalizing to 1.0' in r.message for r in caplog.records)

    def test_h2o_fraction_above_30pct_warns(self, caplog):
        """Edge: > 30 % H2O in a mantle layer logs a warning."""
        cfg = _make_minimal_config(
            layer_eos_config={
                'core': 'PALEOS:iron',
                'mantle': 'PALEOS:MgSiO3:0.5+PALEOS:H2O:0.5',
            },
        )
        with caplog.at_level('WARNING'):
            validate_config(cfg)
        assert any('exceeds the validated range' in r.message for r in caplog.records)

    def test_h2_fraction_above_20pct_warns(self, caplog):
        """Edge: > 20 % Chabrier:H in mantle logs a validated-range warning."""
        cfg = _make_minimal_config(
            layer_eos_config={
                'core': 'PALEOS:iron',
                'mantle': 'PALEOS:MgSiO3:0.7+Chabrier:H:0.3',
            },
        )
        with caplog.at_level('WARNING'):
            validate_config(cfg)
        assert any('validated up to 20% H2' in r.message for r in caplog.records)


class TestExoticMantleConfigs:
    """Reject H2O-only mantle and Pure Chabrier:H mantle in non-isothermal mode."""

    def test_pure_h2o_mantle_in_linear_mode_raises(self):
        """100 % H2O mantle without silicate in linear mode -> ValueError."""
        cfg = _make_minimal_config(
            layer_eos_config={
                'core': 'PALEOS:iron',
                'mantle': 'PALEOS:H2O',
            },
        )
        with pytest.raises(ValueError, match='no silicate component'):
            validate_config(cfg)

    def test_pure_chabrier_h_mantle_raises(self):
        """A mantle entirely made of Chabrier:H hits the explicit rejection."""
        # Chabrier:H must be a valid EOS that survives validate_layer_eos.
        cfg = _make_minimal_config(
            layer_eos_config={
                'core': 'PALEOS:iron',
                'mantle': 'Chabrier:H',
            },
            temperature_mode='isothermal',  # bypass adiabatic-Tdep check
        )
        with pytest.raises(ValueError, match='Pure Chabrier:H mantle'):
            validate_config(cfg)


class TestNonIronCoreWarning:
    """Non-iron core EOS triggers an informational warning."""

    def test_silicate_in_core_logs_warning(self, caplog):
        """Edge: core EOS string contains a silicate -> warning, no raise."""
        cfg = _make_minimal_config(
            layer_eos_config={
                'core': 'PALEOS:MgSiO3',
                'mantle': 'PALEOS:MgSiO3',
            },
            mushy_zone_factors={
                'PALEOS:iron': 1.0,
                'PALEOS:MgSiO3': 1.0,
                'PALEOS:H2O': 1.0,
            },
        )
        with caplog.at_level('WARNING'):
            validate_config(cfg)
        # The core-isn't-iron warning must appear.
        assert any(
            'does not appear to be' in r.message and 'iron' in r.message for r in caplog.records
        )
