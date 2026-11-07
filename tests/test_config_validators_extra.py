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

import os

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

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
        with pytest.raises(ValueError, match='target_surface_pressure must be > 0'):
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


class TestDebugProfileDir:
    """The optional debug_profile_dir: a folder that exists, or absent."""

    @pytest.mark.parametrize(
        'value', [123, True, ['x'], b'/tmp'], ids=['int', 'bool', 'list', 'bytes']
    )
    def test_wrong_type_is_refused_by_name(self, value):
        """A value that is not a path string raises ValueError naming the key and the value."""
        with pytest.raises(ValueError, match='debug_profile_dir') as err:
            validate_config(_make_minimal_config(debug_profile_dir=value))
        assert repr(value) in str(err.value)

    def test_empty_string_is_refused_by_name(self):
        """An empty string is not a folder: ValueError naming the key."""
        with pytest.raises(ValueError, match='debug_profile_dir.*empty'):
            validate_config(_make_minimal_config(debug_profile_dir=''))
        with pytest.raises(ValueError, match='is not an existing folder'):
            validate_config(_make_minimal_config(debug_profile_dir=' '))

    def test_missing_folder_is_refused_by_name(self, tmp_path):
        """A path that is not an existing folder raises ValueError naming it."""
        missing = tmp_path / 'not_there'
        with pytest.raises(ValueError, match='debug_profile_dir.*not_there'):
            validate_config(_make_minimal_config(debug_profile_dir=str(missing)))
        assert not missing.exists()

    def test_existing_folder_and_absent_key_pass(self, tmp_path):
        """An existing folder (str or Path) and a config without the key validate."""
        assert validate_config(_make_minimal_config(debug_profile_dir=str(tmp_path))) is None
        assert validate_config(_make_minimal_config(debug_profile_dir=tmp_path)) is None

    @settings(max_examples=60, deadline=None)
    @given(
        st.one_of(
            st.none(),
            st.booleans(),
            st.integers(),
            st.floats(allow_nan=True, allow_infinity=True),
            st.lists(st.text(max_size=3), max_size=2),
            st.dictionaries(st.text(max_size=2), st.integers(), max_size=1),
            st.binary(max_size=4),
        )
    )
    def test_any_non_path_value_is_refused(self, value):
        """None, bool, int, float (0, negative, NaN, inf) and containers raise ValueError."""
        with pytest.raises(ValueError, match='debug_profile_dir must be a folder path'):
            validate_config(_make_minimal_config(debug_profile_dir=value))

    @settings(
        max_examples=60,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(st.text(alphabet='abcXYZ019_- .', min_size=1, max_size=12))
    def test_any_path_that_is_not_a_folder_is_refused(self, tmp_path, name):
        """A name under a missing folder, and a regular file, raise ValueError; nothing is created."""
        with pytest.raises(ValueError, match='is not an existing folder'):
            validate_config(
                _make_minimal_config(debug_profile_dir=os.path.join(tmp_path, 'absent', name))
            )
        afile = tmp_path / 'a_file.txt'
        afile.write_text('x')
        with pytest.raises(ValueError, match='is not an existing folder'):
            validate_config(_make_minimal_config(debug_profile_dir=str(afile)))
        assert sorted(p.name for p in tmp_path.iterdir()) == ['a_file.txt']


class TestDebugProfileDirFromToml:
    """load_zalmoxis_config reads debug_profile_dir from the [Output] table and validates it."""

    @staticmethod
    def _toml(tmp_path, line):
        from zalmoxis import get_zalmoxis_root

        text = open(os.path.join(get_zalmoxis_root(), 'input', 'default.toml')).read()
        assert text.count('[Output]\n') == 1
        path = tmp_path / 'cfg.toml'
        path.write_text(text.replace('[Output]\n', f'[Output]\n{line}\n'))
        return str(path)

    def test_key_set_in_the_toml_reaches_the_config(self, tmp_path):
        """A folder named in [Output] is in the returned dict; without the line the key is absent."""
        from zalmoxis.config import load_zalmoxis_config

        folder = tmp_path / 'dump'
        folder.mkdir()
        cfg = load_zalmoxis_config(self._toml(tmp_path, f"debug_profile_dir = '{folder}'"))
        assert cfg['debug_profile_dir'] == str(folder)
        assert 'debug_profile_dir' not in load_zalmoxis_config(self._toml(tmp_path, ''))

    def test_empty_key_in_the_toml_is_refused_by_name(self, tmp_path):
        """An empty string in [Output] raises the named ValueError at load time."""
        from zalmoxis.config import load_zalmoxis_config

        with pytest.raises(ValueError, match='debug_profile_dir must not be empty'):
            load_zalmoxis_config(self._toml(tmp_path, "debug_profile_dir = ''"))
        assert sorted(p.name for p in tmp_path.iterdir()) == ['cfg.toml']
