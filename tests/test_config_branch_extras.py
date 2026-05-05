"""Branch tests for ``zalmoxis.config`` validation paths the existing
config_validation suite does not exercise.

Covers:
- ``validate_layer_eos`` accepts a valid Vinet:<material> entry and
  loops past it (the success-path ``continue`` at line 175).
- ``validate_config``'s T-dep / T-indep mix warning emits when a
  layer mixes a tabulated PALEOS-2phase EOS with a tabulated
  Seager2007 EOS (lines 539-545).
- The non-iron core component warning emits when the core uses a
  silicate EOS (line 573).
- The H2O-mantle path's "empty layer string" continue (line 596).
- The Chabrier:H pure-mantle ValueError (line 628 region) - exercised
  by the rejection branch.
- The H2 fraction warning at >20 % H2 by mass.
"""

from __future__ import annotations

import logging
from copy import deepcopy

import pytest

from zalmoxis.config import validate_config, validate_layer_eos

pytestmark = pytest.mark.unit


def _base_config(**overrides):
    """Minimal config for validate_config calls."""
    cfg = {
        'planet_mass': 5.972e24,
        'core_mass_fraction': 0.325,
        'mantle_mass_fraction': 0,
        'temperature_mode': 'linear',
        'surface_temperature': 3000,
        'center_temperature': 6000,
        'temp_profile_file': '',
        'mushy_zone_factor': 1.0,
        'num_layers': 50,
        'target_surface_pressure': 101325,
        'data_output_enabled': False,
        'plotting_enabled': False,
    }
    cfg.update(overrides)
    return cfg


class TestValidateLayerEosVinetSuccess:
    """Valid Vinet:<material> identifier is accepted (continue at line 175)."""

    def test_valid_vinet_iron_passes(self):
        validate_layer_eos({'core': 'Vinet:iron', 'mantle': 'Vinet:MgSiO3'})

    def test_invalid_vinet_material_raises(self):
        """A Vinet:<unknown_key> entry raises with the valid-keys hint."""
        with pytest.raises(ValueError, match='Invalid Vinet material'):
            validate_layer_eos({'core': 'Vinet:tin'})


class TestValidateConfigMixWarnings:
    """Warning emission paths in validate_config."""

    def test_tdep_tindep_mix_warning(self, caplog):
        """Tabulated T-dep + tabulated T-indep in one layer triggers a
        physical-consistency warning."""
        cfg = _base_config(
            rock_solidus='Monteux16-solidus',
            rock_liquidus='Monteux16-liquidus-A-chondritic',
            layer_eos_config={
                'core': 'Seager2007:iron',
                'mantle': 'PALEOS-2phase:MgSiO3:0.7+Seager2007:H2O:0.3',
            },
        )
        with caplog.at_level(logging.WARNING, logger='zalmoxis.config'):
            validate_config(cfg)
        assert any(
            'T-dependent' in r.message and 'T-independent' in r.message for r in caplog.records
        )

    def test_non_iron_core_warning(self, caplog):
        """Core configured with a tabulated silicate EOS triggers the
        non-iron-core physical-implausibility warning. (Analytic: and
        Vinet: cores are exempted from the warning.)"""
        cfg = _base_config(
            layer_eos_config={
                'core': 'Seager2007:MgSiO3',  # tabulated silicate, not iron
                'mantle': 'Seager2007:MgSiO3',
            }
        )
        with caplog.at_level(logging.WARNING, logger='zalmoxis.config'):
            validate_config(cfg)
        assert any('Core EOS' in r.message and 'iron' in r.message for r in caplog.records)

    def test_h2_high_fraction_warning(self, caplog):
        """Mantle with >20% H2 by mass triggers the validation-range warning."""
        cfg = _base_config(
            layer_eos_config={
                'core': 'Seager2007:iron',
                'mantle': 'PALEOS:MgSiO3:0.7+Chabrier:H:0.3',
            }
        )
        with caplog.at_level(logging.WARNING, logger='zalmoxis.config'):
            validate_config(cfg)
        assert any('H2' in r.message and 'validated' in r.message for r in caplog.records)

    def test_chabrier_h_pure_mantle_raises(self):
        """Pure Chabrier:H mantle is rejected (no condensed anchor)."""
        cfg = _base_config(
            layer_eos_config={
                'core': 'Seager2007:iron',
                'mantle': 'Chabrier:H',
            }
        )
        with pytest.raises(ValueError, match='Pure Chabrier:H'):
            validate_config(cfg)

    def test_h2o_dominated_mantle_raises_at_adiabatic_t(self):
        """Mantle with >50% H2O and no silicate raises at adiabatic temperature."""
        cfg = _base_config(
            temperature_mode='adiabatic',
            layer_eos_config={
                'core': 'Seager2007:iron',
                'mantle': 'PALEOS:H2O',  # 100% H2O, no silicate
            },
        )
        with pytest.raises(ValueError, match='H2O'):
            validate_config(cfg)

    def test_h2o_dominated_mantle_allowed_at_isothermal_t(self):
        """Same composition is allowed under isothermal mode (no vapor risk
        in the volume-additive mixing model when T < critical)."""
        cfg = _base_config(
            temperature_mode='isothermal',
            surface_temperature=500.0,  # below H2O critical (647 K)
            layer_eos_config={
                'core': 'Seager2007:iron',
                'mantle': 'PALEOS:H2O',
            },
        )
        # No raise expected
        validate_config(cfg)

    def test_h2o_above_30pct_warning(self, caplog):
        """Mantle with 30 < H2O < 50% triggers the validated-range warning
        without raising."""
        cfg = deepcopy(
            _base_config(
                layer_eos_config={
                    'core': 'Seager2007:iron',
                    'mantle': 'PALEOS:MgSiO3:0.6+PALEOS:H2O:0.4',
                }
            )
        )
        with caplog.at_level(logging.WARNING, logger='zalmoxis.config'):
            validate_config(cfg)
        assert any('40% H2O' in r.message for r in caplog.records)
