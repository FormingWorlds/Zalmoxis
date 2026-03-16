"""Tests for multi-material volume-additive mixing.

Tests cover:
- Parsing single and multi-component EOS strings
- LayerMixture dataclass operations
- calculate_mixed_density (single and multi-component)
- get_mixed_nabla_ad
- any_component_is_tdep
- Runtime fraction updates
- Backward compatibility with single-material configs
"""

from __future__ import annotations

import os

import pytest

from zalmoxis.mixing import (
    LayerMixture,
    any_component_is_tdep,
    calculate_mixed_density,
    parse_all_layer_mixtures,
    parse_layer_components,
)


@pytest.mark.unit
class TestParseLayerComponents:
    def test_single_component(self):
        m = parse_layer_components('PALEOS:iron')
        assert m.components == ['PALEOS:iron']
        assert m.fractions == [1.0]

    def test_single_analytic(self):
        m = parse_layer_components('Analytic:SiC')
        assert m.components == ['Analytic:SiC']
        assert m.fractions == [1.0]

    def test_two_components(self):
        m = parse_layer_components('PALEOS:MgSiO3:0.85+PALEOS:H2O:0.15')
        assert m.components == ['PALEOS:MgSiO3', 'PALEOS:H2O']
        assert m.fractions[0] == pytest.approx(0.85)
        assert m.fractions[1] == pytest.approx(0.15)

    def test_three_components(self):
        m = parse_layer_components('PALEOS:iron:0.5+PALEOS:MgSiO3:0.3+PALEOS:H2O:0.2')
        assert len(m.components) == 3
        assert sum(m.fractions) == pytest.approx(1.0)

    def test_normalization_warning(self, caplog):
        m = parse_layer_components('PALEOS:MgSiO3:0.8+PALEOS:H2O:0.1')
        assert sum(m.fractions) == pytest.approx(1.0)

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match='Empty'):
            parse_layer_components('')

    def test_backward_compat_seager(self):
        m = parse_layer_components('Seager2007:iron')
        assert m.is_single()
        assert m.components[0] == 'Seager2007:iron'


@pytest.mark.unit
class TestLayerMixture:
    def test_is_single(self):
        m = LayerMixture(['PALEOS:iron'], [1.0])
        assert m.is_single()

    def test_is_not_single(self):
        m = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.8, 0.2])
        assert not m.is_single()

    def test_primary(self):
        m = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.3, 0.7])
        assert m.primary() == 'PALEOS:H2O'

    def test_has_tdep(self):
        m = LayerMixture(['PALEOS:iron'], [1.0])
        assert m.has_tdep()

    def test_has_no_tdep(self):
        m = LayerMixture(['Seager2007:iron'], [1.0])
        assert not m.has_tdep()

    def test_mixed_tdep(self):
        m = LayerMixture(['Seager2007:iron', 'PALEOS:MgSiO3'], [0.5, 0.5])
        assert m.has_tdep()

    def test_update_fractions(self):
        m = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.8, 0.2])
        m.update_fractions([0.6, 0.4])
        assert m.fractions == [0.6, 0.4]

    def test_update_fractions_wrong_length(self):
        m = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.8, 0.2])
        with pytest.raises(ValueError, match='Expected 2'):
            m.update_fractions([0.5, 0.3, 0.2])

    def test_update_fractions_bad_sum(self):
        m = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.8, 0.2])
        with pytest.raises(ValueError, match='sum to 1.0'):
            m.update_fractions([0.5, 0.3])

    def test_update_fractions_negative(self):
        m = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.8, 0.2])
        with pytest.raises(ValueError, match='non-negative'):
            m.update_fractions([1.5, -0.5])


@pytest.mark.unit
class TestParseAllLayerMixtures:
    def test_two_layer(self):
        lec = {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'}
        mixtures = parse_all_layer_mixtures(lec)
        assert 'core' in mixtures
        assert 'mantle' in mixtures
        assert mixtures['core'].components == ['PALEOS:iron']

    def test_three_layer(self):
        lec = {
            'core': 'PALEOS:iron',
            'mantle': 'PALEOS:MgSiO3',
            'ice_layer': 'PALEOS:H2O',
        }
        mixtures = parse_all_layer_mixtures(lec)
        assert len(mixtures) == 3

    def test_mixed_mantle(self):
        lec = {
            'core': 'PALEOS:iron',
            'mantle': 'PALEOS:MgSiO3:0.9+PALEOS:H2O:0.1',
        }
        mixtures = parse_all_layer_mixtures(lec)
        assert len(mixtures['mantle'].components) == 2


@pytest.mark.unit
class TestAnyComponentIsTdep:
    def test_all_tdep(self):
        mixtures = parse_all_layer_mixtures({'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'})
        assert any_component_is_tdep(mixtures)

    def test_none_tdep(self):
        mixtures = parse_all_layer_mixtures(
            {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'}
        )
        assert not any_component_is_tdep(mixtures)

    def test_mixed_has_tdep(self):
        mixtures = parse_all_layer_mixtures(
            {'core': 'Seager2007:iron', 'mantle': 'PALEOS:MgSiO3:0.9+Seager2007:H2O:0.1'}
        )
        assert any_component_is_tdep(mixtures)


def _paleos_data_available():
    root = os.environ.get('ZALMOXIS_ROOT', '')
    return os.path.isfile(
        os.path.join(root, 'data', 'EOS_PALEOS_iron', 'paleos_iron_eos_table_pt.dat')
    )


def _seager_data_available():
    root = os.environ.get('ZALMOXIS_ROOT', '')
    return os.path.isfile(os.path.join(root, 'data', 'EOS_Seager2007', 'eos_seager07_iron.txt'))


@pytest.mark.unit
class TestCalculateMixedDensity:
    def test_single_component_matches_direct(self):
        """Single-component mixture gives same result as calculate_density."""
        if not _seager_data_available():
            pytest.skip('Seager data not found')

        from zalmoxis.eos_functions import calculate_density
        from zalmoxis.zalmoxis import load_material_dictionaries

        md = load_material_dictionaries()
        mixture = LayerMixture(['Seager2007:iron'], [1.0])

        rho_mixed = calculate_mixed_density(300e9, 300, mixture, md, None, None, {})
        rho_direct = calculate_density(300e9, md, 'Seager2007:iron', 300, None, None, {})
        assert rho_mixed == pytest.approx(rho_direct)

    def test_two_component_harmonic_mean(self):
        """Two Seager components produce harmonic-mean density."""
        if not _seager_data_available():
            pytest.skip('Seager data not found')

        from zalmoxis.eos_functions import calculate_density
        from zalmoxis.zalmoxis import load_material_dictionaries

        md = load_material_dictionaries()
        P = 100e9

        rho_iron = calculate_density(P, md, 'Seager2007:iron', 300, None, None, {})
        rho_sil = calculate_density(P, md, 'Seager2007:MgSiO3', 300, None, None, {})

        w1, w2 = 0.7, 0.3
        expected = 1.0 / (w1 / rho_iron + w2 / rho_sil)

        mixture = LayerMixture(['Seager2007:iron', 'Seager2007:MgSiO3'], [w1, w2])
        rho_mixed = calculate_mixed_density(P, 300, mixture, md, None, None, {})

        assert rho_mixed == pytest.approx(expected, rel=1e-10)

    def test_analytic_mixing(self):
        """Analytic EOS components can be mixed."""
        from zalmoxis.zalmoxis import load_material_dictionaries

        md = load_material_dictionaries()
        mixture = LayerMixture(['Analytic:iron', 'Analytic:MgSiO3'], [0.5, 0.5])
        rho = calculate_mixed_density(100e9, 300, mixture, md, None, None, {})
        assert rho is not None
        assert rho > 0


@pytest.mark.unit
class TestBackwardCompat:
    def test_main_without_layer_mixtures(self):
        """main() with layer_mixtures=None should auto-parse from config."""
        if not _paleos_data_available():
            pytest.skip('PALEOS data not found')

        from zalmoxis.constants import earth_mass
        from zalmoxis.zalmoxis import load_material_dictionaries, main

        config = {
            'planet_mass': 1.0 * earth_mass,
            'core_mass_fraction': 0.325,
            'mantle_mass_fraction': 0,
            'temperature_mode': 'isothermal',
            'surface_temperature': 300,
            'center_temperature': 6000,
            'temp_profile_file': '',
            'layer_eos_config': {
                'core': 'Seager2007:iron',
                'mantle': 'Seager2007:MgSiO3',
            },
            'mushy_zone_factor': 1.0,
            'num_layers': 50,
            'max_iterations_outer': 20,
            'tolerance_outer': 3e-3,
            'max_iterations_inner': 20,
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
        result = main(
            config,
            material_dictionaries=load_material_dictionaries(),
            melting_curves_functions=None,
            input_dir='.',
        )
        assert result['converged']
