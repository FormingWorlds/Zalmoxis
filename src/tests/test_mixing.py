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
    _condensed_weight,
    any_component_is_tdep,
    calculate_mixed_density,
    get_mixed_nabla_ad,
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
        import logging

        with caplog.at_level(logging.WARNING, logger='zalmoxis.mixing'):
            m = parse_layer_components('PALEOS:MgSiO3:0.8+PALEOS:H2O:0.1')
        assert sum(m.fractions) == pytest.approx(1.0)
        assert 'normalizing' in caplog.text.lower()

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

    def test_post_init_empty_components(self):
        with pytest.raises(ValueError, match='at least one'):
            LayerMixture([], [])

    def test_post_init_mismatched_lengths(self):
        with pytest.raises(ValueError, match='same length'):
            LayerMixture(['PALEOS:iron'], [0.5, 0.5])

    def test_post_init_bad_fraction_sum(self):
        with pytest.raises(ValueError, match='sum to'):
            LayerMixture(['PALEOS:iron', 'PALEOS:MgSiO3'], [0.4, 0.4])

    def test_post_init_single_bad_fraction(self):
        with pytest.raises(ValueError, match='sum to'):
            LayerMixture(['PALEOS:iron'], [0.5])


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
        """Two Seager components produce harmonic-mean density.

        All-condensed limit: both iron (~12000 kg/m^3) and silicate (~5000 kg/m^3)
        are far above the sigmoid center (300 kg/m^3), so sigma ~1.0 to float64
        precision. The suppressed harmonic mean exactly recovers the standard one.
        """
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


@pytest.mark.unit
class TestCondensedWeight:
    """Tests for the sigmoid suppression function used in phase-aware mixing."""

    @pytest.mark.parametrize(
        'rho, expected_min, expected_max',
        [
            (10.0, 0.0, 0.01),  # vapor: near zero
            (100.0, 0.0, 0.03),  # light supercritical: near zero
            (300.0, 0.45, 0.55),  # critical density: ~0.5 (sigmoid center)
            (500.0, 0.97, 1.0),  # dense supercritical: near 1
            (1000.0, 0.999, 1.0),  # liquid: essentially 1
            (5000.0, 0.999, 1.0),  # rock: essentially 1
        ],
    )
    def test_condensed_weight_values(self, rho, expected_min, expected_max):
        """Sigmoid returns expected values at key densities."""
        sigma = _condensed_weight(rho, rho_min=300.0, rho_scale=50.0)
        assert expected_min <= sigma <= expected_max, (
            f'_condensed_weight({rho}) = {sigma}, expected in [{expected_min}, {expected_max}]'
        )

    def test_condensed_weight_zero_density(self):
        """Zero density gives near-zero weight."""
        sigma = _condensed_weight(0.0)
        assert sigma < 0.01

    def test_condensed_weight_large_density(self):
        """Very large density gives weight ~1."""
        sigma = _condensed_weight(1e6)
        assert sigma == pytest.approx(1.0, abs=1e-10)

    def test_condensed_weight_negative_density(self):
        """Negative density gives near-zero weight (no crash)."""
        sigma = _condensed_weight(-100.0)
        assert sigma < 0.01

    def test_condensed_weight_monotonic(self):
        """Weight is monotonically increasing with density."""
        rhos = [1, 10, 50, 100, 200, 300, 500, 1000, 5000]
        sigmas = [_condensed_weight(r) for r in rhos]
        for i in range(len(sigmas) - 1):
            assert sigmas[i] < sigmas[i + 1]

    def test_custom_rho_min_and_scale(self):
        """Custom parameters shift the sigmoid."""
        # With rho_min=30 (H2 envelope), sigma at 30 should be ~0.5
        sigma_at_center = _condensed_weight(30.0, rho_min=30.0, rho_scale=5.0)
        assert sigma_at_center == pytest.approx(0.5, abs=0.01)
        # Well above center should be ~1
        sigma_high = _condensed_weight(100.0, rho_min=30.0, rho_scale=5.0)
        assert sigma_high > 0.99


@pytest.mark.unit
class TestVaporSuppression:
    """Tests for phase-aware suppression in density and nabla_ad mixing."""

    def test_mixed_density_vapor_suppression(self):
        """Vapor component is suppressed in two-component harmonic mean."""
        from unittest.mock import patch

        # Mock calculate_density to return rock=4000, water=50 kg/m^3
        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            if 'MgSiO3' in eos:
                return 4000.0
            if 'H2O' in eos:
                return 50.0
            return None

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])

        # calculate_density is imported inside the function body in mixing.py
        # to break a circular import. Patching 'zalmoxis.eos_functions.calculate_density'
        # works because the deferred import re-reads from the module's attribute dict,
        # which is already replaced by the patch.
        with patch('zalmoxis.eos_functions.calculate_density', side_effect=mock_density):
            rho = calculate_mixed_density(
                1e9,
                3000,
                mixture,
                {},
                None,
                None,
                {},
                condensed_rho_min=322.0,
                condensed_rho_scale=50.0,
            )

        assert rho is not None
        # With suppression, water at 50 kg/m^3 is almost fully suppressed
        # Result should be close to pure rock density (4000), not 311 (unsuppressed)
        assert rho > 3000, f'Expected rho > 3000 (near rock), got {rho}'

    def test_mixed_density_all_condensed(self):
        """All-condensed mixture recovers standard harmonic mean."""
        from unittest.mock import patch

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            if 'MgSiO3' in eos:
                return 4000.0
            if 'H2O' in eos:
                return 1000.0
            return None

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])
        w1, w2 = 0.85, 0.15
        rho1, rho2 = 4000.0, 1000.0

        with patch('zalmoxis.eos_functions.calculate_density', side_effect=mock_density):
            rho = calculate_mixed_density(
                1e11,
                3000,
                mixture,
                {},
                None,
                None,
                {},
                condensed_rho_min=322.0,
                condensed_rho_scale=50.0,
            )

        # Both at high density: sigma ~1.0, so result should match standard harmonic mean
        # sigma for 4000 and 1000 are both > 0.999
        sigma1 = _condensed_weight(rho1)
        sigma2 = _condensed_weight(rho2)
        w_eff1 = w1 * sigma1
        w_eff2 = w2 * sigma2
        expected = (w_eff1 + w_eff2) / (w_eff1 / rho1 + w_eff2 / rho2)
        standard = 1.0 / (w1 / rho1 + w2 / rho2)

        assert rho == pytest.approx(expected, rel=1e-6)
        # And the suppressed result is within 0.1% of the standard harmonic mean
        assert rho == pytest.approx(standard, rel=1e-3)

    def test_mixed_density_all_vapor_returns_small_density(self):
        """All components below condensation threshold return heavily suppressed density."""
        from unittest.mock import patch

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            return 10.0  # vapor-like for all

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])

        with patch('zalmoxis.eos_functions.calculate_density', side_effect=mock_density):
            rho = calculate_mixed_density(
                1e5,
                5000,
                mixture,
                {},
                None,
                None,
                {},
                condensed_rho_min=322.0,
                condensed_rho_scale=50.0,
            )

        # sigma at rho=10 is ~0.003, so w_eff is tiny but not zero.
        # The function should still return a value (not None) since w_eff > 0.
        # This tests the edge case; the actual value is ~10 kg/m^3.
        assert rho is not None

    def test_single_component_no_suppression(self):
        """Single-component layers bypass suppression entirely."""
        from unittest.mock import patch

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            return 50.0  # vapor-like

        mixture = LayerMixture(['PALEOS:H2O'], [1.0])

        with patch('zalmoxis.eos_functions.calculate_density', side_effect=mock_density):
            rho = calculate_mixed_density(
                1e5,
                3000,
                mixture,
                {},
                None,
                None,
                {},
                condensed_rho_min=322.0,
                condensed_rho_scale=50.0,
            )

        # Single-component fast path: returns raw density, no suppression
        assert rho == pytest.approx(50.0)

    def test_mixed_nabla_ad_vapor_suppression(self):
        """Vapor component's nabla_ad is suppressed in weighted average."""
        from unittest.mock import patch

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            if 'MgSiO3' in eos:
                return 4000.0
            if 'H2O' in eos:
                return 50.0
            return None

        def mock_nabla(eos, P, T, md, interp, sf, lf):
            if 'MgSiO3' in eos:
                return 0.3  # rock nabla_ad
            if 'H2O' in eos:
                return 0.9  # vapor nabla_ad (unrealistically large)
            return None

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])

        with (
            patch('zalmoxis.eos_functions.calculate_density', side_effect=mock_density),
            patch('zalmoxis.mixing._nabla_ad_for_component', side_effect=mock_nabla),
        ):
            nabla = get_mixed_nabla_ad(
                1e9,
                3000,
                mixture,
                {},
                {},
                mushy_zone_factor=1.0,
                condensed_rho_min=322.0,
                condensed_rho_scale=50.0,
            )

        assert nabla is not None
        # With suppression, water's nabla_ad=0.9 is nearly excluded
        # Result should be close to rock's 0.3
        assert nabla < 0.35, f'Expected nabla near 0.3 (rock), got {nabla}'
        assert nabla > 0.25
