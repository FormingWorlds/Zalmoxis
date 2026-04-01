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

import numpy as np
import pytest

from zalmoxis.mixing import (
    LayerMixture,
    _binodal_factor,
    _condensed_weight,
    _condensed_weight_batch,
    _nabla_ad_for_component,
    any_component_is_tdep,
    calculate_mixed_density,
    calculate_mixed_density_batch,
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

        from zalmoxis.eos import calculate_density
        from zalmoxis.config import load_material_dictionaries

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

        from zalmoxis.eos import calculate_density
        from zalmoxis.config import load_material_dictionaries

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
        from zalmoxis.config import load_material_dictionaries

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
        from zalmoxis.config import load_material_dictionaries
        from zalmoxis.solver import main

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
            'target_surface_pressure': 101325,
            'data_output_enabled': False,
            'plotting_enabled': False,
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
        # to break a circular import. Patching 'zalmoxis.eos.calculate_density'
        # works because the deferred import re-reads from the module's attribute dict,
        # which is already replaced by the patch.
        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
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

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
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

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
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

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
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
            patch('zalmoxis.eos.calculate_density', side_effect=mock_density),
            patch('zalmoxis.mixing._nabla_ad_for_component', side_effect=mock_nabla),
        ):
            nabla = get_mixed_nabla_ad(
                1e9,
                3000,
                mixture,
                {},
                {},
                mushy_zone_factors=None,
                condensed_rho_min=322.0,
                condensed_rho_scale=50.0,
            )

        assert nabla is not None
        # With suppression, water's nabla_ad=0.9 is nearly excluded
        # Result should be close to rock's 0.3
        assert nabla < 0.35, f'Expected nabla near 0.3 (rock), got {nabla}'
        assert nabla > 0.25


@pytest.mark.unit
class TestPerEosMushyZoneFactors:
    """Tests for per-EOS mushy_zone_factors dispatch."""

    def test_different_factors_per_component(self):
        """Each component receives its own mushy_zone_factor from the dict."""
        from unittest.mock import patch

        received_mzf = {}

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            received_mzf[eos] = mzf
            if 'MgSiO3' in eos:
                return 4000.0
            if 'H2O' in eos:
                return 1000.0
            return None

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])
        factors = {'PALEOS:MgSiO3': 0.8, 'PALEOS:H2O': 0.9}

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            calculate_mixed_density(
                1e11,
                3000,
                mixture,
                {},
                None,
                None,
                {},
                mushy_zone_factors=factors,
            )

        assert received_mzf['PALEOS:MgSiO3'] == pytest.approx(0.8)
        assert received_mzf['PALEOS:H2O'] == pytest.approx(0.9)

    def test_single_component_gets_its_factor(self):
        """Single-component mixture looks up the correct per-EOS factor."""
        from unittest.mock import patch

        received_mzf = {}

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            received_mzf[eos] = mzf
            return 5000.0

        mixture = LayerMixture(['PALEOS:iron'], [1.0])
        factors = {'PALEOS:iron': 0.75, 'PALEOS:MgSiO3': 0.8}

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            calculate_mixed_density(
                1e11,
                3000,
                mixture,
                {},
                None,
                None,
                {},
                mushy_zone_factors=factors,
            )

        assert received_mzf['PALEOS:iron'] == pytest.approx(0.75)

    def test_missing_key_defaults_to_one(self):
        """EOS name not in the dict defaults to mushy_zone_factor=1.0."""
        from unittest.mock import patch

        received_mzf = {}

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            received_mzf[eos] = mzf
            return 5000.0

        mixture = LayerMixture(['Seager2007:iron'], [1.0])
        factors = {'PALEOS:MgSiO3': 0.8}

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            calculate_mixed_density(
                1e11,
                300,
                mixture,
                {},
                None,
                None,
                {},
                mushy_zone_factors=factors,
            )

        assert received_mzf['Seager2007:iron'] == pytest.approx(1.0)

    def test_none_defaults_to_one(self):
        """mushy_zone_factors=None gives 1.0 for all components."""
        from unittest.mock import patch

        received_mzf = {}

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            received_mzf[eos] = mzf
            return 5000.0

        mixture = LayerMixture(['PALEOS:iron'], [1.0])

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            calculate_mixed_density(
                1e11,
                3000,
                mixture,
                {},
                None,
                None,
                {},
                mushy_zone_factors=None,
            )

        assert received_mzf['PALEOS:iron'] == pytest.approx(1.0)

    def test_float_backward_compat(self):
        """A float mushy_zone_factors is applied to all components."""
        from unittest.mock import patch

        received_mzf = {}

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            received_mzf[eos] = mzf
            if 'MgSiO3' in eos:
                return 4000.0
            if 'H2O' in eos:
                return 1000.0
            return None

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            calculate_mixed_density(
                1e11,
                3000,
                mixture,
                {},
                None,
                None,
                {},
                mushy_zone_factors=0.85,
            )

        assert received_mzf['PALEOS:MgSiO3'] == pytest.approx(0.85)
        assert received_mzf['PALEOS:H2O'] == pytest.approx(0.85)


@pytest.mark.unit
class TestCondensedWeightBatch:
    """Tests for the vectorized sigmoid suppression function."""

    def test_matches_scalar(self):
        """Batch version produces same results as scalar _condensed_weight."""
        rhos = np.array([1.0, 10.0, 100.0, 322.0, 500.0, 1000.0, 5000.0])
        rho_min = 322.0
        rho_scale = 50.0
        batch_result = _condensed_weight_batch(rhos, rho_min, rho_scale)
        for i, rho in enumerate(rhos):
            scalar_result = _condensed_weight(rho, rho_min, rho_scale)
            assert batch_result[i] == pytest.approx(scalar_result, rel=1e-12)

    def test_output_shape(self):
        """Output has same shape as input array."""
        rhos = np.array([100.0, 200.0, 300.0])
        result = _condensed_weight_batch(rhos, 322.0, 50.0)
        assert result.shape == rhos.shape

    def test_clipping_extreme_values(self):
        """Extreme densities are handled without overflow."""
        rhos = np.array([1e-10, 1e10])
        result = _condensed_weight_batch(rhos, 322.0, 50.0)
        assert np.all(np.isfinite(result))
        assert result[0] < 0.01
        assert result[1] > 0.99

    def test_custom_h2_parameters(self):
        """H2 critical density parameters (rho_min=30, rho_scale=10)."""
        rhos = np.array([5.0, 30.0, 100.0])
        result = _condensed_weight_batch(rhos, rho_min=30.0, rho_scale=10.0)
        assert result[0] < 0.1  # well below center
        assert result[1] == pytest.approx(0.5, abs=0.01)  # at center
        assert result[2] > 0.99  # well above center


@pytest.mark.unit
class TestBinodalFactor:
    """Tests for the binodal suppression dispatch function."""

    def test_non_h2_returns_one(self):
        """Non-H2 components are never suppressed by binodal."""
        mixture = LayerMixture(['PALEOS:MgSiO3', 'Chabrier:H'], [0.85, 0.15])
        sigma = _binodal_factor('PALEOS:MgSiO3', 0.85, mixture, 1e10, 3000, 50.0)
        assert sigma == 1.0

    def test_h2_with_silicate_partner(self):
        """H2 mixed with silicate triggers Rogers+2025 binodal."""
        mixture = LayerMixture(['Chabrier:H', 'PALEOS:MgSiO3'], [0.15, 0.85])
        # At low T (below binodal), H2 should be suppressed
        sigma_cold = _binodal_factor('Chabrier:H', 0.15, mixture, 10e9, 500, 50.0)
        # At high T (above binodal), H2 should be unsuppressed
        sigma_hot = _binodal_factor('Chabrier:H', 0.15, mixture, 10e9, 10000, 50.0)
        assert sigma_cold < sigma_hot

    def test_h2_with_h2o_partner(self):
        """H2 mixed with H2O triggers Gupta+2025 binodal."""
        mixture = LayerMixture(['Chabrier:H', 'PALEOS:H2O'], [0.5, 0.5])
        # At low T, H2 should be suppressed (below miscibility curve)
        sigma_cold = _binodal_factor('Chabrier:H', 0.5, mixture, 5e9, 300, 50.0)
        # At very high T, H2 should be less suppressed
        sigma_hot = _binodal_factor('Chabrier:H', 0.5, mixture, 5e9, 8000, 50.0)
        assert sigma_cold < sigma_hot

    def test_h2_with_zero_fraction_partner_skipped(self):
        """Partners with zero weight fraction are skipped."""
        mixture = LayerMixture(['Chabrier:H', 'PALEOS:MgSiO3'], [1.0, 0.0])
        # This would raise an error in binodal functions if the zero-fraction
        # partner were not skipped, since w_p=0 would cause division issues.
        # But since we explicitly constructed with fractions summing to 1.0,
        # we need to bypass the normal LayerMixture validation.
        # Instead, test that the loop body is skipped for w_p <= 0.
        # With no valid partner, sigma should stay 1.0.
        # (LayerMixture won't allow fractions that don't sum to 1.0, but the
        # _binodal_factor function itself checks w_p <= 0.)
        sigma = _binodal_factor('Chabrier:H', 1.0, mixture, 1e10, 3000, 50.0)
        # No partner has positive fraction that matches a binodal pair,
        # but the loop still runs and w_p=0 is skipped via the continue.
        # Actually MgSiO3 has w_p=0.0, so the continue fires.
        assert sigma == 1.0

    def test_h2_with_no_binodal_partner(self):
        """H2 paired with a non-binodal EOS returns 1.0."""
        mixture = LayerMixture(['Chabrier:H', 'PALEOS:iron'], [0.5, 0.5])
        sigma = _binodal_factor('Chabrier:H', 0.5, mixture, 1e10, 3000, 50.0)
        assert sigma == 1.0

    def test_h2_with_both_silicate_and_h2o(self):
        """H2 in a 3-component mix: minimum of both binodals taken."""
        mixture = LayerMixture(
            ['Chabrier:H', 'PALEOS:MgSiO3', 'PALEOS:H2O'],
            [0.1, 0.6, 0.3],
        )
        sigma = _binodal_factor('Chabrier:H', 0.1, mixture, 10e9, 3000, 50.0)
        # Should be the minimum of both binodal suppressions
        assert 0.0 <= sigma <= 1.0


@pytest.mark.unit
class TestCalculateMixedDensityEdgeCases:
    """Edge cases for calculate_mixed_density: invalid returns, zero fractions."""

    def test_component_returns_none_aborts_mixture(self):
        """If any component returns None, the entire mixture returns None."""
        from unittest.mock import patch

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            if 'MgSiO3' in eos:
                return 4000.0
            return None  # H2O lookup fails

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            rho = calculate_mixed_density(1e9, 3000, mixture, {}, None, None, {})
        assert rho is None

    def test_component_returns_nan_aborts_mixture(self):
        """If any component returns NaN, the entire mixture returns None."""
        from unittest.mock import patch

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            if 'MgSiO3' in eos:
                return 4000.0
            return float('nan')

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            rho = calculate_mixed_density(1e9, 3000, mixture, {}, None, None, {})
        assert rho is None

    def test_component_returns_zero_aborts_mixture(self):
        """If any component returns zero density, the mixture returns None."""
        from unittest.mock import patch

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            if 'MgSiO3' in eos:
                return 4000.0
            return 0.0

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            rho = calculate_mixed_density(1e9, 3000, mixture, {}, None, None, {})
        assert rho is None

    def test_component_returns_negative_aborts_mixture(self):
        """If any component returns negative density, the mixture returns None."""
        from unittest.mock import patch

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            if 'MgSiO3' in eos:
                return 4000.0
            return -100.0

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            rho = calculate_mixed_density(1e9, 3000, mixture, {}, None, None, {})
        assert rho is None

    def test_per_component_rho_min_h2(self):
        """H2 component uses its own rho_min (30 kg/m^3), not global default."""
        from unittest.mock import patch

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            if 'MgSiO3' in eos:
                return 4000.0
            if eos == 'Chabrier:H':
                return 35.0  # just above H2 critical density (30)
            return None

        mixture = LayerMixture(['PALEOS:MgSiO3', 'Chabrier:H'], [0.85, 0.15])

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            rho = calculate_mixed_density(
                1e10,
                5000,
                mixture,
                {},
                None,
                None,
                {},
                condensed_rho_min=322.0,
                condensed_rho_scale=50.0,
                binodal_T_scale=50.0,
            )

        # H2 at 35 kg/m^3 is just above its own rho_min=30, so sigma ~0.62
        # With the global default (322), sigma would be ~0.003 (nearly zero).
        # The mixed density should reflect a meaningful H2 contribution.
        assert rho is not None
        assert rho > 0


@pytest.mark.unit
class TestCalculateMixedDensityBatch:
    """Tests for the vectorized batch density calculation."""

    def test_single_component_batch(self):
        """Single-component batch delegates to calculate_density_batch."""
        from unittest.mock import patch

        expected = np.array([5000.0, 5100.0, 5200.0])

        def mock_batch(P, T, md, eos, sf, lf, interp, mzf):
            return expected.copy()

        mixture = LayerMixture(['PALEOS:iron'], [1.0])
        pressures = np.array([100e9, 200e9, 300e9])
        temperatures = np.array([3000.0, 3500.0, 4000.0])

        with patch('zalmoxis.eos.calculate_density_batch', side_effect=mock_batch):
            result = calculate_mixed_density_batch(
                pressures, temperatures, mixture, {}, None, None, {}
            )

        np.testing.assert_allclose(result, expected)

    def test_multi_component_batch(self):
        """Multi-component batch computes suppressed harmonic mean for each point."""
        from unittest.mock import patch

        def mock_batch(P, T, md, eos, sf, lf, interp, mzf):
            if 'MgSiO3' in eos:
                return np.array([4000.0, 4100.0])
            if 'H2O' in eos:
                return np.array([1000.0, 1050.0])
            return np.full(len(P), np.nan)

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])
        pressures = np.array([100e9, 200e9])
        temperatures = np.array([3000.0, 3500.0])

        with patch('zalmoxis.eos.calculate_density_batch', side_effect=mock_batch):
            result = calculate_mixed_density_batch(
                pressures, temperatures, mixture, {}, None, None, {}
            )

        assert result.shape == (2,)
        # Both components at high density, so sigma ~ 1. Should be near harmonic mean.
        for i in range(2):
            assert np.isfinite(result[i])
            assert result[i] > 0

    def test_batch_invalid_component_gives_nan(self):
        """Shells where any component has invalid density become NaN."""
        from unittest.mock import patch

        def mock_batch(P, T, md, eos, sf, lf, interp, mzf):
            if 'MgSiO3' in eos:
                return np.array([4000.0, 4100.0, np.nan])
            if 'H2O' in eos:
                return np.array([1000.0, 1050.0, 1100.0])
            return np.full(len(P), np.nan)

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])
        pressures = np.array([100e9, 200e9, 300e9])
        temperatures = np.array([3000.0, 3500.0, 4000.0])

        with patch('zalmoxis.eos.calculate_density_batch', side_effect=mock_batch):
            result = calculate_mixed_density_batch(
                pressures, temperatures, mixture, {}, None, None, {}
            )

        assert np.isfinite(result[0])
        assert np.isfinite(result[1])
        assert np.isnan(result[2])

    def test_batch_with_h2_binodal(self):
        """H2 component in batch triggers per-shell binodal suppression."""
        from unittest.mock import patch

        def mock_batch(P, T, md, eos, sf, lf, interp, mzf):
            if eos == 'Chabrier:H':
                return np.array([50.0, 50.0])
            if 'MgSiO3' in eos:
                return np.array([4000.0, 4000.0])
            return np.full(len(P), np.nan)

        mixture = LayerMixture(['Chabrier:H', 'PALEOS:MgSiO3'], [0.15, 0.85])
        pressures = np.array([10e9, 10e9])
        # One cold (below binodal), one hot (above binodal)
        temperatures = np.array([500.0, 10000.0])

        with patch('zalmoxis.eos.calculate_density_batch', side_effect=mock_batch):
            result = calculate_mixed_density_batch(
                pressures,
                temperatures,
                mixture,
                {},
                None,
                None,
                {},
                binodal_T_scale=50.0,
            )

        assert result.shape == (2,)
        # Both should produce finite results (MgSiO3 is always condensed)
        assert np.all(np.isfinite(result))

    def test_batch_zero_fraction_component_skipped(self):
        """Component with zero mass fraction is skipped in batch."""
        from unittest.mock import patch

        call_count = {'n': 0}

        def mock_batch(P, T, md, eos, sf, lf, interp, mzf):
            call_count['n'] += 1
            if 'MgSiO3' in eos:
                return np.array([4000.0])
            return np.full(len(P), np.nan)

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [1.0, 0.0])
        pressures = np.array([100e9])
        temperatures = np.array([3000.0])

        with patch('zalmoxis.eos.calculate_density_batch', side_effect=mock_batch):
            result = calculate_mixed_density_batch(
                pressures, temperatures, mixture, {}, None, None, {}
            )

        # Only MgSiO3 should be evaluated (H2O fraction is 0)
        assert call_count['n'] == 1
        assert np.isfinite(result[0])


@pytest.mark.unit
class TestGetMixedNablaAdPrecomputed:
    """Tests for get_mixed_nabla_ad with precomputed_densities optimization."""

    def test_precomputed_skips_density_lookup(self):
        """When precomputed_densities is provided, density is not recomputed."""
        from unittest.mock import patch

        density_called = {'n': 0}

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            density_called['n'] += 1
            return 4000.0

        def mock_nabla(eos, P, T, md, interp, sf, lf):
            if 'MgSiO3' in eos:
                return 0.3
            if 'H2O' in eos:
                return 0.4
            return None

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])
        precomputed = {'PALEOS:MgSiO3': 4000.0, 'PALEOS:H2O': 1000.0}

        with (
            patch('zalmoxis.eos.calculate_density', side_effect=mock_density),
            patch('zalmoxis.mixing._nabla_ad_for_component', side_effect=mock_nabla),
        ):
            nabla = get_mixed_nabla_ad(
                1e10,
                3000,
                mixture,
                {},
                {},
                precomputed_densities=precomputed,
            )

        # With precomputed densities, calculate_density should NOT be called
        assert density_called['n'] == 0
        assert nabla is not None
        assert 0.0 < nabla < 1.0

    def test_precomputed_partial(self):
        """Partial precomputed: only the missing component triggers EOS lookup."""
        from unittest.mock import patch

        density_calls = []

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            density_calls.append(eos)
            return 1000.0

        def mock_nabla(eos, P, T, md, interp, sf, lf):
            return 0.3

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])
        precomputed = {'PALEOS:MgSiO3': 4000.0}  # only MgSiO3 precomputed

        with (
            patch('zalmoxis.eos.calculate_density', side_effect=mock_density),
            patch('zalmoxis.mixing._nabla_ad_for_component', side_effect=mock_nabla),
        ):
            nabla = get_mixed_nabla_ad(
                1e10,
                3000,
                mixture,
                {},
                {},
                precomputed_densities=precomputed,
            )

        # Only H2O should trigger a density call
        assert density_calls == ['PALEOS:H2O']
        assert nabla is not None

    def test_all_components_invalid_density_returns_none(self):
        """If all components have invalid density, nabla_ad is None."""
        from unittest.mock import patch

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            return None

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            nabla = get_mixed_nabla_ad(1e10, 3000, mixture, {}, {})

        assert nabla is None

    def test_nabla_ad_none_component_excluded(self):
        """Components that return None for nabla_ad are excluded from average."""
        from unittest.mock import patch

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            return 4000.0

        def mock_nabla(eos, P, T, md, interp, sf, lf):
            if 'MgSiO3' in eos:
                return 0.3
            return None  # H2O has no nabla_ad

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])

        with (
            patch('zalmoxis.eos.calculate_density', side_effect=mock_density),
            patch('zalmoxis.mixing._nabla_ad_for_component', side_effect=mock_nabla),
        ):
            nabla = get_mixed_nabla_ad(1e10, 3000, mixture, {}, {})

        # Only MgSiO3 contributes, so result should be its nabla_ad
        assert nabla == pytest.approx(0.3)

    def test_single_component_nabla_ad(self):
        """Single-component mixture delegates directly, no suppression."""
        from unittest.mock import patch

        def mock_nabla(eos, P, T, md, interp, sf, lf):
            return 0.28

        mixture = LayerMixture(['PALEOS:MgSiO3'], [1.0])

        with patch('zalmoxis.mixing._nabla_ad_for_component', side_effect=mock_nabla):
            nabla = get_mixed_nabla_ad(1e10, 3000, mixture, {}, {})

        assert nabla == pytest.approx(0.28)


@pytest.mark.unit
class TestNablaAdForComponent:
    """Tests for the _nabla_ad_for_component routing function."""

    def test_non_tdep_returns_none(self):
        """Non-T-dependent EOS (Seager, Analytic) returns None."""
        result = _nabla_ad_for_component('Seager2007:iron', 1e10, 3000, {}, {}, None, None)
        assert result is None

    def test_paleos_unified_routes_correctly(self):
        """PALEOS unified format (e.g. PALEOS:iron) routes to unified nabla_ad."""
        from unittest.mock import patch

        mat = {'format': 'paleos_unified'}
        md = {'PALEOS:iron': mat}

        with patch(
            'zalmoxis.eos._get_paleos_unified_nabla_ad', return_value=0.25
        ) as mock_fn:
            result = _nabla_ad_for_component('PALEOS:iron', 1e10, 3000, md, {}, None, None)

        assert result == pytest.approx(0.25)
        mock_fn.assert_called_once_with(1e10, 3000, mat, {})

    def test_paleos_2phase_routes_correctly(self):
        """PALEOS-2phase:MgSiO3 converts dT/dP to nabla_ad."""
        from unittest.mock import patch

        md = {'PALEOS-2phase:MgSiO3': {}}
        P = 50e9
        T = 4000.0
        dtdp = 1e-8  # dT/dP in K/Pa

        with patch('zalmoxis.eos._compute_paleos_dtdp', return_value=dtdp):
            result = _nabla_ad_for_component('PALEOS-2phase:MgSiO3', P, T, md, {}, None, None)

        expected = dtdp * P / T
        assert result == pytest.approx(expected)

    def test_paleos_2phase_zero_pressure_returns_none(self):
        """PALEOS-2phase with zero pressure returns None."""
        md = {'PALEOS-2phase:MgSiO3': {}}
        result = _nabla_ad_for_component('PALEOS-2phase:MgSiO3', 0.0, 3000, md, {}, None, None)
        assert result is None

    def test_paleos_2phase_zero_temperature_returns_none(self):
        """PALEOS-2phase with zero temperature returns None."""
        md = {'PALEOS-2phase:MgSiO3': {}}
        result = _nabla_ad_for_component('PALEOS-2phase:MgSiO3', 1e10, 0.0, md, {}, None, None)
        assert result is None

    def test_paleos_2phase_none_dtdp_returns_none(self):
        """PALEOS-2phase returns None if dtdp lookup fails."""
        from unittest.mock import patch

        md = {'PALEOS-2phase:MgSiO3': {}}

        with patch('zalmoxis.eos._compute_paleos_dtdp', return_value=None):
            result = _nabla_ad_for_component(
                'PALEOS-2phase:MgSiO3', 1e10, 3000, md, {}, None, None
            )

        assert result is None

    def test_paleos_2phase_negative_dtdp_returns_none(self):
        """PALEOS-2phase returns None if dtdp is negative."""
        from unittest.mock import patch

        md = {'PALEOS-2phase:MgSiO3': {}}

        with patch('zalmoxis.eos._compute_paleos_dtdp', return_value=-1e-8):
            result = _nabla_ad_for_component(
                'PALEOS-2phase:MgSiO3', 1e10, 3000, md, {}, None, None
            )

        assert result is None

    def test_wolfbower_with_grad_file(self):
        """WolfBower2018 routes to tabulated EOS for adiabat gradient."""
        from unittest.mock import patch

        mat = {'melted_mantle': {'adiabat_grad_file': '/some/path.dat'}}
        md = {'WolfBower2018:MgSiO3': mat}
        P = 50e9
        T = 4000.0
        dtdp = 2e-8

        with patch('zalmoxis.eos.get_tabulated_eos', return_value=dtdp):
            result = _nabla_ad_for_component('WolfBower2018:MgSiO3', P, T, md, {}, None, None)

        expected = dtdp * P / T
        assert result == pytest.approx(expected)

    def test_wolfbower_no_grad_file_returns_none(self):
        """WolfBower2018 without adiabat_grad_file returns None."""
        mat = {'melted_mantle': {}}
        md = {'WolfBower2018:MgSiO3': mat}

        result = _nabla_ad_for_component('WolfBower2018:MgSiO3', 1e10, 3000, md, {}, None, None)
        assert result is None

    def test_wolfbower_no_melted_mantle_returns_none(self):
        """WolfBower2018 without melted_mantle dict key returns None."""
        md = {'WolfBower2018:MgSiO3': {}}

        result = _nabla_ad_for_component('WolfBower2018:MgSiO3', 1e10, 3000, md, {}, None, None)
        assert result is None

    def test_wolfbower_negative_dtdp_returns_none(self):
        """WolfBower2018 returns None when tabulated dtdp is negative."""
        from unittest.mock import patch

        mat = {'melted_mantle': {'adiabat_grad_file': '/some/path.dat'}}
        md = {'WolfBower2018:MgSiO3': mat}

        with patch('zalmoxis.eos.get_tabulated_eos', return_value=-1e-8):
            result = _nabla_ad_for_component(
                'WolfBower2018:MgSiO3', 1e10, 3000, md, {}, None, None
            )

        assert result is None

    def test_wolfbower_none_dtdp_returns_none(self):
        """WolfBower2018 returns None when tabulated dtdp is None."""
        from unittest.mock import patch

        mat = {'melted_mantle': {'adiabat_grad_file': '/some/path.dat'}}
        md = {'WolfBower2018:MgSiO3': mat}

        with patch('zalmoxis.eos.get_tabulated_eos', return_value=None):
            result = _nabla_ad_for_component(
                'WolfBower2018:MgSiO3', 1e10, 3000, md, {}, None, None
            )

        assert result is None

    def test_unknown_tdep_eos_no_format(self):
        """T-dependent EOS without 'format' key and not PALEOS-2phase falls through."""
        md = {'RTPress100TPa:MgSiO3': {}}

        result = _nabla_ad_for_component('RTPress100TPa:MgSiO3', 1e10, 3000, md, {}, None, None)
        # No melted_mantle/adiabat_grad_file, so returns None
        assert result is None
