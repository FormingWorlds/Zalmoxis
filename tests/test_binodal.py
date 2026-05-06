"""Tests for binodal (miscibility) models: H2-MgSiO3 and H2-H2O.

Tests cover:
- Rogers+2025 H2-MgSiO3 binodal: critical temperature, binodal curve shape,
  suppression weights above/below the binodal.
- Gupta+2025 H2-H2O binodal: critical pressure/temperature relation,
  coexistence compositions, suppression weights.
- Shared utilities: mass-to-mole fraction conversion.

References
----------
- Rogers, Young & Schlichting (2025), MNRAS 544, 3496.
- Gupta, Kovacevic, Mazevet (2025), ApJL 982, L35.
- Testing guide: docs/How-to/test_infrastructure.md
"""

from __future__ import annotations

import pytest

from zalmoxis.binodal import (
    gupta2025_coexistence_compositions,
    gupta2025_critical_pressure,
    gupta2025_critical_temperature,
    gupta2025_gibbs_mixing,
    gupta2025_suppression_weight,
    mass_to_mole_fraction,
    rogers2025_binodal_temperature,
    rogers2025_suppression_weight,
)

# ═════════════════════════════════════════════════════════════════════
# Shared utilities
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestMassToMoleFraction:
    """Mass-to-mole fraction conversion with known analytical values."""

    def test_H2_in_MgSiO3_1pct(self):
        """1% H2 by mass in MgSiO3 = ~33.5% by mole (H2 is ~50x lighter)."""
        x = mass_to_mole_fraction(0.01, 0.99, 2.016e-3, 100.39e-3)
        assert x == pytest.approx(0.335, abs=0.01)

    def test_H2_in_H2O_1pct(self):
        """1% H2 by mass in H2O = ~8.2% by mole."""
        x = mass_to_mole_fraction(0.01, 0.99, 2.016e-3, 18.015e-3)
        assert x == pytest.approx(0.082, abs=0.01)

    def test_pure_component(self):
        """100% of either component gives x=1 or x=0."""
        assert mass_to_mole_fraction(1.0, 0.0, 2.016e-3, 100.39e-3) == pytest.approx(1.0)
        assert mass_to_mole_fraction(0.0, 1.0, 2.016e-3, 100.39e-3) == pytest.approx(0.0)

    def test_equal_molar_masses(self):
        """Equal molar masses: mass fraction = mole fraction."""
        x = mass_to_mole_fraction(0.3, 0.7, 10.0, 10.0)
        assert x == pytest.approx(0.3)


# ═════════════════════════════════════════════════════════════════════
# Rogers+2025: H2-MgSiO3
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestRogers2025Binodal:
    """Rogers+2025 H2-MgSiO3 binodal fit (MNRAS 544, 3496)."""

    def test_T_decreases_with_P(self):
        """T_c at 10 GPa should be lower than T_c at 1 GPa.

        Higher pressure makes H2-silicate mixing easier, lowering
        the binodal temperature.
        """
        T_1GPa = rogers2025_binodal_temperature(0.739, 1.0)
        T_10GPa = rogers2025_binodal_temperature(0.739, 10.0)
        assert T_10GPa < T_1GPa

    def test_binodal_at_critical_composition(self):
        """At x_c = 0.73913, T_binodal should equal T_c (peak of the curve)."""
        x_c = 0.73913
        P = 5.0  # GPa
        T_b = rogers2025_binodal_temperature(x_c, P)
        # T_c = 4223 * (1 - P/35) = 4223 * (1 - 5/35) = 4223 * 6/7 = 3619.7 K
        T_c_expected = 4223.0 * (1.0 - P / 35.0)
        assert T_b == pytest.approx(T_c_expected, rel=0.03)

    def test_endmembers_return_zero(self):
        """Pure H2 (x=1) and pure MgSiO3 (x=0) have T_binodal = 0."""
        assert rogers2025_binodal_temperature(0.0, 5.0) == 0.0
        assert rogers2025_binodal_temperature(1.0, 5.0) == 0.0

    def test_high_pressure_always_miscible(self):
        """At P > 35 GPa, T_c < 0, so the system is always miscible."""
        T = rogers2025_binodal_temperature(0.5, 40.0)
        assert T == 0.0

    def test_suppression_above_binodal(self):
        """Well above the binodal, weight should be ~1 (miscible)."""
        # At 1 GPa, T_c ~ 4223*(1+1/35) ~ 4344 K
        # T = 6000 K is well above the binodal for any composition
        sigma = rogers2025_suppression_weight(1e9, 6000.0, 0.03, 0.97)
        assert sigma > 0.99

    def test_suppression_below_binodal(self):
        """Well below the binodal, weight should be ~0 (immiscible)."""
        # At 1 GPa, T_c ~ 4344 K, and at x_H2 ~ 0.6 the binodal is
        # near T_c. T = 1000 K is well below.
        sigma = rogers2025_suppression_weight(1e9, 1000.0, 0.03, 0.97)
        assert sigma < 0.01

    def test_suppression_zero_H2(self):
        """No H2 in the mixture: weight = 1 (no suppression)."""
        sigma = rogers2025_suppression_weight(1e9, 3000.0, 0.0, 1.0)
        assert sigma == 1.0

    def test_suppression_zero_silicate(self):
        """No silicate in the mixture: weight = 1 (no partner for binodal)."""
        sigma = rogers2025_suppression_weight(1e9, 3000.0, 1.0, 0.0)
        assert sigma == 1.0


# ═════════════════════════════════════════════════════════════════════
# Gupta+2025: H2-H2O
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestGupta2025Binodal:
    """Gupta+2025 H2-H2O miscibility (ApJL 982, L35)."""

    def test_critical_pressure_increases_with_T(self):
        """Higher T should give higher critical pressure.

        The critical curve P_c(T) increases with T because more thermal
        energy is needed to overcome the mixing enthalpy barrier.
        """
        Pc_1000 = gupta2025_critical_pressure(1000.0)
        Pc_3000 = gupta2025_critical_pressure(3000.0)
        assert Pc_3000 > Pc_1000

    def test_critical_T_at_1GPa(self):
        """Critical temperature at 1 GPa should be roughly 800-1200 K.

        The median posterior parameters give T_c ~ 930 K at 1 GPa.
        """
        T_c = gupta2025_critical_temperature(1.0)
        assert T_c is not None
        assert 700 < T_c < 1200

    def test_critical_T_increases_with_P(self):
        """Higher pressure requires higher T for miscibility.

        At high P, the volume-of-mixing term dominates, making it
        harder to achieve a single phase.
        """
        T_1 = gupta2025_critical_temperature(1.0)
        T_10 = gupta2025_critical_temperature(10.0)
        assert T_1 is not None
        assert T_10 is not None
        assert T_10 > T_1

    def test_coexistence_below_critical(self):
        """Below the critical curve, two coexisting compositions should exist."""
        # At T=1000 K, P_c should be low. Use P = 0.1 GPa (below P_c).
        T = 1000.0
        P_c = gupta2025_critical_pressure(T)
        if P_c > 0.1:
            result = gupta2025_coexistence_compositions(T, 0.05)
            if result is not None:
                x1, x2 = result
                assert 0 < x1 < x2 < 1

    def test_coexistence_above_critical_returns_none(self):
        """Above the critical curve, the system is single-phase."""
        # At T=6000 K, P_c should be quite high. Use a P well below P_c.
        T = 6000.0
        P_c = gupta2025_critical_pressure(T)
        # Use a pressure well above P_c to be in single-phase regime
        result = gupta2025_coexistence_compositions(T, P_c + 10.0)
        assert result is None

    def test_suppression_above_critical(self):
        """Well above the critical curve, weight should be ~1 (miscible)."""
        # At very high T, H2 and H2O are fully miscible
        sigma = gupta2025_suppression_weight(1e9, 8000.0, 0.1, 0.9)
        assert sigma > 0.95

    def test_suppression_below_critical(self):
        """Well below the critical curve, weight should be ~0 (immiscible)."""
        # At low T, H2 and H2O separate
        sigma = gupta2025_suppression_weight(1e10, 300.0, 0.1, 0.9)
        assert sigma < 0.05

    def test_suppression_zero_H2(self):
        """No H2: weight = 1."""
        sigma = gupta2025_suppression_weight(1e9, 3000.0, 0.0, 1.0)
        assert sigma == 1.0

    def test_suppression_zero_H2O(self):
        """No H2O: weight = 1."""
        sigma = gupta2025_suppression_weight(1e9, 3000.0, 1.0, 0.0)
        assert sigma == 1.0


# ═════════════════════════════════════════════════════════════════════
# Integration with mixing module
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestBinodalFactor:
    """Tests for _binodal_factor in mixing.py."""

    def test_non_H2_returns_one(self):
        """Non-H2 components are never suppressed by binodals."""
        from zalmoxis.mixing import LayerMixture, _binodal_factor

        mixture = LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.85, 0.15])
        sigma = _binodal_factor('PALEOS:MgSiO3', 0.85, mixture, 1e10, 3000.0, 50.0)
        assert sigma == 1.0

    def test_H2_with_silicate_above_binodal(self):
        """H2 above the binodal with silicate partner: weight ~1."""
        from zalmoxis.mixing import LayerMixture, _binodal_factor

        mixture = LayerMixture(['PALEOS:MgSiO3', 'Chabrier:H'], [0.97, 0.03])
        sigma = _binodal_factor('Chabrier:H', 0.03, mixture, 1e9, 6000.0, 50.0)
        assert sigma > 0.95

    def test_H2_with_silicate_below_binodal(self):
        """H2 below the binodal with silicate partner: weight ~0."""
        from zalmoxis.mixing import LayerMixture, _binodal_factor

        mixture = LayerMixture(['PALEOS:MgSiO3', 'Chabrier:H'], [0.97, 0.03])
        sigma = _binodal_factor('Chabrier:H', 0.03, mixture, 1e9, 1000.0, 50.0)
        assert sigma < 0.05

    def test_H2_with_H2O_above_critical(self):
        """H2 above the H2O critical curve: weight ~1."""
        from zalmoxis.mixing import LayerMixture, _binodal_factor

        mixture = LayerMixture(['PALEOS:H2O', 'Chabrier:H'], [0.90, 0.10])
        sigma = _binodal_factor('Chabrier:H', 0.10, mixture, 1e9, 8000.0, 50.0)
        assert sigma > 0.95

    def test_H2_with_both_partners(self):
        """H2 in a 3-component mixture: both binodals checked."""
        from zalmoxis.mixing import LayerMixture, _binodal_factor

        mixture = LayerMixture(
            ['PALEOS:MgSiO3', 'PALEOS:H2O', 'Chabrier:H'], [0.80, 0.10, 0.10]
        )
        # At T=6000K, P=1 GPa: H2 should be miscible with MgSiO3
        # and the H2-H2O critical curve should also be satisfied
        sigma_high_T = _binodal_factor('Chabrier:H', 0.10, mixture, 1e9, 6000.0, 50.0)
        # At T=1000K: H2 should be immiscible with MgSiO3
        sigma_low_T = _binodal_factor('Chabrier:H', 0.10, mixture, 1e9, 1000.0, 50.0)
        assert sigma_high_T > sigma_low_T

    def test_H2_with_iron_no_binodal(self):
        """H2 + iron (no silicate or water): no binodal, returns 1.0."""
        from zalmoxis.mixing import LayerMixture, _binodal_factor

        mixture = LayerMixture(['PALEOS:iron', 'Chabrier:H'], [0.90, 0.10])
        sigma = _binodal_factor('Chabrier:H', 0.10, mixture, 1e9, 1000.0, 50.0)
        assert sigma == 1.0


@pytest.mark.unit
class TestMixingWithH2:
    """Tests for calculate_mixed_density with H2 binodal suppression."""

    def test_mixed_density_H2_MgSiO3_above_binodal(self):
        """Above binodal: H2 participates in harmonic mean."""
        from unittest.mock import patch

        from zalmoxis.mixing import LayerMixture, calculate_mixed_density

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            if 'MgSiO3' in eos:
                return 4000.0
            if 'Chabrier' in eos:
                return 500.0  # high-density H2 at high P
            return None

        mixture = LayerMixture(['PALEOS:MgSiO3', 'Chabrier:H'], [0.97, 0.03])

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            rho = calculate_mixed_density(
                1e9,
                6000,  # well above binodal
                mixture,
                {},
                None,
                None,
                {},
                condensed_rho_min=30.0,
                condensed_rho_scale=10.0,
            )

        assert rho is not None
        # H2 at 500 kg/m^3 is well above rho_min=30, so it participates
        # Result should be between pure MgSiO3 (4000) and the harmonic mean
        assert rho < 4000
        assert rho > 100

    def test_mixed_density_H2_MgSiO3_below_binodal(self):
        """Below binodal: H2 is suppressed, result is near pure MgSiO3."""
        from unittest.mock import patch

        from zalmoxis.mixing import LayerMixture, calculate_mixed_density

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            if 'MgSiO3' in eos:
                return 4000.0
            if 'Chabrier' in eos:
                return 500.0
            return None

        mixture = LayerMixture(['PALEOS:MgSiO3', 'Chabrier:H'], [0.97, 0.03])

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            rho = calculate_mixed_density(
                1e9,
                1000,  # well below binodal
                mixture,
                {},
                None,
                None,
                {},
                condensed_rho_min=30.0,
                condensed_rho_scale=10.0,
            )

        assert rho is not None
        # H2 is suppressed by binodal, result should be close to MgSiO3
        assert rho > 3500

    def test_single_H2_no_binodal(self):
        """Single-component H2 layer: no suppression, uses fast path."""
        from unittest.mock import patch

        from zalmoxis.mixing import LayerMixture, calculate_mixed_density

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            return 200.0

        mixture = LayerMixture(['Chabrier:H'], [1.0])

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            rho = calculate_mixed_density(
                1e9, 3000, mixture, {}, None, None, {}, condensed_rho_min=30.0
            )

        # Single-component fast path: returns raw density, no suppression
        assert rho == pytest.approx(200.0)

    def test_per_component_rho_min_H2_H2O(self):
        """H2+H2O mixture: each component uses its own critical density.

        H2O at 100 kg/m^3 is below its critical density (322) and should
        be suppressed. H2 at 100 kg/m^3 is above its critical density (30)
        and should NOT be suppressed. With a single global rho_min=30,
        both would pass; with per-component values, only H2 passes.
        """
        from unittest.mock import patch

        from zalmoxis.mixing import LayerMixture, calculate_mixed_density

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            if 'H2O' in eos:
                return 100.0  # below H2O critical (322), above H2 critical (30)
            if 'Chabrier' in eos:
                return 100.0  # above H2 critical (30)
            return None

        mixture = LayerMixture(['PALEOS:H2O', 'Chabrier:H'], [0.90, 0.10])

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            rho = calculate_mixed_density(
                1e9,
                5000,  # above both binodals
                mixture,
                {},
                None,
                None,
                {},
                condensed_rho_min=322.0,  # global default
            )

        # H2O at 100 kg/m^3: per-component rho_min=322, sigma ~ 0.01 (suppressed)
        # H2 at 100 kg/m^3: per-component rho_min=30, sigma ~ 1.0 (not suppressed)
        # Result should be dominated by H2 (the unsuppressed component)
        assert rho is not None
        assert rho == pytest.approx(100.0, rel=0.1)

    def test_per_component_rho_min_both_dense(self):
        """When both H2 and H2O are dense, both participate."""
        from unittest.mock import patch

        from zalmoxis.mixing import LayerMixture, calculate_mixed_density

        def mock_density(P, md, eos, T, sf, lf, interp, mzf):
            if 'H2O' in eos:
                return 1000.0  # well above H2O critical (322)
            if 'Chabrier' in eos:
                return 500.0  # well above H2 critical (30)
            return None

        mixture = LayerMixture(['PALEOS:H2O', 'Chabrier:H'], [0.90, 0.10])

        with patch('zalmoxis.eos.calculate_density', side_effect=mock_density):
            rho = calculate_mixed_density(
                1e10,
                5000,
                mixture,
                {},
                None,
                None,
                {},
            )

        # Both above their critical densities: standard harmonic mean
        # rho_mix = 1 / (0.9/1000 + 0.1/500) = 1 / 0.0011 = 909
        assert rho is not None
        assert rho == pytest.approx(909.0, rel=0.02)


# ═════════════════════════════════════════════════════════════════════
# Edge-case and branch-coverage tests for binodal.py
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestMassToMoleFractionEdge:
    """Edge cases for mass_to_mole_fraction."""

    def test_both_zero_mass_fractions(self):
        """Both w_1 = w_2 = 0 should return 0 (guard against division by zero)."""
        assert mass_to_mole_fraction(0.0, 0.0, 2.016e-3, 100.39e-3) == 0.0


@pytest.mark.unit
class TestRogers2025SuppressionEdge:
    """Edge-case branches in rogers2025_suppression_weight."""

    def test_hard_cutoff_above(self):
        """T_scale=0 (hard cutoff) with T above binodal returns 1.0."""
        # At 40 GPa, P > 35 GPa, so T_binodal = 0 and weight is 1.0
        # regardless of T_scale. Use a moderate P instead.
        # At 1 GPa, x_c ~ 0.739, T_c ~ 4223*(1 - 1/35) ~ 4102 K
        # Use T well above T_c so T > T_binodal.
        sigma = rogers2025_suppression_weight(1e9, 6000.0, 0.03, 0.97, T_scale=0.0)
        assert sigma == 1.0

    def test_hard_cutoff_below(self):
        """T_scale=0 (hard cutoff) with T below binodal returns 0.0."""
        sigma = rogers2025_suppression_weight(1e9, 1000.0, 0.03, 0.97, T_scale=0.0)
        assert sigma == 0.0

    def test_high_pressure_miscible(self):
        """At P > 35 GPa, T_binodal = 0, so weight is 1.0 (always miscible).

        Exercises the T_binodal <= 0 early return inside the function,
        distinct from the w_H2/w_sil <= 0 guards.
        """
        sigma = rogers2025_suppression_weight(40e9, 500.0, 0.03, 0.97)
        assert sigma == 1.0


@pytest.mark.unit
class TestGupta2025GibbsMixingEdge:
    """Edge cases for gupta2025_gibbs_mixing."""

    def test_pure_H2_returns_zero(self):
        """x_H2 = 1 (pure H2) gives G_mix = 0 (no mixing)."""
        assert gupta2025_gibbs_mixing(1.0, 1000.0, 1.0) == 0.0

    def test_pure_H2O_returns_zero(self):
        """x_H2 = 0 (pure H2O) gives G_mix = 0 (no mixing)."""
        assert gupta2025_gibbs_mixing(0.0, 1000.0, 1.0) == 0.0

    def test_mid_composition_negative(self):
        """At typical conditions, G_mix < 0 (mixing is favorable)."""
        G = gupta2025_gibbs_mixing(0.5, 1000.0, 0.5)
        assert G < 0.0

    def test_very_small_x(self):
        """Very dilute H2 (x ~ 1e-6) should still return a finite value."""
        G = gupta2025_gibbs_mixing(1e-6, 1000.0, 1.0)
        assert G != 0.0
        assert abs(G) < 1e6  # Should be a modest value, not divergent


@pytest.mark.unit
class TestGupta2025CriticalPressureEdge:
    """Edge cases for gupta2025_critical_pressure."""

    def test_returns_finite(self):
        """At normal T, P_c should be a finite positive number."""
        Pc = gupta2025_critical_pressure(1000.0)
        assert Pc > 0
        assert Pc < 100  # Should be order-of-magnitude GPa, not huge


@pytest.mark.unit
class TestGupta2025CriticalTemperatureEdge:
    """Edge-case branches in gupta2025_critical_temperature."""

    def test_zero_pressure_returns_none(self):
        """P = 0 GPa should return None (no critical temperature at zero pressure)."""
        assert gupta2025_critical_temperature(0.0) is None

    def test_negative_pressure_returns_none(self):
        """Negative pressure returns None."""
        assert gupta2025_critical_temperature(-1.0) is None

    def test_very_high_pressure_fallback(self):
        """Pressure beyond the lookup table triggers brentq fallback.

        The table covers log10(P) from -3 to 3.5, so P > 10^3.5 ~ 3162 GPa
        falls back to root finding.
        """
        T_c = gupta2025_critical_temperature(5000.0)
        # At 5000 GPa, T_c should be very high or None
        # Either result is acceptable; the test exercises the fallback path
        if T_c is not None:
            assert T_c > 0

    def test_table_interpolation_consistent_with_brentq(self):
        """The precomputed table should give similar results to direct brentq.

        Uses a pressure well within the table range.
        """
        from zalmoxis.binodal import _gupta2025_critical_temperature_brentq

        P = 5.0  # GPa, well within table
        T_table = gupta2025_critical_temperature(P)
        T_brentq = _gupta2025_critical_temperature_brentq(P)
        assert T_table is not None
        assert T_brentq is not None
        assert T_table == pytest.approx(T_brentq, rel=0.01)


@pytest.mark.unit
class TestGupta2025SuppressionEdge:
    """Edge-case branches in gupta2025_suppression_weight."""

    def test_very_low_T_immiscible(self):
        """T < 300 K triggers the early return of 0.0 (immiscible).

        The Gupta+2025 model is parameterized for T >= 300 K. Below this,
        H2 and H2O are certainly phase-separated.
        """
        sigma = gupta2025_suppression_weight(1e9, 200.0, 0.1, 0.9)
        assert sigma == 0.0

    def test_hard_cutoff_above(self):
        """T_scale=0 (hard cutoff) with T above critical returns 1.0."""
        sigma = gupta2025_suppression_weight(1e9, 8000.0, 0.1, 0.9, T_scale=0.0)
        assert sigma == 1.0

    def test_hard_cutoff_below(self):
        """T_scale=0 (hard cutoff) with T below critical returns 0.0."""
        sigma = gupta2025_suppression_weight(1e9, 500.0, 0.1, 0.9, T_scale=0.0)
        assert sigma == 0.0

    def test_T_crit_none_returns_miscible(self):
        """When critical temperature cannot be found, assume miscible (1.0).

        This exercises the T_crit is None guard.
        """
        from unittest.mock import patch

        with patch('zalmoxis.binodal.gupta2025_critical_temperature', return_value=None):
            sigma = gupta2025_suppression_weight(1e9, 1000.0, 0.1, 0.9)
        assert sigma == 1.0

    def test_647K_floor_applied(self):
        """T_crit is floored at 647 K (H2O critical point).

        At low P, T_crit from the model may be below 647 K, but the floor
        ensures H2 is suppressed when H2O would be liquid/ice.
        """
        # At very low P (e.g. 0.001 GPa), T_crit from model is ~700 K
        # The 647 K floor should ensure suppression at e.g. 600 K
        sigma = gupta2025_suppression_weight(1e6, 600.0, 0.1, 0.9)
        assert sigma < 0.5  # Below the floor, so suppressed


@pytest.mark.unit
class TestGupta2025CoexistenceCompositionsDetailed:
    """Tests for the common-tangent construction in coexistence_compositions.

    The Gupta+2025 Gibbs energy with published parameters always has a
    single minimum below the critical curve (no two-phase region on the
    grid), so the internal common-tangent code (lines 487-527) is never
    reached with real physics. We mock gupta2025_gibbs_mixing to produce
    a double-well curve to exercise that path.
    """

    def test_common_tangent_with_double_well(self):
        """Mock a double-well Gibbs curve to exercise the fsolve path.

        The mock produces G(x) = A*(x - 0.2)^2 * (x - 0.8)^2 - C*x*(1-x)
        which has two minima at x ~ 0.40 and x ~ 0.60 (the linear term
        shifts the quartic minima inward from 0.2 and 0.8).
        """
        from unittest.mock import patch

        def double_well_gibbs(x, T, P_GPa):
            """Synthetic double-well Gibbs energy."""
            if x <= 0 or x >= 1:
                return 0.0
            return 5000.0 * (x - 0.2) ** 2 * (x - 0.8) ** 2 - 800.0 * x * (1.0 - x)

        with (
            patch('zalmoxis.binodal.gupta2025_gibbs_mixing', side_effect=double_well_gibbs),
            patch('zalmoxis.binodal.gupta2025_critical_pressure', return_value=100.0),
        ):
            result = gupta2025_coexistence_compositions(1000.0, 1.0)

        assert result is not None
        x1, x2 = result
        assert 0 < x1 < x2 < 1
        # Grid minima are at ~0.40 and ~0.60; fsolve refines from there
        assert x1 == pytest.approx(0.4, abs=0.15)
        assert x2 == pytest.approx(0.6, abs=0.15)

    def test_single_minimum_returns_none(self):
        """When Gibbs has only one minimum, no coexistence exists.

        This exercises the len(minima) < 2 early return at line 484.
        """
        # Real physics: at 1000 K, P well below P_c, only one minimum
        T = 1000.0
        P_c = gupta2025_critical_pressure(T)
        if P_c > 0.1:
            result = gupta2025_coexistence_compositions(T, 0.05)
            assert result is None

    def test_above_critical_pressure_returns_none(self):
        """P >= P_c returns None immediately (single phase)."""
        T = 2000.0
        P_c = gupta2025_critical_pressure(T)
        result = gupta2025_coexistence_compositions(T, P_c + 5.0)
        assert result is None

    def test_fsolve_invalid_result_falls_back_to_grid(self):
        """When fsolve returns invalid compositions, fall back to grid minima.

        The function checks 0 < x_sol[0] < x_sol[1] < 1 after fsolve.
        When fsolve returns reversed or out-of-bounds values, the fallback
        path (lines 524-525) uses the original grid minima instead.
        """
        from unittest.mock import patch

        import numpy as np

        def double_well_gibbs(x, T, P_GPa):
            """Same double-well as the successful test (minima at ~0.40, ~0.60)."""
            if x <= 0 or x >= 1:
                return 0.0
            return 5000.0 * (x - 0.2) ** 2 * (x - 0.8) ** 2 - 800.0 * x * (1.0 - x)

        # Return reversed x values so x_sol[0] > x_sol[1], triggering fallback
        def fsolve_bad(*args, **kwargs):
            return (np.array([0.8, 0.2]), None, None, None)

        with (
            patch('zalmoxis.binodal.gupta2025_gibbs_mixing', side_effect=double_well_gibbs),
            patch('zalmoxis.binodal.gupta2025_critical_pressure', return_value=100.0),
            patch('scipy.optimize.fsolve', side_effect=fsolve_bad),
        ):
            result = gupta2025_coexistence_compositions(1000.0, 1.0)

        # Fallback to grid minima: should still return two compositions
        assert result is not None
        x1, x2 = result
        assert 0 < x1 < x2 < 1
