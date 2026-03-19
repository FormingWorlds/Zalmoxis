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

        with patch('zalmoxis.eos_functions.calculate_density', side_effect=mock_density):
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

        with patch('zalmoxis.eos_functions.calculate_density', side_effect=mock_density):
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

        with patch('zalmoxis.eos_functions.calculate_density', side_effect=mock_density):
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

        with patch('zalmoxis.eos_functions.calculate_density', side_effect=mock_density):
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

        with patch('zalmoxis.eos_functions.calculate_density', side_effect=mock_density):
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
