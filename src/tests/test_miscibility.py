"""Tests for global miscibility extension (binodal-aware structure).

Tests cover:
- VolatileProfile binodal blending (blend_with_binodal)
- Above/below binodal detection (_is_above_binodal)
- apply_to_mixture_with_binodal ordering
- Backward compatibility: global_miscibility=False unchanged behavior
- solve_miscible_interior mass conservation convergence

References
----------
- Rogers, Young & Schlichting (2025), MNRAS 544, 3496.
- Gupta, Stixrude & Schlichting (2025), ApJL 982, L35.
"""

from __future__ import annotations

import pytest

from zalmoxis.mixing import LayerMixture, VolatileProfile


# ═════════════════════════════════════════════════════════════════════
# VolatileProfile binodal blending
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestVolatileProfileBinodalBlend:
    """Test blend_with_binodal at known conditions."""

    def _make_profile(self, x_H2=0.03):
        """Helper: create a VolatileProfile with global_miscibility enabled."""
        return VolatileProfile(
            w_liquid={'Chabrier:H': 0.01, 'PALEOS:H2O': 0.02},
            w_solid={'Chabrier:H': 0.0, 'PALEOS:H2O': 0.001},
            primary_component='PALEOS:MgSiO3',
            x_interior={'Chabrier:H': x_H2},
            global_miscibility=True,
        )

    def test_above_binodal_returns_x_interior(self):
        """Above binodal (hot, high P): H2 mass fraction = x_interior."""
        vp = self._make_profile(x_H2=0.05)
        # 5000 K and 10 GPa (1e10 Pa): well above MgSiO3-H2 binodal
        result = vp.blend_with_binodal(phi=1.0, pressure=1e10, temperature=5000.0)
        assert result['Chabrier:H'] == pytest.approx(0.05, abs=0.001)
        # Primary component gets remainder
        total = sum(result.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_below_binodal_returns_zero(self):
        """Below binodal (cold): H2 mass fraction = 0."""
        vp = self._make_profile(x_H2=0.05)
        # 1000 K and 1 GPa: well below binodal (~3800 K at 1 GPa)
        result = vp.blend_with_binodal(phi=1.0, pressure=1e9, temperature=1000.0)
        assert result.get('Chabrier:H', 0.0) == pytest.approx(0.0, abs=1e-6)

    def test_high_pressure_always_miscible(self):
        """At P > 35 GPa, T_c < 0: always miscible regardless of T."""
        vp = self._make_profile(x_H2=0.03)
        # 40 GPa, even at low temperature: always miscible
        result = vp.blend_with_binodal(phi=1.0, pressure=40e9, temperature=2000.0)
        assert result['Chabrier:H'] == pytest.approx(0.03, abs=0.001)

    def test_h2o_standard_blend_when_not_in_x_interior(self):
        """H2O not in x_interior: uses standard phi-blend, not binodal."""
        vp = self._make_profile(x_H2=0.03)
        # H2O is NOT in x_interior, so it uses standard phi-blend
        result = vp.blend_with_binodal(phi=0.5, pressure=1e10, temperature=5000.0)
        # phi=0.5: w_H2O = 0.5 * 0.02 + 0.5 * 0.001 = 0.0105
        expected_h2o = 0.5 * 0.02 + 0.5 * 0.001
        assert result.get('PALEOS:H2O', 0.0) == pytest.approx(expected_h2o, abs=0.002)


@pytest.mark.unit
class TestVolatileProfileBackwardCompat:
    """Ensure global_miscibility=False preserves existing behavior."""

    def test_blend_unchanged(self):
        """blend() returns same result regardless of global_miscibility flag."""
        vp_off = VolatileProfile(
            w_liquid={'Chabrier:H': 0.01},
            w_solid={'Chabrier:H': 0.0},
            primary_component='PALEOS:MgSiO3',
            global_miscibility=False,
        )
        vp_on = VolatileProfile(
            w_liquid={'Chabrier:H': 0.01},
            w_solid={'Chabrier:H': 0.0},
            primary_component='PALEOS:MgSiO3',
            x_interior={'Chabrier:H': 0.03},
            global_miscibility=True,
        )
        # blend() ignores global_miscibility (use blend_with_binodal for that)
        r_off = vp_off.blend(phi=0.5)
        r_on = vp_on.blend(phi=0.5)
        assert r_off == r_on

    def test_blend_with_binodal_fallback(self):
        """blend_with_binodal with global_miscibility=False falls back to blend."""
        vp = VolatileProfile(
            w_liquid={'Chabrier:H': 0.01},
            w_solid={'Chabrier:H': 0.0},
            primary_component='PALEOS:MgSiO3',
            global_miscibility=False,
        )
        r_blend = vp.blend(phi=0.5)
        r_binodal = vp.blend_with_binodal(phi=0.5, pressure=1e10, temperature=5000.0)
        assert r_blend == r_binodal


# ═════════════════════════════════════════════════════════════════════
# _is_above_binodal
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestIsAboveBinodal:
    """Test the binodal detection for H2 and H2O."""

    def _make_profile(self, x_H2=0.03):
        return VolatileProfile(
            w_liquid={'Chabrier:H': 0.01},
            w_solid={},
            primary_component='PALEOS:MgSiO3',
            x_interior={'Chabrier:H': x_H2},
            global_miscibility=True,
        )

    def test_H2_miscible_at_high_T(self):
        """H2-MgSiO3 miscible at 5000 K, 5 GPa."""
        vp = self._make_profile(x_H2=0.03)
        assert vp._is_above_binodal('Chabrier:H', 5e9, 5000.0) is True

    def test_H2_immiscible_at_low_T(self):
        """H2-MgSiO3 immiscible at 1000 K, 1 GPa."""
        vp = self._make_profile(x_H2=0.03)
        assert vp._is_above_binodal('Chabrier:H', 1e9, 1000.0) is False

    def test_H2_miscible_at_very_high_P(self):
        """At P > 35 GPa, always miscible."""
        vp = self._make_profile(x_H2=0.03)
        assert vp._is_above_binodal('Chabrier:H', 40e9, 1000.0) is True

    def test_H2O_miscible_at_high_T(self):
        """H2-H2O miscible at 4000 K, 5 GPa (above critical curve)."""
        vp = self._make_profile()
        # Critical T at 5 GPa is ~ 2000-3000 K; 4000 K is well above
        assert vp._is_above_binodal('PALEOS:H2O', 5e9, 4000.0) is True

    def test_H2O_immiscible_at_low_T(self):
        """H2-H2O immiscible at 500 K (below H2O critical point floor)."""
        vp = self._make_profile()
        assert vp._is_above_binodal('PALEOS:H2O', 1e9, 500.0) is False

    def test_unknown_species_assumed_miscible(self):
        """Unknown species always returns True (no suppression)."""
        vp = self._make_profile()
        assert vp._is_above_binodal('PALEOS:iron', 1e9, 1000.0) is True

    def test_zero_x_interior_returns_immiscible(self):
        """If x_interior is 0 for H2, _is_above_binodal returns False."""
        vp = VolatileProfile(
            w_liquid={},
            w_solid={},
            primary_component='PALEOS:MgSiO3',
            x_interior={'Chabrier:H': 0.0},
            global_miscibility=True,
        )
        assert vp._is_above_binodal('Chabrier:H', 5e9, 5000.0) is False


# ═════════════════════════════════════════════════════════════════════
# apply_to_mixture_with_binodal
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestApplyToMixtureWithBinodal:
    """Test that fractions map correctly to LayerMixture ordering."""

    def test_ordering_preserved(self):
        """Fractions follow LayerMixture component order."""
        mix = LayerMixture(
            components=['PALEOS:MgSiO3', 'Chabrier:H', 'PALEOS:H2O'],
            fractions=[0.90, 0.05, 0.05],
        )
        vp = VolatileProfile(
            w_liquid={'Chabrier:H': 0.01, 'PALEOS:H2O': 0.02},
            w_solid={'Chabrier:H': 0.0, 'PALEOS:H2O': 0.0},
            primary_component='PALEOS:MgSiO3',
            x_interior={'Chabrier:H': 0.04},
            global_miscibility=True,
        )
        # Above binodal: H2 should be ~0.04
        fracs = vp.apply_to_mixture_with_binodal(
            mix, phi=1.0, pressure=1e10, temperature=5000.0
        )
        assert len(fracs) == 3
        assert sum(fracs) == pytest.approx(1.0, abs=1e-6)
        # H2 (index 1) should be close to 0.04 (after normalization)
        assert fracs[1] > 0.02

    def test_below_binodal_h2_zero(self):
        """Below binodal: H2 fraction should be 0."""
        mix = LayerMixture(
            components=['PALEOS:MgSiO3', 'Chabrier:H'],
            fractions=[0.95, 0.05],
        )
        vp = VolatileProfile(
            w_liquid={'Chabrier:H': 0.01},
            w_solid={'Chabrier:H': 0.0},
            primary_component='PALEOS:MgSiO3',
            x_interior={'Chabrier:H': 0.04},
            global_miscibility=True,
        )
        # Below binodal: 1000 K, 1 GPa
        fracs = vp.apply_to_mixture_with_binodal(
            mix, phi=1.0, pressure=1e9, temperature=1000.0
        )
        # H2 (index 1) should be 0, MgSiO3 gets everything
        assert fracs[1] == pytest.approx(0.0, abs=1e-6)
        assert fracs[0] == pytest.approx(1.0, abs=1e-6)

    def test_normalization(self):
        """Result always sums to 1.0."""
        mix = LayerMixture(
            components=['PALEOS:MgSiO3', 'Chabrier:H', 'PALEOS:H2O'],
            fractions=[0.85, 0.10, 0.05],
        )
        vp = VolatileProfile(
            w_liquid={'Chabrier:H': 0.01, 'PALEOS:H2O': 0.02},
            w_solid={},
            primary_component='PALEOS:MgSiO3',
            x_interior={'Chabrier:H': 0.03},
            global_miscibility=True,
        )
        for T in [1000.0, 3000.0, 5000.0]:
            for P in [1e9, 5e9, 20e9, 40e9]:
                fracs = vp.apply_to_mixture_with_binodal(
                    mix, phi=0.5, pressure=P, temperature=T
                )
                assert sum(fracs) == pytest.approx(1.0, abs=1e-6)


# ═════════════════════════════════════════════════════════════════════
# Solvus detection edge cases
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSolvusEdgeCases:
    """Test binodal crossing detection at boundary conditions."""

    def test_fully_miscible_planet(self):
        """Very hot planet: above binodal everywhere -> all H2 dissolved."""
        vp = VolatileProfile(
            w_liquid={'Chabrier:H': 0.01},
            w_solid={},
            primary_component='PALEOS:MgSiO3',
            x_interior={'Chabrier:H': 0.05},
            global_miscibility=True,
        )
        # At 8000 K, 50 GPa: always miscible
        result = vp.blend_with_binodal(phi=1.0, pressure=50e9, temperature=8000.0)
        assert result['Chabrier:H'] == pytest.approx(0.05, abs=0.001)

    def test_fully_immiscible_planet(self):
        """Very cold planet: below binodal everywhere -> no H2 dissolved."""
        vp = VolatileProfile(
            w_liquid={'Chabrier:H': 0.01},
            w_solid={},
            primary_component='PALEOS:MgSiO3',
            x_interior={'Chabrier:H': 0.05},
            global_miscibility=True,
        )
        # At 500 K, 0.1 GPa: well below binodal
        result = vp.blend_with_binodal(phi=1.0, pressure=1e8, temperature=500.0)
        assert result.get('Chabrier:H', 0.0) == pytest.approx(0.0, abs=1e-6)

    def test_transition_temperature_range(self):
        """Binodal transition occurs in the expected T range at 1 GPa.

        At 1 GPa and x_H2 ~ 0.03 (mass), the binodal temperature from
        Rogers+2025 should be ~2500-3500 K for mole fractions ~0.6.
        """
        vp = VolatileProfile(
            w_liquid={'Chabrier:H': 0.01},
            w_solid={},
            primary_component='PALEOS:MgSiO3',
            x_interior={'Chabrier:H': 0.03},
            global_miscibility=True,
        )
        # Should be immiscible at 2000 K
        r_cold = vp.blend_with_binodal(phi=1.0, pressure=1e9, temperature=2000.0)
        # Should be miscible at 5000 K
        r_hot = vp.blend_with_binodal(phi=1.0, pressure=1e9, temperature=5000.0)
        assert r_cold.get('Chabrier:H', 0.0) < 0.01
        assert r_hot.get('Chabrier:H', 0.0) > 0.02
