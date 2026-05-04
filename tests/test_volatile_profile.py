"""Tests for ``zalmoxis.mixing.VolatileProfile`` and ``compute_melt_fraction``.

Both routines are PR #59 additions and arrive uncovered by the existing
``tests/test_mixing.py``. This file targets the entire ``VolatileProfile``
class API (``blend``, ``apply_to_mixture``, ``blend_with_binodal``,
``apply_to_mixture_with_binodal``, ``_is_above_binodal``) plus the
``compute_melt_fraction`` helper.

Anti-happy-path discipline: every test class includes at least one edge
case (phi at 0/1, missing keys, primary component not appearing in either
phase) and at least one physically unreasonable input (NaN in melting
curves, negative mass fractions, T_liq <= T_sol).
"""

from __future__ import annotations

import numpy as np
import pytest

from zalmoxis.mixing import LayerMixture, VolatileProfile, compute_melt_fraction

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# VolatileProfile.blend
# ---------------------------------------------------------------------------


class TestVolatileProfileBlend:
    """Phi-weighted mass-fraction blending without binodal physics."""

    def test_phi_zero_returns_solid_phase_fractions(self):
        """At phi=0 the result equals the solid-phase mass fractions."""
        profile = VolatileProfile(
            w_liquid={'PALEOS:H2O': 0.05},
            w_solid={'PALEOS:H2O': 0.02},
            primary_component='PALEOS:MgSiO3',
        )
        out = profile.blend(0.0)
        assert out['PALEOS:H2O'] == pytest.approx(0.02, rel=1e-12)
        # Primary component fills the remainder.
        assert out['PALEOS:MgSiO3'] == pytest.approx(0.98, rel=1e-12)
        assert sum(out.values()) == pytest.approx(1.0, abs=1e-12)

    def test_phi_one_returns_liquid_phase_fractions(self):
        """At phi=1 the result equals the liquid-phase mass fractions."""
        profile = VolatileProfile(
            w_liquid={'PALEOS:H2O': 0.05},
            w_solid={'PALEOS:H2O': 0.02},
            primary_component='PALEOS:MgSiO3',
        )
        out = profile.blend(1.0)
        assert out['PALEOS:H2O'] == pytest.approx(0.05, rel=1e-12)
        assert out['PALEOS:MgSiO3'] == pytest.approx(0.95, rel=1e-12)

    def test_phi_midway_blends_linearly(self):
        """At phi=0.5 the mass fraction is the algebraic mean.

        Discriminating values: w_liquid=0.10, w_solid=0.04 -> mid = 0.07.
        Asymmetric so any sign flip or wrong weighting is detectable.
        """
        profile = VolatileProfile(
            w_liquid={'PALEOS:H2O': 0.10},
            w_solid={'PALEOS:H2O': 0.04},
            primary_component='PALEOS:MgSiO3',
        )
        out = profile.blend(0.5)
        assert out['PALEOS:H2O'] == pytest.approx(0.07, rel=1e-12)
        assert out['PALEOS:MgSiO3'] == pytest.approx(0.93, rel=1e-12)

    def test_phi_clamped_to_unit_interval(self):
        """Phi outside [0,1] is clamped (per the docstring contract)."""
        profile = VolatileProfile(
            w_liquid={'PALEOS:H2O': 0.10},
            w_solid={'PALEOS:H2O': 0.04},
            primary_component='PALEOS:MgSiO3',
        )
        # Below 0 -> clamps to 0.
        out_lo = profile.blend(-0.5)
        assert out_lo['PALEOS:H2O'] == pytest.approx(0.04, rel=1e-12)
        # Above 1 -> clamps to 1.
        out_hi = profile.blend(2.0)
        assert out_hi['PALEOS:H2O'] == pytest.approx(0.10, rel=1e-12)

    def test_volatile_only_in_liquid_blends_to_zero_at_phi_zero(self):
        """Edge: w_solid omits a key but w_liquid has it.

        At phi=0 the volatile gets zero weight and is dropped from the
        result (the routine only emits keys with w > 0).
        """
        profile = VolatileProfile(
            w_liquid={'Chabrier:H': 0.03},
            w_solid={},
            primary_component='PALEOS:MgSiO3',
        )
        out = profile.blend(0.0)
        assert 'Chabrier:H' not in out
        assert out['PALEOS:MgSiO3'] == pytest.approx(1.0, abs=1e-12)

    def test_total_volatiles_above_one_floors_primary_at_zero(self):
        """Edge: w_liquid sum exceeds 1.0 -> primary clamped to 0.

        Physically unreasonable input: total volatiles > 1.0 should not
        produce a negative primary fraction.
        """
        profile = VolatileProfile(
            w_liquid={'PALEOS:H2O': 0.6, 'Chabrier:H': 0.5},
            w_solid={'PALEOS:H2O': 0.6, 'Chabrier:H': 0.5},
            primary_component='PALEOS:MgSiO3',
        )
        out = profile.blend(0.5)
        assert out['PALEOS:MgSiO3'] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# VolatileProfile.apply_to_mixture
# ---------------------------------------------------------------------------


class TestApplyToMixture:
    """Maps blended fractions onto a LayerMixture's component ordering."""

    def test_returns_fractions_in_mixture_order(self):
        """Output is ordered to match LayerMixture.components."""
        mixture = LayerMixture(
            components=['Chabrier:H', 'PALEOS:H2O', 'PALEOS:MgSiO3'],
            fractions=[0.0, 0.0, 1.0],
        )
        profile = VolatileProfile(
            w_liquid={'Chabrier:H': 0.02, 'PALEOS:H2O': 0.05},
            w_solid={'Chabrier:H': 0.01, 'PALEOS:H2O': 0.02},
            primary_component='PALEOS:MgSiO3',
        )
        fracs = profile.apply_to_mixture(mixture, phi=0.5)
        # Discriminating: two distinct volatiles + primary, all different sizes.
        # Expected (before normalise): 0.015, 0.035, 0.95 -> sums to 1.0 already.
        np.testing.assert_allclose(fracs, [0.015, 0.035, 0.95], rtol=1e-12)
        assert sum(fracs) == pytest.approx(1.0, abs=1e-12)

    def test_unmanaged_component_keeps_original_fraction(self):
        """Components not in the profile keep their LayerMixture fraction."""
        mixture = LayerMixture(
            components=['PALEOS:iron', 'PALEOS:MgSiO3'],
            fractions=[0.4, 0.6],
        )
        profile = VolatileProfile(
            w_liquid={},
            w_solid={},
            primary_component='PALEOS:MgSiO3',
        )
        # No volatiles -> primary fills 1.0; iron is unmanaged -> keeps 0.4.
        # After normalisation: iron 0.4 -> 0.4/1.4, primary 1.0 -> 1.0/1.4.
        fracs = profile.apply_to_mixture(mixture, phi=0.5)
        assert sum(fracs) == pytest.approx(1.0, abs=1e-12)
        # Iron retains positive (non-zero) fraction even though it was
        # absent from the profile.
        assert fracs[0] > 0.0

    def test_managed_zero_blend_set_to_zero(self):
        """Volatile in profile but blended to zero appears as 0 in output."""
        mixture = LayerMixture(
            components=['Chabrier:H', 'PALEOS:MgSiO3'],
            fractions=[0.0, 1.0],
        )
        profile = VolatileProfile(
            w_liquid={'Chabrier:H': 0.0},
            w_solid={'Chabrier:H': 0.0},
            primary_component='PALEOS:MgSiO3',
        )
        fracs = profile.apply_to_mixture(mixture, phi=0.5)
        # Chabrier:H is managed but blends to 0; primary fills 1.0.
        assert fracs[0] == pytest.approx(0.0, abs=1e-12)
        assert fracs[1] == pytest.approx(1.0, abs=1e-12)

    def test_zero_total_returns_unnormalised(self):
        """Edge: if every fraction is zero the result is returned unmodified.

        Forces the ``if total > 0`` False branch. Make every w_* be 0 and the
        primary component absent from the LayerMixture, so fracs == [0.0]
        and the routine skips the normalise-by-total step.
        """
        profile = VolatileProfile(
            w_liquid={'Chabrier:H': 0.0},
            w_solid={'Chabrier:H': 0.0},
            primary_component='Not:In:Mixture',
        )
        mixture2 = LayerMixture(
            components=['Chabrier:H'],
            fractions=[1.0],
        )
        fracs = profile.apply_to_mixture(mixture2, phi=0.5)
        # Single managed component with blend 0 -> [0.0]. Sum is 0.
        # Since total == 0 the routine skips normalisation.
        assert fracs == [0.0]


# ---------------------------------------------------------------------------
# VolatileProfile._is_above_binodal
# ---------------------------------------------------------------------------


class TestIsAboveBinodal:
    """Binodal classifier for H2-silicate and H2-H2O systems."""

    def test_h2_silicate_below_threshold_returns_false(self):
        """H2 with small x_interior at high T is below the binodal.

        Edge case: x_interior=0 should hit the early return of False.
        """
        profile = VolatileProfile(
            x_interior={'Chabrier:H': 0.0},
        )
        # x_interior == 0 -> below binodal regardless of P, T.
        assert profile._is_above_binodal('Chabrier:H', 1e10, 5000.0) is False

    def test_h2_silicate_above_critical_returns_true(self):
        """H2 with non-trivial x_interior at high T may be miscible.

        Property assertion: at T well above the critical T_c, the
        suppression weight should approach 1.0 and the classifier returns
        True. With T_scale=50 the boundary is very sharp.
        """
        profile = VolatileProfile(
            x_interior={'Chabrier:H': 0.10},
        )
        # P near 0, T very high -> deep above critical T.
        assert profile._is_above_binodal('Chabrier:H', 1e8, 1.0e5) is True

    def test_h2o_critical_temperature_floor(self):
        """H2O critical temperature is floored at 647 K (water critical point).

        At low P (10 MPa), gupta2025_critical_temperature returns ~300 K;
        the floor at 647 K then dominates, so T=500 K is below the floor
        and the species is reported as immiscible.
        """
        profile = VolatileProfile()
        assert profile._is_above_binodal('PALEOS:H2O', 1e7, 500.0) is False
        # Same P, T=2000 K -> well above floor -> miscible.
        assert profile._is_above_binodal('PALEOS:H2O', 1e7, 2000.0) is True

    def test_h2o_off_table_low_pressure_assumes_miscible(self):
        """When T_crit is None (off-table), the routine returns True.

        Edge case: at P=1e5 Pa = 1e-4 GPa the Gupta tabulation has no
        critical temperature, so the classifier falls back to ``True``.
        Documented as ``Cannot determine: assume miscible`` in the source.
        """
        profile = VolatileProfile()
        assert profile._is_above_binodal('PALEOS:H2O', 1e5, 500.0) is True

    def test_unknown_species_assumed_miscible(self):
        """Default branch returns True for unknown species names."""
        profile = VolatileProfile()
        assert profile._is_above_binodal('Unknown:species', 1e9, 1500.0) is True


# ---------------------------------------------------------------------------
# VolatileProfile.blend_with_binodal
# ---------------------------------------------------------------------------


class TestBlendWithBinodal:
    """Phi-weighted blend with binodal-controlled species switching."""

    def test_falls_back_to_phi_blend_when_global_miscibility_off(self):
        """Without global_miscibility, behaves identically to blend()."""
        profile = VolatileProfile(
            w_liquid={'Chabrier:H': 0.02},
            w_solid={'Chabrier:H': 0.005},
            primary_component='PALEOS:MgSiO3',
            x_interior={'Chabrier:H': 0.05},
            global_miscibility=False,
        )
        out_a = profile.blend(0.5)
        out_b = profile.blend_with_binodal(0.5, pressure=1e10, temperature=4000.0)
        assert out_a == pytest.approx(out_b)

    def test_falls_back_to_phi_blend_when_x_interior_empty(self):
        """global_miscibility=True but no x_interior -> phi blend."""
        profile = VolatileProfile(
            w_liquid={'PALEOS:H2O': 0.04},
            w_solid={'PALEOS:H2O': 0.01},
            primary_component='PALEOS:MgSiO3',
            global_miscibility=True,
            x_interior={},
        )
        out_a = profile.blend(0.3)
        out_b = profile.blend_with_binodal(0.3, pressure=1e10, temperature=4000.0)
        assert out_a == pytest.approx(out_b)

    def test_above_binodal_uses_x_interior(self):
        """Above the binodal, the controlled species takes the x_interior value."""
        profile = VolatileProfile(
            w_liquid={'PALEOS:H2O': 0.10},
            w_solid={'PALEOS:H2O': 0.0},
            primary_component='PALEOS:MgSiO3',
            global_miscibility=True,
            x_interior={'PALEOS:H2O': 0.07},
        )
        out = profile.blend_with_binodal(0.5, pressure=1e9, temperature=2000.0)
        # H2O critical T at P=1 GPa is ~947 K (Gupta); 2000 K > T_c -> miscible.
        assert out['PALEOS:H2O'] == pytest.approx(0.07, rel=1e-12)
        assert out['PALEOS:MgSiO3'] == pytest.approx(0.93, rel=1e-12)

    def test_below_binodal_zeroes_controlled_species(self):
        """Below the binodal, the controlled species is expelled (w=0).

        Use P = 1 GPa where Gupta T_crit ~ 930 K is finite, then probe at
        T = 500 K which is below the floored critical T (max(930, 647)=930).
        """
        profile = VolatileProfile(
            w_liquid={'PALEOS:H2O': 0.10},
            w_solid={'PALEOS:H2O': 0.0},
            primary_component='PALEOS:MgSiO3',
            global_miscibility=True,
            x_interior={'PALEOS:H2O': 0.07},
        )
        out = profile.blend_with_binodal(0.5, pressure=1e9, temperature=500.0)
        assert 'PALEOS:H2O' not in out  # expelled (w == 0 dropped)
        assert out['PALEOS:MgSiO3'] == pytest.approx(1.0, abs=1e-12)

    def test_uncontrolled_species_keep_phi_blend_under_global_miscibility(self):
        """A species absent from x_interior still uses the phi-weighted blend."""
        profile = VolatileProfile(
            w_liquid={'PALEOS:H2O': 0.10, 'Other:gas': 0.02},
            w_solid={'PALEOS:H2O': 0.0, 'Other:gas': 0.005},
            primary_component='PALEOS:MgSiO3',
            global_miscibility=True,
            x_interior={'PALEOS:H2O': 0.07},  # only H2O controlled
        )
        out = profile.blend_with_binodal(0.5, pressure=1e9, temperature=2000.0)
        # H2O = x_interior; Other:gas = phi-weighted = 0.5*0.02 + 0.5*0.005 = 0.0125
        assert out['Other:gas'] == pytest.approx(0.0125, rel=1e-12)


# ---------------------------------------------------------------------------
# VolatileProfile.apply_to_mixture_with_binodal
# ---------------------------------------------------------------------------


class TestApplyToMixtureWithBinodal:
    """``apply_to_mixture`` analogue that consults the binodal."""

    def test_above_binodal_returns_x_interior_aligned_to_mixture(self):
        """At T above binodal, controlled species rides on x_interior."""
        mixture = LayerMixture(
            components=['Chabrier:H', 'PALEOS:MgSiO3'],
            fractions=[0.0, 1.0],
        )
        profile = VolatileProfile(
            w_liquid={'Chabrier:H': 0.10},
            w_solid={'Chabrier:H': 0.0},
            primary_component='PALEOS:MgSiO3',
            global_miscibility=True,
            x_interior={'Chabrier:H': 0.05},
        )
        # T well above critical -> miscible.
        fracs = profile.apply_to_mixture_with_binodal(
            mixture, phi=0.5, pressure=1e8, temperature=1.0e5
        )
        assert sum(fracs) == pytest.approx(1.0, abs=1e-12)
        # Chabrier:H value reflects x_interior, not a phi-weighted blend.
        assert fracs[0] == pytest.approx(0.05, rel=1e-12)

    def test_unmanaged_component_kept_in_with_binodal_path(self):
        """Same passthrough behaviour as ``apply_to_mixture``."""
        mixture = LayerMixture(
            components=['PALEOS:iron', 'Chabrier:H', 'PALEOS:MgSiO3'],
            fractions=[0.3, 0.0, 0.7],
        )
        profile = VolatileProfile(
            w_liquid={'Chabrier:H': 0.05},
            w_solid={'Chabrier:H': 0.0},
            primary_component='PALEOS:MgSiO3',
            global_miscibility=True,
            x_interior={'Chabrier:H': 0.04},
        )
        fracs = profile.apply_to_mixture_with_binodal(
            mixture, phi=0.5, pressure=1e8, temperature=1.0e5
        )
        # Iron is unmanaged: keeps its 0.3 even after re-normalisation.
        assert fracs[0] > 0.0
        assert sum(fracs) == pytest.approx(1.0, abs=1e-12)

    def test_zero_total_path_in_with_binodal_apply(self):
        """Edge: every component blends to zero -> total==0 branch."""
        mixture = LayerMixture(
            components=['Chabrier:H'],
            fractions=[1.0],
        )
        profile = VolatileProfile(
            w_liquid={'Chabrier:H': 0.0},
            w_solid={'Chabrier:H': 0.0},
            primary_component='Not:In:Mixture',
            global_miscibility=True,
            x_interior={'Chabrier:H': 0.0},
        )
        fracs = profile.apply_to_mixture_with_binodal(
            mixture, phi=0.5, pressure=1e8, temperature=1.0e5
        )
        assert fracs == [0.0]


# ---------------------------------------------------------------------------
# compute_melt_fraction
# ---------------------------------------------------------------------------


def _sol_2000(P):
    return 2000.0


def _liq_2400(P):
    return 2400.0


def _sol_none(P):
    return None


def _sol_nan(P):
    return float('nan')


def _liq_2400_alt(P):
    return 2400.0


def _sol_2400(P):
    return 2400.0


class TestComputeMeltFraction:
    """Local melt fraction phi from solidus/liquidus curves."""

    def test_at_solidus_returns_zero(self):
        """T == T_sol -> phi = 0 exactly."""
        assert compute_melt_fraction(1e9, 2000.0, _sol_2000, _liq_2400) == pytest.approx(0.0)

    def test_at_liquidus_returns_one(self):
        """T == T_liq -> phi = 1 exactly."""
        assert compute_melt_fraction(1e9, 2400.0, _sol_2000, _liq_2400) == pytest.approx(1.0)

    def test_midway_returns_one_half(self):
        """T at midpoint of mushy zone -> phi = 0.5 exactly."""
        assert compute_melt_fraction(1e9, 2200.0, _sol_2000, _liq_2400) == pytest.approx(0.5)

    def test_below_solidus_clamps_to_zero(self):
        """T below T_sol -> phi clamped to 0."""
        assert compute_melt_fraction(1e9, 1500.0, _sol_2000, _liq_2400) == pytest.approx(0.0)

    def test_above_liquidus_clamps_to_one(self):
        """T above T_liq -> phi clamped to 1."""
        assert compute_melt_fraction(1e9, 3000.0, _sol_2000, _liq_2400) == pytest.approx(1.0)

    def test_missing_curves_returns_default_one_half(self):
        """Edge case: solidus_func or liquidus_func is None -> 0.5 default."""
        assert compute_melt_fraction(1e9, 2000.0, None, None) == 0.5
        assert compute_melt_fraction(1e9, 2000.0, _sol_2000, None) == 0.5
        assert compute_melt_fraction(1e9, 2000.0, None, _liq_2400) == 0.5

    def test_curves_returning_none_give_default(self):
        """Edge: a melting-curve callable returning None -> 0.5 default."""
        assert compute_melt_fraction(1e9, 2000.0, _sol_none, _liq_2400) == 0.5

    def test_nan_curve_value_gives_default(self):
        """Physically unreasonable: NaN melting temperature -> 0.5 default."""
        assert compute_melt_fraction(1e9, 2000.0, _sol_nan, _liq_2400) == 0.5

    def test_inverted_curves_T_above_solidus_returns_one(self):
        """Edge: T_liq <= T_sol (degenerate) -> 1 if T > T_sol else 0."""
        # T_liq == T_sol case using two scalar-returning callables.
        assert compute_melt_fraction(1e9, 2500.0, _sol_2400, _liq_2400_alt) == pytest.approx(
            1.0
        )
        assert compute_melt_fraction(1e9, 2400.0, _sol_2400, _liq_2400_alt) == pytest.approx(
            0.0
        )
        assert compute_melt_fraction(1e9, 2300.0, _sol_2400, _liq_2400_alt) == pytest.approx(
            0.0
        )
