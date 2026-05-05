"""Branch tests for ``zalmoxis.structure_model.get_layer_mixture`` and the
None-density safe path in ``coupled_odes``.

The full structure-solver flow is exercised by integration tests; this
file pins two narrow branches that the integration tests do not
discriminate against:

1. ``get_layer_mixture`` returns the ``ice_layer`` mixture when the
   planet has a 3-layer composition and the query mass is in the
   outermost shell.
2. ``coupled_odes`` returns zero derivatives when the EOS dispatch
   yields ``None`` or NaN (the ODE state freezes for that radius).
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from zalmoxis.mixing import LayerMixture
from zalmoxis.structure_model import coupled_odes, get_layer_mixture

pytestmark = pytest.mark.unit


@pytest.fixture
def three_layer_mixtures():
    """Three-layer (iron + silicate + ice) composition."""
    return {
        'core': LayerMixture(['Seager2007:iron'], [1.0]),
        'mantle': LayerMixture(['Seager2007:MgSiO3'], [1.0]),
        'ice_layer': LayerMixture(['Seager2007:H2O'], [1.0]),
    }


@pytest.fixture
def two_layer_mixtures():
    """Two-layer (iron + silicate) composition; no ice envelope."""
    return {
        'core': LayerMixture(['Seager2007:iron'], [1.0]),
        'mantle': LayerMixture(['Seager2007:MgSiO3'], [1.0]),
    }


class TestGetLayerMixture:
    """Mass-based dispatch into core / mantle / ice_layer."""

    def test_core_returned_below_cmb(self, three_layer_mixtures):
        m = get_layer_mixture(
            mass=1e23,  # below cmb=2e23
            cmb_mass=2e23,
            core_mantle_mass=4e23,
            layer_mixtures=three_layer_mixtures,
        )
        assert m.components == ['Seager2007:iron']

    def test_mantle_returned_between_cmb_and_cmm(self, three_layer_mixtures):
        m = get_layer_mixture(
            mass=3e23,  # between cmb=2e23 and core_mantle=4e23
            cmb_mass=2e23,
            core_mantle_mass=4e23,
            layer_mixtures=three_layer_mixtures,
        )
        assert m.components == ['Seager2007:MgSiO3']

    def test_ice_layer_returned_above_cmm(self, three_layer_mixtures):
        """Mass at or above ``core_mantle_mass`` returns the ice mixture
        when ``ice_layer`` is configured. This is the previously-uncovered
        branch."""
        m = get_layer_mixture(
            mass=5e23,  # above core_mantle=4e23
            cmb_mass=2e23,
            core_mantle_mass=4e23,
            layer_mixtures=three_layer_mixtures,
        )
        assert m.components == ['Seager2007:H2O']

    def test_two_layer_falls_through_to_mantle(self, two_layer_mixtures):
        """Without ``ice_layer``, mass above ``core_mantle_mass`` falls through
        to mantle (no ice envelope)."""
        m = get_layer_mixture(
            mass=5e23,
            cmb_mass=2e23,
            core_mantle_mass=4e23,
            layer_mixtures=two_layer_mixtures,
        )
        assert m.components == ['Seager2007:MgSiO3']

    def test_boundary_inclusion_at_cmb(self, three_layer_mixtures):
        """Mass exactly at ``cmb_mass`` is mantle-side (relative tol 1e-12)."""
        m = get_layer_mixture(
            mass=2e23,
            cmb_mass=2e23,
            core_mantle_mass=4e23,
            layer_mixtures=three_layer_mixtures,
        )
        assert m.components == ['Seager2007:MgSiO3']

    def test_negative_mass_assigns_to_core(self, three_layer_mixtures):
        """Unphysical negative mass still dispatches to core (defensive)."""
        m = get_layer_mixture(
            mass=-1.0,
            cmb_mass=2e23,
            core_mantle_mass=4e23,
            layer_mixtures=three_layer_mixtures,
        )
        assert m.components == ['Seager2007:iron']


class TestCoupledOdesNoneDensity:
    """When the EOS lookup returns ``None``/NaN the ODE state freezes."""

    def test_none_density_returns_zero_derivatives(self, three_layer_mixtures):
        """Mock the mixed-density helper to return None; the ODEs produce
        zero derivatives so the integrator stops advancing this shell."""
        with mock.patch(
            'zalmoxis.structure_model.calculate_mixed_density',
            return_value=None,
        ):
            dy = coupled_odes(
                radius=1e6,
                y=[1e23, 5.0, 1e10],  # plausible mid-mantle state
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                layer_mixtures=three_layer_mixtures,
                interpolation_cache={},
                material_dictionaries={},
                temperature=3000.0,
                solidus_func=None,
                liquidus_func=None,
                mushy_zone_factors=None,
                condensed_rho_min=None,
                condensed_rho_scale=None,
                binodal_T_scale=None,
            )
        assert dy == [0.0, 0.0, 0.0]

    def test_nan_density_returns_zero_derivatives(self, three_layer_mixtures):
        """NaN density is also treated as a frozen-shell signal."""
        with mock.patch(
            'zalmoxis.structure_model.calculate_mixed_density',
            return_value=float('nan'),
        ):
            dy = coupled_odes(
                radius=1e6,
                y=[1e23, 5.0, 1e10],
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                layer_mixtures=three_layer_mixtures,
                interpolation_cache={},
                material_dictionaries={},
                temperature=3000.0,
                solidus_func=None,
                liquidus_func=None,
                mushy_zone_factors=None,
                condensed_rho_min=None,
                condensed_rho_scale=None,
                binodal_T_scale=None,
            )
        assert dy == [0.0, 0.0, 0.0]

    def test_finite_density_yields_normal_derivatives(self, three_layer_mixtures):
        """With a finite density the ODEs return non-trivial derivatives,
        confirming the zero-derivative path is the genuine NaN guard, not a
        sign error elsewhere."""
        with mock.patch(
            'zalmoxis.structure_model.calculate_mixed_density',
            return_value=5500.0,
        ):
            dy = coupled_odes(
                radius=1e6,
                y=[1e23, 5.0, 1e10],
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                layer_mixtures=three_layer_mixtures,
                interpolation_cache={},
                material_dictionaries={},
                temperature=3000.0,
                solidus_func=None,
                liquidus_func=None,
                mushy_zone_factors=None,
                condensed_rho_min=None,
                condensed_rho_scale=None,
                binodal_T_scale=None,
            )
        # dM/dr = 4 pi r^2 rho > 0
        assert dy[0] > 0
        # dP/dr = -rho g < 0 (pressure decreases outward)
        assert dy[2] < 0
        assert np.isfinite(dy[1])
