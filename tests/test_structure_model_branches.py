"""Branch tests for ``zalmoxis.structure_model``.

The full structure-solver flow is exercised by integration tests; this
file pins narrow branches that the integration tests do not
discriminate against:

1. ``get_layer_mixture`` mass-based dispatch (core, mantle, ice_layer).
2. ``coupled_odes`` zero-derivative paths: None/NaN density, P<=0,
   NaN pressure.
3. ``coupled_odes`` radius=0 analytic limit dg/dr = (4/3) pi G rho.
4. ``solve_structure`` truncation/padding when the terminal event
   fires before the outermost grid point.
5. ``solve_structure`` JAX fallback on ValueError.
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from zalmoxis.constants import G
from zalmoxis.mixing import LayerMixture
from zalmoxis.structure_model import coupled_odes, get_layer_mixture, solve_structure

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


class TestCoupledOdesNonphysicalPressure:
    """Pressure <= 0 or NaN freezes the ODE state for that shell."""

    def _call(self, three_layer_mixtures, pressure):
        return coupled_odes(
            radius=1e6,
            y=[1e23, 5.0, pressure],
            cmb_mass=2e23,
            core_mantle_mass=4e23,
            layer_mixtures=three_layer_mixtures,
            interpolation_cache={},
            material_dictionaries={},
            temperature=3000.0,
            solidus_func=None,
            liquidus_func=None,
        )

    def test_zero_pressure_returns_zero_derivatives(self, three_layer_mixtures):
        assert self._call(three_layer_mixtures, 0.0) == [0.0, 0.0, 0.0]

    def test_negative_pressure_returns_zero_derivatives(self, three_layer_mixtures):
        """Negative P is unphysical; the integrator's terminal event will
        catch the zero crossing one step later. Until then the state is frozen."""
        assert self._call(three_layer_mixtures, -1e8) == [0.0, 0.0, 0.0]

    def test_nan_pressure_returns_zero_derivatives(self, three_layer_mixtures):
        assert self._call(three_layer_mixtures, float('nan')) == [0.0, 0.0, 0.0]


class TestCoupledOdesRadiusOriginLimit:
    """At r=0 the 2g/r term is singular; coupled_odes uses dg/dr = (4/3) pi G rho."""

    def test_radius_zero_uses_analytic_dgdr(self, three_layer_mixtures):
        """At r=0 dg/dr should equal (4/3) pi G rho exactly, not 4 pi G rho - 2g/r."""
        rho = 5500.0
        with mock.patch(
            'zalmoxis.structure_model.calculate_mixed_density',
            return_value=rho,
        ):
            dy = coupled_odes(
                radius=0.0,
                y=[0.0, 0.0, 1e11],  # central state: M=0, g=0
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                layer_mixtures=three_layer_mixtures,
                interpolation_cache={},
                material_dictionaries={},
                temperature=3000.0,
                solidus_func=None,
                liquidus_func=None,
            )
        # dM/dr at r=0: 4 pi (0)^2 rho = 0
        assert dy[0] == pytest.approx(0.0)
        # dg/dr at r=0: analytic limit (4/3) pi G rho, not the divergent r=0 form
        expected_dgdr = (4.0 / 3.0) * np.pi * G * rho
        assert dy[1] == pytest.approx(expected_dgdr)
        # dP/dr at r=0: -rho * g = -rho * 0 = 0
        assert dy[2] == pytest.approx(0.0)

    def test_radius_zero_dgdr_three_times_smaller_than_naive(self, three_layer_mixtures):
        """Sanity: the r=0 limit is exactly 4*pi*G*rho/3, i.e. 1/3 of the
        non-zero-r form 4*pi*G*rho. Distinguishes the correct L'Hopital
        limit from a sign or factor-of-3 bug."""
        rho = 9000.0
        with mock.patch(
            'zalmoxis.structure_model.calculate_mixed_density',
            return_value=rho,
        ):
            dy_origin = coupled_odes(
                radius=0.0,
                y=[0.0, 0.0, 1e11],
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                layer_mixtures=three_layer_mixtures,
                interpolation_cache={},
                material_dictionaries={},
                temperature=3000.0,
                solidus_func=None,
                liquidus_func=None,
            )
        # Naive r->0 of 4 pi G rho - 2g/r with g=0 would give 4 pi G rho;
        # the analytic limit is exactly 1/3 of that.
        assert dy_origin[1] == pytest.approx((4.0 / 3.0) * np.pi * G * rho)


class TestSolveStructureNonTdep:
    """Cover the non-Tdep (single solve_ivp) branch of solve_structure."""

    def test_constant_density_yields_monotonic_pressure(self, two_layer_mixtures):
        """With a constant density the pressure profile is monotonically
        decreasing with radius and the mass profile is monotonically
        increasing — both physical invariants of the ODE system."""
        radii = np.linspace(1.0, 1e6, 50)
        rho_const = 5000.0
        with mock.patch(
            'zalmoxis.structure_model.calculate_mixed_density',
            return_value=rho_const,
        ):
            mass, gravity, pressure = solve_structure(
                layer_mixtures=two_layer_mixtures,
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                radii=radii,
                adaptive_radial_fraction=0.5,
                relative_tolerance=1e-6,
                absolute_tolerance=1e-8,
                maximum_step=1e5,
                material_dictionaries={},
                interpolation_cache={},
                y0=[0.0, 0.0, 1e12],  # large central P so we don't terminate early
                solidus_func=None,
                liquidus_func=None,
            )
        assert len(mass) == len(radii)
        assert len(gravity) == len(radii)
        assert len(pressure) == len(radii)
        # Mass strictly increases with radius for positive density
        assert np.all(np.diff(mass) > 0)
        # Pressure strictly decreases with radius for positive density and gravity
        assert np.all(np.diff(pressure) < 0)


class TestSolveStructurePadding:
    """When the terminal event truncates the solution, the output is padded
    to len(radii) with zero pressure and held mass/gravity."""

    def test_padding_when_terminal_event_truncates(self, two_layer_mixtures):
        """Force a low central pressure so the integrator terminates before
        reaching the outermost grid point. Output arrays must still match
        len(radii); the padded slice has pressure=0 (vacuum)."""
        radii = np.linspace(1.0, 1e7, 50)
        rho_const = 8000.0
        with mock.patch(
            'zalmoxis.structure_model.calculate_mixed_density',
            return_value=rho_const,
        ):
            mass, gravity, pressure = solve_structure(
                layer_mixtures=two_layer_mixtures,
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                radii=radii,
                adaptive_radial_fraction=0.5,
                relative_tolerance=1e-6,
                absolute_tolerance=1e-8,
                maximum_step=1e5,
                material_dictionaries={},
                interpolation_cache={},
                # Tiny central pressure: gradient -rho*g exhausts P quickly
                y0=[0.0, 0.0, 1e3],
                solidus_func=None,
                liquidus_func=None,
            )
        assert len(pressure) == len(radii)
        # Last entries (the padded shells) must be exactly zero
        assert pressure[-1] == 0.0
        # Some early shells should still hold non-zero pressure
        assert pressure[0] > 0.0


class TestSolveStructureTdepBranch:
    """Cover the temperature-dependent two-stage solve_ivp branch.

    When any layer mixture has a Tdep EOS component, solve_structure
    splits the radial grid into an adaptive inner part and a max-step
    outer part. Mock ``any_component_is_tdep`` and ``calculate_mixed_density``
    to exercise this branch without needing real Tdep tables.
    """

    def test_tdep_two_stage_solve_concatenates_correctly(self, two_layer_mixtures):
        radii = np.linspace(1.0, 1e6, 60)
        rho_const = 5500.0

        def temp_fn(r, P):
            return 3000.0

        with (
            mock.patch(
                'zalmoxis.structure_model.any_component_is_tdep',
                return_value=True,
            ),
            mock.patch(
                'zalmoxis.structure_model.calculate_mixed_density',
                return_value=rho_const,
            ),
        ):
            mass, gravity, pressure = solve_structure(
                layer_mixtures=two_layer_mixtures,
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                radii=radii,
                adaptive_radial_fraction=0.5,
                relative_tolerance=1e-6,
                absolute_tolerance=1e-8,
                maximum_step=1e5,
                material_dictionaries={},
                interpolation_cache={},
                y0=[0.0, 0.0, 1e12],
                solidus_func=None,
                liquidus_func=None,
                temperature_function=temp_fn,
            )
        # Concatenated arrays must match the input grid length exactly
        assert len(mass) == len(radii)
        assert len(gravity) == len(radii)
        assert len(pressure) == len(radii)
        # The two-stage solve must produce a continuous monotonic profile
        assert np.all(np.diff(mass) > 0)
        assert np.all(np.diff(pressure) < 0)

    def test_tdep_terminal_event_in_first_stage(self, two_layer_mixtures):
        """Terminal event in the inner (adaptive) stage skips the outer stage
        and uses sol1 directly. Force a low central pressure to trigger this."""
        radii = np.linspace(1.0, 1e7, 40)

        def temp_fn(r, P):
            return 3000.0

        with (
            mock.patch(
                'zalmoxis.structure_model.any_component_is_tdep',
                return_value=True,
            ),
            mock.patch(
                'zalmoxis.structure_model.calculate_mixed_density',
                return_value=8000.0,
            ),
        ):
            mass, gravity, pressure = solve_structure(
                layer_mixtures=two_layer_mixtures,
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                radii=radii,
                adaptive_radial_fraction=0.5,
                relative_tolerance=1e-6,
                absolute_tolerance=1e-8,
                maximum_step=1e5,
                material_dictionaries={},
                interpolation_cache={},
                # Tiny central P + high rho exhausts pressure inside the inner stage
                y0=[0.0, 0.0, 1e3],
                solidus_func=None,
                liquidus_func=None,
                temperature_function=temp_fn,
            )
        assert len(pressure) == len(radii)
        # Padding tail must be exactly zero (terminal event truncated, then padded)
        assert pressure[-1] == 0.0


class TestSolveStructureJAXFallback:
    """The JAX path falls back to the numpy path on ValueError, logging a
    warning. Verify the fallback yields the same ODE-driven contract."""

    def test_jax_value_error_falls_back_to_numpy(self, two_layer_mixtures, caplog):
        """If the JAX wrapper raises ValueError (unsupported config), the
        function logs a warning and returns the numpy path's output."""
        radii = np.linspace(1.0, 1e6, 30)
        with (
            mock.patch(
                'zalmoxis.jax_eos.wrapper.solve_structure_via_jax',
                side_effect=ValueError('unsupported config'),
            ),
            mock.patch(
                'zalmoxis.structure_model.calculate_mixed_density',
                return_value=5000.0,
            ),
        ):
            with caplog.at_level('WARNING', logger='zalmoxis.structure_model'):
                mass, gravity, pressure = solve_structure(
                    layer_mixtures=two_layer_mixtures,
                    cmb_mass=2e23,
                    core_mantle_mass=4e23,
                    radii=radii,
                    adaptive_radial_fraction=0.5,
                    relative_tolerance=1e-6,
                    absolute_tolerance=1e-8,
                    maximum_step=1e5,
                    material_dictionaries={},
                    interpolation_cache={},
                    y0=[0.0, 0.0, 1e12],
                    solidus_func=None,
                    liquidus_func=None,
                    use_jax=True,
                )
        assert len(mass) == len(radii)
        # Warning must be emitted on fallback so users notice unsupported configs
        assert any('fell back to numpy path' in m for m in caplog.messages)
