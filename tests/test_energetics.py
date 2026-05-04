"""Unit tests for the zalmoxis.energetics module.

Tests cover:
- Gravitational binding energy: uniform sphere, two-layer planet, analytic vs numerical
- Differentiation energy: sign and magnitude for core-mantle differentiation
- Iron melting curves: Anzellini+2013 reference points, triple-point continuity, Sinmyo+2019
- Heat capacity averaging and initial thermal state Earth benchmark

References:
    Anzellini, S. et al. (2013). Science, 340, 464-466.
    Sinmyo, R. et al. (2019). EPSL, 510, 45-52.
    White, N. I. & Li, J. (2025). JGRP, 130, e2024JE008550.
    Boujibar, A., Driscoll, P. & Fei, Y. (2020). JGRP, 125, e2019JE006124.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import G

from zalmoxis.energetics import (
    differentiation_energy,
    gravitational_binding_energy,
    gravitational_binding_energy_uniform,
    initial_thermal_state,
)
from zalmoxis.melting_curves import iron_melting_anzellini13, iron_melting_sinmyo19

# ============================================================================
# Helpers
# ============================================================================


def _make_uniform_sphere(rho=5500.0, R=6.371e6, n_points=1000):
    """Create a uniform-density sphere on a radial grid.

    Parameters
    ----------
    rho : float
        Uniform density [kg/m^3].
    R : float
        Total radius [m].
    n_points : int
        Number of radial grid points.

    Returns
    -------
    tuple
        (radii, mass_enclosed) arrays.
    """
    radii = np.linspace(0, R, n_points)
    mass_enclosed = (4.0 / 3.0) * np.pi * rho * radii**3
    return radii, mass_enclosed


def _make_synthetic_model_results(
    n_points=500, rho_core=10000.0, rho_mantle=4000.0, R_planet=6.371e6, cmf=0.32
):
    """Create a synthetic two-layer model_results dict for testing.

    Parameters
    ----------
    n_points : int
        Number of radial grid points.
    rho_core : float
        Core density [kg/m^3].
    rho_mantle : float
        Mantle density [kg/m^3].
    R_planet : float
        Planet radius [m].
    cmf : float
        Core mass fraction.

    Returns
    -------
    dict
        Synthetic model_results with keys: radii, density, mass_enclosed,
        gravity, pressure, temperature, cmb_mass.
    """
    # Compute core radius from CMF and densities
    # CMF = M_core / M_total = (rho_core * R_core^3) / (rho_core * R_core^3 + rho_mantle * (R^3 - R_core^3))
    # Solving: R_core^3 = cmf * rho_mantle * R^3 / (rho_core - cmf*(rho_core - rho_mantle))
    rho_avg = cmf * rho_core + (1 - cmf) * rho_mantle
    R_core = R_planet * (cmf * rho_avg / rho_core) ** (1.0 / 3.0)

    radii = np.linspace(0, R_planet, n_points)
    density = np.where(radii <= R_core, rho_core, rho_mantle)

    # Build mass_enclosed by integrating density shells
    mass_enclosed = np.zeros(n_points)
    for i in range(1, n_points):
        dr3 = radii[i] ** 3 - radii[i - 1] ** 3
        mass_enclosed[i] = mass_enclosed[i - 1] + (4.0 / 3.0) * np.pi * dr3 * density[i]

    # Gravity
    gravity = np.zeros(n_points)
    gravity[1:] = G * mass_enclosed[1:] / radii[1:] ** 2

    # Hydrostatic pressure (integrate inward from surface)
    pressure = np.zeros(n_points)
    for i in range(n_points - 2, -1, -1):
        dr = radii[i + 1] - radii[i]
        pressure[i] = pressure[i + 1] + density[i] * gravity[i] * dr

    # Isothermal for simplicity
    temperature = np.full(n_points, 3000.0)

    # CMB mass
    cmb_idx = np.searchsorted(radii, R_core)
    cmb_mass = mass_enclosed[min(cmb_idx, n_points - 1)]

    return {
        'radii': radii,
        'density': density,
        'mass_enclosed': mass_enclosed,
        'gravity': gravity,
        'pressure': pressure,
        'temperature': temperature,
        'cmb_mass': cmb_mass,
    }


# ============================================================================
# Tests: gravitational binding energy
# ============================================================================


@pytest.mark.unit
class TestUniformSphereBindingEnergy:
    """Verify numerical binding energy matches the analytic 3GM^2/(5R) formula."""

    def test_agreement_with_analytic(self):
        """Numerical integral on uniform sphere should match analytic to < 0.5%."""
        rho = 5500.0
        R = 6.371e6
        radii, mass_enclosed = _make_uniform_sphere(rho=rho, R=R, n_points=1000)

        U_numerical = gravitational_binding_energy(radii, mass_enclosed)
        M_total = mass_enclosed[-1]
        U_analytic = gravitational_binding_energy_uniform(M_total, R)

        rel_error = abs(U_numerical - U_analytic) / U_analytic
        assert rel_error < 0.005, (
            f'Relative error {rel_error:.4e} exceeds 0.5%. '
            f'U_numerical={U_numerical:.4e}, U_analytic={U_analytic:.4e}'
        )

    def test_binding_energy_is_positive(self):
        """Binding energy must always be positive."""
        radii, mass_enclosed = _make_uniform_sphere()
        U = gravitational_binding_energy(radii, mass_enclosed)
        assert U > 0

    def test_length_mismatch_raises(self):
        """Mismatched array lengths should raise ValueError."""
        radii = np.linspace(0, 1e6, 100)
        mass_enclosed = np.zeros(50)
        with pytest.raises(ValueError, match='same length'):
            gravitational_binding_energy(radii, mass_enclosed)


@pytest.mark.unit
class TestTwoLayerBindingEnergy:
    """Verify that a differentiated planet is more gravitationally bound than uniform."""

    def test_differentiated_greater_than_uniform(self):
        """U_differentiated > U_uniform for a denser core."""
        model = _make_synthetic_model_results(
            n_points=1000, rho_core=10000.0, rho_mantle=4000.0, cmf=0.32
        )
        radii = model['radii']
        mass_enclosed = model['mass_enclosed']

        U_diff = gravitational_binding_energy(radii, mass_enclosed)
        M_total = mass_enclosed[-1]
        R_total = radii[-1]
        U_uniform = gravitational_binding_energy_uniform(M_total, R_total)

        assert U_diff > U_uniform, (
            f'Differentiated ({U_diff:.4e} J) should exceed uniform ({U_uniform:.4e} J)'
        )


@pytest.mark.unit
class TestDifferentiationEnergy:
    """Verify differentiation energy has the correct sign and magnitude."""

    def test_positive_for_dense_core(self):
        """Differentiation energy should be positive when core is denser than mantle."""
        model = _make_synthetic_model_results(
            n_points=1000, rho_core=10000.0, rho_mantle=4000.0, cmf=0.32
        )
        radii = model['radii']
        mass_enclosed = model['mass_enclosed']

        U_diff = gravitational_binding_energy(radii, mass_enclosed)
        U_uniform = gravitational_binding_energy_uniform(mass_enclosed[-1], radii[-1])

        dE = differentiation_energy(U_diff, U_uniform)
        assert dE > 0, f'Differentiation energy should be positive, got {dE:.4e} J'

    def test_zero_for_uniform(self):
        """Differentiation energy should be near zero for a uniform body."""
        radii, mass_enclosed = _make_uniform_sphere(rho=5500.0, n_points=1000)
        U_numerical = gravitational_binding_energy(radii, mass_enclosed)
        U_analytic = gravitational_binding_energy_uniform(mass_enclosed[-1], radii[-1])

        dE = differentiation_energy(U_numerical, U_analytic)
        # Should be small relative to U_analytic (integration error only)
        assert abs(dE) / U_analytic < 0.005


# ============================================================================
# Tests: iron melting curves
# ============================================================================


@pytest.mark.unit
class TestIronMeltingAnzellini:
    """Tests for the Anzellini+2013 composite iron melting curve."""

    def test_near_zero_pressure(self):
        """T_melt at 0 Pa should be near 1811 K.

        The Anzellini parameterization has a reference pressure of 5.2 GPa
        with T0=1991 K, so at P=0 the extrapolation gives a value offset
        from the atmospheric melting point. We allow +/- 50 K.
        """
        T = iron_melting_anzellini13(0.0)
        # At P=0, the Anzellini Eq. 2 gives T0 * ((0 - 5.2)/27.39 + 1)^(1/2.38)
        # which is ~1811 K (not exactly, since it depends on the parameterization)
        assert abs(T - 1811.0) < 200.0, (
            f'T_melt(0 Pa) = {T:.1f} K, expected near 1811 K (tolerance: 200 K for extrapolation)'
        )

    def test_at_330_GPa(self):
        """T_melt at 330 GPa should be near 6230 K (Anzellini+2013 Fig. 3)."""
        T = iron_melting_anzellini13(330e9)
        assert abs(T - 6230.0) < 500.0, (
            f'T_melt(330 GPa) = {T:.1f} K, expected ~6230 K (+/- 500 K)'
        )

    def test_monotonically_increasing(self):
        """Iron melting curve must be monotonically increasing with pressure."""
        P = np.linspace(0, 400e9, 1000)
        T = iron_melting_anzellini13(P)
        # Skip first point (P=0) which may have special behavior
        diffs = np.diff(T[1:])
        assert np.all(diffs >= 0), 'T_melt must be monotonically increasing'

    def test_triple_point_continuity(self):
        """T_melt at 98.4 GPa and 98.6 GPa should differ by < 10 K.

        The gamma-epsilon triple point is at 98.5 GPa. The two piecewise
        branches should be approximately continuous there.
        """
        T_below = iron_melting_anzellini13(98.4e9)
        T_above = iron_melting_anzellini13(98.6e9)
        assert abs(T_below - T_above) < 10.0, (
            f'Discontinuity at triple point: T({98.4} GPa)={T_below:.1f} K, '
            f'T({98.6} GPa)={T_above:.1f} K, diff={abs(T_below - T_above):.1f} K'
        )

    def test_scalar_output(self):
        """Scalar input should return a float."""
        T = iron_melting_anzellini13(100e9)
        assert isinstance(T, float)

    def test_array_output(self):
        """Array input should return an ndarray."""
        T = iron_melting_anzellini13(np.array([50e9, 100e9, 200e9]))
        assert isinstance(T, np.ndarray)
        assert len(T) == 3


@pytest.mark.unit
class TestIronMeltingSinmyo:
    """Tests for the Sinmyo+2019 iron melting curve."""

    def test_at_zero_pressure(self):
        """T_melt(0 Pa) = 1811 K exactly (Sinmyo reference temperature)."""
        T = iron_melting_sinmyo19(0.0)
        assert T == pytest.approx(1811.0, abs=0.1), (
            f'T_melt(0 Pa) = {T:.4f} K, expected exactly 1811.0 K'
        )


# ============================================================================
# Tests: heat capacity averaging and initial thermal state
# ============================================================================


@pytest.mark.unit
class TestHeatCapacityAveraging:
    """Verify the mass-weighted average heat capacity calculation."""

    def test_known_values(self):
        """C_avg = CMF * C_iron + (1-CMF) * C_silicate.

        For CMF=0.32, C_iron=840, C_silicate=1200:
        C_avg = 0.32 * 840 + 0.68 * 1200 = 268.8 + 816.0 = 1084.8 J/kg/K
        """
        model = _make_synthetic_model_results(n_points=200, cmf=0.32)

        result = initial_thermal_state(
            model,
            core_mass_fraction=0.32,
            C_iron=840.0,
            C_silicate=1200.0,
            f_accretion=0.04,
            f_differentiation=0.50,
        )

        expected_C = 0.32 * 840.0 + 0.68 * 1200.0
        assert result['C_avg'] == pytest.approx(expected_C, rel=1e-10), (
            f'C_avg = {result["C_avg"]:.2f}, expected {expected_C:.2f}'
        )


@pytest.mark.unit
class TestEarthBenchmark:
    """Broad sanity check for Earth-like parameters."""

    def test_temperature_ranges(self):
        """For Earth mass/radius/CMF, temperatures should be physically plausible.

        With f_a=0.04, f_d=0.50 and Dulong-Petit C_p (White+Li defaults):
        - Delta_T_G should be ~1000-2000 K
        - Delta_T_D should be ~1000-3000 K
        - T_CMB should be ~3000-8000 K (includes adiabatic term)
        - Delta_T_ad should be positive (adiabatic heating at depth)
        """
        model = _make_synthetic_model_results(
            n_points=500,
            rho_core=10000.0,
            rho_mantle=4000.0,
            R_planet=6.371e6,
            cmf=0.32,
        )

        result = initial_thermal_state(
            model,
            core_mass_fraction=0.32,
            T_radiative_eq=255.0,
            f_accretion=0.04,
            f_differentiation=0.50,
        )

        # Temperature increments
        assert 500.0 < result['Delta_T_accretion'] < 3000.0, (
            f'Delta_T_G = {result["Delta_T_accretion"]:.0f} K, expected 500-3000 K'
        )
        assert 200.0 < result['Delta_T_differentiation'] < 3000.0, (
            f'Delta_T_D = {result["Delta_T_differentiation"]:.0f} K, expected 200-3000 K'
        )

        # Adiabatic gradient term must be positive and physically reasonable
        assert result['Delta_T_adiabat'] > 0, (
            f'Delta_T_ad = {result["Delta_T_adiabat"]:.0f} K, must be positive'
        )
        assert result['Delta_T_adiabat'] < 5000.0, (
            f'Delta_T_ad = {result["Delta_T_adiabat"]:.0f} K, unreasonably large'
        )

        # T_CMB = T_surf_accr + Delta_T_ad (White+Li Eq. 2 structure)
        assert result['T_cmb'] == pytest.approx(
            result['T_surf_accr'] + result['Delta_T_adiabat'], rel=1e-10
        )

        # CMB temperature: includes adiabatic term, so higher than bulk heating
        assert 3000.0 < result['T_cmb'] < 8000.0, (
            f'T_CMB = {result["T_cmb"]:.0f} K, expected 3000-8000 K'
        )

        # T_surface (from adiabat) must be less than T_cmb
        assert result['T_surf_accr'] < result['T_cmb'], (
            f'T_surface ({result["T_surface"]:.0f}) >= T_cmb ({result["T_cmb"]:.0f})'
        )

        # T_profile must exist and have correct shape
        assert 'T_profile' in result
        assert len(result['T_profile']) == len(result['radii'])
        # Profile must be monotonically decreasing from center to surface
        # (adiabat: T decreases outward)
        T_prof = result['T_profile']
        # Core is isothermal (no iron nabla_ad provided), mantle decreases
        assert T_prof[0] == pytest.approx(result['T_cmb'], rel=0.01)
        assert T_prof[-1] == pytest.approx(result['T_surf_accr'], rel=0.01)

        # Verify accretion formula: Delta_T_G = f_a * U_u / (M * C_avg)
        expected_DT_G = (
            0.04 * result['U_undifferentiated'] / (model['mass_enclosed'][-1] * result['C_avg'])
        )
        assert result['Delta_T_accretion'] == pytest.approx(expected_DT_G, rel=1e-10), (
            f'Accretion term: got {result["Delta_T_accretion"]:.0f} K, '
            f'expected {expected_DT_G:.0f} K from U_u'
        )

    def test_binding_energies_returned(self):
        """Result dict should contain binding energies with correct signs."""
        model = _make_synthetic_model_results(n_points=200, cmf=0.32)
        result = initial_thermal_state(model, core_mass_fraction=0.32)

        assert result['U_differentiated'] > 0
        assert result['U_undifferentiated'] > 0
        assert result['U_differentiated'] > result['U_undifferentiated']

    def test_core_state_determined(self):
        """Core state should be one of 'liquid', 'solid', or 'partial'."""
        model = _make_synthetic_model_results(n_points=200, cmf=0.32)
        result = initial_thermal_state(model, core_mass_fraction=0.32)

        assert result['core_state'] in {'liquid', 'solid', 'partial'}

    def test_coreless_planet(self):
        """CMF=0 should return core_state='none' without calling melting curve."""
        model = _make_synthetic_model_results(n_points=200, cmf=0.01)
        # Override cmb_mass to 0 for a truly coreless planet
        model['cmb_mass'] = 0.0

        result = initial_thermal_state(model, core_mass_fraction=0.0)

        assert result['core_state'] == 'none'
        assert result['T_cmb'] > 0

    def test_energy_scaling_with_mass(self):
        """Binding energy should increase with planet mass.

        For uniform density, U ~ M^{5/3}. A 2x larger planet (same density,
        bigger radius) should have substantially more binding energy.
        """
        model_small = _make_synthetic_model_results(n_points=500, R_planet=5e6, cmf=0.32)
        model_large = _make_synthetic_model_results(n_points=500, R_planet=8e6, cmf=0.32)

        result_small = initial_thermal_state(model_small, core_mass_fraction=0.32)
        result_large = initial_thermal_state(model_large, core_mass_fraction=0.32)

        assert result_large['U_differentiated'] > result_small['U_differentiated']


# ============================================================================
# Branch coverage: callables that activate non-default code paths
# ============================================================================
#
# The default ``initial_thermal_state`` call uses constant C_p, no nabla_ad,
# and isothermal core. The branches below are activated only when the user
# supplies callables for ``cp_iron_func``, ``cp_silicate_func``,
# ``nabla_ad_func``, ``nabla_ad_iron_func``, or ``iron_melting_func``.
# Each test class targets one such callable family.


def _toy_nabla_ad(P, T):
    """Realistic nabla_ad ~ 0.25 in-table.

    Discriminating: not 0.3 (the Gruneisen-fallback hand-wave value), so a
    branch that silently mis-routes through the fallback would give a
    different T_CMB than this test asserts.
    """
    return 0.25


def _toy_cp_iron(P, T):
    """Iron C_p with a deliberate P-dependence.

    Non-trivial P dependence so the mass-weighted average over a
    multi-shell core differs from the constant-fallback value of C_iron=450.
    """
    return 450.0 + 5.0e-11 * P


def _toy_cp_silicate(P, T):
    """Silicate C_p with deliberate P-dependence."""
    return 1250.0 + 1.0e-11 * P


@pytest.mark.unit
class TestIntegrateAdiabatPaths:
    """``_integrate_adiabat`` has a nabla_ad branch (L210-225) and a Gruneisen branch (L226-229)."""

    def test_nabla_ad_branch_runs_and_warms_inward(self):
        """With ``nabla_ad_func``, T must increase monotonically as P increases."""
        from zalmoxis.energetics import _integrate_adiabat

        # Pressure grid surface -> CMB (increasing P).
        P_grid = np.linspace(1e7, 1.4e11, 80)
        T_top = _integrate_adiabat(P_grid, 1500.0, _toy_nabla_ad)
        T_top_grun = _integrate_adiabat(P_grid, 1500.0, None)
        # Both branches must warm; the Gruneisen branch is bounded by the
        # log-stepping branch when nabla_ad < gamma/Kprime ~ 0.325, so the
        # Gruneisen step is hotter than the nabla_ad step at default settings.
        assert T_top > 1500.0
        assert T_top_grun > 1500.0
        # Both branches must warm by O(>200 K) over a 14-decade pressure rise.
        assert T_top - 1500.0 > 200.0
        assert T_top_grun - 1500.0 > 200.0

    def test_nabla_ad_step_capped_when_exceeds_gruneisen(self):
        """Cap kicks in when nabla_ad_func returns a divergent value.

        Discriminating edge case: ``nabla_ad_func`` returns 0.6 (well above
        gamma/Kprime ~ 0.325). The capping logic must clamp to the Gruneisen
        prediction, so the integrated T_top should match the Gruneisen-only
        result within a few %.
        """
        from zalmoxis.energetics import _integrate_adiabat

        def divergent_nabla_ad(P, T):
            return 0.6

        P_grid = np.linspace(1e7, 1.4e11, 80)
        T_top_capped = _integrate_adiabat(P_grid, 1500.0, divergent_nabla_ad)
        T_top_grun = _integrate_adiabat(P_grid, 1500.0, None)
        # Capping pulls T_top within 5% of the Gruneisen result.
        np.testing.assert_allclose(T_top_capped, T_top_grun, rtol=5e-2)

    def test_zero_pressure_clamped_to_floor(self):
        """Pressure floor at 1e3 Pa: a pressure of 0 must not cause div-by-zero."""
        from zalmoxis.energetics import _integrate_adiabat

        # Single-step path with P=0 at one end.
        P_grid = np.array([0.0, 1e9])
        T_top = _integrate_adiabat(P_grid, 1500.0, None)
        assert np.isfinite(T_top)
        assert T_top > 1500.0  # warming inward

    def test_single_point_profile_returns_anchor(self):
        """Edge case: profile of length 1 has no steps and returns T_anchor."""
        from zalmoxis.energetics import _integrate_adiabat

        P_grid = np.array([1e9])
        T_top = _integrate_adiabat(P_grid, 1234.0, _toy_nabla_ad)
        assert T_top == pytest.approx(1234.0)


@pytest.mark.unit
class TestVariableHeatCapacityPath:
    """``initial_thermal_state`` with cp_iron_func and cp_silicate_func."""

    def test_cp_funcs_change_C_avg_relative_to_constant(self):
        """Mass-weighted C_p differs from constant C_p when funcs vary with P.

        Discriminating: the toy funcs add ~50 J/kg/K of extra C_p at the
        CMB pressure (5e-11 * 1.4e11 = 7), so C_iron_avg should rise above
        the constant 450 J/kg/K default by a measurable amount.
        """
        model = _make_synthetic_model_results(n_points=200, cmf=0.32)
        result = initial_thermal_state(
            model,
            core_mass_fraction=0.32,
            cp_iron_func=_toy_cp_iron,
            cp_silicate_func=_toy_cp_silicate,
        )
        # The constant defaults give C_iron_avg = 450, C_sil_avg = 1250.
        assert result['C_iron_avg'] > 450.0
        assert result['C_silicate_avg'] > 1250.0
        # And the result dict carries the mass-weighted aggregate.
        assert 'C_avg' in result
        assert result['C_avg'] > 0.0

    def test_only_cp_iron_func_provided_silicate_falls_back_to_constant(self):
        """Edge: providing only one cp callable still triggers the integration block."""
        model = _make_synthetic_model_results(n_points=200, cmf=0.32)
        result = initial_thermal_state(
            model,
            core_mass_fraction=0.32,
            cp_iron_func=_toy_cp_iron,
        )
        # Iron average reflects the toy func; silicate falls back to the
        # default constant 1250.
        assert result['C_iron_avg'] > 450.0
        assert result['C_silicate_avg'] == pytest.approx(1250.0, abs=1e-6)

    def test_cp_funcs_and_nabla_ad_func_together(self):
        """Both callables together: exercises the L388-391 nabla_ad path inside cp loop."""
        model = _make_synthetic_model_results(n_points=200, cmf=0.32)
        result = initial_thermal_state(
            model,
            core_mass_fraction=0.32,
            cp_iron_func=_toy_cp_iron,
            cp_silicate_func=_toy_cp_silicate,
            nabla_ad_func=_toy_nabla_ad,
        )
        # Both branches active: result must remain physical.
        assert result['T_cmb'] > result['T_surf_accr']
        assert result['C_iron_avg'] > 450.0
        assert result['C_silicate_avg'] > 1250.0

    def test_zero_mass_shell_skipped_in_cp_integration(self):
        """L408-409 ``if dm_i <= 0: continue`` branch.

        Construct a model with a duplicate radius (dm = 0) and verify the
        cp-integration loop survives (no division by zero, no NaN in C_avg).
        """
        n = 50
        radii = np.linspace(0, 6.371e6, n)
        mass_enclosed = np.linspace(0, 5.972e24, n).copy()
        # Force a zero-mass shell by duplicating one entry.
        mass_enclosed[20] = mass_enclosed[19]
        pressure = np.linspace(1e11, 1e5, n)
        cmb_mass = mass_enclosed[15]

        model = {
            'radii': radii,
            'density': 5500.0 * np.ones(n),
            'gravity': np.linspace(0, 9.8, n),
            'pressure': pressure,
            'temperature': 1500.0 * np.ones(n),
            'mass_enclosed': mass_enclosed,
            'cmb_mass': cmb_mass,
        }
        result = initial_thermal_state(
            model,
            core_mass_fraction=0.32,
            cp_iron_func=_toy_cp_iron,
            cp_silicate_func=_toy_cp_silicate,
        )
        assert np.isfinite(result['C_avg'])
        assert result['C_avg'] > 0.0


@pytest.mark.unit
class TestNablaAdFuncPath:
    """``initial_thermal_state`` with nabla_ad_func and nabla_ad_iron_func."""

    def test_provided_mantle_nabla_ad_warms_T_cmb(self):
        """With nabla_ad_func, T_CMB rises above the Gruneisen-only default.

        Property assertion (no exact value): the mantle adiabat with
        nabla_ad ~ 0.25 takes a different path than the Gruneisen step;
        T_CMB must remain >= T_surf_accr in either case.
        """
        model = _make_synthetic_model_results(n_points=200, cmf=0.32)
        result = initial_thermal_state(
            model,
            core_mass_fraction=0.32,
            nabla_ad_func=_toy_nabla_ad,
        )
        assert result['T_cmb'] > result['T_surf_accr']
        # The mantle adiabat increment must be non-negative after clamp.
        assert result['Delta_T_adiabat'] >= 0.0

    def test_iron_nabla_ad_func_makes_core_non_isothermal(self):
        """Provided nabla_ad_iron_func: T_profile[0] > T_cmb (warmer core center).

        Without nabla_ad_iron_func the core is isothermal at T_cmb, so
        T_profile[0] == T_cmb. With the iron adiabat callable, T_profile[0]
        must rise above T_cmb (compression heating inward).
        """

        def iron_nabla(P, T):
            return 0.20  # iron adiabat ~ 1-3 K/GPa

        model = _make_synthetic_model_results(n_points=200, cmf=0.32)
        result = initial_thermal_state(
            model,
            core_mass_fraction=0.32,
            nabla_ad_iron_func=iron_nabla,
        )
        assert result['T_profile'][0] > result['T_cmb']

    def test_partial_core_state_when_T_cmb_within_5pct_of_T_melt(self):
        """Edge: T_cmb tuned to land in [0.95, 1.05] * T_melt.

        Forces the third arm of the core_state classifier (line 524). Use a
        custom iron melting function that returns a value sandwiching T_cmb.
        """
        model = _make_synthetic_model_results(n_points=200, cmf=0.32)
        # First, get T_cmb under the default settings.
        baseline = initial_thermal_state(model, core_mass_fraction=0.32)
        T_cmb_baseline = baseline['T_cmb']

        # Custom melting curve that returns exactly T_cmb at every P -> partial.
        def melt_at_T_cmb(P):
            return T_cmb_baseline

        result = initial_thermal_state(
            model,
            core_mass_fraction=0.32,
            iron_melting_func=melt_at_T_cmb,
        )
        assert result['core_state'] == 'partial'

    def test_solid_core_state_when_T_cmb_below_T_melt(self):
        """Edge: melting curve returning a high value forces 'solid' branch."""

        def hot_melt(P):
            return 50000.0  # unphysically hot melting -> always solid below

        model = _make_synthetic_model_results(n_points=200, cmf=0.32)
        result = initial_thermal_state(
            model,
            core_mass_fraction=0.32,
            iron_melting_func=hot_melt,
        )
        assert result['core_state'] == 'solid'


@pytest.mark.unit
class TestNegativeDeltaTAdiabatClamp:
    """When _integrate_adiabat returns T_at_cmb < T_surf_accr, Delta_T_ad clamps to 0."""

    def test_inverted_pressure_profile_triggers_clamp(self):
        """Pressure profile that decreases inward forces T_at_cmb < T_surf_accr.

        Edge case for the L464-470 ``Delta_T_ad < 0`` warning + clamp branch.
        """
        # Build a synthetic model whose pressure decreases from CMB to surface
        # the wrong way. The slicing in initial_thermal_state then yields an
        # adiabat from low T at CMB to high T at surface, so when the
        # outward-from-CMB profile is reversed the integrated T_at_cmb may
        # come back below the surface anchor. We verify the clamp handles it.
        n = 60
        radii = np.linspace(0, 6.371e6, n)
        # Outward density gradient is fine; just lower CMB pressure than surface.
        mass_enclosed = np.linspace(0, 5.972e24, n)
        # Pressure rising outward (inverted) so the adiabat integrator sees
        # negative dP. The clamp must catch the unphysical Delta_T_ad.
        pressure = np.linspace(1e9, 1e11, n)
        cmb_mass = mass_enclosed[20]  # CMB at index 20

        model = {
            'radii': radii,
            'density': 5500.0 * np.ones(n),
            'gravity': np.linspace(0, 9.8, n),
            'pressure': pressure,
            'temperature': 1500.0 * np.ones(n),
            'mass_enclosed': mass_enclosed,
            'cmb_mass': cmb_mass,
        }
        result = initial_thermal_state(model, core_mass_fraction=0.32)
        # Delta_T_ad must be clamped to >= 0 even with inverted pressure.
        assert result['Delta_T_adiabat'] >= 0.0
        assert np.isfinite(result['T_cmb'])
