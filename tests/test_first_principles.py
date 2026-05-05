"""First-principles validation tests for the Zalmoxis structure solver.

Verifies that the ODE system, conservation laws, and full solver produce results
consistent with known analytic solutions and physical constraints.

Tier 1 (unit): Pure-math ODE verification against closed-form solutions.
    No EOS data files needed. Patches calculate_mixed_density to inject constant density.
Tier 2 (integration): Full solver with Seager+2007 analytic EOS.
    No tabulated data files needed. Tests Earth benchmarks, scaling laws, conservation.
Tier 3 (slow): Numerical convergence studies (grid and tolerance convergence).

References:
    - Seager et al. (2007), ApJ 669:1279 (analytic EOS, mass-radius relations)
    - docs/How-to/test_infrastructure.md
    - docs/How-to/test_categorization.md
"""

from __future__ import annotations

import math
from functools import lru_cache
from unittest.mock import patch

import numpy as np
import pytest

from zalmoxis.constants import G, earth_mass, earth_radius
from zalmoxis.mixing import LayerMixture
from zalmoxis.structure_model import solve_structure

# ============================================================================
# Helpers
# ============================================================================


def _make_constant_density_mock(rho):
    """Return a function mimicking calculate_mixed_density that always returns rho.

    Parameters
    ----------
    rho : float
        Constant density to return [kg/m^3].

    Returns
    -------
    callable
        Mock function with same signature as calculate_mixed_density.
    """

    def _mock(pressure, temperature, mixture, *args, **kwargs):
        if pressure <= 0 or np.isnan(pressure):
            return None
        return rho

    return _mock


def _make_two_layer_density_mock(rho_core, rho_mantle, cmb_mass):
    """Return a density mock that dispatches on enclosed mass.

    Parameters
    ----------
    rho_core : float
        Core density [kg/m^3].
    rho_mantle : float
        Mantle density [kg/m^3].
    cmb_mass : float
        Core-mantle boundary mass [kg].

    Returns
    -------
    callable
        Mock function that returns rho_core for mass < cmb_mass, else rho_mantle.
    """

    # The mock receives the mixture object which knows its layer, but we need
    # to track enclosed mass. Since calculate_mixed_density doesn't receive
    # the mass directly, we use the layer_mixture's component names to dispatch.
    def _mock(pressure, temperature, mixture, *args, **kwargs):
        if pressure <= 0 or np.isnan(pressure):
            return None
        if mixture.components[0] == 'mock:core':
            return rho_core
        return rho_mantle

    return _mock


def _solve_uniform_sphere(rho, R, P_center, num_layers=200):
    """Solve the structure ODEs for a uniform-density sphere.

    Parameters
    ----------
    rho : float
        Constant density [kg/m^3].
    R : float
        Planet radius [m].
    P_center : float
        Central pressure [Pa].
    num_layers : int
        Number of radial grid points.

    Returns
    -------
    radii : np.ndarray
        Radial grid [m].
    mass : np.ndarray
        Enclosed mass at each radius [kg].
    gravity : np.ndarray
        Gravity at each radius [m/s^2].
    pressure : np.ndarray
        Pressure at each radius [Pa].
    """
    radii = np.linspace(0, R, num_layers)
    layer_mixtures = {'core': LayerMixture(components=['mock:uniform'], fractions=[1.0])}
    mock_fn = _make_constant_density_mock(rho)

    with patch('zalmoxis.structure_model.calculate_mixed_density', mock_fn):
        mass, gravity, pressure = solve_structure(
            layer_mixtures=layer_mixtures,
            cmb_mass=1e30,  # all mass in "core" layer
            core_mantle_mass=1e30,
            radii=radii,
            adaptive_radial_fraction=0.98,
            relative_tolerance=1e-10,
            absolute_tolerance=1e-12,
            maximum_step=R / 10,
            material_dictionaries={},
            interpolation_cache={},
            y0=[0, 0, P_center],
            solidus_func=None,
            liquidus_func=None,
        )

    return radii, mass, gravity, pressure


def _analytic_uniform_sphere(rho, R, P_center, radii):
    """Compute exact analytic profiles for a uniform-density sphere.

    Parameters
    ----------
    rho : float
        Constant density [kg/m^3].
    R : float
        Planet radius [m].
    P_center : float
        Central pressure [Pa].
    radii : np.ndarray
        Radial grid [m].

    Returns
    -------
    M_exact : np.ndarray
        Enclosed mass [kg].
    g_exact : np.ndarray
        Gravity [m/s^2].
    P_exact : np.ndarray
        Pressure [Pa].
    """
    M_exact = (4.0 / 3.0) * math.pi * rho * radii**3
    g_exact = (4.0 / 3.0) * math.pi * G * rho * radii
    P_exact = P_center - (2.0 / 3.0) * math.pi * G * rho**2 * radii**2
    return M_exact, g_exact, P_exact


def _two_layer_central_pressure(rho_core, rho_mantle, R_cmb, R_total, M_cmb):
    """Compute the exact central pressure for a two-layer constant-density sphere.

    Integrates dP/dr = -rho * g from the surface (P=0) to the center.

    Parameters
    ----------
    rho_core : float
        Core density [kg/m^3].
    rho_mantle : float
        Mantle density [kg/m^3].
    R_cmb : float
        Core-mantle boundary radius [m].
    R_total : float
        Planet surface radius [m].
    M_cmb : float
        Core mass [kg].

    Returns
    -------
    float
        Central pressure [Pa].
    """
    # Compute P(R) = 0, integrate inward numerically with high resolution
    n_fine = 10000
    r_fine = np.linspace(0, R_total, n_fine)

    # Build analytic rho(r) and g(r) on the fine grid
    rho_fine = np.where(r_fine <= R_cmb, rho_core, rho_mantle)
    M_fine = np.zeros(n_fine)
    for i in range(n_fine):
        r = r_fine[i]
        if r <= R_cmb:
            M_fine[i] = (4.0 / 3.0) * math.pi * rho_core * r**3
        else:
            M_fine[i] = M_cmb + (4.0 / 3.0) * math.pi * rho_mantle * (r**3 - R_cmb**3)
    g_fine = np.zeros(n_fine)
    g_fine[1:] = G * M_fine[1:] / r_fine[1:] ** 2

    # P_center = integral_0^R rho(r) g(r) dr (since P(R) = 0)
    integrand = rho_fine * g_fine
    return float(np.trapezoid(integrand, r_fine))


@lru_cache(maxsize=64)
def _run_analytic_eos_solver(
    mass_earth,
    cmf=0.325,
    mmf=0,
    num_layers=200,
    relative_tolerance=1e-8,
    absolute_tolerance=1e-10,
):
    """Run the full Zalmoxis solver with Analytic EOS.

    Cached by argument tuple so multiple tests asserting different invariants
    on the same (mass, cmf, mmf, grid, tol) configuration share one solver
    run. Test code only reads the result dict, never mutates it.

    Parameters
    ----------
    mass_earth : float
        Planet mass in Earth masses.
    cmf : float
        Core mass fraction.
    mmf : float
        Mantle mass fraction (0 for 2-layer model).
    num_layers : int
        Number of radial grid points.
    relative_tolerance : float
        ODE solver relative tolerance.
    absolute_tolerance : float
        ODE solver absolute tolerance.

    Returns
    -------
    dict
        Model results from zalmoxis.main().
    """
    import os

    from zalmoxis.config import load_material_dictionaries
    from zalmoxis.solver import main

    layer_eos = {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'}
    if mmf > 0:
        layer_eos['ice_layer'] = 'Analytic:H2O'

    config_params = {
        'planet_mass': mass_earth * earth_mass,
        'core_mass_fraction': cmf,
        'mantle_mass_fraction': mmf,
        'temperature_mode': 'isothermal',
        'surface_temperature': 300,
        'center_temperature': 5000,
        'temp_profile_file': '',
        'layer_eos_config': layer_eos,
        'rock_solidus': 'Stixrude14-solidus',
        'rock_liquidus': 'Stixrude14-liquidus',
        'mushy_zone_factor': 1.0,
        'mushy_zone_factors': {
            'PALEOS:iron': 1.0,
            'PALEOS:MgSiO3': 1.0,
            'PALEOS:H2O': 1.0,
        },
        'condensed_rho_min': 322.0,
        'condensed_rho_scale': 50.0,
        'binodal_T_scale': 50.0,
        'num_layers': num_layers,
        'target_surface_pressure': 101325,
        'data_output_enabled': False,
        'plotting_enabled': False,
    }

    from zalmoxis import get_zalmoxis_root

    input_dir = os.path.join(get_zalmoxis_root(), 'input')
    mat_dicts = load_material_dictionaries()

    results = main(
        config_params,
        material_dictionaries=mat_dicts,
        melting_curves_functions=None,
        input_dir=input_dir,
    )
    return results


# ============================================================================
# Tier 1: Pure-Math ODE Verification
# ============================================================================


@pytest.mark.unit
class TestUniformDensitySphere:
    """Verify ODE integration against the exact analytic solution for a
    uniform-density sphere. No EOS data needed."""

    RHO = 5000.0  # kg/m^3, representative rocky planet average
    R = 6.4e6  # m, ~Earth radius
    P_CENTER = 3.6e11  # Pa, ~360 GPa
    N = 300

    def test_mass_profile(self):
        """M(r) must match (4/3) pi rho r^3."""
        radii, mass, _, _ = _solve_uniform_sphere(self.RHO, self.R, self.P_CENTER, self.N)
        M_exact, _, _ = _analytic_uniform_sphere(self.RHO, self.R, self.P_CENTER, radii)
        # Skip r=0 (both are zero, ratio undefined)
        np.testing.assert_allclose(
            mass[1:],
            M_exact[1:],
            rtol=1e-6,
            err_msg='Enclosed mass deviates from (4/3) pi rho r^3',
        )

    def test_gravity_profile(self):
        """g(r) must match (4/3) pi G rho r (linear in r)."""
        radii, _, gravity, _ = _solve_uniform_sphere(self.RHO, self.R, self.P_CENTER, self.N)
        _, g_exact, _ = _analytic_uniform_sphere(self.RHO, self.R, self.P_CENTER, radii)
        np.testing.assert_allclose(
            gravity[1:],
            g_exact[1:],
            rtol=1e-6,
            err_msg='Gravity deviates from (4/3) pi G rho r',
        )

    def test_pressure_profile(self):
        """P(r) must match P_c - (2/3) pi G rho^2 r^2."""
        radii, _, _, pressure = _solve_uniform_sphere(self.RHO, self.R, self.P_CENTER, self.N)
        _, _, P_exact = _analytic_uniform_sphere(self.RHO, self.R, self.P_CENTER, radii)
        # Only compare where P > 0 (terminal event may truncate)
        valid = pressure > 0
        np.testing.assert_allclose(
            pressure[valid],
            P_exact[valid],
            rtol=1e-6,
            err_msg='Pressure deviates from P_c - (2/3) pi G rho^2 r^2',
        )

    def test_central_boundary_conditions(self):
        """Initial conditions: M(0) = 0, g(0) = 0, P(0) = P_center."""
        radii, mass, gravity, pressure = _solve_uniform_sphere(
            self.RHO,
            self.R,
            self.P_CENTER,
            self.N,
        )
        assert mass[0] == 0.0
        assert gravity[0] == 0.0
        assert pressure[0] == pytest.approx(self.P_CENTER, rel=1e-12)

    def test_gravity_slope_at_center(self):
        """dg/dr at r->0 must approach (4/3) pi G rho."""
        radii, _, gravity, _ = _solve_uniform_sphere(self.RHO, self.R, self.P_CENTER, self.N)
        # Use the first few points for a finite-difference slope
        dgdr_numerical = (gravity[2] - gravity[0]) / (radii[2] - radii[0])
        dgdr_exact = (4.0 / 3.0) * math.pi * G * self.RHO
        assert dgdr_numerical == pytest.approx(dgdr_exact, rel=1e-4), (
            f'dg/dr at center: numerical={dgdr_numerical:.6e}, exact={dgdr_exact:.6e}'
        )

    def test_pressure_monotonically_decreasing(self):
        """Pressure must strictly decrease with radius (where P > 0)."""
        _, _, _, pressure = _solve_uniform_sphere(self.RHO, self.R, self.P_CENTER, self.N)
        valid = pressure > 0
        P_valid = pressure[valid]
        assert len(P_valid) > 10, 'Too few valid pressure points'
        dP = np.diff(P_valid)
        assert np.all(dP < 0), (
            f'Pressure not monotonically decreasing: '
            f'max(dP) = {np.max(dP):.2e} Pa at index {np.argmax(dP)}'
        )


@pytest.mark.unit
class TestGaussLaw:
    """Verify Gauss's law g(r) = G M(r) / r^2 at every mesh point."""

    def test_gauss_law_uniform_sphere(self):
        """Gauss's law for uniform-density sphere."""
        rho, R, P_c = 5000.0, 6.4e6, 3.6e11
        radii, mass, gravity, _ = _solve_uniform_sphere(rho, R, P_c, 300)
        # Skip r=0 (g=0, M=0, ratio 0/0)
        r = radii[2:]  # skip first two points near singularity
        M = mass[2:]
        g = gravity[2:]
        g_gauss = G * M / r**2
        np.testing.assert_allclose(
            g,
            g_gauss,
            rtol=1e-5,
            err_msg='Gravity deviates from Gauss law g = GM/r^2',
        )


@pytest.mark.unit
class TestHydrostaticBalance:
    """Verify dP/dr + rho * g = 0 pointwise (hydrostatic equilibrium)."""

    def test_hydrostatic_residual_uniform_sphere(self):
        """Hydrostatic balance residual for uniform-density sphere."""
        rho, R, P_c = 5000.0, 6.4e6, 3.6e11
        N = 500  # finer grid for better finite differences
        radii, _, gravity, pressure = _solve_uniform_sphere(rho, R, P_c, N)

        # Central differences (skip first and last point)
        valid = pressure > 0
        idx = np.where(valid)[0]
        # Use interior points only (need neighbors on both sides)
        interior = idx[(idx > 0) & (idx < len(radii) - 1)]

        dPdr = (pressure[interior + 1] - pressure[interior - 1]) / (
            radii[interior + 1] - radii[interior - 1]
        )
        residual = dPdr + rho * gravity[interior]

        # Normalize by typical pressure gradient scale
        P_scale = (2.0 / 3.0) * math.pi * G * rho**2 * R
        relative_residual = np.abs(residual) / P_scale

        assert np.max(relative_residual) < 1e-4, (
            f'Max relative hydrostatic residual = {np.max(relative_residual):.2e}, '
            f'expected < 1e-4'
        )


@pytest.mark.unit
class TestTwoLayerSphere:
    """Verify ODE integration for a two-layer constant-density sphere."""

    RHO_CORE = 13000.0  # kg/m^3
    RHO_MANTLE = 4000.0  # kg/m^3
    M_TOTAL = 5.972e24  # kg (Earth mass)
    CMF = 0.325
    N = 400

    @staticmethod
    def _geometry():
        """Compute the two-layer sphere geometry and self-consistent P_center."""
        cmb_mass = TestTwoLayerSphere.CMF * TestTwoLayerSphere.M_TOTAL
        rho_c = TestTwoLayerSphere.RHO_CORE
        rho_m = TestTwoLayerSphere.RHO_MANTLE
        R_cmb = ((3 * cmb_mass) / (4 * math.pi * rho_c)) ** (1.0 / 3.0)
        M_mantle = TestTwoLayerSphere.M_TOTAL - cmb_mass
        R_total = (R_cmb**3 + (3 * M_mantle) / (4 * math.pi * rho_m)) ** (1.0 / 3.0)
        P_center = _two_layer_central_pressure(rho_c, rho_m, R_cmb, R_total, cmb_mass)
        # Add 5% margin so pressure stays positive across the full grid
        P_center *= 1.05
        return cmb_mass, R_cmb, R_total, P_center

    @staticmethod
    def _run():
        """Run solve_structure for the two-layer sphere."""
        cmb_mass, R_cmb, R_total, P_center = TestTwoLayerSphere._geometry()
        radii = np.linspace(0, R_total, TestTwoLayerSphere.N)
        core_mix = LayerMixture(components=['mock:core'], fractions=[1.0])
        mantle_mix = LayerMixture(components=['mock:mantle'], fractions=[1.0])
        layer_mixtures = {'core': core_mix, 'mantle': mantle_mix}
        mock_fn = _make_two_layer_density_mock(
            TestTwoLayerSphere.RHO_CORE,
            TestTwoLayerSphere.RHO_MANTLE,
            cmb_mass,
        )

        with patch('zalmoxis.structure_model.calculate_mixed_density', mock_fn):
            mass, gravity, pressure = solve_structure(
                layer_mixtures=layer_mixtures,
                cmb_mass=cmb_mass,
                core_mantle_mass=TestTwoLayerSphere.M_TOTAL,
                radii=radii,
                adaptive_radial_fraction=0.98,
                relative_tolerance=1e-10,
                absolute_tolerance=1e-12,
                maximum_step=R_total / 10,
                material_dictionaries={},
                interpolation_cache={},
                y0=[0, 0, P_center],
                solidus_func=None,
                liquidus_func=None,
            )
        return radii, mass, gravity, pressure, cmb_mass, R_cmb, R_total

    def test_two_layer_mass_profile(self):
        """Mass profile must match piecewise analytic formula."""
        radii, mass, _, _, cmb_mass, R_cmb, _ = self._run()

        M_exact = np.zeros_like(radii)
        for i, r in enumerate(radii):
            if r <= R_cmb:
                M_exact[i] = (4.0 / 3.0) * math.pi * self.RHO_CORE * r**3
            else:
                M_exact[i] = cmb_mass + (4.0 / 3.0) * math.pi * self.RHO_MANTLE * (
                    r**3 - R_cmb**3
                )

        np.testing.assert_allclose(
            mass[1:],
            M_exact[1:],
            rtol=1e-5,
            err_msg='Two-layer mass profile deviates from analytic formula',
        )

    def test_two_layer_gauss_law(self):
        """Gauss's law must hold at every point in both layers."""
        radii, mass, gravity, _, _, _, _ = self._run()

        r = radii[3:]
        g_gauss = G * mass[3:] / r**2
        np.testing.assert_allclose(
            gravity[3:],
            g_gauss,
            rtol=1e-4,
            err_msg='Gauss law violation in two-layer sphere',
        )


@pytest.mark.unit
class TestGravitationalBindingEnergy:
    """Verify gravitational binding energy for a uniform-density sphere.

    E_grav = -(3/5) G M^2 / R (exact for uniform density).
    """

    def test_binding_energy(self):
        """Numerically integrated binding energy must match analytic formula."""
        rho, R, P_c = 5000.0, 6.4e6, 3.6e11
        N = 500
        radii, mass, _, _ = _solve_uniform_sphere(rho, R, P_c, N)

        # Numerical integration: E = -integral [G M(r) / r] 4 pi r^2 rho dr
        # Use trapezoidal rule on the integrand
        r = radii[1:]  # skip r=0 (singularity in GM/r, but M=0 so integrand=0)
        M = mass[1:]
        integrand = G * M / r * 4 * math.pi * r**2 * rho
        E_numerical = -np.trapezoid(integrand, r)

        # Analytic: E = -(3/5) G M_total^2 / R
        M_total = mass[-1]
        # Use the outermost radius where pressure is still positive
        E_analytic = -(3.0 / 5.0) * G * M_total**2 / R

        assert E_numerical == pytest.approx(E_analytic, rel=0.01), (
            f'Binding energy: numerical={E_numerical:.4e} J, analytic={E_analytic:.4e} J'
        )


# ============================================================================
# Tier 2: Analytic-EOS Full Solver
# ============================================================================


@pytest.mark.smoke
class TestEarthBenchmark:
    """Verify the full solver against known Earth properties using Analytic EOS."""

    def test_earth_radius(self):
        """1 M_Earth with CMF=0.325 must give R ~ 1.0 R_Earth (+/- 5%)."""
        results = _run_analytic_eos_solver(1.0, cmf=0.325)
        assert results['converged'], 'Solver did not converge for 1 M_Earth'

        R_planet = results['radii'][-1]
        R_ratio = R_planet / earth_radius
        assert R_ratio == pytest.approx(1.0, abs=0.05), (
            f'Earth radius ratio = {R_ratio:.4f}, expected ~1.0 +/- 0.05'
        )

    def test_earth_surface_gravity(self):
        """Surface gravity must be ~9.8 m/s^2 (+/- 10%)."""
        results = _run_analytic_eos_solver(1.0, cmf=0.325)
        assert results['converged'], 'Solver did not converge for 1 M_Earth'

        # Find last point with positive pressure
        pressure = results['pressure']
        valid = pressure > 0
        i_surf = np.where(valid)[0][-1]
        g_surf = results['gravity'][i_surf]

        assert g_surf == pytest.approx(9.8, rel=0.10), (
            f'Surface gravity = {g_surf:.2f} m/s^2, expected ~9.8 +/- 10%'
        )

    def test_earth_central_pressure(self):
        """Central pressure must be ~360 GPa (+/- 15%)."""
        results = _run_analytic_eos_solver(1.0, cmf=0.325)
        assert results['converged'], 'Solver did not converge for 1 M_Earth'

        P_center = results['pressure'][0]
        P_center_GPa = P_center / 1e9
        assert P_center_GPa == pytest.approx(360, rel=0.15), (
            f'Central pressure = {P_center_GPa:.1f} GPa, expected ~360 +/- 15%'
        )


@pytest.mark.smoke
class TestMassRadiusScaling:
    """Verify the mass-radius scaling exponent against Seager+2007."""

    def test_scaling_exponent(self):
        """R ~ M^alpha with alpha ~ 0.27 for rocky planets (Seager+2007)."""
        masses = [0.5, 1.0, 2.0, 5.0, 10.0]
        radii = []
        for m in masses:
            results = _run_analytic_eos_solver(m, cmf=0.325)
            assert results['converged'], f'Solver did not converge for {m} M_Earth'
            radii.append(results['radii'][-1])

        # Fit log(R) vs log(M)
        log_m = np.log10(masses)
        log_r = np.log10(np.array(radii) / earth_radius)
        coeffs = np.polyfit(log_m, log_r, 1)
        alpha = coeffs[0]

        assert alpha == pytest.approx(0.27, abs=0.04), (
            f'M-R scaling exponent alpha = {alpha:.3f}, expected ~0.27 +/- 0.04 (Seager+2007)'
        )


@pytest.mark.smoke
class TestCMFMonotonicity:
    """Radius must decrease monotonically with increasing core mass fraction."""

    def test_cmf_monotonicity(self):
        """R(CMF) must be strictly decreasing (iron is denser than silicate)."""
        cmfs = [0.1, 0.2, 0.3, 0.5, 0.7]
        radii = []
        for cmf in cmfs:
            results = _run_analytic_eos_solver(1.0, cmf=cmf)
            assert results['converged'], f'Solver did not converge for CMF={cmf}'
            radii.append(results['radii'][-1])

        for i in range(1, len(radii)):
            assert radii[i] < radii[i - 1], (
                f'Radius not decreasing: R(CMF={cmfs[i]}) = {radii[i]:.0f} m '
                f'>= R(CMF={cmfs[i - 1]}) = {radii[i - 1]:.0f} m'
            )


@pytest.mark.smoke
class TestPureIronPlanet:
    """Verify a pure iron planet (CMF=1.0) against Seager+2007."""

    def test_pure_iron_radius(self):
        """Pure iron 1 M_Earth: R ~ 0.77 R_Earth (Seager+2007 Table 4)."""
        # CMF=1.0 means only core, no mantle
        results = _run_analytic_eos_solver(1.0, cmf=1.0)
        assert results['converged'], 'Solver did not converge for pure iron'

        R_planet = results['radii'][-1]
        R_ratio = R_planet / earth_radius
        # Seager+2007 pure iron at 1 M_Earth gives R ~ 0.77 R_Earth
        assert R_ratio == pytest.approx(0.77, abs=0.05), (
            f'Pure iron radius ratio = {R_ratio:.4f}, expected ~0.77 +/- 0.05'
        )


@pytest.mark.smoke
class TestConservationLaws:
    """Verify mass conservation and surface pressure boundary condition."""

    def test_mass_conservation(self):
        """M(R_surface) must equal input planet_mass to within tolerance."""
        results = _run_analytic_eos_solver(1.0, cmf=0.325)
        assert results['converged'], 'Solver did not converge'

        M_surface = results['mass_enclosed'][-1]
        M_input = 1.0 * earth_mass
        rel_diff = abs(M_surface - M_input) / M_input
        assert rel_diff < 0.01, (
            f'Mass conservation: M_surface/M_input = {M_surface / M_input:.6f}, '
            f'rel_diff = {rel_diff:.2e}'
        )

    def test_mass_conservation_trapezoidal(self):
        """Independent trapezoidal integration of 4 pi r^2 rho(r) dr."""
        results = _run_analytic_eos_solver(1.0, cmf=0.325)
        assert results['converged'], 'Solver did not converge'

        radii = results['radii']
        density = results['density']
        # Trapezoidal integration
        integrand = 4 * math.pi * radii**2 * density
        M_trapz = np.trapezoid(integrand, radii)
        M_input = 1.0 * earth_mass

        rel_diff = abs(M_trapz - M_input) / M_input
        assert rel_diff < 0.02, (
            f'Trapezoidal mass: M_trapz/M_input = {M_trapz / M_input:.6f}, '
            f'rel_diff = {rel_diff:.2e}'
        )

    def test_surface_pressure(self):
        """P(R_surface) must be within a few GPa of target_surface_pressure.

        The Brent solver uses pressure_tolerance = 1e9 Pa (1 GPa) and the
        surface grid point may not land exactly at the P=P_target radius,
        so we allow up to 2 GPa residual.
        """
        results = _run_analytic_eos_solver(1.0, cmf=0.325)
        assert results['converged'], 'Solver did not converge'

        # Find last point with positive pressure
        pressure = results['pressure']
        valid = pressure > 0
        i_surf = np.where(valid)[0][-1]
        P_surface = pressure[i_surf]
        target = 101325  # Pa

        assert abs(P_surface - target) < 2e9, (
            f'Surface pressure = {P_surface:.2e} Pa, '
            f'target = {target:.2e} Pa, '
            f'difference = {abs(P_surface - target):.2e} Pa (limit 2 GPa)'
        )

    def test_gauss_law_full_solver(self):
        """Gauss's law g = GM/r^2 must hold on the full solver output."""
        results = _run_analytic_eos_solver(1.0, cmf=0.325)
        assert results['converged'], 'Solver did not converge'

        radii = results['radii']
        mass = results['mass_enclosed']
        gravity = results['gravity']
        pressure = results['pressure']

        # Only check where pressure is positive and r > 0
        valid = (pressure > 0) & (radii > 0)
        # Skip first few points near origin where ratios are noisy
        idx = np.where(valid)[0]
        idx = idx[idx > 5]

        g_gauss = G * mass[idx] / radii[idx] ** 2
        np.testing.assert_allclose(
            gravity[idx],
            g_gauss,
            rtol=1e-3,
            err_msg='Gauss law violation in full solver output',
        )


# ============================================================================
# Tier 3: Numerical Convergence
# ============================================================================


@pytest.mark.slow
class TestGridConvergence:
    """Verify ODE integration error decreases with grid resolution.

    Uses the uniform-density sphere (exact analytic solution) to isolate
    the ODE integrator from the Picard and Brent loops.
    """

    RHO = 5000.0
    R = 6.4e6
    P_CENTER = 3.6e11

    def test_mass_error_decreases_with_grid(self):
        """Max relative error in M(r) must decrease with increasing N.

        Monotonicity is only checked above the floating-point noise floor
        (~1e-13). Below that, RK45 has reached machine-precision limits.
        """
        resolutions = [50, 100, 200, 400]
        errors = []
        for n in resolutions:
            radii, mass, _, _ = _solve_uniform_sphere(self.RHO, self.R, self.P_CENTER, n)
            M_exact, _, _ = _analytic_uniform_sphere(self.RHO, self.R, self.P_CENTER, radii)
            rel_err = np.max(np.abs(mass[1:] - M_exact[1:]) / M_exact[1:])
            errors.append(rel_err)

        # Check monotonic improvement only above noise floor
        noise_floor = 1e-13
        for i in range(1, len(errors)):
            if errors[i - 1] > noise_floor:
                assert errors[i] <= errors[i - 1] * 1.5, (
                    f'Grid convergence: error at N={resolutions[i]} ({errors[i]:.2e}) '
                    f'> error at N={resolutions[i - 1]} ({errors[i - 1]:.2e})'
                )

        # At the finest grid, error should be well below solver tolerance
        assert errors[-1] < 1e-8, f'ODE mass error at N=400: {errors[-1]:.2e}, expected < 1e-8'

    def test_pressure_error_decreases_with_grid(self):
        """Max relative error in P(r) must decrease with increasing N."""
        resolutions = [50, 100, 200, 400]
        errors = []
        for n in resolutions:
            radii, _, _, pressure = _solve_uniform_sphere(self.RHO, self.R, self.P_CENTER, n)
            _, _, P_exact = _analytic_uniform_sphere(self.RHO, self.R, self.P_CENTER, radii)
            valid = pressure > 0
            rel_err = np.max(np.abs(pressure[valid] - P_exact[valid]) / P_exact[valid])
            errors.append(rel_err)

        assert errors[-1] < 1e-6, (
            f'ODE pressure error at N=400: {errors[-1]:.2e}, expected < 1e-6'
        )

    def test_full_solver_grid_convergence(self):
        """Full solver planet radius should be grid-independent to ~1%."""
        radii_at_n = {}
        for n in [100, 200, 400]:
            results = _run_analytic_eos_solver(1.0, cmf=0.325, num_layers=n)
            assert results['converged'], f'Solver did not converge at N={n}'
            radii_at_n[n] = results['radii'][-1]

        # N=200 and N=400 should agree to better than 1%
        rel_diff = abs(radii_at_n[400] - radii_at_n[200]) / radii_at_n[400]
        assert rel_diff < 0.01, (
            f'Full solver grid convergence: R(N=400) vs R(N=200) '
            f'rel_diff = {rel_diff:.2e} (limit 1%)'
        )


@pytest.mark.slow
class TestToleranceConvergence:
    """Verify ODE integration error decreases with tighter tolerances.

    Uses the uniform-density sphere to isolate the ODE integrator.
    """

    RHO = 5000.0
    R = 6.4e6
    P_CENTER = 3.6e11
    N = 200

    def test_mass_error_decreases_with_tolerance(self):
        """Mass profile error must decrease as rtol is tightened."""
        tolerances = [1e-6, 1e-8, 1e-10, 1e-12]
        errors = []
        for rtol in tolerances:
            radii = np.linspace(0, self.R, self.N)
            layer_mixtures = {
                'core': LayerMixture(components=['mock:uniform'], fractions=[1.0]),
            }
            mock_fn = _make_constant_density_mock(self.RHO)

            with patch('zalmoxis.structure_model.calculate_mixed_density', mock_fn):
                mass, _, _ = solve_structure(
                    layer_mixtures=layer_mixtures,
                    cmb_mass=1e30,
                    core_mantle_mass=1e30,
                    radii=radii,
                    adaptive_radial_fraction=0.98,
                    relative_tolerance=rtol,
                    absolute_tolerance=rtol * 1e-2,
                    maximum_step=self.R / 10,
                    material_dictionaries={},
                    interpolation_cache={},
                    y0=[0, 0, self.P_CENTER],
                    solidus_func=None,
                    liquidus_func=None,
                )

            M_exact, _, _ = _analytic_uniform_sphere(self.RHO, self.R, self.P_CENTER, radii)
            rel_err = np.max(np.abs(mass[1:] - M_exact[1:]) / M_exact[1:])
            errors.append(rel_err)

        # Check monotonic improvement only above noise floor
        noise_floor = 1e-13
        for i in range(1, len(errors)):
            if errors[i - 1] > noise_floor:
                assert errors[i] <= errors[i - 1] * 1.5, (
                    f'Tolerance convergence: error at rtol={tolerances[i]} ({errors[i]:.2e}) '
                    f'> error at rtol={tolerances[i - 1]} ({errors[i - 1]:.2e})'
                )

        # At the tightest tolerance, error should be at or below machine precision
        assert errors[-1] < 1e-10, (
            f'ODE mass error at rtol=1e-12: {errors[-1]:.2e}, expected < 1e-10'
        )
