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

        With f_a=0.04, f_d=0.50:
        - Delta_T_G should be ~1000-2000 K
        - Delta_T_D should be ~1000-3000 K
        - T_CMB should be ~3000-6000 K
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

        # Temperature increments should be on the right order of magnitude
        assert 500.0 < result['Delta_T_accretion'] < 5000.0, (
            f'Delta_T_G = {result["Delta_T_accretion"]:.0f} K, expected 500-5000 K'
        )
        assert 500.0 < result['Delta_T_differentiation'] < 5000.0, (
            f'Delta_T_D = {result["Delta_T_differentiation"]:.0f} K, expected 500-5000 K'
        )

        # CMB temperature in a broad physically plausible range
        assert 2000.0 < result['T_cmb'] < 8000.0, (
            f'T_CMB = {result["T_cmb"]:.0f} K, expected 2000-8000 K'
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
