"""
End-to-end verification of solve_structure against the exact n=1 polytrope.

The n=1 polytrope P = K rho^2 has a closed-form Lane-Emden solution
rho(r) = rho_c sin(xi)/xi with xi = pi r / R, the surface at the first zero
xi = pi, and radius R = pi sqrt(K / 2 pi G) fixed by K alone, independent of
central density. K is chosen so R equals one Earth radius, providing an exact
reference for the coupled interior-structure integrator.

This suite drives the real solve_structure ODE integration with the
'Analytic:polytrope_n1' verification material and pins:
  - the emergent surface radius against the exact R = R_earth,
  - the integrated mass against the analytic M = 4 rho_c R^3 / pi,
  - the n=1 mass-independence of the radius, the property that discriminates
    the genuine Chandrasekhar solution from a mass-dependent numerical artifact.

Module under test: zalmoxis.structure_model.solve_structure
References:
    - Chandrasekhar (1939), An Introduction to the Study of Stellar Structure,
      Chapter IV (analytic n=1 Lane-Emden solution).
    - Seager et al. (2007), ApJ 669:1279 (modified-polytrope EOS form).
    - docs/test_categorization.md, docs/test_infrastructure.md
"""

from __future__ import annotations

import numpy as np
import pytest

from zalmoxis.constants import earth_mass, earth_radius
from zalmoxis.eos_analytic import _K_POLYTROPE_N1 as K
from zalmoxis.mixing import LayerMixture
from zalmoxis.structure_model import solve_structure

pytestmark = pytest.mark.smoke

# Analytic n=1 surface radius: R = pi sqrt(K / 2 pi G) = earth_radius by construction.
_R_LE = earth_radius


def _rho_c_for_mass(mass_kg: float) -> float:
    """Central density of an n=1 polytrope of the given total mass (M = 4 rho_c R^3 / pi)."""
    return np.pi * mass_kg / (4.0 * _R_LE**3)


def _integrate_polytrope(mass_kg: float, num_layers: int = 600) -> tuple[float, float]:
    """Integrate the structure ODEs with the polytrope EOS; return (R_surf, M_total).

    A single ``core`` layer carries the whole planet. The surface radius is the linear
    extrapolation of the density zero-crossing: density vanishes linearly at the n=1
    surface, so this recovers R to O(dr^2), whereas extrapolating the quadratically
    vanishing pressure would only give O(dr). The integrated mass is the outermost mass.

    Parameters
    ----------
    mass_kg
        Target total mass; sets the central density and hence the central pressure.
    num_layers
        Radial grid resolution, kept coarse to stay within the smoke-test budget.

    Returns
    -------
    tuple of float
        Emergent surface radius [m] and integrated total mass [kg].
    """
    rho_c = _rho_c_for_mass(mass_kg)
    central_pressure = K * rho_c**2
    radii = np.linspace(0.0, 1.03 * _R_LE, num_layers)
    layer_mixtures = {
        'core': LayerMixture(['Analytic:polytrope_n1'], [1.0]),
    }
    mass, _gravity, pressure = solve_structure(
        layer_mixtures=layer_mixtures,
        cmb_mass=1e30,  # keep all mass in the single core layer
        core_mantle_mass=1e30,
        radii=radii,
        adaptive_radial_fraction=0.98,
        relative_tolerance=1e-8,
        absolute_tolerance=1e-10,
        maximum_step=_R_LE / 50.0,
        material_dictionaries={},  # analytic EOS needs no tabulated registry
        interpolation_cache={},
        y0=[0.0, 0.0, central_pressure],
        solidus_func=None,
        liquidus_func=None,
        temperature_function=None,  # analytic EOS is temperature-independent
    )
    density = np.where(pressure > 0, np.sqrt(np.abs(pressure) / K), 0.0)
    i = int(np.where(pressure > 0)[0][-1])
    drho_dr = (density[i] - density[i - 1]) / (radii[i] - radii[i - 1])
    r_surf = radii[i] + density[i] / (-drho_dr)
    return float(r_surf), float(mass[i])


def test_surface_radius_matches_exact_lane_emden():
    """The emergent radius reproduces the exact n=1 radius R = R_earth.

    Reference-pinned against the closed-form Lane-Emden surface. The 1e-4 tolerance is tight
    enough to require the density extrapolation this test uses: density vanishes linearly at the
    surface, so extrapolating it recovers R to O(dr^2) (~3e-6 residual on this grid), whereas
    extrapolating the quadratically vanishing pressure would only give O(dr) (~7e-4), which this
    tolerance rejects. A mis-set K (which scales R as sqrt(K)) or a wrong density exponent would
    move the radius by percent-level, far outside the tolerance.
    """
    r_surf, _mass = _integrate_polytrope(earth_mass)
    assert r_surf == pytest.approx(_R_LE, rel=1e-4)
    # Discrimination guard: a factor-2 error in K would put R at sqrt(2) R_earth (~41% off).
    assert abs(r_surf / _R_LE - np.sqrt(2.0)) > 0.1


def test_integrated_mass_matches_analytic_mass():
    """The integrated mass recovers the analytic M = 4 rho_c R^3 / pi it was seeded from.

    Closes the mass budget of the ODE integration: the emergent total mass equals the mass
    implied by the seeded central density, verifying mass conservation through the solve.
    """
    _r_surf, mass_total = _integrate_polytrope(2.0 * earth_mass)
    assert mass_total == pytest.approx(2.0 * earth_mass, rel=1e-4)
    # Discrimination guard: dropping the 4 pi in dM/dr = 4 pi rho r^2 would scale the integrated
    # mass by 1/(4 pi) ~ 0.08 of the seeded value, far below the pinned ratio of 1.
    assert abs(mass_total / (2.0 * earth_mass) - 1.0 / (4.0 * np.pi)) > 0.5


def test_radius_is_mass_independent():
    """The n=1 radius is independent of central density: R(1 M) == R(2 M).

    This distinguishes the genuine Chandrasekhar n=1 solution from a mass-dependent
    artifact. Any polytrope index other than 1 yields a radius that depends on central
    density, so agreement across a factor-2 mass change is a strong, non-trivially
    derivable invariant, not a coincidence of a single run.
    """
    r1, _ = _integrate_polytrope(1.0 * earth_mass)
    r2, _ = _integrate_polytrope(2.0 * earth_mass)
    assert r1 == pytest.approx(r2, rel=1e-4)
    # Both land at one Earth radius, not merely equal to each other at some other radius.
    assert r1 == pytest.approx(_R_LE, rel=1e-3)
