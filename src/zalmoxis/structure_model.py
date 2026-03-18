"""Main function to solve the coupled ODEs for the structure model.

!!! Imports
    - [`constants`](zalmoxis.constants.md): G
    - [`mixing`](zalmoxis.mixing.md): LayerMixture, calculate_mixed_density, any_component_is_tdep

"""

# This file contains the main function that solves the coupled ODEs for the structure model.
from __future__ import annotations

import logging

import numpy as np
from scipy.integrate import solve_ivp

from .constants import CONDENSED_RHO_MIN_DEFAULT, CONDENSED_RHO_SCALE_DEFAULT, G
from .mixing import any_component_is_tdep, calculate_mixed_density

# Set up logging
logger = logging.getLogger(__name__)


def get_layer_eos(mass, cmb_mass, core_mantle_mass, layer_mixtures):
    """Determine the per-layer mixture based on enclosed mass (purely geometric).

    Parameters
    ----------
    mass : float
        Enclosed mass at the current radial shell [kg].
    cmb_mass : float
        Core-mantle boundary mass [kg].
    core_mantle_mass : float
        Core + mantle mass [kg].
    layer_mixtures : dict
        Per-layer LayerMixture objects, e.g.
        ``{"core": LayerMixture(...), "mantle": LayerMixture(...)}``.

    Returns
    -------
    LayerMixture
        Mixture for this shell.
    """
    # Use a small relative tolerance for boundary comparisons to avoid
    # float-precision misassignment when a mesh node lands exactly at
    # the CMB or core-mantle boundary.
    rtol = 1e-12
    if mass < cmb_mass * (1.0 - rtol):
        return layer_mixtures['core']
    elif 'ice_layer' in layer_mixtures and mass >= core_mantle_mass * (1.0 - rtol):
        return layer_mixtures['ice_layer']
    else:
        return layer_mixtures['mantle']


# Define the coupled ODEs for the structure model
def coupled_odes(
    radius,
    y,
    cmb_mass,
    core_mantle_mass,
    layer_mixtures,
    interpolation_cache,
    material_dictionaries,
    temperature,
    solidus_func,
    liquidus_func,
    mushy_zone_factors=None,
    condensed_rho_min=CONDENSED_RHO_MIN_DEFAULT,
    condensed_rho_scale=CONDENSED_RHO_SCALE_DEFAULT,
):
    """Calculate derivatives of mass, gravity, and pressure w.r.t. radius.

    Parameters
    ----------
    radius : float
        Current radius [m].
    y : array-like
        State vector [mass, gravity, pressure].
    cmb_mass : float
        Core-mantle boundary mass [kg].
    core_mantle_mass : float
        Core + mantle mass [kg].
    layer_mixtures : dict
        Per-layer LayerMixture objects.
    interpolation_cache : dict
        Cache for interpolation functions.
    material_dictionaries : dict
        EOS registry dict keyed by EOS identifier string.
    temperature : float
        Temperature at current radius [K].
    solidus_func : callable or None
        Solidus melting curve interpolation function.
    liquidus_func : callable or None
        Liquidus melting curve interpolation function.
    mushy_zone_factors : dict or float or None
        Per-EOS mushy zone factors. Dict keyed by EOS name, a single
        float (applied to all), or None (default 1.0 for all).
    condensed_rho_min : float
        Sigmoid center for phase-aware suppression (kg/m^3).
    condensed_rho_scale : float
        Sigmoid width for phase-aware suppression (kg/m^3).

    Returns
    -------
    list
        Derivatives [dM/dr, dg/dr, dP/dr].
    """
    # Unpack the state vector
    mass, gravity, pressure = y

    # Determine per-layer mixture for the current enclosed mass
    mixture = get_layer_eos(mass, cmb_mass, core_mantle_mass, layer_mixtures)

    # Return zero derivatives for non-physical pressure.  The adaptive ODE
    # solver (RK45) may evaluate trial points beyond the physical domain;
    # zero derivatives signal the solver to reject the step and retry smaller.
    if pressure <= 0 or np.isnan(pressure):
        logger.debug(f'Nonphysical pressure encountered: P={pressure} Pa at radius={radius} m')
        return [0.0, 0.0, 0.0]

    # Calculate density at the current radius, using pressure from y
    current_density = calculate_mixed_density(
        pressure,
        temperature,
        mixture,
        material_dictionaries,
        solidus_func,
        liquidus_func,
        interpolation_cache,
        mushy_zone_factors,
        condensed_rho_min,
        condensed_rho_scale,
    )

    # Return zero derivatives for invalid density.  This is intentional:
    # the adaptive ODE solver (RK45) evaluates the RHS at trial points that
    # may be non-physical (e.g. negative pressure).  Zero derivatives cause
    # the solver to reject the step and retry with a smaller step size.
    if current_density is None or not np.isfinite(current_density):
        return [0.0, 0.0, 0.0]

    # Define the ODEs for mass, gravity and pressure
    dMdr = 4 * np.pi * radius**2 * current_density
    # At r=0 the 2g/r term is singular; use the analytic limit dg/dr = 4πGρ/3.
    # For r>0, add a tiny offset (1e-20 m, ~1e-14 R_earth) to guard against
    # the adaptive solver probing exactly r=0.
    dgdr = (
        4 * np.pi * G * current_density - 2 * gravity / (radius + 1e-20)
        if radius > 0
        else (4.0 / 3.0) * np.pi * G * current_density
    )
    dPdr = -current_density * gravity

    # Return the derivatives
    return [dMdr, dgdr, dPdr]


def solve_structure(
    layer_mixtures,
    cmb_mass,
    core_mantle_mass,
    radii,
    adaptive_radial_fraction,
    relative_tolerance,
    absolute_tolerance,
    maximum_step,
    material_dictionaries,
    interpolation_cache,
    y0,
    solidus_func,
    liquidus_func,
    temperature_function=None,
    mushy_zone_factors=None,
    condensed_rho_min=CONDENSED_RHO_MIN_DEFAULT,
    condensed_rho_scale=CONDENSED_RHO_SCALE_DEFAULT,
):
    """Solve the coupled ODEs for the planetary structure model.

    Handles the special case for temperature-dependent EOS where the radial
    grid is split into two parts for better handling of large step sizes
    towards the surface.

    Parameters
    ----------
    layer_mixtures : dict
        Per-layer LayerMixture objects, e.g.
        ``{"core": LayerMixture(...), "mantle": LayerMixture(...)}``.
    cmb_mass : float
        Mass at the core-mantle boundary [kg].
    core_mantle_mass : float
        Core + mantle mass [kg].
    radii : numpy.ndarray
        Radial grid points [m].
    adaptive_radial_fraction : float
        Fraction of radial domain for adaptive-to-fixed step transition.
    relative_tolerance : float
        Relative tolerance for solve_ivp.
    absolute_tolerance : float
        Absolute tolerance for solve_ivp.
    maximum_step : float
        Maximum integration step size [m].
    material_dictionaries : dict
        EOS registry dict keyed by EOS identifier string.
    interpolation_cache : dict
        Cache for interpolation functions.
    y0 : array-like
        Initial conditions [mass, gravity, pressure] at center.
    solidus_func : callable or None
        Solidus melting curve interpolation function.
    liquidus_func : callable or None
        Liquidus melting curve interpolation function.
    temperature_function : callable or None
        Function returning temperature [K]. Signature: ``f(r, P) -> T``
        where ``r`` is radius in m and ``P`` is pressure in Pa. For
        non-adiabatic modes the pressure argument is ignored.
    mushy_zone_factors : dict or float or None
        Per-EOS mushy zone factors. Dict keyed by EOS name, a single
        float (applied to all), or None (default 1.0 for all).
    condensed_rho_min : float
        Sigmoid center for phase-aware suppression (kg/m^3).
    condensed_rho_scale : float
        Sigmoid width for phase-aware suppression (kg/m^3).

    Returns
    -------
    tuple
        (mass_enclosed, gravity, pressure) arrays at each radial grid point.
    """
    uses_Tdep = any_component_is_tdep(layer_mixtures)

    # Terminal event: stop integration when pressure crosses zero.
    # Without this, the ODE solver grinds with tiny step sizes in the
    # zero-derivative region returned by coupled_odes() for P <= 0.
    def _pressure_zero(r, y, *args):
        return y[2]  # pressure component

    _pressure_zero.terminal = True
    _pressure_zero.direction = -1  # trigger on positive → negative crossing

    def _ode_rhs(r, y):
        return coupled_odes(
            r,
            y,
            cmb_mass,
            core_mantle_mass,
            layer_mixtures,
            interpolation_cache,
            material_dictionaries,
            temperature_function(r, y[2]) if temperature_function else 300,
            solidus_func,
            liquidus_func,
            mushy_zone_factors,
            condensed_rho_min,
            condensed_rho_scale,
        )

    if uses_Tdep:
        # Split the radial grid into two parts for better handling of large step sizes
        radial_split_index = max(
            1, min(len(radii) - 1, int(adaptive_radial_fraction * len(radii)))
        )

        # Solve the ODEs in two parts, first part with default max_step (adaptive)
        sol1 = solve_ivp(
            _ode_rhs,
            (radii[0], radii[radial_split_index - 1]),
            y0,
            t_eval=radii[:radial_split_index],
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            method='RK45',
            events=_pressure_zero,
        )

        # If sol1 hit the terminal event (pressure reached zero), skip sol2
        if sol1.status == 1:
            mass_enclosed = sol1.y[0]
            gravity = sol1.y[1]
            pressure = sol1.y[2]
        else:
            # Second part with user-defined max_step
            sol2 = solve_ivp(
                _ode_rhs,
                (radii[radial_split_index - 1], radii[-1]),
                sol1.y[:, -1],
                t_eval=radii[radial_split_index - 1 :],
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                max_step=maximum_step,
                method='RK45',
                events=_pressure_zero,
            )

            # Concatenate the two solutions
            mass_enclosed = np.concatenate([sol1.y[0, :-1], sol2.y[0]])
            gravity = np.concatenate([sol1.y[1, :-1], sol2.y[1]])
            pressure = np.concatenate([sol1.y[2, :-1], sol2.y[2]])
    else:
        # Single integration with fixed temperature (300 K for Seager+2007)
        sol = solve_ivp(
            _ode_rhs,
            (radii[0], radii[-1]),
            y0,
            t_eval=radii,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            method='RK45',
            events=_pressure_zero,
        )

        # Extract mass, gravity, and pressure grids from the solution
        mass_enclosed = sol.y[0]
        gravity = sol.y[1]
        pressure = sol.y[2]

    # Pad to full length if the terminal event truncated the solution
    # (pressure reached zero before the outermost radial grid point).
    n_target = len(radii)
    if len(mass_enclosed) < n_target:
        n_pad = n_target - len(mass_enclosed)
        mass_enclosed = np.concatenate([mass_enclosed, np.full(n_pad, mass_enclosed[-1])])
        gravity = np.concatenate([gravity, np.full(n_pad, gravity[-1])])
        pressure = np.concatenate([pressure, np.zeros(n_pad)])

    return mass_enclosed, gravity, pressure
