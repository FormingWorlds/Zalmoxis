# This file contains the main function that solves the coupled ODEs for the structure model.
from __future__ import annotations

import logging

import numpy as np
from scipy.integrate import solve_ivp

from .constants import G
from .eos_functions import calculate_density

# Set up logging
logger = logging.getLogger(__name__)


def get_layer_eos(mass, cmb_mass, core_mantle_mass, layer_eos_config):
    """Determine the per-layer EOS string based on enclosed mass (purely geometric).

    Parameters
    ----------
    mass : float
        Enclosed mass at the current radial shell [kg].
    cmb_mass : float
        Core-mantle boundary mass [kg].
    core_mantle_mass : float
        Core + mantle mass [kg].
    layer_eos_config : dict
        Per-layer EOS strings, e.g.
        {"core": "Seager2007:iron", "mantle": "WolfBower2018:MgSiO3"}.

    Returns
    -------
    str
        Per-layer EOS identifier for this shell.
    """
    if mass < cmb_mass:
        return layer_eos_config['core']
    elif 'ice_layer' in layer_eos_config and mass >= core_mantle_mass:
        return layer_eos_config['ice_layer']
    else:
        return layer_eos_config['mantle']


# Define the coupled ODEs for the structure model
def coupled_odes(
    radius,
    y,
    cmb_mass,
    core_mantle_mass,
    layer_eos_config,
    interpolation_cache,
    material_dictionaries,
    temperature,
    solidus_func,
    liquidus_func,
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
    layer_eos_config : dict
        Per-layer EOS configuration.
    interpolation_cache : dict
        Cache for interpolation functions.
    material_dictionaries : tuple
        Material property dictionaries.
    temperature : float
        Temperature at current radius [K].
    solidus_func : callable or None
        Solidus melting curve interpolation function.
    liquidus_func : callable or None
        Liquidus melting curve interpolation function.

    Returns
    -------
    list
        Derivatives [dM/dr, dg/dr, dP/dr].
    """
    # Unpack the state vector
    mass, gravity, pressure = y

    # Determine per-layer EOS for the current enclosed mass
    layer_eos = get_layer_eos(mass, cmb_mass, core_mantle_mass, layer_eos_config)

    # Check for nonphysical pressure values
    if pressure <= 0 or np.isnan(pressure):
        logger.debug(f'Nonphysical pressure encountered: P={pressure} Pa at radius={radius} m')

    # Calculate density at the current radius, using pressure from y
    current_density = calculate_density(
        pressure,
        material_dictionaries,
        layer_eos,
        temperature,
        solidus_func,
        liquidus_func,
        interpolation_cache,
    )

    # Fail fast on invalid density — continuing with zero derivatives would
    # silently produce non-physical profiles that pass convergence checks.
    if current_density is None or np.isnan(current_density):
        raise RuntimeError(
            f'Density calculation failed at radius={radius:.4e} m, P={pressure:.4e} Pa, '
            f'layer_eos={layer_eos}. Cannot continue integration.'
        )

    # Define the ODEs for mass, gravity and pressure
    dMdr = 4 * np.pi * radius**2 * current_density
    dgdr = 4 * np.pi * G * current_density - 2 * gravity / (radius + 1e-20) if radius > 0 else 0
    dPdr = -current_density * gravity

    # Return the derivatives
    return [dMdr, dgdr, dPdr]


def solve_structure(
    layer_eos_config,
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
):
    """Solve the coupled ODEs for the planetary structure model.

    Handles the special case for temperature-dependent EOS where the radial
    grid is split into two parts for better handling of large step sizes
    towards the surface.

    Parameters
    ----------
    layer_eos_config : dict
        Per-layer EOS configuration, e.g.
        {"core": "Seager2007:iron", "mantle": "WolfBower2018:MgSiO3"}.
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
    material_dictionaries : tuple
        Material property dictionaries.
    interpolation_cache : dict
        Cache for interpolation functions.
    y0 : array-like
        Initial conditions [mass, gravity, pressure] at center.
    solidus_func : callable or None
        Solidus melting curve interpolation function.
    liquidus_func : callable or None
        Liquidus melting curve interpolation function.
    temperature_function : callable or None
        Function returning temperature [K] as function of radius [m].

    Returns
    -------
    tuple
        (mass_enclosed, gravity, pressure) arrays at each radial grid point.
    """
    uses_Tdep = any(v == 'WolfBower2018:MgSiO3' for v in layer_eos_config.values() if v)

    if uses_Tdep:
        # Split the radial grid into two parts for better handling of large step sizes
        radial_split_index = max(
            1, min(len(radii) - 1, int(adaptive_radial_fraction * len(radii)))
        )

        # Solve the ODEs in two parts, first part with default max_step (adaptive)
        sol1 = solve_ivp(
            lambda r, y: coupled_odes(
                r,
                y,
                cmb_mass,
                core_mantle_mass,
                layer_eos_config,
                interpolation_cache,
                material_dictionaries,
                temperature_function(r),
                solidus_func,
                liquidus_func,
            ),
            (radii[0], radii[radial_split_index - 1]),
            y0,
            t_eval=radii[:radial_split_index],
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            method='RK45',
            dense_output=True,
        )

        # Second part with user-defined max_step
        sol2 = solve_ivp(
            lambda r, y: coupled_odes(
                r,
                y,
                cmb_mass,
                core_mantle_mass,
                layer_eos_config,
                interpolation_cache,
                material_dictionaries,
                temperature_function(r),
                solidus_func,
                liquidus_func,
            ),
            (radii[radial_split_index - 1], radii[-1]),
            sol1.y[:, -1],
            t_eval=radii[radial_split_index - 1 :],
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            max_step=maximum_step,
            method='RK45',
            dense_output=True,
        )

        # Concatenate the two solutions
        mass_enclosed = np.concatenate([sol1.y[0, :-1], sol2.y[0]])
        gravity = np.concatenate([sol1.y[1, :-1], sol2.y[1]])
        pressure = np.concatenate([sol1.y[2, :-1], sol2.y[2]])
    else:
        # Single integration with fixed temperature (300 K for Seager+2007)
        temperature = 300
        sol = solve_ivp(
            lambda r, y: coupled_odes(
                r,
                y,
                cmb_mass,
                core_mantle_mass,
                layer_eos_config,
                interpolation_cache,
                material_dictionaries,
                temperature,
                solidus_func,
                liquidus_func,
            ),
            (radii[0], radii[-1]),
            y0,
            t_eval=radii,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            method='RK45',
            dense_output=True,
        )

        # Extract mass, gravity, and pressure grids from the solution
        mass_enclosed = sol.y[0]
        gravity = sol.y[1]
        pressure = sol.y[2]

    return mass_enclosed, gravity, pressure
