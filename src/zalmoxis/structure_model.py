# This file contains the main function that solves the coupled ODEs for the structure model.
from __future__ import annotations

import logging

import numpy as np
from scipy.integrate import solve_ivp

from .constants import G
from .eos_functions import calculate_density

# Set up logging
logger = logging.getLogger(__name__)


def get_layer_material(mass, cmb_mass, core_mantle_mass, eos_choice, analytic_materials=None):
    """
    Determine the material/layer label for a given enclosed mass.

    For tabulated EOS choices this returns a layer label (e.g., "core", "mantle").
    For the analytic EOS this returns the configured material key (e.g., "iron", "MgSiO3").

    Parameters
    ----------
    mass : float
        Enclosed mass at the current radial shell [kg].
    cmb_mass : float
        Core-mantle boundary mass [kg].
    core_mantle_mass : float
        Core + mantle mass [kg].
    eos_choice : str
        EOS identifier string.
    analytic_materials : dict or None
        Per-layer material keys for Analytic:Seager2007.

    Returns
    -------
    str
        Material or layer identifier.
    """
    if eos_choice == 'Tabulated:iron/silicate':
        return 'core' if mass < cmb_mass else 'mantle'

    elif eos_choice == 'Tabulated:iron/Tdep_silicate':
        # Layer label only; phase assignment is handled inside get_Tdep_density
        return 'core' if mass < cmb_mass else 'mantle'

    elif eos_choice == 'Tabulated:water':
        if mass < cmb_mass:
            return 'core'
        elif mass < core_mantle_mass:
            return 'mantle'
        else:
            return 'water_ice_layer'

    elif eos_choice == 'Analytic:Seager2007':
        if mass < cmb_mass:
            return analytic_materials['core']
        elif core_mantle_mass > 0 and mass < core_mantle_mass:
            return analytic_materials['mantle']
        elif 'water_ice_layer' in analytic_materials:
            return analytic_materials['water_ice_layer']
        else:
            return analytic_materials['mantle']

    else:
        raise ValueError(
            f"Unknown eos_choice '{eos_choice}'. "
            "Valid options: 'Tabulated:iron/silicate', 'Tabulated:iron/Tdep_silicate', "
            "'Tabulated:water', 'Analytic:Seager2007'."
        )


# Define the coupled ODEs for the structure model
def coupled_odes(
    radius,
    y,
    cmb_mass,
    core_mantle_mass,
    EOS_CHOICE,
    interpolation_cache,
    material_dictionaries,
    temperature,
    solidus_func,
    liquidus_func,
    analytic_materials=None,
):
    """
    Calculate the derivatives of mass, gravity, and pressure with respect to radius for a planetary model.

    Parameters:
        radius (float): The current radius at which the ODEs are evaluated.
        y (list or array): The state vector containing mass, gravity, and pressure at the current radius.
        cmb_mass (float): The core-mantle boundary mass.
        core_mantle_mass (float): The mass of the core+mantle.
        EOS_CHOICE (str): The equation of state choice for the material.
        interpolation_cache (dict): A cache for interpolation to speed up calculations.
        material_dictionaries (dict): A tuple containing the material properties dictionaries.
        temperature (float): The temperature at the current radius.
        solidus_func: Interpolation function for the solidus melting curve
        liquidus_func: Interpolation function for the liquidus melting curve
        analytic_materials (dict or None): Per-layer material keys for Analytic:Seager2007 EOS.
            Keys: "core", "mantle", and optionally "water_ice_layer".
            Values: material key strings (e.g., "iron", "MgSiO3").

    Returns:
        list: The derivatives of mass, gravity, and pressure with respect to radius.
    """
    # Unpack the state vector
    mass, gravity, pressure = y

    # Determine material/layer for the current enclosed mass
    material = get_layer_material(
        mass, cmb_mass, core_mantle_mass, EOS_CHOICE, analytic_materials
    )

    # Check for nonphysical pressure values
    if pressure <= 0 or np.isnan(pressure):
        logger.debug(f'Nonphysical pressure encountered: P={pressure} Pa at radius={radius} m')

    # Calculate density at the current radius, using pressure from y
    current_density = calculate_density(
        pressure,
        material_dictionaries,
        material,
        EOS_CHOICE,
        temperature,
        solidus_func,
        liquidus_func,
        interpolation_cache,
    )

    # Guard against failed density calculation to prevent ODE solver crash
    if current_density is None or np.isnan(current_density):
        logger.error(
            f'Density calculation failed at radius={radius}, P={pressure}. '
            'Returning zero derivatives to allow solver to continue.'
        )
        return [0.0, 0.0, 0.0]

    # Define the ODEs for mass, gravity and pressure
    dMdr = 4 * np.pi * radius**2 * current_density
    dgdr = 4 * np.pi * G * current_density - 2 * gravity / (radius + 1e-20) if radius > 0 else 0
    dPdr = -current_density * gravity

    # Return the derivatives
    return [dMdr, dgdr, dPdr]


def solve_structure(
    EOS_CHOICE,
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
    analytic_materials=None,
):
    """
    Solve the coupled ODEs for the planetary structure model using scipy's solve_ivp. Handles special case for temperature-dependent EOS where the radial grid is split into two parts for better handling of large step sizes towards the surface.
    Parameters:
        EOS_CHOICE (str): Specifies the equation of state (EOS) model to use for the interior structure calculation.
        cmb_mass (float): Mass at the core–mantle boundary [kg].
        core_mantle_mass (float): Core+mantle mass [kg].
        radii (numpy.ndarray): 1D array of radial grid points [m] across which the structure equations are solved.
        adaptive_radial_fraction (float): Fraction (0–1) of the radial domain defining where the solver transitions from adaptive integration to fixed-step integration when using a temperature-dependent tabulated EOS.
        relative_tolerance (float): Relative tolerance for solve_ivp
        absolute_tolerance (float): Absolute tolerance for solve_ivp
        maximum_step (float): Maximum integration step size for solve_ivp (m)
        material_dictionaries (dict): A tuple containing the material properties dictionaries for iron/silicate planets, water planets, and temperature-dependent iron/silicate planets.
        interpolation_cache (dict): Cache used to store interpolation functions.
        temperature_function (callable): Function returning temperature [K] as a function of radius [m].
        y0 (list or numpy.ndarray): Initial conditions for the mass, gravity, and pressure at the center of the planet.
        solidus_func: Interpolation function for the solidus melting curve
        liquidus_func: Interpolation function for the liquidus melting curve
        analytic_materials (dict or None): Per-layer material keys for Analytic:Seager2007 EOS.
    Returns:
        tuple: A tuple containing three numpy arrays: mass_enclosed (kg), gravity (m/s²), and pressure (Pa) at each radial grid point.
    """

    if EOS_CHOICE == 'Tabulated:iron/Tdep_silicate':
        # Split the radial grid into two parts for better handling of large step sizes in solve_ivp
        radial_split_index = int(adaptive_radial_fraction * len(radii))

        # Solve the ODEs in two parts, first part with default max_step (adaptive)
        sol1 = solve_ivp(
            lambda r, y: coupled_odes(
                r,
                y,
                cmb_mass,
                core_mantle_mass,
                EOS_CHOICE,
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

        # Solve the ODEs in two parts, second part with user-defined max_step
        sol2 = solve_ivp(
            lambda r, y: coupled_odes(
                r,
                y,
                cmb_mass,
                core_mantle_mass,
                EOS_CHOICE,
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

        # Extract mass, gravity, and pressure grids from the two solutions and concatenate them
        mass_enclosed = np.concatenate([sol1.y[0, :-1], sol2.y[0]])
        gravity = np.concatenate([sol1.y[1, :-1], sol2.y[1]])
        pressure = np.concatenate([sol1.y[2, :-1], sol2.y[2]])
    else:
        # Solve the ODEs using solve_ivp
        temperature = 300  # Fixed-temperature for EOS from Seager et al. 2007
        sol = solve_ivp(
            lambda r, y: coupled_odes(
                r,
                y,
                cmb_mass,
                core_mantle_mass,
                EOS_CHOICE,
                interpolation_cache,
                material_dictionaries,
                temperature,
                solidus_func,
                liquidus_func,
                analytic_materials,
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
