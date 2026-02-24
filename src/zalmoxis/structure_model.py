"""
Structure model ODE solver.

!!! Imports
    - **Constants**: [`zalmoxis.constants`](zalmoxis.constants.md) – `G`
    - **EOS**: [`zalmoxis.eos_functions`](zalmoxis.eos_functions.md) — `calculate_density`

This module contains the coupled ODEs and the solver wrapper used by the Zalmoxis structure model.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.integrate import solve_ivp

from .constants import G
from .eos_functions import calculate_density

# Set up logging
logger = logging.getLogger(__name__)

# Define the coupled ODEs for the structure model
def coupled_odes(radius, y, cmb_mass, core_mantle_mass, EOS_CHOICE, interpolation_cache, material_dictionaries, temperature, solidus_func, liquidus_func):
    """
    Derivatives for the coupled planetary structure ODE system.

    Parameters
    ----------
    radius : float
        Radius where the ODEs are evaluated [m].
    y : array_like
        State vector ``[mass, gravity, pressure]`` at ``radius``.
    cmb_mass : float
        Core–mantle boundary mass [kg].
    core_mantle_mass : float
        Core+mantle mass [kg].
    EOS_CHOICE : str
        EOS identifier (e.g. ``"Tabulated:iron/silicate"``).
    interpolation_cache : dict
        Cache for EOS interpolation functions.
    material_dictionaries : tuple
        Material property dictionaries.
    temperature : float
        Temperature at ``radius`` [K].
    solidus_func : callable
        Solidus melting curve ``T_sol(P)``.
    liquidus_func : callable
        Liquidus melting curve ``T_liq(P)``.

    Returns
    -------
    list[float]
        Derivatives ``[dMdr, dgdr, dPdr]``.
    """
    # Unpack the state vector
    mass, gravity, pressure = y

    # Define material based on enclosed mass within a certain mass fraction
    if EOS_CHOICE == "Tabulated:iron/silicate":
        # Define the material type based on the calculated enclosed mass up to the core-mantle boundary
        if mass < cmb_mass:
            # Core
            material = "core"
        else:
            # Mantle
            material = "mantle"

    elif EOS_CHOICE == "Tabulated:iron/Tdep_silicate":
        # Define the material type based on the calculated enclosed mass up to the core-mantle boundary
        if mass < cmb_mass:
            # Core
            material = "core"
        else:
            # Mantle, uncomment the next line to assign material based on temperature and pressure
            material = "mantle" # placeholder (can be melted or solid depending on T and P)
            #material = get_Tdep_material(pressure, temperature) #optional to assign since get_Tdep_density handles material assignment internally
            pass

    elif EOS_CHOICE == "Tabulated:water":
        # Define the material type based on the calculated enclosed mass up to the core-mantle boundary
        if mass < cmb_mass:
            # Core
            material = "core"
        elif mass < core_mantle_mass:
            # Inner mantle
            material = "mantle"
        else:
            # Outer layer
            material = "water_ice_layer"
    else:
        raise ValueError(f"Unknown EOS_CHOICE '{EOS_CHOICE}'. "
                         "Valid options: 'Tabulated:iron/silicate', 'Tabulated:iron/Tdep_silicate', 'Tabulated:water'.")

    # Check for nonphysical pressure values
    if pressure <= 0 or np.isnan(pressure):
        logger.debug(f"Nonphysical pressure encountered: P={pressure} Pa at radius={radius} m")

    # Calculate density at the current radius, using pressure from y
    current_density = calculate_density(pressure, material_dictionaries, material, EOS_CHOICE, temperature, solidus_func, liquidus_func, interpolation_cache)

    # Handle potential errors in density calculation
    if current_density is None or np.isnan(current_density):
        logger.error(f"Density calculation failed at radius={radius}, P={pressure}")

    # Define the ODEs for mass, gravity and pressure
    dMdr = 4 * np.pi * radius**2 * current_density
    dgdr = 4 * np.pi * G * current_density - 2 * gravity / (radius + 1e-20) if radius > 0 else 0
    dPdr = -current_density * gravity

    # Return the derivatives
    return [dMdr, dgdr, dPdr]

def solve_structure(EOS_CHOICE, cmb_mass, core_mantle_mass, radii, adaptive_radial_fraction, relative_tolerance, absolute_tolerance, maximum_step, material_dictionaries, interpolation_cache, y0, solidus_func, liquidus_func, temperature_function=None):
    """
    Solve the coupled ODEs for the planetary structure model using
    ``scipy.integrate.solve_ivp``.

    For the temperature-dependent EOS (``"Tabulated:iron/Tdep_silicate"``),
    the radial grid is split into two integration regions to improve
    numerical stability and control step sizes near the surface.

    Parameters
    ----------
    EOS_CHOICE : str
        Specifies the equation of state (EOS) model used for the interior
        structure calculation.
    cmb_mass : float
        Mass at the core–mantle boundary [kg].
    core_mantle_mass : float
        Core+mantle mass [kg].
    radii : numpy.ndarray
        One-dimensional radial grid [m] across which the structure equations
        are solved.
    adaptive_radial_fraction : float
        Fraction (0–1) of the radial domain defining where the solver
        transitions from adaptive integration to fixed-step integration
        when using a temperature-dependent EOS.
    relative_tolerance : float
        Relative tolerance passed to ``solve_ivp``.
    absolute_tolerance : float
        Absolute tolerance passed to ``solve_ivp``.
    maximum_step : float
        Maximum integration step size [m].
    material_dictionaries : tuple
        Tuple containing the material property dictionaries for iron/silicate,
        water, and temperature-dependent silicate planets.
    interpolation_cache : dict
        Cache used to store interpolation functions.
    y0 : array_like
        Initial conditions ``[mass, gravity, pressure]`` at the planetary
        center.
    solidus_func : callable
        Interpolation function for the solidus melting curve.
    liquidus_func : callable
        Interpolation function for the liquidus melting curve.
    temperature_function : callable, optional
        Function returning temperature [K] as a function of radius [m].

    Returns
    -------
    mass_enclosed : numpy.ndarray
        Enclosed mass profile [kg].
    gravity : numpy.ndarray
        Gravity profile [m/s²].
    pressure : numpy.ndarray
        Pressure profile [Pa].
    """

    if EOS_CHOICE == "Tabulated:iron/Tdep_silicate":
        # Split the radial grid into two parts for better handling of large step sizes in solve_ivp
        radial_split_index = int(adaptive_radial_fraction * len(radii))

        # Solve the ODEs in two parts, first part with default max_step (adaptive)
        sol1 = solve_ivp(lambda r, y: coupled_odes(r, y, cmb_mass, core_mantle_mass, EOS_CHOICE, interpolation_cache, material_dictionaries, temperature_function(r), solidus_func, liquidus_func),
            (radii[0], radii[radial_split_index-1]), y0, t_eval=radii[:radial_split_index], rtol=relative_tolerance, atol=absolute_tolerance, method='RK45', dense_output=True)

        # Solve the ODEs in two parts, second part with user-defined max_step
        sol2 = solve_ivp(lambda r, y: coupled_odes(r, y, cmb_mass, core_mantle_mass, EOS_CHOICE, interpolation_cache, material_dictionaries, temperature_function(r), solidus_func, liquidus_func),
            (radii[radial_split_index-1], radii[-1]), sol1.y[:, -1], t_eval=radii[radial_split_index-1:], rtol=relative_tolerance, atol=absolute_tolerance, max_step=maximum_step, method='RK45', dense_output=True)

        # Extract mass, gravity, and pressure grids from the two solutions and concatenate them
        mass_enclosed = np.concatenate([sol1.y[0, :-1], sol2.y[0]])
        gravity = np.concatenate([sol1.y[1, :-1], sol2.y[1]])
        pressure = np.concatenate([sol1.y[2, :-1], sol2.y[2]])
    else:
        # Solve the ODEs using solve_ivp
        temperature = 300 # Fixed-temperature for EOS from Seager et al. 2007
        sol = solve_ivp(lambda r, y: coupled_odes(r, y, cmb_mass, core_mantle_mass, EOS_CHOICE, interpolation_cache, material_dictionaries, temperature, solidus_func, liquidus_func),
        (radii[0], radii[-1]), y0, t_eval=radii, rtol=relative_tolerance, atol=absolute_tolerance, method='RK45', dense_output=True)

        # Extract mass, gravity, and pressure grids from the solution
        mass_enclosed = sol.y[0]
        gravity = sol.y[1]
        pressure = sol.y[2]

    return mass_enclosed, gravity, pressure
