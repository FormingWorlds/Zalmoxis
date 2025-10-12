# This file contains the main function that solves the coupled ODEs for the structure model.
from __future__ import annotations

import logging

import numpy as np

from .constants import G
from .eos_functions import calculate_density

# Set up logging
logger = logging.getLogger(__name__)

# Define the coupled ODEs for the structure model
def coupled_odes(radius, y, cmb_mass, core_mantle_mass, EOS_CHOICE, temperature, interpolation_cache, material_dictionaries):
    """
    Calculate the derivatives of mass, gravity, and pressure with respect to radius for a planetary model.

    Parameters:
    radius (float): The current radius at which the ODEs are evaluated.
    y (list or array): The state vector containing mass, gravity, and pressure at the current radius.
    cmb_mass (float): The core-mantle boundary mass.
    core_mantle_mass (float): The mass of the core+mantle.
    EOS_CHOICE (str): The equation of state choice for the material.
    temperature (float): The temperature at the current radius.
    interpolation_cache (dict): A cache for interpolation to speed up calculations.

    Returns:
    list: The derivatives of mass, gravity, and pressure with respect to radius.
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
            material = None # placeholder to avoid material not defined error
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
        logger.error(f"Nonphysical pressure encountered: P={pressure} Pa at radius={radius} m")

    # Calculate density at the current radius, using pressure from y
    current_density = calculate_density(pressure, material_dictionaries, material, EOS_CHOICE, temperature, interpolation_cache)

    # Handle potential errors in density calculation
    if current_density is None or np.isnan(current_density):
        logger.error(f"Density calculation failed at radius={radius}, P={pressure}")

    # Define the ODEs for mass, gravity and pressure
    dMdr = 4 * np.pi * radius**2 * current_density
    dgdr = 4 * np.pi * G * current_density - 2 * gravity / (radius + 1e-20) if radius > 0 else 0
    dPdr = -current_density * gravity

    # Return the derivatives
    return [dMdr, dgdr, dPdr]

