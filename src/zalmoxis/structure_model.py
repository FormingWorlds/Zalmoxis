# This file contains the main function that solves the coupled ODEs for the structure model.

import numpy as np
from .eos_functions import calculate_density
from .constants import *

# Define the coupled ODEs for the structure model
def coupled_odes(radius, y, cmb_mass, eos_choice, interpolation_cache):
    """
    Calculate the derivatives of mass, gravity, and pressure with respect to radius for a planetary model.

    Parameters:
    radius (float): The current radius at which the ODEs are evaluated.
    y (list or array): The state vector containing mass, gravity, and pressure at the current radius.
    cmb_mass (float): The core-mantle boundary mass.
    eos_choice (str): The equation of state choice for material properties.
    interpolation_cache (dict): A cache for interpolation to speed up calculations.
    num_layers (int): The number of layers in the planetary model.

    Returns:
    list: The derivatives of mass, gravity, and pressure with respect to radius.
    """
    # Unpack the state vector
    mass, gravity, pressure = y

    # Define material based on enclosed mass up to the core-mantle boundary
    if mass < cmb_mass:
        material = "core"
    else:
        material = "mantle" 

    # Calculate density at the current radius, using pressure from y
    current_density = calculate_density(pressure, material, eos_choice, interpolation_cache)

    # Handle potential errors in density calculation
    if current_density is None:
        print(f"Warning: Density calculation failed at radius {radius}") 

    # Define the ODEs for mass, gravity and pressure
    dMdr = 4 * np.pi * radius**2 * current_density
    dgdr = 4 * np.pi * G * current_density - 2 * gravity / (radius + 1e-20) if radius > 0 else 0
    dPdr = -current_density * gravity

    # Return the derivatives
    return [dMdr, dgdr, dPdr]

