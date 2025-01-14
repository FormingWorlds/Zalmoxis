# This file contains the main function that solves the coupled ODEs for the structure model.

import numpy as np
from .eos_functions import calculate_density, calculate_temperature
from .constants import *

# Define the coupled ODEs for the structure model
def coupled_odes(radius, y, cmb_radius, radius_guess, cmb_temp_guess, core_temp_guess, EOS_CHOICE, interpolation_cache):
    # Unpack the state vector
    mass, gravity, pressure = y

    # Define material based on radius
    if radius < cmb_radius:
        material = "core"
    else:
        material = "mantle" # Assign material only once per call

    # Calculate density at the current radius, using pressure from y
    current_density = calculate_density(pressure, radius, cmb_radius, material, radius_guess, cmb_temp_guess, core_temp_guess, EOS_CHOICE, interpolation_cache)

    # Handle potential errors in density calculation
    if current_density is None:
        print(f"Warning: Density calculation failed at radius {radius}. Using previous density.") # Print warning only
        #current_density = old_density[np.argmin(np.abs(radii - radius))]

    # Calculate temperature
    temperature = calculate_temperature(radius, cmb_radius, 300, cmb_temp_guess, core_temp_guess, radius_guess)

    # Define the ODEs for mass, gravity and pressure
    dMdr = 4 * np.pi * radius**2 * current_density
    dgdr = 4 * np.pi * G * current_density - 2 * gravity / (radius + 1e-20) if radius > 0 else 0
    dPdr = -current_density * gravity

    # Return the derivatives
    return [dMdr, dgdr, dPdr]