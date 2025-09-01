"""
EOS Data and Functions
"""
from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import interp1d

# Set up logging
logger = logging.getLogger(__name__)

'''def calculate_temperature(radii, core_radius, surface_temp, material_properties, gravity, density, K_s, dr):
    """
    Computes the temperature profile inward from the surface using the Runge-Kutta 4th order method (RK4).
    Parameters:
    - radii: Array of radii (m) from the center to the surface.
    - core_radius: Radius of core-mantle boundary (m).
    - surface_temp: Surface temperature (K).
    - cmb_temp: Core-mantle boundary temperature (K).
    - material_properties: Material-specific parameters (rho, K, gamma).
    - gravity: Array of gravitational accelerations (m/s^2) corresponding to radii.
    - density: Array of densities (kg/m^3) corresponding to radii.
    - K_s: Isentropic bulk modulus (Pa).
    - dr: Step size in radial direction (meters).

    Returns:
    - temperature_profile: Array of temperatures (K) corresponding to the radii, starting from the surface.
    """
    # Initialize temperature profile array
    temperature = np.zeros_like(radii)
    temperature[-1] = surface_temp  # Surface temperature as the starting point

    # Iterate inward from the surface (reverse order in the arrays)
    for i in range(len(radii) - 1, 0, -1):
        radius = radii[i]

        # Determine the material type: mantle or core
        if radius > core_radius:
            material = "mantle"
            gamma = material_properties[material]["gamma0"]

        else:
            material = "core"
            gamma = material_properties[material]["gamma0"]

        # Define the adiabatic gradient equation
        def adiabatic_gradient(T, radius_idx):
            return -(density[radius_idx] * gravity[radius_idx] * gamma * T) / K_s

        # Apply RK4 Method
        current_temp = temperature[i]
        k1 = -dr * adiabatic_gradient(current_temp, i)  # Reverse integration direction
        k2 = -dr * adiabatic_gradient(current_temp + 0.5 * k1, i - 1)
        k3 = -dr * adiabatic_gradient(current_temp + 0.5 * k2, i - 1)
        k4 = -dr * adiabatic_gradient(current_temp + k3, i - 2)

        # Update temperature at the previous radius
        temperature[i - 1] = current_temp + (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return temperature'''

def get_tabulated_eos(pressure, material_dictionary, material, interpolation_functions={}):
    """
    Retrieves density from tabulated EOS data for a given material and choice of EOS.
    """
    props = material_dictionary[material]
    try:
        eos_file = props["eos_file"]
        # Caching: Store interpolation functions for reuse
        if eos_file not in interpolation_functions:
            data = np.loadtxt(eos_file, delimiter=',', skiprows=1)
            pressure_data = data[:, 1] * 1e9
            density_data = data[:, 0] * 1e3
            interpolation_functions[eos_file] = interp1d(pressure_data, density_data, bounds_error=False, fill_value="extrapolate")

        interpolation_function = interpolation_functions[eos_file]  # Retrieve from cache
        density = interpolation_function(pressure)  # Call to cached function

        if density is None or np.isnan(density):
            raise ValueError(f"Density calculation failed for {material} at {pressure:.2e} Pa.")

        return density

    except (ValueError, OSError) as e: # Catch file errors
        logger.error(f"Error with tabulated EOS for {material} at {pressure:.2e} Pa: {e}")
        return None
    except Exception as e: # Other errors
        logger.error(f"Unexpected error with tabulated EOS for {material} at {pressure:.2e} Pa: {e}")
        return None

def calculate_density(pressure, material_dictionaries, material, eos_choice, interpolation_functions={}):
    """Calculates density with caching for tabulated EOS."""

    #T = 0  # Temporary fix for tabulated EOS

    # Unpack material dictionaries
    material_properties_iron_silicate_planets, material_properties_water_planets = material_dictionaries

    if eos_choice == "Tabulated:iron/silicate":
        return get_tabulated_eos(pressure, material_properties_iron_silicate_planets, material, interpolation_functions)
    elif eos_choice == "Tabulated:water":
        return get_tabulated_eos(pressure, material_properties_water_planets, material, interpolation_functions)
    else:
        raise ValueError("Invalid EOS choice.")




