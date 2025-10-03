"""
EOS Data and Functions
"""
from __future__ import annotations

import logging
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

logger = logging.getLogger(__name__)

def get_tabulated_eos(pressure, material_dictionary, material, temperature=None, interpolation_functions={}):
    """
    Retrieves density from tabulated EOS data for a given material and choice of EOS.
    Inputs:
        - pressure: Pressure at which to evaluate the EOS (in Pa)
        - material_dictionary: Dictionary containing material properties and EOS file paths
        - material: Material type (e.g., "core", "mantle", "melted_mantle", "water_ice_layer")
        - temperature: Temperature at which to evaluate the EOS (in K), if applicable
        - interpolation_functions: Cache for interpolation functions to avoid redundant loading
    Returns:
        - density: Density corresponding to the given pressure (and temperature if applicable) in kg/m^3
    """
    props = material_dictionary[material]
    eos_file = props["eos_file"]
    try:
        if eos_file not in interpolation_functions:
            if material == "melted_mantle" or material == "solid_mantle":
                # Load P-T–ρ file
                data = np.loadtxt(eos_file, delimiter="\t", skiprows=1)
                pressures = data[:, 0] # in Pa
                temps = data[:, 1] # in K
                densities = data[:, 2] # in kg/m^3
                unique_pressures = np.unique(pressures)
                unique_temps = np.unique(temps)

                # Check if pressures and temps are sorted as expected
                if not (np.all(np.diff(unique_pressures) > 0) and np.all(np.diff(unique_temps) > 0)):
                    raise ValueError("Pressures or temperatures are not sorted as expected in EOS file.")

                # Reshape densities to a 2D grid for interpolation
                density_grid = densities.reshape(len(unique_pressures), len(unique_temps))

                # Create a RegularGridInterpolator for ρ(P,T)
                interpolation_functions[eos_file] = RegularGridInterpolator((unique_pressures, unique_temps), density_grid, bounds_error=False, fill_value=None)
            else:
                # Load ρ-P file
                data = np.loadtxt(eos_file, delimiter=',', skiprows=1)
                pressure_data = data[:, 1] * 1e9 # Convert from GPa to Pa
                density_data = data[:, 0] * 1e3 # Convert from g/cm^3 to kg/m^3
                interpolation_functions[eos_file] = interp1d(pressure_data, density_data, bounds_error=False, fill_value="extrapolate")

        interpolator = interpolation_functions[eos_file]  # Retrieve from cache

        # Perform interpolation
        if material == "melted_mantle":
            if temperature is None:
                raise ValueError("Temperature must be provided for melted_mantle.")
            density = interpolator((pressure, temperature))
        else:
            density = interpolator(pressure)

        if density is None or np.isnan(density):
            raise ValueError(f"Density calculation failed for {material} at P={pressure:.2e} Pa, T={temperature}.")

        return density

    except (ValueError, OSError) as e:
        logger.error(f"Error with tabulated EOS for {material} at P={pressure:.2e} Pa, T={temperature}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error with tabulated EOS for {material} at P={pressure:.2e} Pa, T={temperature}: {e}")
        return None
    
def load_melting_curves(melt_file):
    """
    Loads melting curve data for MgSiO3 from a text file.
    Returns:
        - melting_curve_func: Interpolation function for melting temperature as a function of pressure.
    """
    try:
        data = np.loadtxt(melt_file, comments="#")
        pressures = data[:, 0]  # in Pa
        temperatures = data[:, 1]  # in K
        return pressures, temperatures
    except Exception as e:
        print(f"Error loading melting curve data: {e}")
        return None, None

def calculate_density(pressure, material_dictionaries, material, eos_choice, temperature, interpolation_functions={}):
    """Calculates density with caching for tabulated EOS.
    Inputs:
        - pressure: Pressure at which to evaluate the EOS (in Pa)
        - material_dictionaries: Tuple of material property dictionaries
        - material: Material type (e.g., "core", "mantle", "melted_mantle", "water_ice_layer")
        - eos_choice: Choice of EOS (e.g., "Tabulated:iron/silicate", "Tabulated:iron/Tdep_silicate", "Tabulated:water")
        - temperature: Temperature at which to evaluate the EOS (in K), required for melted_mantle if applicable
        - interpolation_functions: Cache for interpolation functions to avoid redundant loading
    Returns:
        - density: Density corresponding to the given pressure (and temperature if applicable) in kg/m^3
    """

    # Unpack material dictionaries
    material_properties_iron_silicate_planets, material_properties_iron_Tdep_silicate_planets, material_properties_water_planets = material_dictionaries

    if eos_choice == "Tabulated:iron/silicate":
        return get_tabulated_eos(pressure, material_properties_iron_silicate_planets, material, interpolation_functions)
    elif eos_choice == "Tabulated:iron/Tdep_silicate":
        return get_tabulated_eos(pressure, material_properties_iron_Tdep_silicate_planets, material, temperature, interpolation_functions)
    elif eos_choice == "Tabulated:water":
        return get_tabulated_eos(pressure, material_properties_water_planets, material, interpolation_functions)
    else:
        raise ValueError("Invalid EOS choice.")

def calculate_temperature_profile_function(radii, mode, surface_temperature, center_temperature, temp_profile_file=None):
    """
    Returns a callable temperature function for a planetary interior model.

    Inputs:
    radii : array_like
        Radial grid of the planet [m].
    mode : str
        Temperature profile mode. Options:
        - "isothermal": constant temperature equal to surface_temperature
        - "linear": linear profile from center_temperature (r=0) to surface_temperature (r=R)
        - "prescribed": read temperature profile from a text file
    surface_temperature : float
        Temperature at the surface [K] (used for "linear" and "isothermal")
    center_temperature : float
        Temperature at the center [K] (used for "linear")
    temp_profile_file : str, optional
        Name of the file containing the prescribed temperature profile. Must have same length as `radii` if mode="prescribed".

    Returns:
    temperature_func : callable
        Function of radius or array of radii: temperature_func(r) -> float or np.ndarray
    """
    radii = np.array(radii)

    if mode == "isothermal":
        return lambda r: np.full_like(r, surface_temperature, dtype=float)

    elif mode == "linear":
        return lambda r: surface_temperature + (center_temperature - surface_temperature) * (1 - np.array(r)/radii[-1])

    elif mode == "prescribed":
        temp_profile_path = os.path.join(ZALMOXIS_ROOT, "input", temp_profile_file)
        if temp_profile_path is None or not os.path.exists(os.path.join(ZALMOXIS_ROOT, "input", temp_profile_file)):
            raise ValueError("Temperature profile file must be provided and exist for 'prescribed' mode.")
        temp_profile = np.loadtxt(temp_profile_path)
        if len(temp_profile) != len(radii):
            raise ValueError("Temperature profile length does not match radii length.")
        # Vectorized interpolation for arbitrary radius points
        return lambda r: np.interp(np.array(r), radii, temp_profile)

    else:
        raise ValueError(f"Unknown mode '{mode}'. Valid options: 'isothermal', 'linear', 'prescribed'.")


