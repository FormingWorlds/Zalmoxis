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
    Parameters:
        pressure: Pressure at which to evaluate the EOS (in Pa)
        material_dictionary: Dictionary containing material properties and EOS file paths
        material: Material type (e.g., "core", "mantle", "water_ice_layer", "melted_mantle", "solid_mantle")
        temperature: Temperature at which to evaluate the EOS (in K), if applicable
        interpolation_functions: Cache for interpolation functions to avoid redundant loading
    Returns:
        density: Density corresponding to the given pressure (and temperature if applicable) in kg/m^3
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
        if material == "melted_mantle" or material == "solid_mantle":
            if temperature is None:
                raise ValueError("Temperature must be provided.")
            if temperature < np.min(interpolator.grid[1]) or temperature > np.max(interpolator.grid[1]):
                raise ValueError(f"Temperature {temperature:.2f} K is out of bounds for EOS data.")
            if pressure < np.min(interpolator.grid[0]) or pressure > np.max(interpolator.grid[0]):
                raise ValueError(f"Pressure {pressure:.2e} Pa is out of bounds for the EOS data.")
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

def load_melting_curve(melt_file):
    """
    Loads melting curve data for MgSiO3 from a text file.
    Parameters:
        melt_file: Path to the melting curve data file
    Returns:
        interp_func: Interpolation function for T(P)
    """
    try:
        data = np.loadtxt(melt_file, comments="#")
        pressures = data[:, 0]  # in Pa
        temperatures = data[:, 1]  # in K
        interp_func = interp1d(pressures, temperatures, kind="linear", bounds_error=False, fill_value=np.nan)
        return interp_func
    except Exception as e:
        print(f"Error loading melting curve data: {e}")
        return None

def get_solidus_liquidus_functions():
    """
    Loads and returns the solidus and liquidus melting curves for temperature-dependent silicate mantle EOS.
    Returns: A tuple containing the solidus and liquidus functions.
    """
    solidus_func = load_melting_curve(os.path.join(ZALMOXIS_ROOT, "data", "melting_curves_Monteux-600", "solidus.dat"))
    liquidus_func = load_melting_curve(os.path.join(ZALMOXIS_ROOT, "data", "melting_curves_Monteux-600", "liquidus.dat"))

    return solidus_func, liquidus_func

def get_Tdep_density(pressure, temperature, material_properties_iron_Tdep_silicate_planets, solidus_func, liquidus_func, interpolation_functions={}):
    """
    Returns density for mantle material, considering temperature-dependent phase changes.
    Parameters:
        pressure: Pressure at which to evaluate the EOS (in Pa)
        temperature: Temperature at which to evaluate the EOS (in K)
        material_properties_iron_Tdep_silicate_planets: Dictionary containing temperature-dependent material properties for temperature-dependent MgSiO3 EOS
        solidus_func: Interpolation function for the solidus melting curve
        liquidus_func: Interpolation function for the liquidus melting curve
        interpolation_functions: Cache for interpolation functions to avoid redundant loading
    Returns:
        density: Density corresponding to the given pressure and temperature in kg/m^3
    """

    if solidus_func is None or liquidus_func is None:
        raise ValueError("solidus_func and liquidus_func must be provided for Tabulated:iron/Tdep_silicate EOS.")

    T_sol = solidus_func(pressure)
    T_liq = liquidus_func(pressure)

    if temperature <= T_sol:
        # Solid phase
        rho = get_tabulated_eos(pressure, material_properties_iron_Tdep_silicate_planets, "solid_mantle", temperature, interpolation_functions)
        return rho

    elif temperature >= T_liq:
        # Liquid phase
        rho = get_tabulated_eos(pressure, material_properties_iron_Tdep_silicate_planets, "melted_mantle", temperature, interpolation_functions)
        return rho

    else:
        # Mixed phase: linear melt fraction between solidus and liquidus
        frac_melt = (temperature - T_sol) / (T_liq - T_sol)
        rho_solid = get_tabulated_eos(pressure, material_properties_iron_Tdep_silicate_planets, "solid_mantle", temperature, interpolation_functions)
        rho_liquid = get_tabulated_eos(pressure, material_properties_iron_Tdep_silicate_planets, "melted_mantle", temperature, interpolation_functions)
        # Calculate mixed density by volume additivity
        specific_volume_mixed = (frac_melt * (1 / rho_liquid) + (1 - frac_melt) * (1 / rho_solid))
        rho_mixed = 1 / specific_volume_mixed
        return rho_mixed

def get_Tdep_material(pressure, temperature, solidus_func, liquidus_func):
    """
    Returns type for mantle material, considering temperature-dependent phase changes. Supports scalar and array inputs.
    Parameters:
        pressure: Pressure at which to evaluate the EOS (in Pa), can be scalar or array
        temperature: Temperature at which to evaluate the EOS (in K), can be scalar or array
    Returns:
        material: Material type ("solid_mantle", "melted_mantle", or "mixed_mantle")
    """
    # Define per-point evaluation
    def evaluate_phase(P, T):
        T_sol = solidus_func(P)
        T_liq = liquidus_func(P)
        frac_melt = (T - T_sol) / (T_liq - T_sol)
        if frac_melt < 0.05:
            return "solid_mantle"
        elif frac_melt <= 0.95:
            return "mixed_mantle"
        else:
            return "melted_mantle"

    # Vectorize function for array support
    vectorized_eval = np.vectorize(evaluate_phase, otypes=[str])

    # Apply depending on input type
    if np.isscalar(pressure) and np.isscalar(temperature):
        return evaluate_phase(pressure, temperature)
    else:
        return vectorized_eval(pressure, temperature)

def calculate_density(pressure, material_dictionaries, material, eos_choice, temperature, solidus_func, liquidus_func, interpolation_functions={}):
    """Calculates density with caching for tabulated EOS.
    Parameters:
        pressure: Pressure at which to evaluate the EOS (in Pa)
        material_dictionaries: Tuple of material property dictionaries
        material: Material type (e.g., "core", "mantle", "water_ice_layer", "melted_mantle", "solid_mantle" or a combination of "melted_mantle" and "solid_mantle")
        eos_choice: Choice of EOS (e.g., "Tabulated:iron/silicate", "Tabulated:iron/Tdep_silicate", "Tabulated:water")
        temperature: Temperature at which to evaluate the EOS (in K), if applicable
        solidus_func: Interpolation function for the solidus melting curve
        liquidus_func: Interpolation function for the liquidus melting curve
        interpolation_functions: Cache for interpolation functions to avoid redundant loading
    Returns:
        density: Density corresponding to the given pressure (and temperature if applicable) in kg/m^3
    """

    # Unpack material dictionaries
    material_properties_iron_silicate_planets, material_properties_iron_Tdep_silicate_planets, material_properties_water_planets = material_dictionaries

    if eos_choice == "Tabulated:iron/silicate":
        return get_tabulated_eos(pressure, material_properties_iron_silicate_planets, material, interpolation_functions)
    elif eos_choice == "Tabulated:iron/Tdep_silicate":
        return get_Tdep_density(pressure, temperature, material_properties_iron_Tdep_silicate_planets, solidus_func, liquidus_func, interpolation_functions)
    elif eos_choice == "Tabulated:water":
        return get_tabulated_eos(pressure, material_properties_water_planets, material, interpolation_functions)
    else:
        raise ValueError("Invalid EOS choice.")

def calculate_temperature_profile(radii, temperature_mode, surface_temperature, center_temperature, input_dir, temp_profile_file=None):
    """
    Returns a callable temperature function for a planetary interior model.

    Parameters:
    radii: Radial grid of the planet [m].
    temperature_mode: Temperature profile mode. Options:
        - "isothermal": constant temperature equal to surface_temperature
        - "linear": linear profile from center_temperature (r=0) to surface_temperature (r=R)
        - "prescribed": read temperature profile from a text file
    surface_temperature: Temperature at the surface [K] (used for "linear" and "isothermal")
    center_temperature: Temperature at the center [K] (used for "linear")
    input_dir: Directory where the temperature profile file is located.
    temp_profile_file: Name of the file containing the prescribed temperature profile from center to surface. Must have same length as `radii` if temperature_mode="prescribed".

    Returns:
    temperature_func: Function of radius or array of radii for temperature [K]: temperature_func(r) -> float or np.ndarray
    """
    radii = np.array(radii)

    if temperature_mode == "isothermal":
        return lambda r: np.full_like(r, surface_temperature, dtype=float)

    elif temperature_mode == "linear":
        return lambda r: surface_temperature + (center_temperature - surface_temperature) * (1 - np.array(r)/radii[-1])

    elif temperature_mode == "prescribed":
        temp_profile_path = os.path.join(input_dir, temp_profile_file)
        if temp_profile_path is None or not os.path.exists(os.path.join(input_dir, temp_profile_file)):
            raise ValueError("Temperature profile file must be provided and exist for 'prescribed' temperature mode.")
        temp_profile = np.loadtxt(temp_profile_path)
        if len(temp_profile) != len(radii):
            raise ValueError("Temperature profile length does not match radii length.")
        # Vectorized interpolation for arbitrary radius points
        return lambda r: np.interp(np.array(r), radii, temp_profile)

    else:
        raise ValueError(f"Unknown temperature mode '{temperature_mode}'. Valid options: 'isothermal', 'linear', 'prescribed'.")

def create_pressure_density_files(outer_iter, inner_iter, pressure_iter, radii, pressure, density):
    """
    Create and append pressure and density profiles to output files for each pressure iteration.
    Parameters:
        outer_iter (int): Current outer iteration index.
        inner_iter (int): Current inner iteration index.
        pressure_iter (int): Current pressure iteration index.
        radii (np.ndarray): Array of radial positions.
        pressure (np.ndarray): Array of pressure values corresponding to the radii.
        density (np.ndarray): Array of density values corresponding to the radii.
    """

    pressure_file = os.path.join(ZALMOXIS_ROOT, "output_files", "pressure_profiles.txt")
    density_file = os.path.join(ZALMOXIS_ROOT, "output_files", "density_profiles.txt")

    # Only delete the files once at the beginning of the run
    if outer_iter == 0 and inner_iter == 0 and pressure_iter == 0:
        for file_path in [pressure_file, density_file]:
            if os.path.exists(file_path):
                os.remove(file_path)

    # Append current iteration's pressure profile to file
    with open(pressure_file, "a") as f:
        f.write(f"# Pressure iteration {pressure_iter}\n")
        np.savetxt(f, np.column_stack((radii, pressure)), header="radius pressure", comments='')
        f.write("\n")

    # Append current iteration's density profile to file
    with open(density_file, "a") as f:
        f.write(f"# Pressure iteration {pressure_iter}\n")
        np.savetxt(f, np.column_stack((radii, density)), header="radius density", comments='')
        f.write("\n")
