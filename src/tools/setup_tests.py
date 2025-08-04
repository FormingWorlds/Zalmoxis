from __future__ import annotations

import os
import tempfile

import toml

from src.zalmoxis import zalmoxis
from zalmoxis.constants import earth_mass, earth_radius

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

def run_zalmoxis_rocky_water(id_mass, config_type, cmf, immf):
    """
    Runs the Zalmoxis model for a given planet mass and configuration type (rocky or water).
    Parameters:
        id_mass (float): Mass of the planet in Earth masses.
        config_type (str): Type of planet configuration ('rocky' or 'water').
        cmf (float): Core mass fraction for water planets.
        immf (float): Inner mantle mass fraction for water planets.
    Returns:
        output_file (str): Path to the output file containing mass and radius.
        profile_output_file (str): Path to the output file containing the density profile.
    Raises:
        ValueError: If an unknown config_type is provided.
    """
    # Load default config and adjust parameters per config_type and mass
    default_config_path = os.path.join(ZALMOXIS_ROOT, "input", "default.toml")
    with open(default_config_path, 'r') as file:
        config = toml.load(file)

    config['InputParameter']['planet_mass'] = id_mass * 5.972e24  # Earth mass in kg
    config['Calculations']['num_layers'] = 150
    config['IterativeProcess']['max_iterations_outer'] = 20
    config['IterativeProcess']['tolerance_outer'] = 1e-3
    config['IterativeProcess']['tolerance_inner'] = 1e-4
    config['IterativeProcess']['relative_tolerance'] = 1e-5
    config['IterativeProcess']['absolute_tolerance'] = 1e-6
    config['PressureAdjustment']['target_surface_pressure'] = 101325
    config['PressureAdjustment']['pressure_tolerance'] = 1e9
    config['PressureAdjustment']['max_iterations_pressure'] = 200
    config['PressureAdjustment']['pressure_adjustment_factor'] = 1.1

    if config_type == "rocky":
        config['AssumptionsAndInitialGuesses']['core_mass_fraction'] = 0.325
        config['AssumptionsAndInitialGuesses']['inner_mantle_mass_fraction'] = 0
        config['AssumptionsAndInitialGuesses']['weight_iron_fraction'] = 0.325
        config['EOS']['choice'] = "Tabulated:iron/silicate"
    elif config_type == "water":
        config['AssumptionsAndInitialGuesses']['core_mass_fraction'] = cmf
        config['AssumptionsAndInitialGuesses']['inner_mantle_mass_fraction'] = immf
        config['AssumptionsAndInitialGuesses']['weight_iron_fraction'] = cmf
        config['EOS']['choice'] = "Tabulated:water"
    else:
        raise ValueError(f"Unknown config_type: {config_type}")

    suffix = "_rocky.txt" if config_type == "rocky" else "_water.txt"
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=suffix) as temp_output_file:
        output_file = temp_output_file.name

    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.toml') as temp_config_file:
        toml.dump(config, temp_config_file)
        temp_config_path = temp_config_file.name

    # Delete existing output file to start fresh
    if os.path.exists(output_file):
        os.remove(output_file)

    radii, density, gravity, pressure, temperature, mass_enclosed, total_time = zalmoxis.main(temp_config_path, id_mass, output_file=output_file)

    os.remove(temp_config_path)

    # Write profile data (radii and density) to a temporary profile file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix="_profile.txt") as profile_file:
        profile_output_file = profile_file.name
        profile_file.write("radius (m),density (kg/m^3)\n")
        for r, d in zip(radii, density):
            profile_file.write(f"{r},{d}\n")

    return output_file, profile_output_file

def load_zeng_curve(filename):
    """
    Load Zeng et al. (2019) mass-radius data from a specified file.
    Parameters:
        filename (str): Name of the file containing Zeng et al. (2019) mass-radius data.
    Returns:
        masses (list): List of planet masses in Earth masses.
        radii (list): List of planet radii in Earth radii.
    Raises:
        FileNotFoundError: If the specified file does not exist.
    """
    # Load Zeng et al. (2019) mass-radius data from the specified file
    data_path = os.path.join(ZALMOXIS_ROOT, "data", filename)

    masses = []
    radii = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip() == "" or line.startswith("#"):
                continue
            mass, radius = map(float, line.split())
            masses.append(mass)
            radii.append(radius)
    return masses, radii

def load_model_output(output_file):
    """
    Load the mass and radius from the Zalmoxis model output file.
    Parameters:
        output_file (str): Path to the output file containing mass and radius.
    Returns:
        tuple: A tuple containing the mass in Earth masses and radius in Earth radii.
    Raises:
        RuntimeError: If the output file does not contain valid data.
    """
    # Your output file should contain just one line: mass radius
    with open(output_file, 'r') as f:
        next(f)  # Skip the header line
        for line in f:
            if line.strip():
                mass, radius = map(float, line.split())
                return mass/earth_mass, radius/earth_radius
    raise RuntimeError(f"No valid data found in {output_file}")

def load_profile_output(profile_output_file):
    """
    Load the density profile from the Zalmoxis model output file.
    Parameters:
        profile_output_file (str): Path to the output file containing the density profile.
    Returns:
        tuple: A tuple containing two lists: radii in meters and densities in kg/m^3.
    Raises:
        RuntimeError: If the profile output file does not contain valid data.
    """
    radii = []
    densities = []

    with open(profile_output_file, 'r') as f:
        next(f)  # Skip header line
        for line in f:
            if line.strip():
                try:
                    radius_str, density_str = line.strip().split(',')
                    radius = float(radius_str)
                    density = float(density_str)
                    radii.append(radius)
                    densities.append(density)
                except ValueError as e:
                    raise RuntimeError(f"Failed to parse line: '{line.strip()}'. Error: {e}")

    if not radii or not densities:
        raise RuntimeError(f"No valid data found in {profile_output_file}")

    return radii, densities

def load_Seager_data(filename):
    """
    Load Seager et al. (2007) radius and density data from a specified file.
    Parameters:
        filename (str): Name of the file containing Seager et al. (2007) radius and density data.
    Returns:
        data_by_mass (dict): A dictionary where keys are planet masses and values are lists of radii and densities.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file format is incorrect or malformed.
    """
    # Load Seager et al. (2007) profiles: mass, radius, density per line (comma-separated)
    data_path = os.path.join(ZALMOXIS_ROOT, "data", filename)

    data_by_mass = {}
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip() == "":
                continue
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue  # skip malformed lines
            mass, radius, density = map(float, parts)
            if mass not in data_by_mass:
                data_by_mass[mass] = {
                    "radius": [],
                    "density": []
                }
            data_by_mass[mass]["radius"].append(radius * 1000)      # km
            data_by_mass[mass]["density"].append(density * 1000)    # kg/m^3
    return data_by_mass
