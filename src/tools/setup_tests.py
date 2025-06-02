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
    config['PressureAdjustment']['pressure_tolerance'] = 1e11
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

    radii, density, gravity, pressure, temperature, mass_enclosed = zalmoxis.main(temp_config_path, id_mass, output_file=output_file)

    os.remove(temp_config_path)

    # Write profile data (radii and density) to a temporary profile file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix="_profile.txt") as profile_file:
        profile_output_file = profile_file.name
        profile_file.write("radius (m),density (kg/m^3)\n")
        for r, d in zip(radii, density):
            profile_file.write(f"{r},{d}\n")

    return output_file, profile_output_file

def load_zeng_curve(filename):
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
    # Your output file should contain just one line: mass radius
    with open(output_file, 'r') as f:
        next(f)  # Skip the header line
        for line in f:
            if line.strip():
                mass, radius = map(float, line.split())
                return mass/earth_mass, radius/earth_radius
    raise RuntimeError(f"No valid data found in {output_file}")

def load_profile_output(profile_output_file):
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
