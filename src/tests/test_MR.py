import os
import tempfile
import toml
import pytest
import numpy as np
from src.zalmoxis import zalmoxis
from zalmoxis.constants import earth_mass, earth_radius

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

def run_zalmoxis(id_mass, config_type):
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
        #output_file = os.path.join(ZALMOXIS_ROOT, "src", "zalmoxis", "output_files", "calculated_planet_mass_radius_Earth.txt")
    elif config_type == "water":
        config['AssumptionsAndInitialGuesses']['core_mass_fraction'] = 0.1625
        config['AssumptionsAndInitialGuesses']['inner_mantle_mass_fraction'] = 0.3375
        config['AssumptionsAndInitialGuesses']['weight_iron_fraction'] = 0.1625
        config['EOS']['choice'] = "Tabulated:water"
        #output_file = os.path.join(ZALMOXIS_ROOT, "src", "zalmoxis", "output_files", "calculated_planet_mass_radius_water.txt")
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

    zalmoxis.main(temp_config_path, id_mass, output_file=output_file)

    os.remove(temp_config_path)

    return output_file

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

@pytest.mark.parametrize("config_type,zeng_file", [
    ("rocky", "massradiusEarthlikeRockyZeng.txt"),
    ("water", "massradiuswaterZeng.txt"),
])
@pytest.mark.parametrize("mass", range(1, 51))  # 1 to 50 Earth masses
def test_mass_radius(config_type, zeng_file, mass):
    zeng_masses, zeng_radii = load_zeng_curve(zeng_file)

    output_file = run_zalmoxis(mass, config_type)

    model_mass, model_radius = load_model_output(output_file)

    # Interpolate Zeng radius for the model mass
    zeng_radius_interp = np.interp(model_mass, zeng_masses, zeng_radii)

    assert np.isclose(model_radius, zeng_radius_interp, rtol=0.05), (
        f"{config_type} planet mass {model_mass} Earth masses: "
        f"model radius = {model_radius}, "
        f"Zeng radius = {zeng_radius_interp} (within 5%)"
    )
