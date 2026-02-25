from __future__ import annotations

import os
import tempfile

from src.zalmoxis import zalmoxis
from src.zalmoxis.zalmoxis import load_solidus_liquidus_functions
from zalmoxis.constants import earth_mass, earth_radius

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')


def run_zalmoxis_rocky_water(id_mass, config_type, cmf, immf, layer_eos_override=None):
    """Run the Zalmoxis model for a given planet mass and configuration type.

    Parameters
    ----------
    id_mass : float
        Mass of the planet in Earth masses.
    config_type : str
        Type of planet configuration ('rocky' or 'water').
    cmf : float
        Core mass fraction.
    immf : float
        Inner mantle mass fraction.
    layer_eos_override : dict or None
        If set, overrides the per-layer EOS config for this run.
        E.g. {"core": "Analytic:iron", "mantle": "Analytic:MgSiO3"}.

    Returns
    -------
    tuple
        (output_file, profile_output_file) paths.
    """
    # Load default configuration
    default_config_path = os.path.join(ZALMOXIS_ROOT, 'input', 'default.toml')
    config_params = zalmoxis.load_zalmoxis_config(default_config_path)

    # Modify the configuration parameters as needed
    config_params['planet_mass'] = id_mass * earth_mass

    if config_type == 'rocky':
        config_params['core_mass_fraction'] = 0.325
        config_params['mantle_mass_fraction'] = 0
        config_params['layer_eos_config'] = {
            'core': 'Seager2007:iron',
            'mantle': 'Seager2007:MgSiO3',
        }
    elif config_type == 'water':
        config_params['core_mass_fraction'] = cmf
        config_params['mantle_mass_fraction'] = immf
        config_params['layer_eos_config'] = {
            'core': 'Seager2007:iron',
            'mantle': 'Seager2007:MgSiO3',
            'ice_layer': 'Seager2007:H2O',
        }
    else:
        raise ValueError(f'Unknown config_type: {config_type}')

    # Apply layer EOS override if specified
    if layer_eos_override:
        config_params['layer_eos_config'] = layer_eos_override

    # Create a temporary output file
    suffix = '_rocky.txt' if config_type == 'rocky' else '_water.txt'
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix=suffix) as temp_output_file:
        output_file = temp_output_file.name

    # Delete existing output file to start fresh
    if os.path.exists(output_file):
        os.remove(output_file)

    layer_eos_config = config_params['layer_eos_config']

    # Run the main function and post-processing
    model_results = zalmoxis.main(
        config_params,
        material_dictionaries=zalmoxis.load_material_dictionaries(),
        melting_curves_functions=load_solidus_liquidus_functions(layer_eos_config),
        input_dir=os.path.join(ZALMOXIS_ROOT, 'input'),
    )
    zalmoxis.post_processing(config_params, id_mass, output_file=output_file)

    # Write profile data (radii and density) to a temporary profile file
    with tempfile.NamedTemporaryFile(
        delete=False, mode='w', suffix='_profile.txt'
    ) as profile_file:
        profile_output_file = profile_file.name
        profile_file.write('radius (m),density (kg/m^3)\n')
        for r, d in zip(model_results['radii'], model_results['density']):
            profile_file.write(f'{r},{d}\n')

    return output_file, profile_output_file


def run_zalmoxis_TdepEOS(id_mass):
    """Run Zalmoxis with the default Tdep EOS config for a given mass.

    Parameters
    ----------
    id_mass : float
        Mass of the planet in Earth masses.

    Returns
    -------
    list
        [(id_mass, converged, model_results)] status tuple with full model results.
    """
    # Load default configuration
    default_config_path = os.path.join(ZALMOXIS_ROOT, 'input', 'default.toml')
    config_params = zalmoxis.load_zalmoxis_config(default_config_path)

    # Modify the configuration parameters as needed
    config_params['planet_mass'] = id_mass * earth_mass

    # Create a temporary output file
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_output_file:
        output_file = temp_output_file.name

    # Delete existing output file to start fresh
    if os.path.exists(output_file):
        os.remove(output_file)

    layer_eos_config = config_params['layer_eos_config']

    # Unpack outputs directly from Zalmoxis
    model_results = zalmoxis.main(
        config_params,
        material_dictionaries=zalmoxis.load_material_dictionaries(),
        melting_curves_functions=load_solidus_liquidus_functions(layer_eos_config),
        input_dir=os.path.join(ZALMOXIS_ROOT, 'input'),
    )
    converged = model_results.get('converged', False)

    # Check if model converged before proceeding
    if not converged:
        print(f'Model did not converge for mass {id_mass} Earth masses.')
        return [(id_mass, False, model_results)]

    # Extract the results from the model output
    radii = model_results['radii']
    total_time = model_results['total_time']
    planet_radius = radii[-1]

    # Log the mass and radius only if converged
    custom_log_file = os.path.join(
        ZALMOXIS_ROOT, 'output_files', 'composition_TdepEOS_mass_log.txt'
    )
    with open(custom_log_file, 'a') as log:
        log.write(f'{id_mass:.4f}\t{planet_radius:.4e}\t{total_time:.4e}\n')
    return [(id_mass, converged, model_results)]


def load_zeng_curve(filename):
    """Load Zeng et al. (2019) mass-radius data from a specified file.

    Parameters
    ----------
    filename : str
        Name of the file containing Zeng et al. (2019) mass-radius data.

    Returns
    -------
    tuple
        (masses, radii) lists in Earth units.
    """
    data_path = os.path.join(ZALMOXIS_ROOT, 'data', 'mass_radius_curves', filename)

    masses = []
    radii = []
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip() == '' or line.startswith('#'):
                continue
            mass, radius = map(float, line.split())
            masses.append(mass)
            radii.append(radius)
    return masses, radii


def load_model_output(output_file):
    """Load mass and radius from the Zalmoxis model output file.

    Parameters
    ----------
    output_file : str
        Path to the output file.

    Returns
    -------
    tuple
        (mass_earth_masses, radius_earth_radii).
    """
    with open(output_file, 'r') as f:
        next(f)  # Skip the header line
        for line in f:
            if line.strip():
                mass, radius = map(float, line.split())
                return mass / earth_mass, radius / earth_radius
    raise RuntimeError(f'No valid data found in {output_file}')


def load_profile_output(profile_output_file):
    """Load density profile from the Zalmoxis model output file.

    Parameters
    ----------
    profile_output_file : str
        Path to the profile output file.

    Returns
    -------
    tuple
        (radii_m, densities_kgm3) lists.
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
        raise RuntimeError(f'No valid data found in {profile_output_file}')

    return radii, densities


def load_Seager_data(filename):
    """Load Seager et al. (2007) radius and density data from a specified file.

    Parameters
    ----------
    filename : str
        Name of the file containing Seager et al. (2007) data.

    Returns
    -------
    dict
        Dictionary keyed by planet mass, with 'radius' and 'density' lists.
    """
    data_path = os.path.join(ZALMOXIS_ROOT, 'data', 'radial_profiles', filename)

    data_by_mass = {}
    with open(data_path, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue
            parts = line.strip().split(',')
            if len(parts) != 3:
                continue
            mass, radius, density = map(float, parts)
            if mass not in data_by_mass:
                data_by_mass[mass] = {'radius': [], 'density': []}
            data_by_mass[mass]['radius'].append(radius * 1000)  # km
            data_by_mass[mass]['density'].append(density * 1000)  # kg/m^3
    return data_by_mass
