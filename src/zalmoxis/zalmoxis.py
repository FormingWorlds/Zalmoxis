from __future__ import annotations

import logging
import math
import os
import sys
import time

import numpy as np
import toml
from scipy.optimize import brentq

from .constants import TDEP_EOS_NAMES, earth_center_pressure, earth_mass, earth_radius
from .eos_analytic import VALID_MATERIAL_KEYS
from .eos_functions import (
    calculate_density,
    calculate_temperature_profile,
    compute_adiabatic_temperature,
    create_pressure_density_files,
    get_solidus_liquidus_functions,
    get_Tdep_material,
)
from .eos_properties import (
    material_properties_iron_RTPress100TPa_silicate_planets,
    material_properties_iron_silicate_planets,
    material_properties_iron_Tdep_silicate_planets,
    material_properties_water_planets,
)
from .plots.plot_phase_vs_radius import plot_PT_with_phases
from .plots.plot_profiles import plot_planet_profile_single
from .structure_model import get_layer_eos, solve_structure

# Run file via command line with default configuration file: python -m zalmoxis -c input/default.toml

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')

logger = logging.getLogger(__name__)

# Mapping from legacy global EOS choice strings to per-layer config dicts
LEGACY_EOS_MAP = {
    'Tabulated:iron/silicate': {
        'core': 'Seager2007:iron',
        'mantle': 'Seager2007:MgSiO3',
    },
    'Tabulated:iron/Tdep_silicate': {
        'core': 'Seager2007:iron',
        'mantle': 'WolfBower2018:MgSiO3',
    },
    'Tabulated:water': {
        'core': 'Seager2007:iron',
        'mantle': 'Seager2007:MgSiO3',
        'ice_layer': 'Seager2007:H2O',
    },
}

VALID_TABULATED_EOS = {
    'Seager2007:iron',
    'Seager2007:MgSiO3',
    'WolfBower2018:MgSiO3',
    'RTPress100TPa:MgSiO3',
    'Seager2007:H2O',
}

# WolfBower2018 tables are valid up to ~1 TPa. The Brent pressure solver with
# out-of-bounds clamping handles mantle pressures exceeding the table boundary
# for planets up to ~7 M_earth. Beyond this mass, deep-mantle pressures are far
# enough above the table ceiling that clamped densities become unreliable.
WOLFBOWER2018_MAX_MASS_EARTH = 7.0

# RTPress100TPa melt table extends to 100 TPa, matching the Seager2007 iron
# range. The solid table is still WolfBower2018 (1 TPa, clamped), but at high
# temperatures the mantle is predominantly molten, so the solid table limitation
# is less constraining. Safe up to ~50 M_earth.
RTPRESS100TPA_MAX_MASS_EARTH = 50.0


def parse_eos_config(eos_section):
    """Parse [EOS] TOML section into per-layer EOS dict.

    Supports new per-layer format (core/mantle/ice_layer fields) and legacy
    format (choice field) for backward compatibility.

    Parameters
    ----------
    eos_section : dict
        The [EOS] section from the TOML config.

    Returns
    -------
    dict
        Per-layer EOS config, e.g.
        {"core": "Seager2007:iron", "mantle": "WolfBower2018:MgSiO3"}.
    """
    # New format: per-layer fields present
    if 'core' in eos_section:
        if 'mantle' not in eos_section:
            raise ValueError(
                "EOS config has 'core' but missing 'mantle'. "
                "Both 'core' and 'mantle' are required."
            )
        layer_eos = {
            'core': eos_section['core'],
            'mantle': eos_section['mantle'],
        }
        water = eos_section.get('ice_layer', '')
        if water:
            layer_eos['ice_layer'] = water
        return layer_eos

    # Legacy format: expand 'choice' field
    choice = eos_section.get('choice', '')
    if choice in LEGACY_EOS_MAP:
        return dict(LEGACY_EOS_MAP[choice])

    if choice == 'Analytic:Seager2007':
        layer_eos = {
            'core': f'Analytic:{eos_section.get("core_material", "iron")}',
            'mantle': f'Analytic:{eos_section.get("mantle_material", "MgSiO3")}',
        }
        water_mat = eos_section.get('water_layer_material', '')
        if water_mat:
            layer_eos['ice_layer'] = f'Analytic:{water_mat}'
        return layer_eos

    raise ValueError(
        f"Unknown EOS config. Set per-layer fields (core, mantle) or legacy 'choice'. "
        f'Got: {eos_section}'
    )


def validate_layer_eos(layer_eos_config):
    """Validate all per-layer EOS strings.

    Parameters
    ----------
    layer_eos_config : dict
        Per-layer EOS config from parse_eos_config().

    Raises
    ------
    ValueError
        If any layer EOS string is invalid.
    """
    for layer, eos in layer_eos_config.items():
        if eos in VALID_TABULATED_EOS:
            continue
        if eos.startswith('Analytic:'):
            material_key = eos.split(':', 1)[1]
            if material_key not in VALID_MATERIAL_KEYS:
                raise ValueError(
                    f"Invalid analytic material '{material_key}' for layer '{layer}'. "
                    f'Valid keys: {sorted(VALID_MATERIAL_KEYS)}'
                )
            continue
        raise ValueError(
            f"Invalid EOS '{eos}' for layer '{layer}'. "
            f'Valid tabulated: {sorted(VALID_TABULATED_EOS)}. '
            f'Valid analytic: Analytic:<material> with {sorted(VALID_MATERIAL_KEYS)}.'
        )


def choose_config_file(temp_config_path=None):
    """
    Function to choose the configuration file to run the main function.
    The function will first check if a temporary configuration file is provided.
    If not, it will check if the -c flag is provided in the command line arguments.
    If the -c flag is provided, the function will read the configuration file path from the next argument.
    If no temporary configuration file or -c flag is provided, the function will read the default configuration file.
    """

    # Load the configuration file either from terminal (-c flag) or default path
    if temp_config_path:
        try:
            config = toml.load(temp_config_path)
            logger.info(f'Reading temporary config file from: {temp_config_path}')
        except FileNotFoundError:
            logger.error(f'Error: Temporary config file not found at {temp_config_path}')
            sys.exit(1)
    elif '-c' in sys.argv:
        index = sys.argv.index('-c')
        try:
            config_file_path = sys.argv[index + 1]
            config = toml.load(config_file_path)
            logger.info(f'Reading config file from: {config_file_path}')
        except IndexError:
            logger.error('Error: -c flag provided but no config file path specified.')
            sys.exit(1)  # Exit with error code
        except FileNotFoundError:
            logger.error(f'Error: Config file not found at {config_file_path}')
            sys.exit(1)
    else:
        config_default_path = os.path.join(ZALMOXIS_ROOT, 'input', 'default.toml')
        try:
            config = toml.load(config_default_path)
            logger.info(f'Reading default config file from {config_default_path}')
        except FileNotFoundError:
            logger.info(f'Error: Default config file not found at {config_default_path}')
            sys.exit(1)

    return config


def load_zalmoxis_config(temp_config_path=None):
    """Load and return configuration parameters for the Zalmoxis model.

    Returns
    -------
    dict
        All relevant configuration parameters.
    """
    config = choose_config_file(temp_config_path)

    # Parse per-layer EOS config (supports both new and legacy formats)
    layer_eos_config = parse_eos_config(config['EOS'])
    validate_layer_eos(layer_eos_config)

    return {
        'planet_mass': config['InputParameter']['planet_mass'] * earth_mass,
        'core_mass_fraction': config['AssumptionsAndInitialGuesses']['core_mass_fraction'],
        'mantle_mass_fraction': config['AssumptionsAndInitialGuesses']['mantle_mass_fraction'],
        'temperature_mode': config['AssumptionsAndInitialGuesses']['temperature_mode'],
        'surface_temperature': config['AssumptionsAndInitialGuesses']['surface_temperature'],
        'center_temperature': config['AssumptionsAndInitialGuesses']['center_temperature'],
        'temp_profile_file': config['AssumptionsAndInitialGuesses']['temperature_profile_file'],
        'layer_eos_config': layer_eos_config,
        'num_layers': config['Calculations']['num_layers'],
        'max_iterations_outer': config['IterativeProcess']['max_iterations_outer'],
        'tolerance_outer': config['IterativeProcess']['tolerance_outer'],
        'max_iterations_inner': config['IterativeProcess']['max_iterations_inner'],
        'tolerance_inner': config['IterativeProcess']['tolerance_inner'],
        'relative_tolerance': config['IterativeProcess']['relative_tolerance'],
        'absolute_tolerance': config['IterativeProcess']['absolute_tolerance'],
        'maximum_step': config['IterativeProcess']['maximum_step'],
        'adaptive_radial_fraction': config['IterativeProcess']['adaptive_radial_fraction'],
        'max_center_pressure_guess': config['IterativeProcess']['max_center_pressure_guess'],
        'target_surface_pressure': config['PressureAdjustment']['target_surface_pressure'],
        'pressure_tolerance': config['PressureAdjustment']['pressure_tolerance'],
        'max_iterations_pressure': config['PressureAdjustment']['max_iterations_pressure'],
        'data_output_enabled': config['Output']['data_enabled'],
        'plotting_enabled': config['Output']['plots_enabled'],
        'verbose': config['Output']['verbose'],
        'iteration_profiles_enabled': config['Output']['iteration_profiles_enabled'],
    }


def load_material_dictionaries():
    """Load and return the material properties dictionaries.

    Returns
    -------
    tuple
        (iron_silicate, iron_Tdep_silicate, water, iron_RTPress100TPa_silicate)
        material property dicts.
    """
    material_dictionaries = (
        material_properties_iron_silicate_planets,
        material_properties_iron_Tdep_silicate_planets,
        material_properties_water_planets,
        material_properties_iron_RTPress100TPa_silicate_planets,
    )
    return material_dictionaries


def load_solidus_liquidus_functions(layer_eos_config):
    """Load solidus and liquidus functions if any layer uses a T-dependent EOS.

    Parameters
    ----------
    layer_eos_config : dict
        Per-layer EOS config.

    Returns
    -------
    tuple or None
        (solidus_func, liquidus_func) if Tdep EOS is used, else None.
    """
    uses_Tdep = any(v in TDEP_EOS_NAMES for v in layer_eos_config.values() if v)
    if uses_Tdep:
        solidus_func, liquidus_func = get_solidus_liquidus_functions()
        return (solidus_func, liquidus_func)
    return None


def main(config_params, material_dictionaries, melting_curves_functions, input_dir):
    """Run the exoplanet internal structure model.

    Iteratively adjusts the internal structure based on configuration parameters,
    calculating the planet's radius, core-mantle boundary, densities, pressures,
    and other properties.

    Parameters
    ----------
    config_params : dict
        Configuration parameters for the model.
    material_dictionaries : tuple
        Material properties dictionaries.
    melting_curves_functions : tuple or None
        (solidus_func, liquidus_func) for Tdep EOS, or None.
    input_dir : str
        Directory containing input files.

    Returns
    -------
    dict
        Model results including radii, density, gravity, pressure, temperature,
        mass enclosed, convergence status, and timing.
    """
    # Initialize convergence flags
    converged = False
    converged_pressure = False
    converged_density = False
    converged_mass = False

    # Unpack configuration parameters
    planet_mass = config_params['planet_mass']
    core_mass_fraction = config_params['core_mass_fraction']
    mantle_mass_fraction = config_params['mantle_mass_fraction']
    temperature_mode = config_params['temperature_mode']
    surface_temperature = config_params['surface_temperature']
    center_temperature = config_params['center_temperature']
    temp_profile_file = config_params['temp_profile_file']
    layer_eos_config = config_params['layer_eos_config']
    num_layers = config_params['num_layers']
    max_iterations_outer = config_params['max_iterations_outer']
    tolerance_outer = config_params['tolerance_outer']
    max_iterations_inner = config_params['max_iterations_inner']
    tolerance_inner = config_params['tolerance_inner']
    relative_tolerance = config_params['relative_tolerance']
    absolute_tolerance = config_params['absolute_tolerance']
    maximum_step = config_params['maximum_step']
    adaptive_radial_fraction = config_params['adaptive_radial_fraction']
    max_center_pressure_guess = config_params['max_center_pressure_guess']
    target_surface_pressure = config_params['target_surface_pressure']
    pressure_tolerance = config_params['pressure_tolerance']
    max_iterations_pressure = config_params['max_iterations_pressure']
    verbose = config_params['verbose']
    iteration_profiles_enabled = config_params['iteration_profiles_enabled']

    # Check if any layer uses T-dependent EOS
    uses_Tdep = any(v in TDEP_EOS_NAMES for v in layer_eos_config.values() if v)

    # Enforce per-EOS mass limits for T-dependent tables
    mass_in_earth = planet_mass / earth_mass
    if uses_Tdep:
        tdep_eos_used = {v for v in layer_eos_config.values() if v in TDEP_EOS_NAMES}
        for eos_name in tdep_eos_used:
            if eos_name == 'WolfBower2018:MgSiO3':
                max_mass = WOLFBOWER2018_MAX_MASS_EARTH
                reason = (
                    'Deep-mantle pressures far exceed the 1 TPa table boundary '
                    'at higher masses, making clamped densities unreliable. '
                    'Use RTPress100TPa:MgSiO3 for higher masses.'
                )
            elif eos_name == 'RTPress100TPa:MgSiO3':
                max_mass = RTPRESS100TPA_MAX_MASS_EARTH
                reason = (
                    'The RTPress100TPa melt table extends to 100 TPa but '
                    'the solid table is limited to 1 TPa.'
                )
            else:
                continue
            if mass_in_earth > max_mass:
                raise ValueError(
                    f'{eos_name} EOS is limited to planets <= '
                    f'{max_mass} M_earth (requested {mass_in_earth:.2f} M_earth). '
                    f'{reason}'
                )

    # Setup initial guesses
    radius_guess = (
        1000 * (7030 - 1840 * core_mass_fraction) * (planet_mass / earth_mass) ** 0.282
    )
    cmb_mass = 0
    core_mantle_mass = 0

    logger.info(
        f'Starting structure model for a {planet_mass / earth_mass} Earth masses planet '
        f"with EOS config {layer_eos_config} and temperature mode '{temperature_mode}'."
    )

    start_time = time.time()

    # Initialize empty cache for interpolation functions
    interpolation_cache = {}

    # Load solidus and liquidus functions if using Tdep EOS
    if uses_Tdep:
        solidus_func, liquidus_func = melting_curves_functions
    else:
        solidus_func, liquidus_func = None, None

    # Storage for the previous iteration's converged profiles.
    # Used by adiabatic mode to compute T(r) from the last P(r), g(r).
    prev_radii = None
    prev_pressure = None
    prev_mass_enclosed = None

    # Solve the interior structure
    for outer_iter in range(max_iterations_outer):
        radii = np.linspace(0, radius_guess, num_layers)

        density = np.zeros(num_layers)
        mass_enclosed = np.zeros(num_layers)
        gravity = np.zeros(num_layers)
        pressure = np.zeros(num_layers)

        if uses_Tdep:
            if temperature_mode == 'adiabatic' and outer_iter > 0 and prev_pressure is not None:
                # Recompute adiabat from previous iteration's converged structure
                adiabat_T = compute_adiabatic_temperature(
                    prev_radii,
                    prev_pressure,
                    prev_mass_enclosed,
                    surface_temperature,
                    cmb_mass,
                    core_mantle_mass,
                    layer_eos_config,
                    material_dictionaries,
                    interpolation_cache,
                )

                # Interpolate adiabat (on prev grid) onto current grid
                def temperature_function(r, _T=adiabat_T, _r=prev_radii):
                    return np.interp(np.array(r), _r, _T)

                temperatures = temperature_function(radii)
            else:
                temperature_function = calculate_temperature_profile(
                    radii,
                    temperature_mode,
                    surface_temperature,
                    center_temperature,
                    input_dir,
                    temp_profile_file,
                )
                temperatures = temperature_function(radii)
        else:
            temperatures = np.ones(num_layers) * 300

        cmb_mass = core_mass_fraction * planet_mass
        core_mantle_mass = (core_mass_fraction + mantle_mass_fraction) * planet_mass

        pressure[0] = earth_center_pressure

        for inner_iter in range(max_iterations_inner):
            old_density = density.copy()

            # Scaling-law estimate for central pressure
            pressure_guess = (
                earth_center_pressure
                * (planet_mass / earth_mass) ** 2
                * (radius_guess / earth_radius) ** (-4)
            )
            # Cap the central pressure guess for WolfBower2018 (1 TPa table)
            # but not for RTPress100TPa (100 TPa melt table)
            uses_WB2018 = any(
                v == 'WolfBower2018:MgSiO3' for v in layer_eos_config.values() if v
            )
            if uses_WB2018:
                pressure_guess = min(pressure_guess, max_center_pressure_guess)

            # Mutable state to capture the last ODE solution from inside
            # the residual function (brentq doesn't return intermediate
            # results, only the root)
            _state = {'mass_enclosed': None, 'gravity': None, 'pressure': None, 'n_evals': 0}

            def _pressure_residual(p_center):
                """Surface pressure residual f(P_c) = P_surface(P_c) - P_target.

                When the ODE integration terminates early (pressure hit zero
                before reaching the planet surface), P_center is too low and
                the residual is negative.
                """
                y0 = [0, 0, p_center]
                m, g, p = solve_structure(
                    layer_eos_config,
                    cmb_mass,
                    core_mantle_mass,
                    radii,
                    adaptive_radial_fraction,
                    relative_tolerance,
                    absolute_tolerance,
                    maximum_step,
                    material_dictionaries,
                    interpolation_cache,
                    y0,
                    solidus_func,
                    liquidus_func,
                    temperature_function if uses_Tdep else None,
                )
                if iteration_profiles_enabled:
                    create_pressure_density_files(
                        outer_iter, inner_iter, _state['n_evals'], radii, p, density
                    )
                _state['mass_enclosed'] = m
                _state['gravity'] = g
                _state['pressure'] = p
                _state['n_evals'] += 1

                # Early termination: pressure reached zero before the
                # surface (padded with zeros) → P_center is too low
                if p[-1] <= 0:
                    return -target_surface_pressure

                return p[-1] - target_surface_pressure

            # Bracket: P_center must be positive and wide enough to
            # straddle the root (where surface P = target P)
            p_low = max(1e6, 0.1 * pressure_guess)
            p_high = 10.0 * pressure_guess
            if uses_WB2018:
                p_high = min(p_high, max_center_pressure_guess)

            try:
                p_solution, root_info = brentq(
                    _pressure_residual,
                    p_low,
                    p_high,
                    xtol=1e6,
                    rtol=1e-10,
                    maxiter=max_iterations_pressure,
                    full_output=True,
                )
                # Re-run solve_structure at the exact root to get clean profiles
                # (brentq may have evaluated _state at a slightly different P)
                y0_root = [0, 0, p_solution]
                mass_enclosed, gravity, pressure = solve_structure(
                    layer_eos_config,
                    cmb_mass,
                    core_mantle_mass,
                    radii,
                    adaptive_radial_fraction,
                    relative_tolerance,
                    absolute_tolerance,
                    maximum_step,
                    material_dictionaries,
                    interpolation_cache,
                    y0_root,
                    solidus_func,
                    liquidus_func,
                    temperature_function if uses_Tdep else None,
                )

                surface_residual = abs(pressure[-1] - target_surface_pressure)
                # Allow zero pressure at the surface: the terminal event
                # pads truncated points with P=0, so check >= 0
                if (
                    root_info.converged
                    and surface_residual < pressure_tolerance
                    and np.min(pressure) >= 0
                ):
                    converged_pressure = True
                    verbose and logger.info(
                        f'Surface pressure converged after '
                        f'{root_info.function_calls} evaluations (Brent method).'
                    )
                else:
                    converged_pressure = False
                    verbose and logger.warning(
                        f'Brent method: converged={root_info.converged}, '
                        f'residual={surface_residual:.2e} Pa, '
                        f'min_P={np.min(pressure):.2e} Pa.'
                    )
            except ValueError:
                # f(p_low) and f(p_high) have the same sign — bracket
                # invalid.  Use the last evaluated solution if available.
                verbose and logger.warning(
                    f'Could not bracket pressure root in [{p_low:.2e}, {p_high:.2e}] Pa.'
                )
                if _state['mass_enclosed'] is not None:
                    mass_enclosed = _state['mass_enclosed']
                    gravity = _state['gravity']
                    pressure = _state['pressure']
                else:
                    # No evaluations succeeded — keep profiles from previous
                    # outer iteration (already initialised above).
                    verbose and logger.warning(
                        'No valid ODE solutions obtained during bracket search. '
                        'Keeping previous profiles.'
                    )
                converged_pressure = False

            # Update density grid (solve_structure may return fewer points
            # than num_layers if the ODE solver terminated early)
            for i in range(min(num_layers, len(mass_enclosed))):
                layer_eos = get_layer_eos(
                    mass_enclosed[i],
                    cmb_mass,
                    core_mantle_mass,
                    layer_eos_config,
                )

                new_density = calculate_density(
                    pressure[i],
                    material_dictionaries,
                    layer_eos,
                    temperatures[i],
                    solidus_func,
                    liquidus_func,
                    interpolation_cache,
                )

                if new_density is None:
                    verbose and logger.warning(
                        f'Density calculation failed at radius {radii[i]}. Using previous density.'
                    )
                    new_density = old_density[i]

                density[i] = 0.5 * (new_density + old_density[i])

            # Check density convergence
            relative_diff_inner = np.max(
                np.abs((density - old_density) / (old_density + 1e-20))
            )
            if relative_diff_inner < tolerance_inner:
                verbose and logger.info(
                    f'Inner loop converged after {inner_iter + 1} iterations.'
                )
                converged_density = True
                break

            if inner_iter == max_iterations_inner - 1:
                verbose and logger.warning(
                    f'Maximum inner iterations ({max_iterations_inner}) reached. '
                    'Density may not be fully converged.'
                )

        # Save converged profiles for the next outer iteration's adiabat
        prev_radii = radii.copy()
        prev_pressure = np.asarray(pressure).copy()
        prev_mass_enclosed = np.asarray(mass_enclosed).copy()

        # Update radius guess
        calculated_mass = mass_enclosed[-1]
        radius_guess = radius_guess * (planet_mass / calculated_mass) ** (1 / 3)
        cmb_mass = core_mass_fraction * calculated_mass
        core_mantle_mass = (core_mass_fraction + mantle_mass_fraction) * calculated_mass

        relative_diff_outer_mass = np.abs((calculated_mass - planet_mass) / planet_mass)

        if relative_diff_outer_mass < tolerance_outer:
            logger.info(f'Outer loop (total mass) converged after {outer_iter + 1} iterations.')
            converged_mass = True
            break

        if outer_iter == max_iterations_outer - 1:
            verbose and logger.warning(
                f'Maximum outer iterations ({max_iterations_outer}) reached. '
                'Total mass may not be fully converged.'
            )

    if converged_mass and converged_density and converged_pressure:
        converged = True

    end_time = time.time()
    total_time = end_time - start_time

    model_results = {
        'layer_eos_config': layer_eos_config,
        'radii': radii,
        'density': density,
        'gravity': gravity,
        'pressure': pressure,
        'temperature': temperatures,
        'mass_enclosed': mass_enclosed,
        'cmb_mass': cmb_mass,
        'core_mantle_mass': core_mantle_mass,
        'total_time': total_time,
        'converged': converged,
        'converged_pressure': converged_pressure,
        'converged_density': converged_density,
        'converged_mass': converged_mass,
    }
    return model_results


def post_processing(config_params, id_mass=None, output_file=None):
    """Post-process model results by saving output data and plotting.

    Parameters
    ----------
    config_params : dict
        Configuration parameters for the model.
    id_mass : str or None
        Identifier for the planet mass, used in output file naming.
    output_file : str or None
        Path to the output file for calculated mass and radius.
    """
    data_output_enabled = config_params['data_output_enabled']
    plotting_enabled = config_params['plotting_enabled']

    layer_eos_config = config_params['layer_eos_config']

    model_results = main(
        config_params,
        material_dictionaries=load_material_dictionaries(),
        melting_curves_functions=load_solidus_liquidus_functions(layer_eos_config),
        input_dir=os.path.join(ZALMOXIS_ROOT, 'input'),
    )

    # Extract results
    radii = model_results['radii']
    density = model_results['density']
    gravity = model_results['gravity']
    pressure = model_results['pressure']
    temperature = model_results['temperature']
    mass_enclosed = model_results['mass_enclosed']
    cmb_mass = model_results['cmb_mass']
    core_mantle_mass = model_results['core_mantle_mass']
    total_time = model_results['total_time']
    converged = model_results['converged']
    converged_pressure = model_results['converged_pressure']
    converged_density = model_results['converged_density']
    converged_mass = model_results['converged_mass']

    cmb_index = np.argmax(mass_enclosed >= cmb_mass)

    average_density = mass_enclosed[-1] / (4 / 3 * math.pi * radii[-1] ** 3)

    # Check if mantle uses Tdep EOS for phase detection
    uses_Tdep_mantle = layer_eos_config.get('mantle') in TDEP_EOS_NAMES

    if uses_Tdep_mantle:
        mantle_pressures = pressure[cmb_index:]
        mantle_temperatures = temperature[cmb_index:]
        mantle_radii = radii[cmb_index:]

        solidus_func, liquidus_func = load_solidus_liquidus_functions(layer_eos_config)

        mantle_phases = get_Tdep_material(
            mantle_pressures, mantle_temperatures, solidus_func, liquidus_func
        )

    logger.info('Exoplanet Internal Structure Model Results:')
    logger.info('----------------------------------------------------------------------')
    logger.info(
        f'Calculated Planet Mass: {mass_enclosed[-1]:.2e} kg or '
        f'{mass_enclosed[-1] / earth_mass:.2f} Earth masses'
    )
    logger.info(
        f'Calculated Planet Radius: {radii[-1]:.2e} m or '
        f'{radii[-1] / earth_radius:.2f} Earth radii'
    )
    logger.info(f'Core Radius: {radii[cmb_index]:.2e} m')
    logger.info(f'Mantle Density (at CMB): {density[cmb_index]:.2f} kg/m^3')
    logger.info(f'Core Density (at CMB): {density[cmb_index - 1]:.2f} kg/m^3')
    logger.info(f'Pressure at Core-Mantle Boundary (CMB): {pressure[cmb_index]:.2e} Pa')
    logger.info(f'Pressure at Center: {pressure[0]:.2e} Pa')
    logger.info(f'Average Density: {average_density:.2f} kg/m^3')
    logger.info(f'CMB Mass Fraction: {mass_enclosed[cmb_index] / mass_enclosed[-1]:.3f}')
    logger.info(
        f'Core+Mantle Mass Fraction: '
        f'{(core_mantle_mass - mass_enclosed[cmb_index]) / mass_enclosed[-1]:.3f}'
    )
    logger.info(f'Calculated Core Radius Fraction: {radii[cmb_index] / radii[-1]:.2f}')
    logger.info(
        f'Calculated Core+Mantle Radius Fraction: '
        f'{(radii[np.argmax(mass_enclosed >= core_mantle_mass)] / radii[-1]):.2f}'
    )
    logger.info(f'Total Computation Time: {total_time:.2f} seconds')
    logger.info(
        f'Overall Convergence Status: {converged} with Pressure: {converged_pressure}, '
        f'Density: {converged_density}, Mass: {converged_mass}'
    )

    if data_output_enabled:
        output_data = np.column_stack(
            (radii, density, gravity, pressure, temperature, mass_enclosed)
        )
        header = (
            'Radius (m)\tDensity (kg/m^3)\tGravity (m/s^2)\t'
            'Pressure (Pa)\tTemperature (K)\tMass Enclosed (kg)'
        )
        if id_mass is None:
            np.savetxt(
                os.path.join(ZALMOXIS_ROOT, 'output_files', 'planet_profile.txt'),
                output_data,
                header=header,
            )
        else:
            np.savetxt(
                os.path.join(ZALMOXIS_ROOT, 'output_files', f'planet_profile{id_mass}.txt'),
                output_data,
                header=header,
            )
        if output_file is None:
            output_file = os.path.join(
                ZALMOXIS_ROOT, 'output_files', 'calculated_planet_mass_radius.txt'
            )
        if not os.path.exists(output_file):
            header = 'Calculated Mass (kg)\tCalculated Radius (m)'
            with open(output_file, 'w') as file:
                file.write(header + '\n')
        with open(output_file, 'a') as file:
            file.write(f'{mass_enclosed[-1]}\t{radii[-1]}\n')

    if plotting_enabled:
        plot_planet_profile_single(
            radii,
            density,
            gravity,
            pressure,
            temperature,
            radii[np.argmax(mass_enclosed >= cmb_mass)],
            cmb_mass,
            mass_enclosed[-1] / (4 / 3 * math.pi * radii[-1] ** 3),
            mass_enclosed,
            id_mass,
        )

        if uses_Tdep_mantle:
            plot_PT_with_phases(
                mantle_pressures,
                mantle_temperatures,
                mantle_radii,
                mantle_phases,
                radii[cmb_index],
            )
