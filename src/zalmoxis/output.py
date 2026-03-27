"""Post-processing, file output, and plotting orchestration for Zalmoxis."""

from __future__ import annotations

import logging
import math
import os

import numpy as np

from . import get_zalmoxis_root
from .config import _NEEDS_MELTING_CURVES
from .constants import earth_mass, earth_radius
from .eos import get_Tdep_material
from .mixing import parse_layer_components

logger = logging.getLogger(__name__)


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
    from .config import load_material_dictionaries, load_solidus_liquidus_functions
    from .solver import main

    data_output_enabled = config_params['data_output_enabled']
    plotting_enabled = config_params['plotting_enabled']

    layer_eos_config = config_params['layer_eos_config']
    solidus_id = config_params.get('rock_solidus', 'Stixrude14-solidus')
    liquidus_id = config_params.get('rock_liquidus', 'Stixrude14-liquidus')

    model_results = main(
        config_params,
        material_dictionaries=load_material_dictionaries(),
        melting_curves_functions=load_solidus_liquidus_functions(
            layer_eos_config, solidus_id, liquidus_id
        ),
        input_dir=os.path.join(get_zalmoxis_root(), 'input'),
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

    # Check if mantle uses a Tdep EOS that needs external melting curves
    # for phase detection. Unified PALEOS tables derive phases from the
    # table itself and do not need (or support) get_Tdep_material().
    mantle_str = layer_eos_config.get('mantle', '')
    if mantle_str:
        _mantle_mix = parse_layer_components(mantle_str)
        uses_phase_detection = bool(set(_mantle_mix.components) & _NEEDS_MELTING_CURVES)
    else:
        uses_phase_detection = False

    if uses_phase_detection:
        mantle_pressures = pressure[cmb_index:]
        mantle_temperatures = temperature[cmb_index:]
        mantle_radii = radii[cmb_index:]

        solidus_func, liquidus_func = load_solidus_liquidus_functions(
            layer_eos_config, solidus_id, liquidus_id
        )

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
                os.path.join(get_zalmoxis_root(), 'output', 'planet_profile.txt'),
                output_data,
                header=header,
            )
        else:
            np.savetxt(
                os.path.join(
                    get_zalmoxis_root(), 'output', f'planet_profile{id_mass}.txt'
                ),
                output_data,
                header=header,
            )
        if output_file is None:
            output_file = os.path.join(
                get_zalmoxis_root(), 'output', 'calculated_planet_mass_radius.txt'
            )
        if not os.path.exists(output_file):
            header = 'Calculated Mass (kg)\tCalculated Radius (m)'
            with open(output_file, 'w') as file:
                file.write(header + '\n')
        with open(output_file, 'a') as file:
            file.write(f'{mass_enclosed[-1]}\t{radii[-1]}\n')

    if plotting_enabled:
        from tools.plots.plot_phase_vs_radius import plot_PT_with_phases
        from tools.plots.plot_profiles import plot_planet_profile_single

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
            layer_eos_config=layer_eos_config,
        )

        if uses_phase_detection:
            plot_PT_with_phases(
                mantle_pressures,
                mantle_temperatures,
                mantle_radii,
                mantle_phases,
                radii[cmb_index],
            )
