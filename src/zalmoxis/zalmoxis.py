from __future__ import annotations

import logging
import math
import os
import sys
import time

import numpy as np
import toml
from scipy.integrate import solve_ivp

from .constants import earth_center_pressure, earth_mass, earth_radius
from .eos_functions import calculate_density
from .eos_properties import (
    material_properties_iron_silicate_planets,
    material_properties_water_planets,
)
from .plots.plot_eos import plot_eos_material
from .plots.plot_profiles import plot_planet_profile_single
from .structure_model import coupled_odes

# Run file via command line with default configuration file: python -m zalmoxis -c input/default.toml

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

logger = logging.getLogger(__name__)

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
            logger.info(f"Reading temporary config file from: {temp_config_path}")
        except FileNotFoundError:
            logger.error(f"Error: Temporary config file not found at {temp_config_path}")
            sys.exit(1)
    elif "-c" in sys.argv:
        index = sys.argv.index("-c")
        try:
            config_file_path = sys.argv[index + 1]
            config = toml.load(config_file_path)
            logger.info(f"Reading config file from: {config_file_path}")
        except IndexError:
            logger.error("Error: -c flag provided but no config file path specified.")
            sys.exit(1)  # Exit with error code
        except FileNotFoundError:
            logger.error(f"Error: Config file not found at {config_file_path}")
            sys.exit(1)
    else:
        config_default_path = os.path.join(ZALMOXIS_ROOT, "input", "default.toml")
        try:
            config = toml.load(config_default_path)
            logger.info(f"Reading default config file from {config_default_path}")
        except FileNotFoundError:
            logger.info(f"Error: Default config file not found at {config_default_path}")
            sys.exit(1)

    return config

def load_zalmoxis_config(temp_config_path=None):
    """
    Loads and returns configuration parameters for the Zalmoxis model.
    Returns:
        dict: Dictionary containing all relevant configuration parameters.
    """
    config = choose_config_file(temp_config_path) # Choose the configuration file

    # Extract and return all relevant configuration parameters
    return {
        "planet_mass": config['InputParameter']['planet_mass'] * earth_mass,  # Convert Earth masses to kg
        "core_mass_fraction": config['AssumptionsAndInitialGuesses']['core_mass_fraction'],
        "mantle_mass_fraction": config['AssumptionsAndInitialGuesses']['mantle_mass_fraction'],
        "weight_iron_fraction": config['AssumptionsAndInitialGuesses']['weight_iron_fraction'],
        "EOS_CHOICE": config['EOS']['choice'],
        "num_layers": config['Calculations']['num_layers'],
        "max_iterations_outer": config['IterativeProcess']['max_iterations_outer'],
        "tolerance_outer": config['IterativeProcess']['tolerance_outer'],
        "max_iterations_inner": config['IterativeProcess']['max_iterations_inner'],
        "tolerance_inner": config['IterativeProcess']['tolerance_inner'],
        "relative_tolerance": config['IterativeProcess']['relative_tolerance'],
        "absolute_tolerance": config['IterativeProcess']['absolute_tolerance'],
        "target_surface_pressure": config['PressureAdjustment']['target_surface_pressure'],
        "pressure_tolerance": config['PressureAdjustment']['pressure_tolerance'],
        "max_iterations_pressure": config['PressureAdjustment']['max_iterations_pressure'],
        "pressure_adjustment_factor": config['PressureAdjustment']['pressure_adjustment_factor'],
        "data_output_enabled": config['Output']['data_enabled'],
        "plotting_enabled": config['Output']['plots_enabled'],
        "verbose": config['Output']['verbose']
    }

def load_material_dictionaries():
    """
    Loads and returns the material properties dictionaries for the Zalmoxis model.
    Returns:
        tuple: A tuple containing two dictionaries with material properties for iron/silicate and water planets.
    """
    material_dictionaries = (material_properties_iron_silicate_planets, material_properties_water_planets)
    return material_dictionaries

def main(config_params, material_dictionaries):

    """
    Runs the exoplanet internal structure model.

    Iteratively adjusts the internal structure of an exoplanet based on the provided configuration parameters,
    calculating the planet's radius, core-mantle boundary, densities, pressures, and other properties.

    Parameters:
        config_params (dict): Dictionary containing configuration parameters for the model.

    Returns:
        dict: Dictionary containing the calculated radii, density, gravity, pressure, temperature, mass enclosed,
                core-mantle boundary mass, core+mantle mass, total computation time, and convergence status of the model.
    """
    # Initialize convergence flags for the model
    converged = False  # Overall convergence flag for the model, assume not converged until proven otherwise
    converged_pressure = False  # Assume pressure not converged until proven otherwise
    converged_density = False  # Assume density not converged until proven otherwise
    converged_mass = False  # Assume mass not converged until proven otherwise

    # Unpack configuration parameters
    planet_mass = config_params["planet_mass"]
    core_mass_fraction = config_params["core_mass_fraction"]
    mantle_mass_fraction = config_params["mantle_mass_fraction"]
    weight_iron_fraction = config_params["weight_iron_fraction"]
    EOS_CHOICE = config_params["EOS_CHOICE"]
    num_layers = config_params["num_layers"]
    max_iterations_outer = config_params["max_iterations_outer"]
    tolerance_outer = config_params["tolerance_outer"]
    max_iterations_inner = config_params["max_iterations_inner"]
    tolerance_inner = config_params["tolerance_inner"]
    relative_tolerance = config_params["relative_tolerance"]
    absolute_tolerance = config_params["absolute_tolerance"]
    target_surface_pressure = config_params["target_surface_pressure"]
    pressure_tolerance = config_params["pressure_tolerance"]
    max_iterations_pressure = config_params["max_iterations_pressure"]
    pressure_adjustment_factor = config_params["pressure_adjustment_factor"]
    verbose = config_params["verbose"]

    # Setup initial guesses for the planet radius and core-mantle boundary mass
    radius_guess = 1000*(7030-1840*weight_iron_fraction)*(planet_mass/earth_mass)**0.282 # Initial guess for the interior planet radius [m] based on the scaling law in Noack et al. 2020
    cmb_mass = 0 # Initial guess for the core-mantle boundary mass [kg]
    core_mantle_mass = 0 # Initial guess for the core+mantle mass [kg]

    # Initialize temperature profile
    temperature = np.zeros(num_layers) # Dummy initialization, will be calculated later

    # Time the entire process
    start_time = time.time()

    # Solve the interior structure
    for outer_iter in range(max_iterations_outer): # Outer loop for radius and mass convergence

        # Setup initial guess for the radial grid based on the radius guess
        radii = np.linspace(0, radius_guess, num_layers)

        # Initialize arrays for mass, gravity, density, and pressure grids
        density = np.zeros(num_layers)
        mass_enclosed = np.zeros(num_layers)
        gravity = np.zeros(num_layers)
        pressure = np.zeros(num_layers)

        # Setup initial guess for the core-mantle boundary mass
        cmb_mass = core_mass_fraction * planet_mass

        # Setup initial guess for the core+mantle mass
        core_mantle_mass = (core_mass_fraction + mantle_mass_fraction) * planet_mass

        # Setup initial guess for the pressure at the center of the planet (needed for solving the ODEs)
        pressure[0] = earth_center_pressure

        for inner_iter in range(max_iterations_inner): # Inner loop for density adjustment
            old_density = density.copy() # Store the old density for convergence check

            # Initialize empty cache for interpolation functions for density calculations
            interpolation_cache = {}

            # Setup initial pressure guess at the center of the planet based on empirical scaling law derived from the hydrostatic equilibrium equation
            pressure_guess = earth_center_pressure * (planet_mass/earth_mass)**2 * (radius_guess/earth_radius)**(-4)

            for pressure_iter in range(max_iterations_pressure): # Innermost loop for pressure adjustment

                # Setup initial conditions for the mass, gravity, and pressure at the center of the planet
                y0 = [0, 0, pressure_guess]

                # Solve the ODEs using solve_ivp
                sol = solve_ivp(lambda r, y: coupled_odes(r, y, cmb_mass, core_mantle_mass, EOS_CHOICE, interpolation_cache, material_dictionaries),
                    (radii[0], radii[-1]), y0, t_eval=radii, rtol=relative_tolerance, atol=absolute_tolerance, method='RK45', dense_output=True)

                # Extract mass, gravity, and pressure grids from the solution
                mass_enclosed = sol.y[0]
                gravity = sol.y[1]
                pressure = sol.y[2]

                # Extract the calculated surface pressure from the last element of the pressure array
                surface_pressure = pressure[-1]

                # Calculate the pressure difference between the calculated surface pressure and the target surface pressure
                pressure_diff = surface_pressure - target_surface_pressure

                # Check for convergence of the surface pressure and overall pressure positivity
                if np.abs(pressure_diff) < pressure_tolerance and np.all(pressure > 0):
                    verbose and logger.info(f"Surface pressure converged after {pressure_iter + 1} iterations and all pressures are positive.")
                    converged_pressure = True  # Set convergence flag for pressure to True if converged
                    break  # Exit the pressure adjustment loop

                # Update the pressure guess at the center of the planet based on the pressure difference at the surface using an adjustment factor
                pressure_guess -= pressure_diff * pressure_adjustment_factor

                # Check if maximum iterations for pressure adjustment are reached
                if pressure_iter == max_iterations_pressure - 1:
                    verbose and logger.warning(f"Maximum pressure iterations ({max_iterations_pressure}) reached. Surface pressure may not be fully converged.")

            # Update density grid based on the mass enclosed within a certain mass fraction
            for i in range(num_layers):
                if EOS_CHOICE == "Tabulated:iron/silicate":
                    # Define the material type based on the calculated enclosed mass up to the core-mantle boundary
                    if mass_enclosed[i] < cmb_mass:
                        # Core
                        material = "core"
                    else:
                        # Mantle
                        material = "mantle"
                elif EOS_CHOICE == "Tabulated:water":
                    # Define the material type based on the calculated enclosed mass up to the core-mantle boundary
                    if mass_enclosed[i] < cmb_mass:
                        # Core
                        material = "core"
                    elif mass_enclosed[i] < core_mantle_mass:
                        # Inner mantle
                        material = "bridgmanite_shell"
                    else:
                        # Outer layer
                        material = "water_ice_layer"

                # Calculate the new density using the equation of state
                new_density = calculate_density(pressure[i], material_dictionaries, material, EOS_CHOICE)

                # Handle potential errors in density calculation
                if new_density is None:
                    verbose and logger.warning(f"Density calculation failed at radius {radii[i]}. Using previous density.")
                    new_density = old_density[i]

                # Update the density grid with a weighted average of the new and old density
                density[i] = 0.5 * (new_density + old_density[i])

            # Check for convergence of density using relative difference between old and new density
            relative_diff_inner = np.max(np.abs((density - old_density) / (old_density + 1e-20)))
            if relative_diff_inner < tolerance_inner:
                verbose and logger.info(f"Inner loop converged after {inner_iter + 1} iterations.")
                converged_density = True  # Set convergence flag for density to True if converged
                break # Exit the inner loop

            # Check if maximum iterations for inner loop are reached
            if inner_iter == max_iterations_inner - 1:
                verbose and logger.warning(f"Maximum inner iterations ({max_iterations_inner}) reached. Density may not be fully converged.")

        # Extract the calculated total interior mass of the planet from the last element of the mass array
        calculated_mass = mass_enclosed[-1]

        # Update the total interior radius by scaling the initial guess based on the calculated mass
        radius_guess = radius_guess * (planet_mass / calculated_mass)**(1/3)

        # Update the core-mantle boundary mass based on the core mass fraction and calculated total interior mass of the planet
        cmb_mass = core_mass_fraction * calculated_mass

        # Update the inner mantle mass based on the inner mantle mass fraction and calculated total interior mass of the planet
        core_mantle_mass = (core_mass_fraction + mantle_mass_fraction) * calculated_mass

        # Calculate relative differences of the calculated total interior mass
        relative_diff_outer_mass = np.abs((calculated_mass - planet_mass) / planet_mass)

        # Check for convergence of the calculated total interior mass
        if relative_diff_outer_mass < tolerance_outer:
            logger.info(f"Outer loop (total mass) converged after {outer_iter + 1} iterations.")
            converged_mass = True  # Set convergence flag to True if converged
            break  # Exit the outer loop

        # Check if maximum iterations for outer loop are reached
        if outer_iter == max_iterations_outer - 1:
            verbose and logger.warning(f"Maximum outer iterations ({max_iterations_outer}) reached. Total mass may not be fully converged.")

    # Calculate the temperature profile
    #temperature = calculate_temperature(radii, cmb_radius, 300, material_properties_iron_silicate_planets, gravity, density, material_properties_iron_silicate_planets["mantle"]["K0"], dr=planet_radius/num_layers)

    # Check for overall convergence of the model
    if converged_mass and converged_density and converged_pressure:
        converged = True

    # End timing the entire process
    end_time = time.time()

    # Calculate the total time taken for the entire process
    total_time = end_time - start_time

    # Save the calculated values for further use in a dictionary
    model_results = {
        "radii": radii,
        "density": density,
        "gravity": gravity,
        "pressure": pressure,
        "temperature": temperature,
        "mass_enclosed": mass_enclosed,
        "cmb_mass": cmb_mass,
        "core_mantle_mass": core_mantle_mass,
        "total_time": total_time,
        "converged": converged,
        "converged_pressure": converged_pressure,
        "converged_density": converged_density,
        "converged_mass": converged_mass
    }
    return model_results

def post_processing(config_params, id_mass=None, output_file=None):
    """
    Post-processes the results of the Zalmoxis model by saving output data to a file and plotting results.
    Parameters:
        config_params (dict): Dictionary containing configuration parameters for the model.
        id_mass (str, optional): Identifier for the mass of the planet, used in output file naming.
        output_file (str, optional): Path to the output file where calculated mass and radius will be saved.
    """

    # Unpack configuration parameters related to output
    data_output_enabled = config_params["data_output_enabled"]
    plotting_enabled = config_params["plotting_enabled"]

    # Load the model output data
    model_results = main(config_params, material_dictionaries=load_material_dictionaries())

    # Extract the results from the model output
    radii = model_results["radii"]
    density = model_results["density"]
    gravity = model_results["gravity"]
    pressure = model_results["pressure"]
    temperature = model_results["temperature"]
    mass_enclosed = model_results["mass_enclosed"]
    cmb_mass = model_results["cmb_mass"]
    core_mantle_mass = model_results["core_mantle_mass"]
    total_time = model_results["total_time"]
    converged = model_results["converged"]
    converged_pressure = model_results["converged_pressure"]
    converged_density = model_results["converged_density"]
    converged_mass = model_results["converged_mass"]

    # Extract the index of the core-mantle boundary mass in the mass array
    cmb_index = np.argmax(mass_enclosed >= cmb_mass)

    # Calculate the average density of the planet using the calculated mass and radius
    average_density = mass_enclosed[-1] / (4/3 * math.pi * radii[-1]**3)

    logger.info("Exoplanet Internal Structure Model Results:")
    logger.info("----------------------------------------------------------------------")
    logger.info(f"Calculated Planet Mass: {mass_enclosed[-1]:.2e} kg or {mass_enclosed[-1]/earth_mass:.2f} Earth masses")
    logger.info(f"Calculated Planet Radius: {radii[-1]:.2e} m or {radii[-1]/earth_radius:.2f} Earth radii")
    logger.info(f"Core Radius: {radii[cmb_index]:.2e} m")
    logger.info(f"Mantle Density (at CMB): {density[cmb_index]:.2f} kg/m^3")
    logger.info(f"Core Density (at CMB): {density[cmb_index- 1]:.2f} kg/m^3")
    logger.info(f"Pressure at Core-Mantle Boundary (CMB): {pressure[cmb_index]:.2e} Pa")
    logger.info(f"Pressure at Center: {pressure[0]:.2e} Pa")
    logger.info(f"Average Density: {average_density:.2f} kg/m^3")
    logger.info(f"CMB Mass Fraction: {mass_enclosed[cmb_index] / mass_enclosed[-1]:.3f}")
    logger.info(f"Core+Mantle Mass Fraction: {(core_mantle_mass - mass_enclosed[cmb_index]) / mass_enclosed[-1]:.3f}")
    logger.info(f"Calculated Core Radius Fraction: {radii[cmb_index] / radii[-1]:.2f}")
    logger.info(f"Calculated Core+Mantle Radius Fraction: {(radii[np.argmax(mass_enclosed >= core_mantle_mass)] / radii[-1]):.2f}")
    logger.info(f"Total Computation Time: {total_time:.2f} seconds")
    logger.info(f"Overall Convergence Status: {converged} with Pressure: {converged_pressure}, Density: {converged_density}, Mass: {converged_mass}")

    # Save output data to a file
    if data_output_enabled:
        # Combine and save plotted data to a single output file
        output_data = np.column_stack((radii, density, gravity, pressure, temperature, mass_enclosed))
        header = "Radius (m)\tDensity (kg/m^3)\tGravity (m/s^2)\tPressure (Pa)\tTemperature (K)\tMass Enclosed (kg)"
        if id_mass is None:
            np.savetxt(os.path.join(ZALMOXIS_ROOT, "output_files", "planet_profile.txt"), output_data, header=header)
        else:
            np.savetxt(os.path.join(ZALMOXIS_ROOT, "output_files", f"planet_profile{id_mass}.txt"), output_data, header=header)
        # Append calculated mass and radius of the planet to a file in dedicated columns
        if output_file is None:
            output_file = os.path.join(ZALMOXIS_ROOT, "output_files", "calculated_planet_mass_radius.txt")
        if not os.path.exists(output_file):
            header = "Calculated Mass (kg)\tCalculated Radius (m)"
            with open(output_file, "w") as file:
                file.write(header + "\n")
        with open(output_file, "a") as file:
            file.write(f"{mass_enclosed[-1]}\t{radii[-1]}\n")

    # Plotting results
    if plotting_enabled:
        plot_planet_profile_single(radii, density, gravity, pressure, temperature, radii[np.argmax(mass_enclosed >= cmb_mass)] / radii[-1], cmb_mass, mass_enclosed[-1] / (4/3 * math.pi * radii[-1]**3), mass_enclosed, id_mass) # Plot planet profile for a single planet
        eos_data_files = ['eos_seager07_iron.txt', 'eos_seager07_silicate.txt', 'eos_seager07_water.txt']
        eos_data_folder = os.path.join(ZALMOXIS_ROOT, "data", "EOS_Seager2007")
        plot_eos_material(eos_data_files, eos_data_folder)  # Plot the equation of state data for the materials used in the model
        #plt.show()
