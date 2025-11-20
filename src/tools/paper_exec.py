#!/usr/bin/env python3
# Run as `python src/tools/paper_exec.py`
from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from glob import glob

import numpy as np

import zalmoxis.zalmoxis as zalmoxis
from zalmoxis.constants import earth_mass

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(levelname)s] %(message)s', datefmt='%H:%M:%S')

def run_zalmoxis(mass_and_core):
    """
    Runs the Zalmoxis model for a given total mass and core mass fraction.
    """

    # Path to the default configuration file
    default_config_path = os.path.join(ZALMOXIS_ROOT, "input", "default.toml")
    config_params = zalmoxis.load_zalmoxis_config(temp_config_path=default_config_path)

    mass = mass_and_core[0]
    core = mass_and_core[1]

    # Modify the configuration parameters as needed
    config_params["planet_mass"]          = mass * earth_mass
    config_params["core_mass_fraction"]   = core
    config_params["weight_iron_fraction"] = core
    config_params["verbose"] = False

    # Run the main function with the temporary configuration file
    id = str(f"_M{mass:0.4f}_C{core:0.4f}")
    zalmoxis.main(config_params, material_dictionaries=zalmoxis.load_material_dictionaries())
    zalmoxis.post_processing(config_params, id_mass=id, print_output=False)

def run_zalmoxis_in_parallel():
    """
    Runs zalmoxis in parallel for a range of planet masses based on the provided choice,
    deletes the contents of the calculated_planet_mass_radius.txt file if it exists,
    and calls the function to plot the profiles of all planets in one plot.

    Raises:
        ValueError: If an invalid choice is provided for the comparison data.

    Returns:
        None
    """

    # Parameters
    workers = 60
    target_mass_array = np.round(np.arange(0.25, 10.25, 0.25), 4)
    target_core_array = np.round(np.arange(0.1,  0.75,  0.05), 4)


    # Setup output
    output_dir = os.path.join(ZALMOXIS_ROOT, "output_files") + "/"
    logger.info(f"Output folder: {output_dir}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    # Delete existing profiles
    for f in glob(output_dir + "planet_profile*.txt"):
        os.remove(f)

    # Delete the contents of the calculated_planet_mass_radius.txt file if it exists
    calculated_file_path = os.path.join(output_dir, "calculated_planet_mass_radius.txt")
    logger.info(f"Output data file: {calculated_file_path}")
    if os.path.exists(calculated_file_path):
        os.remove(calculated_file_path)
    with open(calculated_file_path, 'w') as file:
        file.truncate(0)
        header = "Calculated Mass (kg)\tCalculated Radius (m)"
        file.write(header + "\n")


    num_pts = len(target_mass_array) * len(target_core_array)
    logger.info(f"Will run in parallel with {workers} workers")
    logger.info(f"Number of points: {num_pts}")
    time.sleep(3.0)

    target_array = []
    for m in target_mass_array:
        for c in target_core_array:
            target_array.append([m,c])

    # Run zalmoxis in parallel for a range of planet masses
    logger.info("Running...")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        executor.map(run_zalmoxis, target_array)

    # Plot the mass-radius relationship
    # plot_mass_radius_relationship(target_mass_array) # if rocky and water planets are simulated in two separate runs

    # Call the function to plot the profiles of all planets in one plot
    # plot_profiles_all_in_one(target_mass_array, choice)

if __name__ == "__main__":
    start_time = time.time()

    logging.info("Direct execution")
    run_zalmoxis_in_parallel()

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total time for running Zalmoxis in parallel is: {total_time:.2f} seconds")
