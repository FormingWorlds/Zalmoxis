from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from src.zalmoxis import zalmoxis
from src.zalmoxis.plots.plot_profiles_all_in_one import plot_profiles_all_in_one

# Run file via command line: python -m src.tools.run_parallel Wagner/Boujibar/default/SeagerEarth/Seagerwater/custom

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

# Set up logging
logger = logging.getLogger(__name__)

def run_zalmoxis(id_mass=None):
    """
    Runs the Zalmoxis model for a given planet mass.
    Parameters:
        id_mass (float): Mass of the planet in Earth masses.
    """

    # Path to the default configuration file
    default_config_path = os.path.join(ZALMOXIS_ROOT, "input", "default.toml")
    config_params = zalmoxis.load_zalmoxis_config(default_config_path)

    # Modify the configuration parameters as needed
    config_params["planet_mass"] = id_mass * 5.972e24

    # Run the main function with the temporary configuration file
    zalmoxis.post_processing(config_params, id_mass)

def run_zalmoxis_in_parallel(choice):
    """
    Runs zalmoxis in parallel for a range of planet masses based on the provided choice,
    deletes the contents of the calculated_planet_mass_radius.txt file if it exists,
    and calls the function to plot the profiles of all planets in one plot.

    Parameters:
        choice (str): Choice of comparison data. Options are 'Wagner', 'Boujibar', 'default', 'SeagerEarth', 'Seagerwater', or 'custom'.

    Raises:
        ValueError: If an invalid choice is provided for the comparison data.

    Returns:
        None
    """

    # Delete the contents of the calculated_planet_mass_radius.txt file if it exists
    calculated_file_path = os.path.join(ZALMOXIS_ROOT, "output_files", "calculated_planet_mass_radius.txt")
    if os.path.exists(calculated_file_path):
        with open(calculated_file_path, 'w') as file:
            file.truncate(0)
            header = "Calculated Mass (kg)\tCalculated Radius (m)"
            file.write(header + "\n")

    if choice == "Wagner":
        target_mass_array = [1, 2.5, 5, 7.5, 10, 12.5, 15]
    elif choice == "Boujibar":
        target_mass_array = range(1, 11)
    elif choice == "default":
        logger.info("No choice selected for the comparison, defaulting to 1 to 10 Earth masses simulation.")
        target_mass_array = range(1, 11)
    elif choice == "SeagerEarth":
        target_mass_array = [1, 5, 10, 50]
    elif choice == "Seagerwater":
        target_mass_array = [1, 5, 10, 50]
    elif choice == "custom":
        target_mass_array = range(1,51,1)
    else:
        raise ValueError("Invalid choice. Please select 'Wagner', 'Boujibar', 'default', 'SeagerEarth', 'Seagerwater', or 'custom'.")

    # Run zalmoxis in parallel for a range of planet masses
    with ProcessPoolExecutor() as executor:
        executor.map(run_zalmoxis, target_mass_array)

    # Plot the mass-radius relationship
    # plot_mass_radius_relationship(target_mass_array) # if rocky and water planets are simulated in two separate runs

    # Call the function to plot the profiles of all planets in one plot
    plot_profiles_all_in_one(target_mass_array, choice)

if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) != 2:
        logger.info("Usage: python -m src.tools.run_parallel <choice>")
        sys.exit(1)

    choice = sys.argv[1]
    run_zalmoxis_in_parallel(choice)

    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total time for running Zalmoxis in parallel with choice '{choice}' is: {total_time:.2f} seconds")
