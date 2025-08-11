from __future__ import annotations

import itertools
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from ternary import figure
from tqdm import tqdm

from src.zalmoxis import zalmoxis
from zalmoxis.constants import earth_radius

# Run file via command line: python -m zalmoxis.plots.plot_ternary

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

# Set up logging
logger = logging.getLogger(__name__)

def run_zalmoxis_for_ternary(args):
    """
    Runs the zalmoxis main function for a given set of core and mantle fractions,
    and returns the core fraction, mantle fraction, and planet radius.
    Parameters:
        args (tuple): A tuple containing the planet mass, core fraction, and mantle fraction.
    Returns:
        tuple: A tuple containing the core fraction, mantle fraction, and planet radius.
    """
    id_mass, core_frac, mantle_frac = args
    id_mass = float(id_mass)
    core_frac = float(core_frac)
    mantle_frac = float(mantle_frac)
    water_frac = 1.0 - core_frac - mantle_frac

    # Path to the default configuration file
    default_config_path = os.path.join(ZALMOXIS_ROOT, "input", "default.toml")
    config_params = zalmoxis.load_zalmoxis_config(default_config_path)

    # Modify the configuration parameters as needed
    config_params["planet_mass"] = id_mass * 5.972e24
    config_params["core_mass_fraction"] = core_frac
    config_params["mantle_mass_fraction"] = mantle_frac
    config_params["weight_iron_fraction"] = core_frac  # must be equal to core_mass_fraction
    config_params["EOS_CHOICE"] = "Tabulated:water"

    # Unpack outputs directly from Zalmoxis
    model_results = zalmoxis.main(config_params, material_dictionaries=zalmoxis.load_material_dictionaries())
    converged = model_results.get("converged", False)

    # Check if model converged before proceeding
    if not model_results.get("converged", False):
        logger.warning(f"Model did not converge for core: {core_frac}, mantle: {mantle_frac}")
        return converged

    # Extract the results from the model output
    radii = model_results["radii"]
    total_time = model_results["total_time"]
    planet_radius = radii[-1]

    # Log the composition and radius only if converged
    custom_log_file = os.path.join(ZALMOXIS_ROOT, "output_files", f"composition_radius_log{id_mass}.txt")
    with open(custom_log_file, "a") as log:
        log.write(f"{core_frac:.4f}\t{mantle_frac:.4f}\t{water_frac:.4f}\t{planet_radius:.4e}\t{total_time:.4e}\n")
    return core_frac, mantle_frac, water_frac, converged

def generate_composition_grid(step=0.05):
    """
    Generates a list of valid (core_frac, mantle_frac) combinations where:
    core_frac + mantle_frac <= 1
    Parameters:
        step (float): Step size for generating fractions, controls the resolution of the grid.
    """
    grid = []
    fractions = np.arange(0.0, 1.01, step)
    for core, mantle in itertools.product(fractions, repeat=2):
        if core + mantle <= 1.0:
            grid.append((core, mantle))
    return grid

def run_ternary_grid_for_mass(id_mass=None):
    """
    Run zalmoxis for a grid of core and mantle fractions for a given planet mass,
    showing progress with tqdm.
    Parameters:
        id_mass (float): Mass of the planet in Earth masses.
    Returns:
        list: A list of results from the zalmoxis runs.
    """
    grid = generate_composition_grid(step=0.05)
    args_list = [(id_mass, core, mantle) for (core, mantle) in grid]

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(run_zalmoxis_for_ternary, args) for args in args_list]

        results = []
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing mass {id_mass}"):
            result = future.result()
            results.append(result)
    logger.info(f"Completed Zalmoxis runs for {len(args_list)} composition points.")
    return results

def read_results(id_mass=None):
    """
    Reads the results from the log file and returns a list of tuples containing
    (core_fraction, mantle_fraction, water_fraction, radius, total_time).
    Parameters:
        id_mass (float): Mass of the planet in Earth masses.
    Returns:
        list: A list of tuples with the composition and radius data.
    """
    log_path = os.path.join(ZALMOXIS_ROOT, "output_files", f"composition_radius_log{id_mass}.txt")
    data = []
    with open(log_path, 'r') as file:
        for line in file:
            try:
                core, mantle, water, radius, total_time = map(float, line.strip().split())
                data.append((core, mantle, water, radius, total_time))
            except ValueError:
                continue  # skip malformed lines
    return data

def plot_ternary(data):
    """
    Plot a ternary diagram of (core, mantle, water) mass fractions as percentages.
    Points are coloured by planet radius, normalised to Earth radii (R⊕).
    """

    # Normalise radii to Earth units
    radii_re = [radius / earth_radius for (_, _, _, radius, _) in data]
    rmin, rmax = min(radii_re), max(radii_re)

    # Convert fractions to percentages by multiplying by 100
    points = [(core * 100, water * 100, mantle * 100) for (core, mantle, water, _, _) in data]

    # Colours mapped
    colours = [(r - rmin) / (rmax - rmin) for r in radii_re]
    colour_mapped = [plt.cm.viridis(val) for val in colours]

    # Set scale to 100 (percent scale)
    scale = 100.0
    fig, tax = figure(scale=scale)
    tax.boundary()
    tax.gridlines(color="gray", multiple=5)  # gridlines every 10%

    tax.scatter(points, marker='o', color=colour_mapped, s=24)

    # Mark the special point with an X
    special_point = (40, 45, 15)  # Core=40%, Water=45%, Mantle=15%
    tax.scatter([special_point], marker='x', color='red', s=100, linewidths=2, label='Special Point (40,45,15)')

    # Axis labels with percent signs
    tax.left_axis_label("Mantle (%)", fontsize=12, offset=0.14)
    tax.right_axis_label("Water (%)", fontsize=12, offset=0.14)
    tax.bottom_axis_label("Core (%)", fontsize=12, offset=0.07)

    # Annotate the apices in percentage
    tax.annotate("100 % Core", position=(100, 0, 0), fontsize=10,
             xytext=(+5, -25), textcoords='offset points',
             horizontalalignment='right', verticalalignment='bottom')

    tax.annotate("100 % Water", position=(0, 100, 0), fontsize=10,
                xytext=(-10, 15), textcoords='offset points',
                verticalalignment='bottom')

    tax.annotate("100 % Mantle", position=(0, 0, 100), fontsize=10,
                xytext=(5, -15), textcoords='offset points',
                verticalalignment='top')

    tax.ticks(axis='lbr', multiple=10, linewidth=1, fontsize=8)  # ticks every 10%

    tax.clear_matplotlib_ticks()

    # Colour-bar (in Earth radii)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=rmin, vmax=rmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=tax.get_axes(), orientation='vertical')
    cbar.set_label("Radius (R⊕)")

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(ZALMOXIS_ROOT, "output_files", "ternary_diagram.png"), dpi=300)

def plot_ternary_time(data):
    """
    Plot a ternary diagram of (core, mantle, water) mass fractions as percentages.
    Points are coloured by total time taken for the simulation.
    """

    # Extract total time values
    total_times = [total_time for (_, _, _, _, total_time) in data]
    tmin, tmax = min(total_times), max(total_times)

    # Convert fractions to percentages by multiplying by 100
    points = [(core * 100, water * 100, mantle * 100) for (core, mantle, water, _, _) in data]

    # Colours mapped
    colours = [(t - tmin) / (tmax - tmin) for t in total_times]
    colour_mapped = [plt.cm.viridis(val) for val in colours]

    # Set scale to 100 (percent scale)
    scale = 100.0
    fig, tax = figure(scale=scale)
    tax.boundary()
    tax.gridlines(color="gray", multiple=5)  # gridlines every 10%

    tax.scatter(points, marker='o', color=colour_mapped, s=24)

    # Axis labels with percent signs
    tax.left_axis_label("Mantle (%)", fontsize=12, offset=0.14)
    tax.right_axis_label("Water (%)", fontsize=12, offset=0.14)
    tax.bottom_axis_label("Core (%)", fontsize=12, offset=0.07)

    # Annotate the apices in percentage
    tax.annotate("100 % Core", position=(100, 0, 0), fontsize=10,
             xytext=(+5, -25), textcoords='offset points',
             horizontalalignment='right', verticalalignment='bottom')

    tax.annotate("100 % Water", position=(0, 100, 0), fontsize=10,
                xytext=(-10, 15), textcoords='offset points',
                verticalalignment='bottom')

    tax.annotate("100 % Mantle", position=(0, 0, 100), fontsize=10,
                xytext=(5, -15), textcoords='offset points',
                verticalalignment='top')

    tax.ticks(axis='lbr', multiple=10, linewidth=1, fontsize=8)  # ticks every 10%

    tax.clear_matplotlib_ticks()

    # Colour-bar (in seconds)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=tmin, vmax=tmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=tax.get_axes(), orientation='vertical')
    cbar.set_label("Total Time (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(ZALMOXIS_ROOT, "output_files", "ternary_diagram_time.png"), dpi=300)

def wrapper_ternary(id_mass):
    """ Wrapper function to run the ternary grid and plot the results.
    It deletes the composition_radius_log file if it exists, runs the ternary grid for a default planet mass,
    reads the results, and plots the ternary diagrams.
    Parameters:
        id_mass (float): Mass of the planet in Earth masses.
    """
    # Delete composition_radius_log file if it exists
    custom_log_file = os.path.join(ZALMOXIS_ROOT, "output_files", f"composition_radius_log{id_mass}.txt")
    if os.path.exists(custom_log_file):
        os.remove(custom_log_file)

    # Run the ternary grid for the specified planet mass
    run_ternary_grid_for_mass(id_mass)

    # Read the results and plot the ternary diagrams
    data = read_results(id_mass)
    plot_ternary(data)
    plot_ternary_time(data)

if __name__ == "__main__":
    wrapper_ternary(id_mass=10.0)
