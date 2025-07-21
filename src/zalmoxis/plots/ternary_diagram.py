import os, sys
import tempfile
import toml
from zalmoxis import zalmoxis  
from concurrent.futures import ProcessPoolExecutor
import time
import itertools
import numpy as np
import ternary
import matplotlib.pyplot as plt
import ternary
from zalmoxis.constants import earth_radius

# Run file via command line: python -m src.tools.run_parallel Wagner/Boujibar/default/SeagerEarth/Seagerwater/custom

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

# Function to run the main function with a temporary configuration file
def run_zalmoxis_for_ternary(args):
    """
    Runs the zalmoxis main function with a temporary configuration file.

    Parameters:
        id_mass (float, optional): The mass of the planet in Earth masses.

    Raises:
        FileNotFoundError: If the default configuration file is not found.
        toml.TomlDecodeError: If there is an error decoding the TOML configuration file.
        Exception: If there is an error during the execution of the main function.

    Returns:
        None
    """
    id_mass, core_frac, mantle_frac = args
    id_mass = float(id_mass)
    core_frac = float(core_frac)
    mantle_frac = float(mantle_frac)
    water_frac = 1.0 - core_frac - mantle_frac

    # Path to the default configuration file
    default_config_path = os.path.join(ZALMOXIS_ROOT, "input", "default.toml")

    # Load the default configuration
    with open(default_config_path, 'r') as file:
        config = toml.load(file)

    # Modify the configuration parameters as needed
    config['InputParameter']['planet_mass'] = id_mass*5.972e24
    config['AssumptionsAndInitialGuesses']['core_mass_fraction'] = core_frac # first mass fraction
    config['AssumptionsAndInitialGuesses']['inner_mantle_mass_fraction'] = mantle_frac # second mass fraction
    config['AssumptionsAndInitialGuesses']['weight_iron_fraction'] = core_frac# must be equal to core_mass_fraction
    config['EOS']['choice'] = "Tabulated:water"
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

    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.toml') as temp_config_file:
        toml.dump(config, temp_config_file)  # Dump the updated config into the file
        temp_config_path = temp_config_file.name  # Get the path to the temporary file

    # Run the main function with the temporary configuration file
    try:
        # Unpack outputs directly from Zalmoxis
        radii, density, gravity, pressure, temperature, mass_enclosed, total_time = zalmoxis.main(temp_config_path, id_mass)

        planet_radius = max(radii)  # or radii[-1], depending on Zalmoxis' radius order

        # Log the composition and radius
        custom_log_file = os.path.join(ZALMOXIS_ROOT, "output_files", "composition_radius_log.txt")
        with open(custom_log_file, "a") as log:
            log.write(f"{core_frac:.4f}\t{mantle_frac:.4f}\t{water_frac:.4f}\t{planet_radius:.4e}\t{total_time:.4e}\n")

    except Exception as e:
        print(f"Failed for core: {core_frac}, mantle: {mantle_frac} -> {e}")

    # Clean up the temporary configuration file after running
    os.remove(temp_config_path)

def generate_composition_grid(step=0.05):
    """
    Generates a list of valid (core_frac, mantle_frac) combinations where:
    core_frac + mantle_frac <= 1
    """
    grid = []
    fractions = np.arange(0.0, 1.01, step)
    for core, mantle in itertools.product(fractions, repeat=2):
        if core + mantle <= 1.0:
            grid.append((core, mantle))
    return grid

def run_ternary_grid_for_mass(planet_mass=1.0):
    """
    Run zalmoxis for all valid 3-layer compositions for a fixed mass and generate ternary data.
    """
    grid = generate_composition_grid(step=0.05)
    args_list = [(planet_mass, core, mantle) for (core, mantle) in grid]

    with ProcessPoolExecutor() as executor:
        executor.map(run_zalmoxis_for_ternary, args_list)

    print(f"Completed Zalmoxis runs for {len(args_list)} composition points.")

def read_results():
    log_path = os.path.join(ZALMOXIS_ROOT, "output_files", "composition_radius_log.txt")
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
    radii_re = [radius / earth_radius for (*_, radius) in data]
    rmin, rmax = min(radii_re), max(radii_re)

    # Convert fractions to percentages by multiplying by 100
    points = [(core * 100, water * 100, mantle * 100) for (core, mantle, water, _) in data]

    # Colours mapped as before
    colours = [(r - rmin) / (rmax - rmin) for r in radii_re]
    colour_mapped = [plt.cm.viridis(val) for val in colours]

    # Set scale to 100 (percent scale)
    scale = 100.0
    fig, tax = ternary.figure(scale=scale)
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
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=rmin, vmax=rmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=tax.get_axes(), orientation='vertical')
    cbar.set_label("Radius (R⊕)")

    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(ZALMOXIS_ROOT, "output_files", "ternary_diagram.png"), dpi=300)

#create another ternary function that plots the time instead of radius
def plot_ternary_time(data):
    """
    Plot a ternary diagram of (core, mantle, water) mass fractions as percentages.
    Points are coloured by total time taken for the simulation.
    """

    # Extract total time values
    total_times = [total_time for (*_, total_time) in data]
    tmin, tmax = min(total_times), max(total_times)

    # Convert fractions to percentages by multiplying by 100
    points = [(core * 100, water * 100, mantle * 100) for (core, mantle, water, _, _) in data]

    # Colours mapped as before
    colours = [(t - tmin) / (tmax - tmin) for t in total_times]
    colour_mapped = [plt.cm.viridis(val) for val in colours]

    # Set scale to 100 (percent scale)
    scale = 100.0
    fig, tax = ternary.figure(scale=scale)
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
    sm = plt.cm.ScalarMappable(cmap="viridis",
                               norm=plt.Normalize(vmin=tmin, vmax=tmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=tax.get_axes(), orientation='vertical')
    cbar.set_label("Total Time (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(ZALMOXIS_ROOT, "output_files", "ternary_diagram_time.png"), dpi=300)

if __name__ == "__main__":
    run_ternary_grid_for_mass(planet_mass=50)  # runs all models and writes the log file
    data = read_results()                       # reads the log file
    plot_ternary_time(data)                          # plots the ternary diagram
