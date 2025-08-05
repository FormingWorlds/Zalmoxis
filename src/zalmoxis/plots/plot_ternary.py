from __future__ import annotations

import os

import matplotlib.pyplot as plt
import ternary

from zalmoxis.constants import earth_radius

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

def read_results():
    log_path = os.path.join(ZALMOXIS_ROOT, "output_files", "composition_radius_log.txt")
    data = []
    with open(log_path, 'r') as file:
        for line in file:
            try:
                core, mantle, water, radius = map(float, line.strip().split())
                data.append((core, mantle, water, radius))
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
    #run_ternary_grid_for_mass(planet_mass=1.0)  # runs all models and writes the log file
    data = read_results()                       # reads the log file
    plot_ternary(data)                          # plots the ternary diagram
    plot_ternary_time(data)                     # plots the ternary diagram with time
