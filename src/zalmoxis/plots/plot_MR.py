from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from ..constants import earth_mass, earth_radius

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

# Function to plot the mass-radius relationship of planets and compare with Earth-like Rocky (32.5% Fe+67.5% MgSiO3) planets from Zeng et al. (2019)
def plot_mass_radius_relationship(target_mass_array):
    """
    Plots the mass-radius relationship of planets using data from a specified file and compares it with Earth-like rocky and water planet data from Zeng et al. (2019).

    Parameters:
    target_mass_array (list or array): Array of target masses for which the model is run.

    The function performs the following steps:
    1. Reads mass and radius data for Earth-like rocky and water planets from Zeng et al. (2019).
    2. Reads mass and radius data for modeled planets from output files.
    3. Plots the mass-radius relationship for all datasets.
    4. Saves the plot as a PDF file named 'MR_plot.pdf'.

    Notes:
    - Input data files should have two columns: mass and radius, separated by whitespace.
    - Masses and radii in the modeled data files are normalized to Earth units using `earth_mass` and `earth_radius`.
    - The function requires the ZALMOXIS_ROOT environment variable to be set.
    """
    # Read data from Zeng et al. (2019) for Earth-like Rocky (32.5% Fe+67.5% MgSiO3) planets
    zeng_masses_Earth = []
    zeng_radii_Earth = []
    with open(os.path.join(ZALMOXIS_ROOT, "data", "mass_radius_curves", "massradiusEarthlikeRocky.txt"), 'r') as zeng_file:
        next(zeng_file)  # Skip the header line
        for line in zeng_file:
            mass, radius = map(float, line.split())
            zeng_masses_Earth.append(mass)
            zeng_radii_Earth.append(radius)

    # Read data from Zeng et al. (2019) for water planets (50 % H2O + 50% Earth-like rocky core)
    zeng_masses_water = []
    zeng_radii_water = []
    with open(os.path.join(ZALMOXIS_ROOT, "data", "mass_radius_curves", "massradius_50percentH2O_300K_1mbar.txt"), 'r') as zeng_file:
        next(zeng_file)  # Skip the header line
        for line in zeng_file:
            mass, radius = map(float, line.split())
            zeng_masses_water.append(mass)
            zeng_radii_water.append(radius)

    # Read data from file with calculated planet masses and radii by the model
    masses_Earth = []
    radii_Earth = []
    with open(os.path.join(ZALMOXIS_ROOT, "output_files", "calculated_planet_mass_radius_Earth.txt"), 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            mass, radius = map(float, line.split())
            masses_Earth.append(mass/earth_mass)
            radii_Earth.append(radius/earth_radius)

    # Read data from file with calculated planet masses and radii by the model
    masses_water = []
    radii_water = []
    with open(os.path.join(ZALMOXIS_ROOT, "output_files", "calculated_planet_mass_radius_water.txt"), 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            mass, radius = map(float, line.split())
            masses_water.append(mass/earth_mass)
            radii_water.append(radius/earth_radius)

    # Plot the MR graph
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(masses_Earth, radii_Earth, color='red', label='Modeled Earth-like Rocky planets')
    ax.scatter(masses_water, radii_water, color='blue', label='Modeled Water planets')
    ax.plot(zeng_masses_Earth, zeng_radii_Earth, color='darkred', label='Earth-like Rocky Planets (Zeng et al. 2019)')
    ax.plot(zeng_masses_water, zeng_radii_water, color='darkblue', label='Water Planets (Zeng et al. 2019)')
    ax.set_xlabel('Planet Mass (Earth Masses)')
    ax.set_ylabel('Planet Radius (Earth Radii)')
    ax.set_title('Calculated Mass-Radius Relationship of Planets')
    ax.set_xlim(0, np.max(target_mass_array))
    ax.set_ylim(0, 5)
    ax.legend()
    ax.grid(True)
    plt.savefig(os.path.join(ZALMOXIS_ROOT, "output_files", "MR_plot.pdf"))
    #plt.show()
    plt.close(fig)

