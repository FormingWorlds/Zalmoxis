from ..constants import *
import matplotlib.pyplot as plt
import os
import numpy as np

# Run file via command line: python -m src.jord.plots.plot_MR

# Function to plot the mass-radius relationship of planets and compare with Earth-like Rocky (32.5% Fe+67.5% MgSiO3) planets from Zeng et al. (2019)
def plot_mass_radius_relationship(target_mass_array):
    """
    Plots the mass-radius relationship of planets using data from a specified file and compares it with Earth-like rocky planets data from Zeng et al. (2019).

    Parameters:
    target_mass_array (list or array): Array of target masses for which the model is run.

    The function performs the following steps:
    1. Sets the working directory to the current file's directory.
    2. Reads mass and radius data for Earth-like rocky planets from Zeng et al. (2019).
    3. Reads mass and radius data for model planets from the specified data file.
    4. Plots the mass-radius relationship for both datasets.
    5. Saves the plot as a PDF file named 'MR_plot.pdf'.

    Note:
    - The input data file should have two columns: mass and radius, separated by whitespace.
    - The function assumes that the masses and radii in the input data file are in Earth masses and Earth radii, respectively.
    - The function uses the constants `earth_mass` and `earth_radius` for normalization.
    """

    # Set the working directory to the current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    data_file = '../output_files/calculated_planet_mass_radius.txt'

    # Read data from Zeng et al. (2019) for Earth-like Rocky (32.5% Fe+67.5% MgSiO3) planets
    zeng_masses = []
    zeng_radii = []

    with open("../../../data/massradiusEarthlikeRockyZeng.txt", 'r') as zeng_file:
        next(zeng_file)  # Skip the header line
        for line in zeng_file:
            mass, radius = map(float, line.split())
            zeng_masses.append(mass)
            zeng_radii.append(radius)

    # Read data from file with calculated planet masses and radii by the model
    masses = []
    radii = []

    with open(data_file, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            mass, radius = map(float, line.split())
            masses.append(mass/earth_mass)
            radii.append(radius/earth_radius)

    # Plot the MR graph
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(masses, radii, color='red', label='Model Planets')
    ax.plot(zeng_masses, zeng_radii, color='blue', label='Earth-like Rocky Planets (Zeng et al. 2019)')
    ax.set_xlabel('Planet Mass (Earth Masses)')
    ax.set_ylabel('Planet Radius (Earth Radii)')
    ax.set_title('Calculated Mass-Radius Relationship of Planets')
    ax.set_xlim(0, np.max(target_mass_array))
    ax.set_ylim(0, 5)
    ax.legend()
    ax.grid(True)
    plt.savefig("../MR_plot.pdf")
    #plt.show()
    plt.close(fig)

