import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from ..constants import *

# Run file via command line: python -m src.jord.plots.plot_profiles_all_in_one

# Function to plot the profiles of all planets in one plot for comparison
def plot_profiles_all_in_one():
    """
    Plots various profiles (Density, Gravity, Pressure, and Mass Enclosed) for planets with different masses.

    This function reads planet profile data from text files, processes the data, and generates comparative plots
    for different planet masses. The plots include:
    - Radius vs Density
    - Radius vs Gravity
    - Radius vs Pressure
    - Radius vs Mass Enclosed

    The function also adds a colorbar to indicate the planet mass for each profile.

    The data files should be named in the format 'planet_profile{id_mass}.txt' and located in the '../output_files/' directory.
    Each file should contain space-separated values with columns representing:
    - Radius (m)
    - Density (kg/m^3)
    - Gravity (m/s^2)
    - Pressure (Pa)
    - Temperature (K)
    - Mass Enclosed (kg)

    The generated plot is saved as 'all_profiles_with_colorbar.pdf' in the parent directory.

    Raises:
        FileNotFoundError: If any of the expected data files are not found.

    """
    # Set the working directory to the current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Initialize a list to hold the data for plotting
    data_list = []  # List of dictionaries to hold data for each planet

    # Read data from files with calculated planet profiles
    for id_mass in range(1, 11):
        # Generate file path for each planet profile
        file_path = f"../output_files/planet_profile{id_mass}.txt"

        # Check if the file exists
        if os.path.exists(file_path):
            # Load the data from the file (assuming the format is space-separated)
            data = np.loadtxt(file_path)

            # Extract data for plotting: Radius, Density, Gravity, Pressure, Temperature, Mass Enclosed
            radius = data[:, 0]  # Radius (m)
            density = data[:, 1]  # Density (kg/m^3)
            gravity = data[:, 2]  # Gravity (m/s^2)
            pressure = data[:, 3]  # Pressure (Pa)
            temperature = data[:, 4]  # Temperature (K)
            mass = data[:, 5]  # Mass Enclosed (kg)

            # Append the data along with id_mass to the list
            data_dict = {
                'id_mass': id_mass,
                'radius': radius / 1e3,  # Convert to km
                'density': density,  # in kg/m^3
                'gravity': gravity,  # in m/s^2
                'pressure': pressure / 1e9,  # Convert to GPa
                'temperature': temperature,  # in K
                'mass': mass / earth_mass  # Convert to Earth masses
            }
            data_list.append(data_dict)
        else:
            print(f"File not found: {file_path}")

    # Create a colormap based on the id_mass values
    cmap = cm.inferno
    norm = Normalize(vmin=1, vmax=10)  # Normalize the id_mass range to map to colors

    # Plot the profiles for comparison using ax method
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot Radius vs Density
    for data in data_list:
        color = cmap(norm(data['id_mass']))
        axs[0, 0].plot(data['radius'], data['density'], color=color)
    axs[0, 0].set_xlabel("Radius (km)")
    axs[0, 0].set_ylabel("Density (kg/m$^3$)")
    axs[0, 0].set_title("Radius vs Density")
    axs[0, 0].grid()

    # Plot Radius vs Gravity
    for data in data_list:
        color = cmap(norm(data['id_mass']))
        axs[0, 1].plot(data['radius'], data['gravity'], color=color)
    axs[0, 1].set_xlabel("Radius (km)")
    axs[0, 1].set_ylabel("Gravity (m/s$^2$)")
    axs[0, 1].set_title("Radius vs Gravity")
    axs[0, 1].grid()

    # Plot Radius vs Pressure
    for data in data_list:
        color = cmap(norm(data['id_mass']))
        axs[1, 0].plot(data['radius'], data['pressure'], color=color)
    axs[1, 0].set_xlabel("Radius (km)")
    axs[1, 0].set_ylabel("Pressure (GPa)")
    axs[1, 0].set_title("Radius vs Pressure")
    axs[1, 0].grid()

    # Plot Radius vs Mass Enclosed
    for data in data_list:
        color = cmap(norm(data['id_mass']))
        axs[1, 1].plot(data['radius'], data['mass'], color=color)
    axs[1, 1].set_xlabel("Radius (km)")
    axs[1, 1].set_ylabel("Mass Enclosed (M$_\oplus$)")
    axs[1, 1].set_title("Radius vs Mass Enclosed")
    axs[1, 1].grid()

    # Add a colorbar to the plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array to avoid warning
    cbar = plt.colorbar(sm, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label("Planet Mass (M$_\oplus$)")

    # Adjust layout and show plot
    #plt.tight_layout()
    plt.suptitle("Planet Profiles Comparison")
    plt.savefig("../all_profiles_with_colorbar.pdf")
    #plt.show()

# Call the function to plot the profiles
plot_profiles_all_in_one()
