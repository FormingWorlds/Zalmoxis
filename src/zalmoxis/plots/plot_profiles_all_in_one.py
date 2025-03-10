import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.colors import Normalize
from ..constants import *

# Run file via command line: python -m src.zalmoxis.plots.plot_profiles_all_in_one

# Function to plot the profiles of all planets in one plot for comparison
def plot_profiles_all_in_one(target_mass_array, choice):
    """
    Plots various profiles (Density, Gravity, Pressure, and Mass Enclosed) for planets with different masses.

    This function reads planet profile data from text files, processes the data, and generates comparative plots
    for different planet masses. The plots include:
    - Radius vs Density
    - Radius vs Gravity
    - Radius vs Pressure
    - Radius vs Mass Enclosed

    The function also adds a colorbar to indicate the planet mass for each profile.

    Parameters:
    target_mass_array (list): List of planet masses to plot profiles for.
    choice (str): Choice of comparison data. Options are 'Wagner', 'Boujibar', or 'default'.

    The data files should be named in the format 'planet_profile{id_mass}.txt' and located in the '../output_files/' directory.
    Each file should contain space-separated values with columns representing:
    - Radius (m)
    - Density (kg/m^3)
    - Gravity (m/s^2)
    - Pressure (Pa)
    - Temperature (K)
    - Mass Enclosed (kg)

    The generated plot is saved as 'all_profiles_with_colorbar_vs_{choice}.pdf' in the parent directory.

    Raises:
        FileNotFoundError: If any of the expected data files are not found.
        ValueError: If an invalid choice is provided for the comparison data.

    """
    # Set the working directory to the current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Initialize a list to hold the data for plotting
    data_list = []  # List of dictionaries to hold data for each planet

    # Read data from files with calculated planet profiles
    for id_mass in target_mass_array:
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

    if choice == "Wagner":
        # Read data from Wagner et al. (2012) for comparison
        wagner_radii_for_densities = []
        wagner_densities = []

        with open("../../../data/radiusdensityWagner.txt", 'r') as wagner_file:
            for line in wagner_file:
                radius, density = map(float, line.split(','))
                wagner_radii_for_densities.append(radius*earth_radius/1000) # Convert to km
                wagner_densities.append(density) # in kg/m^3 

        wagner_radii_for_pressures = []
        wagner_pressures = []

        with open("../../../data/radiuspressureWagner.txt", 'r') as wagner_file:
            for line in wagner_file:
                radius, pressure = map(float, line.split(','))
                wagner_radii_for_pressures.append(radius*earth_radius/1000) # Convert to km
                wagner_pressures.append(pressure) #in GPa

        wagner_radii_for_gravities = []
        wagner_gravities = []

        with open("../../../data/radiusgravityWagner.txt", 'r') as wagner_file:
            for line in wagner_file:
                radius, gravity = map(float, line.split(','))
                wagner_radii_for_gravities.append(radius*earth_radius/1000) # Convert to km
                wagner_gravities.append(gravity) #in GPa

    elif choice == "Boujibar":
        # Read data from Boujibar et al. (2020) for comparison
        boujibar_radii_for_densities = []
        boujibar_densities = []

        with open("../../../data/radiusdensityEarthBoujibar.txt", 'r') as boujibar_file:
            for line in boujibar_file:
                radius, density = map(float, line.split(','))
                boujibar_radii_for_densities.append(radius)
                boujibar_densities.append(density * 1000) # Convert to kg/m^3 

        boujibar_radii_for_pressures = []
        boujibar_pressures = []

        with open("../../../data/radiuspressureEarthBoujibar.txt", 'r') as boujibar_file:
            for line in boujibar_file:
                radius, pressure = map(float, line.split(','))
                boujibar_radii_for_pressures.append(radius)
                boujibar_pressures.append(pressure) #in GPa

    elif choice == "default":
        pass
    elif choice == "Seager":
        # Read data from Seager et al. (2007) for comparison
        seager_radii_for_densities = []
        seager_densities = []

        with open("../../../data/radiusdensitySeager.txt", 'r') as seager_file:
            for line in seager_file:
                radius, density = map(float, line.split(','))
                seager_radii_for_densities.append(radius*1000) # Convert to km
                seager_densities.append(density*1000) # Convert to kg/m^3
    elif choice == "custom":
        pass
    else:
        raise ValueError("Invalid choice. Please select 'Wagner', 'Boujibar', or 'default'.")

    # Create a colormap based on the id_mass values
    cmap = cm.inferno
    norm = Normalize(vmin=1, vmax=np.max(target_mass_array))  # Normalize the id_mass range to map to colors

    # Plot the profiles for comparison using ax method
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot Radius vs Density
    for data in data_list:
        color = cmap(norm(data['id_mass']))
        axs[0, 0].plot(data['radius'], data['density'], color=color)
    if choice == "Wagner":
        axs[0, 0].scatter(wagner_radii_for_densities, wagner_densities, color='green', s=1, label='Earth-like super-Earths (Wagner et al. 2012)')
    elif choice == "Boujibar":
        axs[0, 0].scatter(boujibar_radii_for_densities, boujibar_densities, color='green', s=1, label='Earth-like super-Earths (Boujibar et al. 2020)')
    elif choice == "default":
        pass
    elif choice == "Seager":
        axs[0, 0].scatter(seager_radii_for_densities, seager_densities, color='green', s=1, label='Earth-like super-Earths (Seager et al. 2007)')           
    elif choice == "custom":
        pass
    axs[0, 0].set_xlabel("Radius (km)")
    axs[0, 0].set_ylabel("Density (kg/m$^3$)")
    axs[0, 0].set_title("Radius vs Density")
    axs[0, 0].grid()

    # Plot Radius vs Gravity
    for data in data_list:
        color = cmap(norm(data['id_mass']))
        axs[0, 1].plot(data['radius'], data['gravity'], color=color)
    if choice == "Wagner":
        axs[0, 1].scatter(wagner_radii_for_gravities, wagner_gravities, color='green', s=1, label='Earth-like super-Earths (Wagner et al. 2012)')
    elif choice == "Boujibar":
        pass
    elif choice == "default":
        pass
    elif choice == "Seager":
        pass
    elif choice == "custom":
        pass
    axs[0, 1].set_xlabel("Radius (km)")
    axs[0, 1].set_ylabel("Gravity (m/s$^2$)")
    axs[0, 1].set_title("Radius vs Gravity")
    axs[0, 1].grid()

    # Plot Radius vs Pressure
    for data in data_list:
        color = cmap(norm(data['id_mass']))
        axs[1, 0].plot(data['radius'], data['pressure'], color=color)
    if choice == "Wagner":
        axs[1, 0].scatter(wagner_radii_for_pressures, wagner_pressures, color='green', s=1, label='Earth-like super-Earths (Wagner et al. 2012)')
    elif choice == "Boujibar":
        axs[1, 0].scatter(boujibar_radii_for_pressures, boujibar_pressures, color='green', s=1, label='Earth-like super-Earths (Boujibar et al. 2020)')
    elif choice == "default":
        pass
    elif choice == "Seager":
        pass    
    elif choice == "custom":
        pass
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
    plt.suptitle(f"Planet Profiles Comparison ({choice})")
    plt.savefig(f"../all_profiles_with_colorbar_vs_{choice}.pdf")
    #plt.show()
    plt.close(fig)
