from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

from ..constants import earth_mass, earth_radius

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

# Function to plot the profiles of all planets in one plot for comparison
def plot_profiles_all_in_one(target_mass_array, choice):
    """
    Plots various planetary profiles (Density, Gravity, Pressure, and Mass Enclosed) for a set of planet masses.

    This function reads planet profile data from text files, processes the data, and generates comparative plots
    for different planet masses. The plots include:
    - Radius vs Density
    - Radius vs Gravity
    - Radius vs Pressure
    - Radius vs Mass Enclosed

    Optionally, comparison data from literature (Wagner et al. 2012, Boujibar et al. 2020, Seager et al. 2007) can be overlaid.

    Parameters:
        target_mass_array (list or array-like): List of planet masses (in Earth masses) to plot profiles for.
        choice (str): Choice of comparison data. Options are 'Wagner', 'Boujibar', 'SeagerEarth', 'Seagerwater', 'custom', or 'default'.

    Data files:
        - Planet profile files: 'planet_profile{id_mass}.txt' in the output_files directory.
          Each file should contain space-separated columns:
            Radius (m), Density (kg/m^3), Gravity (m/s^2), Pressure (Pa), Temperature (K), Mass Enclosed (kg)
        - Literature comparison files are expected in the 'data' directory.

    Output:
        The generated plot is saved as 'all_profiles_with_colorbar_vs_{choice}.pdf' in the output_files directory.

    Raises:
        RuntimeError: If the ZALMOXIS_ROOT environment variable is not set.
        ValueError: If an invalid choice is provided for the comparison data.
    """
    # Initialize a list to hold the data for plotting
    data_list = []  # List of dictionaries to hold data for each planet

    # Read data from files with calculated planet profiles
    for id_mass in target_mass_array:
        # Generate file path for each planet profile
        file_path = os.path.join(ZALMOXIS_ROOT, "output_files", f"planet_profile{id_mass}.txt")

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

        with open(os.path.join(ZALMOXIS_ROOT, "data", "radial_profiles", "radiusdensityWagner.txt"), 'r') as wagner_file:
            for line in wagner_file:
                radius, density = map(float, line.split(','))
                wagner_radii_for_densities.append(radius*earth_radius/1000) # Convert to km
                wagner_densities.append(density) # in kg/m^3

        wagner_radii_for_pressures = []
        wagner_pressures = []

        with open(os.path.join(ZALMOXIS_ROOT, "data", "radial_profiles", "radiuspressureWagner.txt"), 'r') as wagner_file:
            for line in wagner_file:
                radius, pressure = map(float, line.split(','))
                wagner_radii_for_pressures.append(radius*earth_radius/1000) # Convert to km
                wagner_pressures.append(pressure) #in GPa

        wagner_radii_for_gravities = []
        wagner_gravities = []

        with open(os.path.join(ZALMOXIS_ROOT, "data", "radial_profiles", "radiusgravityWagner.txt"), 'r') as wagner_file:
            for line in wagner_file:
                radius, gravity = map(float, line.split(','))
                wagner_radii_for_gravities.append(radius*earth_radius/1000) # Convert to km
                wagner_gravities.append(gravity) #in GPa

    elif choice == "Boujibar":
        # Read data from Boujibar et al. (2020) for comparison
        boujibar_radii_for_densities = []
        boujibar_densities = []

        with open(os.path.join(ZALMOXIS_ROOT, "data", "radial_profiles", "radiusdensityEarthBoujibar.txt"), 'r') as boujibar_file:
            for line in boujibar_file:
                radius, density = map(float, line.split(','))
                boujibar_radii_for_densities.append(radius)
                boujibar_densities.append(density * 1000) # Convert to kg/m^3

        boujibar_radii_for_pressures = []
        boujibar_pressures = []

        with open(os.path.join(ZALMOXIS_ROOT, "data", "radial_profiles", "radiuspressureEarthBoujibar.txt"), 'r') as boujibar_file:
            for line in boujibar_file:
                radius, pressure = map(float, line.split(','))
                boujibar_radii_for_pressures.append(radius)
                boujibar_pressures.append(pressure) #in GPa

    elif choice == "default":
        pass
    elif choice == "SeagerEarth":
        # Read data from Seager et al. (2007) for Earth-like super-Earths
        seagerEarth_radii_for_densities = []
        seagerEarth_densities = []

        with open(os.path.join(ZALMOXIS_ROOT, "data", "radial_profiles", "radiusdensitySeagerEarth.txt"), 'r') as seagerEarth_file:
            for line in seagerEarth_file:
                radius, density = map(float, line.split(','))
                seagerEarth_radii_for_densities.append(radius*1000) # Convert to km
                seagerEarth_densities.append(density*1000) # Convert to kg/m^3
    elif choice == "Seagerwater":
        # Read data from Seager et al. (2007) for water planets
        seagerwater_radii_for_densities = []
        seagerwater_densities = []

        with open(os.path.join(ZALMOXIS_ROOT, "data", "radial_profiles", "radiusdensitySeagerwater.txt"), 'r') as seagerwater_file:
            for line in seagerwater_file:
                radius, density = map(float, line.split(','))
                seagerwater_radii_for_densities.append(radius*1000) # Convert to km
                seagerwater_densities.append(density*1000) # Convert to kg/m^3
    elif choice == "custom":
        pass
    else:
        raise ValueError("Invalid choice. Please select 'Wagner', 'Boujibar', 'SeagerEarth', 'Seagerwater', 'SeagerHHe' or 'default'.")

    # Create a colormap based on the id_mass values
    cmap = cm.plasma
    norm = Normalize(vmin=1, vmax=np.max(target_mass_array))  # Normalize the id_mass range to map to colors

    # Plot the profiles for comparison using ax method
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot Radius vs Density
    for data in data_list:
        color = cmap(norm(data['id_mass']))
        axs[0, 0].plot(data['radius'], data['density'], color=color)
    if choice == "Wagner":
        axs[0, 0].scatter(wagner_radii_for_densities, wagner_densities, color='black', s=1, label='Earth-like super-Earths (Wagner et al. 2012)')
    elif choice == "Boujibar":
        axs[0, 0].scatter(boujibar_radii_for_densities, boujibar_densities, color='black', s=1, label='Earth-like super-Earths (Boujibar et al. 2020)')
    elif choice == "default":
        pass
    elif choice == "SeagerEarth":
        axs[0, 0].scatter(seagerEarth_radii_for_densities, seagerEarth_densities, color='black', s=5, label='Earth-like super-Earths (Seager et al. 2007)', zorder=10)
    elif choice == "Seagerwater":
        axs[0, 0].scatter(seagerwater_radii_for_densities, seagerwater_densities, color='black', s=5, label='Water planets with iron core and silicate mantles (Seager et al. 2007)', zorder=10)
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
        axs[0, 1].scatter(wagner_radii_for_gravities, wagner_gravities, color='black', s=1, label='Earth-like super-Earths (Wagner et al. 2012)')
    elif choice == "Boujibar":
        pass
    elif choice == "default":
        pass
    elif choice == "SeagerEarth":
        pass
    elif choice == "Seagerwater":
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
        axs[1, 0].scatter(wagner_radii_for_pressures, wagner_pressures, color='black', s=1, label='Earth-like super-Earths (Wagner et al. 2012)')
    elif choice == "Boujibar":
        axs[1, 0].scatter(boujibar_radii_for_pressures, boujibar_pressures, color='black', s=1, label='Earth-like super-Earths (Boujibar et al. 2020)')
    elif choice == "default":
        pass
    elif choice == "SeagerEarth":
        pass
    elif choice == "Seagerwater":
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
    axs[1, 1].set_ylabel(r"Mass Enclosed (M$_\oplus$)")
    axs[1, 1].set_title("Radius vs Mass Enclosed")
    axs[1, 1].grid()

    # Add a colorbar to the plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # Empty array to avoid warning
    cbar = plt.colorbar(sm, ax=axs, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label(r"Planet Mass (M$_\oplus$)")

    # Adjust layout and show plot
    #plt.tight_layout()
    plt.suptitle(f"Planet Profiles Comparison ({choice})")
    plt.savefig(os.path.join(ZALMOXIS_ROOT, "output_files", f"all_profiles_with_colorbar_vs_{choice}.pdf"))
    #plt.show()
    plt.close(fig)
