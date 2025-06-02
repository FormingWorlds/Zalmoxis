import matplotlib.pyplot as plt
import numpy as np
import os
from ..constants import earth_center_pressure, earth_cmb_pressure

# Run this file via command line: python -m src.zalmoxis.plots.plot_eos

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

# Function to read data from a file
def read_eos_data(filename):
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    pressure = data[:, 1]  # Assuming pressure is in the second column (GPa)
    density = data[:, 0] * 1000  # Assuming density is in the first column (g/cm³), convert to kg/m³
    return pressure, density

# Function to plot EOS data
def plot_eos_material(data_files, data_folder):
    """
    Plots the equation of state (EOS) data for different materials.
    Parameters:
    data_files (list): List of filenames containing the EOS data.   
    data_folder (str): Path to the folder containing the data files.

    The function reads the EOS data from the specified files and plots the pressure-density relationship for each material.
    The data files should be CSV files with two columns: density (in g/cm³) and pressure (in GPa).
    The function assumes that the data files are located in the specified data_folder.
    The function plots the data on a log-log scale and inverts the y-axis to make it downward-increasing.
    """
    custom_labels = {
    'eos_seager07_iron.txt': 'Iron',
    'eos_seager07_silicate.txt': 'Silicate',
    'eos_seager07_water.txt': 'Water ice',
    # Add more as needed
    }

    custom_colors = {
    'eos_seager07_iron.txt': 'red',
    'eos_seager07_silicate.txt': 'orange',
    'eos_seager07_water.txt': 'blue',
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for file in data_files:
        filepath = os.path.join(data_folder, file)  # Use the passed data_folder
        pressure, density = read_eos_data(filepath)
        # Use the custom label if available, otherwise default to filename
        label = custom_labels.get(file, file)
        color = custom_colors.get(file, None)
        ax.plot(pressure, density, label=label, color=color)

    # Set plot labels and title
    ax.set_xlabel('Pressure (GPa)')
    ax.set_ylabel('Density (kg/m³)')
    ax.set_title('Equation of State (EOS) of Materials from Seager et al. (2007)')
    ax.legend()

    # Set log scale for both axes
    ax.set_xscale('log')
    ax.set_yscale('log')
    #ax.axvline(x=earth_cmb_pressure/10**9, color='gray', linestyle=':', label="Earth's core-mantle boundary (136 GPa)")
    #ax.axvline(x=earth_center_pressure/10**9, color='gray', linestyle='--', label="Earth's center (364 GPa)")

    # Show the plot
    ax.legend()
    fig.savefig(os.path.join(ZALMOXIS_ROOT, "src", "zalmoxis", "output_files", "planet_eos.pdf"))
    #plt.show()
    plt.close(fig)
