from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

def read_eos_data(filename):
    """
    Reads the equation of state (EOS) data from Seager et al. (2007) files.
    Parameters:
    filename (str): Path to the file containing EOS data.
    Returns:
    tuple: A tuple containing two numpy arrays: pressure (in GPa) and density (in kg/m³).
    The file should contain two columns: density (in g/cm³) and pressure (in GPa).
    """
    data = np.loadtxt(filename, delimiter=',', skiprows=1)
    pressure = data[:, 1]  # Assuming pressure is in the second column (GPa)
    density = data[:, 0] * 1000  # Assuming density is in the first column (g/cm³), convert to kg/m³
    return pressure, density

def read_eos_WolfBower2018_data(filename):
    """
    Reads the equation of state (EOS) data from Wolf & Bower (2018) files.
    Parameters:
    filename (str): Path to the file containing EOS data.
    Returns:
    tuple: A tuple containing three numpy arrays: pressures (in Pa), temps (in K), and densities (in kg/m³).
    """
    data = np.loadtxt(filename, delimiter="\t", skiprows=1)
    pressures = data[:, 0] # in Pa
    temps = data[:, 1] # in K
    densities = data[:, 2] # in kg/m^3

    return pressures, temps, densities

def plot_eos_Seager2007(data_files, data_folder):
    """
    Plots the equation of state (EOS) data for different materials from Seager et al. (2007).
    Parameters:
    data_files (list): List of filenames containing the EOS data.
    data_folder (str): Path to the folder containing the data files.

    The function reads the EOS data from the specified files and plots the pressure-density relationship for each material.
    The data files should be CSV files with two columns: density (in g/cm³) and pressure (in GPa).
    The function assumes that the data files are located in the specified data_folder.
    The function plots the data on a log-log scale.
    """
    custom_labels = {
    'eos_seager07_iron.txt': 'Iron',
    'eos_seager07_silicate.txt': 'Silicate',
    'eos_seager07_water.txt': 'Water ice',
    }

    custom_colors = {
    'eos_seager07_iron.txt': 'red',
    'eos_seager07_silicate.txt': 'orange',
    'eos_seager07_water.txt': 'blue',
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    for file in data_files:
        filepath = os.path.join(data_folder, file)
        pressure, density = read_eos_data(filepath)
        # Use the custom label if available, otherwise default to filename
        label = custom_labels.get(file, file)
        color = custom_colors.get(file, None)
        ax.plot(pressure, density, label=label, color=color)

    ax.set_xlabel('Pressure (GPa)')
    ax.set_ylabel('Density (kg/m³)')
    ax.set_title('Equation of State (EOS) of Materials from Seager et al. (2007)')
    ax.legend()
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    fig.savefig(os.path.join(ZALMOXIS_ROOT, "output_files", "planet_eos_Seager2007.pdf"))
    #plt.show()
    plt.close(fig)

def plot_eos_WolfBower2018(data_file, data_folder, melting_data_folder, melting_data_files):
    """
    Plots the equation of state (EOS) data for different materials from Wolf & Bower (2018).
    Parameters:
    data_files (list): List of filenames containing the EOS data.
    data_folder (str): Path to the folder containing the data files.

    The function reads the EOS data from the specified files and plots the pressure-density relationship for each material.
    The data files should be tab-separated files with three columns: pressure (in Pa), temperature (in K), and density (in kg/m³).
    The function assumes that the data files are located in the specified data_folder.
    The function plots the data on a log-log scale.
    The melting curves from the specified melting_data_files are also plotted for reference.
    """
    filepath = os.path.join(data_folder, data_file)
    pressures, temps, densities = read_eos_WolfBower2018_data(filepath)

    pressures_gpa = pressures / 1e9 # Convert to GPa

    fig, ax = plt.subplots(figsize=(10, 6))

    for file in melting_data_files:
        melting_data = np.loadtxt(os.path.join(melting_data_folder, file), comments="#")
        melting_pressures = melting_data[:, 0] / 1e9  # in GPa
        melting_temps = melting_data[:, 1]  # in K
        ax.plot(melting_temps, melting_pressures, label=file.split('.')[0])

    sc = ax.scatter(temps, pressures_gpa, c=densities, cmap="viridis", s=12, alpha=0.85)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("Pressure (GPa)")
    ax.set_title("Equation of State (EOS) of MgSiO3 from Wolf & Bower (2018)")
    ax.set_xlim(0, temps.max())
    ax.set_ylim(0, pressures_gpa.max())
    ax.legend()
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Density (kg/m³)")
    if data_file == 'density_melt.dat':
        label = 'MgSiO3 melt'
    elif data_file == 'density_solid.dat':
        label = 'MgSiO3 solid'
    fig.savefig(os.path.join(ZALMOXIS_ROOT, "output_files", f"planet_eos_WolfBower2018_{label}.pdf"))
    #plt.show()
    plt.close(fig)

if __name__ == "__main__":
    # Example usage
    eos_data_files = ['eos_seager07_iron.txt', 'eos_seager07_silicate.txt', 'eos_seager07_water.txt']
    eos_data_folder = os.path.join(ZALMOXIS_ROOT, "data", "EOS_Seager2007")
    plot_eos_Seager2007(eos_data_files, eos_data_folder)

    wolf_bower_files = ['density_melt.dat', 'density_solid.dat']
    wolf_bower_folder = os.path.join(ZALMOXIS_ROOT, "data", "EOS_WolfBower2018_1TPa")
    melting_curve_files = ['liquidus.dat', 'solidus.dat']
    melting_curve_folder = os.path.join(ZALMOXIS_ROOT, "data", "melting_curves_Monteux-600")
    for wolf_bower_file in wolf_bower_files:
        plot_eos_WolfBower2018(wolf_bower_file, wolf_bower_folder, melting_curve_folder, melting_curve_files)
