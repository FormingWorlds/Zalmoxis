# This script generates a plot of the planet's internal structure, including density, gravity, pressure, and temperature profiles.

import matplotlib.pyplot as plt
import numpy as np
from ..constants import *

def plot_planet_profile_single(radii, density, gravity, pressure, temperature, cmb_radius, average_density, output_filename="planet_profile.png"):
    """
    Generates a plot of the planet's internal structure, including density, 
    gravity, pressure, and temperature profiles.

    Args:
        radii (numpy.ndarray): Array of radial distances (m).
        density (numpy.ndarray): Array of densities (kg/m^3).
        gravity (numpy.ndarray): Array of gravitational accelerations (m/s^2).
        pressure (numpy.ndarray): Array of pressures (Pa).
        temperature (numpy.ndarray): Array of temperatures (K).
        cmb_radius (float): Radius of the core-mantle boundary (m).
        average_density (float): Average density of the planet (kg/m^3).
        output_filename (str, optional): Name of the output file. Defaults to "planet_profile.png".
    """

    fig, ax = plt.subplots(1, 4, figsize=(16, 6))

    # Density vs. Radius
    ax[0].plot(density, radii / 1e3, color='b', lw=2, label=r'Model profile')
    ax[0].axhline(y=cmb_radius / 1e3, color='b', linestyle='--', label="Model CMB")
    ax[0].set_xlabel(r'Density (kg/m$^3$)')
    ax[0].set_ylabel("Radius (km)")
    ax[0].set_title("Model density structure")
    ax[0].grid()

    # Add average density as a vertical line
    ax[0].axvline(x=average_density, color='b', linestyle='-.', label=f"Model average density\n = {average_density:.0f} kg/m^3")

    # Gravity vs. Radius
    ax[1].plot(gravity, radii / 1e3, color='b', lw=2, label="Model")
    ax[1].set_xlabel(r"Gravity (m/$s^2$)")
    ax[1].set_ylabel("Radius (km)")
    ax[1].axhline(y=cmb_radius / 1e3, color='b', linestyle='--', label="Model CMB radius")
    ax[1].set_title("Model gravity structure")
    ax[1].grid()

    # Pressure vs. Radius
    ax[2].plot(pressure / 1e9, radii / 1e3, color='b', lw=2, label="Model")
    ax[2].set_xlabel("Pressure (GPa)")
    ax[2].set_ylabel("Radius (km)")
    ax[2].axhline(y=cmb_radius / 1e3, color='b', linestyle='--', label="Model CMB radius")
    ax[2].set_title("Model pressure structure")
    ax[2].grid()

    # Temperature vs. Radius
    ax[3].plot(temperature, radii / 1e3, color='b', lw=2, label="Model")
    ax[3].set_xlabel("Temperature (K)")
    ax[3].set_ylabel("Radius (km)")
    ax[3].axhline(y=cmb_radius / 1e3, color='b', linestyle='--', label="Model CMB radius")
    ax[3].set_title("Model temperature structure")
    ax[3].grid()


    # Add reference Earth values to the plots
    ax[0].axhline(y=(earth_radius/1e3), color='g', linestyle=':', label=f"Earth Surface")
    ax[0].axhline(y=earth_cmb_radius / 1e3, color='g', linestyle='--', label="Earth CMB radius")
    ax[0].axvline(x=5515, color='g', linestyle='-.', label=f"Earth average density\n = 5515 kg/m^3")
    ax[0].axvline(x=earth_center_density, color='g', linestyle=':', label="Earth center density")

    ax[1].axvline(x=0, color='g', linestyle=':', label="Center gravity\n"+r"= 0 $m/s^2$")
    ax[1].axvline(x=9.81, color='g', linestyle='--', label="Earth surface gravity\n"+r"= 9.81 $m/s^2$")
    ax[1].axhline(y=earth_cmb_radius / 1e3, color='g', linestyle='--', label="Earth CMB")

    ax[2].axvline(x=earth_surface_pressure / 1e9, color='g', linestyle=':', label="Earth surface pressure")
    ax[2].axhline(y=earth_cmb_radius / 1e3, color='g', linestyle='--', label="Earth CMB radius")
    ax[2].axvline(x=earth_cmb_pressure / 1e9, color='g', linestyle='--', label="Earth CMB pressure")
    ax[2].axvline(x=earth_center_pressure / 1e9, color='g', linestyle='-.', label="Earth center pressure")

    ax[3].axvline(x=earth_surface_temperature , color='g', linestyle=':', label="Earth surface temperature")
    ax[3].axhline(y=earth_cmb_radius / 1e3, color='g', linestyle='--', label="Earth CMB radius")
    ax[3].axvline(x=earth_cmb_temperature , color='g', linestyle='-.', label="Earth CMB temperature")
    ax[3].axvline(x=earth_center_temperature , color='g', linestyle='--', label="Earth center temperature")

    # Add legends
    for a in ax:
        a.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("planet_profile_develop.png")
