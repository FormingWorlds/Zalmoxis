# This script generates a plot of the planet's internal structure, including density, gravity, pressure, and temperature profiles.
from __future__ import annotations

import os

import matplotlib.pyplot as plt

from ..constants import (
    earth_center_density,
    earth_center_pressure,
    earth_center_temperature,
    earth_cmb_pressure,
    earth_cmb_radius,
    earth_cmb_temperature,
    earth_mass,
    earth_radius,
    earth_surface_pressure,
    earth_surface_temperature,
)

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

def plot_planet_profile_single(radii, density, gravity, pressure, temperature, cmb_radius, cmb_mass, average_density, mass_enclosed, id_mass):
    """
    Generates a plot of the planet's internal structure, including density, gravity, pressure, temperature, and mass profiles.

    Args:
        radii (numpy.ndarray): Array of radial distances (m).
        density (numpy.ndarray): Array of densities (kg/m^3).
        gravity (numpy.ndarray): Array of gravitational accelerations (m/s^2).
        pressure (numpy.ndarray): Array of pressures (Pa).
        temperature (numpy.ndarray): Array of temperatures (K).
        cmb_radius (float): Radius of the core-mantle boundary (m).
        cmb_mass (float): Mass enclosed within the core-mantle boundary (kg).
        average_density (float): Average density of the planet (kg/m^3).
        mass_enclosed (numpy.ndarray): Array of enclosed masses (kg).
        id_mass (int): Identifier for the planet mass.
    """

    fig, ax = plt.subplots(1, 5, figsize=(16, 6))

    # Density vs. Radius
    ax[0].plot(radii / 1e3, density, color='b', lw=2, label=r'Model profile')
    ax[0].axvline(x=cmb_radius / 1e3, color='b', linestyle='--', label="Model CMB radius")
    ax[0].set_xlabel("Radius (km)")
    ax[0].set_ylabel(r'Density (kg/m$^3$)')
    ax[0].set_title("Model density structure")
    ax[0].grid()

    # Add average density as a vertical line
    ax[0].axhline(y=average_density, color='b', linestyle='-.', label=f"Model average density\n = {average_density:.0f} kg/m^3")

    # Gravity vs. Radius
    ax[1].plot(radii / 1e3, gravity, color='b', lw=2, label="Model")
    ax[1].set_xlabel("Radius (km)")
    ax[1].set_ylabel(r"Gravity (m/$s^2$)")
    ax[1].axvline(x=cmb_radius / 1e3, color='b', linestyle='--', label="Model CMB radius")
    ax[1].set_title("Model gravity structure")
    ax[1].grid()

    # Pressure vs. Radius
    ax[2].plot(radii / 1e3, pressure / 1e9, color='b', lw=2, label="Model")
    ax[2].set_xlabel("Radius (km)")
    ax[2].set_ylabel("Pressure (GPa)")
    ax[2].axvline(x=cmb_radius / 1e3, color='b', linestyle='--', label="Model CMB radius")
    ax[2].set_title("Model pressure structure")
    ax[2].grid()

    # Temperature vs. Radius
    ax[3].plot(radii / 1e3, temperature, color='b', lw=2, label="Model")
    ax[3].set_xlabel("Radius (km)")
    ax[3].set_ylabel("Temperature (K)")
    ax[3].axvline(x=cmb_radius / 1e3, color='b', linestyle='--', label="Model CMB radius")
    ax[3].set_title("Model temperature structure")
    ax[3].grid()

    # Mass vs. Radius
    ax[4].plot(radii / 1e3, mass_enclosed/earth_mass, color='b', lw=2, label="Model")
    ax[4].set_xlabel("Radius (km)")
    ax[4].set_ylabel(r"Mass enclosed (M$_\oplus$)")
    ax[4].axvline(x=cmb_radius / 1e3, color='b', linestyle='--', label="Model CMB radius")
    ax[4].set_title("Model mass enclosed structure")
    ax[4].grid()


    # Add reference Earth values to the plots
    ax[0].axvline(x=(earth_radius/1e3), color='g', linestyle=':', label="Earth Surface")
    ax[0].axvline(x=earth_cmb_radius / 1e3, color='g', linestyle='--', label="Earth CMB radius")
    ax[0].axhline(y=5515, color='g', linestyle='-.', label="Earth average density\n = 5515 kg/m^3")
    ax[0].axhline(y=earth_center_density, color='g', linestyle=':', label="Earth center density")

    ax[1].axhline(y=0, color='g', linestyle=':', label="Center gravity\n"+r"= 0 $m/s^2$")
    ax[1].axhline(y=9.81, color='g', linestyle='--', label="Earth surface gravity\n"+r"= 9.81 $m/s^2$")
    ax[1].axvline(x=earth_cmb_radius / 1e3, color='g', linestyle='--', label="Earth CMB")

    ax[2].axhline(y=earth_surface_pressure / 1e9, color='g', linestyle=':', label="Earth surface pressure")
    ax[2].axvline(x=earth_cmb_radius / 1e3, color='g', linestyle='--', label="Earth CMB radius")
    ax[2].axhline(y=earth_cmb_pressure / 1e9, color='g', linestyle='--', label="Earth CMB pressure")
    ax[2].axhline(y=earth_center_pressure / 1e9, color='g', linestyle='-.', label="Earth center pressure")

    ax[3].axhline(y=earth_surface_temperature , color='g', linestyle=':', label="Earth surface temperature")
    ax[3].axvline(x=earth_cmb_radius / 1e3, color='g', linestyle='--', label="Earth CMB radius")
    ax[3].axhline(y=earth_cmb_temperature , color='g', linestyle='-.', label="Earth CMB temperature")
    ax[3].axhline(y=earth_center_temperature , color='g', linestyle='--', label="Earth center temperature")

    ax[4].axvline(x=(earth_radius/1e3), color='g', linestyle=':', label="Earth Surface")
    ax[4].axvline(x=earth_cmb_radius / 1e3, color='g', linestyle='--', label="Earth CMB radius")
    ax[4].axhline(y=earth_mass/earth_mass, color='g', linestyle='-.', label="Earth mass")
    ax[4].axhline(y=cmb_mass/earth_mass, color='g', linestyle='--', label="Model CMB mass")

    # Add legends
    for a in ax:
        a.legend(fontsize=8)

    plt.tight_layout()
    if id_mass is None:
        plt.savefig(os.path.join(ZALMOXIS_ROOT, "output_files", "planet_profile.pdf"))
    #plt.show()
    plt.close(fig)
