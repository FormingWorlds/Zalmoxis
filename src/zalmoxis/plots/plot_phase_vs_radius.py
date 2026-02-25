from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')


def plot_PT_with_phases(pressure, temperature, radii, mantle_phases, cmb_radius):
    """
    Plot pressure–temperature profile highlighting solid/mixed/melted mantle regions.
    Parameters:
    pressure (array-like): Pressure profile in Pa.
    temperature (array-like): Temperature profile in K.
    radii (array-like): Radial positions corresponding to the pressure and temperature profiles in meters.
    mantle_phases (array-like): List of mantle phases ("solid_mantle", "mixed_mantle", "melted_mantle") corresponding to each radial position.
    cmb_radius (float or None): Radius of the core-mantle boundary in meters. If None, CMB line is not plotted.
    """

    # Convert pressure to GPa and temperature to kK for readability
    P = np.array(pressure) * 1e-9  # in GPa
    T = np.array(temperature) / 1e3  # in kK

    # Create masks for each phase
    mask_solid = np.array(mantle_phases) == 'solid_mantle'
    mask_mixed = np.array(mantle_phases) == 'mixed_mantle'
    mask_melted = np.array(mantle_phases) == 'melted_mantle'

    # Plot setup
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(T[mask_solid], P[mask_solid], color='black', lw=2, ls='solid', label='Solid mantle')
    ax.plot(
        T[mask_mixed], P[mask_mixed], color='orange', lw=2, ls='dashed', label='Mixed mantle'
    )
    ax.plot(
        T[mask_melted], P[mask_melted], color='red', lw=2, ls='dotted', label='Melted mantle'
    )
    if cmb_radius is not None:
        cmb_P = P[np.argmax(radii >= cmb_radius)]
        ax.axhline(cmb_P, color='k', linestyle='--', alpha=0.6, label='CMB')
    ax.set_xlabel('Temperature (1000 K)')
    ax.set_ylabel('Pressure (GPa)')
    ax.set_title('Mantle P–T Profile with Phase Transitions')
    ax.invert_yaxis()
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(ZALMOXIS_ROOT, 'output_files', 'mantle_PT_profile.pdf'))
    # plt.show()
