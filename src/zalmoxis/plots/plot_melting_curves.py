from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from zalmoxis.eos_functions import load_melting_curves

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

def plot_melting_curves(data_files, data_folder):

    fig, ax = plt.subplots(figsize=(10, 6))

    for file in data_files:
        pressures, temps = load_melting_curves(os.path.join(data_folder, file))
        ax.plot(pressures / 1e9, temps, label=file.split('.')[0])

    ax.set_xlabel('Pressure (GPa)')
    ax.set_ylabel('Temperature (K)')
    ax.set_title('Melting Curves of MgSiO3 from Wolf & Bower (2018)')
    ax.legend()
    fig.savefig(os.path.join(ZALMOXIS_ROOT, "output_files", "melting_curves.pdf"))
    #plt.show()
    plt.close(fig)

if __name__ == "__main__":  
    melting_curve_files = ['liquidus.dat', 'solidus.dat']
    melting_curve_folder = os.path.join(ZALMOXIS_ROOT, "data", "melting_curves_WolfBower2018")
    plot_melting_curves(melting_curve_files, melting_curve_folder)