from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

# Read the environment variable for get_zalmoxis_root()
from zalmoxis import get_zalmoxis_root


def plot_melting_curves(data_files, data_folder):
    fig, ax = plt.subplots(figsize=(10, 6))

    for file in data_files:
        data = np.loadtxt(os.path.join(data_folder, file), comments='#')
        pressures = data[:, 0] / 1e9  # in GPa
        temps = data[:, 1]  # in K
        ax.plot(temps, pressures, label=file.split('.')[0])

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (GPa)')
    ax.set_title('Melting Curves of MgSiO3 from Wolf & Bower (2018)')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    fig.savefig(os.path.join(get_zalmoxis_root(), 'output', 'melting_curves.pdf'))
    # plt.show()
    plt.close(fig)


if __name__ == '__main__':
    melting_curve_files = ['liquidus.dat', 'solidus.dat']
    melting_curve_folder = os.path.join(get_zalmoxis_root(), 'data', 'melting_curves_Monteux-600')
    plot_melting_curves(melting_curve_files, melting_curve_folder)
