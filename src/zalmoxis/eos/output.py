"""Output file creation for pressure and density profiles."""

from __future__ import annotations

import logging
import os

import numpy as np

from .. import get_zalmoxis_root

logger = logging.getLogger(__name__)


def create_pressure_density_files(
    outer_iter, inner_iter, pressure_iter, radii, pressure, density
):
    """Append pressure and density profiles to output files for a given iteration.

    Parameters
    ----------
    outer_iter : int
        Current outer iteration index.
    inner_iter : int
        Current inner iteration index.
    pressure_iter : int
        Current pressure iteration index.
    radii : numpy.ndarray
        Radial positions, in m.
    pressure : numpy.ndarray
        Pressure values corresponding to ``radii``, in Pa.
    density : numpy.ndarray
        Density values corresponding to ``radii``, in kg/m^3.

    Returns
    -------
    None
    """

    output_dir = os.path.join(get_zalmoxis_root(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    pressure_file = os.path.join(output_dir, 'pressure_profiles.txt')
    density_file = os.path.join(output_dir, 'density_profiles.txt')

    # Only delete the files once at the beginning of the run
    if outer_iter == 0 and inner_iter == 0 and pressure_iter == 0:
        for file_path in [pressure_file, density_file]:
            if os.path.exists(file_path):
                os.remove(file_path)

    # Append current iteration's pressure profile to file
    with open(pressure_file, 'a') as f:
        f.write(f'# Pressure iteration {pressure_iter}\n')
        np.savetxt(f, np.column_stack((radii, pressure)), header='radius pressure', comments='')
        f.write('\n')

    # Append current iteration's density profile to file
    with open(density_file, 'a') as f:
        f.write(f'# Pressure iteration {pressure_iter}\n')
        np.savetxt(f, np.column_stack((radii, density)), header='radius density', comments='')
        f.write('\n')
