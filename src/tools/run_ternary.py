from __future__ import annotations

import os

from zalmoxis.plots.plot_ternary import wrapper_ternary

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

def run_ternary_diagrams(target_mass_array):
    """
    Runs the ternary diagrams for a list of target planet masses.
    Parameters:
        target_mass_array (list of float): List of planet masses in Earth masses.
    """

    for id_mass in target_mass_array:
        wrapper_ternary(id_mass)

if __name__ == "__main__":
    run_ternary_diagrams(target_mass_array=[1, 5, 10, 50])
