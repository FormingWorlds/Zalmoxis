"""
Analytic EOS from Seager et al. (2007), Table 3, Eq. 11.

Modified polytropic fit: rho(P) = rho_0 + c * P^n
Valid for P < 1e16 Pa. Approximates the full merged Vinet/BME + TFD EOS
to 2-12% accuracy for all planetary pressures.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Maximum pressure for which the analytic fit is valid (Pa)
P_MAX = 1e16

# Seager et al. (2007) Table 3: modified polytropic EOS parameters
# Each entry: (rho_0 [kg/m^3], c [kg m^-3 Pa^-n], n [dimensionless])
SEAGER2007_MATERIALS: dict[str, tuple[float, float, float]] = {
    'iron': (8300.0, 0.00349, 0.528),  # Fe (epsilon)
    'MgSiO3': (4100.0, 0.00161, 0.541),  # MgSiO3 perovskite
    'MgFeSiO3': (4260.0, 0.00127, 0.549),  # (Mg,Fe)SiO3
    'H2O': (1460.0, 0.00311, 0.513),  # Water ice (phases VII, VIII, X)
    'graphite': (2250.0, 0.00350, 0.514),  # C (graphite)
    'SiC': (3220.0, 0.00172, 0.537),  # Silicon carbide
}

VALID_MATERIAL_KEYS: set[str] = set(SEAGER2007_MATERIALS.keys())


def get_analytic_density(pressure: float, material_key: str) -> float | None:
    """
    Compute density from the Seager et al. (2007) analytic modified polytrope.

    Parameters
    ----------
    pressure : float
        Pressure in Pa.
    material_key : str
        One of the keys in SEAGER2007_MATERIALS
        (e.g., "iron", "MgSiO3", "H2O", "graphite", "SiC", "MgFeSiO3").

    Returns
    -------
    float
        Density in kg/m^3. Returns rho_0 for non-positive pressure (allows
        the ODE solver to remain stable during pressure adjustment iterations).
        Returns None only for NaN pressure.

    Raises
    ------
    ValueError
        If material_key is not recognized.
    """
    if material_key not in SEAGER2007_MATERIALS:
        raise ValueError(
            f"Unknown material key '{material_key}'. Valid keys: {sorted(VALID_MATERIAL_KEYS)}"
        )

    rho_0, c, n = SEAGER2007_MATERIALS[material_key]

    # Guard against nonphysical pressure
    if np.isnan(pressure):
        return None
    if pressure <= 0:
        return float(rho_0)

    if pressure > P_MAX:
        logger.warning(
            f'Pressure {pressure:.2e} Pa exceeds validity limit of {P_MAX:.0e} Pa '
            f'for Seager+2007 analytic EOS. Results may be inaccurate.'
        )

    density = rho_0 + c * pressure**n

    return float(density)
