"""Rose-Vinet equation of state for planetary interior materials.

Implements the 3rd-order Vinet (Rose-Vinet) EOS used by White & Li
(2025) and Boujibar et al. (2020) for computing density as a function
of pressure. The Vinet EOS relates pressure to the Eulerian strain
parameter f = (V/V_0)^(1/3):

    P(f) = 3 K_0 f^{-2} (1 - f) exp[eta (1 - f)]

where eta = 3/2 (K_0' - 1), K_0 is the zero-pressure isothermal bulk
modulus, and K_0' is its pressure derivative. Inversion to rho(P) uses
Brent root-finding.

Material parameters from:
    - Fe (solid): Smith et al. (2018), with 12.5% thermal+light-element
      density reduction following White & Li (2025).
    - Fe (liquid): Wicks et al. (2018), Fe-7Si alloy.
    - MgSiO3 perovskite: Fei et al. (2021), via Boujibar et al. (2020).
    - MgSiO3 post-perovskite: Sakai et al. (2016).
    - Peridotite: Stixrude & Lithgow-Bertelloni (2005).

References
----------
Boujibar, A., Driscoll, P. & Fei, Y. (2020). JGRP, 125, e2019JE006124.
White, N. I. & Li, J. (2025). JGRP, 130, e2024JE008550.
Smith, R. F. et al. (2018). Nature Astronomy, 2, 452-458.
Fei, Y. et al. (2021). Phys. Rev. Lett., 127, 080501.
Wicks, J. K. et al. (2018). Science Advances, 4, eaao5864.
Sakai, T. et al. (2016). Geophys. Res. Lett., 43, 7648-7656.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import brentq

logger = logging.getLogger(__name__)


# ── Material parameters ──────────────────────────────────────────────
# Each entry: (rho_0 [kg/m³], K_0 [Pa], K_0' [dimensionless],
#              thermal_correction [dimensionless, multiplied to rho])
#
# thermal_correction < 1 reduces density (thermal expansion, light elements).
# White+Li (2025) p.3: iron gets 12.5% reduction (fusion 2%, thermal 4%,
# light elements 6.5%). MgSiO3: thermal effects cancel (+3%, -3%).

VINET_MATERIALS: dict[str, dict] = {
    # Iron (solid, 300 K) - Boujibar+2020 Table 1 (citing Smith+2018)
    # Note: Boujibar's adopted values (8160/165/4.9) differ from
    # Smith+2018's own Vinet fit (8219/177.7/4.51).
    # With White+Li thermal+light-element correction (12.5% reduction)
    'iron': {
        'rho_0': 8160.0,  # kg/m³
        'K_0': 165.0e9,  # Pa
        'K_prime': 4.9,
        'thermal_correction': 0.875,  # (1 - 0.125)
        'description': 'Fe epsilon (Boujibar+2020 Table 1), '
        '12.5% thermal correction (White+Li 2025)',
    },
    # Iron (solid, 300 K) - Smith+2018 original Vinet fit
    # From Smith et al. (2018) Nature Astronomy 2, 452-458
    # These are the parameters White+Li likely reference directly.
    'iron_smith2018': {
        'rho_0': 8219.0,  # kg/m³
        'K_0': 177.7e9,  # Pa
        'K_prime': 4.51,
        'thermal_correction': 0.875,
        'description': 'Fe epsilon (Smith+2018 Vinet fit), '
        '12.5% thermal correction (White+Li 2025)',
    },
    # Iron - White+Li (2025) code (github.com/wnate1373/superEarth)
    # Uses ambient iron density (7874), K_0=162.5, K'=5.5 with 12.5% correction.
    # These are the EXACT parameters from their dif_constR.m.
    'iron_whiteli': {
        'rho_0': 7874.0,  # kg/m³ (ambient alpha-iron)
        'K_0': 162.5e9,  # Pa
        'K_prime': 5.5,
        'thermal_correction': 0.875,
        'description': 'Fe (White+Li 2025 code, dif_constR.m)',
    },
    # MgSiO3 - White+Li (2025) code (differentiated model)
    # rho_0=4103, K_0=265.5, K'=4.16 from their dif_constR.m
    'MgSiO3_whiteli': {
        'rho_0': 4103.0,
        'K_0': 265.5e9,
        'K_prime': 4.16,
        'thermal_correction': 1.0,
        'description': 'MgSiO3 (White+Li 2025 code, dif_constR.m)',
    },
    # Iron (liquid, Fe-7Si alloy) - Wicks+2018
    'iron_liquid': {
        'rho_0': 7700.0,
        'K_0': 125.0e9,
        'K_prime': 5.5,
        'thermal_correction': 1.0,  # already liquid properties
        'description': 'Fe-7Si liquid (Wicks+2018)',
    },
    # MgSiO3 perovskite (bridgmanite) - Fei+2021 (Nature Comm. 12:876)
    # Optimized Vinet parameters from shock compression to 1254 GPa.
    # These are the parameters White+Li (2025) use for their structure model.
    'MgSiO3': {
        'rho_0': 4176.0,  # kg/m³ (Fei+2021 p.5)
        'K_0': 265.5e9,  # Pa (Fei+2021 p.5)
        'K_prime': 4.16,  # (Fei+2021 p.5)
        'thermal_correction': 1.0,  # thermal effects cancel (White+Li p.3)
        'description': 'MgSiO3 bridgmanite (Fei+2021, shock to 1254 GPa)',
    },
    # MgSiO3 perovskite - Boujibar+2020 Table 1 (pre-Fei+2021 values)
    # From Dorfman+2013 and Lundin+2008, used in Boujibar's structure model.
    'MgSiO3_boujibar': {
        'rho_0': 4109.0,
        'K_0': 261.0e9,
        'K_prime': 4.0,
        'thermal_correction': 1.0,
        'description': 'MgSiO3 perovskite (Boujibar+2020 Table 1, Dorfman+2013/Lundin+2008)',
    },
    # MgSiO3 post-perovskite - Sakai+2016 / Boujibar+2020 Table 1
    'MgSiO3_ppv': {
        'rho_0': 4260.0,
        'K_0': 324.0e9,
        'K_prime': 3.3,
        'thermal_correction': 1.0,
        'description': 'MgSiO3 post-perovskite (Sakai+2016)',
    },
    # Peridotite (upper mantle) - Stixrude & Lithgow-Bertelloni 2005
    'peridotite': {
        'rho_0': 3226.0,
        'K_0': 128.0e9,
        'K_prime': 4.2,
        'thermal_correction': 1.0,
        'description': 'Peridotite (Stixrude+2005)',
    },
}

VALID_VINET_KEYS: set[str] = set(VINET_MATERIALS.keys())

# Maximum pressure for root-finding bracket [Pa]
P_MAX_VINET = 1e16  # 10,000 TPa


def _vinet_pressure(f, K_0, eta):
    """Vinet EOS: pressure as a function of strain parameter f.

    Parameters
    ----------
    f : float
        Vinet linear compression parameter, f = (V/V_0)^(1/3) = (rho_0/rho)^(1/3).
        f = 1 at zero pressure, f -> 0 at infinite compression.
    K_0 : float
        Zero-pressure isothermal bulk modulus [Pa].
    eta : float
        Vinet parameter eta = 3/2 (K_0' - 1).

    Returns
    -------
    float
        Pressure [Pa].
    """
    if f >= 1.0:
        return 0.0
    if f <= 0.0:
        return np.inf
    return 3.0 * K_0 * f ** (-2) * (1.0 - f) * np.exp(eta * (1.0 - f))


def get_vinet_density(pressure: float, material_key: str) -> float | None:
    """Compute density from the Rose-Vinet EOS.

    Parameters
    ----------
    pressure : float
        Pressure in Pa.
    material_key : str
        One of the keys in VINET_MATERIALS (e.g. 'iron', 'MgSiO3').

    Returns
    -------
    float or None
        Density in kg/m³, or None for invalid input.

    Raises
    ------
    ValueError
        If material_key is not recognized.
    """
    if material_key not in VINET_MATERIALS:
        raise ValueError(
            f"Unknown Vinet material '{material_key}'. Valid keys: {sorted(VALID_VINET_KEYS)}"
        )

    if np.isnan(pressure):
        return None

    mat = VINET_MATERIALS[material_key]
    rho_0 = mat['rho_0']
    K_0 = mat['K_0']
    K_prime = mat['K_prime']
    thermal = mat['thermal_correction']

    if pressure <= 0:
        return float(rho_0 * thermal)

    if pressure > P_MAX_VINET:
        logger.warning(
            'Pressure %.2e Pa exceeds Vinet validity limit %.2e Pa for material %s. Clamping.',
            pressure,
            P_MAX_VINET,
            material_key,
        )
        pressure = P_MAX_VINET

    eta = 1.5 * (K_prime - 1.0)

    # Find f such that P_vinet(f) = pressure, using Brent root-finding.
    # f = 1 at P = 0, f -> 0 at P -> inf.
    # Search bracket: f in [f_min, 1 - epsilon]
    f_min = 0.01  # extreme compression (rho ~ 1e6 * rho_0)

    def residual(f):
        return _vinet_pressure(f, K_0, eta) - pressure

    # Check bracket validity
    if residual(1.0 - 1e-12) > 0:
        # P_target < P(f~1), which means P ~ 0
        return float(rho_0 * thermal)

    if residual(f_min) < 0:
        # P_target > P(f_min), need even more compression
        f_min = 0.001

    try:
        f_solution = brentq(residual, f_min, 1.0 - 1e-12, xtol=1e-14, rtol=1e-12)
    except ValueError:
        logger.warning(
            'Vinet root-finding failed for P=%.2e Pa, material=%s. Bracket: [%.4e, %.4e].',
            pressure,
            material_key,
            residual(f_min),
            residual(1.0 - 1e-12),
        )
        return None

    # rho = rho_0 / f^3, with thermal correction
    density = rho_0 / f_solution**3 * thermal

    return float(density)
