"""Gravitational energy and initial thermal state computation.

Computes the self-consistent initial temperature profile of a rocky
planet from its structure, following White & Li (2025, JGRP) and
Boujibar et al. (2020, JGRP). The initial CMB temperature is:

    T_CMB = T_i + f_a * U_u / (M * C) + f_d * (U_d - U_u) / (M * C)

where U_u is the gravitational binding energy of the undifferentiated
(homogeneous) planet (= energy of accretion), U_d is the binding energy
of the differentiated planet (iron core + silicate mantle), and f_a, f_d
are heat retention efficiencies for accretion and differentiation.

References
----------
White, N. I. & Li, J. (2025). JGRP, 130, e2024JE008550.
Boujibar, A., Driscoll, P. & Fei, Y. (2020). JGRP, 125, e2019JE006124.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
from scipy.constants import G

logger = logging.getLogger(__name__)


def gravitational_binding_energy(radii, mass_enclosed):
    """Gravitational binding energy of a spherically symmetric body.

    Computes U = integral_0^M (G m / r) dm via trapezoidal integration
    on the Zalmoxis radial grid. The integrand is G * m(r) / r, and the
    integration variable is the enclosed mass m(r).

    Parameters
    ----------
    radii : array-like
        Radial grid from center to surface [m]. Shape (N,).
    mass_enclosed : array-like
        Enclosed mass at each radial point [kg]. Shape (N,).

    Returns
    -------
    float
        Gravitational binding energy [J], always positive.
    """
    radii = np.asarray(radii, dtype=float)
    mass_enclosed = np.asarray(mass_enclosed, dtype=float)

    if len(radii) != len(mass_enclosed):
        raise ValueError(
            f'radii and mass_enclosed must have the same length, '
            f'got {len(radii)} and {len(mass_enclosed)}.'
        )

    # Skip index 0 where r=0 (0/0 indeterminate limit)
    r = radii[1:]
    m = mass_enclosed[1:]

    # Integrand: G * m / r
    integrand = G * m / r

    # Trapezoidal integration over enclosed mass
    U = np.trapz(integrand, m)

    return float(abs(U))


def gravitational_binding_energy_uniform(total_mass, total_radius):
    """Gravitational binding energy of a uniform-density sphere.

    U = 3 G M^2 / (5 R)

    Parameters
    ----------
    total_mass : float
        Total mass [kg].
    total_radius : float
        Total radius [m].

    Returns
    -------
    float
        Gravitational binding energy [J], always positive.
    """
    return 3.0 * G * total_mass**2 / (5.0 * total_radius)


def differentiation_energy(U_differentiated, U_undifferentiated):
    """Energy released by differentiation (core formation).

    Delta_E = U_differentiated - U_undifferentiated. Positive when the
    differentiated body is more gravitationally bound (denser core sinks
    to center, releasing potential energy).

    Parameters
    ----------
    U_differentiated : float
        Binding energy of the differentiated (actual) body [J].
    U_undifferentiated : float
        Binding energy of the uniform-density body [J].

    Returns
    -------
    float
        Differentiation energy [J].
    """
    return U_differentiated - U_undifferentiated


def initial_thermal_state(
    model_results: dict,
    core_mass_fraction: float,
    T_radiative_eq: float = 255.0,
    f_accretion: float = 0.04,
    f_differentiation: float = 0.50,
    C_iron: float = 840.0,
    C_silicate: float = 1200.0,
    iron_melting_func: callable | None = None,
    nabla_ad_func: callable | None = None,
) -> dict:
    """Compute the initial thermal state of a rocky planet.

    Uses the gravitational binding energy from the converged structure
    model to estimate the post-accretion CMB temperature, surface
    temperature, and core state.

    Parameters
    ----------
    model_results : dict
        Output from `zalmoxis.solver.main()`. Must contain keys
        'radii', 'mass_enclosed', 'pressure', 'cmb_mass'.
    core_mass_fraction : float
        Core mass fraction (0 to 1).
    T_radiative_eq : float
        Radiative equilibrium temperature [K] (starting point before
        gravitational heating). Default 255 K.
    f_accretion : float
        Fraction of accretional gravitational energy retained as heat.
        Default 0.04 (White & Li 2025).
    f_differentiation : float
        Fraction of differentiation energy retained as heat.
        Default 0.50 (White & Li 2025).
    C_iron : float
        Specific heat capacity of iron [J kg^-1 K^-1]. Default 840.
    C_silicate : float
        Specific heat capacity of silicate [J kg^-1 K^-1]. Default 1200.
    iron_melting_func : callable or None
        Function f(P [Pa]) -> T_melt [K] for the iron melting curve.
        If None, uses ``iron_melting_anzellini13`` from
        ``zalmoxis.melting_curves``.
    nabla_ad_func : callable or None
        Function f(P [Pa], T [K]) -> nabla_ad (dimensionless adiabatic
        gradient d ln T / d ln P). If None, uses constant 0.3 with a
        warning.

    Returns
    -------
    dict
        Keys:
        - 'T_cmb' : float, CMB temperature [K]
        - 'T_surface' : float, surface temperature [K]
        - 'U_differentiated' : float, binding energy of real planet [J]
        - 'U_undifferentiated' : float, binding energy of uniform planet [J]
        - 'Delta_T_accretion' : float, temperature rise from accretion [K]
        - 'Delta_T_differentiation' : float, temperature rise from differentiation [K]
        - 'C_avg' : float, mass-weighted average specific heat [J kg^-1 K^-1]
        - 'core_state' : str, 'liquid', 'solid', or 'partial'
    """
    # Import default iron melting curve if none provided
    if iron_melting_func is None:
        from zalmoxis.melting_curves import iron_melting_anzellini13

        iron_melting_func = iron_melting_anzellini13

    if nabla_ad_func is None:
        warnings.warn(
            'No nabla_ad_func provided; using constant nabla_ad = 0.3. '
            'This is a rough approximation for silicate melt.',
            stacklevel=2,
        )

    # Extract structure profiles
    radii = np.asarray(model_results['radii'], dtype=float)
    mass_enclosed = np.asarray(model_results['mass_enclosed'], dtype=float)
    pressure = np.asarray(model_results['pressure'], dtype=float)

    total_mass = float(mass_enclosed[-1])
    total_radius = float(radii[-1])

    # Gravitational binding energies
    U_d = gravitational_binding_energy(radii, mass_enclosed)
    U_u = gravitational_binding_energy_uniform(total_mass, total_radius)

    # Mass-weighted average specific heat
    C_avg = core_mass_fraction * C_iron + (1.0 - core_mass_fraction) * C_silicate

    # Temperature increments (White+Li 2025 Eq. 3, 5):
    # Accretion heats the UNDIFFERENTIATED body (energy = U_u).
    # Differentiation releases ADDITIONAL energy (U_d - U_u) from core formation.
    Delta_T_G = f_accretion * U_u / (total_mass * C_avg)
    Delta_T_D = f_differentiation * differentiation_energy(U_d, U_u) / (total_mass * C_avg)

    # CMB temperature
    T_cmb = T_radiative_eq + Delta_T_G + Delta_T_D

    # Find CMB index (closest mass_enclosed to cmb_mass)
    cmb_mass = float(model_results['cmb_mass'])
    cmb_index = int(np.argmin(np.abs(mass_enclosed - cmb_mass)))

    # Compute surface temperature via adiabatic integration from CMB outward
    # Pressure decreases from CMB to surface
    P_mantle = pressure[cmb_index:]
    T = T_cmb
    for i in range(len(P_mantle) - 1):
        dP = P_mantle[i + 1] - P_mantle[i]  # negative (decreasing outward)
        if nabla_ad_func is not None:
            nad = nabla_ad_func(P_mantle[i], T)
        else:
            # Approximate nabla_ad for MgSiO3 melt at mid-mantle conditions.
            # Stixrude (2014) gives nabla_ad ~ 0.2-0.4 for liquid MgSiO3
            # across the 0-140 GPa range. 0.3 is a reasonable mid-range value.
            # For production runs, PROTEUS provides nabla_ad from the EOS tables.
            nad = 0.3
        if P_mantle[i] > 0:
            T = T + nad * T / P_mantle[i] * dP
    T_surface = T

    # Validate: surface must be cooler than CMB (adiabatic gradient is positive)
    if T_surface > T_cmb:
        logger.warning(
            'T_surface (%.0f K) > T_cmb (%.0f K): adiabatic integration '
            'produced unphysical result. Clamping T_surface to T_cmb.',
            T_surface, T_cmb,
        )
        T_surface = T_cmb

    # Determine core state from iron melting curve at CMB pressure
    P_cmb = float(pressure[cmb_index])
    T_melt_cmb = iron_melting_func(P_cmb)

    if T_cmb > 1.05 * T_melt_cmb:
        core_state = 'liquid'
    elif T_cmb < 0.95 * T_melt_cmb:
        core_state = 'solid'
    else:
        core_state = 'partial'

    logger.info(
        f'Initial thermal state: T_CMB={T_cmb:.0f} K, T_surface={T_surface:.0f} K, '
        f'core_state={core_state}, U_d={U_d:.3e} J, U_u={U_u:.3e} J'
    )

    return {
        'T_cmb': T_cmb,
        'T_surface': T_surface,
        'U_differentiated': U_d,
        'U_undifferentiated': U_u,
        'Delta_T_accretion': Delta_T_G,
        'Delta_T_differentiation': Delta_T_D,
        'C_avg': C_avg,
        'core_state': core_state,
    }
