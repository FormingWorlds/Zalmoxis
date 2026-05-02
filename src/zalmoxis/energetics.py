"""Gravitational energy and initial thermal state computation.

Computes the self-consistent initial temperature profile of a rocky
planet from its structure, following White & Li (2025, JGRP). The
initial CMB temperature is (White & Li Eq. 2):

    T_CMB = T_eq + Delta_T_G + Delta_T_D + Delta_T_ad

where Delta_T_G = f_a * U_u / (M * C) is the bulk heating from
accretion, Delta_T_D = f_d * (U_d - U_u) / (M * C) is the bulk
heating from core-mantle differentiation, and Delta_T_ad is the
adiabatic temperature increase from the surface to the CMB depth.
U_u = 3 G M^2 / (5 R) is the gravitational binding energy of the
undifferentiated planet, U_d is the binding energy of the
differentiated planet, and f_a, f_d are heat retention efficiencies.

The gravitational heating terms give the average temperature rise of
the whole planet. The adiabatic term corrects from the average to the
actual CMB temperature, which is hotter than the average because
adiabatic compression raises temperature at depth.

The post-accretion surface temperature T_surf_accr = T_eq + DT_G + DT_D
is the temperature at the top of the convecting mantle adiabat,
before atmospheric equilibration. PROTEUS uses this as the initial
condition; the atmosphere module then equilibrates the surface.

Boujibar et al. (2020) use a related but different framework: their
accretional energy is the surface gravitational potential GM/R per
unit mass (their Eq. 18), they adopt f = 0.04, and they ignore
differentiation (f_d = 0). Their Table 3 polynomial gives the T_CMB
at which the core starts crystallizing (a melting-curve intersection),
not the initial post-accretion temperature.

References
----------
White, N. I. & Li, J. (2025). JGRP, 130, e2024JE008550.
Boujibar, A., Driscoll, P. & Fei, Y. (2020). JGRP, 125, e2019JE006124.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np

from .constants import G

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
    # np.trapezoid (numpy >= 2.0) or np.trapz (numpy < 2.0)
    _trapz = getattr(np, 'trapezoid', np.trapz)
    U = _trapz(integrand, m)

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


def _gruneisen_adiabat_step(P_curr, P_next, T, gamma=1.3, K0=250e9, Kprime=4.0):
    """Single step of the Gruneisen adiabat: dT/dP = gamma * T / K(P).

    For linear K(P) = K0 + P * K', the exact solution is:
        T_next = T * (K(P_next) / K(P_curr))^(gamma / K')

    This is the formulation used by White & Li (2025) Eq. 6-7.
    It gives ~2-5 K/GPa for silicate mantle conditions.

    Default K0 = 250 GPa is a bulk lower-mantle average (perovskite
    K0=261 GPa from Fei+2021, post-perovskite K0=324 GPa from
    Sakai+2016, weighted toward the dominant perovskite phase).

    Parameters
    ----------
    P_curr, P_next : float
        Current and next pressure [Pa].
    T : float
        Current temperature [K].
    gamma : float
        Gruneisen parameter. Default 1.3 (typical silicate).
    K0 : float
        Bulk modulus at zero pressure [Pa]. Default 250 GPa.
    Kprime : float
        Pressure derivative of bulk modulus. Default 4.0.

    Returns
    -------
    float
        Temperature at P_next [K].
    """
    # Exact analytical solution for linear K(P) = K0 + P*K'
    K_curr = K0 + P_curr * Kprime
    K_next = K0 + P_next * Kprime
    return T * (K_next / K_curr) ** (gamma / Kprime)


def _integrate_adiabat(pressure_profile, T_anchor, nabla_ad_func):
    """Integrate the adiabatic gradient along a pressure profile.

    When nabla_ad_func is provided (e.g. from PALEOS), uses the
    log-stepping formula T_new = T * (P_new/P_old)^nabla_ad.

    When nabla_ad_func is None, uses the Gruneisen parameter
    formulation dT/dP = gamma * T / K(P) with default silicate
    parameters (gamma=1.3, K0=250 GPa, K'=4), following White & Li
    (2025) Eq. 6-7. This gives ~2-5 K/GPa, unlike constant
    nabla_ad = 0.3 which diverges over large pressure ranges.

    Parameters
    ----------
    pressure_profile : array-like
        Pressure values along the profile [Pa], in the direction of
        integration.
    T_anchor : float
        Temperature [K] at the starting end of the profile.
    nabla_ad_func : callable or None
        f(P [Pa], T [K]) -> nabla_ad (dimensionless). If None, uses
        Gruneisen parameter formulation.

    Returns
    -------
    float
        Temperature [K] at the far end of the profile.
    """
    P = np.asarray(pressure_profile, dtype=float)
    T = T_anchor

    if nabla_ad_func is not None:
        # PALEOS or user-provided nabla_ad: log-stepping with Gruneisen cap.
        # When nabla_ad_func returns its fallback (e.g. 0.3 outside table),
        # the log-stepping T*(P2/P1)^0.3 can diverge over large dP. Cap each
        # step so it never exceeds the Gruneisen prediction by more than 50%.
        for i in range(len(P) - 1):
            P_curr = max(float(P[i]), 1e3)
            P_next = max(float(P[i + 1]), 1e3)
            nad = nabla_ad_func(P_curr, T)
            T_nabla = T * (P_next / P_curr) ** nad
            T_grun = _gruneisen_adiabat_step(P_curr, P_next, T)
            # Use nabla_ad result unless it exceeds Gruneisen by >50%
            if abs(T_nabla - T) > 1.5 * abs(T_grun - T) + 1.0:
                T = T_grun
            else:
                T = T_nabla
    else:
        # Gruneisen parameter fallback (White+Li 2025 Eq. 6-7)
        for i in range(len(P) - 1):
            T = _gruneisen_adiabat_step(float(P[i]), float(P[i + 1]), T)

    return T


def initial_thermal_state(
    model_results: dict,
    core_mass_fraction: float,
    T_radiative_eq: float = 255.0,
    f_accretion: float = 0.04,
    f_differentiation: float = 0.50,
    C_iron: float = 450.0,
    C_silicate: float = 1250.0,
    iron_melting_func: callable | None = None,
    nabla_ad_func: callable | None = None,
    nabla_ad_iron_func: callable | None = None,
    cp_iron_func: callable | None = None,
    cp_silicate_func: callable | None = None,
) -> dict:
    """Compute the initial thermal state and temperature profile.

    Follows White & Li (2025) Eq. 2:
        T_CMB = T_eq + Delta_T_G + Delta_T_D + Delta_T_ad

    where Delta_T_G and Delta_T_D are the average bulk heating from
    accretion and differentiation, and Delta_T_ad is the adiabatic
    temperature increase from the surface to the CMB depth.

    Also computes the full adiabatic T(r) profile for initializing
    interior evolution solvers (SPIDER, Aragog). The profile is:
    - Core (r < R_CMB): adiabat inward from T_CMB using iron nabla_ad
      (or isothermal if nabla_ad_iron_func is None)
    - Mantle (r > R_CMB): adiabat outward from T_CMB using silicate
      nabla_ad (PALEOS or Gruneisen fallback)
    T_surf_accr = T_profile[-1] is the post-accretion surface
    temperature (top of the mantle adiabat), used as the initial
    condition for PROTEUS before atmospheric equilibration.

    Parameters
    ----------
    model_results : dict
        Output from ``zalmoxis.solver.main()``. Must contain keys
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
        Specific heat capacity of iron [J kg^-1 K^-1]. Default 450
        (Dulong-Petit, White & Li 2025). Used as constant fallback
        when ``cp_iron_func`` is None.
    C_silicate : float
        Specific heat capacity of silicate [J kg^-1 K^-1]. Default 1250
        (Dulong-Petit, White & Li 2025). Used as constant fallback
        when ``cp_silicate_func`` is None.
    iron_melting_func : callable or None
        Function f(P [Pa]) -> T_melt [K] for the iron melting curve.
        If None, uses ``iron_melting_anzellini13`` from
        ``zalmoxis.melting_curves``.
    nabla_ad_func : callable or None
        Function f(P [Pa], T [K]) -> nabla_ad for the mantle (silicate).
        If None, uses Gruneisen fallback (gamma=1.3, K0=250 GPa, K'=4).
    nabla_ad_iron_func : callable or None
        Function f(P [Pa], T [K]) -> nabla_ad for the core (iron).
        If None, the core is set to isothermal at T_CMB.
    cp_iron_func : callable or None
        Function f(P [Pa], T [K]) -> C_p [J kg^-1 K^-1] for iron.
        If provided, C_p is integrated over the core shells to compute
        a mass-weighted average instead of using the constant ``C_iron``.
    cp_silicate_func : callable or None
        Function f(P [Pa], T [K]) -> C_p [J kg^-1 K^-1] for silicate.
        If provided, C_p is integrated over the mantle shells.

    Returns
    -------
    dict
        Keys:
        - 'T_cmb' : float, CMB temperature [K]
        - 'T_surf_accr' : float, post-accretion surface temperature [K]
          (= T_eq + DT_G + DT_D = top of mantle adiabat). This is the
          initial condition for PROTEUS; the atmosphere module then
          equilibrates the surface with the interior.
        - 'T_profile' : ndarray, T(r) at each Zalmoxis grid point [K]
        - 'radii' : ndarray, radial grid [m] (from model_results)
        - 'pressure' : ndarray, pressure profile [Pa]
        - 'U_differentiated' : float, binding energy of real planet [J]
        - 'U_undifferentiated' : float, binding energy of uniform planet [J]
        - 'Delta_T_accretion' : float, temperature rise from accretion [K]
        - 'Delta_T_differentiation' : float, temperature rise from differentiation [K]
        - 'Delta_T_adiabat' : float, adiabatic T increase surface->CMB [K]
        - 'C_avg' : float, mass-weighted average specific heat [J kg^-1 K^-1]
        - 'C_iron_avg' : float, average iron C_p [J kg^-1 K^-1]
        - 'C_silicate_avg' : float, average silicate C_p [J kg^-1 K^-1]
        - 'core_state' : str, 'liquid', 'solid', 'partial', or 'none'
    """
    # Import default iron melting curve if none provided
    if iron_melting_func is None:
        from zalmoxis.melting_curves import iron_melting_anzellini13

        iron_melting_func = iron_melting_anzellini13

    if nabla_ad_func is None:
        warnings.warn(
            'No nabla_ad_func provided; using Gruneisen adiabat fallback '
            '(gamma=1.3, K0=250 GPa, K\'=4). Accurate at ~1 M_Earth; '
            'use PALEOS nabla_ad for super-Earths.',
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

    # Find CMB index (closest mass_enclosed to cmb_mass)
    cmb_mass = float(model_results['cmb_mass'])
    cmb_index = int(np.argmin(np.abs(mass_enclosed - cmb_mass)))

    # Mass-weighted average specific heat.
    # When cp_iron_func / cp_silicate_func are provided, integrate C_p(P, T)
    # over the radial shells weighted by shell mass. This accounts for the
    # pressure and temperature dependence of C_p from the EOS tables.
    # The temperature estimate uses a rough adiabat from a first-pass T_CMB.
    if cp_iron_func is not None or cp_silicate_func is not None:
        # First-pass T_CMB estimate using constant C_p
        C_avg_const = core_mass_fraction * C_iron + (1.0 - core_mass_fraction) * C_silicate
        _dT_G_est = f_accretion * U_u / (total_mass * C_avg_const)
        _dT_D_est = f_differentiation * differentiation_energy(U_d, U_u) / (total_mass * C_avg_const)
        # Include rough adiabatic correction for first-pass estimate
        P_mantle_est = pressure[cmb_index:]
        _dT_ad_est = _integrate_adiabat(
            P_mantle_est[::-1], T_radiative_eq + _dT_G_est + _dT_D_est, nabla_ad_func
        ) - (T_radiative_eq + _dT_G_est + _dT_D_est)
        T_cmb_est = T_radiative_eq + _dT_G_est + _dT_D_est + max(_dT_ad_est, 0.0)

        # Rough temperature profile for C_p evaluation:
        # Core: isothermal at T_cmb_est
        # Mantle: adiabatic from T_cmb_est to surface
        P_cmb_val = float(pressure[cmb_index])
        T_profile = np.full_like(radii, T_cmb_est)
        if P_cmb_val > 0:
            for i in range(cmb_index + 1, len(radii)):
                if nabla_ad_func is not None:
                    P_prev = max(float(pressure[i - 1]), 1e3)
                    P_i = max(float(pressure[i]), 1e3)
                    nad = nabla_ad_func(P_prev, T_profile[i - 1])
                    T_profile[i] = T_profile[i - 1] * (P_i / P_prev) ** nad
                else:
                    T_profile[i] = _gruneisen_adiabat_step(
                        float(pressure[i - 1]), float(pressure[i]), T_profile[i - 1]
                    )

        # Shell masses (dm = M[i+1] - M[i])
        dm = np.diff(mass_enclosed)

        # Integrate C_p * dm for core and mantle separately
        C_iron_sum, M_core_sum = 0.0, 0.0
        C_sil_sum, M_mantle_sum = 0.0, 0.0

        for i in range(len(dm)):
            P_i = max(float(pressure[i]), 1e3)
            T_i = float(T_profile[i])
            dm_i = float(dm[i])
            if dm_i <= 0:
                continue

            if i < cmb_index:
                # Core shell
                cp_i = cp_iron_func(P_i, T_i) if cp_iron_func is not None else C_iron
                C_iron_sum += cp_i * dm_i
                M_core_sum += dm_i
            else:
                # Mantle shell
                cp_i = cp_silicate_func(P_i, T_i) if cp_silicate_func is not None else C_silicate
                C_sil_sum += cp_i * dm_i
                M_mantle_sum += dm_i

        C_iron_avg = C_iron_sum / M_core_sum if M_core_sum > 0 else C_iron
        C_sil_avg = C_sil_sum / M_mantle_sum if M_mantle_sum > 0 else C_silicate
        C_avg = (C_iron_sum + C_sil_sum) / (M_core_sum + M_mantle_sum)

        logger.info(
            'Mass-weighted C_p: C_Fe_avg=%.0f, C_sil_avg=%.0f, C_avg=%.0f J/kg/K '
            '(T_cmb_est=%.0f K for T profile)',
            C_iron_avg, C_sil_avg, C_avg, T_cmb_est,
        )
    else:
        # Constant C_p (White+Li 2025 Dulong-Petit defaults)
        C_avg = core_mass_fraction * C_iron + (1.0 - core_mass_fraction) * C_silicate
        C_iron_avg = C_iron
        C_sil_avg = C_silicate

    # Temperature increments (White+Li 2025 Eq. 3, 5):
    # These represent the AVERAGE bulk heating of the entire planet.
    # Accretion heats the UNDIFFERENTIATED body (energy = U_u).
    # Differentiation releases ADDITIONAL energy (U_d - U_u) from core formation.
    Delta_T_G = f_accretion * U_u / (total_mass * C_avg)
    Delta_T_D = f_differentiation * differentiation_energy(U_d, U_u) / (total_mass * C_avg)

    # Surface temperature after accretion: the temperature at the top
    # of the convecting mantle adiabat, before atmospheric equilibration.
    # Equals T_eq + DT_G + DT_D (bulk gravitational heating).
    # PROTEUS uses this as the initial condition; the atmosphere module
    # then equilibrates the surface with the interior.
    T_surf_accr = T_radiative_eq + Delta_T_G + Delta_T_D

    # ── Compute T_CMB via adiabatic integration ──────────────────────
    # Integrate the mantle adiabat inward from T_surf_accr to get T_CMB.
    P_mantle = pressure[cmb_index:]  # CMB to surface (decreasing P)
    P_surface_to_cmb = P_mantle[::-1]  # surface to CMB (increasing P)

    T_at_cmb = _integrate_adiabat(P_surface_to_cmb, T_surf_accr, nabla_ad_func)
    Delta_T_ad = T_at_cmb - T_surf_accr

    if Delta_T_ad < 0:
        logger.warning(
            'Delta_T_ad = %.0f K (negative): adiabat integration produced '
            'unphysical result. Setting Delta_T_ad = 0.',
            Delta_T_ad,
        )
        Delta_T_ad = 0.0

    T_cmb = T_surf_accr + Delta_T_ad

    # ── Build full T(r) profile ──────────────────────────────────────
    # Mantle: integrate adiabat OUTWARD from T_CMB to surface.
    # Core: integrate adiabat INWARD from T_CMB to center (or isothermal).
    T_profile = np.full_like(radii, T_cmb)

    # Mantle adiabat: CMB -> surface (decreasing P, decreasing T)
    if cmb_index < len(radii) - 1:
        T = T_cmb
        for i in range(cmb_index, len(radii) - 1):
            P_curr = max(float(pressure[i]), 1e3)
            P_next = max(float(pressure[i + 1]), 1e3)
            if nabla_ad_func is not None:
                nad = nabla_ad_func(P_curr, T)
                T_nabla = T * (P_next / P_curr) ** nad
                T_grun = _gruneisen_adiabat_step(P_curr, P_next, T)
                if abs(T_nabla - T) > 1.5 * abs(T_grun - T) + 1.0:
                    T = T_grun
                else:
                    T = T_nabla
            else:
                T = _gruneisen_adiabat_step(P_curr, P_next, T)
            T_profile[i + 1] = T

    # Core adiabat: CMB -> center (increasing P inward, increasing T)
    if cmb_index > 0:
        if nabla_ad_iron_func is not None:
            T = T_cmb
            for i in range(cmb_index, 0, -1):
                P_curr = max(float(pressure[i]), 1e3)
                P_next = max(float(pressure[i - 1]), 1e3)
                nad = nabla_ad_iron_func(P_curr, T)
                T = T * (P_next / P_curr) ** nad
                T_profile[i - 1] = T
        # else: core stays isothermal at T_CMB (already set)

    # T at the top of the adiabat = T_surf_accr by construction:
    # the outward adiabat from T_CMB is the inverse of the inward integration.

    # ── Core state detection ─────────────────────────────────────────
    if core_mass_fraction <= 0:
        core_state = 'none'
    else:
        P_cmb = float(pressure[cmb_index])
        T_melt_cmb = iron_melting_func(P_cmb)

        if T_cmb > 1.05 * T_melt_cmb:
            core_state = 'liquid'
        elif T_cmb < 0.95 * T_melt_cmb:
            core_state = 'solid'
        else:
            core_state = 'partial'

    logger.info(
        f'Initial thermal state: T_CMB={T_cmb:.0f} K (DT_G={Delta_T_G:.0f}, '
        f'DT_D={Delta_T_D:.0f}, DT_ad={Delta_T_ad:.0f}), '
        f'T_surf_accr={T_surf_accr:.0f} K, '
        f'core_state={core_state}, U_d={U_d:.3e} J, U_u={U_u:.3e} J'
    )

    return {
        'T_cmb': T_cmb,
        'T_surf_accr': T_surf_accr,
        'T_profile': T_profile,
        'radii': radii,
        'pressure': pressure,
        'U_differentiated': U_d,
        'U_undifferentiated': U_u,
        'Delta_T_accretion': Delta_T_G,
        'Delta_T_differentiation': Delta_T_D,
        'Delta_T_adiabat': Delta_T_ad,
        'C_avg': C_avg,
        'C_iron_avg': C_iron_avg,
        'C_silicate_avg': C_sil_avg,
        'core_state': core_state,
    }
