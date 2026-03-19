"""Binodal (miscibility) models for H2-silicate and H2-H2O systems.

Provides thermodynamic suppression weights for multi-component mixtures
involving molecular hydrogen. Two independent binodal models determine
whether H2 is miscible with its partner material at given (P, T, x):

- **Rogers+2025** (MNRAS 544, 3496): H2-MgSiO3 miscibility boundary.
  Analytic fit to the binodal temperature T_b(x_H2, P) from Eqs. A1-A11.

- **Gupta+2025** (ApJL 982, L35): H2-H2O miscibility from a Gibbs free
  energy model with asymmetric Margules mixing (Eqs. A2-A9).

Both models return a smooth sigmoid weight in [0, 1]:
- ~1 above the binodal (miscible, H2 participates in the harmonic mean)
- ~0 below the binodal (immiscible, H2 is suppressed)

This module has no dependencies on the rest of Zalmoxis (pure thermodynamics).

References
----------
Rogers, Young & Schlichting (2025), MNRAS 544, 3496.
    H2-MgSiO3 miscibility: doi:10.1093/mnras/stae2268

Gupta, Kovacevic, Mazevet (2025), ApJL 982, L35.
    H2-H2O miscibility: doi:10.3847/2041-8213/adb8f5

Gilmore & Stixrude (2025), Nature 650, 60.
    DFT-MD source for H2-MgSiO3 Gibbs parameters.

Stixrude & Gilmore (2025), Icarus 432, 116401.
    Margules parameters for H2-MgSiO3.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

# ── Molar masses ─────────────────────────────────────────────────────
MU_H2 = 2.016e-3  # kg/mol
MU_MGSIO3 = 100.39e-3  # kg/mol
MU_H2O = 18.015e-3  # kg/mol

# ── Universal gas constant ───────────────────────────────────────────
R_GAS = 8.314462  # J/(mol K)


# ═════════════════════════════════════════════════════════════════════
# Shared utilities
# ═════════════════════════════════════════════════════════════════════


def mass_to_mole_fraction(w_1, w_2, mu_1, mu_2):
    """Convert mass fractions to mole fraction of component 1.

    Parameters
    ----------
    w_1 : float
        Mass fraction of component 1.
    w_2 : float
        Mass fraction of component 2.
    mu_1 : float
        Molar mass of component 1 (kg/mol).
    mu_2 : float
        Molar mass of component 2 (kg/mol).

    Returns
    -------
    float
        Mole fraction of component 1.
    """
    n1 = w_1 / mu_1
    n2 = w_2 / mu_2
    total = n1 + n2
    if total <= 0:
        return 0.0
    return n1 / total


# ═════════════════════════════════════════════════════════════════════
# Rogers+2025: H2-MgSiO3 binodal (MNRAS 544, 3496, Eqs. A1-A11)
# ═════════════════════════════════════════════════════════════════════

# Critical mole fraction (Eq. A4)
_R25_XC = 0.73913

# T_c parameters (Eq. A3): T_c = E * (1 - P/D), where P in GPa
_R25_E = 4223.0  # K
_R25_D = -35.0  # GPa (negative, so T_c increases as P decreases)

# Generalized logistic parameters for ascending branch (x < x_c)
# Eq. A7: T_asc(x) = T_c / (1 + alpha_2 * exp(-alpha_3 * (x - alpha_4)))^(1/alpha_5)
_R25_ALPHA_2 = 1.40299
_R25_ALPHA_3 = 9.75751
_R25_ALPHA_4 = 0.42587
_R25_ALPHA_5 = 2.72591

# Generalized logistic parameters for descending branch (x > x_c)
# Eq. A10: T_desc(x) = T_c / (1 + beta_2 * exp(-beta_3 * (x - beta_4)))^(1/beta_5)
_R25_BETA_2 = 0.02653
_R25_BETA_3 = -39.51206
_R25_BETA_4 = 0.85862
_R25_BETA_5 = 1.14413


def _rogers2025_T_critical(P_GPa):
    """Critical temperature for H2-MgSiO3 binodal at pressure P.

    Parameters
    ----------
    P_GPa : float
        Pressure in GPa.

    Returns
    -------
    float
        Critical temperature in K. May be negative at P > 35 GPa
        (always miscible regime).
    """
    return _R25_E * (1.0 + P_GPa / _R25_D)


def rogers2025_binodal_temperature(x_H2, P_GPa):
    """Analytic binodal temperature for the H2-MgSiO3 system.

    Piecewise generalized logistic fit from Rogers+2025 Eqs. A1-A11.

    Parameters
    ----------
    x_H2 : float
        H2 mole fraction in the H2-MgSiO3 binary system.
    P_GPa : float
        Pressure in GPa.

    Returns
    -------
    float
        Binodal temperature in K. Returns 0 for x_H2 <= 0 or x_H2 >= 1.
    """
    if x_H2 <= 0.0 or x_H2 >= 1.0:
        return 0.0

    T_c = _rogers2025_T_critical(P_GPa)
    if T_c <= 0:
        # At P > 35 GPa, T_c < 0: always miscible
        return 0.0

    if x_H2 <= _R25_XC:
        # Ascending branch (Eq. A7)
        arg = _R25_ALPHA_3 * (x_H2 - _R25_ALPHA_4)
        # Clamp to prevent overflow
        arg = max(min(arg, 500.0), -500.0)
        denom = (1.0 + _R25_ALPHA_2 * np.exp(-arg)) ** (1.0 / _R25_ALPHA_5)
        return T_c / denom
    else:
        # Descending branch (Eq. A10)
        arg = _R25_BETA_3 * (x_H2 - _R25_BETA_4)
        arg = max(min(arg, 500.0), -500.0)
        denom = (1.0 + _R25_BETA_2 * np.exp(-arg)) ** (1.0 / _R25_BETA_5)
        return T_c / denom


def rogers2025_suppression_weight(P_Pa, T_K, w_H2, w_sil, T_scale=50.0):
    """Sigmoid suppression weight for H2-MgSiO3 miscibility.

    Returns ~1 when the system is above the binodal (H2 miscible with
    silicate) and ~0 when below (H2 immiscible, should be suppressed).

    Parameters
    ----------
    P_Pa : float
        Pressure in Pa.
    T_K : float
        Temperature in K.
    w_H2 : float
        Mass fraction of H2 in the binary system.
    w_sil : float
        Mass fraction of silicate (MgSiO3) in the binary system.
    T_scale : float
        Sigmoid width in K. Default 50 K gives a ~200 K transition zone.

    Returns
    -------
    float
        Weight in [0, 1].
    """
    if w_H2 <= 0 or w_sil <= 0:
        return 1.0

    x_H2 = mass_to_mole_fraction(w_H2, w_sil, MU_H2, MU_MGSIO3)
    P_GPa = P_Pa * 1e-9

    T_binodal = rogers2025_binodal_temperature(x_H2, P_GPa)

    if T_binodal <= 0:
        # No binodal (pure endmember or P > 35 GPa): always miscible
        return 1.0

    if T_scale <= 0:
        # Hard cutoff
        return 1.0 if T_K >= T_binodal else 0.0

    arg = (T_K - T_binodal) / T_scale
    arg = max(min(arg, 500.0), -500.0)
    return 1.0 / (1.0 + np.exp(-arg))


# ═════════════════════════════════════════════════════════════════════
# Gupta+2025: H2-H2O binodal (ApJL 982, L35, Eqs. A2-A9)
# ═════════════════════════════════════════════════════════════════════

# Parameters from Table 1 (median values of posterior)
_G25_W_H = -599.08  # J/mol, enthalpy of mixing
_G25_WV1 = -26.12  # J/(mol GPa), volume of mixing (linear in P)
_G25_WV2 = 981.78  # J/(mol GPa K^2), volume of mixing (T-dependent)
_G25_W_S = -16.08  # J/(mol K), entropy of mixing
_G25_LAMBDA1 = 2.62  # asymmetry parameter (constant)
_G25_LAMBDA2 = -0.68  # asymmetry parameter (T-dependent)
_G25_T0 = 1000.0  # K, reference temperature


def _gupta2025_W_V(T):
    """T-dependent volume of mixing parameter (Eq. A5).

    W_V = W_{V,1} + W_{V,2} / (T/T_0)^2

    Parameters
    ----------
    T : float
        Temperature in K.

    Returns
    -------
    float
        W_V in J/(mol GPa).
    """
    ratio = T / _G25_T0
    return _G25_WV1 + _G25_WV2 / (ratio * ratio)


def _gupta2025_lambda(T):
    """T-dependent asymmetry parameter (Eq. A6).

    lambda = lambda_1 + lambda_2 / (T/T_0)^2

    Parameters
    ----------
    T : float
        Temperature in K.

    Returns
    -------
    float
        Asymmetry parameter (dimensionless).
    """
    ratio = T / _G25_T0
    return _G25_LAMBDA1 + _G25_LAMBDA2 / (ratio * ratio)


def _gupta2025_W(T, P_GPa):
    """Margules interaction parameter W (Eq. A4).

    W = W_H - T * W_S + P * W_V(T)

    Parameters
    ----------
    T : float
        Temperature in K.
    P_GPa : float
        Pressure in GPa.

    Returns
    -------
    float
        W in J/mol.
    """
    return _G25_W_H - T * _G25_W_S + P_GPa * _gupta2025_W_V(T)


def gupta2025_gibbs_mixing(x_H2, T, P_GPa):
    """Gibbs free energy of H2-H2O mixing per mole (Eq. A2).

    G_mix = RT [y ln(y) + (1-y) ln(1-y)] + W * y * (1-y)

    where y = x / (x + lambda * (1-x)) is the transformed composition
    (Eq. A3) accounting for asymmetric mixing.

    Parameters
    ----------
    x_H2 : float
        H2 mole fraction.
    T : float
        Temperature in K.
    P_GPa : float
        Pressure in GPa.

    Returns
    -------
    float
        Gibbs free energy of mixing in J/mol.
    """
    if x_H2 <= 0 or x_H2 >= 1:
        return 0.0

    lam = _gupta2025_lambda(T)
    # Transformed composition (Eq. A3)
    denom = x_H2 + lam * (1.0 - x_H2)
    if denom <= 0:
        return 0.0
    y = x_H2 / denom

    if y <= 0 or y >= 1:
        return 0.0

    W = _gupta2025_W(T, P_GPa)

    # Ideal mixing + asymmetric Margules
    G = R_GAS * T * (y * np.log(y) + (1.0 - y) * np.log(1.0 - y)) + W * y * (1.0 - y)
    return G


def gupta2025_critical_pressure(T):
    """Critical pressure at temperature T (Eq. A8).

    P_c = [2RT - (W_H - T * W_S)] / W_V(T)

    Above this pressure, the system is in a single phase for all compositions.

    Parameters
    ----------
    T : float
        Temperature in K.

    Returns
    -------
    float
        Critical pressure in GPa. May be negative (system is always
        single-phase at this T for all P > 0).
    """
    W_V = _gupta2025_W_V(T)
    if abs(W_V) < 1e-30:
        return np.inf
    numerator = 2.0 * R_GAS * T - (_G25_W_H - T * _G25_W_S)
    return numerator / W_V


def gupta2025_critical_temperature(P_GPa, T_bounds=(200.0, 6000.0)):
    """Critical temperature at given pressure (inverse of P_c(T)).

    Finds T such that P_c(T) = P via root finding on the ascending
    branch of the critical curve. P_c(T) has a singularity near
    T ~ 6130 K where W_V crosses zero; the search is limited to
    T < 6000 K to avoid this.

    Parameters
    ----------
    P_GPa : float
        Pressure in GPa.
    T_bounds : tuple of float
        Search interval for temperature in K. Default (200, 6000).
        The upper bound should stay below the W_V singularity (~6130 K).

    Returns
    -------
    float or None
        Critical temperature in K, or None if no root exists in bounds.
    """

    def residual(T):
        return gupta2025_critical_pressure(T) - P_GPa

    # Check bracket
    f_lo = residual(T_bounds[0])
    f_hi = residual(T_bounds[1])

    if f_lo * f_hi > 0:
        return None

    try:
        T_c = brentq(residual, T_bounds[0], T_bounds[1], xtol=1.0, rtol=1e-8)
        return T_c
    except ValueError:
        return None


def gupta2025_coexistence_compositions(T, P_GPa):
    """Coexisting H2 mole fractions from common-tangent construction (Eq. A7).

    At temperatures below the critical curve, the system separates into
    two phases: an H2-rich phase and an H2O-rich phase. This function
    finds the compositions of both phases.

    Parameters
    ----------
    T : float
        Temperature in K.
    P_GPa : float
        Pressure in GPa.

    Returns
    -------
    tuple of (float, float) or None
        (x_H2O_rich, x_H2_rich) mole fractions of H2 in each phase,
        or None if the system is single-phase (above critical curve).
    """
    P_c = gupta2025_critical_pressure(T)
    if P_GPa >= P_c:
        # Above critical pressure: single phase
        return None

    # Compute Gibbs energy on a fine grid to locate the two minima
    # of the common tangent
    n_pts = 500
    x_arr = np.linspace(0.001, 0.999, n_pts)
    G_arr = np.array([gupta2025_gibbs_mixing(x, T, P_GPa) for x in x_arr])

    # Find local minima of G
    minima = []
    for i in range(1, n_pts - 1):
        if G_arr[i] < G_arr[i - 1] and G_arr[i] < G_arr[i + 1]:
            minima.append(i)

    if len(minima) < 2:
        # No two-phase region
        return None

    # Refine: find the common tangent between first and last minimum
    i1, i2 = minima[0], minima[-1]
    x1, x2 = x_arr[i1], x_arr[i2]

    # Common tangent condition: the line from (x1, G1) to (x2, G2)
    # must be tangent to G(x) at both endpoints. Use Brent on:
    # f(x1, x2) = dG/dx(x1) - (G(x2)-G(x1))/(x2-x1) = 0
    # and similarly for x2.

    def _dG_dx(x):
        """Numerical derivative of G_mix at x."""
        dx = 1e-6
        return (
            gupta2025_gibbs_mixing(x + dx, T, P_GPa) - gupta2025_gibbs_mixing(x - dx, T, P_GPa)
        ) / (2 * dx)

    def _tangent_residual(x_pair):
        """Residual for common tangent: slope at x1 = slope at x2 = chord slope."""
        xa, xb = x_pair
        if xa >= xb or xa <= 0 or xb >= 1:
            return [1e10, 1e10]
        Ga = gupta2025_gibbs_mixing(xa, T, P_GPa)
        Gb = gupta2025_gibbs_mixing(xb, T, P_GPa)
        chord = (Gb - Ga) / (xb - xa)
        return [_dG_dx(xa) - chord, _dG_dx(xb) - chord]

    # Simple iterative refinement from the grid minima
    from scipy.optimize import fsolve

    try:
        sol = fsolve(_tangent_residual, [x1, x2], full_output=True)
        x_sol = sol[0]
        if x_sol[0] < x_sol[1] and 0 < x_sol[0] < 1 and 0 < x_sol[1] < 1:
            return (float(x_sol[0]), float(x_sol[1]))
    except Exception:
        pass

    # Fallback: return grid minima
    if 0 < x1 < x2 < 1:
        return (float(x1), float(x2))

    return None


def gupta2025_suppression_weight(P_Pa, T_K, w_H2, w_H2O, T_scale=50.0):
    """Sigmoid suppression weight for H2-H2O miscibility.

    Returns ~1 when the system is above the critical curve (H2 miscible
    with H2O) and ~0 when below (immiscible, H2 should be suppressed).

    Parameters
    ----------
    P_Pa : float
        Pressure in Pa.
    T_K : float
        Temperature in K.
    w_H2 : float
        Mass fraction of H2.
    w_H2O : float
        Mass fraction of H2O.
    T_scale : float
        Sigmoid width in K. Default 50 K.

    Returns
    -------
    float
        Weight in [0, 1].
    """
    if w_H2 <= 0 or w_H2O <= 0:
        return 1.0

    P_GPa = P_Pa * 1e-9

    # The critical temperature at this pressure
    T_crit = gupta2025_critical_temperature(P_GPa)

    if T_crit is None:
        # Could not find a critical temperature: assume miscible
        return 1.0

    if T_scale <= 0:
        return 1.0 if T_K >= T_crit else 0.0

    arg = (T_K - T_crit) / T_scale
    arg = max(min(arg, 500.0), -500.0)
    return 1.0 / (1.0 + np.exp(-arg))
