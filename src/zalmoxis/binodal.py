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
    H2-MgSiO3 miscibility: doi:10.1093/mnras/staf1940

Gupta, Stixrude & Schlichting (2025), ApJL 982, L35.
    H2-H2O miscibility: doi:10.3847/2041-8213/adb631

Gilmore & Stixrude (2026), Nature 650, 60.
    DFT-MD source for H2-MgSiO3 miscibility and Margules parameters.
"""

from __future__ import annotations

import math

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

# T_c parameters (Eq. A3): T_c = E * (1 + P/D), where D = -35 GPa.
# Equivalently T_c = E * (1 - P/35). T_c decreases with P, reaching
# zero at P = 35 GPa (always miscible above this pressure).
_R25_E = 4223.0  # K
_R25_D = -35.0  # GPa

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

    # Evaluate both branches and take the minimum (lower envelope).
    # The ascending branch rises from x=0 toward T_c; the descending
    # branch falls from T_c toward x=1. Their natural crossing defines
    # the binodal peak. Using min() avoids the artificial kink that a
    # hard if/else cutoff at x_c would produce.
    arg_asc = _R25_ALPHA_3 * (x_H2 - _R25_ALPHA_4)
    arg_asc = max(min(arg_asc, 500.0), -500.0)
    denom_asc = (1.0 + _R25_ALPHA_2 * math.exp(-arg_asc)) ** (1.0 / _R25_ALPHA_5)
    T_asc = T_c / denom_asc

    arg_desc = _R25_BETA_3 * (x_H2 - _R25_BETA_4)
    arg_desc = max(min(arg_desc, 500.0), -500.0)
    denom_desc = (1.0 + _R25_BETA_2 * math.exp(-arg_desc)) ** (1.0 / _R25_BETA_5)
    T_desc = T_c / denom_desc

    return min(T_asc, T_desc)


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
    if w_H2 <= 0:
        return 1.0  # No H2 present: fully miscible (no suppression)
    if w_sil <= 0:
        return 0.0  # No silicate melt: H2 cannot dissolve

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
    return 1.0 / (1.0 + math.exp(-arg))


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

    lambda = lambda_1 + lambda_2 / (T/T_0)

    Note: first power of (T/T_0), not squared. W_V (Eq. A5) uses the
    square; lambda does not. See Gupta, Stixrude & Schlichting (2025),
    ApJL 982, L35, Eq. A6.

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
    return _G25_LAMBDA1 + _G25_LAMBDA2 / ratio


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


def _gupta2025_critical_temperature_brentq(P_GPa, T_bounds=(300.0, 6000.0)):
    """Critical temperature via root finding (internal, use the fast version).

    Parameters
    ----------
    P_GPa : float
        Pressure in GPa.
    T_bounds : tuple of float
        Search interval for temperature in K.

    Returns
    -------
    float or None
        Critical temperature in K, or None if no root exists in bounds.
    """

    def residual(T):
        return gupta2025_critical_pressure(T) - P_GPa

    f_lo = residual(T_bounds[0])
    f_hi = residual(T_bounds[1])

    if f_lo * f_hi > 0:
        return None

    try:
        return brentq(residual, T_bounds[0], T_bounds[1], xtol=1.0, rtol=1e-8)
    except ValueError:
        return None


# ── Precomputed T_crit(P) lookup table ───────────────────────────────
# gupta2025_critical_temperature is called per ODE step (~150,000 times
# per structure solve). The brentq root finder is too expensive for this.
# Precompute T_crit on a log-spaced pressure grid at import time and
# interpolate in the hot path. The table build costs ~2000 brentq calls
# once; each subsequent lookup is a single np.interp.
_G25_TCRIT_LOG_P = np.linspace(-3.0, 3.5, 2000)  # log10(P/GPa): 1 MPa to ~3 TPa
_G25_TCRIT_VALS = np.full(2000, np.nan)
for _i, _logp in enumerate(_G25_TCRIT_LOG_P):
    _tc = _gupta2025_critical_temperature_brentq(10.0**_logp)
    if _tc is not None:
        _G25_TCRIT_VALS[_i] = _tc
# Fill NaN edges with boundary values for safe extrapolation
_valid = np.isfinite(_G25_TCRIT_VALS)
if np.any(_valid):
    _first = np.argmax(_valid)
    _last = len(_valid) - 1 - np.argmax(_valid[::-1])
    _G25_TCRIT_VALS[:_first] = _G25_TCRIT_VALS[_first]
    _G25_TCRIT_VALS[_last + 1 :] = _G25_TCRIT_VALS[_last]
else:
    import warnings

    warnings.warn(
        'Gupta+2025 T_crit(P) table: all brentq solves failed. '
        'H2-H2O binodal suppression will be disabled (H2 treated as '
        'always miscible with H2O). Check binodal parameter values.',
        stacklevel=1,
    )


def gupta2025_critical_temperature(P_GPa, T_bounds=(300.0, 6000.0)):
    """Critical temperature at given pressure (inverse of P_c(T)).

    Uses a precomputed interpolation table for speed. Falls back to
    brentq for pressures outside the table range.

    Parameters
    ----------
    P_GPa : float
        Pressure in GPa.
    T_bounds : tuple of float
        Search interval for brentq fallback. Default (300, 6000).

    Returns
    -------
    float or None
        Critical temperature in K, or None if no root exists.
    """
    if P_GPa <= 0:
        return None
    log_p = math.log10(P_GPa)
    if _G25_TCRIT_LOG_P[0] <= log_p <= _G25_TCRIT_LOG_P[-1]:
        T_c = float(np.interp(log_p, _G25_TCRIT_LOG_P, _G25_TCRIT_VALS))
        if np.isfinite(T_c):
            return T_c
    # Fallback for out-of-range pressures
    return _gupta2025_critical_temperature_brentq(P_GPa, T_bounds)


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

    # The Gupta+2025 model is parameterized for T >= 300 K.
    # At lower T, the W_V and lambda terms diverge (1/T^2 dependence),
    # producing unphysical critical temperatures. Return 0 (immiscible)
    # as a safe default: at T < 300 K, H2 and H2O are certainly phase-
    # separated.
    if T_K < 300.0:
        return 0.0

    P_GPa = P_Pa * 1e-9

    # The critical temperature at this pressure (fast interpolated lookup)
    T_crit = gupta2025_critical_temperature(P_GPa)

    if T_crit is None:
        # Could not find a critical temperature: assume miscible
        return 1.0

    # Floor at the H2O critical temperature (647 K). Below this, H2O
    # condenses to liquid/ice and the Margules Gibbs model (fitted to
    # supercritical DFT-MD data) is not valid. H2 and condensed H2O
    # are always immiscible, so the floor ensures correct suppression
    # for cold sub-Neptune models with condensed water layers.
    T_crit = max(T_crit, 647.0)

    if T_scale <= 0:
        return 1.0 if T_K >= T_crit else 0.0

    arg = (T_K - T_crit) / T_scale
    arg = max(min(arg, 500.0), -500.0)
    return 1.0 / (1.0 + math.exp(-arg))


# ═════════════════════════════════════════════════════════════════════
# Young+2025: Ternary MgSiO3-Fe-H2 phase diagram (PSJ 6:251)
# ═════════════════════════════════════════════════════════════════════

# Molar masses for the ternary system
MU_FE = 55.845e-3  # kg/mol (iron)

# Binary join interaction parameters from Young+2025 and references therein.
# Subregular solution model: G_excess = (L_ij*x_i + L_ji*x_j) * x_i * x_j
# with T,P dependence: * (1 - T/tau + P/pi)

# MgSiO3-H2 (CB join): Gilmore & Stixrude (2025), Nature 650, 60
_Y25_L_CB = 62000.0  # J/mol
_Y25_L_BC = -4950.0  # J/mol
_Y25_TAU_CB = 4800.0  # K
_Y25_PI_CB = -35.0  # GPa

# Fe-H2 (AB join): Young+2025 Eq. 4
# L_AB = 138,000 - 9500*P(GPa), L_BA = 17,000 - 9500*P(GPa)
_Y25_L_AB_0 = 138000.0  # J/mol (at P=0)
_Y25_L_BA_0 = 17000.0  # J/mol (at P=0)
_Y25_L_AB_P = -9500.0  # J/(mol GPa)

# MgSiO3-Fe (CA join): Insixiengmay & Stixrude (2025)
# Regular solution: L_AC = L_CA = 240,000 - 28*T + 1116*P(GPa)
_Y25_L_AC_0 = 240000.0  # J/mol (at T=0, P=0)
_Y25_L_AC_T = -28.0  # J/(mol K)
_Y25_L_AC_P = 1116.0  # J/(mol GPa)

# Ternary interaction parameter (unknown, set to zero)
_Y25_L_ABC = 0.0  # J/mol


def _y25_L_AB(P_GPa):
    """Fe-H2 interaction parameter L_AB (Young+2025 Eq. 4).

    Parameters
    ----------
    P_GPa : float
        Pressure in GPa.

    Returns
    -------
    float
        L_AB in J/mol.
    """
    return _Y25_L_AB_0 + _Y25_L_AB_P * P_GPa


def _y25_L_BA(P_GPa):
    """Fe-H2 interaction parameter L_BA (Young+2025 Eq. 4).

    Parameters
    ----------
    P_GPa : float
        Pressure in GPa.

    Returns
    -------
    float
        L_BA in J/mol.
    """
    return _Y25_L_BA_0 + _Y25_L_AB_P * P_GPa


def _y25_L_AC(T, P_GPa):
    """MgSiO3-Fe interaction parameter (Insixiengmay & Stixrude 2025).

    Regular solution: L_AC = L_CA = 240,000 - 28*T + 1116*P.

    Parameters
    ----------
    T : float
        Temperature in K.
    P_GPa : float
        Pressure in GPa.

    Returns
    -------
    float
        L_AC = L_CA in J/mol.
    """
    return _Y25_L_AC_0 + _Y25_L_AC_T * T + _Y25_L_AC_P * P_GPa


def ternary_gibbs_mixing(x_A, x_B, T, P_GPa):
    """Gibbs free energy of mixing for the MgSiO3-Fe-H2 ternary system.

    Uses the Muggianu-Jacob projection (Young+2025 Eq. 6) to construct
    the ternary Gibbs surface from three binary joins.

    Components: A = Fe, B = H2, C = MgSiO3 (x_C = 1 - x_A - x_B).

    Parameters
    ----------
    x_A : float
        Mole fraction of Fe.
    x_B : float
        Mole fraction of H2.
    T : float
        Temperature in K.
    P_GPa : float
        Pressure in GPa.

    Returns
    -------
    float
        Gibbs free energy of mixing in J/mol. Returns 0 for pure
        endmembers or invalid compositions.
    """
    x_C = 1.0 - x_A - x_B
    if x_A < 0 or x_B < 0 or x_C < 0:
        return 0.0
    if x_A > 1 or x_B > 1 or x_C > 1:
        return 0.0

    # Ideal entropy of mixing
    G_ideal = 0.0
    if x_A > 0:
        G_ideal += x_A * math.log(x_A)
    if x_B > 0:
        G_ideal += x_B * math.log(x_B)
    if x_C > 0:
        G_ideal += x_C * math.log(x_C)
    G_ideal *= R_GAS * T

    # T,P dependence factor for the CB and AB joins
    # Factor = (1 - T/tau + P/pi)
    tp_factor_CB = 1.0 - T / _Y25_TAU_CB + P_GPa / _Y25_PI_CB

    # Excess Gibbs energy from Muggianu-Jacob projection (Eq. 6)
    # Each binary join contributes via the subregular form:
    # G_ij = (L_ij * nu_i + L_ji * nu_j) * nu_i * nu_j
    # where nu_i = 0.5*(1 + x_i - x_j) is the Muggianu projection

    # Muggianu-Jacob projection: for a subregular binary i-j with
    # G_excess = (L_ij*x_j + L_ji*x_i) * x_i * x_j, the ternary
    # projection replaces x_i -> nu_i = 0.5*(1 + x_i - x_j) in the
    # L coefficients but keeps x_i * x_j for the weighting.
    # Convention: L_ij multiplies x_j (the SECOND subscript).

    # AB join (Fe-H2)
    nu_A_AB = 0.5 * (1.0 + x_A - x_B)
    nu_B_AB = 0.5 * (1.0 + x_B - x_A)
    L_AB = _y25_L_AB(P_GPa)
    L_BA = _y25_L_BA(P_GPa)
    G_AB = (L_AB * nu_B_AB + L_BA * nu_A_AB) * x_A * x_B * tp_factor_CB

    # BC join (H2-MgSiO3)
    nu_B_BC = 0.5 * (1.0 + x_B - x_C)
    nu_C_BC = 0.5 * (1.0 + x_C - x_B)
    G_BC = (_Y25_L_BC * nu_C_BC + _Y25_L_CB * nu_B_BC) * x_B * x_C * tp_factor_CB

    # CA join (MgSiO3-Fe): regular solution (L_CA = L_AC), with its
    # own T,P dependence already in the parameter (not via tau/pi)
    L_CA = _y25_L_AC(T, P_GPa)
    G_CA = L_CA * x_C * x_A

    # Ternary term (L_ABC = 0)
    G_ABC = _Y25_L_ABC * x_A * x_B * x_C

    return G_ideal + G_AB + G_BC + G_CA + G_ABC


def ternary_gibbs_hessian_det(x_A, x_B, T, P_GPa, dx=1e-6):
    """Determinant of the Hessian of the ternary Gibbs free energy.

    The spinodal is defined by det(H) = 0 (Young+2025 Eq. 7).
    Negative det(H) indicates spontaneous decomposition into two phases.

    Parameters
    ----------
    x_A : float
        Mole fraction of Fe.
    x_B : float
        Mole fraction of H2.
    T : float
        Temperature in K.
    P_GPa : float
        Pressure in GPa.
    dx : float
        Step size for numerical second derivatives.

    Returns
    -------
    float
        Determinant of the 2x2 Hessian matrix. Negative = two-phase
        (inside spinodal), positive = single-phase (outside spinodal).
    """
    G = ternary_gibbs_mixing

    # Second derivatives via central finite differences
    G0 = G(x_A, x_B, T, P_GPa)

    # d2G/dx_A^2
    d2G_AA = (G(x_A + dx, x_B, T, P_GPa) - 2 * G0 + G(x_A - dx, x_B, T, P_GPa)) / (
        dx * dx
    )

    # d2G/dx_B^2
    d2G_BB = (G(x_A, x_B + dx, T, P_GPa) - 2 * G0 + G(x_A, x_B - dx, T, P_GPa)) / (
        dx * dx
    )

    # d2G/dx_A dx_B
    d2G_AB = (
        G(x_A + dx, x_B + dx, T, P_GPa)
        - G(x_A + dx, x_B - dx, T, P_GPa)
        - G(x_A - dx, x_B + dx, T, P_GPa)
        + G(x_A - dx, x_B - dx, T, P_GPa)
    ) / (4 * dx * dx)

    return d2G_AA * d2G_BB - d2G_AB * d2G_AB


def ternary_is_single_phase(x_A, x_B, T, P_GPa):
    """Check if a ternary composition is in the single-phase region.

    Uses the spinodal criterion: positive Hessian determinant means
    the free energy surface is convex (single phase stable).

    Parameters
    ----------
    x_A : float
        Mole fraction of Fe.
    x_B : float
        Mole fraction of H2.
    T : float
        Temperature in K.
    P_GPa : float
        Pressure in GPa.

    Returns
    -------
    bool
        True if single-phase (outside spinodal), False if two-phase
        or metastable (inside spinodal).
    """
    return ternary_gibbs_hessian_det(x_A, x_B, T, P_GPa) > 0
