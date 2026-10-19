"""Exact analytic EOS tables for the table consistency tests.

An ideal gas with constant heat capacities and a Murnaghan solid with thermal expansion, both
derived from a Gibbs free energy g(P, T), so every thermodynamic identity holds exactly. The
tables have the layout of ``zalmoxis.eos_export.load_paleos_all_properties``.
"""

from __future__ import annotations

import numpy as np

COLUMNS = ('rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad')


def ideal_gas(P, T, R_s=287.05, cp=1004.675, P0=1.0e5, T0=300.0, s0=7000.0):
    """Columns of an ideal gas at the points (P [Pa], T [K]), SI units."""
    P, T = np.broadcast_arrays(np.asarray(P, float), np.asarray(T, float))
    one = np.ones_like(P)
    return {
        'rho': P / (R_s * T),
        'u': cp * (T - T0) - R_s * T,
        's': s0 + cp * np.log(T / T0) - R_s * np.log(P / P0),
        'cp': cp * one,
        'cv': (cp - R_s) * one,
        'alpha': 1.0 / T,
        'nabla_ad': R_s / cp * one,
    }


def murnaghan(
    P,
    T,
    rho0=4100.0,
    K0=250.0e9,
    Kp0=4.0,
    alpha0=2.0e-5,
    cp0=1100.0,
    cp1=0.05,
    T0=1500.0,
    s0=2200.0,
    h0=-1.5e7,
):
    """Columns of a Murnaghan solid with constant alpha at the points (P [Pa], T [K]).

    g(P, T) = g0(T) + v0(T) K0 / (Kp0 - 1) ((1 + Kp0 P / K0)^(1 - 1/Kp0) - 1), with
    v0(T) = exp(alpha0 (T - T0)) / rho0 and g0 = h0 - s0 T + cp0 (T - T0 - T ln(T/T0))
    - cp1 (T - T0)^2 / 2.
    """
    P, T = np.broadcast_arrays(np.asarray(P, float), np.asarray(T, float))
    pi = 1.0 + Kp0 * P / K0
    phi = K0 / (Kp0 - 1.0) * (pi ** (1.0 - 1.0 / Kp0) - 1.0)
    v0 = np.exp(alpha0 * (T - T0)) / rho0
    v = v0 * pi ** (-1.0 / Kp0)
    s = s0 + cp0 * np.log(T / T0) + cp1 * (T - T0) - alpha0 * v0 * phi
    g = h0 - s0 * T + cp0 * (T - T0 - T * np.log(T / T0)) - 0.5 * cp1 * (T - T0) ** 2 + v0 * phi
    cp = cp0 + cp1 * T - T * alpha0**2 * v0 * phi
    kappa = 1.0 / (K0 + Kp0 * P)
    cv = cp - T * alpha0**2 * v / kappa
    return {
        'rho': 1.0 / v,
        'u': g + T * s - P * v,
        's': s,
        'cp': cp,
        'cv': cv,
        'alpha': alpha0 * np.ones_like(P),
        'nabla_ad': P * alpha0 * v / cp,
    }


def table(eos, log_p, log_t, digits=None):
    """Table of ``eos`` on the grid (log10 P [Pa], log10 T [K]) in the PALEOS layout.

    ``digits`` rounds every column to that many significant digits, as a file stores them.
    """
    P, T = np.meshgrid(
        10.0 ** np.asarray(log_p, float), 10.0 ** np.asarray(log_t, float), indexing='ij'
    )
    cols = eos(P, T)
    if digits is not None:
        cols = {
            k: np.array([float(f'{x:.{digits - 1}e}') for x in c.ravel()]).reshape(c.shape)
            for k, c in cols.items()
        }
    return {
        'unique_log_p': np.asarray(log_p, float),
        'unique_log_t': np.asarray(log_t, float),
        **cols,
        'phase': np.full(P.shape, 'single', dtype=object),
    }
