"""Melting curve functions for the mantle EOS phase routing.

Provides analytic solidus and liquidus curves from:

- Monteux et al. (2016, EPSL 448, 140-149): piecewise Simon-Glatzel fits
  (Eqs. 10-13). Valid up to ~400-500 GPa.
- Stixrude (2014, Phil. Trans. R. Soc. A 372, 20130076): Simon-like power
  laws fitted to Lindemann melting theory and experiments (Eqs. 1.9-1.10).
  Valid over the full super-Earth pressure range.

Available identifiers
---------------------
Solidus:
    ``'Monteux16-solidus'``
        Monteux+2016 Eqs. 10/12, piecewise (P <= 20 GPa / P > 20 GPa).
    ``'Stixrude14-solidus'``
        Stixrude (2014) Eq. 1.9 + cryoscopic Eq. 1.10 with x0=0.79.
    ``'Monteux600-solidus-tabulated'``
        Tabulated: ``melting_curves_Monteux-600/solidus.dat``.

Liquidus:
    ``'Monteux16-liquidus-A-chondritic'``
        Monteux+2016 Eqs. 11/13 A-chondritic.
    ``'Monteux16-liquidus-F-peridotitic'``
        Monteux+2016 Eqs. 11/13 F-peridotitic.
    ``'Stixrude14-liquidus'``
        Stixrude (2014) Eq. 1.9: pure MgSiO3 Simon-like power law.
    ``'Monteux600-liquidus-tabulated'``
        Tabulated: ``melting_curves_Monteux-600/liquidus.dat``.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from scipy.interpolate import interp1d

logger = logging.getLogger(__name__)

ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')


# ── Valid identifiers ──────────────────────────────────────────────────

VALID_SOLIDUS = {'Monteux16-solidus', 'Stixrude14-solidus', 'Monteux600-solidus-tabulated'}
VALID_LIQUIDUS = {
    'Monteux16-liquidus-A-chondritic',
    'Monteux16-liquidus-F-peridotitic',
    'Stixrude14-liquidus',
    'Monteux600-liquidus-tabulated',
}


# ── Monteux+2016 analytic curves (Eqs. 10-13) ─────────────────────────

# Transition pressure between low-P and high-P parameterizations [Pa]
_P_TRANSITION = 20.0e9  # 20 GPa


def _solidus_low(P):
    """Solidus for P <= 20 GPa (Herzberg & Zhang 1996, Eq. 10)."""
    return 1661.2 * (P / 1.336e9 + 1.0) ** (1.0 / 7.437)


def _solidus_high(P):
    """Solidus for P > 20 GPa (Andrault et al. 2011 A-chondritic, Eq. 12)."""
    return 2081.8 * (P / 101.69e9 + 1.0) ** (1.0 / 1.226)


def monteux16_solidus(P):
    """Monteux+2016 solidus temperature.

    Piecewise: Eq. 10 (P <= 20 GPa) and Eq. 12 (P > 20 GPa).
    Both composition models share the same solidus.

    Parameters
    ----------
    P : float or array-like
        Pressure in Pa (must be >= 0).

    Returns
    -------
    float or ndarray
        Solidus temperature in K.
    """
    P_arr = np.atleast_1d(np.asarray(P, dtype=float))
    T = np.empty_like(P_arr)
    lo = P_arr <= _P_TRANSITION
    hi = ~lo
    T[lo] = _solidus_low(P_arr[lo])
    T[hi] = _solidus_high(P_arr[hi])
    return float(T[0]) if np.ndim(P) == 0 else T


def _liquidus_low(P):
    """Liquidus for P <= 20 GPa (Herzberg & Zhang 1996, Eq. 11)."""
    return 1982.1 * (P / 6.594e9 + 1.0) ** (1.0 / 5.374)


def _liquidus_high_F(P):
    """Liquidus for P > 20 GPa, F-peridotitic (Fiquet et al. 2010, Eq. 13)."""
    return 78.74 * (P / 4.054e6 + 1.0) ** (1.0 / 2.44)


def _liquidus_high_A(P):
    """Liquidus for P > 20 GPa, A-chondritic (Andrault et al. 2011, Eq. 13)."""
    return 2006.8 * (P / 34.65e9 + 1.0) ** (1.0 / 1.844)


def monteux16_liquidus(P, model='A-chondritic'):
    """Monteux+2016 liquidus temperature.

    Piecewise: Eq. 11 (P <= 20 GPa) and Eq. 13 (P > 20 GPa).

    Parameters
    ----------
    P : float or array-like
        Pressure in Pa (must be >= 0).
    model : str
        ``'A-chondritic'`` or ``'F-peridotitic'``.

    Returns
    -------
    float or ndarray
        Liquidus temperature in K.
    """
    P_arr = np.atleast_1d(np.asarray(P, dtype=float))
    T = np.empty_like(P_arr)
    lo = P_arr <= _P_TRANSITION
    hi = ~lo
    T[lo] = _liquidus_low(P_arr[lo])

    if model == 'A-chondritic':
        T[hi] = _liquidus_high_A(P_arr[hi])
    elif model == 'F-peridotitic':
        T[hi] = _liquidus_high_F(P_arr[hi])
    else:
        raise ValueError(f"Unknown model '{model}'. Use 'A-chondritic' or 'F-peridotitic'.")

    return float(T[0]) if np.ndim(P) == 0 else T


# ── Stixrude (2014) melting curves (Eqs. 1.9-1.10) ────────────────────

# Reference pressure and temperature for the silicate liquidus (Eq. 1.9)
_STIX14_T_REF = 5400.0  # K, MgSiO3 liquidus at P_ref
_STIX14_P_REF = 140.0e9  # Pa (140 GPa)
_STIX14_EXPONENT = 0.480

# Cryoscopic depression factor for Earth-like mantle solidus (Eq. 1.10)
# x0 = 0.79 (mole fraction of pure MgSiO3 in Earth-like mantle)
# depression = (1 - ln(x0))^{-1} = (1 - ln(0.79))^{-1}
_STIX14_X0_ROCK = 0.79
_STIX14_CRYO_FACTOR = 1.0 / (1.0 - np.log(_STIX14_X0_ROCK))  # ~0.809


def stixrude14_liquidus(P):
    """Silicate (MgSiO3) liquidus from Stixrude (2014) Eq. 1.9.

    Simon-like power law fitted to Lindemann melting theory:
    T_rock = 5400 K * (P / 140 GPa)^0.480

    Defined for all P > 0. At P = 0, returns 0 K (extrapolation).

    Parameters
    ----------
    P : float or array-like
        Pressure in Pa (must be >= 0).

    Returns
    -------
    float or ndarray
        Liquidus temperature in K.
    """
    P_arr = np.atleast_1d(np.asarray(P, dtype=float))
    # Guard P=0: the power law gives 0 K at P=0
    T = np.where(
        P_arr > 0,
        _STIX14_T_REF * (P_arr / _STIX14_P_REF) ** _STIX14_EXPONENT,
        0.0,
    )
    return float(T[0]) if np.ndim(P) == 0 else T


def stixrude14_solidus(P):
    """Silicate solidus from Stixrude (2014) Eqs. 1.9 + 1.10.

    The solidus is the pure-substance liquidus (Eq. 1.9) depressed by
    the cryoscopic equation (Eq. 1.10) with x0 = 0.79 (mole fraction
    of pure MgSiO3 in an Earth-like mantle composition):

    T_solidus = T_liquidus * (1 - ln(x0))^{-1}

    With x0 = 0.79 this reproduces the experimentally measured solidus
    of 4100 K at 140 GPa (Fiquet et al. 2010).

    Parameters
    ----------
    P : float or array-like
        Pressure in Pa (must be >= 0).

    Returns
    -------
    float or ndarray
        Solidus temperature in K.
    """
    return stixrude14_liquidus(P) * _STIX14_CRYO_FACTOR


# ── Tabulated melting curves ───────────────────────────────────────────


def _load_tabulated_curve(filepath):
    """Load a tabulated melting curve from a text file.

    Parameters
    ----------
    filepath : str
        Path to file with two columns: P [Pa], T [K].

    Returns
    -------
    callable
        Interpolation function f(P) -> T. Returns NaN outside the table range.
    """
    data = np.loadtxt(filepath, comments='#')
    pressures = data[:, 0]
    temperatures = data[:, 1]
    return interp1d(
        pressures, temperatures, kind='linear', bounds_error=False, fill_value=np.nan
    )


# ── Dispatcher ─────────────────────────────────────────────────────────


def get_melting_curve_function(curve_id):
    """Return a callable melting curve f(P [Pa]) -> T [K] for the given identifier.

    Parameters
    ----------
    curve_id : str
        Melting curve identifier. See module docstring for valid values.

    Returns
    -------
    callable
        Function accepting pressure in Pa (scalar or array) and returning
        temperature in K.

    Raises
    ------
    ValueError
        If ``curve_id`` is not recognized.
    """
    if curve_id == 'Monteux16-solidus':
        return monteux16_solidus

    elif curve_id == 'Monteux16-liquidus-A-chondritic':

        def _liq(P):
            return monteux16_liquidus(P, model='A-chondritic')

        return _liq

    elif curve_id == 'Monteux16-liquidus-F-peridotitic':

        def _liq(P):
            return monteux16_liquidus(P, model='F-peridotitic')

        return _liq

    elif curve_id == 'Stixrude14-solidus':
        return stixrude14_solidus

    elif curve_id == 'Stixrude14-liquidus':
        return stixrude14_liquidus

    elif curve_id == 'Monteux600-solidus-tabulated':
        return _load_tabulated_curve(
            os.path.join(ZALMOXIS_ROOT, 'data', 'melting_curves_Monteux-600', 'solidus.dat')
        )

    elif curve_id == 'Monteux600-liquidus-tabulated':
        return _load_tabulated_curve(
            os.path.join(ZALMOXIS_ROOT, 'data', 'melting_curves_Monteux-600', 'liquidus.dat')
        )

    else:
        all_valid = sorted(VALID_SOLIDUS | VALID_LIQUIDUS)
        raise ValueError(f"Unknown melting curve '{curve_id}'. Valid values: {all_valid}")


def get_solidus_liquidus_functions(
    solidus_id='Stixrude14-solidus',
    liquidus_id='Stixrude14-liquidus',
):
    """Load solidus and liquidus functions by config identifier.

    Parameters
    ----------
    solidus_id : str
        Solidus curve identifier.
    liquidus_id : str
        Liquidus curve identifier.

    Returns
    -------
    tuple of callable
        ``(solidus_func, liquidus_func)``
    """
    return get_melting_curve_function(solidus_id), get_melting_curve_function(liquidus_id)
