"""Density of a liquid iron core that carries light elements.

A core layer is a :class:`~zalmoxis.mixing.LayerMixture` of one iron base EOS
(``PALEOS:iron``) and ``CoreAlloy:<element>`` components with their mass
fractions, for example ``PALEOS:iron:0.97+CoreAlloy:H:0.005+CoreAlloy:O:0.025``.
The base density comes from the iron table at (P, T); the light elements change it
by one of two laws.

Linear law (Li et al. 2019, Table 3.3): each element ``i`` lowers the iron density
by ``beta_i`` percent per wt%, ``rho_lin = rho_Fe (1 - sum_i beta_i w_i)`` with
``beta_i`` as a fraction per unit mass fraction. ``beta`` values are those of the
liquid at the core-mantle boundary for C, O, Si and S. Table 3.3 gives no liquid
value for H, so ``CoreAlloy:H-linear`` uses the solid value 8.7, and no value for
N, so N does not change the density.

Fe-H-O member (``CoreAlloy:H``): Luo, Dorn & Deng (2024), Methods Eq. 14, give the
equations of state of liquid Fe (64 Fe) and of liquid Fe with water (64 Fe + 8
H2O). Their volume difference at the same (P, T), per kg of H2O, is the excess
specific volume ``dv_c`` of an H2O carrier dissolved in iron. The H of the core
enters as this carrier, mass fraction ``w_c = w_H M_H2O / (2 M_H)``; the carrier O,
``w_c - w_H``, counts against the core O, and the remaining O, which is negative
when the core holds less O than the carrier, enters the linear law. The core
density is ``1/rho = (1 - w_c)/rho_lin + w_c dv_c``, where ``rho_lin`` is the
linear law for the remaining alloy, with fractions ``w_i / (1 - w_c)``. To first
order the negative O remainder splits ``dv_c`` into an H part and the linear-law O
part, so the H effect is that of an effective ``beta_H`` of about 12.8 % per wt%
(w_H -> 0) at 136 GPa and 8000 K, against the solid linear value of 8.7. A remaining
alloy with ``1 - lin`` below 0.8 is refused; above it the density falls with H at every
pressure up to 16000 K. The Luo fits cover 8000-14000 K and 50-1300 GPa; outside that
box ``dv_c`` is taken at the nearest point of the box (the iron base stays at the true
P, T), and the point is counted in
:data:`LUO_OUT_OF_RANGE` with one warning per process. The wet cell has a carrier mass
fraction of 0.0388 (0.43 wt% H); a core above it warns once per process. Core-mantle boundary
temperatures of Earth-like planets, about 4000-6000 K, lie below the box, so there the
carrier volume is its 8000 K value.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.constants import Avogadro
from scipy.optimize import brentq

from ..constants import MOLAR_MASS

logger = logging.getLogger('fwl.' + __name__)

ALLOY_PREFIX = 'CoreAlloy:'

# Li et al. 2019, Table 3.3 [% density decrease per wt%], as fractions per unit mass fraction
LINEAR_BETA = {'H-linear': 8.7, 'C': 1.3, 'N': 0.0, 'O': 1.1, 'Si': 0.7, 'S': 0.8}
CARRIER = 'H'
ALLOY_ELEMENTS = frozenset(LINEAR_BETA) | {CARRIER}

# Luo, Dorn & Deng 2024, Methods Eq. 14: K [GPa], V0 [A^3 per cell], a, b, c [GPa]
LUO_DRY = {'K': 49.249, 'Kp': 4.976, 'V0': 1043.912, 'a': -15.957, 'b': 20.946, 'c': -3.811}
LUO_WET = {'K': 57.162, 'Kp': 4.535, 'V0': 1181.389, 'a': -8.478, 'b': 12.049, 'c': -1.258}
LUO_T0 = 8000.0  # K
LUO_T_RANGE = (8000.0, 14000.0)  # K
LUO_P_RANGE = (50e9, 1300e9)  # Pa
LUO_N_FE, LUO_N_H2O = 64, 8

LUO_OUT_OF_RANGE = {'count': 0}

_M_H2O = 2 * MOLAR_MASS['H'] + MOLAR_MASS['O']
CARRIER_PER_H = _M_H2O / (2 * MOLAR_MASS['H'])
_M_DRY_CELL = LUO_N_FE * MOLAR_MASS['Fe'] / Avogadro  # kg
_M_CARRIER_CELL = LUO_N_H2O * _M_H2O / Avogadro  # kg
_A3 = 1e-30  # m^3 per A^3
LUO_DRY['cell_mass'] = _M_DRY_CELL
LUO_WET['cell_mass'] = _M_DRY_CELL + _M_CARRIER_CELL
# carrier mass fraction of the Luo wet cell (0.0388, 0.43 wt% H): the composition of the fit
LUO_CARRIER_FRACTION = _M_CARRIER_CELL / (_M_DRY_CELL + _M_CARRIER_CELL)
_CELL_WARNED = [False]
# lowest O rest (w_O - 7.94 w_H) / (1 - w_c) of the first-order carrier split (w_H 0.0206 at w_O 0)
O_REST_MIN = -0.2
# lowest linear-law factor 1 - lin: above it the density falls with w_H up to 16000 K
LIN_FACTOR_MIN = 0.8


class CoreAlloyError(ValueError):
    """A core alloy composition or density that breaks a physical bound."""


def is_alloy_mixture(mixture) -> bool:
    """Return True when a layer mixture has a ``CoreAlloy:`` component."""
    return any(c.startswith(ALLOY_PREFIX) for c in mixture.components)


def split_alloy(mixture):
    """Split a core mixture into its iron base and its light-element fractions.

    Parameters
    ----------
    mixture : LayerMixture
        Core mixture with one base component and ``CoreAlloy:<element>`` components.

    Returns
    -------
    base : str
        EOS name of the iron base.
    w : dict
        Mass fraction per element key of :data:`ALLOY_ELEMENTS` (absent keys are 0).

    Raises
    ------
    CoreAlloyError
        Unknown element, element given twice, both H forms, not exactly one base,
        a negative or non-finite fraction, or light elements that leave no iron.
    """
    base, w = [], {}
    for comp, frac in zip(mixture.components, mixture.fractions):
        if not math.isfinite(frac) or frac < 0:
            raise CoreAlloyError(f'core component {comp!r} has mass fraction {frac!r}')
        if not comp.startswith(ALLOY_PREFIX):
            base.append(comp)
            continue
        el = comp[len(ALLOY_PREFIX) :]
        if el not in ALLOY_ELEMENTS:
            raise CoreAlloyError(f'unknown core light element {el!r} in {comp!r}')
        if el in w:
            raise CoreAlloyError(f'core light element {el!r} given twice')
        w[el] = float(frac)
    if len(base) != 1:
        raise CoreAlloyError(f'a light-element core needs exactly one iron base, got {base!r}')
    if base[0] != 'PALEOS:iron':
        raise CoreAlloyError(
            f'a light-element core needs the PALEOS:iron base, got {base[0]!r}'
        )
    if 'H' in w and 'H-linear' in w:
        raise CoreAlloyError('CoreAlloy:H and CoreAlloy:H-linear exclude each other')
    if sum(w.values()) >= 1.0:
        raise CoreAlloyError(f'light-element mass fractions sum to {sum(w.values())!r} >= 1')
    w_c, _ = _linear_factor(w)
    if w_c > LUO_CARRIER_FRACTION and not _CELL_WARNED[0]:
        _CELL_WARNED[0] = True
        logger.warning(
            'Core Fe-H-O carrier mass fraction %.4f is above that of the Luo et al. (2024) '
            'wet cell (%.4f); its excess volume is used beyond the cell composition',
            w_c,
            LUO_CARRIER_FRACTION,
        )
    return base[0], w


def _linear_factor(w):
    """Carrier mass fraction ``w_c`` and linear-law sum ``lin`` of the remaining alloy.

    Raises
    ------
    CoreAlloyError
        ``w_c >= 1`` (no alloy mass left), an O rest below :data:`O_REST_MIN` of the
        remaining alloy, or ``1 - lin`` below :data:`LIN_FACTOR_MIN` or not finite.
    """
    w_c = w.get(CARRIER, 0.0) * CARRIER_PER_H
    if w_c >= 1.0:
        raise CoreAlloyError(f'carrier mass fraction {w_c!r} leaves no alloy mass for {w!r}')
    # Total core O is w['O'] >= 0 (split_alloy); only the remainder can be negative.
    o_rem = w.get('O', 0.0) - (w_c - w.get(CARRIER, 0.0))
    if o_rem / (1.0 - w_c) < O_REST_MIN:
        raise CoreAlloyError(
            f'normalised O rest {o_rem / (1.0 - w_c):.3f} of the carrier is below the bound '
            f'{O_REST_MIN} of its first-order split, for {w!r}'
        )
    lin = sum(LINEAR_BETA[el] * f for el, f in w.items() if el not in (CARRIER, 'O'))
    lin = (lin + LINEAR_BETA['O'] * o_rem) / (1.0 - w_c)
    if not 1.0 - lin >= LIN_FACTOR_MIN:
        raise CoreAlloyError(
            f'linear-law core density factor {1.0 - lin:.4f} is below the bound '
            f'{LIN_FACTOR_MIN} for {w!r}'
        )
    return w_c, lin


def luo_pressure(volume, temperature, member):
    """Pressure of a Luo et al. (2024) liquid cell, Eq. 14.

    Parameters
    ----------
    volume : float
        Cell volume in A^3 (64 Fe, or 64 Fe + 8 H2O).
    temperature : float
        Temperature in K.
    member : dict
        :data:`LUO_DRY` or :data:`LUO_WET`.

    Returns
    -------
    float
        Pressure in Pa.
    """
    x = member['V0'] / volume
    x23 = x ** (2.0 / 3.0)
    cold = 1.5 * member['K'] * (x ** (7.0 / 3.0) - x ** (5.0 / 3.0))
    cold *= 1.0 + 0.75 * (member['Kp'] - 4.0) * (x23 - 1.0)
    thermal = (
        (temperature - LUO_T0) / 1000.0 * (member['a'] + member['b'] * x + member['c'] * x * x)
    )
    return (cold + thermal) * 1e9


def luo_cell_volume(pressure, temperature, member):
    """Cell volume in A^3 at (P [Pa], T [K]) from Eq. 14, or None without a root.

    The root is bracketed in [0.15, 1.5] V0, which holds 50-1300 GPa at 8000-14000 K
    with a wide margin.
    """
    v0 = member['V0']
    lo, hi = 0.15 * v0, 1.5 * v0
    f_lo = luo_pressure(lo, temperature, member) - pressure
    f_hi = luo_pressure(hi, temperature, member) - pressure
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)) or f_lo * f_hi > 0:
        return None
    return brentq(
        lambda v: luo_pressure(v, temperature, member) - pressure, lo, hi, xtol=1e-12 * v0
    )


def carrier_excess_volume(pressure, temperature):
    """Excess specific volume of the H2O carrier in liquid iron, m^3 per kg of H2O.

    Evaluated at (P [Pa], T [K]) clamped into the Luo fit box; a clamped point is counted
    in :data:`LUO_OUT_OF_RANGE`. Returns None when a Luo member has no volume there.
    """
    p = min(max(pressure, LUO_P_RANGE[0]), LUO_P_RANGE[1])
    t = min(max(temperature, LUO_T_RANGE[0]), LUO_T_RANGE[1])
    if (p, t) != (pressure, temperature):
        if LUO_OUT_OF_RANGE['count'] == 0:
            logger.warning(
                'Core Fe-H-O carrier at P = %.3e Pa, T = %.0f K is outside the Luo et al. '
                '(2024) fit range (8000-14000 K, 50-1300 GPa); its volume is taken at the edge',
                pressure,
                temperature,
            )
        LUO_OUT_OF_RANGE['count'] += 1
    v_dry = luo_cell_volume(p, t, LUO_DRY)
    v_wet = luo_cell_volume(p, t, LUO_WET)
    if v_dry is None or v_wet is None:
        return None
    return (v_wet - v_dry) * _A3 / _M_CARRIER_CELL


def luo_density(pressure, temperature, member):
    """Density in kg/m^3 of a Luo et al. (2024) member at (P [Pa], T [K]), or None."""
    v = luo_cell_volume(pressure, temperature, member)
    if v is None:
        return None
    return member['cell_mass'] / (v * _A3)


def alloy_density(rho_fe, w, pressure, temperature):
    """Density of the light-element core at one (P, T) point.

    Parameters
    ----------
    rho_fe : float
        Density of the iron base at (P, T), kg/m^3.
    w : dict
        Light-element mass fractions from :func:`split_alloy`.
    pressure : float
        Pressure in Pa.
    temperature : float
        Temperature in K.

    Returns
    -------
    float or None
        Density in kg/m^3; None when the iron base or the Luo members have no value
        at (P, T).

    Raises
    ------
    CoreAlloyError
        A carrier that leaves no alloy mass, a composition outside the bounds of
        :func:`_linear_factor`, or a core without positive finite density.
    """
    w_c, lin = _linear_factor(w)
    if rho_fe is None or not np.isfinite(rho_fe) or rho_fe <= 0:
        return None
    rho_lin = rho_fe * (1.0 - lin)
    if w_c <= 0:
        return rho_lin
    dv_c = carrier_excess_volume(pressure, temperature)
    if dv_c is None:
        return None
    rho = 1.0 / ((1.0 - w_c) / rho_lin + w_c * dv_c)
    if not np.isfinite(rho) or rho <= 0:
        raise CoreAlloyError(f'core alloy density {rho!r} kg/m^3 for {w!r}')
    return rho


def alloy_density_batch(rho_fe, w, pressures, temperatures):
    """Vectorised :func:`alloy_density`; NaN where a point has no density."""
    rho = (alloy_density(r, w, p, t) for r, p, t in zip(rho_fe, pressures, temperatures))
    return np.array(list(rho), dtype=float)  # None becomes NaN
