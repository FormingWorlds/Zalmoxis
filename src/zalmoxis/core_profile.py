"""Inner-core boundary and core arrays of a structure solve.

A two-layer core has an inner core on the solid iron side and an outer core on the liquid side
(``layer_eos_config['inner_core']`` and ``['core']``). Its inner-core boundary (ICB) is either a
given mass, ``inner_core_mass_fraction`` of the core mass, or the first point, from the centre
outward, where the temperature of the profile reaches the liquidus ``T_liq(P)``. The structure
equations place a shell in the inner core by its enclosed mass, as they place the core-mantle
boundary, so the liquidus rule is applied to the profile of the previous outer pass and the
outer loop converges it with the structure.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from .eos.core_alloy import IRON_SIDES, is_alloy_mixture, split_alloy

_RTOL_LAYER = 1e-12  # the layer rule of structure_model.get_layer_mixture


def icb_from_profile(mass, pressure, temperature, cmb_mass, liquidus):
    """Inner-core boundary on a structure profile by the liquidus rule.

    Parameters
    ----------
    mass, pressure, temperature : array-like
        Enclosed mass [kg], pressure [Pa] and temperature [K] at the nodes, centre first.
    cmb_mass : float
        Core mass [kg].
    liquidus : callable
        ``T_liq(P)`` [K] of the outer core at pressure [Pa].

    Returns
    -------
    tuple of float
        Inner-core mass [kg], ICB pressure [Pa] and the core mass above the ICB that lies below
        the liquidus [kg], at node resolution. With the centre at or above the liquidus the
        inner-core mass is 0 and the pressure that of the centre; with every core node below it
        the inner core is the core and the pressure that of the last core node.
    """
    m, p, t = (np.asarray(a, dtype=float) for a in (mass, pressure, temperature))
    core = (m < cmb_mass * (1.0 - _RTOL_LAYER)) & (p > 0)
    m, p, t = m[core], p[core], t[core]
    f = t - np.array([float(liquidus(x)) for x in p])
    above = np.flatnonzero(f >= 0.0)
    if above.size and above[0] == 0:
        return 0.0, float(p[0]), _solid_mass(m, f, 0.0)
    if not above.size:
        return float(cmb_mass), float(p[-1]), 0.0
    k = int(above[0])
    lp_up, t_up = np.log(p[::-1]), t[::-1]  # ascending in pressure for np.interp
    p_icb = brentq(
        lambda x: float(np.interp(math.log(x), lp_up, t_up)) - float(liquidus(x)),
        p[k],
        p[k - 1],
        xtol=1e-3,
        rtol=4.0 * np.finfo(float).eps,
    )
    m_icb = float(np.interp(math.log(p_icb), lp_up, m[::-1]))
    return m_icb, float(p_icb), _solid_mass(m, f, m_icb)


def _solid_mass(m, f, m_icb):
    """Core mass above ``m_icb`` in node segments whose ends both lie below the liquidus."""
    seg = (m[:-1] >= m_icb) & (f[:-1] < 0.0) & (f[1:] < 0.0)
    return float(np.sum(np.diff(m)[seg]))


def core_arrays(result, layer_mixtures, rhs, temperature_at, icb_mass, p_icb, solid_above):
    """Arrays of the core from the centre to a node at the core mass, and its ICB scalars.

    Parameters
    ----------
    result : dict
        Structure result with ``radii``, ``mass_enclosed``, ``gravity``, ``pressure``,
        ``density``, ``temperature`` and ``cmb_mass``.
    layer_mixtures : dict
        Per-layer mixtures; ``inner_core`` present for a two-layer core.
    rhs : callable
        ``rhs(r, y)`` of the structure equations with the core mixture, y = (m, g, P).
    temperature_at : callable
        ``T(r, P)`` [K] of the solve.
    icb_mass, p_icb, solid_above : float
        Inner-core mass [kg], ICB pressure [Pa] and core mass below the liquidus above the ICB.

    Returns
    -------
    dict
        ``r, m, P, rho, g, T`` (m, kg, Pa, kg/m^3, m/s^2, K) and ``phase`` (int8: 1 inner core,
        0 outer core) per node, ``w_<el>`` (mass fraction) per light element of either layer,
        and ``M_ic`` [kg], ``R_icb`` [m], ``P_icb`` [Pa], ``solid_above_icb_mass_fraction`` and
        the flag ``solid_above_icb``.
    """
    r, m, g, p, rho, t = (
        np.asarray(result[k], dtype=float)
        for k in ('radii', 'mass_enclosed', 'gravity', 'pressure', 'density', 'temperature')
    )
    cmb_mass = float(result['cmb_mass'])
    n = int(np.count_nonzero(m < cmb_mass * (1.0 - _RTOL_LAYER)))

    def hit(r_, y):
        return y[0] - cmb_mass

    hit.terminal, hit.direction = True, 1
    sol = solve_ivp(
        rhs,
        (r[n - 1], r[n]),
        [m[n - 1], g[n - 1], p[n - 1]],
        events=hit,
        rtol=1e-12,
        atol=[1e-12 * cmb_mass, 1e-12, 1e-6],
    )
    if sol.t_events[0].size:
        r_c, (m_c, g_c, p_c) = float(sol.t_events[0][0]), sol.y_events[0][0]
    else:  # the mass of the last core node rounds to the core mass
        r_c, m_c, g_c, p_c = r[n - 1], m[n - 1], g[n - 1], p[n - 1]
    t_c = float(temperature_at(r_c, p_c))
    rho_c = float(rhs(r_c, [m_c, g_c, p_c])[0]) / (4.0 * np.pi * r_c**2)
    out = {
        'r': np.r_[r[:n], r_c],
        'm': np.r_[m[:n], m_c],
        'P': np.r_[p[:n], p_c],
        'rho': np.r_[rho[:n], rho_c],
        'g': np.r_[g[:n], g_c],
        'T': np.r_[t[:n], t_c],
    }
    inner = 'inner_core' in layer_mixtures
    out['phase'] = (inner & (out['m'] < icb_mass * (1.0 - _RTOL_LAYER))).astype(np.int8)
    comps = {
        layer: split_alloy(mix)[1] if is_alloy_mixture(mix) else {}
        for layer, mix in layer_mixtures.items()
        if layer in ('core', 'inner_core')
    }
    for el in sorted(set().union(*comps.values())):
        w = [comps.get(layer, {}).get(el, 0.0) for layer in ('core', 'inner_core')]
        out[f'w_{el}'] = np.where(out['phase'] == 1, w[1], w[0])
    out['M_ic'] = float(icb_mass)
    out['R_icb'] = float(np.interp(icb_mass, out['m'], out['r']))
    out['P_icb'] = float(p_icb)
    out['solid_above_icb_mass_fraction'] = float(solid_above) / cmb_mass
    out['solid_above_icb'] = bool(solid_above > 0.0)
    return out


def outside_side_tables(P, T, mixture, material_dictionaries, cache):
    """Nodes whose (P, T) an iron side table of ``mixture`` clamps or cannot read.

    A node is outside where its pressure is outside the table, its temperature outside the
    valid band of its pressure row (where the read clamps it) or the bilinear density there is
    not finite (where the read takes the nearest neighbour). Mixtures without a side table give
    no node.
    """
    from .eos.interpolation import (
        _ensure_unified_cache,
        _fast_bilinear,
        _paleos_clamp_temperature,
    )

    out = np.zeros(len(P), dtype=bool)
    for comp in mixture.components:
        if comp not in IRON_SIDES:
            continue
        c = _ensure_unified_cache(material_dictionaries[comp]['eos_file'], cache)
        for i, (p, t) in enumerate(zip(P, T)):
            if not c['p_min'] <= p <= c['p_max'] or not t > 0:
                out[i] = True
                continue
            lp, lt = math.log10(p), math.log10(t)
            clamped = _paleos_clamp_temperature(lp, lt, c)[1]
            out[i] |= clamped or not math.isfinite(_fast_bilinear(lp, lt, c['density_grid'], c))
    return out
