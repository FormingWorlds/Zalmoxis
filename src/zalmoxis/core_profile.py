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


def icb_from_profile(mass, pressure, temperature, cmb_mass, liquidus, icb_mass=None):
    """Inner-core boundary on a structure profile by the liquidus rule.

    Parameters
    ----------
    mass, pressure, temperature : array-like
        Enclosed mass [kg], pressure [Pa] and temperature [K] at the nodes, centre first.
    cmb_mass : float
        Core mass [kg].
    liquidus : callable
        ``T_liq(P)`` [K] of the outer core at pressure [Pa].
    icb_mass : float or None
        A given inner-core mass [kg]: the result then reports it, the pressure at it and the
        core mass above it that lies below the liquidus.

    Returns
    -------
    tuple of float
        Inner-core mass [kg], ICB pressure [Pa] and the core mass above the ICB that lies below
        the liquidus [kg], at node resolution. With the centre at or above the liquidus the
        inner-core mass is 0 and the pressure that of the centre; with the core below it to the
        core mass the inner core is the core and the pressure that at the core mass. A node past
        the core mass adds a point at the core mass to the core nodes (``_cmb_point``).
    """
    m, p, t = (np.asarray(a, dtype=float) for a in (mass, pressure, temperature))
    below = m < cmb_mass * (1.0 - _RTOL_LAYER)
    cmb = _cmb_point(m, p, t, cmb_mass, int(np.count_nonzero(below)))
    core = below & (p > 0)
    m, p, t = m[core], p[core], t[core]
    if cmb is not None:
        m, p, t = np.r_[m, cmb[0]], np.r_[p, cmb[1]], np.r_[t, cmb[2]]
    if not m.size:
        raise ValueError('the profile has no core node with a positive pressure')
    f = t - np.array([float(liquidus(x)) for x in p])
    if not np.all(np.isfinite(f)):
        raise ValueError('the temperature or the core liquidus is not finite on the core nodes')
    if icb_mass is not None:
        return float(icb_mass), float(np.interp(icb_mass, m, p)), _solid_mass(m, f, icb_mass)
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


def _cmb_point(m, p, t, cmb_mass, n):
    """(m, P, T) at the core mass on the segment from the last core node n - 1 to node n, with
    ln P linear in m and T linear in ln P (the adiabat steps that segment with the core
    gradient); None without such a segment of positive pressure."""
    if not 0 < n < len(m) or p[n] <= 0 or p[n - 1] <= 0:
        return None
    lp = float(np.interp(cmb_mass, m[n - 1 : n + 1], np.log(p[n - 1 : n + 1])))
    return cmb_mass, math.exp(lp), float(np.interp(lp, np.log(p[[n, n - 1]]), t[[n, n - 1]]))


def icb_misses(mass, pressure, temperature, cmb_mass, liquidus, icb_mass, icb_adiabat):
    """Distances [kg] of the layer inner-core mass ``icb_mass`` from the liquidus root of the
    profile and from ``icb_adiabat``, the inner-core mass that its adiabat took (0 if None)."""
    root = icb_from_profile(mass, pressure, temperature, cmb_mass, liquidus)[0]
    return abs(root - icb_mass), 0.0 if icb_adiabat is None else abs(icb_adiabat - icb_mass)


def _solid_mass(m, f, m_icb):
    """Core mass above ``m_icb`` in node segments whose ends both lie below the liquidus."""
    seg = (m[:-1] >= m_icb) & (f[:-1] < 0.0) & (f[1:] < 0.0)
    return float(np.sum(np.diff(m)[seg]))


def core_arrays(result, layer_mixtures, rhs, temperature_at, icb_mass, solid_above):
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
    icb_mass, solid_above : float
        Inner-core mass [kg] and the core mass below the liquidus above the ICB [kg].

    Returns
    -------
    dict
        ``r, m, P, rho, g, T`` (m, kg, Pa, kg/m^3, m/s^2, K) and ``phase`` (int8: 1 inner core,
        0 outer core) per node, ``w_<el>`` (mass fraction) per light element of either layer,
        and ``M_ic`` [kg], ``R_icb`` [m], ``P_icb`` [Pa], ``solid_above_icb_mass_fraction`` and
        the flag ``solid_above_icb``.
        The node at the core mass takes the core side of the layer rule for its phase and
        density. ``R_icb`` takes r^3 linear in m and ``P_icb`` P linear in m^(2/3) on the node
        segment of M_ic, both exact for uniform density. ``w_layers`` maps each core layer to
        its light-element fractions.

    Raises
    ------
    ValueError
        When the profile has no node at or past the core mass, or the core mass is not reached
        within 2 node segments.
    """
    r, m, g, p, rho, t = (
        np.asarray(result[k], dtype=float)
        for k in ('radii', 'mass_enclosed', 'gravity', 'pressure', 'density', 'temperature')
    )
    cmb_mass = float(result['cmb_mass'])
    n = int(np.count_nonzero(m < cmb_mass * (1.0 - _RTOL_LAYER)))
    if n == len(m):
        raise ValueError('the profile has no node at or past the core mass')

    def hit(r_, y):
        return y[0] - cmb_mass

    hit.terminal, hit.direction = True, 1
    sol = solve_ivp(
        rhs,
        (r[n - 1], 2.0 * r[n] - r[n - 1]),  # one more segment: node n can lie 1e-12 short
        [m[n - 1], g[n - 1], p[n - 1]],
        events=hit,
        rtol=1e-12,
        atol=[1e-12 * cmb_mass, 1e-12, 1e-6],
    )
    if not sol.t_events[0].size:
        raise ValueError(f'the core mass is not reached within 2 segments of node {n - 1}')
    r_c, (m_c, g_c, p_c) = float(sol.t_events[0][0]), sol.y_events[0][0]
    t_c = float(temperature_at(r_c, p_c))
    m_in = min(m_c, cmb_mass * (1.0 - 4.0 * _RTOL_LAYER))  # the core side of the layer rule
    rho_c = float(rhs(r_c, [m_in, g_c, p_c])[0]) / (4.0 * np.pi * r_c**2)
    out = {
        'r': np.r_[r[:n], r_c],
        'm': np.r_[m[:n], m_c],
        'P': np.r_[p[:n], p_c],
        'rho': np.r_[rho[:n], rho_c],
        'g': np.r_[g[:n], g_c],
        'T': np.r_[t[:n], t_c],
    }
    inner = 'inner_core' in layer_mixtures
    side_m = np.r_[m[:n], m_in]
    out['phase'] = (inner & (side_m < icb_mass * (1.0 - _RTOL_LAYER))).astype(np.int8)
    comps = {
        layer: split_alloy(mix)[1] if is_alloy_mixture(mix) else {}
        for layer, mix in layer_mixtures.items()
        if layer in ('core', 'inner_core')
    }
    out['w_layers'] = comps
    for el in sorted(set().union(*comps.values())):
        w = [comps.get(layer, {}).get(el, 0.0) for layer in ('core', 'inner_core')]
        in_core = side_m < cmb_mass * (1.0 - _RTOL_LAYER)
        out[f'w_{el}'] = np.where(out['phase'] == 1, w[1], np.where(in_core, w[0], 0.0))
    out['M_ic'] = float(icb_mass)
    out['R_icb'] = float(np.interp(icb_mass, out['m'], out['r'] ** 3)) ** (1.0 / 3.0)
    out['P_icb'] = float(np.interp(icb_mass ** (2.0 / 3.0), out['m'] ** (2.0 / 3.0), out['P']))
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
