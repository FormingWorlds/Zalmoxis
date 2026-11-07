"""Physical invariants of a structure result that ``main`` accepts."""

from __future__ import annotations

import numbers

import numpy as np

from .constants import G

# Largest deviation: a count of bad nodes (padding, centre, positive, *_node), a relative rise
# (monotone) or a relative difference; gravity is a factor on the RK45 bound of ``deviations``;
# density_eos is about 50 times the batch vs scalar EOS gap on T-dependent tables (1.8e-8).
TOLERANCES = {
    'padding': 0,
    'centre': 0,
    'positive': 0,
    'pressure_monotone': 0.0,
    'mass_monotone': 0.0,
    'cmb_mass': 1e-12,
    'core_mantle_mass': 1e-12,
    'cmb_node': 0,
    'core_mantle_node': 0,
    'gravity': 100.0,
    'density_eos': 1e-6,
    'core_cmb_node': 1e-10,
    'core_table_box': 0,
    'core_density_eos': 1e-6,
    'core_element_mass': 1e-10,
}
_RTOL_LAYER = 1e-12  # the layer rule of structure_model.get_layer_mixture
_RTOL_GAUSS = 1e-12  # padded gravity against G m_stop / r^2, both computed in floats
_SCALAR = ('cmb_mass', 'core_mantle_mass', 'mass_closure')  # no node in the message
_ARRAYS = 'radii mass_enclosed gravity pressure density temperature'.split()
_MASSES = ['cmb_mass', 'core_mantle_mass']


class StructureInvariantError(Exception):
    """An accepted structure result breaks a physical invariant.

    Not a RuntimeError: a caller that keeps its previous structure after a failed
    solve must not catch it.
    """


def _layer(m, cmb_mass, core_mantle_mass, ice, icb_mass=0.0):
    """Layer name of each node with enclosed mass ``m``, by the rule of the solver."""
    top = np.where(ice & (m >= core_mantle_mass * (1.0 - _RTOL_LAYER)), 'ice_layer', 'mantle')
    core = np.where(m < icb_mass * (1.0 - _RTOL_LAYER), 'inner_core', 'core')
    return np.where(m < cmb_mass * (1.0 - _RTOL_LAYER), core, top)


def deviations(
    result,
    config_params,
    mass_limit,
    rtol,
    atol,
    density_at=None,
    ice=False,
    details=None,
    box_at=None,
):
    """Return the deviation of each invariant of an accepted result.

    Parameters
    ----------
    result : dict
        Result of ``main`` with ``converged`` True.
    config_params : dict
        Configuration; reads ``planet_mass``, ``core_mass_fraction`` and
        ``mantle_mass_fraction``, and ``target_surface_pressure`` for the density check of
        padded nodes.
    mass_limit : float
        Largest relative mass error that the accept path of ``result`` allows.
    rtol, atol : float
        ``relative_tolerance`` and ``absolute_tolerance`` of the structure integration of
        ``result``. The gravity tolerance at node i is ``TOLERANCES['gravity']`` times
        ``rtol + atol (1 / M_i + 1 / g_i)``, with ``g_i = G M_i / r_i^2``. The RK45 dense
        output keeps ``g r^2 - G M`` per step to about 19.9 rtol + 13.6 atol (1/M_i + 1/g_i)
        relative to ``G M``, an estimate from the Butcher tableau. ``g r^2 - G M`` is constant
        under the exact equations for any density, so an error left at a step end stays, and
        relative to ``G M`` it falls outward as ``M_i / M``. The largest node deviation is the
        dense-output error inside a long step: on the PALEOS-2phase adiabat below it is
        ``+6.4e-6 G M`` at r/R 0.22 in the core, 20.8 units, and ``1.4e-7 G M`` at the last node
        with P > 0.
        The factor 100 is therefore an empirical bound: accepted real solves through ``main``
        of 26 setups, 0.1 to 10 M_E, reach 2.83 to 20.78 times ``rtol + atol (1 / M_i +
        1 / g_i)``, a margin of at least 4.8. They cover the analytic, Wolf & Bower (isothermal
        3000 and 4000 K, linear), PALEOS, PALEOS-2phase (linear, adiabatic), PALEOS with 3 %
        Chabrier H and 3-layer PALEOS ice EOS, 150 and 2000 nodes, rtol 1e-5 and 1e-9, Picard,
        Newton and the JAX kernel. The largest is ``input/default.toml`` at 1 M_E with the
        layers of the PALEOS smoke tests (``Seager2007:iron``, ``PALEOS-2phase:MgSiO3``), the
        adiabatic profile and the default tolerances; the smallest is a 3 % Chabrier H mantle
        at rtol 1e-9. At rtol 1e-5 and atol 1e-6 the tolerance at the surface is about 1e-3
        for any node count, so there the check sees gravity faults of about 0.1 % or more;
        at the first nodes of a fine mesh it is several percent (2.1e-2 at node 1 of 2000).
    density_at : callable or None
        ``(P, T, layer) -> rho`` for the arrays of the nodes of one layer, by the EOS path
        that the solver used at the returned nodes; called once per layer that has nodes.
        A reply that is not a finite number breaks the check. The density check is skipped
        when ``density_at`` is None or when ``result['density_from_eos']`` is False.
    ice : bool
        True when the planet has an ice layer.
    box_at : callable or None
        ``(P, T, layer) -> bool array`` of the core nodes of one layer that its EOS table clamps
        or cannot read; with ``result['core']`` (the core arrays of
        :func:`zalmoxis.core_profile.core_arrays`) ``core_table_box`` counts them and
        ``core_cmb_node`` is the relative difference of the last core-array mass from
        ``cmb_mass``. The layer of a node takes the inner core below ``result['core']['M_ic']``.
        With ``density_at``, ``core_density_eos`` compares each core-array density with the
        EOS of its layer, the inner core where ``phase`` is 1.
    details : dict or None
        If given, filled with {invariant: text} naming the values that enter the deviation and
        their expected values, for the error message, and with ``'density_eos cause'``: the
        exception of a density lookup that raised.

    Returns
    -------
    dict
        {invariant: (deviation, node index, tolerance)}. ``pressure_monotone`` uses the nodes
        before the first node with P <= 0, and ``padding`` the nodes from there on, the surface
        past a stop of the integration, which must have P = 0, the mass of that first node
        and, since no mass lies above the stop, the gravity of that mass at their own radius
        (Gauss), not the gravity at the stop; ``gravity`` checks every node but the centre.
        ``positive`` and ``density_eos`` read every node, ``density_eos`` with the lookup of a
        padded node at ``target_surface_pressure``, the surface pressure that the solve targets;
        ``mass_monotone``, the layer masses, the node checks and ``mass_closure`` read all nodes
        or the last one, and ``centre`` reads node 0.
        ``cmb_node`` and ``core_mantle_node`` mirror the PROTEUS reading of the profile: the
        layer boundary is the first node with m >= cmb_mass (core_mantle_mass with an ice
        layer), compared exactly, and a result where no node reaches it would give the caller
        R_core = 0; its node index is then -1. So a standalone core_mass_fraction 1 run passes
        only with a mass surplus, and with an ice layer so does a run with core_mass_fraction
        plus mantle_mass_fraction 1: a mass deficit is a fault even where no caller reads
        that node. A density lookup that raises breaks ``density_eos`` with an infinite
        deviation and the error in its text.
    """
    P, rho, m, r, g, T = (
        np.asarray(result[k], float)
        for k in ('pressure', 'density', 'mass_enclosed', 'radii', 'gravity', 'temperature')
    )
    ok = np.isfinite(P) & (P > 0)
    k = len(P) if ok.all() else int(np.argmin(ok))
    out = {}

    def put(name, dev, i, tol=None, text=''):
        out[name] = (float(dev), int(i), TOLERANCES[name] if tol is None else tol)
        if details is not None:
            details[name] = text

    put(
        'centre',
        np.count_nonzero([r[0], m[0], g[0]]),
        0,
        text=f'radius, mass, gravity {r[0]:.6e}, {m[0]:.6e}, {g[0]:.6e}, expected 0',
    )
    bad = ~(np.isfinite(rho) & (rho > 0) & np.isfinite(T) & (T > 0))
    i = int(np.argmax(bad)) if bad.any() else k
    put(
        'positive',
        np.count_nonzero(bad) + (k < 2),
        i,
        text=f'density {rho[i]:.6e}, temperature {T[i]:.6e}'
        if bad.any()
        else f'{k} nodes with P > 0',
    )
    # the solver pads a surface stop with zero pressure, the stop mass and its gravity at r
    with np.errstate(divide='ignore', invalid='ignore'):
        gauss = np.abs(g[k:] * r[k:] ** 2 / (G * m[k : k + 1]) - 1) <= _RTOL_GAUSS
    off = (P[k:] != 0) | (m[k:] != m[k : k + 1]) | ~gauss
    i = k + int(np.argmax(off)) if off.any() else k
    put(
        'padding',
        np.count_nonzero(P[k:] != 0)
        + np.count_nonzero(m[k:] != m[k : k + 1])
        + np.count_nonzero(~gauss),
        i,
        text=f'pressure, mass, gravity {P[i]:.6e}, {m[i]:.6e}, {g[i]:.6e}; expected 0, '
        f'the mass of node {k} and G m / r^2 {G * m[k] / r[i] ** 2:.6e}'
        if off.any()
        else '',
    )
    for name, x, sign in (('pressure_monotone', P[:k], 1), ('mass_monotone', m, -1)):
        rise = sign * np.diff(x) / np.abs(x).max(initial=1.0)
        i = int(np.argmax(rise)) + 1 if len(rise) else 0
        put(
            name,
            rise.max(initial=0.0),
            i,
            text=f'{x[i - 1]:.6e} at node {i - 1}, {x[i]:.6e} here' if i else '',
        )
    M, cmf = config_params['planet_mass'], config_params['core_mass_fraction']
    mmf = config_params.get('mantle_mass_fraction', 0.0)
    for name, f in (('cmb_mass', cmf), ('core_mantle_mass', cmf + mmf)):
        put(
            name,
            abs(result[name] / (f * M) - 1) if f > 0 else float(result[name] != 0),
            len(m) - 1,
            text=f'{name} {result[name]:.13e} kg, expected {f * M:.13e} kg',
        )
    # PROTEUS takes the CMB (and with an ice layer the mantle top) at argmax(m >= mass)
    for name, f, key in (
        ('cmb_node', cmf, 'cmb_mass'),
        ('core_mantle_node', cmf + mmf, 'core_mantle_mass'),
    ):
        if f > 0 and (name == 'cmb_node' or ice):
            reach = m >= result[key]
            i = int(np.argmax(reach)) if reach.any() else -1
            put(
                name,
                not (reach[i] and r[i] > 0),
                i,
                text=f'{key} {result[key]:.6e} kg, largest mass_enclosed {m.max():.6e} kg, '
                f'radius of the first node at or above it {r[i] if reach[i] else "none"}',
            )
    with np.errstate(divide='ignore', invalid='ignore'):
        # the bound takes G M / r^2, not the gravity under test, which could widen it
        g_ref = G * m[1:] / r[1:] ** 2
        dg = np.abs(g[1:] / g_ref - 1)
        tol_g = TOLERANCES['gravity'] * (rtol + atol * (1 / m[1:] + 1 / g_ref))
        # a node without mass breaks the check
        ok_g = (m[1:] > 0) & np.isfinite(dg)
        ratio = np.where(ok_g, dg / tol_g, np.inf)
        i = int(np.argmax(ratio)) if len(ratio) else -1
        dev, tol = (dg[i], tol_g[i]) if i >= 0 and ok_g[i] else (np.inf, 0.0)
        j = i + 1
        text = (
            f'gravity {g[j]:.6e} m/s^2, expected {G * m[j] / r[j] ** 2:.6e} (G m / r^2)'
            if i >= 0
            else ''
        )
    put('gravity', dev, j, tol, text=text)
    # the form of the accept test
    put(
        'mass_closure',
        abs(m[-1] - M) / M,
        len(m) - 1,
        mass_limit,
        text=f'mass_enclosed[-1] {m[-1]:.6e} kg, target {M:.6e} kg',
    )
    core = result.get('core')
    icb = core['M_ic'] if core else 0.0
    if core:
        put(
            'core_cmb_node',
            abs(core['m'][-1] / result['cmb_mass'] - 1.0),
            len(core['m']) - 1,
            text=f'core array mass {core["m"][-1]:.6e} kg, cmb_mass {result["cmb_mass"]:.6e} kg',
        )
        put(
            'core_element_mass',
            _element_mass_deviation(core, result['cmb_mass']),
            len(core['m']) - 1,
            text='light-element masses of the core arrays against M_ic w_ic + M_oc w_oc',
        )
    if core and box_at is not None:
        names = np.where(core['phase'] == 1, 'inner_core', 'core')
        bad = np.zeros(len(names), bool)
        for name in np.unique(names):
            sel = names == name
            bad[sel] = box_at(core['P'][sel], core['T'][sel], str(name))
        put(
            'core_table_box',
            bad.sum(),
            int(np.argmax(bad)),
            text='core nodes outside the table',
        )
    if density_at is not None and result.get('density_from_eos'):
        n = len(P)
        lookup = P.copy()
        if k < n:  # a padded node is the surface, at the target surface pressure
            lookup[k:] = config_params['target_surface_pressure']
        layer = _layer(m, result['cmb_mass'], result['core_mantle_mass'], ice, icb)
        ref = np.full(n, np.nan)
        for name in np.unique(layer):
            sel = layer == name
            try:
                ref[sel] = np.asarray(density_at(lookup[sel], T[sel], str(name)), dtype=float)
            except Exception as err:  # reported as the density_eos deviation
                failed = (
                    f'density lookup of the {name} layer failed: {type(err).__name__}: {err}'
                )
                put('density_eos', np.inf, int(np.argmax(sel)), text=failed)
                if details is not None:
                    details['density_eos cause'] = err
                return out
        with np.errstate(divide='ignore', invalid='ignore'):
            d = np.abs(rho / ref - 1)
        d[~np.isfinite(d)] = np.inf
        i = int(np.argmax(d))
        put(
            'density_eos',
            d.max(initial=0.0),
            i,
            text=f'density {rho[i]:.6e}, expected {ref[i]:.6e} ({layer[i]} EOS)',
        )
    if core and density_at is not None and result.get('density_from_eos'):
        names = np.where(core['phase'] == 1, 'inner_core', 'core')
        ref = np.full(len(names), np.nan)
        for name in np.unique(names):
            sel = names == name
            ref[sel] = np.asarray(density_at(core['P'][sel], core['T'][sel], str(name)), float)
        with np.errstate(divide='ignore', invalid='ignore'):
            d = np.abs(np.asarray(core['rho'], float) / ref - 1)
        d[~np.isfinite(d)] = np.inf
        i = int(np.argmax(d))
        put(
            'core_density_eos',
            d.max(initial=0.0),
            i,
            text=f'core array density {core["rho"][i]:.6e}, expected {ref[i]:.6e} ({names[i]} EOS)',
        )
    return out


def _element_mass_deviation(core, cmb_mass):
    """Largest error, over cmb_mass times the largest fraction, of the light-element masses of
    the core arrays: the trapezoid of each ``w_<el>`` over ``m``, exact across the step at
    ``M_ic``, against ``M_ic w_ic + (m_end - M_ic) w_oc`` with the layer compositions
    ``w_layers``."""
    m, m_ic, layers = np.asarray(core['m'], float), core['M_ic'], core.get('w_layers', {})
    columns = {k[2:] for k in core if k.startswith('w_') and k != 'w_layers'}
    dev = 0.0
    for el in columns | set().union(*layers.values()):
        if el not in columns:
            return np.inf
        w = np.asarray(core[f'w_{el}'], float)
        w_ic, w_oc = (layers.get(k, {}).get(el, 0.0) for k in ('inner_core', 'core'))
        got = float(np.sum(0.5 * (w[1:] + w[:-1]) * np.diff(m)))
        step = m_ic * (1.0 - _RTOL_LAYER)  # where the layer rule changes from inner to outer
        for j in np.flatnonzero((m[:-1] < step) & (m[1:] >= step)):  # the step inside a segment
            exact = w_ic * (step - m[j]) + w_oc * (m[j + 1] - step)
            got += exact - 0.5 * (w[j] + w[j + 1]) * (m[j + 1] - m[j])
        want = m_ic * w_ic + (m[-1] - m_ic) * w_oc
        scale = cmb_mass * max(w_ic, w_oc, float(np.abs(w).max()))
        ratio = abs(got - want) / scale if scale > 0 else abs(got - want)
        dev = max(dev, ratio if np.isfinite(ratio) else np.inf)
    return dev


def _finite_positive(x):
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and 0 < x < np.inf


def _real_scalar(x):
    """Whether ``x`` is one real number of any numeric type that a float holds, a 0-d array
    too, not a bool."""
    if isinstance(x, np.ndarray):
        return x.ndim == 0 and x.dtype.kind in 'iuf'
    if not isinstance(x, numbers.Real) or isinstance(x, bool):
        return False
    try:
        float(x)
    except OverflowError:
        return False
    return True


def check_accepted(result, config_params, rtol, atol, density_at=None, ice=False, box_at=None):
    """Raise StructureInvariantError when an accepted result breaks an invariant.

    See ``deviations`` for the parameters; the mass limit is ``result['accept_mass_limit']``,
    which the accept path writes. A result with ``converged`` False is not checked.

    Raises
    ------
    StructureInvariantError
        Names the first broken invariant in the order of ``deviations``, its deviation and
        tolerance, the node for the profile invariants, the values that enter it and the
        ``accepted_on`` label; or the profile arrays or layer masses that the result lacks; or
        a malformed result (arrays that are not numeric, of more than one shape or dimension,
        fewer than 2 nodes, a layer mass that is not one real number); or an
        ``accept_mass_limit``, ``rtol`` or ``atol`` that is not a finite positive number.
    """
    if not result.get('converged'):
        return
    accepted_on = result.get('accepted_on')
    missing = [k for k in _ARRAYS + _MASSES if result.get(k) is None]
    if 'core' in result and result['core'] is None:
        missing.append('core arrays')
    if missing:
        raise StructureInvariantError(
            f'Accepted structure ({accepted_on=}) has no {", ".join(missing)}.'
        )
    try:
        shapes = [np.asarray(result[k], float).shape for k in _ARRAYS]
    except (TypeError, ValueError) as err:
        raise StructureInvariantError(
            f'Accepted structure ({accepted_on=}) is malformed: {err}'
        ) from err
    scalars = [_real_scalar(result[k]) for k in _MASSES]
    if len(set(shapes)) != 1 or len(shapes[0]) != 1 or shapes[0][0] < 2 or not all(scalars):
        raise StructureInvariantError(
            f'Accepted structure ({accepted_on=}) is malformed: the profile arrays need one '
            f'1-D shape of at least 2 nodes and cmb_mass, core_mantle_mass one real number '
            f'each; got shapes {shapes}, cmb_mass {result["cmb_mass"]!r}, core_mantle_mass '
            f'{result["core_mantle_mass"]!r}.'
        )
    limit = result.get('accept_mass_limit')
    if not _finite_positive(limit):
        raise StructureInvariantError(
            f'Accepted structure ({accepted_on=}) has no valid accept_mass_limit: {limit!r}.'
        )
    if not (_finite_positive(rtol) and _finite_positive(atol)):
        raise StructureInvariantError(
            f'Accepted structure ({accepted_on=}) has invalid integration tolerances: '
            f'rtol={rtol!r}, atol={atol!r}.'
        )
    details = {}
    for name, (dev, i, tol) in deviations(
        result, config_params, limit, rtol, atol, density_at, ice, details, box_at
    ).items():
        if not dev <= tol:
            node = '' if name in _SCALAR or i < 0 else f' at node {i}'
            err = StructureInvariantError(
                f'Accepted structure ({accepted_on=}) breaks {name}: deviation {dev:.3e} > '
                f'tolerance {tol:.3e}{node}; {details[name]}.'
            )
            cause = details.get(f'{name} cause')
            if cause is not None:
                raise err from cause
            raise err
