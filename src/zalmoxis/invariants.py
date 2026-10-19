"""Physical invariants of a structure result that ``main`` accepts."""

from __future__ import annotations

import numbers

import numpy as np

from .constants import G

# Largest allowed deviation of each invariant; a count of bad nodes where the tolerance is 0.
# gravity: a factor on the per-node RK45 bound of ``deviations``; density_eos: round-off, since
# both sides call the same batch EOS at the same (P, T).
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
}
_RTOL_LAYER = 1e-12  # the layer rule of structure_model.get_layer_mixture
_SCALAR = ('cmb_mass', 'core_mantle_mass', 'mass_closure')  # no node in the message
_ARRAYS = 'radii mass_enclosed gravity pressure density temperature'.split()
_MASSES = ['cmb_mass', 'core_mantle_mass']


class StructureInvariantError(Exception):
    """An accepted structure result breaks a physical invariant.

    Not a RuntimeError: a caller that keeps its previous structure after a failed
    solve must not catch it.
    """


def _layer(m, cmb_mass, core_mantle_mass, ice):
    """Layer name of each node with enclosed mass ``m``, by the rule of the solver."""
    top = np.where(ice & (m >= core_mantle_mass * (1.0 - _RTOL_LAYER)), 'ice_layer', 'mantle')
    return np.where(m < cmb_mass * (1.0 - _RTOL_LAYER), 'core', top)


def deviations(
    result, config_params, mass_limit, rtol, atol, density_at=None, ice=False, details=None
):
    """Return the deviation of each invariant of an accepted result.

    Parameters
    ----------
    result : dict
        Result of ``main`` with ``converged`` True.
    config_params : dict
        Configuration; reads ``planet_mass``, ``core_mass_fraction`` and
        ``mantle_mass_fraction``.
    mass_limit : float
        Largest relative mass error that the accept path of ``result`` allows.
    rtol, atol : float
        ``relative_tolerance`` and ``absolute_tolerance`` of the structure integration of
        ``result``. The gravity tolerance at node i is ``TOLERANCES['gravity']`` times
        ``rtol + atol (1 / M_i + 1 / g_i)``, with ``g_i = G M_i / r_i^2``. The RK45 dense
        output keeps ``g r^2 - G M`` per step to about 19.9 rtol + 13.6 atol (1/M_i + 1/g_i)
        relative to ``G M``, an estimate from the Butcher tableau. ``g r^2 - G M`` is constant
        under the exact equations for any density, so a step error stays, the step errors add,
        and relative to ``G M`` an error made at mass M_i falls outward only as ``M_i / M``.
        The factor 100 is an empirical bound on that sum: accepted real solves through ``main`` of 26 setups, 0.1 to 10 M_E, reach 2.83 to
        20.78 times ``rtol + atol (1 / M_i + 1 / g_i)``, a margin of at least 4.8. They cover
        the analytic, Wolf & Bower (isothermal 3000 and 4000 K, linear), PALEOS, PALEOS-2phase
        (linear, adiabatic), PALEOS with 3 % Chabrier H and 3-layer PALEOS ice EOS, 150 and 2000
        nodes, rtol 1e-5 and 1e-9, Picard, Newton and the JAX kernel. The largest is the 1 M_E
        planet of ``input/default.toml`` with a Seager iron core, a PALEOS-2phase mantle and the
        adiabatic profile, at the default tolerances; the smallest is a 3 % Chabrier H mantle at
        rtol 1e-9. At rtol 1e-5 and atol 1e-6 the tolerance at the surface is about 1e-3 for
        any node count.
    density_at : callable or None
        ``(P, T, layer) -> rho`` for the arrays of the nodes of one layer, by the EOS path
        that the solver used at the returned nodes; called once per layer that has nodes.
        A reply that is not a finite number breaks the check. The density check is skipped
        when ``density_at`` is None or when ``result['density_from_eos']`` is False.
    ice : bool
        True when the planet has an ice layer.
    details : dict or None
        If given, filled with {invariant: text} naming the values that enter the deviation and
        their expected values, for the error message.

    Returns
    -------
    dict
        {invariant: (deviation, node index, tolerance)}. The checks use the nodes
        before the first node with P <= 0; the nodes after it are padding with P = 0.
        ``cmb_node`` and ``core_mantle_node`` mirror the PROTEUS reading of the profile: the
        layer boundary is the first node with m >= cmb_mass (core_mantle_mass with an ice
        layer), compared exactly, and a result where no node reaches it would give the caller
        R_core = 0; its node index is then -1. So a standalone core_mass_fraction 1 run passes
        only with a mass surplus. A density lookup that raises breaks ``density_eos`` with an
        infinite deviation and the error in its text.
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
    bad = ~(np.isfinite(rho[:k]) & (rho[:k] > 0) & np.isfinite(T[:k]) & (T[:k] > 0))
    i = int(np.argmax(bad)) if bad.any() else k
    put(
        'positive',
        np.count_nonzero(bad) + (k < 2),
        i,
        text=f'density {rho[i]:.6e}, temperature {T[i]:.6e}'
        if i < k
        else f'{k} nodes with P > 0',
    )
    # the solver pads a surface stop with zero pressure and the mass and gravity at the stop
    off = (P[k:] != 0) | (m[k:] != m[k : k + 1]) | (g[k:] != g[k : k + 1])
    i = k + int(np.argmax(off)) if off.any() else k
    pad = np.count_nonzero(P[k:] != 0) + np.count_nonzero(m[k:] != m[k : k + 1])
    put(
        'padding',
        pad + np.count_nonzero(g[k:] != g[k : k + 1]),
        i,
        text=f'pressure, mass, gravity {P[i]:.6e}, {m[i]:.6e}, {g[i]:.6e}; expected 0 and '
        f'the mass and gravity of node {k}'
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
        g_ref = G * m[1:k] / r[1:k] ** 2
        dg = np.abs(g[1:k] / g_ref - 1)
        tol_g = TOLERANCES['gravity'] * (rtol + atol * (1 / m[1:k] + 1 / g_ref))
        # a node without mass breaks the check
        ok_g = (m[1:k] > 0) & np.isfinite(dg)
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
    if density_at is not None and result.get('density_from_eos'):
        layer = _layer(m[:k], result['cmb_mass'], result['core_mantle_mass'], ice)
        ref = np.full(k, np.nan)
        for name in np.unique(layer):
            sel = layer == name
            try:
                ref[sel] = np.asarray(
                    density_at(P[:k][sel], T[:k][sel], str(name)), dtype=float
                )
            except Exception as err:  # noqa: BLE001 -- reported as the density_eos deviation
                failed = f'density lookup of the {name} layer failed: {err}'
                put('density_eos', np.inf, int(np.argmax(sel)), text=failed)
                return out
        with np.errstate(divide='ignore', invalid='ignore'):
            d = np.abs(rho[:k] / ref - 1)
        d[~np.isfinite(d)] = np.inf
        i = int(np.argmax(d)) if k else 0
        put(
            'density_eos',
            d.max(initial=0.0),
            i,
            text=f'density {rho[i]:.6e}, expected {ref[i]:.6e} ({layer[i]} EOS)' if k else '',
        )
    return out


def _finite_positive(x):
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and 0 < x < np.inf


def _real_scalar(x):
    """Whether ``x`` is one integer or floating number (a 0-d array included), not a bool."""
    return np.ndim(x) == 0 and np.asarray(x).dtype.kind in 'iuf'


def check_accepted(result, config_params, rtol, atol, density_at=None, ice=False):
    """Raise StructureInvariantError when an accepted result breaks an invariant.

    See ``deviations`` for the parameters; the mass limit is ``result['accept_mass_limit']``,
    which the accept path writes. A result with ``converged`` False is not checked.

    Raises
    ------
    StructureInvariantError
        Names the first broken invariant in the order of ``deviations``, its deviation and
        tolerance, the node for the profile invariants, the values that enter it and the
        ``accepted_on`` label; or the profile arrays that the result lacks; or a malformed
        result (arrays of more than one shape or dimension, fewer than 2 nodes, a layer mass
        that is not one real number); or an ``accept_mass_limit``, ``rtol`` or ``atol`` that
        is not a finite positive number.
    """
    if not result.get('converged'):
        return
    accepted_on = result.get('accepted_on')
    missing = [k for k in _ARRAYS + _MASSES if result.get(k) is None]
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
        result, config_params, limit, rtol, atol, density_at, ice, details
    ).items():
        if not dev <= tol:
            node = '' if name in _SCALAR or i < 0 else f' at node {i}'
            raise StructureInvariantError(
                f'Accepted structure ({accepted_on=}) breaks {name}: deviation {dev:.3e} > '
                f'tolerance {tol:.3e}{node}; {details[name]}.'
            )
