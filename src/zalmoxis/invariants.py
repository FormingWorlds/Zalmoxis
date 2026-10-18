"""Physical invariants of a structure result that ``main`` accepts."""

from __future__ import annotations

import numbers

import numpy as np

from .constants import G

# Largest allowed deviation of each invariant; a count of bad nodes where the tolerance is 0.
# gravity: a factor on the per-node RK45 bound of ``deviations``; density_eos: below the PALEOS
# table interpolation error of 1e-5.
TOLERANCES = {
    'padding': 0,
    'positive': 0,
    'pressure_monotone': 0.0,
    'mass_monotone': 0.0,
    'cmb_mass': 1e-12,
    'core_mantle_mass': 1e-12,
    'core_radius': 0,
    'gravity': 100.0,
    'density_eos': 1e-6,
}
_RTOL_LAYER = 1e-12  # the layer rule of structure_model.get_layer_mixture
_PROFILE = 'radii mass_enclosed gravity pressure density cmb_mass core_mantle_mass'.split()


class StructureInvariantError(Exception):
    """An accepted structure result breaks a physical invariant.

    Not a RuntimeError: a caller that keeps its previous structure after a failed
    solve must not catch it.
    """


def _layer(m, cmb_mass, core_mantle_mass, ice):
    """Layer name of each node with enclosed mass ``m``, by the rule of the solver."""
    top = np.where(ice & (m >= core_mantle_mass * (1.0 - _RTOL_LAYER)), 'ice_layer', 'mantle')
    return np.where(m < cmb_mass * (1.0 - _RTOL_LAYER), 'core', top)


def deviations(result, config_params, mass_limit, rtol, atol, density_at=None, ice=False):
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
        ``rtol + atol (1 / M_i + 1 / g_i)``. The RK45 dense output keeps ``g r^2 - G M`` per
        step to about 19.9 rtol + 13.6 atol (1/M_i + 1/g_i) relative to ``G M``, an estimate
        from the Butcher tableau, and an error made at one radius decays outward as
        ``(r_i / r)^2``, since the homogeneous part of ``dg/dr`` is ``-2 g / r``. The bound
        is per step and the factor 100 covers the sum over the steps: real solves of 0.1 to
        10 M_E planets (analytic and tabulated EOS, 150 and 2000 nodes) reach 3 to 14 times
        the per-step bound, a margin of at least 7. At rtol 1e-5 and atol 1e-6 the tolerance
        at the surface is about 1e-3 for any node count.
    density_at : callable or None
        ``(P, T, layer) -> rho`` for the arrays of the nodes of one layer, by the EOS path
        that the solver used at the returned nodes; called once per layer that has nodes.
        A reply that is not a finite number breaks the check. The density check is skipped
        when ``density_at`` is None or when ``result['density_from_eos']`` is False.
    ice : bool
        True when the planet has an ice layer.

    Returns
    -------
    dict
        {invariant: (deviation, node index, tolerance)}. The checks use the nodes
        before the first node with P <= 0; the nodes after it are padding with P = 0.
    """
    P, rho, m, r, g = (
        np.asarray(result[k], float)
        for k in ('pressure', 'density', 'mass_enclosed', 'radii', 'gravity')
    )
    T = result.get('temperature')
    T = np.full(len(P), 300.0) if T is None else np.asarray(T, float)
    ok = np.isfinite(P) & (P > 0)
    k = len(P) if ok.all() else int(np.argmin(ok))
    out = {}

    def put(name, dev, i, tol=None):
        out[name] = (float(dev), int(i), TOLERANCES[name] if tol is None else tol)

    put('padding', np.count_nonzero(P[k:] != 0), k)
    bad = ~(np.isfinite(rho[:k]) & (rho[:k] > 0) & np.isfinite(T[:k]) & (T[:k] > 0))
    put('positive', np.count_nonzero(bad) + (k < 2), int(np.argmax(bad)) if bad.any() else k)
    for name, x, sign in (('pressure_monotone', P, 1), ('mass_monotone', m, -1)):
        rise = sign * np.diff(x[:k]) / np.abs(x[:k]).max(initial=1.0)
        put(name, max(rise.max(initial=0.0), 0.0), int(np.argmax(rise)) + 1 if len(rise) else 0)
    M, cmf = config_params['planet_mass'], config_params['core_mass_fraction']
    mmf = config_params.get('mantle_mass_fraction', 0.0)
    for name, f in (('cmb_mass', cmf), ('core_mantle_mass', cmf + mmf)):
        if f > 0:
            put(name, abs(result[name] / (f * M) - 1), len(m) - 1)
    if cmf > 0:
        n_core = int(np.count_nonzero(m[:k] < result['cmb_mass'] * (1.0 - _RTOL_LAYER)))
        put('core_radius', not 2 <= n_core < k + (cmf == 1), n_core)  # all nodes only at cmf 1
    with np.errstate(divide='ignore', invalid='ignore'):
        dg = np.abs(g[1:k] * r[1:k] ** 2 / (G * m[1:k]) - 1)
        tol_g = TOLERANCES['gravity'] * (rtol + atol * (1 / m[1:k] + 1 / g[1:k]))
        # a node without mass or gravity breaks the check
        ok_g = (m[1:k] > 0) & (g[1:k] > 0) & np.isfinite(dg)
        ratio = np.where(ok_g, dg / tol_g, np.inf)
    i = int(np.argmax(ratio)) if len(ratio) else -1
    dev, tol = (dg[i], tol_g[i]) if i >= 0 and ok_g[i] else (np.inf, 0.0)
    put('gravity', dev, i + 1, tol)
    # the form of the accept test
    put('mass_closure', abs(m[-1] - M) / M, len(m) - 1, mass_limit)
    if density_at is not None and result.get('density_from_eos'):
        layer = _layer(m[:k], result['cmb_mass'], result['core_mantle_mass'], ice)
        ref = np.full(k, np.nan)
        for name in np.unique(layer):
            sel = layer == name
            ref[sel] = np.asarray(density_at(P[:k][sel], T[:k][sel], str(name)), dtype=float)
        with np.errstate(divide='ignore', invalid='ignore'):
            d = np.abs(rho[:k] / ref - 1)
        d[~np.isfinite(d)] = np.inf
        put('density_eos', d.max(initial=0.0), int(np.argmax(d)) if k else 0)
    return out


def _finite_positive(x):
    return isinstance(x, numbers.Real) and not isinstance(x, bool) and 0 < x < np.inf


def check_accepted(result, config_params, rtol, atol, density_at=None, ice=False):
    """Raise StructureInvariantError when an accepted result breaks an invariant.

    See ``deviations`` for the parameters; the mass limit is ``result['accept_mass_limit']``,
    which the accept path writes. A result with ``converged`` False is not checked.

    Raises
    ------
    StructureInvariantError
        Names the invariant, the node, the deviation, the tolerance and the
        ``accepted_on`` label; or the profile arrays that the result lacks; or an
        ``accept_mass_limit``, ``rtol`` or ``atol`` that is not a finite positive number.
    """
    if not result.get('converged'):
        return
    accepted_on = result.get('accepted_on')
    missing = [k for k in _PROFILE if result.get(k) is None]
    if missing:
        raise StructureInvariantError(
            f'Accepted structure ({accepted_on=}) has no {", ".join(missing)}.'
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
    for name, (dev, i, tol) in deviations(
        result, config_params, limit, rtol, atol, density_at, ice
    ).items():
        if not dev <= tol:
            raise StructureInvariantError(
                f'Accepted structure ({accepted_on=}) breaks '
                f'{name}: deviation {dev:.3e} > {tol:.1e} at node {i}.'
            )
