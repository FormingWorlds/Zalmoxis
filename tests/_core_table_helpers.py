"""Shared inputs and checks of the core temperature table tests."""

from __future__ import annotations

import numpy as np
from scipy.integrate import trapezoid

from tests.test_jax_temperature_arrays import _INPUT_DIR, _chili_like_profile, _load_fixtures
from zalmoxis import solver as zs
from zalmoxis.structure_model import core_temperature, is_core

STEP = 150.0  # K: the first table point lies this far above the mantle T at the CMB
CLOSURE = 2e-3  # core mass closure; an ODE on the mantle T in the core misses by 7.7e-3


def _table(t_cmb, p_lo=1.0e11, p_hi=4.5e11):
    """A 3-point core table [Pa, K], kinked so the end slopes differ, at T_cmb + STEP at p_lo."""
    p = np.array([p_lo, 0.5 * (p_lo + p_hi), p_hi])
    return p, t_cmb + STEP + np.array([0.0, 700.0, 1000.0])


def _solve(jax=True, table=None, check=True, **over):
    """1 M_E PALEOS iron core and PALEOS-2phase mantle with the r-indexed CHILI-like mantle T;
    ``table(t_cmb)`` gives the core table."""
    cp, mat, mf = _load_fixtures()
    cp.update(use_jax=jax, **over)
    r, t, _, t_cmb = _chili_like_profile()
    kw = {} if table is None else {'core_temperature_arrays': table(t_cmb)}
    res = zs.main(
        cp,
        mat,
        mf,
        _INPUT_DIR,
        temperature_function=lambda r_, p_: float(np.interp(r_, r, t)),
        temperature_arrays=(r, t),
        **kw,
    )
    assert res['converged'] or not check, res.get('accepted_on')
    return res, (r, t)


def _key(res):
    """R, P_c, P_cmb and rho_c of a result."""
    p_cmb = np.interp(res['cmb_mass'], res['mass_enclosed'], res['pressure'])
    return res['radii'][-1], res['pressure'][0], p_cmb, res['density'][0]


def _layers(res):
    """Core mask, pressure and temperature of the returned nodes."""
    core = is_core(np.asarray(res['mass_enclosed']), res['cmb_mass'])
    return core, np.asarray(res['pressure'], float), np.asarray(res['temperature'], float)


def _mass_closure(res):
    """Relative miss of the integral of 4 pi r^2 rho over the returned core nodes against the
    enclosed mass at the top core node: an ODE that read another T than the nodes misses it."""
    core, _, _ = _layers(res)
    r, rho = np.asarray(res['radii'])[core], np.asarray(res['density'])[core]
    return trapezoid(4.0 * np.pi * r**2 * rho, r) / res['mass_enclosed'][core][-1] - 1.0


def _assert_node_rule(res, mantle, table):
    """Core nodes carry table(P), mantle nodes the mantle arrays, both to a relative 1e-9."""
    core, p, t = _layers(res)
    np.testing.assert_allclose(t[core], core_temperature(p[core], table), rtol=1e-9)
    r = np.asarray(res['radii'])
    np.testing.assert_allclose(t[~core], np.interp(r[~core], *mantle), rtol=1e-9)
