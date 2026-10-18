"""Invariant checks on an accepted structure result (``zalmoxis.invariants``).

A uniform sphere is an exact accepted result: each test breaks one invariant and expects
the error that names it. ``main`` runs the check at its Picard and Newton exits.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import zalmoxis.solver as zs
from tests.test_failed_structure_solve import _cfg, _run
from zalmoxis.constants import G
from zalmoxis.invariants import (
    TOLERANCES,
    StructureInvariantError,
    check_accepted,
    deviations,
)

RHO, N, CMF = 5500.0, 50, 0.325


def _sphere(mmf=0.0):
    """Accepted result of a uniform sphere of density RHO and radius 6.4e6 m: exact m, g
    and P; the last node has P = 0 and density 0, as the solver returns it."""
    R = 6.4e6
    r = np.linspace(0.0, R, N)
    m = 4 / 3 * np.pi * RHO * r**3
    g = np.zeros(N)
    g[1:] = G * m[1:] / r[1:] ** 2
    rho = np.full(N, RHO)
    rho[-1] = 0.0
    result = {
        'radii': r,
        'mass_enclosed': m,
        'gravity': g,
        'pressure': 2 / 3 * np.pi * G * RHO**2 * (R**2 - r**2),
        'density': rho,
        'temperature': np.full(N, 3000.0),
        'cmb_mass': CMF * m[-1],
        'core_mantle_mass': (CMF + mmf) * m[-1],
        'converged': True,
        'accepted_on': 'strict',
        'density_from_eos': True,
    }
    cfg = {'planet_mass': m[-1], 'core_mass_fraction': CMF, 'mantle_mass_fraction': mmf}
    return result, cfg


def _check(result, cfg, limit=1e-3):
    check_accepted(result, cfg, limit, lambda P, T, layer: RHO)


def _raises(result, cfg, name, limit=1e-3):
    with pytest.raises(StructureInvariantError, match=f'breaks {name}:'):
        _check(result, cfg, limit)


@pytest.mark.unit
def test_the_sphere_passes_with_every_deviation_inside_its_tolerance():
    result, cfg = _sphere(mmf=0.5)
    devs = deviations(result, cfg, 1e-3, lambda P, T, layer: RHO)
    assert set(devs) == set(TOLERANCES) | {'mass_closure'}
    assert all(dev <= tol for dev, _, tol in devs.values()), devs
    _check(result, cfg)


@pytest.mark.unit
def test_the_error_is_no_runtime_value_or_arithmetic_error():
    # PROTEUS keeps its previous structure after a RuntimeError of a solve
    for base in (RuntimeError, ValueError, ArithmeticError):
        assert not issubclass(StructureInvariantError, base)


def _break(field, i, factor):
    def edit(result, cfg):
        result[field] = np.array(result[field], float)
        result[field][i] *= factor

    return edit


def _set(field, value):
    def edit(result, cfg):
        result[field] = value(result)

    return edit


def _beyond_surface(result, cfg):
    """cmb_mass consistent with mass_enclosed[-1] but beyond the last node with P > 0: the
    CMB index is 0 for a caller that reads argmax(m >= cmb_mass), so R_core = 0."""
    m = result['mass_enclosed']
    m[-1] = 4 * m[-2]
    result['cmb_mass'] = CMF * m[-1]
    result['core_mantle_mass'] = (CMF + cfg['mantle_mass_fraction']) * m[-1]
    cfg['planet_mass'] = m[-1]


BREAKS = {
    'padding': _break('pressure', N - 3, 0.0),
    'positive': _break('density', 7, -1.0),
    'pressure_monotone': _set(
        'pressure',
        lambda r: np.where(np.arange(N) == 9, r['pressure'][8] * 1.01, r['pressure']),
    ),
    'mass_monotone': _set(
        'mass_enclosed',
        lambda r: np.where(np.arange(N) == 9, r['mass_enclosed'][8] * 0.99, r['mass_enclosed']),
    ),
    'cmb_mass': _set('cmb_mass', lambda r: r['cmb_mass'] * (1 + 1e-9)),
    'core_mantle_mass': _set('core_mantle_mass', lambda r: r['core_mantle_mass'] * (1 + 1e-9)),
    'core_radius': _beyond_surface,
    'surface_gravity': _break('gravity', N - 2, 1 + 1e-5),
    'gravity': _break('gravity', N // 2, 1 + 1e-5),
    'density_eos': _break('density', N // 2, 1 + 1e-7),
}


@pytest.mark.unit
@pytest.mark.parametrize('name', BREAKS)
def test_a_broken_invariant_raises_the_error_that_names_it(name):
    result, cfg = _sphere(mmf=0.5)
    BREAKS[name](result, cfg)
    _raises(result, cfg, name)


@pytest.mark.unit
def test_cmb_mass_zero_raises():
    result, cfg = _sphere()
    result['cmb_mass'] = 0.0
    _raises(result, cfg, 'cmb_mass')


@pytest.mark.unit
def test_mass_closure_reads_the_limit_of_the_accept_path():
    result, cfg = _sphere()
    cfg['planet_mass'] *= 1.02
    _check(result, cfg, limit=0.03)
    _raises(result, cfg, 'mass_closure', limit=0.01)


@pytest.mark.unit
def test_a_nan_raises():
    result, cfg = _sphere()
    result['gravity'][N - 2] = np.nan
    _raises(result, cfg, 'surface_gravity')


@pytest.mark.unit
def test_density_layer_follows_cmb_mass():
    result, cfg = _sphere(mmf=0.5)
    m = result['mass_enclosed']
    layers = []
    check_accepted(result, cfg, 1e-3, lambda P, T, layer: layers.append(layer) or RHO, ice=True)
    core = m[: len(layers)] < result['cmb_mass']
    ice = m[: len(layers)] >= result['core_mantle_mass']
    assert layers == [
        'core' if c else 'ice_layer' if i else 'mantle' for c, i in zip(core, ice)
    ]
    assert {'core', 'mantle', 'ice_layer'} <= set(layers)


@pytest.mark.unit
def test_an_unconverged_result_or_a_picard_density_is_not_checked():
    result, cfg = _sphere()
    result['density'][N // 2] *= 2
    result['density_from_eos'] = False
    _check(result, cfg)
    result['converged'], result['cmb_mass'] = False, 0.0
    _check(result, cfg)


@pytest.mark.unit
@settings(max_examples=60, deadline=None)
@given(
    field=st.sampled_from(['gravity', 'density', 'cmb_mass']),
    node=st.integers(1, N - 2),
    over=st.floats(2.0, 1e6),
    sign=st.sampled_from([-1, 1]),
)
def test_a_change_above_the_tolerance_raises_and_one_below_does_not(field, node, over, sign):
    """A relative change of ``over`` times the tolerance raises, one of 1/4 of it does not."""
    name = {'gravity': 'gravity', 'density': 'density_eos', 'cmb_mass': 'cmb_mass'}[field]
    tol = TOLERANCES[name]
    for factor, raises in ((over, True), (0.25, False)):
        result, cfg = _sphere()
        x = 1 + sign * min(factor * tol, 0.5)
        if field == 'cmb_mass':
            result['cmb_mass'] *= x
        else:
            result[field][node] *= x
        if raises:
            with pytest.raises(StructureInvariantError):
                _check(result, cfg)
        else:
            _check(result, cfg)


def _spy_limits(monkeypatch):
    """Record the mass limit of every check that main runs."""
    limits, real = [], zs.check_accepted

    def spy(result, cfg, limit, *args, **kwargs):
        limits.append((result['accepted_on'], limit))
        return real(result, cfg, limit, *args, **kwargs)

    monkeypatch.setattr(zs, 'check_accepted', spy)
    return limits


def _canned(label, converged=True, **flags):
    """The sphere as a ``_solve`` result with the accept label and flags given."""
    result, cfg = _sphere()
    result.update(accepted_on=label, converged=converged, density_from_eos=False, **flags)
    return result, cfg


TOL0 = zs._default_solver_params(_sphere()[1]['planet_mass'])['tolerance_outer']
SKIP = {'converged_density': True, 'converged_pressure': True, 'best_mass_error': 0.05}


@pytest.mark.unit
@pytest.mark.parametrize(
    'first, retry, limit',
    [
        (('strict',), None, TOL0),
        (('mass_only',), None, TOL0),
        (('max_iter',), None, zs._TIMEOUT_ACCEPT_MASS_ERR),
        (('oscillation',), None, zs._TIMEOUT_ACCEPT_MASS_ERR),
        (('timeout',), None, zs._TIMEOUT_ACCEPT_MASS_ERR),
        ((None, False), None, 0.07),  # retry_skip
        ((None, False), ('strict',), 0.1 * TOL0),  # the retry has tightened tolerances
    ],
)
def test_main_checks_with_the_limit_of_the_accept_path(monkeypatch, first, retry, limit):
    calls = [first, retry]

    def solve(*args, **kwargs):
        return _canned(*calls.pop(0), **SKIP if retry is None else {})[0]

    monkeypatch.setattr(zs, '_solve', solve)
    limits = _spy_limits(monkeypatch)
    _run(_cfg(planet_mass=_sphere()[1]['planet_mass']))
    assert [x for _, x in limits] == [pytest.approx(limit, rel=1e-12)]


@pytest.mark.smoke
@pytest.mark.timeout(600)
def test_main_checks_a_picard_result_with_tolerance_outer(monkeypatch):
    limits = _spy_limits(monkeypatch)
    r = _run(_cfg(wall_timeout=1e6, tolerance_outer=2e-3))
    assert limits == [(r['accepted_on'], 2e-3)] and r['accepted_on'] in ('strict', 'mass_only')


@pytest.mark.smoke
@pytest.mark.timeout(600)
def test_main_raises_on_a_broken_picard_result(monkeypatch):
    real = zs._report_table_bands

    def bands(result, *args):
        result['cmb_mass'] = 0.0
        return real(result, *args)

    monkeypatch.setattr(zs, '_report_table_bands', bands)
    with pytest.raises(StructureInvariantError, match='breaks cmb_mass:'):
        _run(_cfg(wall_timeout=1e6))


@pytest.mark.smoke
@pytest.mark.timeout(900)
def test_main_checks_a_newton_result_with_newton_tol(monkeypatch):
    limits = _spy_limits(monkeypatch)
    cfg = _cfg(
        wall_timeout=1e6,
        outer_solver='newton',
        newton_tol=2e-4,
        relative_tolerance=1e-9,
        absolute_tolerance=1e-10,
    )
    r = _run(cfg)
    assert r['converged'] and limits == [(r['accepted_on'], 2e-4)]
