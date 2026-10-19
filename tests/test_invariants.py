"""Invariant checks on an accepted structure result (``zalmoxis.invariants``).

A uniform sphere is an exact accepted result: each test breaks one invariant and expects
the error that names it. ``main`` runs the check at its Picard and Newton exits.
"""

from __future__ import annotations

import numbers
from contextlib import nullcontext

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import zalmoxis.solver as zs
from tests.test_failed_structure_solve import _cfg, _run
from zalmoxis.config import load_material_dictionaries
from zalmoxis.constants import G, earth_mass
from zalmoxis.invariants import (
    _RTOL_LAYER,
    TOLERANCES,
    StructureInvariantError,
    _layer,
    check_accepted,
    deviations,
)
from zalmoxis.mixing import calculate_mixed_density, parse_all_layer_mixtures

RHO, N, CMF, RTOL, ATOL = 5500.0, 50, 0.325, 1e-9, 1e-10


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
    result['accept_mass_limit'] = limit
    check_accepted(result, cfg, RTOL, ATOL, lambda P, T, layer: RHO)


def _sphere_off(err):
    """The sphere as the result for a target mass ``1 + err`` times below its mass."""
    result, cfg = _sphere()
    cfg['planet_mass'] = result['mass_enclosed'][-1] / (1 + err)
    result['cmb_mass'] = result['core_mantle_mass'] = CMF * cfg['planet_mass']
    return result, cfg


def _raises(result, cfg, name, limit=1e-3):
    with pytest.raises(StructureInvariantError, match=f'breaks {name}:'):
        _check(result, cfg, limit)


@pytest.mark.unit
def test_the_sphere_passes_with_every_deviation_inside_its_tolerance():
    result, cfg = _sphere(mmf=0.5)
    devs = deviations(result, cfg, 1e-3, RTOL, ATOL, lambda P, T, layer: RHO)
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
    'gravity': _break('gravity', N // 2, 1 + 1e-5),
    'density_eos': _break('density', N // 2, 1 + 1e-5),
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
    result['cmb_mass'] *= 1.02
    result['core_mantle_mass'] *= 1.02
    _check(result, cfg, limit=0.03)
    _raises(result, cfg, 'mass_closure', limit=0.01)


@pytest.mark.unit
@pytest.mark.parametrize('key', ['pressure', 'cmb_mass', 'temperature'])
def test_a_converged_result_without_a_profile_raises(key):
    result, cfg = _sphere()
    del result[key]
    with pytest.raises(StructureInvariantError, match=f'has no {key}'):
        _check(result, cfg)


@pytest.mark.unit
def test_a_nan_raises():
    result, cfg = _sphere()
    result['gravity'][N - 2] = np.nan
    _raises(result, cfg, 'gravity')


@pytest.mark.unit
def test_a_nan_gravity_reports_an_infinite_deviation():
    result, cfg = _sphere()
    result['gravity'][N - 2] = np.nan
    dev, node, _ = deviations(result, cfg, 1e-3, RTOL, ATOL)['gravity']
    assert dev == np.inf and node == N - 2


@pytest.mark.unit
@pytest.mark.parametrize('key', ['cmb_mass', 'core_mantle_mass'])
def test_a_nan_layer_boundary_raises_the_error_that_names_it(key):
    result, cfg = _sphere(mmf=0.5)
    result[key] = np.nan
    _raises(result, cfg, key)


@pytest.mark.unit
def test_a_node_of_negative_mass_has_an_infinite_gravity_deviation():
    """g = G m / r^2 holds at a negative m too, so only the sign of m marks the node."""
    result, cfg = _sphere()
    result['mass_enclosed'][5] *= -1
    result['gravity'][5] *= -1
    dev, node, _ = deviations(result, cfg, 1e-3, RTOL, ATOL)['gravity']
    assert dev == np.inf and node == 5


@pytest.mark.unit
def test_density_layer_follows_cmb_mass():
    result, cfg = _sphere(mmf=0.5)
    m, P = result['mass_enclosed'][: N - 1], result['pressure'][: N - 1]  # node N - 1 has P = 0
    calls, result['accept_mass_limit'] = {}, 1e-3

    def density_at(P, T, layer):  # one call per layer, with the nodes of that layer
        calls[layer] = np.array(P)
        return np.full(len(P), RHO)

    check_accepted(result, cfg, RTOL, ATOL, density_at, ice=True)
    assert set(calls) == {'core', 'mantle', 'ice_layer'}
    core, ice = m < result['cmb_mass'], m >= result['core_mantle_mass']
    for layer, nodes in (('core', core), ('mantle', ~core & ~ice), ('ice_layer', ice)):
        assert nodes.any() and np.array_equal(calls[layer], P[nodes]), layer


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
    tol = deviations(*_sphere(), 1e-3, RTOL, ATOL, lambda P, T, layer: RHO)[name][2]
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


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(2400)
def test_cmb_mass_is_the_layer_boundary_of_the_returned_profile(monkeypatch):
    """Every pass splits the layers at cmf * planet_mass; at cmf 0.3 and 200 nodes a node
    lies between that mass and cmf * M[-1], so a cmb_mass of cmf * M[-1] puts it in the
    wrong layer."""
    used, real = [], zs.solve_structure
    monkeypatch.setattr(
        zs, 'solve_structure', lambda *a, **k: (used.append(a[1]), real(*a, **k))[1]
    )
    r = _run(_cfg(core_mass_fraction=0.3, num_layers=200, relative_tolerance=1e-8))
    m, P = np.asarray(r['mass_enclosed']), np.asarray(r['pressure'])
    target, last = 0.3 * earth_mass, 0.3 * m[-1]
    assert set(used) == {target}
    assert np.any((min(target, last) <= m) & (m < max(target, last)))  # the case discriminates
    assert r['cmb_mass'] == pytest.approx(target, rel=1e-12)
    mixes, md = (
        parse_all_layer_mixtures(_cfg()['layer_eos_config']),
        load_material_dictionaries(),
    )
    for i in np.flatnonzero(P > 0):
        layer = 'core' if m[i] < r['cmb_mass'] * (1 - 1e-12) else 'mantle'
        ref = calculate_mixed_density(
            P[i], r['temperature'][i], mixes[layer], md, None, None, {}
        )
        assert r['density'][i] == pytest.approx(ref, rel=1e-8), (i, layer)


def _spy_limits(monkeypatch):
    """Record the accept label and mass limit of every check that main runs."""
    limits, real = [], zs.check_accepted

    def spy(result, *args, **kwargs):
        limits.append((result['accepted_on'], result.get('accept_mass_limit')))
        return real(result, *args, **kwargs)

    monkeypatch.setattr(zs, 'check_accepted', spy)
    return limits


def _canned(label, converged=True, **flags):
    """The sphere as a ``_solve`` result with the accept label and flags given."""
    result, cfg = _sphere()
    result.update(accepted_on=label, converged=converged, density_from_eos=False, **flags)
    return result, cfg


SKIP = {'converged_density': True, 'converged_pressure': True, 'mass_error_at_accept': 0.05}


@pytest.mark.unit
def test_main_checks_retry_skip_with_its_own_limit(monkeypatch):
    monkeypatch.setattr(zs, '_solve', lambda *a, **k: _canned(None, False, **SKIP)[0])
    limits = _spy_limits(monkeypatch)
    _run(_cfg(planet_mass=_sphere()[1]['planet_mass']))
    assert limits == [('retry_skip', 0.07)]


@pytest.mark.unit
@pytest.mark.parametrize('label', ['strict', 'mass_only', 'max_iter', 'timeout'])
@pytest.mark.parametrize('limit', [0.0123, None])
def test_main_checks_the_limit_that_the_solve_wrote(monkeypatch, label, limit):
    """main does not derive a limit from the label; a result without one fails."""
    result = _canned(label)[0]
    if limit is not None:
        result['accept_mass_limit'] = limit
    monkeypatch.setattr(zs, '_solve', lambda *a, **k: result)
    limits = _spy_limits(monkeypatch)
    cfg = _cfg(planet_mass=_sphere()[1]['planet_mass'])
    if limit is None:
        with pytest.raises(StructureInvariantError, match='accept_mass_limit'):
            _run(cfg)
    else:
        _run(cfg)
    assert limits == [(label, limit)]


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(2400)
def test_main_checks_a_picard_result_with_tolerance_outer(monkeypatch):
    limits = _spy_limits(monkeypatch)
    r = _run(_cfg(wall_timeout=1e6, tolerance_outer=2e-3))
    assert limits == [(r['accepted_on'], 2e-3)] and r['accepted_on'] in ('strict', 'mass_only')


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(2400)
def test_main_raises_on_a_broken_picard_result(monkeypatch):
    real = zs._report_table_bands

    def bands(result, *args):
        result['cmb_mass'] = 0.0
        return real(result, *args)

    monkeypatch.setattr(zs, '_report_table_bands', bands)
    with pytest.raises(StructureInvariantError, match='breaks cmb_mass:'):
        _run(_cfg(wall_timeout=1e6))


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(2400)
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


@pytest.mark.unit
@pytest.mark.parametrize(
    'limit', ['missing', None, True, 0, 0.0, -1e-3, np.nan, np.inf, '0.05']
)
def test_an_accept_without_a_valid_mass_limit_raises(limit):
    result, cfg = _sphere()
    if not isinstance(limit, str) or limit != 'missing':
        result['accept_mass_limit'] = limit
    with pytest.raises(StructureInvariantError, match='accept_mass_limit'):
        check_accepted(result, cfg, RTOL, ATOL, lambda P, T, layer: RHO)


@pytest.mark.unit
@pytest.mark.parametrize(
    'limit, raises', [(0.06, False), (0.0401, False), (0.0399, True), (0.03, True)]
)
def test_mass_closure_reads_the_limit_of_the_result(limit, raises):
    """A 4 % mass error against the limit that the accept path wrote."""
    result, cfg = _sphere_off(0.04)
    if raises:
        _raises(result, cfg, 'mass_closure', limit)
    else:
        _check(result, cfg, limit)


@pytest.mark.unit
@settings(max_examples=80, deadline=None)
@given(
    limit=st.one_of(
        st.none(),
        st.booleans(),
        st.text(max_size=4),
        st.integers(max_value=0),
        st.integers(min_value=1, max_value=10**6),
        st.floats(max_value=0.0),
        st.floats(min_value=1e-300, max_value=0.0399),
        st.floats(min_value=0.0401, max_value=1e300),
        st.sampled_from(
            [np.nan, np.inf, -np.inf, np.float64(0.05), np.float32(0.05), np.int64(1)]
        ),
    )
)
def test_accept_mass_limit_domain(limit):
    """Only a finite positive number is a limit; it then decides the 4 % mass error."""
    result, cfg = _sphere_off(0.04)
    result['accept_mass_limit'] = limit
    number = isinstance(limit, numbers.Real) and not isinstance(limit, bool)
    if not (number and 0 < limit < np.inf):
        with pytest.raises(StructureInvariantError, match='accept_mass_limit'):
            check_accepted(result, cfg, RTOL, ATOL, lambda P, T, layer: RHO)
    elif limit < 0.04:
        with pytest.raises(StructureInvariantError, match='breaks mass_closure:'):
            check_accepted(result, cfg, RTOL, ATOL, lambda P, T, layer: RHO)
    else:
        check_accepted(result, cfg, RTOL, ATOL, lambda P, T, layer: RHO)


def _sphere_core(f):
    """The sphere with a core mass fraction ``f`` and no ice layer."""
    result, cfg = _sphere()
    cfg['core_mass_fraction'] = f
    result['cmb_mass'] = result['core_mantle_mass'] = f * cfg['planet_mass']
    return result, cfg


@pytest.mark.unit
def test_a_core_of_every_node_passes_only_at_core_mass_fraction_1():
    _check(*_sphere_core(1.0))
    _raises(*_sphere_core(0.97), 'core_radius')  # every node with P > 0 is below 0.97 M


@pytest.mark.unit
def test_a_core_of_one_node_raises():
    m = _sphere()[0]['mass_enclosed']
    _raises(*_sphere_core(0.5 * m[1] / m[-1]), 'core_radius')


@pytest.mark.unit
@settings(max_examples=60, deadline=None)
@given(u=st.floats(0.0, 1.0))
def test_core_radius_over_the_core_mass_fraction(u):
    """Nodes 0 and 1 at least, and every node with P > 0 only at core_mass_fraction 1."""
    m = _sphere()[0]['mass_enclosed'] / _sphere()[0]['mass_enclosed'][-1]
    lo, hi = 1.01 * m[1], 0.99 * m[N - 2]  # the last node with P > 0 is N - 2
    for f, raises in (
        (lo + u * (hi - lo), False),
        (u * 0.99 * m[1], True),
        (1.01 * m[N - 2] + u * (0.99 - m[N - 2]), True),
    ):
        if f <= 0:
            continue
        if raises:
            _raises(*_sphere_core(f), 'core_radius')
        else:
            _check(*_sphere_core(f))


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(2400)
def test_a_planet_of_core_only_passes_every_invariant():
    """core_mass_fraction 1.0: every node is core, with no mantle node."""
    r = _run(_cfg(wall_timeout=1e6, core_mass_fraction=1.0))
    assert r['converged'] and r['cmb_mass'] == pytest.approx(earth_mass, rel=1e-12)


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(2400)
@pytest.mark.parametrize('n', [150, 2000])  # the PROTEUS default node count, and a fine grid
@pytest.mark.parametrize('factor', [2.0, 0.5])
def test_a_gravity_error_at_one_inner_node_against_the_bound(monkeypatch, n, factor):
    """A real solve; the gravity of one inner node moves to ``factor`` times the bound of that
    node off G M / r^2: twice the bound raises, half of it passes."""
    real = zs._report_table_bands

    def bands(result, *args):
        result = real(result, *args)
        g, r, m = (np.array(result[k], float) for k in ('gravity', 'radii', 'mass_enclosed'))
        i = n // 2
        bound = TOLERANCES['gravity'] * (RTOL + ATOL * (1 / m[i] + 1 / g[i]))
        g[i] = G * m[i] / r[i] ** 2 * (1 + factor * bound)
        result['gravity'] = g
        return result

    monkeypatch.setattr(zs, '_report_table_bands', bands)
    cfg = _cfg(wall_timeout=1e6, num_layers=n, relative_tolerance=RTOL, absolute_tolerance=ATOL)
    if factor > 1:
        with pytest.raises(
            StructureInvariantError, match=f'breaks gravity: .* at node {n // 2}'
        ):
            _run(cfg)
    else:
        assert _run(cfg)['converged']


@pytest.mark.unit
def test_a_density_reference_of_none_raises():
    result, cfg = _sphere()
    result['accept_mass_limit'] = 1e-3
    with pytest.raises(StructureInvariantError, match='breaks density_eos:'):
        check_accepted(result, cfg, RTOL, ATOL, lambda P, T, layer: None)


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(2400)
def test_a_chabrier_h2_mantle_passes_every_invariant():
    """The 3 % H2 mantle of input/grids/h2_mixing.toml with a PALEOS iron core."""
    from tests.test_spider_coupling_convergence import _has_paleos_data

    if not _has_paleos_data():
        pytest.skip('PALEOS tables not on disk')
    layers = {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3:0.97+Chabrier:H:0.03'}
    r = _run(_cfg(wall_timeout=1e6, layer_eos_config=layers))
    assert r['converged'] and r['density_from_eos']


@pytest.mark.unit
def test_the_tolerances_are_the_documented_values():
    assert TOLERANCES == {
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
    assert _RTOL_LAYER == 1e-12


@pytest.mark.unit
@pytest.mark.parametrize(
    'key, value', [('gravity', 0.0), ('mass_enclosed', 0.0), ('gravity', np.nan)]
)
def test_a_first_node_without_gravity_or_mass_does_not_turn_the_gravity_check_off(key, value):
    """A node with a gravity or a mass of 0 or NaN breaks the check at that node. (A NaN M_1
    breaks mass_monotone first.)"""
    result, cfg = _sphere()
    result[key][1] = value
    _raises(result, cfg, 'gravity')


@pytest.mark.unit
@pytest.mark.parametrize(
    'field, node, value, name',
    [
        ('temperature', 7, 0.0, 'positive'),
        ('temperature', 7, -5.0, 'positive'),
        ('temperature', 7, np.nan, 'positive'),
        ('temperature', 7, np.inf, 'positive'),
        ('density', 7, 0.0, 'positive'),
        ('density', 7, np.nan, 'positive'),
        ('density', 7, np.inf, 'positive'),
        ('pressure', 20, np.inf, 'padding'),
    ],
)
def test_an_edge_value_at_one_node_raises(field, node, value, name):
    result, cfg = _sphere()
    result[field] = np.array(result[field], float)
    result[field][node] = value
    _raises(result, cfg, name)


@pytest.mark.unit
def test_a_profile_with_pressure_only_at_the_centre_raises_positive():
    result, cfg = _sphere()
    result['pressure'][1:] = 0.0
    _raises(result, cfg, 'positive')


@pytest.mark.unit
def test_the_layer_of_a_node_on_each_boundary():
    cmb, top = 2.0e24, 4.0e24
    assert _layer(cmb * (1 - 5e-12), cmb, top, True) == 'core'
    assert _layer(cmb * (1 - 1e-12), cmb, top, True) == 'mantle'
    assert _layer(cmb, cmb, top, True) == 'mantle'
    assert _layer(top * (1 - 5e-12), cmb, top, True) == 'mantle'
    assert _layer(top * (1 - 1e-12), cmb, top, True) == 'ice_layer'
    assert _layer(top, cmb, top, False) == 'mantle'


@pytest.mark.unit
def test_main_checks_a_retry_with_its_own_integration_tolerances(monkeypatch):
    """The retry runs with rtol and atol tightened 10x; the check reads those."""
    retry = {**_canned('strict')[0], 'accept_mass_limit': 1e-3}
    calls = [_canned(None, False)[0], retry]
    monkeypatch.setattr(zs, '_solve', lambda *a, **k: calls.pop(0))
    seen = []
    monkeypatch.setattr(
        zs, 'check_accepted', lambda r, c, rtol, atol, *a, **k: seen.append((rtol, atol))
    )
    cfg = _cfg(
        planet_mass=_sphere()[1]['planet_mass'],
        relative_tolerance=1e-9,
        absolute_tolerance=1e-10,
    )
    _run(cfg)
    assert seen == [(pytest.approx(1e-10, rel=1e-12), pytest.approx(1e-11, rel=1e-12))]


@pytest.mark.unit
def test_the_density_check_gives_the_volatile_profile_only_to_the_mantle(monkeypatch):
    """One batch call per layer of a 3-layer planet; only the mantle call gets the profile."""
    result, cfg = _sphere(mmf=0.5)
    result['accept_mass_limit'] = 1e-3
    layers = {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3', 'ice_layer': 'Analytic:H2O'}
    mixes, seen, vp = parse_all_layer_mixtures(layers), {}, object()

    def batch(P, T, mix, *args, volatile_profile=None, **kwargs):
        seen[next(name for name, x in mixes.items() if x is mix)] = volatile_profile
        return np.full(len(P), RHO)

    monkeypatch.setattr(zs, 'calculate_mixed_density_batch', batch)
    run = {'relative_tolerance': RTOL, 'absolute_tolerance': ATOL}
    zs._check_accepted(result, run, {**cfg, 'layer_eos_config': layers}, {}, None, mixes, vp)
    assert seen == {'core': None, 'mantle': vp, 'ice_layer': None}


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(2400)
def test_a_max_iter_accept_returns_the_pass_it_tested(monkeypatch):
    """Two passes at 1.25 % and 10.1 % mass error (mass and gravity scaled together, so the
    gravity invariant holds): max_iter accepts on the best pass and returns that pass."""
    radius, real_tp, real = [], zs.calculate_temperature_profile, zs.solve_structure
    errors = (0.0125, 0.101)

    def tp(radii, *args, **kwargs):
        radius.append(radii[-1])
        return real_tp(radii, *args, **kwargs)

    def solve(*args, **kwargs):
        m, g, P = real(*args, **kwargs)
        f = earth_mass * (1 + errors[min(len(radius), 2) - 1]) / m[-1]
        return m * f, g * f, P

    monkeypatch.setattr(zs, 'calculate_temperature_profile', tp)
    monkeypatch.setattr(zs, 'solve_structure', solve)
    r = _run(_cfg(wall_timeout=1e6, max_iterations_outer=2))
    assert r['converged'] and r['accepted_on'] == 'max_iter' and len(radius) == 2
    assert r['radii'][-1] == pytest.approx(radius[0], rel=1e-12)
    assert r['mass_error_at_accept'] == pytest.approx(0.0125, rel=1e-9)
    assert r['mass_error_at_accept'] == pytest.approx(r['best_mass_error'], rel=1e-12)


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(2400)
def test_a_1_percent_gravity_error_is_caught_on_a_fine_grid(monkeypatch):
    """2000 nodes at the default tolerances (rtol 1e-5, atol 1e-6): a 1 % gravity error at a
    node near the surface raises."""
    real = zs._report_table_bands

    def bands(result, *args):
        result = real(result, *args)
        result['gravity'] = np.array(result['gravity'], float)
        result['gravity'][1990] *= 1.01
        return result

    monkeypatch.setattr(zs, '_report_table_bands', bands)
    cfg = _cfg(
        wall_timeout=1e6, num_layers=2000, relative_tolerance=1e-5, absolute_tolerance=1e-6
    )
    with pytest.raises(StructureInvariantError, match='breaks gravity: .* at node 1990'):
        _run(cfg)


@pytest.mark.unit
@pytest.mark.parametrize(
    'rtol, atol', [(1e-9, 1e-10), (1e-5, 1e-6), (1e-9, 1e-30), (1e-30, 1e-10)]
)
@pytest.mark.parametrize('node', [2, N // 2, N - 2])
@pytest.mark.parametrize('factor, raises', [(2.0, True), (0.5, False)])
def test_the_gravity_tolerance_of_each_node(rtol, atol, node, factor, raises):
    """The tolerance at node i is 100 (rtol + atol (1/M_i + 1/g_i))."""
    result, cfg = _sphere()
    m, g = result['mass_enclosed'], result['gravity']
    g[node] *= 1 + factor * 100.0 * (rtol + atol * (1 / m[node] + 1 / g[node]))
    result['accept_mass_limit'] = 1e-3
    run = lambda: check_accepted(result, cfg, rtol, atol, lambda P, T, layer: RHO)  # noqa: E731
    if raises:
        with pytest.raises(StructureInvariantError, match=f'breaks gravity: .* at node {node}'):
            run()
    else:
        run()


@pytest.mark.unit
@pytest.mark.parametrize('name', ['cmb_mass', 'core_mantle_mass', 'density_eos'])
@pytest.mark.parametrize('factor, raises', [(2.0, True), (0.5, False)])
def test_each_tolerance_at_twice_and_half_its_value(name, factor, raises):
    result, cfg = _sphere(mmf=0.5)
    tol = {'cmb_mass': 1e-12, 'core_mantle_mass': 1e-12, 'density_eos': 1e-6}[name]
    if name == 'density_eos':
        result['density'][N // 2] *= 1 + factor * tol
    else:
        result[name] *= 1 + factor * tol
    if raises:
        _raises(result, cfg, name)
    else:
        _check(result, cfg)


@pytest.mark.unit
@pytest.mark.parametrize('which', ['rtol', 'atol'])
@pytest.mark.parametrize('value', [0.0, -1e-9, np.nan, np.inf, None, True, '1e-9'])
def test_an_invalid_integration_tolerance_raises(which, value):
    result, cfg = _sphere()
    result['accept_mass_limit'] = 1e-3
    tols = {'rtol': RTOL, 'atol': ATOL, which: value}
    with pytest.raises(StructureInvariantError, match='integration tolerances'):
        check_accepted(result, cfg, tols['rtol'], tols['atol'], lambda P, T, layer: RHO)


@pytest.mark.unit
@settings(max_examples=80, deadline=None)
@given(
    rtol=st.floats(1e-12, 1e-3),
    atol=st.floats(1e-14, 1e-3),
    node=st.integers(1, N - 2),
    raises=st.booleans(),
)
def test_the_per_node_gravity_bound_over_the_tolerance_domain(rtol, atol, node, raises):
    """Twice the bound of a node raises at that node, half of it passes."""
    result, cfg = _sphere()
    m, g = result['mass_enclosed'], result['gravity']
    g[node] *= 1 + (2.0 if raises else 0.5) * 100.0 * (
        rtol + atol * (1 / m[node] + 1 / g[node])
    )
    result['accept_mass_limit'] = 1e-3
    expect = pytest.raises(StructureInvariantError, match=f'breaks gravity: .* at node {node}')
    with expect if raises else nullcontext():
        check_accepted(result, cfg, rtol, atol, lambda P, T, layer: RHO)


@pytest.mark.unit
def test_the_gravity_check_reports_the_node_with_the_largest_ratio():
    """A deviation of 0.9 times the loose bound of an inner node and of 2 times the tight bound
    of an outer node: the outer node breaks the check, though its deviation is smaller."""
    result, cfg = _sphere()
    m, g = result['mass_enclosed'], result['gravity']
    rtol, atol = 1e-9, 1e-8
    bound = lambda i: 100.0 * (rtol + atol * (1 / m[i] + 1 / g[i]))  # noqa: E731
    assert 0.9 * bound(2) > 2 * bound(N - 2)
    g[2] *= 1 + 0.9 * bound(2)
    g[N - 2] *= 1 + 2 * bound(N - 2)
    result['accept_mass_limit'] = 1e-3
    with pytest.raises(StructureInvariantError, match=f'breaks gravity: .* at node {N - 2}'):
        check_accepted(result, cfg, rtol, atol, lambda P, T, layer: RHO)


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(2400)
def test_a_relative_tolerance_below_the_floor_of_the_integrator_passes():
    """scipy raises an rtol below 100 eps to 100 eps; the check uses the rtol the integration
    ran with."""
    cfg = _cfg(
        wall_timeout=1e6, relative_tolerance=1e-15, absolute_tolerance=1e-20, num_layers=150
    )
    assert _run(cfg)['converged']


@pytest.mark.unit
def test_the_mass_closure_uses_the_form_of_the_accept_test():
    """|M - M_p| / M_p is just below 1e-3 here while |M / M_p - 1| rounds above it: a result
    that the accept test takes at 1e-3 must pass the closure at 1e-3."""
    M, Mp = 2.9831139e25, 2.9861e25
    assert abs(M - Mp) / Mp < 1e-3 < abs(M / Mp - 1)
    result, cfg = _sphere()
    f = M / result['mass_enclosed'][-1]
    result['mass_enclosed'] = result['mass_enclosed'] * f
    result['gravity'] = result['gravity'] * f
    result['mass_enclosed'][-1] = M  # the padded node, outside the gravity check
    cfg['planet_mass'] = Mp
    result['cmb_mass'] = result['core_mantle_mass'] = CMF * Mp
    _check(result, cfg, 1e-3)


@pytest.mark.unit
def test_the_layer_rule_is_that_of_the_structure_model():
    from zalmoxis.structure_model import get_layer_mixture

    cmb, top = 2.0e24, 4.0e24
    mixes = {'core': 'core', 'mantle': 'mantle', 'ice_layer': 'ice_layer'}
    for x in (cmb, top):
        for m in x * (1 + np.array([-1e-11, -5e-12, -1e-12, -5e-13, 0.0, 1e-12])):
            for ice in (True, False):
                layers = mixes if ice else {k: v for k, v in mixes.items() if k != 'ice_layer'}
                assert _layer(m, cmb, top, ice) == get_layer_mixture(m, cmb, top, layers), (
                    m,
                    ice,
                )


@pytest.mark.unit
@pytest.mark.parametrize('node', [1, N // 2, N - 2])
def test_a_gravity_near_zero_breaks_the_check(node):
    """The atol term uses G M_i / r_i^2, not the gravity under test, so a gravity scaled
    down by 1e-6 does not widen its own tolerance."""
    result, cfg = _sphere()
    result['gravity'][node] *= 1e-6
    result['accept_mass_limit'] = 1e-3
    with pytest.raises(StructureInvariantError, match=f'breaks gravity: .* at node {node}'):
        check_accepted(result, cfg, 1e-5, 1e-6, lambda P, T, layer: RHO)


def _malformed_arrays(result):
    result['pressure'] = result['pressure'].reshape(5, 10)


def _short_radii(result):
    result['radii'] = result['radii'][:-1]


def _one_node(result):
    for key in ('radii', 'mass_enclosed', 'gravity', 'pressure', 'density', 'temperature'):
        result[key] = result[key][:1]


def _vector_cmb(result):
    result['cmb_mass'] = np.array([result['cmb_mass']] * 2)


@pytest.mark.unit
@pytest.mark.parametrize('edit', [_malformed_arrays, _short_radii, _one_node, _vector_cmb])
def test_a_malformed_result_raises_the_invariant_error(edit):
    """A ValueError, IndexError or TypeError would be caught and rolled back by the caller."""
    result, cfg = _sphere()
    edit(result)
    with pytest.raises(StructureInvariantError, match='malformed'):
        _check(result, cfg)


@pytest.mark.unit
def test_a_failing_density_lookup_raises_the_invariant_error_with_its_cause():
    result, cfg = _sphere()
    result['accept_mass_limit'] = 1e-3

    def broken(P, T, layer):
        raise ValueError('table lookup failed')

    with pytest.raises(StructureInvariantError, match='density lookup') as info:
        check_accepted(result, cfg, RTOL, ATOL, broken)
    assert isinstance(info.value.__cause__, ValueError)
