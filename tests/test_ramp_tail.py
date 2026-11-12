"""The secant tail and the seeded start of the adiabat blend ramp: ``_secant_radius``,
``_adiabat_seed`` and how ``main`` routes a seed."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import zalmoxis.solver as zs
from tests.test_accepted_on import (  # noqa: F401
    EXTERNAL_T,
    M_T,
    _adiabatic_cfg,
    _inner,
    canned,
    no_check,
)
from zalmoxis.constants import earth_mass, earth_radius

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

A, B = 6.0e18, -3.6e25  # M(R) = A R + B = M_T at R = 7e6 m
FALLBACK = 1.234e6  # a radius no secant of these tests returns
N = 30


def _lin(R):
    return A * R + B


def test_the_secant_finds_the_root_of_a_linear_mass():
    """M(R) linear: the secant through 2 passes is the root 7e6 m, from either side, and not the
    step with the wrong sign (7.1e6 m from 7.05e6 m)."""
    for R, R_prev in ((7.05e6, 7.08e6), (6.95e6, 6.93e6), (7.01e6, 6.99e6)):
        got = zs._secant_radius(R, _lin(R), R_prev, _lin(R_prev), M_T, FALLBACK)
        assert got == pytest.approx(7.0e6, rel=1e-12)
    wrong = 7.05e6 + (_lin(7.05e6) - M_T) / A
    assert abs(wrong - 7.0e6) > 1e3


@pytest.mark.parametrize('R, R_prev, bound', [(7.5e6, 7.6e6, 0.98), (6.5e6, 6.4e6, 1.02)])
def test_a_root_beyond_2_percent_is_clamped(R, R_prev, bound):
    """A root 7 % below or 8 % above the radius of the pass stops at 2 % from it."""
    got = zs._secant_radius(R, _lin(R), R_prev, _lin(R_prev), M_T, FALLBACK)
    assert got == pytest.approx(bound * R, rel=1e-15)
    assert zs._SECANT_CLAMP == 0.02


NAN, INF = float('nan'), float('inf')


@pytest.mark.parametrize(
    'R, M, R_prev, M_prev, M_target',
    [
        (7.05e6, _lin(7.05e6), 7.05e6, _lin(7.05e6), 0.0),  # equal radii
        (7.05e6, 1.0e23, 7.08e6, 1.0e23, 0.0),  # equal masses: slope 0
        (7.05e6, _lin(7.08e6), 7.08e6, _lin(7.05e6), 0.0),  # mass falls with radius
        (NAN, 1.0e23, 7.08e6, 1.0e24, 0.0),
        (7.05e6, NAN, 7.08e6, 1.0e24, 0.0),
        (7.05e6, 1.0e23, NAN, 1.0e24, 0.0),
        (7.05e6, 1.0e23, 7.08e6, NAN, 0.0),
        (7.05e6, 1.0e23, 7.08e6, 1.0e24, NAN),
        (INF, 1.0e23, 7.08e6, 1.0e24, 0.0),
        (7.05e6, 1.0e23, 7.08e6, -INF, 0.0),
        (0.0, 1.0e23, 7.08e6, 1.0e24, 0.0),  # R = 0
        (-7.05e6, 1.0e23, 7.08e6, 1.0e24, 0.0),  # R < 0
        (7.05e6, 1.0e300, 7.05e6 * (1 + 1e-15), -1.0e300, 0.0),  # slope overflows
    ],
)
def test_an_unusable_secant_returns_the_fallback(R, M, R_prev, M_prev, M_target):
    """Equal radii, a slope that is zero, negative or not finite, a non-finite input and a
    non-positive radius all give the cube-root radius of the caller."""
    assert zs._secant_radius(R, M, R_prev, M_prev, M_target, FALLBACK) == FALLBACK


FLOATS = st.one_of(
    st.floats(allow_nan=True, allow_infinity=True),
    st.floats(1e5, 1e8),
    st.sampled_from([0.0, -1.0, True, False]),
)


@settings(max_examples=300, deadline=None)
@given(R=FLOATS, M=FLOATS, R_prev=FLOATS, M_prev=FLOATS, M_target=FLOATS)
def test_the_secant_radius_is_the_fallback_or_within_the_clamp(R, M, R_prev, M_prev, M_target):
    """Any input, including NaN, inf, zero, negative values and bools: the result is the
    fallback or a finite positive radius within 2 % of R."""
    got = zs._secant_radius(R, M, R_prev, M_prev, M_target, FALLBACK)
    if got != FALLBACK:
        assert np.isfinite(got) and got > 0
        assert 0.98 * R - 1e-9 * R <= got <= 1.02 * R + 1e-9 * R


def _seed(n=N, R=6.4e6):
    """A seed of a 1 Earth-mass profile: radii up, pressure down to 0, mass up to M."""
    r = np.linspace(0.0, R, n)
    return {
        'radii': r,
        'pressure': 3.6e11 * (1.0 - (r / R) ** 2),
        'mass_enclosed': earth_mass * (r / R) ** 3,
    }


def _cfg(**kw):
    return {'planet_mass': earth_mass, 'num_layers': N, **kw}


def test_a_valid_seed_is_kept_as_float_copies():
    """A valid seed comes back as float arrays equal to the input and not the same objects."""
    seed = _seed()
    got = zs._adiabat_seed(seed, _cfg())
    assert set(got) == {'radii', 'pressure', 'mass_enclosed'}
    for k, a in seed.items():
        np.testing.assert_array_equal(got[k], a)
        assert got[k] is not a and got[k].dtype == np.float64


def _bad(key, fn):
    seed = _seed()
    seed[key] = fn(seed[key].copy())
    return seed


def _set(i, v):
    def f(a):
        a[i] = v
        return a

    return f


REFUSED = {
    'missing key': ({k: v for k, v in _seed().items() if k != 'pressure'}, 'not readable'),
    'not a mapping': ([1.0, 2.0], 'not readable'),
    'None': (None, 'not readable'),
    'a string': ('seed', 'not readable'),
    'ragged': (_bad('radii', lambda a: [a[:3], a[:2]]), 'not readable'),
    'short': (_bad('radii', lambda a: a[:-1]), 'shapes'),
    'empty': (_bad('mass_enclosed', lambda a: a[:0]), 'shapes'),
    '2-D': (_bad('pressure', lambda a: a.reshape(5, 6)), 'shapes'),
    'a bool': (_bad('pressure', lambda a: True), 'shapes'),
    'NaN radius': (_bad('radii', _set(4, NAN)), 'not finite'),
    'inf pressure': (_bad('pressure', _set(0, INF)), 'not finite'),
    'NaN mass': (_bad('mass_enclosed', _set(-1, NAN)), 'not finite'),
    'radii fall': (_bad('radii', _set(5, 0.0)), 'radii do not increase'),
    'negative r[0]': (_bad('radii', _set(0, -1.0)), 'radii do not increase'),
    'mass falls': (_bad('mass_enclosed', _set(10, 0.0)), 'mass_enclosed decreases'),
    'zero mass': (_bad('mass_enclosed', lambda a: 0.0 * a), 'mass_enclosed decreases'),
    'pressure rises': (_bad('pressure', _set(1, 4.0e11)), 'pressure increases'),
    'negative P': (_bad('pressure', _set(-1, -1.0)), 'pressure increases'),
    'zero P': (_bad('pressure', lambda a: 0.0 * a), 'pressure increases'),
    'other planet mass': (_bad('mass_enclosed', lambda a: 1.2 * a), 'planet mass'),
    'mass in Earth masses': (_bad('mass_enclosed', lambda a: a / earth_mass), 'planet mass'),
    'radius x0.1': (_seed(R=0.1 * earth_radius), 'outer radius'),
    'radius x10': (_seed(R=10.0 * earth_radius), 'outer radius'),
}


@pytest.mark.parametrize('seed, fault', list(REFUSED.values()), ids=list(REFUSED))
def test_a_seed_that_cannot_start_a_solve_is_refused_with_its_fault(seed, fault, caplog):
    """Every fault of the content refuses the seed and names itself in one warning."""
    with caplog.at_level(logging.WARNING, logger='fwl.zalmoxis.solver'):
        assert zs._adiabat_seed(seed, _cfg()) is None
    (rec,) = [r for r in caplog.records if 'adiabat_seed refused' in r.getMessage()]
    assert fault in rec.getMessage()


def test_the_radius_window_is_open_at_0_2_and_5_times_the_estimate():
    """The outer radius must lie strictly between 0.2 and 5 times the Seager estimate: just
    inside passes, the limits themselves are refused."""
    r_est = earth_radius
    for f, ok in ((0.2 * 1.001, True), (0.2, False), (5.0 * 0.999, True), (5.0, False)):
        assert (zs._adiabat_seed(_seed(R=f * r_est), _cfg()) is not None) is ok, f


def test_the_seed_mass_window_is_5_percent_of_the_planet_mass():
    """The top of mass_enclosed must lie within 5 % of the planet mass: 4.9 % off on either
    side passes, 5.1 % is refused."""
    for f, ok in ((1.049, True), (0.951, True), (1.051, False), (0.949, False)):
        assert (
            zs._adiabat_seed(_bad('mass_enclosed', lambda a: f * a), _cfg()) is not None
        ) is ok, f


ARR = st.one_of(
    st.lists(st.floats(allow_nan=True, allow_infinity=True), min_size=0, max_size=N + 1),
    st.lists(st.booleans(), min_size=N, max_size=N),
    st.just(None),
    st.floats(),
)


@settings(max_examples=200, deadline=None)
@given(r=ARR, P=ARR, m=ARR, keep=st.integers(0, 7))
def test_any_seed_content_is_refused_or_valid(r, P, m, keep):
    """Any content (NaN, inf, empty or wrong-length lists, bools, None, a scalar), mixed with
    valid arrays: the seed is refused or passes every check; it never raises."""
    seed = _seed()
    for bit, (k, v) in enumerate((('radii', r), ('pressure', P), ('mass_enclosed', m))):
        if not keep >> bit & 1:
            seed[k] = v
    got = zs._adiabat_seed(seed, _cfg())
    if got is not None:
        rr, PP, mm = (got[k] for k in ('radii', 'pressure', 'mass_enclosed'))
        assert all(a.shape == (N,) and np.all(np.isfinite(a)) for a in (rr, PP, mm))
        assert np.all(np.diff(rr) > 0) and np.all(np.diff(mm) >= 0) and np.all(np.diff(PP) <= 0)
        assert PP[0] > 0 and PP[-1] >= 0 and mm[-1] > 0


# main: routing of a seed (canned _solve and _solve_newton_outer)


def _full(R=7.0e6, t=1.0):
    r = {**_inner(R, accepted_on='strict'), 'adiabat_computed': True, 'total_time': t}
    r.update(adiabat_blend=1.0, adiabat_table_blend=1.0)
    return r


def _newton():
    r = {**_inner(6.9e6, accepted_on='newton_tol'), 'total_time': 5.0}
    r.update(pressure=np.array([3.5e11, 0.0]), density=np.array([1.2e4, 3.0e3]))
    r.update(newton_n_iter=3, newton_history=[], newton_used_brentq=False)
    return r


@pytest.fixture
def spies(canned, monkeypatch):  # noqa: F811  (the imported fixture)
    """Record every _solve call (config, kwargs) and every Newton call; _solve returns the
    results queued in ``out`` (a converged full-adiabat pass when empty)."""
    calls = {'solve': [], 'newton': [], 'out': []}

    def solve(cp, *a, **k):
        calls['solve'].append((cp, k))
        return dict(calls['out'].pop(0)) if calls['out'] else _full()

    monkeypatch.setattr(zs, '_solve', solve)
    monkeypatch.setattr(
        zs, '_solve_newton_outer', lambda *a, **k: calls['newton'].append(1) or _newton()
    )
    return calls


def _seed_main(R=6.95e6):
    return _seed(n=2, R=R)  # the canned configs below use num_layers 2


def test_a_seeded_newton_solve_skips_the_newton_root(spies):
    """With a valid seed the ramp runs once, from the seed: no Newton root, the seed radius as
    first guess, the Newton mass tolerance, the secant tail, and the seed pressure at the
    centre when the caller gives no hint; the caller's config is not changed."""
    cfg = _adiabatic_cfg(outer_solver='newton', num_layers=2, tolerance_outer=3e-3)
    seed = _seed_main()
    r = zs.main(cfg, {}, None, '/tmp', adiabat_seed=seed)
    assert r['accepted_on'] == 'strict' and not spies['newton']
    ((cp, kw),) = spies['solve']
    assert cp['_initial_radius_guess'] == 6.95e6 and cp['_secant_tail'] is True
    assert cp['tolerance_outer'] == 1e-4 and kw['p_center_hint'] == 3.6e11
    np.testing.assert_array_equal(kw['adiabat_seed']['radii'], seed['radii'])
    assert '_secant_tail' not in cfg and cfg['tolerance_outer'] == 3e-3


def test_a_caller_hint_wins_over_the_seed_pressure_and_picard_keeps_its_tolerance(spies):
    """The Picard outer solver keeps tolerance_outer on a seeded solve, and a central-pressure
    hint of the caller replaces the seed pressure."""
    cfg = _adiabatic_cfg(outer_solver='picard', num_layers=2, tolerance_outer=3e-3)
    zs.main(cfg, {}, None, '/tmp', adiabat_seed=_seed_main(), p_center_hint=2.0e11)
    ((cp, kw),) = spies['solve']
    assert cp['tolerance_outer'] == 3e-3 and cp['_secant_tail'] is True
    assert kw['p_center_hint'] == 2.0e11


def test_a_seeded_solve_that_fails_runs_the_newton_ramp_without_the_seed(spies, caplog):
    """The seeded solve does not converge: main solves again from the Newton root, without the
    seed, and returns that ramp; total_time adds the time of the seeded solve."""
    failed = {**_full(t=40.0), 'converged': False, 'accepted_on': None}
    spies['out'].append(failed)
    cfg = _adiabatic_cfg(outer_solver='newton', num_layers=2)
    with caplog.at_level(logging.INFO, logger='fwl.zalmoxis.solver'):
        r = zs.main(cfg, {}, None, '/tmp', adiabat_seed=_seed_main())
    assert len(spies['newton']) == 1 and len(spies['solve']) == 2
    (cp1, kw1), (cp2, kw2) = spies['solve']
    assert 'adiabat_seed' in kw1 and 'adiabat_seed' not in kw2
    assert cp2['_initial_radius_guess'] == 6.9e6 and cp2['_secant_tail'] is True
    assert (
        r['accepted_on'] == 'strict' and r['total_time'] == 46.0
    )  # seeded 40 s, Newton 5 s, ramp 1 s
    assert 'seeded solve did not converge' in caplog.text


def test_a_refused_seed_leaves_the_newton_ramp_as_without_a_seed(spies):
    """A seed with a NaN is refused: main runs the Newton root and its ramp exactly as with no
    seed (same _solve config and kwargs)."""
    bad = _seed_main()
    bad['pressure'][1] = NAN
    cfg = _adiabatic_cfg(outer_solver='newton', num_layers=2)
    zs.main(cfg, {}, None, '/tmp', adiabat_seed=bad)
    zs.main(cfg, {}, None, '/tmp')
    (cp1, kw1), (cp2, kw2) = spies['solve']
    assert cp1 == cp2 and kw1.keys() == kw2.keys() and 'adiabat_seed' not in kw1
    assert len(spies['newton']) == 2


def test_picard_without_a_seed_runs_no_secant_tail(spies):
    """A plain Picard solve gets the caller's config: no secant key, no seed."""
    cfg = _adiabatic_cfg(outer_solver='picard', num_layers=2)
    zs.main(cfg, {}, None, '/tmp')
    ((cp, kw),) = spies['solve']
    assert cp == cfg and '_secant_tail' not in cp and 'adiabat_seed' not in kw


@EXTERNAL_T
def test_an_external_temperature_ignores_the_seed(spies, kwargs, use_jax, external):
    """With the caller's temperature no adiabat is asked: the seed is not used and the Newton
    result returns; temperature arrays on the numpy path leave the adiabat asked, so the seed
    starts the ramp."""
    cfg = _adiabatic_cfg(outer_solver='newton', num_layers=2, use_jax=use_jax)
    r = zs.main(cfg, {}, None, '/tmp', adiabat_seed=_seed_main(), **kwargs)
    if external:
        assert spies['newton'] and not spies['solve'] and r['accepted_on'] == 'newton_tol'
        return
    ((cp, kw),) = spies['solve']
    assert not spies['newton'] and 'adiabat_seed' in kw
    assert kw['temperature_arrays'] is kwargs['temperature_arrays']


def test_the_seed_constants():
    """The seeded first table carries a blend no accept test takes, and the secant stops at 2
    tolerances."""
    assert zs._SEED_TABLE_BLEND < 0.0 and zs._SECANT_FLOOR == 2.0
    assert zs._short_of_adiabat(
        {'adiabat_computed': True, 'adiabat_blend': 1.0, 'adiabat_table_blend': -1.0}, True
    )
