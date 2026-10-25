"""The central-pressure root of the surface pressure solve (``zalmoxis.solver``).

``_pressure_residual`` returns -P_t for a profile that stops before the surface, and
brentq with xtol 1e6 Pa returns the end of its last bracket with the smaller |f|, so it
can return a stopped (padded) end. ``_refine_pressure_root`` then solves again on the
nearest sign pair with an xtol of 0.1 P_t over the slope, and replaces a root that is
still padded by a reached end of its nearest sign pair within the cap
``_REACHED_END_MAX_MISS_PT`` P_t. Invariants: the refined root is never further from
P_t than the first one or than the result without the second solve, a reached end that
misses by more than the cap never replaces a padded root, and the returned result flags
a padded surface node.
"""

from __future__ import annotations

import logging
import math
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import zalmoxis.solver as zs

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

P_T = 101325.0  # target surface pressure [Pa]


class _Residual:
    """A residual f(P_c) [Pa] that records every evaluation, as the solver's closure does."""

    def __init__(self, f):
        self.f, self.evals, self.calls = f, {}, []

    def __call__(self, x):
        self.calls.append(float(x))
        v = float(self.f(x))
        self.evals[float(x)] = v
        return v


def _kink(x0=2.0e11, p_stop=2.7e4, slope=0.47):
    """Stopped (-P_t) below x0, reached above it with P(R) rising from p_stop at slope [1]."""
    return lambda x: -P_T if x < x0 else p_stop + slope * (x - x0) - P_T


def _refine(res, root):
    return zs._refine_pressure_root(root, None, res.evals, res, P_T, 100)[0]


@pytest.mark.physics_invariant
def test_a_padded_root_on_a_smooth_kink_is_replaced_by_the_reached_root():
    """Smooth kink at slope 0.47 (above the limit 2 P_t / 1e6 Pa = 0.20): the solve on the
    sign pair with xtol 0.1 P_t / slope reaches P_t within 0.1 P_t."""
    res = _Residual(_kink())
    res(1.999e11)  # stopped
    res(2.003e11)  # reached, P(R) = 1.41e8 Pa
    root = _refine(res, 1.999e11)
    assert res.f(root) != -P_T
    assert abs(res.f(root)) <= 0.1 * P_T
    assert abs(res.f(1.999e11)) == P_T  # the first root was padded


@pytest.mark.physics_invariant
def test_a_jump_beyond_the_cap_keeps_the_padded_root_and_logs_it(caplog):
    """A jump from the stop to P(R) = 1.956e8 Pa leaves no reached point near P_t: the reached
    end misses by 1930 P_t, above the cap of 1 P_t, so the padded root stays."""
    res = _Residual(lambda x: -P_T if x < 2.0e11 else 1.956e8 - P_T)  # a jump at the stop
    res(2.0e11 - 5e5)
    res(2.0e11 + 5e5)
    with caplog.at_level(logging.DEBUG, logger='zalmoxis.solver'):
        root = _refine(res, 2.0e11 - 5e5)
    assert res.f(root) == -P_T
    assert len(res.evals) == 2  # a slope of 196 over the 1e6 Pa pair: no second solve
    kept = [r for r in caplog.records if 'keeps the padded root' in r.getMessage()]
    assert [r.levelno for r in kept] == [logging.DEBUG]
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('miss_pt', 'replaced'), [(0.5, True), (1.0, True), (1.0 + 1e-9, False), (1.5, False)]
)
def test_the_reached_end_replaces_a_padded_root_only_within_the_cap(miss_pt, replaced):
    """With a pair 1e8 Pa wide the second solve stands aside (xtol >= 1e6 Pa); the reached
    end then replaces the padded root when it misses P_t by at most the cap of 1 P_t."""
    res = _Residual(lambda x: -P_T if x < 1.05e9 else miss_pt * P_T)
    res(1.0e9)
    res(1.1e9)
    root = _refine(res, 1.0e9)
    assert zs._REACHED_END_MAX_MISS_PT == 1.0
    np.testing.assert_equal(root, 1.1e9 if replaced else 1.0e9)
    assert len(res.evals) == 2  # no second solve


def test_a_root_within_a_tenth_of_p_t_is_kept_without_a_new_solve():
    """|f(root)| <= 0.1 P_t needs no refinement: the root and the evaluations stay."""
    res = _Residual(_kink())
    near = 2.0e11 + (1.05 * P_T - 2.7e4) / 0.47  # P(R) = 1.05 P_t
    res(1.0e11)
    res(near)
    np.testing.assert_equal(_refine(res, near), near)
    assert len(res.evals) == 2


def test_a_root_without_a_sign_pair_is_kept():
    """No evaluation pair brackets P_t: neither step has a bracket, the root stays."""
    res = _Residual(lambda x: -P_T)
    for x in (1e9, 2e9, 3e9):
        res(x)
    np.testing.assert_equal(_refine(res, 2e9), 2e9)
    assert len(res.evals) == 3


@settings(derandomize=True, deadline=None, max_examples=300)
@given(
    points=st.dictionaries(
        st.floats(1e6, 1e13),
        st.one_of(
            st.floats(-1e9, 1e9),
            st.sampled_from([math.nan, math.inf, -math.inf, 0.0, -P_T]),
        ),
        max_size=12,
    ),
    root=st.floats(1e6, 1e13),
)
def test_the_sign_pair_is_the_nearest_adjacent_finite_bracket(points, root):
    """The pair holds adjacent evaluations with finite f(a) < 0 < f(b), the one nearest the
    root; a non-finite or zero residual never ends a pair, and no bracket gives None."""
    pair = zs._nearest_sign_pair(points, root)
    brackets = [
        (a, b)
        for a in points
        for b in points
        if a < b
        and not any(a < x < b for x in points)
        and -math.inf < points[a] < 0.0 < points[b] < math.inf
    ]
    assert (pair is None) == (not brackets)
    if pair is not None:
        a, b = pair
        assert pair in brackets
        assert abs(a - root) == min(abs(x - root) for x, _ in brackets)


@pytest.mark.parametrize('padded', [True, False])
def test_main_warns_once_when_the_accepted_profile_is_padded(monkeypatch, caplog, padded):
    """A Picard run can keep hundreds of padded pressure roots on its way, logged at DEBUG;
    main warns once, and only when the profile it returns is padded."""
    from tests.test_invariants import _canned, _cfg, _run, _sphere

    result = _canned('strict')[0]
    result.update(accept_mass_limit=1e-3, padded=padded)
    monkeypatch.setattr(zs, '_solve', lambda *a, **k: result)
    with caplog.at_level(logging.WARNING, logger='zalmoxis.solver'):
        out = _run(_cfg(planet_mass=_sphere()[1]['planet_mass']))
    hits = [r for r in caplog.records if 'padded' in r.getMessage()]
    assert out['padded'] is padded
    assert len(hits) == (1 if padded else 0)


@pytest.mark.physics_invariant
def test_the_second_solve_never_returns_a_worse_root():
    """A residual that is not monotone near the stop: stopped below x0, P(R) = 3 P_t on
    [x0, b) and 1.5 P_t at b. The second solve converges on the jump at x0 and returns its
    padded end; the brentq root b (miss 0.5 P_t) is the better one and stays."""
    x0, a, b = 2.0e11, 2.0e11 - 1e6, 2.0e11 + 1e6

    def f(x):
        if x < x0:
            return -P_T
        return 2.0 * P_T if x < b else 0.5 * P_T

    res = _Residual(f)
    res(a)
    res(b)
    root = _refine(res, b)
    np.testing.assert_equal(root, b)
    assert len(res.evals) > 2  # the second solve ran


@pytest.mark.parametrize('exc', [zs.StructureSolveError, ValueError])
def test_a_failing_second_solve_keeps_the_first_root(exc):
    """The second solve evaluates new central pressures; a structure solve that fails there
    (StructureSolveError) or an EOS lookup that raises ValueError leaves the first root."""
    a, b = 1.999e11, 2.003e11

    def f(x):
        if x not in (a, b):
            raise exc('solve at P_c not finite')
        return -P_T if x == a else 1.41e8 - P_T

    res = _Residual(f)
    res(a)
    res(b)
    try:
        root = _refine(res, a)
    except exc as err:
        pytest.fail(f'the second solve raised through the refinement: {err!r}')
    np.testing.assert_equal(root, a)
    assert len(res.evals) == 2


def test_a_second_solve_that_does_not_converge_keeps_the_first_root():
    """brentq raises RuntimeError when it does not converge in maxiter iterations."""
    res = _Residual(_kink())
    res(1.999e11)
    res(2.003e11)
    try:
        root = zs._refine_pressure_root(1.999e11, None, res.evals, res, P_T, 1)[0]
    except RuntimeError as err:
        pytest.fail(f'the non-converged second solve raised through the refinement: {err!r}')
    np.testing.assert_equal(root, 1.999e11)  # the reached end misses by 1390 P_t, above the cap
    assert len(res.evals) == 3  # one evaluation before brentq gave up


@pytest.mark.physics_invariant
def test_a_reached_root_far_from_p_t_is_refined():
    """A reached brentq root that misses P_t by 0.5 P_t on the smooth kink is solved again
    on its sign pair and ends within 0.1 P_t."""
    res = _Residual(_kink())
    far = 2.0e11 + (1.5 * P_T - 2.7e4) / 0.47  # P(R) = 1.5 P_t
    res(2.0e11 - 1e6)
    res(far)
    root = _refine(res, far)
    assert res.f(root) != -P_T
    assert abs(res.f(root)) <= 0.1 * P_T


def test_the_second_solve_reuses_the_ends_of_its_pair():
    """Each residual call is a full structure solve: the second solve takes f(a) and f(b)
    from the evaluations and solves the structure only at new central pressures."""
    res = _Residual(_kink())
    res(1.999e11)
    res(2.003e11)
    _refine(res, 1.999e11)
    assert len(res.calls) > 2
    assert len(res.calls) == len(set(res.calls))


@pytest.mark.physics_invariant
@pytest.mark.parametrize(('slope', 'solved'), [(1.0, True), (1.0 + 1e-6, False), (50.0, False)])
def test_the_second_solve_runs_only_on_a_pair_no_steeper_than_a_smooth_profile(slope, solved):
    """dP(R)/dP_c of a smooth profile is at most 1; a steeper pair whose reached end misses P_t
    by more than the cap holds a jump or scatter, where the second solve only costs solves."""
    a, b = 2.0e11, 2.0e11 + 2.5 * P_T / slope  # f(b) = 1.5 P_t at the secant slope
    res = _Residual(lambda x: -P_T if x < (a + b) / 2 else -P_T + slope * (x - a))
    res(a)
    res(b)
    _refine(res, a)
    assert (len(res.evals) > 2) is solved


@pytest.mark.physics_invariant
def test_a_steep_narrow_pair_over_a_small_jump_is_still_refined():
    """A jump of 0.95 P_t at the stop, then P(R) rising at slope 0.47: a pair 1.41e5 Pa wide
    has a secant slope of 1.15 but a reached end within P_t of the target, so it is no large
    jump and the second solve reaches P_t within 0.1 P_t."""
    x0 = 2.0e11
    res = _Residual(lambda x: -P_T if x < x0 else 0.95 * P_T + 0.47 * (x - x0) - P_T)
    res(x0 - 1e3)
    res(x0 + 1.4e5)
    root = _refine(res, x0 - 1e3)
    assert res.f(root) != -P_T
    assert abs(res.f(root)) <= 0.1 * P_T


def test_the_returned_record_is_that_of_the_solve_that_gave_the_root():
    """A root from the second solve comes with that solve's record; a kept root keeps the
    first."""
    res = _Residual(_kink())
    res(1.999e11)
    res(2.003e11)
    first = object()
    root, info = zs._refine_pressure_root(1.999e11, first, res.evals, res, P_T, 100)
    assert info is not first and info.converged
    np.testing.assert_equal(info.root, root)
    near = _Residual(_kink())
    x = 2.0e11 + (1.05 * P_T - 2.7e4) / 0.47
    near(1.0e11)
    near(x)
    assert zs._refine_pressure_root(x, first, near.evals, near, P_T, 100)[1] is first


def test_a_steep_pair_whose_reached_end_is_at_the_cap_is_solved():
    """Slope 2 and f(b) = P_t exactly: the reached end is within the cap, so the second solve
    runs."""
    a, b = 2.0e11, 2.0e11 + P_T
    res = _Residual(lambda x: -P_T if x < a + 0.5 * P_T else -P_T + 2.0 * (x - a))
    res(a)
    res(b)
    assert res.evals[b] == P_T
    _refine(res, a)
    assert len(res.evals) > 2


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('width', 'high', 'end'),
    [(5e4, 2.0, 0.5), (1e5, 1.5, 0.5), (3e5, 1.5, 0.5), (5e4, 2.0, 1.0)],
)
def test_a_second_solve_into_a_jump_keeps_the_reached_end_within_the_cap(width, high, end):
    """Padded first root a, reached end b at end P_t, and P(R) = high P_t on the first 0.8 of
    the pair: the second solve ends on the jump at its padded end. The result is b, the
    root step (e) gives without the second solve, not the padded end; at end = 1 the tie
    of |f| goes to the reached end."""
    x0 = 2.0e11
    res = _Residual(
        lambda x: -P_T if x < x0 else (end * P_T if x >= x0 + 0.8 * width else high * P_T)
    )
    res(x0 - 1e3)
    res(x0 + width)
    root = _refine(res, x0 - 1e3)
    assert len(res.evals) > 2  # the second solve ran
    assert res.evals[root] == end * P_T


@pytest.mark.physics_invariant
@settings(derandomize=True, deadline=None, max_examples=300)
@given(
    width=st.floats(1e4, 1e6),
    levels=st.lists(st.floats(0.01, 3.0), min_size=1, max_size=6),
)
def test_the_second_solve_is_never_worse_than_none(width, levels):
    """Stopped below x0, then P(R) - P_t in steps of ``levels`` P_t over the pair: the
    refined root misses P_t by no more than the result without the second solve."""
    x0 = 2.0e11

    def f(x):
        if x < x0:
            return -P_T
        return levels[min(int((x - x0) / width * len(levels)), len(levels) - 1)] * P_T

    out = {}
    for name, miss_pt in (('with', zs._REFINE_MISS_PT), ('without', math.inf)):
        with patch.object(zs, '_REFINE_MISS_PT', miss_pt):
            res = _Residual(f)
            res(x0 - 1e3)
            res(x0 + width)
            out[name] = abs(res.evals[_refine(res, x0 - 1e3)])
    assert out['with'] <= out['without']


def test_a_tie_of_padded_roots_returns_the_second_solve_root():
    """A jump to 1.5 P_t at x0 over a pair 1e6 Pa wide: the second solve ends on the padded
    end of the jump, no reached end lies within the cap, and the tie of the two padded roots
    goes to the later one, as the first-root guard did."""
    x0, a = 2.0e11, 2.0e11 - 5e5
    res = _Residual(lambda x: -P_T if x < x0 else 1.5 * P_T)
    res(a)
    res(x0 + 5e5)
    root = _refine(res, a)
    assert len(res.evals) > 2
    assert res.evals[root] == -P_T and root != a


@pytest.mark.parametrize(('first', 'want'), [('padded', 0.5), ('reached', 0.5)])
def test_a_second_solve_root_with_a_non_finite_residual_never_wins(first, want):
    """A second-solve root whose recorded f is NaN never wins: a padded first root still takes
    its pair's reached end at 0.5 P_t, and a reached first root at 0.5 P_t stays."""
    x0 = 2.0e11
    res = _Residual(lambda x: -P_T if x < x0 else 0.5 * P_T)
    res(x0 - 1e3)
    res(x0 + 5e4)
    root = x0 - 1e3 if first == 'padded' else x0 + 5e4

    def nan_root(f, a, b, **kw):
        x = 0.5 * (a + b)
        res.evals[x] = math.nan
        return x, 'second'

    with patch.object(zs, 'brentq', nan_root):
        out, info = zs._refine_pressure_root(root, 'first', res.evals, res, P_T, 100)
    assert res.evals[out] == want * P_T and info == 'first'
