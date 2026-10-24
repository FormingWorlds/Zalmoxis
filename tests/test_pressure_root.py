"""The central-pressure root of the surface pressure solve (``zalmoxis.solver``).

``_pressure_residual`` returns -P_t for a profile that stops before the surface, and
brentq with xtol 1e6 Pa returns the end of its last bracket with the smaller |f|, so it
can return a stopped (padded) end. ``_refine_pressure_root`` then solves again on the
nearest sign pair with an xtol of 0.1 P_t over the slope, and replaces a root that is
still padded by the reached end of that pair when that end misses P_t by at most
``E_MAX_MISS_PT`` P_t. Invariants: the refined root is never further from P_t than the
first one, a reached end that misses by more than the cap never replaces a padded root,
and the returned result flags a padded surface node.
"""

from __future__ import annotations

import logging
import math

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
        self.f, self.evals = f, {}

    def __call__(self, x):
        v = float(self.f(x))
        self.evals[float(x)] = v
        return v


def _kink(x0=2.0e11, p_stop=2.7e4, slope=0.47):
    """Stopped (-P_t) below x0, reached above it with P(R) rising from p_stop at slope [1]."""
    return lambda x: -P_T if x < x0 else p_stop + slope * (x - x0) - P_T


def _jump(x0=2.0e11, p_reached=1.956e8):
    """Stopped below x0 and reached far above P_t above it, with nothing in between."""
    return lambda x: -P_T if x < x0 else p_reached - P_T


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
def test_a_jump_beyond_the_cap_keeps_the_padded_root_and_warns(caplog):
    """A jump from the stop to P(R) = 1.956e8 Pa leaves no reached point near P_t: the reached
    end misses by 1930 P_t, above the cap of 1 P_t, so the padded root stays."""
    res = _Residual(_jump())
    res(1.999e11)
    res(2.001e11)
    with caplog.at_level(logging.WARNING, logger='zalmoxis.solver'):
        root = _refine(res, 1.999e11)
    assert res.f(root) == -P_T
    assert 'keeps the padded root' in caplog.text


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('miss_pt', 'replaced'), [(0.5, True), (1.0, True), (1.0 + 1e-9, False), (1.5, False)]
)
def test_the_reached_end_replaces_a_padded_root_only_within_the_cap(miss_pt, replaced):
    """With a pair 1e8 Pa wide the second solve stands aside (xtol >= 1e6 Pa); the reached
    end then replaces the padded root when it misses P_t by at most E_MAX_MISS_PT P_t."""
    res = _Residual(lambda x: -P_T if x < 1.05e9 else miss_pt * P_T)
    res(1.0e9)
    res(1.1e9)
    root = _refine(res, 1.0e9)
    assert zs.E_MAX_MISS_PT == 1.0
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
    xs = sorted(points)
    brackets = [
        (a, b)
        for a, b in zip(xs, xs[1:])
        if np.isfinite(points[a]) and np.isfinite(points[b]) and points[a] < 0.0 < points[b]
    ]
    assert (pair is None) == (not brackets)
    if pair is not None:
        a, b = pair
        assert pair in brackets
        assert abs(a - root) == min(abs(x - root) for x, _ in brackets)
