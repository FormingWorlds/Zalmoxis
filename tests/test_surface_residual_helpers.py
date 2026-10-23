"""The residual of a structure solve that stops before R, and the central pressure tolerance.

``_stopped_residual(p_stop, p_target, radius, r_stop, k)`` continues the surface residual of
a solve that stops at r_stop < radius, at the pressure p_stop: p_stop - p_target - k (radius -
r_stop). ``_surface_residual(pressure, stop, p_target, radius, k)`` is P(R) - p_target for a
solve that reaches R and the stopped branch for one that does not. ``_pressure_xtol(slope,
p_target, xtol_first)`` is the central pressure tolerance that holds the surface residual to
0.1 p_target at a slope dP(R)/dP_c, or None when it is no finer than the first brentq call's.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from zalmoxis.solver import _pressure_xtol, _stopped_residual, _surface_residual

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

XTOL_FIRST = 1e6


@pytest.mark.physics_invariant
@settings(max_examples=300, deadline=None)
@given(
    p_stop=st.floats(0.0, 1e7),
    p_target=st.floats(1e2, 1e8),
    radius=st.floats(1e6, 3e7),
    frac=st.floats(0.0, 1.0),
    k=st.floats(1e2, 1e6),
)
def test_a_stopped_solve_lies_below_its_stop_and_meets_it_at_the_radius(
    p_stop, p_target, radius, frac, k
):
    """The stopped branch is below p_stop - p_target, equals it at r_stop = R and rises with r_stop."""
    r_stop = frac * radius
    f = _stopped_residual(p_stop, p_target, radius, r_stop, k)
    assert f <= p_stop - p_target
    at_radius = _stopped_residual(p_stop, p_target, radius, radius, k)
    assert at_radius == pytest.approx(p_stop - p_target, rel=1e-12, abs=1e-6)
    assert f == pytest.approx(p_stop - p_target - k * (radius - r_stop), rel=1e-12, abs=1e-6)
    farther = min(radius, r_stop + 0.01 * radius)
    assert _stopped_residual(p_stop, p_target, radius, farther, k) >= f


@pytest.mark.parametrize('k', [0.0, -1.0, math.nan, math.inf, True, '1e4', None])
def test_a_slope_that_is_not_a_positive_number_is_refused(k):
    """k must be a finite positive number: zero gives the flat branch back."""
    with pytest.raises(ValueError, match='k must be'):
        _stopped_residual(0.0, 1e5, 6.4e6, 6.0e6, k)


@pytest.mark.parametrize('r_stop', [-1.0, 6.4e6 + 1.0, math.nan, math.inf, True, None])
def test_a_stop_outside_the_planet_is_refused(r_stop):
    """r_stop must lie in [0, R]."""
    with pytest.raises(ValueError, match='r_stop must be'):
        _stopped_residual(0.0, 1e5, 6.4e6, r_stop, 1e4)


@pytest.mark.parametrize('p_stop', [-1.0, math.nan, math.inf, True, None])
def test_a_stop_pressure_that_is_not_a_pressure_is_refused(p_stop):
    """p_stop must be finite and not negative: the integration stops at P >= 0."""
    with pytest.raises(ValueError, match='p_stop must be'):
        _stopped_residual(p_stop, 1e5, 6.4e6, 6.0e6, 1e4)


@pytest.mark.physics_invariant
def test_the_surface_residual_joins_both_branches_for_any_stop_pressure():
    """At the central pressure where the stop reaches R, the stopped branch with the stop
    pressure and the unpadded branch with P(R) equal to it give the same residual, for a
    zero-pressure event and for a stop above zero pressure."""
    radius, p_target, k = 6.4e6, 1e7, 2e4
    for p_stop in (0.0, 5e6):
        reached = np.array([3e11, 1e9, p_stop + 1e-3])
        stopped = np.array([3e11, 1e9, 0.0])
        near = _surface_residual(
            stopped, {'r': radius - 1e-3, 'p': p_stop}, p_target, radius, k
        )
        assert _surface_residual(reached, {}, p_target, radius, k) == pytest.approx(
            p_stop - p_target + 1e-3, rel=1e-12
        )
        assert near == pytest.approx(p_stop - p_target - k * 1e-3, rel=1e-12)
        # The flat branch the stop replaces: -p_target whatever the stop pressure.
        if p_stop:
            assert abs(near - (-p_target)) > 0.4 * p_target
    with pytest.raises(ValueError, match='stop'):
        _surface_residual(np.array([3e11, 0.0]), {}, p_target, radius, k)


@settings(max_examples=300, deadline=None)
@given(slope=st.floats(1e-4, 1e4), p_target=st.floats(1e2, 1e8))
def test_the_tolerance_holds_the_surface_residual_to_a_tenth_of_the_target(slope, p_target):
    """A positive slope gives a tolerance that moves P(R) by at most 0.1 p_target, and none
    when that tolerance is no finer than the first call's."""
    x = _pressure_xtol(slope, p_target, XTOL_FIRST)
    if 0.1 * p_target / slope < XTOL_FIRST:
        assert 0.0 < x < XTOL_FIRST
        assert x * slope <= 0.1 * p_target * (1 + 1e-12)
    else:
        assert x is None


def test_the_second_call_compares_with_the_first_calls_tolerance():
    """The skip compares with the xtol of the first call, not with a fixed number."""
    assert _pressure_xtol(0.2, 1e5, 1e6) == pytest.approx(5e4, rel=1e-12)
    assert _pressure_xtol(0.2, 1e5, 4e4) is None
    assert _pressure_xtol(0.2, 1e5, 5e4) is None


@pytest.mark.parametrize('slope', [0.0, -0.2, math.nan, math.inf, -math.inf, True, None])
def test_a_slope_that_is_not_positive_skips_the_second_call(slope):
    """Without a usable slope the first brentq root stays."""
    assert _pressure_xtol(slope, 1e5, XTOL_FIRST) is None


@pytest.mark.parametrize('p_target', [0.0, -1.0, math.nan, math.inf, True, None])
def test_a_target_that_is_not_positive_skips_the_second_call(p_target):
    """Without a usable target the first brentq root stays."""
    assert _pressure_xtol(0.2, p_target, XTOL_FIRST) is None
