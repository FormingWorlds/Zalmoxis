"""The residual of a structure solve that stops before R, and the central pressure tolerance.

``_stopped_residual(p_target, radius, r_stop, k)`` continues the surface residual below
-p_target for a solve whose pressure reaches zero at r_stop < radius: -p_target - k (radius -
r_stop). ``_pressure_xtol(slope, p_target)`` is the central pressure tolerance that holds the
surface residual to 0.1 p_target at a slope dP(R)/dP_c, capped at 1e6 Pa.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from zalmoxis.solver import _pressure_xtol, _stopped_residual

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

XTOL_CAP = 1e6


@pytest.mark.physics_invariant
@settings(max_examples=300, deadline=None)
@given(
    p_target=st.floats(1e2, 1e8),
    radius=st.floats(1e6, 3e7),
    frac=st.floats(0.0, 1.0),
    k=st.floats(1e2, 1e6),
)
def test_a_stopped_solve_lies_below_the_target_and_meets_it_at_the_radius(
    p_target, radius, frac, k
):
    """The stopped branch is below -p_target, equals it at r_stop = R and rises with r_stop."""
    r_stop = frac * radius
    f = _stopped_residual(p_target, radius, r_stop, k)
    assert f <= -p_target
    assert _stopped_residual(p_target, radius, radius, k) == pytest.approx(-p_target, rel=1e-12)
    assert f == pytest.approx(-p_target - k * (radius - r_stop), rel=1e-12)
    farther = min(radius, r_stop + 0.01 * radius)
    assert _stopped_residual(p_target, radius, farther, k) >= f


@pytest.mark.parametrize('k', [0.0, -1.0, math.nan, math.inf, True, '1e4', None])
def test_a_slope_that_is_not_a_positive_number_is_refused(k):
    """k must be a finite positive number: zero gives the flat branch back."""
    with pytest.raises(ValueError, match='k must be'):
        _stopped_residual(1e5, 6.4e6, 6.0e6, k)


@pytest.mark.parametrize('r_stop', [-1.0, 6.4e6 + 1.0, math.nan, math.inf, True, None])
def test_a_stop_outside_the_planet_is_refused(r_stop):
    """r_stop must lie in [0, R]."""
    with pytest.raises(ValueError, match='r_stop must be'):
        _stopped_residual(1e5, 6.4e6, r_stop, 1e4)


@settings(max_examples=300, deadline=None)
@given(slope=st.floats(1e-4, 1e4), p_target=st.floats(1e2, 1e8))
def test_the_tolerance_holds_the_surface_residual_to_a_tenth_of_the_target(slope, p_target):
    """A positive slope gives a tolerance that moves P(R) by at most 0.1 p_target."""
    x = _pressure_xtol(slope, p_target)
    assert 0.0 < x <= XTOL_CAP
    assert x * slope <= 0.1 * p_target * (1 + 1e-12)


@pytest.mark.parametrize('slope', [0.0, -0.2, math.nan, math.inf, -math.inf, True, None])
def test_a_slope_that_is_not_positive_keeps_the_coarse_tolerance(slope):
    """Without a usable slope the first brentq tolerance stays."""
    assert _pressure_xtol(slope, 1e5) == XTOL_CAP


@pytest.mark.parametrize('p_target', [0.0, -1.0, math.nan, math.inf, True, None])
def test_a_target_that_is_not_positive_keeps_the_coarse_tolerance(p_target):
    """Without a usable target the first brentq tolerance stays."""
    assert _pressure_xtol(0.2, p_target) == XTOL_CAP
