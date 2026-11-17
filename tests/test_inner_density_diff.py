"""The density residual of the inner Picard loop leaves out padded nodes (P <= 0)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.unit


def test_a_padded_node_at_the_alpha_floor_does_not_set_the_residual():
    """A padded outer node has density 0 from the EOS, so damped Picard with alpha 0.1 moves it
    by exactly 0.1 of its value; the residual is that of the nodes with P > 0."""
    from zalmoxis.solver import _inner_density_diff

    old = np.array([1.0e4, 8.0e3, 5.0e3, 200.0])
    new = np.array([1.0e4 * (1 + 2e-5), 8.0e3, 5.0e3 * (1 - 3e-5), 180.0])  # 180 = 0.9 x 200
    pressure = np.array([3.6e11, 1.0e11, 1.0e9, 0.0])
    assert _inner_density_diff(new, old, pressure) == pytest.approx(3e-5, rel=1e-9)


def test_without_a_node_above_zero_pressure_every_node_counts():
    from zalmoxis.solver import _inner_density_diff

    old, new = np.array([100.0, 50.0]), np.array([90.0, 50.0])
    assert _inner_density_diff(new, old, np.zeros(2)) == pytest.approx(0.1)


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.floats(1.0, 2.0e4), min_size=2, max_size=12),
    st.integers(1, 3),
    st.floats(0.0, 1.0),
)
def test_the_residual_is_the_largest_change_over_the_nodes_with_pressure(rho, n_pad, shrink):
    """Any profile with n_pad padded outer nodes: the result is the largest relative change over
    the nodes with P > 0, whatever the padded nodes do."""
    from zalmoxis.solver import _inner_density_diff

    old = np.array(rho)
    n_pad = min(n_pad, old.size - 1)
    new = old * (1 + np.linspace(-1e-3, 2e-3, old.size))
    new[-n_pad:] = old[-n_pad:] * shrink  # any change at the padded nodes
    pressure = np.r_[np.geomspace(3e11, 1e6, old.size - n_pad), np.zeros(n_pad)]
    live = slice(0, old.size - n_pad)
    want = np.max(np.abs((new[live] - old[live]) / old[live]))
    assert _inner_density_diff(new, old, pressure) == pytest.approx(want, rel=1e-12)
