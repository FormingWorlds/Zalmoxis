"""The density residual of the inner Picard loop leaves out padded nodes (P <= 0)."""

from __future__ import annotations

import logging
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._paleos_helpers import _InnerBails
from zalmoxis.solver import _inner_density_diff

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def test_a_padded_node_at_the_alpha_floor_does_not_set_the_residual():
    """A padded outer node has density 0 from the EOS, so damped Picard with alpha 0.1 moves it
    by exactly 0.1 of its value; the residual is that of the nodes with P > 0."""
    old = np.array([1.0e4, 8.0e3, 5.0e3, 200.0])
    new = np.array([1.0e4 * (1 + 2e-5), 8.0e3, 5.0e3 * (1 - 3e-5), 180.0])  # 180 = 0.9 x 200
    pressure = np.array([3.6e11, 1.0e11, 1.0e9, 0.0])
    assert _inner_density_diff(new, old, pressure) == pytest.approx(3e-5, rel=1e-9)
    assert _inner_density_diff(new, old, np.full(4, 1e9)) == pytest.approx(0.1, rel=1e-12)


def test_without_a_node_above_zero_pressure_every_node_counts():
    """With no node at P > 0 the residual is the largest change over every node; with one such
    node it is the change of that node alone."""
    old, new = np.array([100.0, 50.0]), np.array([90.0, 50.0])
    assert _inner_density_diff(new, old, np.zeros(2)) == pytest.approx(0.1)
    assert _inner_density_diff(new, old, np.array([0.0, 1.0])) == 0.0


@settings(max_examples=200, deadline=None)
@given(
    st.lists(st.floats(1.0, 2.0e4), min_size=4, max_size=12),
    st.integers(1, 3),
    st.floats(0.0, 1.0),
)
def test_the_residual_is_the_largest_change_over_the_nodes_with_pressure(rho, n_pad, shrink):
    """Any profile with n_pad padded outer nodes: the result is the largest relative change over
    the nodes with P > 0, whatever the padded nodes do."""
    old = np.array(rho)
    new = old * (1 + np.linspace(-1e-3, 2e-3, old.size))
    new[-n_pad:] = old[-n_pad:] * shrink  # any change at the padded nodes
    pressure = np.r_[np.geomspace(3e11, 1e6, old.size - n_pad), np.zeros(n_pad)]
    want = np.max(np.abs((new[:-n_pad] - old[:-n_pad]) / old[:-n_pad]))
    assert _inner_density_diff(new, old, pressure) == pytest.approx(want, rel=1e-12)
    assert want > 0


def test_the_inner_exit_recorder_reads_each_exit_record():
    """Each bail record of the solver gives its best diff; a wall-clock exit and a bail record
    without its diff give inf; a converged inner loop is counted; only records at or above the
    level of the logger before the solve pass on."""
    rec = _InnerBails(logging.WARNING)

    def log(level, msg, *args):
        return rec.filter(
            logging.LogRecord('fwl.zalmoxis.solver', level, 'f', 1, msg, args, None)
        )

    shown = [
        log(
            10,
            'Inner loop: stuck-bail after %d iters w/o improvement (best diff=%.2e, '
            'target=%.2e). Outer loop will iterate.',
            15,
            0.3,
            1e-4,
        ),
        log(
            20,
            'Inner loop: accepting best density after %d oscillations (best diff=%.2e, '
            'target=%.2e)',
            15,
            0.05,
            1e-4,
        ),
        log(
            30,
            'Maximum inner iterations (%d) reached. Best density diff=%.2e, threshold=%.2e. '
            'Density not converged.',
            100,
            0.2,
            0.1,
        ),
        log(30, 'Wall-clock timeout in inner loop (outer=%d, inner=%d).', 3, 7),
        log(20, 'Inner loop: timeout'),
        log(10, 'Inner loop converged after %d iterations.', 14),
    ]
    assert rec.diffs == [0.3, 0.05, 0.2, math.inf, math.inf]
    assert rec.converged == 1
    assert shown == [False, False, True, True, False, False]
