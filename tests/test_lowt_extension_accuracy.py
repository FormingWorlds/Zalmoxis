"""Tests for ``tools.validation.lowt_extension_accuracy`` on analytic entropy curves."""

from __future__ import annotations

import numpy as np
import pytest

from tools.validation import lowt_extension_accuracy as acc

pytestmark = pytest.mark.unit


def test_deviation_of_a_logarithmic_entropy_matches_the_closed_form():
    """For s = c ln T the line between the node temperatures lies above T(S) = exp(S/c); its
    largest relative excess is a c / exp(S*/c) - 1 at S* = S0 + c - T0 / a, a the line slope."""
    c, S0, S1 = 40.0, 100.0, 180.0
    log_t = np.linspace(0.5, 3.0, 4001)
    dev, S_at, T0, T1 = acc.interval_deviation(log_t, c * np.log(10.0**log_t), S0, S1, n=20001)
    a = (np.exp(S1 / c) - np.exp(S0 / c)) / (S1 - S0)
    S_star = S0 + c - np.exp(S0 / c) / a
    assert dev == pytest.approx(a * c / np.exp(S_star / c) - 1.0, rel=1e-6)
    assert S_at == pytest.approx(S_star, abs=0.01)
    assert (T0, T1) == pytest.approx((np.exp(S0 / c), np.exp(S1 / c)), rel=1e-9)


@pytest.mark.parametrize(
    ('S0', 'S1'), [(5.0, 50.0), (20.0, 500.0), (5.0, 500.0)], ids=['below', 'above', 'both']
)
def test_deviation_is_nan_when_the_curve_does_not_span_the_interval(S0, S1):
    log_t = np.linspace(1.0, 2.0, 11)
    assert np.isnan(acc.interval_deviation(log_t, 10.0 + 100.0 * (log_t - 1.0), S0, S1)[0])


def test_deviation_needs_a_rising_curve_only_between_the_nodes():
    """A fall of s inside the interval gives NaN; one above the first sample at S1 does not."""
    log_t = np.linspace(1.0, 2.0, 11)
    s = 100.0 * log_t
    fall_in, fall_out = s.copy(), s.copy()
    fall_in[4] = fall_in[3]
    fall_out[9] = fall_out[8] - 1.0
    ref = acc.interval_deviation(log_t, s, 110.0, 160.0)[0]
    assert np.isnan(acc.interval_deviation(log_t, fall_in, 110.0, 160.0)[0])
    assert acc.interval_deviation(log_t, fall_out, 110.0, 160.0)[0] == ref > 0.0


def test_a_crossing_row_changes_phase_between_the_cells_that_bracket_the_nodes():
    """Row 1 jumps across both nodes from phase a to phase b, with no cell between them; row 2
    changes phase between its cells at 60 and 90, which bracket S1 = 70 but not S1 = 50; row 3
    holds no entropy in the interval."""
    s = np.array(
        [
            [10.0, 30.0, 60.0, 90.0],
            [5.0, 7.0, 95.0, 99.0],
            [10.0, 30.0, 60.0, 90.0],
            [200.0, 210.0, np.nan, np.nan],
        ]
    )
    phase = np.array([list('aaaa'), list('aabb'), list('aaab'), list('ab  ')])
    assert acc.crossing_rows(s, phase, 20.0, 50.0) == [1]
    assert acc.crossing_rows(s, phase, 20.0, 70.0) == [1, 2]


@pytest.mark.parametrize(('mass', 'p'), [(1.0, 350e9), (0.5, 275e9), (100.0, 1e13)])
def test_grid_top_follows_the_proteus_rule(mass, p):
    assert acc.p_max(mass) == pytest.approx(p)


def test_the_table_has_one_row_per_mass_and_names_band_columns_and_crossing_rows():
    r = dict(
        mass=0.5,
        added=1,
        S0=48.51,
        S1=125.84,
        cols=[(1e9, 0.02, False), (9.38e9, 0.0997, False), (6.931e9, 0.056, True)],
        cross=[(6.918e9, 0.056)],
    )
    table = acc.markdown([r, dict(r, mass=10.0, added=0, cols=[], cross=[])]).splitlines()
    assert len(table) == 4
    assert table[2] == (
        '| 0.5 | 1 | 48.5, 125.8 | 3, up to 9.38 GPa | 9.97 % at 9.38 GPa '
        '| 5.60 % at 6.931 GPa | 5.6 % at 6.918 GPa |'
    )
    assert table[3] == '| 10 | 0 | 48.5, 125.8 | 0 | none | none | none |'
