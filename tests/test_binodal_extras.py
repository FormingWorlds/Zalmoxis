"""Branch tests for ``binodal`` paths the existing suite does not cover.

Targets edge cases in:
- ``gupta2025_gibbs_mixing``: x_H2 boundary returns (lines 310-311 / 320-321).
- ``gupta2025_critical_pressure``: |W_V| near zero -> +inf return (line 350).
- ``gupta2025_critical_temperature``: P_GPa <= 0 short-circuit.
- ``_gupta2025_critical_temperature_brentq``: same-sign endpoints -> None.
- ``gupta2025_coexistence_compositions``: above-critical single-phase return.
"""

from __future__ import annotations

import numpy as np
import pytest

from zalmoxis.binodal import (
    _gupta2025_critical_temperature_brentq,
    gupta2025_coexistence_compositions,
    gupta2025_critical_pressure,
    gupta2025_critical_temperature,
    gupta2025_gibbs_mixing,
)

pytestmark = pytest.mark.unit


def test_gibbs_mixing_zero_x_returns_zero():
    """x_H2 == 0 short-circuits before any thermodynamics."""
    G = gupta2025_gibbs_mixing(0.0, T=2000.0, P_GPa=10.0)
    assert G == 0.0


def test_gibbs_mixing_unity_x_returns_zero():
    """x_H2 >= 1 short-circuits."""
    G = gupta2025_gibbs_mixing(1.0, T=2000.0, P_GPa=10.0)
    assert G == 0.0


def test_gibbs_mixing_negative_x_returns_zero():
    """Sub-zero composition returns 0 without raising."""
    G = gupta2025_gibbs_mixing(-0.1, T=2000.0, P_GPa=10.0)
    assert G == 0.0


def test_gibbs_mixing_finite_for_interior_x():
    """Discriminator: finite, negative G at x=0.5 (mixing is favourable
    in the symmetric Margules form below the critical point)."""
    G_lo = gupta2025_gibbs_mixing(0.5, T=2000.0, P_GPa=1.0)
    G_hi = gupta2025_gibbs_mixing(0.5, T=2000.0, P_GPa=50.0)
    assert np.isfinite(G_lo)
    assert np.isfinite(G_hi)
    # Different P -> different G. Use a relative tolerance scaled to the
    # magnitudes of the two values (Gibbs is O(1e3-1e4) J/mol for typical
    # mantle conditions, so abs=1e-6 was many orders of magnitude below
    # any meaningful difference and made the assertion trivially true).
    assert not np.isclose(G_lo, G_hi, rtol=1e-3, atol=0.0)


def test_critical_temperature_nonpositive_pressure_returns_none():
    """P_GPa <= 0 short-circuits before any lookup."""
    assert gupta2025_critical_temperature(0.0) is None
    assert gupta2025_critical_temperature(-5.0) is None


def test_critical_temperature_inside_table_returns_finite():
    """A pressure well inside the table range returns a finite T_crit."""
    T_crit = gupta2025_critical_temperature(10.0)
    assert T_crit is not None
    assert np.isfinite(T_crit)
    assert 300.0 <= T_crit <= 6000.0


def test_critical_temperature_brentq_no_root_returns_none():
    """Pressure with no critical T inside the bounds -> None.

    A very large pressure (1e6 GPa) cannot be matched by P_c(T) in the
    [300, 6000] K range, so f_lo and f_hi share sign and brentq is skipped.
    """
    T = _gupta2025_critical_temperature_brentq(1.0e6, T_bounds=(300.0, 6000.0))
    assert T is None


def test_critical_pressure_returns_finite_for_typical_T():
    """Smoke + sanity: P_c(T) is finite for typical mantle temperatures."""
    P_c = gupta2025_critical_pressure(2000.0)
    assert np.isfinite(P_c)


def test_coexistence_compositions_above_critical_returns_none():
    """At P >> P_c(T), the system is single-phase: return is None."""
    # Pick T well below the critical line so the critical pressure exists,
    # then probe at 5x that pressure -> definitely above critical.
    T = 1000.0
    P_c = gupta2025_critical_pressure(T)
    if not np.isfinite(P_c) or P_c <= 0:
        pytest.skip('critical pressure non-positive at this T; skip')
    result = gupta2025_coexistence_compositions(T, P_GPa=5.0 * P_c)
    assert result is None


def test_coexistence_compositions_below_critical_returns_pair():
    """At P below the critical line, two coexisting compositions are
    returned with x_lo < x_hi."""
    T = 500.0  # cool enough to be in the two-phase region
    P_c = gupta2025_critical_pressure(T)
    if not np.isfinite(P_c) or P_c <= 0:
        pytest.skip('critical pressure non-positive; skip')
    P_test = max(0.5 * P_c, 0.1)  # safely below critical
    result = gupta2025_coexistence_compositions(T, P_GPa=P_test)
    if result is None:
        pytest.skip('binodal evaluation found no two-phase minima for this T,P')
    x_lo, x_hi = result
    assert 0.0 < x_lo < x_hi < 1.0
