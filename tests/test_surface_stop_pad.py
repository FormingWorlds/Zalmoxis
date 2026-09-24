"""Mass and gravity padding when the structure integration stops below the outer radius.

A uniform-density sphere has the closed form m(r) = 4/3 pi rho r^3 and
P(r) = P_c - 2/3 pi G rho^2 r^2, so the radius r0 where P reaches zero is
known. P_c is chosen so that r0 falls inside the last radial shell; the
padded mass must then equal m(r0), not the mass at the grid node below r0.
"""

from __future__ import annotations

import numpy as np
import pytest

import zalmoxis.structure_model as sm
from zalmoxis.constants import G

pytestmark = pytest.mark.unit

RHO = 5000.0
R_OUT = 6.0e6
N = 50


def _setup(frac):
    """Return radii and the P_c that puts P = 0 at radii[-2] + frac * dr."""
    radii = np.linspace(0.0, R_OUT, N)
    r0 = radii[-2] + frac * (radii[-1] - radii[-2])
    return radii, r0, 2.0 / 3.0 * np.pi * G * RHO**2 * r0**2


def _rhs(below_zero):
    """Uniform-density structure RHS; ``below_zero`` sets its value at P < 0."""

    def rhs(r, y, *args, **kwargs):
        m, g, p = y
        if p < 0 and below_zero == 'nan':
            return np.full(3, np.nan)
        dgdr = 4.0 * np.pi * G * RHO - (2.0 * g / r if r > 0 else 8.0 / 3.0 * np.pi * G * RHO)
        return np.array([4.0 * np.pi * r**2 * RHO, dgdr, -RHO * g])

    return rhs


def _solve_numpy(monkeypatch, radii, p_c, below_zero, tdep):
    monkeypatch.setattr(sm, 'coupled_odes', _rhs(below_zero))
    monkeypatch.setattr(sm, 'any_component_is_tdep', lambda _: tdep)
    return sm.solve_structure(
        {}, 0.0, 0.0, radii, 0.5, 1e-10, 1e-12, np.inf, {}, {}, [0.0, 0.0, p_c], None, None
    )


class TestNumpyStopPad:
    """numpy solve_structure pads with the state where the integration stopped."""

    @pytest.mark.parametrize('below_zero', ['continue', 'nan'])
    @pytest.mark.parametrize('tdep', [False, True])
    def test_mass_at_stop_radius(self, monkeypatch, below_zero, tdep):
        radii, r0, p_c = _setup(0.6)
        m, g, p = _solve_numpy(monkeypatch, radii, p_c, below_zero, tdep)
        m_r0 = 4.0 / 3.0 * np.pi * RHO * r0**3
        assert p[-1] == 0.0
        assert m[-1] == pytest.approx(m_r0, rel=1e-6)
        assert g[-1] == pytest.approx(G * m_r0 / r0**2, rel=1e-6)
        # The node below the stop keeps its own value; only the pad changes.
        assert m[-2] == pytest.approx(4.0 / 3.0 * np.pi * RHO * radii[-2] ** 3, rel=1e-6)

    def test_mass_continuous_across_stop_onset(self, monkeypatch):
        """M at the outer node varies smoothly as the stop moves out through R."""
        radii, _, p_c = _setup(0.999)
        m_in = _solve_numpy(monkeypatch, radii, p_c, 'continue', False)[0][-1]
        m_out = _solve_numpy(monkeypatch, radii, p_c * 1.003, 'continue', False)[0][-1]
        assert abs(m_out / m_in - 1.0) < 1e-3

    def test_no_stop_is_unchanged(self, monkeypatch):
        """With P > 0 at R there is no pad and m[-1] is m(R)."""
        radii, _, p_c = _setup(1.5)
        m, _, p = _solve_numpy(monkeypatch, radii, p_c, 'continue', False)
        assert p[-1] > 0
        assert m[-1] == pytest.approx(4.0 / 3.0 * np.pi * RHO * R_OUT**3, rel=1e-6)


class TestJaxStopState:
    """solve_structure_jax(return_end=True) returns the state at the P = 0 event."""

    def test_end_state_at_event(self, monkeypatch):
        jnp = pytest.importorskip('jax.numpy')
        import zalmoxis.jax_eos.solver as js

        def rhs(t, y, **kwargs):
            m, g, p = y
            dgdr = jnp.where(
                t > 0,
                4.0 * jnp.pi * G * RHO - 2.0 * g / jnp.where(t > 0, t, 1.0),
                8.0 / 3.0 * jnp.pi * G * RHO,
            )
            return jnp.array([4.0 * jnp.pi * t**2 * RHO, dgdr, -RHO * g])

        monkeypatch.setattr(js, 'coupled_odes_jax', rhs)
        monkeypatch.setattr(js, '_SOLVE_CACHE', {})
        radii, r0, p_c = _setup(0.6)
        ys, y_end = js.solve_structure_jax(
            radii, [0.0, 0.0, p_c], rtol=1e-10, atol=1e-12, return_end=True
        )
        ys, y_end = np.asarray(ys), np.asarray(y_end)
        assert not np.isfinite(ys[-1, 2])
        assert y_end[0] == pytest.approx(4.0 / 3.0 * np.pi * RHO * r0**3, rel=1e-6)
        assert abs(y_end[2]) < 1e-3 * p_c
