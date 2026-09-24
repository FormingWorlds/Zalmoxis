"""Mass and gravity padding when the structure integration stops below the outer radius.

A uniform-density sphere has the closed form m(r) = 4/3 pi rho r^3 and
P(r) = P_c - 2/3 pi G rho^2 r^2, so the radius r0 where P reaches zero is
known. P_c is chosen so that r0 falls between two grid nodes; the padded mass
must then equal m(r0), not the mass at the grid node below r0.
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


def _mass(r):
    return 4.0 / 3.0 * np.pi * RHO * r**3


def _setup(frac, node=-2):
    """Return radii, r0 and the P_c that puts P = 0 at radii[node] + frac * dr."""
    radii = np.linspace(0.0, R_OUT, N)
    r0 = radii[node] + frac * (radii[1] - radii[0])
    return radii, r0, 2.0 / 3.0 * np.pi * G * RHO**2 * r0**2


def _rhs(below_zero):
    """Uniform-density structure RHS; ``below_zero`` sets its value at P <= 0.

    'continue' keeps integrating (the event fires), 'zero' freezes the state as
    coupled_odes does, 'nan' makes every step past P = 0 fail.
    """

    def rhs(r, y, *args, **kwargs):
        m, g, p = y
        if p <= 0 and below_zero == 'zero':
            return np.zeros(3)
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

    @pytest.mark.parametrize('below_zero', ['continue', 'zero', 'nan'])
    @pytest.mark.parametrize('tdep', [False, True])
    @pytest.mark.parametrize('node', [-2, 10])
    def test_mass_at_stop_radius(self, monkeypatch, below_zero, tdep, node):
        """Stop in the last shell, or at node 10 (inside the first of the two Tdep solves)."""
        radii, r0, p_c = _setup(0.6, node)
        m, g, p = _solve_numpy(monkeypatch, radii, p_c, below_zero, tdep)
        i_stop = node % N + 1
        assert np.all(p[i_stop:] == 0.0)
        assert m[i_stop:] == pytest.approx(np.full(N - i_stop, _mass(r0)), rel=1e-6)
        assert g[-1] == pytest.approx(G * _mass(r0) / r0**2, rel=1e-6)
        # Nodes below the stop keep their own values; only the pad changes.
        assert m[1:i_stop] == pytest.approx(_mass(radii[1:i_stop]), rel=1e-6)

    def test_interior_failure_does_not_restart_at_the_split(self, monkeypatch):
        """A step failure inside the first Tdep solve stops there; the second solve,
        which starts at the split radius, must not supply the pad state. Only a
        failure (status -1) with a healthy RHS past the split separates the two."""
        node = 10
        radii, r0, p_c = _setup(0.6, node)
        r_split = radii[int(0.5 * N) - 1]  # adaptive_radial_fraction 0.5 below
        inner, outer = _rhs('nan'), _rhs('continue')
        monkeypatch.setattr(
            sm, 'coupled_odes', lambda r, y, *a, **k: (inner if r < r_split else outer)(r, y)
        )
        monkeypatch.setattr(sm, 'any_component_is_tdep', lambda _: True)
        m, g, p = sm.solve_structure(
            {}, 0.0, 0.0, radii, 0.5, 1e-10, 1e-12, np.inf, {}, {}, [0.0, 0.0, p_c], None, None
        )
        k = node + 1
        assert m[k:] == pytest.approx(np.full(N - k, _mass(r0)), rel=1e-6)
        assert g[-1] == pytest.approx(G * _mass(r0) / r0**2, rel=1e-6)
        assert np.all(p[k:] == 0.0)
        assert m[1:k] == pytest.approx(_mass(radii[1:k]), rel=1e-6)

    def test_mass_continuous_across_stop_onset(self, monkeypatch):
        """M at the outer node varies smoothly as the stop moves out through R."""
        radii, _, p_c = _setup(0.999)
        m_in = _solve_numpy(monkeypatch, radii, p_c, 'continue', False)[0][-1]
        m_out = _solve_numpy(monkeypatch, radii, p_c * 1.003, 'continue', False)[0][-1]
        assert m_out / m_in == pytest.approx(1.0, abs=1e-3)

    def test_no_stop_is_unchanged(self, monkeypatch):
        """With P > 0 at R there is no pad and m[-1] is m(R)."""
        radii, _, p_c = _setup(1.5)
        m, _, p = _solve_numpy(monkeypatch, radii, p_c, 'continue', False)
        assert p[-1] > 0
        assert m[-1] == pytest.approx(_mass(R_OUT), rel=1e-6)


def _band_rhs(r_lo, r_hi):
    """Exponential-density RHS that returns NaN for r_lo < r < r_hi, where P > 0.

    Unlike the uniform sphere, RK45 is not exact here, so its steps are short
    enough to run into the band.
    """

    def rhs(r, y, *args, **kwargs):
        if r_lo < r < r_hi:
            return np.full(3, np.nan)
        m, g, p = y
        rho = RHO * np.exp(-r / R_OUT)
        dgdr = 4.0 * np.pi * G * rho - (2.0 * g / r if r > 0 else 8.0 / 3.0 * np.pi * G * RHO)
        return np.array([4.0 * np.pi * r**2 * rho, dgdr, -rho * g])

    return rhs


class TestInteriorStopFails:
    """A stop far below the surface is a failed solve, unless a restart passes it."""

    # P reaches zero near 0.77 R_OUT without a band.
    P_C = 2.0 / 3.0 * np.pi * G * RHO**2 * (0.8 * R_OUT) ** 2 * 0.4

    def _solve(self, monkeypatch, band, tdep=False, surface_pressure=0.0, max_step=np.inf):
        radii = np.linspace(0.0, R_OUT, N)
        r_lo = band[0] * R_OUT
        monkeypatch.setattr(sm, 'coupled_odes', _band_rhs(r_lo, r_lo + band[1] * radii[1]))
        monkeypatch.setattr(sm, 'any_component_is_tdep', lambda _: tdep)
        m, g, p = sm.solve_structure(
            {},
            0.0,
            0.0,
            radii,
            0.5,
            1e-10,
            1e-12,
            max_step,
            {},
            {},
            [0.0, 0.0, self.P_C],
            None,
            None,
            surface_pressure=surface_pressure,
        )
        return radii, r_lo, m, g, p

    @pytest.mark.parametrize(
        'band, tdep', [((0.34, 0.1), False), ((0.38, 0.1), False), ((0.34, 0.1), True)]
    )
    def test_nan_band_is_a_failed_solve(self, monkeypatch, caplog, band, tdep):
        with caplog.at_level('WARNING', logger='zalmoxis.structure_model'):
            radii, r_lo, m, g, p = self._solve(monkeypatch, band, tdep)
        live = radii <= r_lo
        assert np.all(np.isfinite(m[live])) and np.all(p[live] > 0)
        assert np.all(np.isnan(m[~live])) and np.all(np.isnan(g[~live]))
        assert np.all(np.isnan(p[~live]))
        assert 'treating the solve as failed' in caplog.text

    @pytest.mark.parametrize('factor, padded', [(1.01, True), (0.5, False)])
    def test_stop_below_target_pressure_is_a_surface(self, monkeypatch, factor, padded):
        """Below the target surface pressure the stop pads with P = 0, so the
        pressure solve sees a central pressure that is too low."""
        radii, r_lo, m, _, p = self._solve(monkeypatch, (0.34, 0.1))
        p_last = p[radii <= r_lo][-1]
        _, _, m, _, p = self._solve(monkeypatch, (0.34, 0.1), surface_pressure=factor * p_last)
        dead = radii > r_lo
        if padded:
            assert np.all(p[dead] == 0.0) and np.all(np.isfinite(m))
        else:
            assert np.all(np.isnan(p[dead]))

    def test_restart_that_passes_resumes_the_grid(self, monkeypatch):
        """The one-shell restart steps over this band; integration resumes on the
        grid and pads at the real P = 0 crossing."""
        calls = []
        real = sm.solve_ivp

        def spy(f, t_span, y0, **kwargs):
            calls.append((t_span[0], kwargs.get('t_eval') is not None))
            return real(f, t_span, y0, **kwargs)

        monkeypatch.setattr(sm, 'solve_ivp', spy)
        radii, _, m, g, p = self._solve(monkeypatch, (0.46, 0.3))
        assert any(t0 > 0 and grid for t0, grid in calls)
        clean = self._solve(monkeypatch, (2.0, 0.0))[2:]
        assert m == pytest.approx(clean[0], rel=1e-8)
        assert g == pytest.approx(clean[1], rel=1e-8)
        assert np.all(p[clean[2] == 0.0] == 0.0)

    @pytest.mark.parametrize(
        'band, tdep, expected', [((0.6, 0.3), True, 7e4), ((0.34, 0.1), True, np.inf)]
    )
    def test_tail_uses_the_segment_max_step(self, monkeypatch, band, tdep, expected):
        """After the Tdep split the tail takes maximum_step; before it, none."""
        steps = []
        real = sm.solve_ivp

        def spy(f, t_span, y0, **kwargs):
            if kwargs.get('t_eval') is None:
                steps.append(kwargs['max_step'])
            return real(f, t_span, y0, **kwargs)

        monkeypatch.setattr(sm, 'solve_ivp', spy)
        self._solve(monkeypatch, band, tdep, max_step=7e4)
        assert steps and steps[0] == expected

    def _tail_case(self, monkeypatch, shell, p_c):
        """Grid solve on a NaN band inside ``shell``; the tail runs on a healthy RHS."""
        radii = np.linspace(0.0, R_OUT, N)
        lo = radii[shell - 1]
        band, healthy = _band_rhs(lo + 0.2 * radii[1], lo + 0.6 * radii[1]), _band_rhs(-1, -1)
        tails, real = [], sm.solve_ivp

        def spy(f, t_span, y0, **kwargs):
            grid = kwargs.get('t_eval') is not None
            sol = real(band if grid else healthy, t_span, y0, **kwargs)
            if not grid:
                tails.append(sol.status)
            return sol

        monkeypatch.setattr(sm, 'solve_ivp', spy)
        monkeypatch.setattr(sm, 'any_component_is_tdep', lambda _: False)
        m, g, p = sm.solve_structure(
            {}, 0.0, 0.0, radii, 0.5, 1e-10, 1e-12, np.inf, {}, {}, [0.0, 0.0, p_c], None, None
        )
        clean = real(
            healthy,
            (0.0, R_OUT),
            [0.0, 0.0, p_c],
            rtol=1e-10,
            atol=1e-12,
            events=lambda r, y: y[2],
            t_eval=radii,
        )
        return m, g, p, tails, clean

    def test_tail_reaching_the_event_pads_there(self, monkeypatch):
        """The tail reaches P = 0 inside the stop shell and supplies the event state."""
        m, g, p, tails, clean = self._tail_case(monkeypatch, 38, self.P_C)
        assert tails == [1]
        m0, g0, _ = clean.y_events[0][0]
        assert m[38:] == pytest.approx(np.full(N - 38, m0), rel=1e-8)
        assert g[-1] == pytest.approx(g0, rel=1e-8) and np.all(p[38:] == 0.0)

    def test_restart_that_reaches_the_outer_radius(self, monkeypatch):
        """A restart that passes a failure in the last shell completes the profile."""
        m, g, p, tails, clean = self._tail_case(monkeypatch, N - 1, 3.0 * self.P_C)
        assert tails == [0]
        assert len(m) == N and p[-1] > 0
        assert m[-1] == pytest.approx(clean.y[0, -1], rel=1e-8)

    def test_every_passing_restart_resumes(self, monkeypatch):
        """Five narrow bands; three stop the grid solve and each restart passes
        them. The profile ends as the band-free one."""
        radii = np.linspace(0.0, R_OUT, N)
        dr, base = radii[1], _band_rhs(-1, -1)
        los = [(k + 0.5) * dr for k in (7, 12, 17, 21, 24)]

        def rhs(r, y, *args, **kwargs):
            if any(lo < r < lo + 0.1 * dr for lo in los):
                return np.full(3, np.nan)
            return base(r, y)

        tails, real = [], sm.solve_ivp

        def spy(f, t_span, y0, **kwargs):
            sol = real(f, t_span, y0, **kwargs)
            if kwargs.get('t_eval') is None:
                tails.append(sol.status)
            return sol

        monkeypatch.setattr(sm, 'solve_ivp', spy)
        monkeypatch.setattr(sm, 'coupled_odes', rhs)
        monkeypatch.setattr(sm, 'any_component_is_tdep', lambda _: False)
        m, g, p = sm.solve_structure(
            {},
            0.0,
            0.0,
            radii,
            0.5,
            1e-10,
            1e-12,
            np.inf,
            {},
            {},
            [0.0, 0.0, self.P_C],
            None,
            None,
        )
        clean = self._solve(monkeypatch, (2.0, 0.0))[2:]
        assert tails == [0, 0, 0]
        assert m == pytest.approx(clean[0], rel=1e-6)
        assert np.all((p == 0.0) == (clean[2] == 0.0))


class TestPadAfterStop:
    """pad_after_stop pads at or below the surface pressure limit and fails above it."""

    @pytest.mark.parametrize(
        'p_frac, p_surface, padded',
        [(0.0, 0.0, True), (0.9e-6, 0.0, True), (1.1e-6, 0.0, False), (1e-3, 2e8, True)],
    )
    def test_pressure_limit(self, p_frac, p_surface, padded):
        radii = np.linspace(0.0, 1.0, 5)
        m, g, p = sm.pad_after_stop(
            radii,
            np.ones(3),
            np.ones(3),
            np.ones(3),
            [2.0, 3.0, p_frac * 1e11],
            1e11,
            p_surface,
        )
        assert len(m) == 5
        if padded:
            assert list(m[3:]) == [2.0, 2.0] and list(g[3:]) == [3.0, 3.0]
            assert list(p[3:]) == [0.0, 0.0]
        else:
            assert np.all(np.isnan(m[3:])) and np.all(np.isnan(p[3:]))


class TestJaxStopState:
    """solve_structure_jax returns the state at the P = 0 event as its end state."""

    @staticmethod
    def _rhs(rho=RHO):
        jnp = pytest.importorskip('jax.numpy')

        def rhs(t, y, **kwargs):
            m, g, p = y
            dgdr = jnp.where(
                t > 0,
                4.0 * jnp.pi * G * rho - 2.0 * g / jnp.where(t > 0, t, 1.0),
                8.0 / 3.0 * jnp.pi * G * rho,
            )
            return jnp.array([4.0 * jnp.pi * t**2 * rho, dgdr, -rho * g])

        return rhs

    # Earth-like and Mars-like uniform spheres, at a tight and the default tolerance.
    @pytest.mark.parametrize('rho, r_out', [(RHO, R_OUT), (3900.0, 3.4e6)])
    @pytest.mark.parametrize('rtol, atol', [(1e-10, 1e-12), (1e-5, 1e-6)])
    def test_end_state_at_event(self, monkeypatch, rho, r_out, rtol, atol):
        """The event state is within the surface pad limit, so a JAX surface stop pads."""
        pytest.importorskip('jax')
        import zalmoxis.jax_eos.solver as js

        monkeypatch.setattr(js, 'coupled_odes_jax', self._rhs(rho))
        monkeypatch.setattr(js, '_SOLVE_CACHE', {})
        radii = np.linspace(0.0, r_out, N)
        r0 = radii[-2] + 0.6 * radii[1]
        p_c = 2.0 / 3.0 * np.pi * G * rho**2 * r0**2
        m0 = 4.0 / 3.0 * np.pi * rho * r0**3
        ys, y_end = js.solve_structure_jax(radii, [0.0, 0.0, p_c], rtol=rtol, atol=atol)
        ys, y_end = np.asarray(ys), np.asarray(y_end)
        assert not np.isfinite(ys[-1, 2])
        assert y_end[0] == pytest.approx(m0, rel=10 * rtol)
        assert y_end[1] == pytest.approx(G * m0 / r0**2, rel=10 * rtol)
        assert abs(y_end[2]) <= sm.SURFACE_STOP_P_FRACTION * p_c

    def test_end_state_without_event(self, monkeypatch):
        """With P > 0 at R the end state is the state at radii[-1]."""
        pytest.importorskip('jax')
        import zalmoxis.jax_eos.solver as js

        monkeypatch.setattr(js, 'coupled_odes_jax', self._rhs())
        monkeypatch.setattr(js, '_SOLVE_CACHE', {})
        radii, _, p_c = _setup(1.5)
        ys, y_end = js.solve_structure_jax(radii, [0.0, 0.0, p_c], rtol=1e-10, atol=1e-12)
        ys, y_end = np.asarray(ys), np.asarray(y_end)
        assert np.all(np.isfinite(ys)) and ys[-1, 2] > 0
        assert y_end == pytest.approx(ys[-1], rel=1e-12)
        assert y_end[0] == pytest.approx(_mass(R_OUT), rel=1e-6)
