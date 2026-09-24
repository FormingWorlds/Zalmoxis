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

    def test_tail_reaching_the_event_pads_there(self, monkeypatch):
        """The grid solve fails at a band just below r0; the tail, on a healthy
        RHS, reaches P = 0 inside the same shell and supplies the event state."""
        radii, r0, p_c = _setup(0.6)
        dr = radii[1]
        band = _rhs_band_uniform(radii[-2] + 0.1 * dr, radii[-2] + 0.5 * dr)
        real = sm.solve_ivp

        def spy(f, t_span, y0, **kwargs):
            rhs = band if kwargs.get('t_eval') is not None else _rhs('continue')
            return real(rhs, t_span, y0, **kwargs)

        monkeypatch.setattr(sm, 'solve_ivp', spy)
        m, g, p = _solve_numpy(monkeypatch, radii, p_c, 'continue', False)
        assert p[-1] == 0.0
        assert m[-1] == pytest.approx(_mass(r0), rel=1e-6)
        assert g[-1] == pytest.approx(G * _mass(r0) / r0**2, rel=1e-6)


def _rhs_band_uniform(r_lo, r_hi):
    """Uniform-density RHS that returns NaN for r_lo < r < r_hi."""
    healthy = _rhs('continue')

    def rhs(r, y, *args, **kwargs):
        return np.full(3, np.nan) if r_lo < r < r_hi else healthy(r, y)

    return rhs


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


class TestFailedSolveInMain:
    """A failed solve inside the pressure bracket leaves no NaN in the outer state."""

    @pytest.mark.parametrize('scope', ['one', 'all'])
    def test_nan_evaluation_is_not_adopted(self, monkeypatch, caplog, scope):
        """'one': one failed evaluation inside brentq. 'all': every solve of one
        outer iteration fails, which leaves no finite profile at all."""
        import os

        import zalmoxis
        import zalmoxis.solver as zs

        root = os.path.normpath(os.path.join(os.path.dirname(zalmoxis.__file__), '..', '..'))
        for sub, name in (
            ('EOS_PALEOS_iron', 'paleos_iron_eos_table_pt.dat'),
            ('EOS_PALEOS_MgSiO3_unified', 'paleos_mgsio3_eos_table_pt.dat'),
        ):
            if not os.path.isfile(os.path.join(root, 'data', sub, name)):
                pytest.skip('PALEOS unified EOS data not available')
        from zalmoxis.config import load_material_dictionaries
        from zalmoxis.constants import earth_mass

        state = {'in_brentq': False, 'injected': None, 'adiabat_args': []}
        real_solve, real_brentq, real_adiabat = (
            zs.solve_structure,
            zs.brentq,
            zs.compute_adiabatic_temperature,
        )

        def brentq(*args, **kwargs):
            state['in_brentq'] = True
            try:
                return real_brentq(*args, **kwargs)
            finally:
                state['in_brentq'] = False

        def solve(*args, **kwargs):
            m, g, p = real_solve(*args, **kwargs)
            # Fail once the adiabat is active, so the next outer iteration
            # anchors its adiabat on the state this one leaves.
            n_adiabat = len(state['adiabat_args'])
            if scope == 'one':
                hit = state['in_brentq'] and n_adiabat and state['injected'] is None
            else:
                hit = n_adiabat and state['injected'] in (None, n_adiabat)
            if hit:
                state['injected'] = n_adiabat
                m, g, p = (
                    np.where(np.arange(len(m)) >= len(m) // 2, np.nan, a) for a in (m, g, p)
                )
            return m, g, p

        def adiabat(radii, p_prev, m_prev, t_surf, cmb, core_mantle, *args, **kwargs):
            state['adiabat_args'].append((p_prev, m_prev, cmb, core_mantle))
            return real_adiabat(
                radii, p_prev, m_prev, t_surf, cmb, core_mantle, *args, **kwargs
            )

        monkeypatch.setattr(zs, 'brentq', brentq)
        monkeypatch.setattr(zs, 'solve_structure', solve)
        monkeypatch.setattr(zs, 'compute_adiabatic_temperature', adiabat)
        cfg = {
            'planet_mass': earth_mass,
            'core_mass_fraction': 0.325,
            'mantle_mass_fraction': 0,
            'temperature_mode': 'adiabatic',
            'surface_temperature': 3000.0,
            'center_temperature': 6000.0,
            'temp_profile_file': '',
            'layer_eos_config': {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
            'mushy_zone_factor': 1.0,
            'num_layers': 50,
            'target_surface_pressure': 101325,
            'max_iterations_outer': 8,
            'max_iterations_inner': 1,
            'data_output_enabled': False,
            'plotting_enabled': False,
        }
        with caplog.at_level('DEBUG', logger='zalmoxis.solver'):
            zs.main(cfg, load_material_dictionaries(), None, os.path.join(root, 'input'))
        assert state['injected'] and len(state['adiabat_args']) > state['injected']
        assert 'calculated_mass=nan' not in caplog.text
        for p_prev, m_prev, cmb, core_mantle in state['adiabat_args']:
            assert np.all(np.isfinite(p_prev)) and np.all(np.isfinite(m_prev))
            assert p_prev[0] > 0 and m_prev[-1] > 0
            assert cmb > 0 and core_mantle > 0

    def test_newton_reports_a_radius_where_every_solve_fails(self, monkeypatch):
        """A failure at every central pressure of the first radius has no mass to
        offer; Newton must not read the zero-mass profile as M(R) = 0."""
        import os

        import zalmoxis
        import zalmoxis.solver as zs
        from zalmoxis.config import load_material_dictionaries
        from zalmoxis.constants import earth_mass

        root = os.path.normpath(os.path.join(os.path.dirname(zalmoxis.__file__), '..', '..'))
        if not os.path.isfile(
            os.path.join(root, 'data', 'EOS_Seager2007', 'eos_seager07_iron.txt')
        ):
            pytest.skip('Seager2007 EOS data not available')
        real_solve, first_r = zs.solve_structure, []

        def solve(*args, **kwargs):
            m, g, p = real_solve(*args, **kwargs)
            first_r.append(first_r[0] if first_r else args[3][-1])
            if args[3][-1] == first_r[0]:
                m, g, p = (
                    np.where(np.arange(len(m)) >= len(m) // 2, np.nan, a) for a in (m, g, p)
                )
            return m, g, p

        monkeypatch.setattr(zs, 'solve_structure', solve)
        cfg = {
            'planet_mass': earth_mass,
            'core_mass_fraction': 0.325,
            'mantle_mass_fraction': 0,
            'temperature_mode': 'isothermal',
            'surface_temperature': 3000.0,
            'center_temperature': 6000.0,
            'temp_profile_file': '',
            'layer_eos_config': {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'},
            'mushy_zone_factor': 1.0,
            'num_layers': 50,
            'target_surface_pressure': 101325,
            'outer_solver': 'newton',
            'relative_tolerance': 1e-9,
            'absolute_tolerance': 1e-10,
            'data_output_enabled': False,
            'plotting_enabled': False,
        }
        with pytest.raises(RuntimeError, match='failed at every central pressure'):
            zs.main(cfg, load_material_dictionaries(), None, os.path.join(root, 'input'))
