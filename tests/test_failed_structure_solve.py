"""Failed structure solves inside the pressure and mass loops of ``main``.

A structure solve that fails away from the surface returns NaN past its stop
(``pad_after_stop``). These tests inject such failures into real solver runs
with analytic EOS and check that ``main`` raises ``StructureSolveError``
instead of returning a result.
"""

from __future__ import annotations

import os
import re

import numpy as np
import pytest

import zalmoxis
import zalmoxis.solver as zs
import zalmoxis.structure_model as sm
from zalmoxis.config import load_material_dictionaries
from zalmoxis.constants import earth_mass
from zalmoxis.solver import StructureSolveError

ROOT = os.path.normpath(os.path.join(os.path.dirname(zalmoxis.__file__), '..', '..'))


def _cfg(**kwargs):
    """1 M_earth analytic iron/MgSiO3 planet, isothermal, 50 layers."""
    cfg = {
        'planet_mass': earth_mass,
        'core_mass_fraction': 0.325,
        'mantle_mass_fraction': 0,
        'temperature_mode': 'isothermal',
        'surface_temperature': 3000.0,
        'center_temperature': 6000.0,
        'temp_profile_file': '',
        'layer_eos_config': {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'},
        'mushy_zone_factor': 1.0,
        'num_layers': 50,
        'target_surface_pressure': 101325,
        'relative_tolerance': 1e-9,
        'absolute_tolerance': 1e-10,
        'data_output_enabled': False,
        'plotting_enabled': False,
    }
    cfg.update(kwargs)
    return cfg


def _run(cfg):
    return zs.main(cfg, load_material_dictionaries(), None, os.path.join(ROOT, 'input'))


def _fail(m, g, p):
    """The profile of a structure solve that failed half way out."""
    return tuple(np.where(np.arange(len(m)) >= len(m) // 2, np.nan, a) for a in (m, g, p))


def _spy_solve(monkeypatch, fails):
    """Replace structure solves for which ``fails(radii, y0)`` holds by a failed one."""
    real = zs.solve_structure

    def solve(*args, **kwargs):
        out = real(*args, **kwargs)
        return _fail(*out) if fails(args[3], args[10]) else out

    monkeypatch.setattr(zs, 'solve_structure', solve)


def _count_outer_iterations(monkeypatch):
    """Record the outer radius of each outer iteration (one temperature profile each)."""
    radius, real_tp = [], zs.calculate_temperature_profile

    def temperature_profile(radii, *args, **kwargs):
        radius.append(radii[-1])
        return real_tp(radii, *args, **kwargs)

    monkeypatch.setattr(zs, 'calculate_temperature_profile', temperature_profile)
    return radius


@pytest.mark.unit
class TestFailedPressureSolve:
    """A failed structure solve in the pressure loop raises StructureSolveError."""

    def test_failure_above_the_root_raises(self, monkeypatch):
        """Structure solves fail above P_c = 5e11 Pa, above this radius's root
        (3.9e11 Pa): brentq meets a NaN and no finite solution exists."""
        _spy_solve(monkeypatch, lambda radii, y0: y0[2] > 5e11)
        cfg = _cfg(
            outer_solver='picard', max_iterations_outer=1, _initial_radius_guess=6113601.77
        )
        with pytest.raises(
            StructureSolveError,
            match=r'at R = 6\.113602e\+06 m \(outer iteration 0, inner 0\): solve at '
            r'P_c = [0-9.]+e\+\d+ Pa not finite; stop between r = ',
        ):
            _run(cfg)

    def test_one_failed_evaluation_inside_brentq_raises(self, monkeypatch):
        """Only the first structure solve inside brentq fails."""
        state, real_brentq = {'in': False, 'n': 0}, zs.brentq

        def brentq(*args, **kwargs):
            state['in'] = True
            try:
                return real_brentq(*args, **kwargs)
            finally:
                state['in'] = False

        def fails(radii, y0):
            state['n'] += state['in']
            return state['in'] and state['n'] == 1

        monkeypatch.setattr(zs, 'brentq', brentq)
        _spy_solve(monkeypatch, fails)
        with pytest.raises(StructureSolveError, match='outer iteration 0'):
            _run(_cfg(outer_solver='picard'))

    def test_failure_below_the_root_raises(self, monkeypatch):
        """Structure solves fail below P_c = 1e11 Pa, so the low bracket end fails
        while the high one is finite."""
        _spy_solve(monkeypatch, lambda radii, y0: y0[2] < 1e11)
        with pytest.raises(
            StructureSolveError,
            match=r'\(outer iteration 0, inner 0\): solve at P_c = [0-9.]+e\+10 Pa',
        ):
            _run(_cfg(outer_solver='picard'))

    def test_failure_at_the_brent_root_raises(self, monkeypatch):
        """Every solve succeeds except the re-solve at the root brentq returns."""
        calls, real_brentq = {'root': None}, zs.brentq

        def brentq(*args, **kwargs):
            out = real_brentq(*args, **kwargs)
            calls['root'] = out[0]
            return out

        monkeypatch.setattr(zs, 'brentq', brentq)
        _spy_solve(monkeypatch, lambda radii, y0: y0[2] == calls['root'])
        with pytest.raises(StructureSolveError, match='solve at the Brent root'):
            _run(_cfg(outer_solver='picard'))

    def test_every_solve_gets_the_target_surface_pressure(self, monkeypatch):
        """The pad rule reads the target surface pressure of the configuration, in
        the pressure search and in the re-solve at its root (the run stops there)."""
        pressures, state = [], {'root': False}
        real_solve, real_brentq = zs.solve_structure, zs.brentq

        class _Stop(Exception):
            pass

        def brentq(*args, **kwargs):
            out = real_brentq(*args, **kwargs)
            state['root'] = True
            return out

        def solve(*args, **kwargs):
            pressures.append(kwargs['surface_pressure'])
            if state['root']:
                raise _Stop
            return real_solve(*args, **kwargs)

        monkeypatch.setattr(zs, 'brentq', brentq)
        monkeypatch.setattr(zs, 'solve_structure', solve)
        with pytest.raises(_Stop):
            _run(_cfg(outer_solver='picard', target_surface_pressure=2e5))
        assert len(pressures) > 2 and set(pressures) == {2e5}

    @pytest.mark.parametrize('outer_solver', ['picard', 'newton'])
    def test_non_finite_mass_at_the_brent_root_raises(self, monkeypatch, outer_solver):
        """The re-solve at the root has NaN mass in its last nodes and finite pressure."""
        state, real_solve, real_brentq = {'root': False}, zs.solve_structure, zs.brentq

        def brentq(*args, **kwargs):
            out = real_brentq(*args, **kwargs)
            state['root'] = True
            return out

        def solve(*args, **kwargs):
            m, g, p = real_solve(*args, **kwargs)
            if state['root']:
                m = np.array(m, dtype=float)
                m[-3:] = np.nan
            return m, g, p

        monkeypatch.setattr(zs, 'brentq', brentq)
        monkeypatch.setattr(zs, 'solve_structure', solve)
        with pytest.raises(StructureSolveError, match='solve at the Brent root'):
            _run(_cfg(outer_solver=outer_solver, max_iterations_outer=1))


@pytest.mark.unit
class TestNonFiniteDensity:
    """An EOS failure inside the planet stops the integration and fails the solve."""

    @staticmethod
    def _nan_band(monkeypatch, lo, hi):
        state, real_odes, real_rho = {'r': 0.0}, sm.coupled_odes, sm.calculate_mixed_density

        def odes(radius, y, *args, **kwargs):
            state['r'] = radius
            return real_odes(radius, y, *args, **kwargs)

        def rho(*args, **kwargs):
            return np.nan if lo < state['r'] < hi else real_rho(*args, **kwargs)

        monkeypatch.setattr(sm, 'coupled_odes', odes)
        monkeypatch.setattr(sm, 'calculate_mixed_density', rho)

    @pytest.mark.timeout(60)
    def test_nan_density_band_raises_at_the_band(self, monkeypatch):
        """The density is NaN for 2.5e6 m < r < 3.5e6 m."""
        self._nan_band(monkeypatch, 2.5e6, 3.5e6)
        with pytest.raises(StructureSolveError, match='stop between r = ') as exc:
            _run(_cfg(outer_solver='picard'))
        lo, hi = (
            float(v)
            for v in re.findall(r'r = ([0-9.e+]+) and ([0-9.e+]+) m', str(exc.value))[0]
        )
        assert lo < 2.5e6 < hi

    @pytest.mark.timeout(60)
    def test_nan_band_just_below_the_centre_pressure_raises(self, monkeypatch):
        """NaN for P between P_c (1 - 1e-3) / 3 and P_c (1 - 1e-3) of each structure solve."""
        state, real_solve, real_rho = {}, zs.solve_structure, sm.calculate_mixed_density

        def solve(*args, **kwargs):
            state['hi'] = args[10][2] * (1.0 - 1e-3)
            return real_solve(*args, **kwargs)

        def rho(pressure, *args, **kwargs):
            band = state.get('hi', 0.0) / 3.0 < pressure < state.get('hi', 0.0)
            return np.nan if band else real_rho(pressure, *args, **kwargs)

        monkeypatch.setattr(zs, 'solve_structure', solve)
        monkeypatch.setattr(sm, 'calculate_mixed_density', rho)
        with pytest.raises(StructureSolveError, match='not finite; stop '):
            _run(_cfg(outer_solver='picard'))

    @pytest.mark.timeout(60)
    def test_nan_density_at_a_node_raises(self, monkeypatch):
        """The integration is clean, but the density update meets a NaN at a node."""
        real = zs.calculate_mixed_density_batch

        def batch(pressure, *args, **kwargs):
            rho = np.array(real(pressure, *args, **kwargs), dtype=float)
            rho[len(rho) // 2] = np.nan
            return rho

        monkeypatch.setattr(zs, 'calculate_mixed_density_batch', batch)
        with pytest.raises(StructureSolveError, match='density not finite at r = '):
            _run(_cfg(outer_solver='picard'))

    @pytest.mark.timeout(60)
    def test_nan_density_at_the_centre_raises(self, monkeypatch):
        """A NaN density at r = 0 fails the solve at the centre instead of a NaN first step."""
        self._nan_band(monkeypatch, -1.0, 1e5)
        with pytest.raises(StructureSolveError, match='stop at r = 0'):
            _run(_cfg(outer_solver='picard'))


class _FellBack(Exception):
    """Ends main after the first structure solve that fell back to numpy."""


def _synthetic_jax_world(monkeypatch, fill=False):
    """Synthetic PALEOS-format tables (no data files) with NaN density in the core-table
    row at log P = 9.86, which covers 2.7e9 to 1.95e10 Pa. numpy's nearest-neighbour
    fallback fills that row from the valid cells if ``fill``, else it returns NaN."""
    pytest.importorskip('jax')
    from scipy.interpolate import NearestNDInterpolator

    from tests.test_jax_parity_synthetic import _synthetic_world

    world = _synthetic_world()
    core = dict(world['interp_cache']['/synthetic/core.dat'])
    grid = np.array(core['density_grid'], dtype=float)
    grid[(core['unique_log_p'] > 9.8) & (core['unique_log_p'] < 9.9)] = np.nan
    ip, it = np.nonzero(np.isfinite(grid))
    nodes = np.column_stack([core['unique_log_p'][ip], core['unique_log_t'][it]])
    core['density_nn'] = (
        NearestNDInterpolator(nodes, grid[ip, it]) if fill else lambda _: np.nan
    )
    core['density_grid'] = grid
    world['interp_cache']['/synthetic/core.dat'] = core
    world['jax_args']['core_density_grid'] = grid
    monkeypatch.setattr(zs, '_interpolation_cache', dict(world['interp_cache']))
    return world


@pytest.mark.smoke
class TestNonFiniteDensityJax:
    """The JAX path: an EOS failure inside the planet ends the solve and fails it."""

    def test_rhs_non_finite_density_gives_nan_above_zero_pressure(self):
        """NaN derivatives at P > 0 (the solve stops there), zeros at P <= 0 (the
        pressure-zero event ends the integration)."""
        pytest.importorskip('jax')
        from tests.test_jax_parity_synthetic import _synthetic_world
        from zalmoxis.jax_eos.rhs import coupled_odes_jax

        world = _synthetic_world()
        args = dict(world['jax_args'])
        args['core_density_grid'] = np.full_like(args['core_density_grid'], np.nan)
        for pressure, expected in ((1e11, 'nan'), (0.0, 'zero'), (-1e3, 'zero')):
            y = np.array([0.5 * world['cmb_mass'], 5.0, pressure])  # inside the core
            dy = np.asarray(coupled_odes_jax(2e6, y, mantle_is_unified=True, **args))
            if expected == 'nan':
                assert np.all(np.isnan(dy)), (pressure, dy)
            else:
                assert np.all(dy == 0.0), (pressure, dy)

    @pytest.mark.timeout(120)
    def test_nan_table_band_ends_the_jax_solve(self, monkeypatch):
        """The solve stops where P enters the band, with the last accepted state."""
        import zalmoxis.jax_eos.solver as js

        world = _synthetic_jax_world(monkeypatch)
        radii = np.linspace(0.0, 6.4e6, 150)

        out = js.solve_structure_jax(
            radii,
            [0.0, 0.0, 3e11],
            rtol=1e-8,
            atol=1e-10,
            mantle_is_unified=True,
            **world['jax_args'],
        )
        ys, y_end = (np.asarray(a) for a in out)
        n = int(np.argmax(~np.isfinite(ys[:, 2])))
        assert n > 0 and np.all(np.isfinite(ys[:n])) and not np.any(np.isfinite(ys[n:, 2]))
        assert np.all(np.isfinite(y_end)) and 1.9e10 < y_end[2] < ys[n - 1, 2]

    def _main(self, monkeypatch, caplog, fill):
        """Run main on the JAX path until it ends, fails, or a solve fell back to numpy;
        return main's result (None after a fallback) and the ValueErrors the wrapper raised."""
        import zalmoxis.jax_eos.wrapper as jw

        world = _synthetic_jax_world(monkeypatch, fill)
        raised, real, real_solve, out = [], jw.solve_structure_via_jax, zs.solve_structure, []

        def solve(*args, **kwargs):
            out.append(real_solve(*args, **kwargs))
            if raised and np.all(np.isfinite(out[-1])):
                raise _FellBack
            return out[-1]

        def spy(*args, **kwargs):
            try:
                return real(*args, **kwargs)
            except ValueError as exc:
                raised.append(str(exc))
                raise

        monkeypatch.setattr(jw, 'solve_structure_via_jax', spy)
        monkeypatch.setattr(zs, 'solve_structure', solve)
        cfg = _cfg(
            outer_solver='picard',
            use_jax=True,
            relative_tolerance=1e-8,
            layer_eos_config={'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
        )
        with caplog.at_level('WARNING', logger='zalmoxis.structure_model'):
            try:
                return zs.main(cfg, world['mats'], None, os.path.join(ROOT, 'input')), raised
            except _FellBack:
                return None, raised

    @pytest.mark.timeout(600)
    def test_nan_table_band_filled_runs_on_jax(self, monkeypatch, caplog):
        """The JAX grid holds numpy's nearest-cell fill, so main runs to the end on JAX."""
        result, raised = self._main(monkeypatch, caplog, fill=True)
        assert not raised and 'fell back to numpy path' not in caplog.text
        assert result['converged'] and np.all(np.isfinite(result['pressure']))

    @pytest.mark.timeout(120)
    def test_nan_table_band_fails_on_numpy_too(self, monkeypatch, caplog):
        """With no fill on numpy either, the solve that the JAX wrapper handed over fails."""
        with pytest.raises(StructureSolveError, match='stop between r = '):
            self._main(monkeypatch, caplog, fill=False)
        assert 'fell back to numpy path' in caplog.text


@pytest.mark.smoke
class TestFailedPicardIteration:
    """A failed outer iteration raises; no earlier solution stands in for it."""

    @pytest.mark.parametrize('n_outer', [1, 3])
    def test_failed_last_iteration_raises(self, monkeypatch, n_outer):
        """Solves of the last outer iteration fail above P_c = 1e11 Pa, below the root."""
        radius = _count_outer_iterations(monkeypatch)
        _spy_solve(monkeypatch, lambda radii, y0: len(radius) == n_outer and y0[2] > 1e11)
        cfg = _cfg(outer_solver='picard', max_iterations_outer=n_outer, max_iterations_inner=1)
        with pytest.raises(StructureSolveError, match=f'outer iteration {n_outer - 1},'):
            _run(cfg)

    def test_failure_at_one_radius_raises(self, monkeypatch):
        """Every solve at the radius of outer iteration 2 fails."""
        radius = _count_outer_iterations(monkeypatch)
        _spy_solve(monkeypatch, lambda radii, y0: len(radius) >= 2 and radii[-1] == radius[1])
        cfg = _cfg(outer_solver='picard', max_iterations_outer=15, max_iterations_inner=3)
        with pytest.raises(StructureSolveError, match='outer iteration 1,') as exc:
            _run(cfg)
        assert f'at R = {radius[1]:.6e} m' in str(exc.value)
