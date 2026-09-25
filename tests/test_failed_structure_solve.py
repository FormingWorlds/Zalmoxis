"""Failed structure solves inside the pressure and mass loops of ``main``.

A structure solve that fails away from the surface returns NaN past its stop
(``pad_after_stop``). These tests inject such failures into real solver runs
and check that no failed profile is taken as a solution. The analytic-EOS
cases stop ``main`` after its first ``_solve`` and take a few seconds, so they
are unit tests; the real Newton run and the PALEOS adiabat case are smoke tests.
"""

from __future__ import annotations

import os
import time as _time

import numpy as np
import pytest

import zalmoxis
import zalmoxis.solver as zs
from zalmoxis.config import load_material_dictionaries
from zalmoxis.constants import G, earth_mass

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


class _Stop(Exception):
    """Ends ``main`` after its first ``_solve``, before any retry."""


def _first_solve_result(monkeypatch, cfg):
    """Run main and return the result of its first ``_solve`` call."""
    results, real = [], zs._solve

    def solve(*args, **kwargs):
        results.append(real(*args, **kwargs))
        raise _Stop

    monkeypatch.setattr(zs, '_solve', solve)
    with pytest.raises(_Stop):
        _run(cfg)
    return results[0]


@pytest.mark.unit
class TestFailedPressureSolve:
    """A pressure solve with no root is a failed solve, whatever profile remains."""

    def test_no_root_is_reported_and_not_used(self, monkeypatch):
        """Structure solves fail above P_c = 5e11 Pa, above this radius's root
        (3.9e11 Pa): brentq stops on a NaN and the finite profiles left are bracket
        ends, which must not stand in for the solution."""
        _spy_solve(monkeypatch, lambda radii, y0: y0[2] > 5e11)
        cfg = _cfg(
            outer_solver='picard', max_iterations_outer=1, _initial_radius_guess=6113601.77
        )
        first = _first_solve_result(monkeypatch, cfg)
        assert first['structure_failed'] is True
        assert first['cmb_mass'] == pytest.approx(0.325 * earth_mass)

    def test_timeout_before_any_solve_is_a_failure(self, monkeypatch):
        """A wall-clock stop before the first pressure solve leaves the zero profile."""
        first = _first_solve_result(monkeypatch, _cfg(outer_solver='picard', wall_timeout=-1.0))
        assert first['structure_failed'] is True


def _check_restored_or_failed(result, failed):
    """A failed result is not converged; a restored one is a consistent profile."""
    assert result['structure_failed'] is failed
    if failed:
        assert result['converged'] is False
    else:
        M, R = result['mass_enclosed'][-1], result['radii'][-1]
        assert result['gravity'][-1] == pytest.approx(G * M / R**2, rel=1e-4)
        assert result['pressure'][0] == pytest.approx(result['p_center'])


@pytest.mark.unit
class TestWallClockStop:
    """A wall-clock stop inside the last outer iteration restores the best earlier
    solution, or reports a failure when there is none."""

    @pytest.mark.parametrize(
        'n_outer, fail_first, failed', [(2, False, False), (3, False, False), (3, True, True)]
    )
    def test_stop_before_the_first_pressure_solve(
        self, monkeypatch, n_outer, fail_first, failed
    ):
        """The clock jumps past the limit when the second outer iteration sets up
        its temperatures, so its inner loop stops before any structure solve. With
        3 iterations the third restores the best earlier solution; if every solve
        of the first iteration failed there is none. With 2 iterations the restore
        happens after the loop."""
        from types import SimpleNamespace

        clock, calls, real_tp = {'jump': 0.0}, [], zs.calculate_temperature_profile

        def temperature_profile(*args, **kwargs):
            calls.append(1)
            if len(calls) == 2:
                clock['jump'] = 1e9
            return real_tp(*args, **kwargs)

        monkeypatch.setattr(zs, 'calculate_temperature_profile', temperature_profile)
        if fail_first:
            _spy_solve(monkeypatch, lambda radii, y0: len(calls) == 1)
        monkeypatch.setattr(
            zs, 'time', SimpleNamespace(time=lambda: _time.time() + clock['jump'])
        )
        cfg = _cfg(outer_solver='picard', max_iterations_outer=n_outer, max_iterations_inner=1)
        first = _first_solve_result(monkeypatch, cfg)
        assert len(calls) == 2
        _check_restored_or_failed(first, failed)


@pytest.mark.unit
class TestFailedLastPicardIteration:
    """The last outer iteration's pressure solve finds no root, but leaves finite
    bracket-end profiles (solves fail above P_c = 1e11 Pa, below the root)."""

    @pytest.mark.parametrize('n_outer, failed', [(3, False), (1, True)])
    def test_best_solution_or_failure(self, monkeypatch, n_outer, failed):
        calls, real_tp = [], zs.calculate_temperature_profile

        def temperature_profile(*args, **kwargs):
            calls.append(1)
            return real_tp(*args, **kwargs)

        monkeypatch.setattr(zs, 'calculate_temperature_profile', temperature_profile)
        _spy_solve(monkeypatch, lambda radii, y0: len(calls) == n_outer and y0[2] > 1e11)
        cfg = _cfg(outer_solver='picard', max_iterations_outer=n_outer, max_iterations_inner=1)
        first = _first_solve_result(monkeypatch, cfg)
        _check_restored_or_failed(first, failed)
        if not failed:
            assert first['mass_enclosed'][-1] == pytest.approx(earth_mass, rel=0.05)

    def test_failed_middle_iteration_keeps_the_radius(self, monkeypatch):
        """Outer iteration 2 of 3 fails; iteration 3 starts from the same radius."""
        calls, radius, real_tp = [], {}, zs.calculate_temperature_profile

        def temperature_profile(radii, *args, **kwargs):
            calls.append(1)
            radius[len(calls)] = radii[-1]
            return real_tp(radii, *args, **kwargs)

        monkeypatch.setattr(zs, 'calculate_temperature_profile', temperature_profile)
        _spy_solve(monkeypatch, lambda radii, y0: len(calls) == 2 and y0[2] > 1e11)
        cfg = _cfg(outer_solver='picard', max_iterations_outer=3, max_iterations_inner=1)
        _first_solve_result(monkeypatch, cfg)
        assert len(calls) == 3 and radius[1] != radius[2]
        assert radius[3] == radius[2]


@pytest.mark.smoke
class TestNewtonWithFailedRadius:
    """A full Newton run with a radius that has no pressure root (the fall-back
    cases on a synthetic M(R) are unit tests in test_outer_solver_newton.py)."""

    FAST = {
        'outer_solver': 'newton',
        'num_layers': 30,
        'relative_tolerance': 1e-7,
        'absolute_tolerance': 1e-8,
    }

    def test_failure_at_a_later_radius_recovers(self, monkeypatch, caplog):
        """Every solve fails at the fourth radius (the first Newton step after R0
        and R0 +/- dR); the fall-back sweeps from the best evaluated radius."""
        seen = []

        def fails(radii, y0):
            if radii[-1] not in seen:
                seen.append(radii[-1])
            return len(seen) >= 4 and radii[-1] == seen[3]

        _spy_solve(monkeypatch, fails)
        with caplog.at_level('WARNING', logger='zalmoxis.solver'):
            result = _run(_cfg(**self.FAST))
        assert 'sweeping from the best evaluated' in caplog.text
        assert len(seen) > 4 and result['converged'] is True
        assert result['mass_enclosed'][-1] == pytest.approx(earth_mass, rel=1e-3)


@pytest.mark.smoke
class TestFailedSolveInPaleosAdiabat:
    """A failed solve leaves no NaN in the state the next adiabat step reads."""

    @pytest.mark.parametrize('scope', ['one', 'all'])
    def test_nan_evaluation_is_not_adopted(self, monkeypatch, caplog, scope):
        """'one': one failed evaluation inside brentq. 'all': every solve of one
        outer iteration fails, which leaves no finite profile at all."""
        for sub, name in (
            ('EOS_PALEOS_iron', 'paleos_iron_eos_table_pt.dat'),
            ('EOS_PALEOS_MgSiO3_unified', 'paleos_mgsio3_eos_table_pt.dat'),
        ):
            if not os.path.isfile(os.path.join(ROOT, 'data', sub, name)):
                pytest.skip('PALEOS unified EOS data not available')

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
            _run(cfg)
        assert state['injected'] and len(state['adiabat_args']) > state['injected']
        assert 'calculated_mass=nan' not in caplog.text
        for p_prev, m_prev, cmb, core_mantle in state['adiabat_args']:
            assert np.all(np.isfinite(p_prev)) and np.all(np.isfinite(m_prev))
            assert p_prev[0] > 0 and m_prev[-1] > 0
            assert cmb > 0 and core_mantle > 0
