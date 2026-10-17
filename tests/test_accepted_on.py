"""The ``accepted_on`` and ``mass_error_at_accept`` fields of a structure solve result.

Each accept path of ``main`` names itself in ``accepted_on``; a Newton step result is
converged only when the inner solve of its returned profile reports converged pressure
and density.
"""

from __future__ import annotations

import logging
import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import zalmoxis.solver as zs
from tests.test_failed_structure_solve import ROOT, _cfg, _count_outer_iterations, _run
from tests.test_table_band_report import _need_wb_tables
from zalmoxis.config import load_material_dictionaries
from zalmoxis.constants import earth_mass

INPUT = os.path.join(ROOT, 'input')
M_T = 6.0e24
A, B = 6.0e18, -3.6e25  # M(R) = A R + B, root at R = 7e6 m


def _inner(R, *, pressure=True, density=True, accepted_on='timeout'):
    """Inner Picard result at radius R on the linear M(R)."""
    M = A * R + B
    return {
        'radii': np.array([0.0, R]),
        'mass_enclosed': np.array([0.0, M]),
        'converged': True,
        'converged_pressure': pressure,
        'converged_density': density,
        'accepted_on': accepted_on,
        'mass_error_at_accept': abs(M - M_T) / M_T,
    }


def _newton(cp_over=(), **flags):
    cp = {
        'planet_mass': M_T,
        'core_mass_fraction': 0.325,
        'outer_solver': 'newton',
        'relative_tolerance': 1e-9,
        'absolute_tolerance': 1e-10,
        '_initial_radius_guess': 6.0e6,
        'newton_tol': 1e-6,
        'newton_max_iter': 5,
        **dict(cp_over),
    }
    side = lambda c, *a, **k: _inner(c['_initial_radius_guess'], **flags)  # noqa: E731
    with patch('zalmoxis.solver._solve', side_effect=side):
        return zs._solve_newton_outer(cp, {}, None, '/tmp')


def _brentq(history, eval_M, last_result):
    return zs._brentq_fallback_outer(
        6.0e6,
        M_T,
        eval_M,
        history,
        R_min=2e6,
        R_max=3e7,
        tol=1e-6,
        start_time=0.0,
        last_result=last_result,
    )


def _hist(*radii):
    return [(R, A * R + B, abs(A * R + B - M_T) / M_T) for R in radii]


@pytest.mark.unit
class TestNewton:
    def test_newton_tol(self):
        r = _newton()
        assert r['converged'] is True and r['accepted_on'] == 'newton_tol'
        assert r['mass_error_at_accept'] == pytest.approx(r['newton_history'][-1][2], abs=1e-15)

    def test_bracket_sweep_hit_is_brentq_best(self):
        """One Newton step lands on the root; the bracket sweep meets it there, and the profile
        is that of the step before."""
        r = _newton({'_initial_radius_guess': 6.9e6, 'newton_max_iter': 1}, density=False)
        assert r['converged'] is True and r['accepted_on'] == 'brentq_best'
        assert r['mass_error_at_accept'] == pytest.approx(0.1) and r['best_mass_error'] < 1e-6

    @pytest.mark.parametrize('flag', ['pressure', 'density'])
    def test_inner_flag_false_is_not_converged(self, flag):
        r = _newton(**{flag: False})
        assert r['converged'] is False and r['converged_mass'] is True
        assert r['accepted_on'] is None


@pytest.mark.unit
class TestBrentq:
    def test_root_is_newton_tol(self):
        r = _brentq(_hist(6.0e6, 8.0e6), lambda R: (A * R + B, _inner(R)), _inner(8.0e6))
        assert r['newton_used_brentq'] and r['converged'] is True
        assert r['accepted_on'] == 'newton_tol'

    def test_root_does_not_read_inner_flags(self):
        def eval_M(R):
            return A * R + B, _inner(R, density=False)

        r = _brentq(_hist(6.0e6, 8.0e6), eval_M, _inner(8.0e6))
        assert r['converged'] is True and r['accepted_on'] == 'newton_tol'
        assert r['converged_density'] is False

    def test_narrow_bracket_is_brentq_best(self):
        r = _brentq(_hist(7.0e6 - 0.2, 7.0e6 + 0.2), None, _inner(6.5e6))
        assert r['converged'] is True and r['accepted_on'] == 'brentq_best'
        # the returned profile is the last inner solve, far from the best R
        assert r['mass_error_at_accept'] == pytest.approx(0.5)
        assert r['best_mass_error'] < 1e-6

    def test_brentq_error_is_brentq_best(self):
        def boom(R):
            raise ValueError('stale bracket')

        r = _brentq(_hist(6.0e6, 7.0e6 + 1e-3), boom, _inner(6.0e6))
        assert r['converged'] is True and r['accepted_on'] == 'brentq_best'
        r = _brentq(_hist(6.0e6, 8.0e6), boom, _inner(6.0e6))
        assert r['converged'] is False and r['accepted_on'] is None

    def test_no_bracket_is_not_converged(self):
        """No sign change in [R_min, R_max]: not converged, also with a history entry at the
        root."""
        below = lambda R: (0.5 * M_T, _inner(R))  # noqa: E731
        r = _brentq(_hist(7.0e6), below, _inner(6.0e6))
        assert r['converged'] is False and r['accepted_on'] is None and r['converged_mass']


@pytest.mark.unit
@pytest.mark.parametrize('m', [[], [0.0, np.nan], [0.0, np.inf]])
def test_mass_error_of_a_non_finite_profile_is_none(m):
    assert zs._mass_error(m, M_T) is None


def _first_attempt(pressure, density):
    """A Picard first attempt that was not accepted, at 5 % mass error."""
    r = _inner(7.0e6, pressure=pressure, density=density, accepted_on=None)
    r.update(converged=False, converged_mass=False, best_mass_error=0.05)
    return r


@pytest.mark.unit
class TestRetry:
    def _main(self, first):
        cfg = {
            'planet_mass': earth_mass,
            'layer_eos_config': {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'},
        }
        retry = _inner(7.0e6, accepted_on='strict')
        with patch('zalmoxis.solver._solve', side_effect=[first, retry]) as solve:
            return zs.main(cfg, {}, None, '/tmp'), solve.call_count

    def test_retry_skip(self):
        r, n = self._main(_first_attempt(True, True))
        assert n == 1 and r['converged'] is True and r['accepted_on'] == 'retry_skip'

    def test_retry_reports_its_own_path(self):
        r, n = self._main(_first_attempt(True, False))
        assert n == 2 and r['accepted_on'] == 'strict'


def _picard(monkeypatch, wall_at=None, **kwargs):
    """Fast Picard solve (1 M_earth, analytic EOS); the clock jumps at outer iteration
    ``wall_at``."""
    if wall_at is not None:
        radius = _count_outer_iterations(monkeypatch)
        clock = time.time
        monkeypatch.setattr(
            zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * (len(radius) >= wall_at))
        )
    r = _run(_cfg(**kwargs))
    m = np.asarray(r['mass_enclosed'])
    assert r['mass_error_at_accept'] == pytest.approx(abs(m[-1] - earth_mass) / earth_mass)
    return r


@pytest.mark.smoke
@pytest.mark.timeout(120)
class TestRealSolve:
    def test_strict(self, monkeypatch):
        r = _picard(monkeypatch, max_iterations_inner=5)
        assert r['converged_density'] and r['converged_pressure']
        assert r['accepted_on'] == 'strict'

    def test_mass_only(self, monkeypatch):
        r = _picard(monkeypatch, max_iterations_inner=1)
        assert not r['converged_density'] and r['accepted_on'] == 'mass_only'

    def test_iteration_cap_is_timeout(self, monkeypatch):
        r = _picard(monkeypatch, max_iterations_outer=1, max_iterations_inner=1)
        assert r['converged'] is True and r['accepted_on'] == 'timeout'
        assert r['mass_error_at_accept'] > 1e-3  # above tolerance_outer

    def test_wall_clock_is_timeout(self, monkeypatch, caplog):
        r = _picard(monkeypatch, wall_at=3, max_iterations_inner=1, wall_timeout=100.0)
        assert 'Wall-clock timeout (100s)' in caplog.text
        assert not r['converged_pressure'] and r['accepted_on'] == 'timeout'

    def test_inner_wall_clock_is_timeout(self, monkeypatch, caplog):
        """The clock runs out in the inner loop of the last outer iteration, after its first
        density lookup; without the clock the same solve is mass_only."""
        cfg = _cfg(max_iterations_inner=2, wall_timeout=100.0)
        radius = _count_outer_iterations(monkeypatch)
        assert _run(cfg)['accepted_on'] == 'mass_only'
        last, lookups, real = len(radius), [], zs.calculate_mixed_density_batch
        radius.clear()

        def lookup(*args, **kwargs):
            lookups.append(len(radius))
            return real(*args, **kwargs)

        def late():
            return bool(lookups) and lookups[-1] >= last

        clock = time.time
        monkeypatch.setattr(zs, 'calculate_mixed_density_batch', lookup)
        monkeypatch.setattr(zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * late()))
        r = _run(cfg)
        assert 'Wall-clock timeout in inner loop' in caplog.text
        assert 'Wall-clock timeout (100s)' not in caplog.text
        assert r['converged'] is True and r['accepted_on'] == 'timeout'

    def test_wall_clock_without_an_accept(self):
        """The clock runs out before the first profile: not accepted (``main`` would retry)."""
        r = zs._solve(_cfg(wall_timeout=-1.0), load_material_dictionaries(), None, INPUT)
        assert r['converged'] is False and r['accepted_on'] is None

    def test_newton(self):
        r = _run(_cfg(outer_solver='newton', max_iterations_inner=8))
        assert r['converged'] is True and r['accepted_on'] == 'newton_tol'
        assert r['mass_error_at_accept'] < 1e-4  # newton_tol

    def test_newton_with_unconverged_density_is_not_converged(self):
        r = _run(_cfg(outer_solver='newton', max_iterations_inner=1))
        assert r['converged_mass'] and not r['converged_density']
        assert r['converged'] is False and r['accepted_on'] is None


@pytest.mark.smoke
@pytest.mark.wolfbower
@pytest.mark.timeout(300)
def test_wall_clock_in_the_adiabat_ramp_is_timeout(monkeypatch, caplog):
    """The clock runs out after the mass converged, in the adiabat blend ramp: still
    'timeout'."""
    from zalmoxis.eos import get_solidus_liquidus_functions

    _need_wb_tables()
    caplog.set_level(logging.INFO, logger='zalmoxis.solver')
    ramp = lambda: any('activating adiabat blend' in m for m in caplog.messages)  # noqa: E731
    clock = time.time
    monkeypatch.setattr(zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * ramp()))
    cfg = _cfg(
        temperature_mode='adiabatic',
        layer_eos_config={'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'},
        tolerance_outer=0.02,
        max_iterations_inner=2,
        wall_timeout=100.0,
    )
    r = zs.main(cfg, load_material_dictionaries(), get_solidus_liquidus_functions(), '/tmp')
    assert 'Wall-clock timeout (100s)' in caplog.text
    assert r['best_mass_error'] < 0.02  # below tolerance_outer: the chain alone says mass_only
    assert r['converged'] is True and r['accepted_on'] == 'timeout'
