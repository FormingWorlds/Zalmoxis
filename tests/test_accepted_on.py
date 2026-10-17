"""The ``accepted_on`` and ``mass_error_at_accept`` fields of a structure solve result.

Each accept path of ``main`` names itself in ``accepted_on``; a Newton result is
converged only when the last inner solve reports converged pressure and density.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import zalmoxis.solver as zs
from tests.test_failed_structure_solve import _cfg, _count_outer_iterations, _run
from zalmoxis.constants import earth_mass

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
        'converged_mass': True,
        'accepted_on': accepted_on,
        'mass_error_at_accept': abs(M - M_T) / M_T,
    }


def _newton(**flags):
    cp = {
        'planet_mass': M_T,
        'core_mass_fraction': 0.325,
        'outer_solver': 'newton',
        'relative_tolerance': 1e-9,
        'absolute_tolerance': 1e-10,
        '_initial_radius_guess': 6.0e6,
        'newton_tol': 1e-6,
        'newton_max_iter': 5,
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

    def test_root_with_inner_density_false(self):
        r = _brentq(
            _hist(6.0e6, 8.0e6), lambda R: (A * R + B, _inner(R, density=False)), _inner(8.0e6)
        )
        assert (
            r['converged'] is False and r['converged_mass'] is True and r['accepted_on'] is None
        )

    def test_narrow_bracket_is_brentq_best(self):
        r = _brentq(_hist(7.0e6 - 0.2, 7.0e6 + 0.2), None, _inner(6.5e6))
        assert r['converged'] is True and r['accepted_on'] == 'brentq_best'
        # the returned profile is the last inner solve, far from the best R
        assert r['mass_error_at_accept'] == pytest.approx(0.5)
        assert r['best_mass_error'] < 1e-6

    def test_brentq_error_is_brentq_best(self):
        def boom(R):
            raise ValueError('stale bracket')

        r = _brentq(_hist(6.0e6, 7.0e6 + 1e-3), boom, _inner(6.0e6, pressure=True))
        assert r['converged'] is True and r['accepted_on'] == 'brentq_best'
        r = _brentq(_hist(6.0e6, 7.0e6 + 1e-3), boom, _inner(6.0e6, pressure=False))
        assert r['converged'] is False and r['accepted_on'] is None

    def test_no_bracket_clears_inner_accepted_on(self):
        below = lambda R: (0.5 * M_T, _inner(R))  # noqa: E731
        r = _brentq(_hist(6.0e6), below, _inner(6.0e6))
        assert r['converged'] is False and r['accepted_on'] is None


def _first_attempt(pressure, density):
    """A Picard first attempt that was not accepted, at 5 % mass error."""
    r = _inner(7.0e6, pressure=pressure, density=density, accepted_on=None)
    r.update(converged=False, converged_mass=False, best_mass_error=0.05)
    return r


@pytest.mark.unit
class TestRetry:
    def _main(self, first):
        calls = []

        def solve(cp, *a, **k):
            calls.append(cp.get('wall_timeout'))
            return first if len(calls) == 1 else _inner(7.0e6, accepted_on='strict')

        cfg = {
            'planet_mass': earth_mass,
            'layer_eos_config': {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'},
        }
        with patch('zalmoxis.solver._solve', side_effect=solve):
            return zs.main(cfg, {}, None, '/tmp'), calls

    def test_retry_skip(self):
        r, calls = self._main(_first_attempt(True, True))
        assert len(calls) == 1 and r['converged'] is True and r['accepted_on'] == 'retry_skip'

    def test_retry_reports_its_own_path(self):
        r, calls = self._main(_first_attempt(True, False))
        assert len(calls) == 2 and r['accepted_on'] == 'strict'


def _picard(monkeypatch, wall_at=None, **kwargs):
    """Fast Picard solve (1 M_earth, analytic EOS); the clock jumps at outer iteration ``wall_at``."""
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


@pytest.mark.unit
@pytest.mark.timeout(120)
class TestPicard:
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
