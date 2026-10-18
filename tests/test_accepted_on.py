"""The ``accepted_on`` and ``mass_error_at_accept`` fields of a structure solve result.

Each accept path of ``main`` names itself in ``accepted_on``; the returned density is
rho_EOS(P, T) at the returned nodes, and a Newton step result is converged only when the
inner solve of its returned profile reports converged pressure and that density.
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
from zalmoxis.constants import G, earth_mass
from zalmoxis.mixing import calculate_mixed_density, parse_all_layer_mixtures

INPUT = os.path.join(ROOT, 'input')
M_T = 6.0e24
A, B = 6.0e18, -3.6e25  # M(R) = A R + B, root at R = 7e6 m
ICE = {  # 3 layers: cmb_mass and core_mantle_mass differ
    'mantle_mass_fraction': 0.6,
    'num_layers': 200,  # the trapezoid error at the ice density jump stays below 1e-2
    'layer_eos_config': {
        'core': 'Analytic:iron',
        'mantle': 'Analytic:MgSiO3',
        'ice_layer': 'Analytic:H2O',
    },
}
WALL = 1e6  # wall_timeout [s]: the test clock jumps by 1e9 s, a real solve never reaches it


def _inner(R, *, pressure=True, density=True, eos=True, accepted_on='max_iter', wall=False):
    """Inner Picard result at radius R on the linear M(R)."""
    M = A * R + B
    return {
        'radii': np.array([0.0, R]),
        'mass_enclosed': np.array([0.0, M]),
        'converged': True,
        'converged_pressure': pressure,
        'converged_density': density,
        'density_from_eos': eos,
        'accepted_on': accepted_on,
        'mass_error_at_accept': abs(M - M_T) / M_T,
        'stopped_on_wall_clock': wall,
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
        r = _newton({'_initial_radius_guess': 6.9e6, 'newton_max_iter': 1}, eos=False)
        assert r['converged'] is True and r['accepted_on'] == 'brentq_best'
        assert r['mass_error_at_accept'] == pytest.approx(0.1) and r['best_mass_error'] < 1e-6

    @pytest.mark.parametrize('false', [False, np.False_])
    @pytest.mark.parametrize('flag', ['pressure', 'eos'])
    def test_inner_flag_false_is_not_converged(self, flag, false):
        r = _newton(**{'pressure': np.True_, 'eos': np.True_, flag: false})
        assert r['converged'] is False and r['converged_mass'] is True
        assert r['accepted_on'] is None

    def test_density_flag_is_not_read(self):
        r = _newton(density=False)
        assert r['converged'] is True and r['accepted_on'] == 'newton_tol'

    def test_numpy_flags_give_a_python_bool(self):
        r = _newton(pressure=np.True_, eos=np.True_)
        assert type(r['converged']) is bool and r['accepted_on'] == 'newton_tol'

    def test_inner_wall_clock_stop_is_timeout(self):
        assert _newton(wall=True)['accepted_on'] == 'timeout'
        assert _newton(wall=True, eos=False)['accepted_on'] is None


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

    def test_root_on_a_wall_clock_stop_is_timeout(self):
        r = _brentq(_hist(6.0e6, 8.0e6), lambda R: (A * R + B, _inner(R, wall=True)), None)
        assert r['converged'] is True and r['accepted_on'] == 'timeout'

    def test_result_without_a_profile_has_the_keys(self):
        r = _brentq(_hist(7.0e6 - 0.2, 7.0e6 + 0.2), None, None)
        assert r['accepted_on'] == 'brentq_best' and r['mass_error_at_accept'] is None
        assert r['stopped_on_wall_clock'] is False and r['density_from_eos'] is False

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

    def test_retry_skip_on_a_wall_clock_stop_is_timeout(self):
        first = _first_attempt(True, True)
        first['stopped_on_wall_clock'] = True
        r, n = self._main(first)
        assert n == 1 and r['converged'] is True and r['accepted_on'] == 'timeout'

    def test_retry_reports_its_own_path(self):
        r, n = self._main(_first_attempt(True, False))
        assert n == 2 and r['accepted_on'] == 'strict'

    @pytest.mark.parametrize(
        'R, seed', [(1.26e6, None), (1.27e6, 1.27e6), (3.16e7, 3.16e7), (3.17e7, None)]
    )
    def test_retry_radius_seed_window(self, R, seed):
        """The retry starts from the returned radius only between 0.2 and 5 times
        R_E (M/M_E)^0.282, 1.2671e6 to 3.1677e7 m at 1 M_E."""
        first = _first_attempt(True, False)
        first['radii'] = np.array([0.0, R])
        cfg = {
            'planet_mass': earth_mass,
            'layer_eos_config': {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'},
        }
        with patch('zalmoxis.solver._solve', side_effect=[first, _inner(7.0e6)]) as solve:
            zs.main(cfg, {}, None, '/tmp')
        assert solve.call_args_list[1].args[0].get('_initial_radius_guess') == seed


def _assert_eos_density(r, eos=None, melt=(None, None), mzf=None):
    """The returned density is rho_EOS(P, T) at every P > 0 node, from a direct EOS call;
    the core nodes form a block from the centre."""
    mixes = parse_all_layer_mixtures(eos or _cfg()['layer_eos_config'])
    md = load_material_dictionaries()
    P, T, rho = (np.asarray(r[k], float) for k in ('pressure', 'temperature', 'density'))
    assert r['density_from_eos'] is True
    core = []
    for i in np.where(P > 0)[0]:
        ref = {
            k: calculate_mixed_density(P[i], T[i], mixes[k], md, *melt, {}, mzf) for k in mixes
        }
        # rel 1e-8: batch and scalar table lookups differ
        hit = [k for k, v in ref.items() if rho[i] == pytest.approx(v, rel=1e-8)]
        assert hit, (i, rho[i], ref)
        core.append(hit[0] == 'core')
    assert core == sorted(core, reverse=True) and core[0]


def _picard(integrated=True, **kwargs):
    """Fast Picard solve (1 M_earth, analytic EOS) through ``main``; ``cmb_mass`` and
    ``core_mantle_mass`` are 0.325 and 0.325 + ``mantle_mass_fraction`` of the returned mass.
    When ``mass_enclosed`` is the integral of the returned profile (``integrated``), at the
    last node with P > 0 the gravity is G M / r^2 and the trapezoid mass of the returned
    density matches ``mass_enclosed``."""
    r = _run(_cfg(**{'wall_timeout': WALL, **kwargs}))
    m, rad = np.asarray(r['mass_enclosed']), np.asarray(r['radii'])
    assert r['mass_error_at_accept'] == pytest.approx(abs(m[-1] - earth_mass) / earth_mass)
    _assert_eos_density(r, kwargs.get('layer_eos_config'))
    mmf = kwargs.get('mantle_mass_fraction', 0)
    for key, f in (('cmb_mass', 0.325), ('core_mantle_mass', 0.325 + mmf)):
        assert r[key] == pytest.approx(f * m[-1], rel=1e-12), key
    k = np.flatnonzero(np.asarray(r['pressure']) > 0)[-1] + 1  # padded nodes have P = 0
    m, rad, rho = m[:k], rad[:k], np.asarray(r['density'])[:k]
    if integrated:
        assert r['gravity'][k - 1] == pytest.approx(G * m[-1] / rad[-1] ** 2, rel=1e-6)
        assert abs(np.trapezoid(4 * np.pi * rad**2 * rho, rad) / m[-1] - 1) < 1e-2
    return r


def _clock_after_lookup(monkeypatch, outer=0, count=1):
    """The clock jumps by 1e9 s after ``count`` density lookups in outer iteration ``outer``
    or later (0-based); returns the outer radius list."""
    radius, lookups = _count_outer_iterations(monkeypatch), []
    real, clock = zs.calculate_mixed_density_batch, time.time

    def lookup(*args, **kwargs):
        if len(radius) > outer:
            lookups.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(zs, 'calculate_mixed_density_batch', lookup)
    late = lambda: len(lookups) >= count  # noqa: E731
    monkeypatch.setattr(zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * late()))
    return radius


@pytest.mark.unit
@pytest.mark.timeout(60)
def test_outer_wall_clock_is_timeout(monkeypatch, caplog):
    """The clock runs out after the single inner iteration of outer iteration 1, so only the
    outer check sees it; without the clock the stop is 'max_iter'."""
    _clock_after_lookup(monkeypatch, outer=1)
    r = _picard(max_iterations_outer=3, max_iterations_inner=1)
    assert 'Wall-clock timeout (1000000s)' in caplog.text
    assert 'Wall-clock timeout in inner loop' not in caplog.text
    assert r['converged'] is True and r['accepted_on'] == 'timeout'
    assert r['stopped_on_wall_clock'] is True


@pytest.mark.smoke
class TestRealSolve:
    def test_strict(self):
        r = _picard()  # inner cap 50: the density converges below tolerance_inner
        assert r['converged_density'] and r['converged_pressure']
        assert r['accepted_on'] == 'strict' and r['stopped_on_wall_clock'] is False

    def test_mass_only(self):
        r = _picard(max_iterations_inner=1)
        assert not r['converged_density'] and r['accepted_on'] == 'mass_only'

    def test_iteration_cap_is_max_iter(self):  # the last node is padded (P = 0) on macOS
        r = _picard(max_iterations_outer=3, max_iterations_inner=1)
        assert r['converged'] is True and r['accepted_on'] == 'max_iter'
        assert r['mass_error_at_accept'] > 1e-3  # above tolerance_outer

    def test_iteration_cap_of_one_pass_is_max_iter(self):
        """``_solve`` itself: through ``main`` a first attempt that is not accepted would
        reach the retry, with the outer cap doubled. The guess lies 0.4 % below the root
        (6.114e6 m): one pass ends at about 1 % mass error, under the 3 % limit."""
        cfg = _cfg(max_iterations_outer=1, max_iterations_inner=1, _initial_radius_guess=6.09e6)
        r = zs._solve({**cfg, 'wall_timeout': WALL}, load_material_dictionaries(), None, INPUT)
        assert r['converged'] is True and r['accepted_on'] == 'max_iter'
        assert 1e-3 < r['mass_error_at_accept'] < 0.03  # above tolerance_outer

    def test_returned_profile_is_not_the_best(self, monkeypatch):
        """The last outer iteration lands farther from the target mass than the one before;
        ``mass_error_at_accept`` is that of the returned (last) profile."""
        radius, real = _count_outer_iterations(monkeypatch), zs.solve_structure

        def solve(*args, **kwargs):
            m, *rest = real(*args, **kwargs)
            return (m * (1.02 if len(radius) == 3 else 1.0), *rest)

        monkeypatch.setattr(zs, 'solve_structure', solve)
        r = _picard(False, max_iterations_outer=3, max_iterations_inner=1)  # mass scaled
        assert r['accepted_on'] == 'max_iter'
        assert r['mass_error_at_accept'] > 1.5 * r['best_mass_error']

    @pytest.mark.parametrize(
        'inner, count, label',
        [(2, 1, 'mass_only'), (50, 20, 'strict')],
        ids=['mass_only', 'strict'],
    )
    def test_inner_wall_clock_is_timeout(self, monkeypatch, caplog, inner, count, label):
        """The clock runs out in the inner loop of the last outer iteration, after ``count``
        density lookups; without the clock the same solve is accepted on ``label``."""
        cfg = _cfg(max_iterations_inner=inner, wall_timeout=WALL)
        radius = _count_outer_iterations(monkeypatch)
        assert _run(cfg)['accepted_on'] == label
        monkeypatch.undo()
        _clock_after_lookup(monkeypatch, outer=len(radius) - 1, count=count)
        r = _run(cfg)
        assert 'Wall-clock timeout in inner loop' in caplog.text
        assert 'Wall-clock timeout (1000000s)' not in caplog.text
        assert bool(r['converged_density']) == (label == 'strict')
        assert r['converged'] is True and r['accepted_on'] == 'timeout'

    def test_retry_skip_on_an_inner_wall_clock_cut_is_timeout(self, monkeypatch, caplog):
        """One outer iteration cut by the clock in its inner loop, 5.6 % mass error: ``main``
        skips the retry on the density flag of the cut."""
        caplog.set_level(logging.INFO, logger='zalmoxis.solver')
        _clock_after_lookup(monkeypatch, count=8)
        r = _picard(
            max_iterations_outer=1,
            max_iterations_inner=30,
            _initial_radius_guess=0.97 * 6432000.0,
        )
        assert 'Wall-clock timeout in inner loop' in caplog.text
        assert 'Skipping retry' in caplog.text
        assert r['converged'] is True and r['accepted_on'] == 'timeout'

    def test_wall_clock_without_an_accept(self):
        """The clock runs out before the first profile: not accepted (``main`` would retry)."""
        r = zs._solve(_cfg(wall_timeout=-1.0), load_material_dictionaries(), None, INPUT)
        assert r['converged'] is False and r['accepted_on'] is None
        assert r['stopped_on_wall_clock'] is True and r['density_from_eos'] is False
        assert not any(np.any(r[k]) for k in ('pressure', 'density', 'mass_enclosed'))
        assert r['mass_error_at_accept'] == pytest.approx(1.0)

    @pytest.mark.parametrize('layers', [{}, {**ICE, '_initial_radius_guess': 6.37e6}])
    def test_outer_wall_clock_restores_a_best_pass_that_is_not_the_last(
        self, monkeypatch, layers
    ):
        """Pass 2 lands farther from the target mass than pass 1; the clock runs out after it,
        and the outer check returns pass 1 with its own EOS density."""
        radius, real = _clock_after_lookup(monkeypatch, outer=2), zs.solve_structure

        def solve(*args, **kwargs):
            m, *rest = real(*args, **kwargs)
            away = 1.02 if m[-1] > earth_mass else 0.98
            return (m * (away if len(radius) == 3 else 1.0), *rest)

        monkeypatch.setattr(zs, 'solve_structure', solve)
        r = _picard(max_iterations_outer=4, max_iterations_inner=1, **layers)
        assert r['accepted_on'] == 'timeout'
        assert r['mass_error_at_accept'] == pytest.approx(r['best_mass_error'])

    def test_unaccepted_outer_wall_clock_stop_returns_the_best_pass(self, monkeypatch):
        """Each pass has a mass error of 6 to 9 %, above every accept limit, and the outer
        check stops the solve after the third pass: the best, second pass comes back
        unaccepted and seeds the retry."""
        radius, real = _clock_after_lookup(monkeypatch, outer=2), zs.solve_structure
        errors = (0.08, 0.06, 0.09)

        def solve(*args, **kwargs):
            m, *rest = real(*args, **kwargs)
            return (m * earth_mass * (1 + errors[min(len(radius), 3) - 1]) / m[-1], *rest)

        seeds, first, real_solve = [], [], zs._solve

        def spy(cp, *args, **kwargs):  # the retry only records its seed
            seeds.append(cp.get('_initial_radius_guess'))
            first.extend([] if first else [real_solve(cp, *args, **kwargs)])
            return first[0]

        monkeypatch.setattr(zs, 'solve_structure', solve)
        monkeypatch.setattr(zs, '_solve', spy)
        _run(_cfg(max_iterations_outer=4, max_iterations_inner=1, wall_timeout=WALL))
        r = first[0]
        assert len(radius) == 3
        assert r['converged'] is False and r['accepted_on'] is None
        assert r['stopped_on_wall_clock'] is True and r['density_from_eos'] is True
        assert r['radii'][-1] == pytest.approx(radius[1], rel=1e-12)
        assert r['mass_error_at_accept'] == pytest.approx(0.06, rel=1e-9)
        assert r['cmb_mass'] == pytest.approx(0.325 * 1.06 * earth_mass, rel=1e-9)
        assert seeds == [None, radius[1]]

    @pytest.mark.parametrize('outer', [2, 4])
    def test_cut_before_the_first_pressure_solve_of_the_last_pass(
        self, monkeypatch, caplog, outer
    ):
        """The clock runs out after the temperature profile of outer pass 2, before its first
        pressure solve (the last pass, or the outer check stops the next): the best pass comes
        back, with its core mass and gravity."""
        radius, clock = _count_outer_iterations(monkeypatch), time.time
        monkeypatch.setattr(
            zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * (len(radius) >= 2))
        )
        r = _picard(
            max_iterations_outer=outer,
            max_iterations_inner=1,
            _initial_radius_guess=6.15e6,
        )
        assert 'Wall-clock timeout in inner loop (outer=1, inner=0)' in caplog.text
        assert r['converged'] is True and r['accepted_on'] == 'timeout'
        assert r['mass_error_at_accept'] == pytest.approx(r['best_mass_error'])
        m, rho = np.asarray(r['mass_enclosed']), np.asarray(r['density'])
        assert np.argmax(m >= r['cmb_mass']) == np.argmax(np.diff(rho) < -0.2 * rho[:-1]) + 1

    def test_cut_before_the_first_pressure_solve_of_a_pass(self, monkeypatch, caplog):
        """The clock runs out after the temperature profile of the second outer pass, before
        its first pressure solve, and the solve is not accepted: the cut pass comes back, with
        its newest radius and the core mass of the target mass."""
        radius, clock = _count_outer_iterations(monkeypatch), time.time
        monkeypatch.setattr(
            zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * (len(radius) >= 2))
        )
        cfg = _cfg(max_iterations_outer=2, max_iterations_inner=1, wall_timeout=WALL)
        r = zs._solve(cfg, load_material_dictionaries(), None, INPUT)
        assert 'Wall-clock timeout in inner loop (outer=1, inner=0)' in caplog.text
        assert r['converged'] is False and r['density_from_eos'] is False
        assert r['radii'][-1] == pytest.approx(radius[1], rel=1e-12)
        assert r['cmb_mass'] == pytest.approx(0.325 * earth_mass)
        P, rho, m = (np.asarray(r[k]) for k in ('pressure', 'density', 'mass_enclosed'))
        assert np.flatnonzero(P > 0).tolist() == [0]  # no pressure solve in the cut pass
        assert not rho[1:].any() and not m.any()
        assert r['mass_error_at_accept'] == pytest.approx(1.0)

    def test_cut_with_passes_left_returns_the_best_pass(self, monkeypatch):
        """The same cut with passes left: the next outer check stops the solve and returns
        the best earlier pass, with its EOS density."""
        radius, clock = _count_outer_iterations(monkeypatch), time.time
        monkeypatch.setattr(
            zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * (len(radius) >= 2))
        )
        cfg = _cfg(max_iterations_outer=4, max_iterations_inner=1, wall_timeout=WALL)
        r = zs._solve(cfg, load_material_dictionaries(), None, INPUT)
        assert len(radius) == 2 and r['converged'] is False
        assert r['radii'][-1] == pytest.approx(radius[0], rel=1e-12)
        _assert_eos_density(r)

    @pytest.mark.parametrize('n_seed', [1, 50])
    def test_seeded_cut_keeps_a_density_only_where_the_pressure_is_set(
        self, monkeypatch, caplog, n_seed
    ):
        """A caller's density seed fills the first pass when more than one of its nodes is
        above zero; the clock runs out before its first pressure solve, and the density
        comes back zero wherever the pressure is zero."""
        caplog.set_level(logging.INFO, logger='zalmoxis.solver')
        radius, clock = _count_outer_iterations(monkeypatch), time.time
        monkeypatch.setattr(
            zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * (len(radius) >= 1))
        )
        seed = {
            'initial_density': np.linspace(1.3e4, 4e3, 50) * (np.arange(50) < n_seed),
            'initial_radii': np.linspace(0, 6.4e6, 50),
        }
        cfg = _cfg(max_iterations_inner=1, wall_timeout=WALL)
        r = zs._solve(cfg, load_material_dictionaries(), None, INPUT, **seed)
        assert 'Wall-clock timeout in inner loop (outer=0, inner=0)' in caplog.text
        assert ('Density seeded from previous solve' in caplog.text) is (n_seed > 1)
        assert r['converged'] is False and r['density_from_eos'] is False
        assert np.flatnonzero(r['pressure'] > 0).tolist() == [0]
        assert r['density'][0] == pytest.approx(1.3e4 if n_seed > 1 else 0.0)
        assert not r['density'][1:].any()

    def test_retry_after_an_unaccepted_cut_starts_from_the_newest_radius(self, monkeypatch):
        """The first attempt of the cut above is not accepted; the retry starts from the
        radius of the cut pass and converges."""
        radius, clock = _count_outer_iterations(monkeypatch), time.time
        monkeypatch.setattr(
            zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * (len(radius) >= 2))
        )
        seeds, real = [], zs._solve

        def solve(cp, *args, **kwargs):
            seeds.append(cp['_initial_radius_guess'])
            return real(cp, *args, **kwargs)

        monkeypatch.setattr(zs, '_solve', solve)
        r = _picard(
            max_iterations_outer=2,
            max_iterations_inner=1,
            _initial_radius_guess=0.8 * 6.43e6,
        )
        assert seeds == [0.8 * 6.43e6, radius[1]]
        assert r['converged'] is True and r['mass_error_at_accept'] < 1e-4

    def test_newton(self):
        r = _run(_cfg(outer_solver='newton', max_iterations_inner=8, wall_timeout=WALL))
        assert r['converged'] is True and r['accepted_on'] == 'newton_tol'
        assert r['mass_error_at_accept'] < 1e-4  # newton_tol
        _assert_eos_density(r)

    @pytest.mark.parametrize('inner', [3, 5])
    def test_newton_with_unconverged_density_flag(self, inner):
        """The inner caps at which the Picard density flag is False: the returned density is
        the EOS density, so the result is converged."""
        r = _run(_cfg(outer_solver='newton', max_iterations_inner=inner, wall_timeout=WALL))
        assert not r['converged_density']
        assert r['converged'] is True and r['accepted_on'] == 'newton_tol'
        _assert_eos_density(r)


@pytest.mark.smoke
@pytest.mark.wolfbower
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
        wall_timeout=WALL,
    )
    melt = get_solidus_liquidus_functions()
    r = zs.main(cfg, load_material_dictionaries(), melt, '/tmp')
    _assert_eos_density(r, cfg['layer_eos_config'], melt)  # T-dependent mantle, restored pass
    assert 'Wall-clock timeout (1000000s)' in caplog.text
    assert r['best_mass_error'] < 0.02  # below tolerance_outer: the chain alone says mass_only
    assert r['converged'] is True and r['accepted_on'] == 'timeout'


@pytest.mark.smoke
def test_cool_paleos_surface_keeps_the_eos_density():
    """At a 1500 K surface rho_EOS of PALEOS-2phase MgSiO3 halves in the outermost shells;
    the returned density keeps those values, which the structure integration used."""
    from tests._paleos_helpers import _paleos_data_available
    from zalmoxis.melting_curves import get_solidus_liquidus_functions

    if not _paleos_data_available():
        pytest.skip('PALEOS data files not found')
    eos = {'core': 'PALEOS:iron', 'mantle': 'PALEOS-2phase:MgSiO3'}
    cfg = _cfg(
        layer_eos_config=eos,
        temperature_mode='linear',
        surface_temperature=1500.0,
        center_temperature=5000.0,
        mushy_zone_factor=0.8,
        num_layers=100,
        wall_timeout=WALL,
    )
    melt = get_solidus_liquidus_functions()
    r = zs.main(cfg, load_material_dictionaries(), melt, INPUT)
    rho = np.asarray(r['density'])[np.asarray(r['pressure']) > 0]
    assert rho[-1] < 0.6 * rho[-4]  # the outer-shell density drop is kept
    _assert_eos_density(r, eos, melt, 0.8)
