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
from zalmoxis.constants import G, earth_mass, earth_radius
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


@pytest.fixture
def no_check(monkeypatch):
    """main skips the invariant check: the test builds a result or a profile the solver cannot
    make (a canned result, or masses scaled by a spy)."""
    monkeypatch.setattr(zs, 'check_accepted', lambda *args, **kwargs: None)


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


def _brentq(history, eval_M, best_R, **flags):
    """``best_R``: radius of the evaluation with the smallest mass error, or None for none;
    ``flags`` go to the inner result of that evaluation."""
    best = {'rel': np.inf, 'result': None}
    if best_R is not None:
        best = {'rel': abs(A * best_R + B - M_T) / M_T, 'result': _inner(best_R, **flags)}
    return zs._brentq_fallback_outer(
        6.0e6, M_T, eval_M, history, R_min=2e6, R_max=3e7, tol=1e-6, start_time=0.0, best=best
    )


def _hist(*radii):
    return [(R, A * R + B, abs(A * R + B - M_T) / M_T) for R in radii]


@pytest.mark.unit
class TestNewton:
    def test_newton_tol(self):
        r = _newton()
        assert r['converged'] is True and r['accepted_on'] == 'newton_tol'
        assert r['accept_mass_limit'] == 1e-6  # newton_tol
        assert r['mass_error_at_accept'] == pytest.approx(r['newton_history'][-1][2], abs=1e-15)

    def test_bracket_sweep_hit_is_brentq_best(self):
        """One Newton step lands on the root; the bracket sweep meets it there, and the profile
        is the one at the root."""
        r = _newton({'_initial_radius_guess': 6.9e6, 'newton_max_iter': 1})
        assert r['converged'] is True and r['accepted_on'] == 'brentq_best'
        assert r['radii'][-1] == pytest.approx(7.0e6, rel=1e-9)
        assert r['mass_error_at_accept'] == pytest.approx(r['best_mass_error'], abs=1e-15)
        assert r['best_mass_error'] < 1e-6

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
        r = _newton(wall=True)
        assert r['accepted_on'] == 'timeout' and r['accept_mass_limit'] == 1e-6
        assert _newton(wall=True, eos=False)['accepted_on'] is None


@pytest.mark.unit
class TestBrentq:
    def test_root_is_newton_tol(self):
        r = _brentq(_hist(6.0e6, 8.0e6), lambda R: (A * R + B, _inner(R)), 6.0e6)
        assert r['newton_used_brentq'] and r['converged'] is True
        assert r['accepted_on'] == 'newton_tol' and r['accept_mass_limit'] == 1e-6

    def test_root_does_not_read_inner_flags(self):
        def eval_M(R):
            return A * R + B, _inner(R, density=False)

        r = _brentq(_hist(6.0e6, 8.0e6), eval_M, 6.0e6)
        assert r['converged'] is True and r['accepted_on'] == 'newton_tol'
        assert r['converged_density'] is False

    def test_narrow_bracket_is_brentq_best(self):
        r = _brentq(_hist(7.0e6 - 0.2, 7.0e6 + 0.2), None, 7.0e6 - 0.2)
        assert r['converged'] is True and r['accepted_on'] == 'brentq_best'
        assert r['accept_mass_limit'] == 1e-6
        assert r['radii'][-1] == 7.0e6 - 0.2  # the profile of the best entry
        assert r['mass_error_at_accept'] == pytest.approx(r['best_mass_error'], abs=1e-15)
        assert r['best_mass_error'] < 1e-6

    def test_brentq_error_is_brentq_best(self):
        def boom(R):
            raise ValueError('stale bracket')

        r = _brentq(_hist(6.0e6, 7.0e6 + 1e-3), boom, 7.0e6 + 1e-3)
        assert r['converged'] is True and r['accepted_on'] == 'brentq_best'
        assert r['radii'][-1] == 7.0e6 + 1e-3
        assert r['mass_error_at_accept'] == pytest.approx(r['best_mass_error'], abs=1e-15)
        r = _brentq(_hist(6.0e6, 8.0e6), boom, 6.0e6)
        assert r['converged'] is False and r['accepted_on'] is None
        assert r['radii'][-1] == 6.0e6

    def test_root_on_a_wall_clock_stop_is_timeout(self):
        r = _brentq(_hist(6.0e6, 8.0e6), lambda R: (A * R + B, _inner(R, wall=True)), None)
        assert r['converged'] is True and r['accepted_on'] == 'timeout'

    def test_result_without_a_profile_has_the_keys(self):
        """No evaluation, so no profile: not accepted, with the per-profile keys."""
        r = _brentq(_hist(7.0e6 - 0.2, 7.0e6 + 0.2), None, None)
        assert r['converged'] is False and r['accepted_on'] is None
        assert r['mass_error_at_accept'] is None
        assert r['stopped_on_wall_clock'] is False and r['density_from_eos'] is False

    def test_no_bracket_is_not_converged(self):
        """No sign change in [R_min, R_max]: not converged, also with a history entry at the
        root."""
        below = lambda R: (0.5 * M_T, _inner(R))  # noqa: E731
        r = _brentq(_hist(7.0e6), below, 7.0e6)
        assert r['converged'] is False and r['accepted_on'] is None and r['converged_mass']
        assert r['radii'][-1] == 7.0e6 and r['mass_error_at_accept'] < 1e-15
        assert r['accept_mass_limit'] is None


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
@pytest.mark.usefixtures('no_check')
class TestRetry:
    def _main(self, first, planet_mass=earth_mass):
        cfg = {
            'planet_mass': planet_mass,
            'layer_eos_config': {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'},
        }
        retry = _inner(7.0e6, accepted_on='strict')
        with patch('zalmoxis.solver._solve', side_effect=[first, retry]) as self.solve:
            return zs.main(cfg, {}, None, '/tmp'), self.solve.call_count

    def test_retry_skip(self):
        r, n = self._main(_first_attempt(True, True))
        assert n == 1 and r['converged'] is True and r['accepted_on'] == 'retry_skip'
        assert r['accept_mass_limit'] == 0.07

    def test_retry_skip_on_a_wall_clock_stop_is_timeout(self):
        first = _first_attempt(True, True)
        first['stopped_on_wall_clock'] = True
        r, n = self._main(first)
        assert n == 1 and r['converged'] is True and r['accepted_on'] == 'timeout'
        assert r['accept_mass_limit'] == 0.07  # the retry-skip test it passed

    def test_retry_reports_its_own_path(self):
        r, n = self._main(_first_attempt(True, False))
        assert n == 2 and r['accepted_on'] == 'strict'

    @pytest.mark.parametrize(
        'err, pressure, density, skipped',
        [(0.0699, True, True, True), (0.07, True, True, False), (0.2, True, True, False)]
        + [(None, True, True, False), (0.05, False, True, False), (0.05, True, False, False)],
    )
    def test_retry_skip_limit_and_flags(self, err, pressure, density, skipped):
        """The retry is skipped only below 7 % mass error of the returned profile, with the
        density and the pressure converged."""
        first = _first_attempt(pressure, density)
        first['mass_error_at_accept'] = err
        r, n = self._main(first)
        assert n == (1 if skipped else 2)
        assert r['accepted_on'] == ('retry_skip' if skipped else 'strict')

    @pytest.mark.parametrize(
        'mass, f, seeded',
        [
            *(
                (1.0, f, s)
                for f, s in ((0.2, False), (0.201, True), (4.99, True), (5.0, False))
            ),
            *(
                (10.0, f, s)
                for f, s in ((0.199, False), (0.201, True), (4.99, True), (5.01, False))
            ),
            (1e-3, 4.99, True),  # inside the window only with the mass floored at 0.01 M_E
            (1e-3, 5.01, False),
        ],
    )
    def test_retry_radius_seed_window(self, mass, f, seeded):
        """The retry starts from the returned radius only strictly between 0.2 and 5 times
        R_E (max(M, 0.01 M_E)/M_E)^0.282."""
        R = f * (earth_radius * max(mass, 0.01) ** 0.282)
        first = _first_attempt(True, False)
        first['radii'] = np.array([0.0, R])
        self._main(first, mass * earth_mass)
        guess = self.solve.call_args_list[1].args[0].get('_initial_radius_guess')
        assert guess == pytest.approx(R, rel=1e-15) if seeded else guess is None

    @pytest.mark.parametrize('rho', [[1.0e4, 0.0], [0.0, 0.0]])
    def test_retry_density_seed(self, rho):
        """main hands the returned density and radii to the retry when a node of the density
        is above zero; ``_solve`` decides whether to use them."""
        first = _first_attempt(True, False)
        first['density'] = np.array(rho)
        self._main(first)
        kw = self.solve.call_args_list[1].kwargs
        seeded = any(rho)
        assert (kw['initial_density'] is first['density']) is seeded
        assert (kw['initial_radii'] is first['radii']) is seeded
        assert seeded or kw['initial_density'] is None and kw['initial_radii'] is None


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
    ``core_mantle_mass`` are 0.325 and 0.325 + ``mantle_mass_fraction`` of the planet mass.
    When ``mass_enclosed`` is the integral of the returned profile (``integrated``), at the
    last node with P > 0 the gravity is G M / r^2 and the trapezoid mass of the returned
    density matches ``mass_enclosed``."""
    r = _run(_cfg(**{'wall_timeout': WALL, **kwargs}))
    m, rad = np.asarray(r['mass_enclosed']), np.asarray(r['radii'])
    assert r['mass_error_at_accept'] == pytest.approx(abs(m[-1] - earth_mass) / earth_mass)
    _assert_eos_density(r, kwargs.get('layer_eos_config'))
    mmf = kwargs.get('mantle_mass_fraction', 0)
    for key, f in (('cmb_mass', 0.325), ('core_mantle_mass', 0.325 + mmf)):
        assert r[key] == pytest.approx(f * earth_mass, rel=1e-12), key
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
@pytest.mark.accept_solves
class TestRealSolve:
    def test_strict(self):
        r = _picard(tolerance_outer=4e-3)  # inner cap 50: the density converges
        assert r['converged_density'] and r['converged_pressure']
        assert r['accepted_on'] == 'strict' and r['stopped_on_wall_clock'] is False
        assert r['accept_mass_limit'] == 4e-3

    def test_mass_only(self):
        r = _picard(max_iterations_inner=1, tolerance_outer=4e-3)
        assert not r['converged_density'] and r['accepted_on'] == 'mass_only'
        assert r['accept_mass_limit'] == 4e-3

    def test_iteration_cap_is_max_iter(self):  # the last node is padded (P = 0) on macOS
        r = _picard(max_iterations_outer=3, max_iterations_inner=1)
        assert r['converged'] is True and r['accepted_on'] == 'max_iter'
        assert r['mass_error_at_accept'] > 1e-3  # above tolerance_outer
        assert r['accept_mass_limit'] == 0.03

    def test_iteration_cap_of_one_pass_is_max_iter(self):
        """``_solve`` itself: through ``main`` a first attempt that is not accepted would
        reach the retry, with the outer cap doubled. The guess lies 0.4 % below the root
        (6.114e6 m): one pass ends at about 1 % mass error, under the 3 % limit."""
        cfg = _cfg(max_iterations_outer=1, max_iterations_inner=1, _initial_radius_guess=6.09e6)
        r = zs._solve({**cfg, 'wall_timeout': WALL}, load_material_dictionaries(), None, INPUT)
        assert r['converged'] is True and r['accepted_on'] == 'max_iter'
        assert 1e-3 < r['mass_error_at_accept'] < 0.03  # above tolerance_outer

    @pytest.mark.usefixtures('no_check')
    def test_max_iter_returns_the_best_pass(self, monkeypatch):
        """The last outer iteration lands farther from the target mass than the one before;
        max_iter returns the best pass, whose error is ``mass_error_at_accept``."""
        radius, real = _count_outer_iterations(monkeypatch), zs.solve_structure

        def solve(*args, **kwargs):
            m, *rest = real(*args, **kwargs)
            return (m * (1.02 if len(radius) == 3 else 1.0), *rest)

        monkeypatch.setattr(zs, 'solve_structure', solve)
        r = _picard(False, max_iterations_outer=3, max_iterations_inner=1)  # mass scaled
        assert r['accepted_on'] == 'max_iter'
        assert r['mass_error_at_accept'] == pytest.approx(r['best_mass_error'], rel=1e-12)
        assert r['radii'][-1] == pytest.approx(radius[1], rel=1e-12)

    @pytest.mark.parametrize(
        'inner, count, label',
        [(2, 1, 'mass_only'), (50, 20, 'strict')],
        ids=['mass_only', 'strict'],
    )
    def test_inner_wall_clock_is_timeout(self, monkeypatch, caplog, inner, count, label):
        """The clock runs out in the inner loop of the last outer iteration, after ``count``
        density lookups; without the clock the same solve is accepted on ``label``."""
        cfg = _cfg(max_iterations_inner=inner, tolerance_outer=1e-3, wall_timeout=WALL)
        radius = _count_outer_iterations(monkeypatch)
        assert _run(cfg)['accepted_on'] == label
        monkeypatch.undo()
        _clock_after_lookup(monkeypatch, outer=len(radius) - 1, count=count)
        r = _run(cfg)
        assert 'Wall-clock timeout in inner loop' in caplog.text
        assert 'Wall-clock timeout (1000000s)' not in caplog.text
        assert bool(r['converged_density']) == (label == 'strict')
        assert r['converged'] is True and r['accepted_on'] == 'timeout'
        assert r['accept_mass_limit'] == 1e-3  # tolerance_outer, also where it replaces strict

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
        assert r['accept_mass_limit'] == 0.07

    def test_wall_clock_without_an_accept(self):
        """The clock runs out before the first profile: not accepted (``main`` would retry)."""
        r = zs._solve(_cfg(wall_timeout=-1.0), load_material_dictionaries(), None, INPUT)
        assert r['converged'] is False and r['accepted_on'] is None
        assert r['stopped_on_wall_clock'] is True and r['density_from_eos'] is False
        assert not any(np.any(r[k]) for k in ('pressure', 'density', 'mass_enclosed'))
        assert r['mass_error_at_accept'] == pytest.approx(1.0)

    @pytest.mark.usefixtures('no_check')
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
        assert r['cmb_mass'] == pytest.approx(0.325 * earth_mass, rel=1e-12)
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
        assert r['converged_pressure'] and not r['converged_density']  # the best pass, 1 inner
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

    @pytest.mark.usefixtures('no_check')
    def test_restored_pass_keeps_its_convergence_flags(self, monkeypatch):
        """Passes at 8, 6 and 9 % mass error, each with a converged inner loop; the clock cuts
        the third, and the outer check returns the second with its own density and pressure
        flags, so main skips the retry."""
        radius, real = _clock_after_lookup(monkeypatch, outer=2), zs.solve_structure
        errors = (0.08, 0.06, 0.09)

        def solve(*args, **kwargs):
            m, *rest = real(*args, **kwargs)
            return (m * earth_mass * (1 + errors[min(len(radius), 3) - 1]) / m[-1], *rest)

        calls, real_solve = [], zs._solve
        monkeypatch.setattr(zs, 'solve_structure', solve)
        monkeypatch.setattr(
            zs, '_solve', lambda *a, **k: calls.append(1) or real_solve(*a, **k)
        )
        r = _run(_cfg(max_iterations_outer=4, max_iterations_inner=50, wall_timeout=WALL))
        assert len(radius) == 3 and len(calls) == 1
        assert r['converged_density'] and r['converged_pressure']
        assert r['radii'][-1] == pytest.approx(radius[1], rel=1e-12)
        assert r['accepted_on'] == 'timeout'  # the retry skip of a wall-clock stop
        assert r['mass_error_at_accept'] == pytest.approx(0.06, rel=1e-9)

    def test_restored_pass_keeps_which_flag_converged(self, monkeypatch):
        """As above with one inner iteration per pass: the restored pass comes back with its
        pressure converged and its density not."""
        radius, real = _clock_after_lookup(monkeypatch, outer=2), zs.solve_structure
        errors = (0.08, 0.06, 0.09)

        def solve(*args, **kwargs):
            m, *rest = real(*args, **kwargs)
            return (m * earth_mass * (1 + errors[min(len(radius), 3) - 1]) / m[-1], *rest)

        monkeypatch.setattr(zs, 'solve_structure', solve)
        cfg = _cfg(max_iterations_outer=4, max_iterations_inner=1, wall_timeout=WALL)
        r = zs._solve(cfg, load_material_dictionaries(), None, INPUT)
        assert len(radius) == 3 and r['stopped_on_wall_clock'] is True
        assert r['radii'][-1] == pytest.approx(radius[1], rel=1e-12)
        assert r['converged_pressure'] and not r['converged_density']

    def test_an_outer_stop_accepts_on_the_flags_of_the_best_pass(self, monkeypatch):
        """Passes at 8, 4 and 9 % mass error with converged inner loops, tolerance_outer 0.02
        and the clock in pass 3: the outer stop restores the 4 % pass before the accept tests,
        which accept it on its own flags, as a timeout at 3 times tolerance_outer."""
        radius, real = _clock_after_lookup(monkeypatch, outer=2), zs.solve_structure
        errors = (0.08, 0.04, 0.09)

        def solve(*args, **kwargs):
            m, *rest = real(*args, **kwargs)
            return (m * earth_mass * (1 + errors[min(len(radius), 3) - 1]) / m[-1], *rest)

        monkeypatch.setattr(zs, 'solve_structure', solve)
        cfg = _cfg(max_iterations_outer=4, max_iterations_inner=50, tolerance_outer=0.02)
        r = zs._solve({**cfg, 'wall_timeout': WALL}, load_material_dictionaries(), None, INPUT)
        assert len(radius) == 3 and r['radii'][-1] == pytest.approx(radius[1], rel=1e-12)
        assert r['converged_density'] and r['converged_pressure']
        assert r['converged'] is True and r['accepted_on'] == 'timeout'
        assert r['accept_mass_limit'] == pytest.approx(0.06, rel=1e-12)
        assert r['mass_error_at_accept'] == pytest.approx(0.04, rel=1e-9)

    def test_restored_pass_keeps_an_unconverged_pressure(self, monkeypatch):
        """As above with a pressure tolerance of 1e-3 Pa, which no pass meets: the restored pass
        comes back with its pressure not converged, so main retries."""
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
        cfg = _cfg(max_iterations_outer=4, max_iterations_inner=50, pressure_tolerance=1e-3)
        _run({**cfg, 'wall_timeout': WALL})
        r = first[0]
        assert r['radii'][-1] == pytest.approx(radius[1], rel=1e-12)
        assert r['converged_density'] and not r['converged_pressure']
        assert len(seeds) == 2

    def test_retry_skip_reads_the_returned_profile(self, monkeypatch):
        """Passes at 5, 4 and 10 % mass error with converged inner loops, the cap at 3: the
        last pass comes back at 10 %, above the 7 % of the retry skip, so main retries."""
        radius, real = _count_outer_iterations(monkeypatch), zs.solve_structure
        errors = (0.05, 0.04, 0.10)

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
        _run(_cfg(max_iterations_outer=3, max_iterations_inner=50, wall_timeout=WALL))
        r = first[0]
        assert r['converged_density'] and r['converged_pressure']
        assert r['best_mass_error'] == pytest.approx(0.04, rel=1e-9)
        assert r['mass_error_at_accept'] == pytest.approx(0.10, rel=1e-9)
        assert len(seeds) == 2

    def test_a_pass_without_a_pressure_solve_is_never_the_best_pass(self, monkeypatch):
        """A start at 1.2e7 m: the first pass ends more than twice above the target mass, and
        the clock cuts the second before its first pressure solve; the outer check returns
        the first pass, with its EOS density, not the cut pass."""
        radius, clock = _count_outer_iterations(monkeypatch), time.time
        monkeypatch.setattr(
            zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * (len(radius) >= 2))
        )
        cfg = _cfg(max_iterations_outer=4, max_iterations_inner=1, _initial_radius_guess=1.2e7)
        r = zs._solve({**cfg, 'wall_timeout': WALL}, load_material_dictionaries(), None, INPUT)
        assert len(radius) == 2 and r['converged'] is False
        assert r['radii'][-1] == pytest.approx(1.2e7, rel=1e-12)
        assert r['best_mass_error'] > 1 and r['density_from_eos'] is True

    def test_cut_with_passes_left_returns_the_best_pass(self, monkeypatch):
        """The clock cuts the third pass before its first pressure solve, with passes left:
        the next outer check returns the best pass, the first (the second ends 50 % above
        the target mass), with its radius, enclosed mass and EOS density."""
        radius, clock = _count_outer_iterations(monkeypatch), time.time
        monkeypatch.setattr(
            zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * (len(radius) >= 3))
        )
        real, mass = zs.solve_structure, {}

        def solve(*args, **kwargs):
            m, *rest = real(*args, **kwargs)
            m = m * (1.5 * earth_mass / m[-1] if len(radius) == 2 else 1.0)
            mass[len(radius)] = m[-1]
            return (m, *rest)

        monkeypatch.setattr(zs, 'solve_structure', solve)
        cfg = _cfg(max_iterations_outer=5, max_iterations_inner=1, wall_timeout=WALL)
        r = zs._solve(cfg, load_material_dictionaries(), None, INPUT)
        assert len(radius) == 3 and r['converged'] is False
        assert abs(mass[1] / earth_mass - 1) < 0.5  # the first pass is the best
        assert r['radii'][-1] == pytest.approx(radius[0], rel=1e-12)
        assert r['mass_enclosed'][-1] == pytest.approx(mass[1], rel=1e-12)
        assert r['cmb_mass'] == pytest.approx(0.325 * earth_mass, rel=1e-12)
        _assert_eos_density(r)

    @pytest.mark.parametrize('n_seed', [1, 2, 50])
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

    def test_newton_cap_with_the_root_in_the_bracket_sweep(self):
        """newton_max_iter = 1: the first evaluation of the bracket sweep lands within
        newton_tol, and the result is the profile at that radius, not at the first step."""
        cfg = _cfg(outer_solver='newton', newton_max_iter=1, newton_tol=2e-2, wall_timeout=WALL)
        r = _run({**cfg, '_initial_radius_guess': 1.03 * 6.3e6, 'max_iterations_inner': 20})
        assert r['converged'] is True and r['accepted_on'] == 'brentq_best'
        best = min(r['newton_history'], key=lambda h: h[2])
        assert r['radii'][-1] == pytest.approx(best[0], rel=1e-12)
        assert r['mass_error_at_accept'] == pytest.approx(best[2], rel=1e-9)
        assert best[2] < 2e-2  # newton_tol

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
    assert r['accept_mass_limit'] == 0.02


@pytest.mark.smoke
@pytest.mark.wolfbower
def test_wall_clock_in_the_adiabat_ramp_keeps_tolerance_outer_as_the_limit(monkeypatch, caplog):
    """As above with tolerance_outer 0.05 and a 3.9 % mass error: accepted below
    tolerance_outer, so the invariant check reads that limit and does not raise."""
    from zalmoxis.eos import get_solidus_liquidus_functions

    _need_wb_tables()
    caplog.set_level(logging.INFO, logger='zalmoxis.solver')
    ramp = lambda: any('activating adiabat blend' in m for m in caplog.messages)  # noqa: E731
    clock = time.time
    monkeypatch.setattr(zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * ramp()))
    cfg = _cfg(
        temperature_mode='adiabatic',
        layer_eos_config={'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'},
        tolerance_outer=0.05,
        max_iterations_inner=2,
        wall_timeout=WALL,
        _initial_radius_guess=5.9e6,
    )
    r = zs.main(cfg, load_material_dictionaries(), get_solidus_liquidus_functions(), '/tmp')
    assert r['converged'] is True and r['accepted_on'] == 'timeout'
    assert 0.03 < r['mass_error_at_accept'] < 0.05
    assert r['accept_mass_limit'] == 0.05


@pytest.mark.smoke
@pytest.mark.wolfbower
@pytest.mark.usefixtures('no_check')
def test_a_restore_in_the_blend_ramp_stays_on_the_newest_temperature(monkeypatch):
    """The passes after the adiabat blend starts end 5 % off the target mass, and the clock
    runs out after the first density lookup of the first blended pass: the best pass restarts
    at each blend step, so the restored pass is that blended pass, not the linear pass that
    ended closer to the mass."""
    from zalmoxis.eos import get_solidus_liquidus_functions

    _need_wb_tables()
    radius, adiabat, lookups = _count_outer_iterations(monkeypatch), [], []
    real_adi, real, clock = zs.compute_adiabatic_temperature, zs.solve_structure, time.time
    real_lookup = zs.calculate_mixed_density_batch

    def adi(*args, **kwargs):
        adiabat.append(len(radius))
        return real_adi(*args, **kwargs)

    def solve(*args, **kwargs):
        m, g, P = real(*args, **kwargs)
        f = 1.05 * earth_mass / m[-1] if adiabat else 1.0
        return m * f, g * f, P

    def lookup(*args, **kwargs):
        lookups.extend([1] if adiabat else [])
        return real_lookup(*args, **kwargs)

    monkeypatch.setattr(zs, 'calculate_mixed_density_batch', lookup)
    monkeypatch.setattr(zs, 'compute_adiabatic_temperature', adi)
    monkeypatch.setattr(zs, 'solve_structure', solve)
    monkeypatch.setattr(zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * bool(lookups)))
    cfg = _cfg(
        temperature_mode='adiabatic',
        layer_eos_config={'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'},
        tolerance_outer=0.02,
        max_iterations_inner=2,
        wall_timeout=WALL,
    )
    r = zs._solve(cfg, load_material_dictionaries(), get_solidus_liquidus_functions(), '/tmp')
    assert adiabat and r['stopped_on_wall_clock'] is True
    assert r['adiabat_computed'] is True
    assert r['mass_error_at_accept'] == pytest.approx(0.05, rel=1e-9)


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(900)
def test_a_max_iter_stop_in_the_blend_ramp_returns_a_pass_on_the_adiabat(caplog):
    """PALEOS, adiabatic, 4 outer passes: the mass converges on the linear temperature, and
    the cap stops the solve in the blend ramp. The returned pass is on the newest blend (0.25),
    not the linear pass with the smallest mass error, and main warns about the blend."""
    from tests.test_spider_coupling_convergence import _has_paleos_data

    if not _has_paleos_data():
        pytest.skip('PALEOS tables not on disk')
    cfg = _cfg(
        temperature_mode='adiabatic',
        layer_eos_config={'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
        num_layers=150,
        relative_tolerance=1e-5,
        absolute_tolerance=1e-6,
        max_iterations_outer=4,
        wall_timeout=WALL,
    )
    r = zs.main(cfg, load_material_dictionaries(), None, INPUT)
    T = np.asarray(r['temperature'])
    off_linear = np.max(np.abs(T - np.linspace(6000.0, 3000.0, len(T))))
    assert r['converged'] is True and r['accepted_on'] == 'max_iter'
    assert r['adiabat_computed'] is True and off_linear > 100.0
    assert r['adiabat_blend'] == 0.25
    assert 'accepted a temperature profile at adiabat blend 0.25' in caplog.text


@pytest.mark.smoke
@pytest.mark.accept_solves
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


@pytest.mark.unit
@pytest.mark.parametrize('flag', ['pressure', 'eos'])
class TestNewtonExitsReadTheInnerFlags:
    """Every Newton exit accepts only a profile whose inner solve reports converged pressure
    and the EOS density, as the Newton converge path does."""

    def test_bracket_sweep_hit(self, flag):
        r = _newton({'_initial_radius_guess': 6.9e6, 'newton_max_iter': 1}, **{flag: False})
        assert r['newton_used_brentq'] and r['best_mass_error'] < 1e-6
        assert r['converged'] is False and r['accepted_on'] is None

    def test_root(self, flag):
        r = _brentq(
            _hist(6.0e6, 8.0e6), lambda R: (A * R + B, _inner(R, **{flag: False})), None
        )
        assert r['converged'] is False and r['accepted_on'] is None

    def test_narrow_bracket(self, flag):
        r = _brentq(_hist(7.0e6 - 0.2, 7.0e6 + 0.2), None, 7.0e6 - 0.2, **{flag: False})
        assert r['best_mass_error'] < 1e-6
        assert r['converged'] is False and r['accepted_on'] is None

    def test_brentq_error(self, flag):
        def boom(R):
            raise ValueError('stale bracket')

        r = _brentq(_hist(6.0e6, 7.0e6 + 1e-3), boom, 7.0e6 + 1e-3, **{flag: False})
        assert r['converged'] is False and r['accepted_on'] is None


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(900)
def test_a_clock_stop_before_the_first_blended_pressure_solve_keeps_the_linear_pass(
    monkeypatch,
):
    """PALEOS, adiabatic: the clock runs out when the first blended pass computes the adiabat,
    before its first pressure solve. The linear pass that met the mass test comes back, with
    its EOS density and adiabat_blend 0, not an empty profile."""
    from tests.test_spider_coupling_convergence import _has_paleos_data

    if not _has_paleos_data():
        pytest.skip('PALEOS tables not on disk')
    adiabat, real_adi, clock = [], zs.compute_adiabatic_temperature, time.time

    def adi(*args, **kwargs):
        adiabat.append(1)
        return real_adi(*args, **kwargs)

    monkeypatch.setattr(zs, 'compute_adiabatic_temperature', adi)
    monkeypatch.setattr(zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * bool(adiabat)))
    cfg = _cfg(
        temperature_mode='adiabatic',
        layer_eos_config={'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
        num_layers=150,
        relative_tolerance=1e-5,
        absolute_tolerance=1e-6,
        wall_timeout=WALL,
    )
    r = zs._solve(cfg, load_material_dictionaries(), None, INPUT)
    assert adiabat and r['stopped_on_wall_clock'] is True
    assert r['converged'] is True and r['density_from_eos'] is True
    assert r['adiabat_blend'] == 0.0 and r['mass_error_at_accept'] < 1e-3


@pytest.mark.unit
@pytest.mark.usefixtures('no_check')
@pytest.mark.parametrize(
    'eos, computed, blend, warned',
    [
        ('PALEOS', False, 0.0, True),
        ('PALEOS', True, 0.5, True),
        ('PALEOS', True, 1.0, False),
        ('Analytic', False, 0.0, False),  # no T-dependent component: no ramp by design
    ],
)
def test_main_warns_on_an_accepted_profile_short_of_the_adiabat(
    monkeypatch, caplog, eos, computed, blend, warned
):
    first = {**_inner(7.0e6, accepted_on='strict'), 'adiabat_computed': computed}
    first['adiabat_blend'] = blend
    monkeypatch.setattr(zs, '_solve', lambda *a, **k: dict(first))
    monkeypatch.setattr(zs, '_report_table_bands', lambda r, *a: r)
    cfg = {
        'planet_mass': M_T,
        'temperature_mode': 'adiabatic',
        'layer_eos_config': {'core': f'{eos}:iron', 'mantle': f'{eos}:MgSiO3'},
    }
    zs.main(cfg, {}, None, '/tmp')
    assert ('accepted a temperature profile at adiabat blend' in caplog.text) is warned
