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
        assert r.get('accept_mass_limit') == 0.07

    def test_retry_skip_on_a_wall_clock_stop_is_timeout(self):
        first = _first_attempt(True, True)
        first['stopped_on_wall_clock'] = True
        r, n = self._main(first)
        assert n == 1 and r['converged'] is True and r['accepted_on'] == 'timeout'
        assert r.get('accept_mass_limit') == 0.07  # the retry-skip test it passed

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


@pytest.mark.unit
@pytest.mark.timeout(60)
@pytest.mark.usefixtures('no_check')
def test_outer_wall_clock_above_tolerance_outer_accepts_on_the_loose_limit(monkeypatch):
    """The same stop, with the mass error above tolerance_outer, accepts on the 3 % limit."""
    _clock_after_lookup(monkeypatch, outer=1)
    r = _picard(max_iterations_outer=3, max_iterations_inner=1)
    assert r['mass_error_at_accept'] > 1e-3  # above tolerance_outer
    assert r['accepted_on'] == 'timeout' and r['accept_mass_limit'] == 0.03


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
        caplog.set_level(logging.INFO, logger='fwl.zalmoxis.solver')
        _clock_after_lookup(monkeypatch, count=8)
        r = _picard(
            max_iterations_outer=1,
            max_iterations_inner=30,
            _initial_radius_guess=0.97 * 6432000.0,
        )
        assert 'Wall-clock timeout in inner loop' in caplog.text
        assert 'Skipping retry' in caplog.text
        assert r['converged'] is True and r['accepted_on'] == 'timeout'
        assert r.get('accept_mass_limit') == 0.07

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
        caplog.set_level(logging.INFO, logger='fwl.zalmoxis.solver')
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
    caplog.set_level(logging.INFO, logger='fwl.zalmoxis.solver')
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
    caplog.set_level(logging.INFO, logger='fwl.zalmoxis.solver')
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


def _wb_blended(
    monkeypatch, factor, tol, clock_stop=True, inner=2, outer=100, clock_after=None, linear=None
):
    """A Wolf & Bower adiabatic _solve whose passes after the adiabat blend starts end ``factor``
    times the target mass, and the passes before it ``linear`` times (unscaled when None); with
    ``clock_stop`` the clock runs out after the first density lookup of the first blended pass,
    or with ``clock_after`` once the adiabat was computed that many times."""
    from zalmoxis.eos import get_solidus_liquidus_functions

    _need_wb_tables()
    adiabat, lookups = [], []
    real_adi, real, clock = zs.compute_adiabatic_temperature, zs.solve_structure, time.time
    real_lookup = zs.calculate_mixed_density_batch

    def adi(*args, **kwargs):
        adiabat.append(1)
        return real_adi(*args, **kwargs)

    def solve(*args, **kwargs):
        m, g, P = real(*args, **kwargs)
        scale = factor if adiabat else linear
        f = scale * earth_mass / m[-1] if scale else 1.0
        return m * f, g * f, P

    def lookup(*args, **kwargs):
        lookups.extend([1] if adiabat else [])
        return real_lookup(*args, **kwargs)

    monkeypatch.setattr(zs, 'calculate_mixed_density_batch', lookup)
    monkeypatch.setattr(zs, 'compute_adiabatic_temperature', adi)
    monkeypatch.setattr(zs, 'solve_structure', solve)

    def stopped():
        return bool(lookups) if clock_after is None else len(adiabat) >= clock_after

    if clock_stop:
        monkeypatch.setattr(zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * stopped()))
    cfg = _cfg(
        temperature_mode='adiabatic',
        layer_eos_config={'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'},
        tolerance_outer=tol,
        max_iterations_inner=inner,
        max_iterations_outer=outer,
        wall_timeout=WALL,
    )
    r = zs._solve(cfg, load_material_dictionaries(), get_solidus_liquidus_functions(), '/tmp')
    assert adiabat
    return r


@pytest.mark.smoke
@pytest.mark.wolfbower
@pytest.mark.usefixtures('no_check')
def test_a_clock_stop_keeps_a_blended_pass_that_its_mass_test_accepts(monkeypatch, caplog):
    """tolerance_outer 0.05 and blended passes 4 % off: the mass test of the clock stop accepts
    the newest pass above the 3 % ceiling, so it is returned, not the linear one."""
    with caplog.at_level(logging.WARNING, logger='fwl.zalmoxis.solver'):
        r = _wb_blended(monkeypatch, 1.04, 0.05)
    assert r['stopped_on_wall_clock'] is True and r['accepted_on'] == 'timeout'
    assert r['adiabat_blend'] == 0.25
    assert r['mass_error_at_accept'] == pytest.approx(0.04, rel=1e-9)
    assert 'No pass of adiabat blend' not in caplog.text


@pytest.mark.smoke
@pytest.mark.wolfbower
@pytest.mark.usefixtures('no_check')
def test_a_clock_stop_beyond_its_limit_returns_the_linear_pass_under_that_limit(monkeypatch):
    """The blended passes end 10 % off, beyond the 6 % limit of a clock stop: the linear pass
    that started the ramp is returned, and its mass verdict is taken again, so the mass test of
    tolerance_outer accepts it (its inner flags are not both converged), not the 3 % ceiling."""
    r = _wb_blended(monkeypatch, 1.10, 0.02)
    assert r['stopped_on_wall_clock'] is True and r['accepted_on'] == 'timeout'
    assert r['adiabat_blend'] == 0.0
    assert r['accept_mass_limit'] == pytest.approx(0.02)
    assert r['mass_error_at_accept'] < 0.02


@pytest.mark.smoke
@pytest.mark.wolfbower
@pytest.mark.usefixtures('no_check')
def test_a_clock_stop_keeps_a_blended_pass_whose_inner_solves_converged(monkeypatch):
    """Blended passes 5 % off, inside the 6 % limit of a clock stop but above tolerance_outer
    0.02 and the 3 % ceiling: the first blended pass ran its inner solves to convergence, so
    the strict test of the stop accepts it and no earlier pass replaces it."""
    r = _wb_blended(monkeypatch, 1.05, 0.02, inner=100, clock_after=2)
    assert r['accepted_on'] == 'timeout' and r['accept_mass_limit'] == pytest.approx(0.06)
    assert r['converged_density'] and r['converged_pressure']
    assert r['adiabat_blend'] == 0.25
    assert r['mass_error_at_accept'] == pytest.approx(0.05, rel=1e-9)


@pytest.mark.smoke
@pytest.mark.wolfbower
@pytest.mark.usefixtures('no_check')
def test_a_clock_stop_inside_its_limit_without_converged_inner_solves_falls_back(
    monkeypatch, caplog
):
    """Blended passes 5 % off, inside the 6 % limit of a clock stop, with inner solves cut at 2
    passes: no accept test takes the newest pass, so the linear pass is returned; the warning
    names both blends and best_mass_error describes the returned pass."""
    with caplog.at_level(logging.WARNING, logger='fwl.zalmoxis.solver'):
        r = _wb_blended(monkeypatch, 1.05, 0.02)
    assert r['stopped_on_wall_clock'] is True and r['accepted_on'] == 'timeout'
    assert r['adiabat_blend'] == 0.0
    assert r['mass_error_at_accept'] < 0.02
    assert r['best_mass_error'] == pytest.approx(r['mass_error_at_accept'], rel=1e-9)
    assert 'No pass of adiabat blend 0.25' in caplog.text
    assert (
        f'best pass of blend 0.00 (mass error {r["best_mass_error"] * 100:.4f}%)' in caplog.text
    )


@pytest.mark.smoke
@pytest.mark.wolfbower
@pytest.mark.usefixtures('no_check')
@pytest.mark.parametrize(
    ('linear', 'factor', 'outer', 'clock_after'),
    [(1.04, 1.10, 8, None), (1.04, 1.035, 2, None), (1.045, 1.035, 3, 2)],
    ids=['cap-earlier-above-ceiling', 'cap-earlier-worse', 'inner-clock-earlier-worse'],
)
def test_a_stop_that_cannot_return_a_better_earlier_pass_keeps_its_own(
    monkeypatch, caplog, linear, factor, outer, clock_after
):
    """tolerance_outer 0.05 above the 3 % ceiling: an iteration cap whose earlier pass is not
    under 3 %, or whose earlier pass is worse, and an inner clock stop on the last pass whose
    earlier pass is worse, keep the newest best pass; no warning names an earlier one."""
    with caplog.at_level(logging.WARNING, logger='fwl.zalmoxis.solver'):
        r = _wb_blended(
            monkeypatch, factor, 0.05, clock_stop=clock_after is not None, outer=outer,
            clock_after=clock_after, linear=linear,
        )  # fmt: skip
    assert 'No pass of adiabat blend' not in caplog.text
    assert r['best_mass_error'] == pytest.approx(factor - 1, rel=1e-9)


@pytest.mark.smoke
@pytest.mark.wolfbower
@pytest.mark.usefixtures('no_check')
def test_an_inner_clock_stop_on_the_last_pass_returns_a_better_earlier_pass(monkeypatch):
    """tolerance_outer 0.05: the clock runs out in the inner loop of the last outer pass, the
    blended pass is 4.5 % off and the linear pass 4 %: the linear pass is returned and the mass
    test of the stop accepts it, although it is above the 3 % ceiling."""
    r = _wb_blended(monkeypatch, 1.045, 0.05, outer=3, clock_after=2, linear=1.04)
    assert r['accepted_on'] == 'timeout' and r['adiabat_blend'] == 0.0
    assert r['mass_error_at_accept'] == pytest.approx(0.04, rel=1e-9)


@pytest.mark.smoke
@pytest.mark.wolfbower
@pytest.mark.usefixtures('no_check')
def test_a_clock_stop_swap_returns_the_pass_its_mass_test_read(monkeypatch):
    """tolerance_outer 0.05, the clock runs out in the first blended pass, 6 % off, and the
    linear pass is 4.5 % off: the swap returns the linear pass, and the accepted mass error is
    the one of the returned pass, not of the blended pass."""
    r = _wb_blended(monkeypatch, 1.06, 0.05, inner=5, outer=2, linear=1.045)
    assert r['accepted_on'] == 'timeout' and r['adiabat_blend'] == 0.0
    assert r['mass_error_at_accept'] == pytest.approx(r['best_mass_error'], rel=1e-9)
    assert r['mass_error_at_accept'] == pytest.approx(0.045, rel=1e-9)


@pytest.mark.smoke
@pytest.mark.wolfbower
@pytest.mark.usefixtures('no_check')
def test_a_clock_stop_with_converged_density_but_not_pressure_falls_back(monkeypatch):
    """Blended passes 5 % off, inside the 6 % limit of a clock stop but above tolerance_outer
    0.02 and the 3 % ceiling; their inner density converges but their pressure solve reports
    no root, so the strict test cannot take the newest pass and the linear pass is returned."""
    blended, real_adi, real_brentq = [], zs.compute_adiabatic_temperature, zs.brentq

    def adi(*args, **kwargs):
        blended.append(1)
        return real_adi(*args, **kwargs)

    def brent(*args, **kwargs):
        x, info = real_brentq(*args, **kwargs)
        cut = SimpleNamespace(converged=False, function_calls=info.function_calls)
        return x, cut if blended else info

    monkeypatch.setattr(zs, 'compute_adiabatic_temperature', adi)
    monkeypatch.setattr(zs, 'brentq', brent)
    r = _wb_blended(monkeypatch, 1.05, 0.02, inner=100, clock_after=2)
    assert r['accepted_on'] == 'timeout' and r['adiabat_blend'] == 0.0
    assert r['mass_error_at_accept'] < 0.02


@pytest.mark.smoke
@pytest.mark.wolfbower
@pytest.mark.usefixtures('no_check')
def test_an_accepted_mass_break_returns_its_own_pass_without_a_fallback(monkeypatch, caplog):
    """tolerance_outer 0.05 and blended passes 4 % off: the mass test breaks at blend 1.0 and
    accepts that pass, so no earlier pass replaces it and best_mass_error describes it."""
    with caplog.at_level(logging.WARNING, logger='fwl.zalmoxis.solver'):
        r = _wb_blended(monkeypatch, 1.04, 0.05, clock_stop=False, outer=50)
    assert r['accepted_on'] == 'mass_only' and r['adiabat_blend'] == 1.0
    assert r['best_mass_error'] == pytest.approx(r['mass_error_at_accept'], rel=1e-9)
    assert r['mass_error_at_accept'] == pytest.approx(0.04, rel=1e-9)
    assert 'No pass of adiabat blend' not in caplog.text


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
        ('PALEOS', False, 1.0, True),  # a blend without a computed adiabat counts as 0
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


@pytest.mark.unit
@pytest.mark.parametrize('converged', [True, False])
def test_only_an_accepted_result_gets_the_adiabat_warning(caplog, converged):
    cfg = {
        'temperature_mode': 'adiabatic',
        'layer_eos_config': {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
    }
    result = {'converged': converged, 'adiabat_computed': True, 'adiabat_blend': 0.5}
    zs._warn_adiabat(result, cfg, None, None, None)
    assert ('at adiabat blend 0.50' in caplog.text) is converged


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(2400)
def test_an_iteration_cap_whose_newest_blend_step_diverges_returns_an_earlier_best_pass(
    monkeypatch, caplog
):
    """3-layer PALEOS ice planet (cmf 0.3, mmf 0.4), adiabatic Picard, 20 outer passes and
    tolerance_outer 1e-2, so a linear pass starts the blend ramp also on the CI runner (whose
    linear passes end near 2.4e-3, above the default 1e-3): every pass of the newest blend step
    misses the mass by more than 3 %, so the cap returns the best pass of an earlier step (blend
    0, R about 1.2978e7 m), the log names the ramp start and the fallback, and main does not
    retry."""
    from tests.test_spider_coupling_convergence import _has_paleos_data
    from zalmoxis.config import load_solidus_liquidus_functions, load_zalmoxis_config

    if not _has_paleos_data():
        pytest.skip('PALEOS tables not on disk')
    cfg = load_zalmoxis_config(os.path.join(ROOT, 'input', 'default.toml'))
    cfg.update(
        data_output_enabled=False,
        plotting_enabled=False,
        core_mass_fraction=0.3,
        mantle_mass_fraction=0.4,
        layer_eos_config={
            'core': 'PALEOS:iron',
            'mantle': 'PALEOS:MgSiO3',
            'ice_layer': 'PALEOS:H2O',
        },
        wall_timeout=WALL,
        max_iterations_outer=20,
        tolerance_outer=1e-2,
    )
    assert cfg['temperature_mode'] == 'adiabatic'
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        'Stixrude14-solidus',
        'Stixrude14-liquidus',
        cfg['mushy_zone_factor'],
    )
    calls, real = [], zs._solve

    def once(*args, **kwargs):
        if calls:
            pytest.fail('main retried the solve')
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(zs, '_solve', once)
    with caplog.at_level(logging.INFO, logger='fwl.zalmoxis.solver'):
        r = zs.main(cfg, load_material_dictionaries(), melt, os.path.join(ROOT, 'input'))
    assert 'activating adiabat blend' in caplog.text
    assert 'No pass of adiabat blend' in caplog.text
    assert r['converged'] and r['accepted_on'] == 'max_iter'
    assert r['adiabat_blend'] == 0.0
    assert r['mass_error_at_accept'] < 1e-2
    assert r['radii'][-1] == pytest.approx(1.2978e7, rel=1e-2)


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(2400)
def test_a_single_pass_between_the_two_ceilings_is_a_retry_skip(monkeypatch):
    """0.5 M_E PALEOS iron/MgSiO3, linear T, one outer pass: the pass misses the mass by about
    5 %, above the 3 % max_iter ceiling and below the retry skip one, so main accepts it as
    retry_skip without a second solve, and the invariant check passes."""
    from tests.test_spider_coupling_convergence import _has_paleos_data
    from zalmoxis.config import load_solidus_liquidus_functions, load_zalmoxis_config

    if not _has_paleos_data():
        pytest.skip('PALEOS tables not on disk')
    cfg = load_zalmoxis_config(os.path.join(ROOT, 'input', 'default.toml'))
    cfg.update(
        data_output_enabled=False,
        plotting_enabled=False,
        planet_mass=0.5 * earth_mass,
        temperature_mode='linear',
        max_iterations_outer=1,
        wall_timeout=WALL,
        layer_eos_config={'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
    )
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        'Stixrude14-solidus',
        'Stixrude14-liquidus',
        cfg['mushy_zone_factor'],
    )
    calls, real = [], zs._solve

    def once(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(zs, '_solve', once)
    r = zs.main(cfg, load_material_dictionaries(), melt, os.path.join(ROOT, 'input'))
    assert r['converged'] and r['accepted_on'] == 'retry_skip' and len(calls) == 1
    assert r['accept_mass_limit'] == 0.07
    assert 0.03 < r['mass_error_at_accept'] < 0.07


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(2400)
def test_the_adiabat_of_the_default_solve_reads_the_layer_boundaries_of_its_profile(
    monkeypatch,
):
    """input/default.toml (1 M_E PALEOS iron/MgSiO3, adiabatic): every adiabat call gets the
    layer boundaries the previous pass integrated with, the planet-mass fractions, and the solve
    is accepted strict on the full adiabat. A boundary from the mass of the previous pass puts a
    node of that profile in the other layer and gives another pass."""
    from tests.test_spider_coupling_convergence import _has_paleos_data
    from zalmoxis.config import load_solidus_liquidus_functions, load_zalmoxis_config

    if not _has_paleos_data():
        pytest.skip('PALEOS tables not on disk')
    cfg = load_zalmoxis_config(os.path.join(ROOT, 'input', 'default.toml'))
    cfg.update(data_output_enabled=False, plotting_enabled=False, wall_timeout=WALL)
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        cfg['rock_solidus'],
        cfg['rock_liquidus'],
        cfg['mushy_zone_factor'],
    )
    seen, real = [], zs.compute_adiabatic_temperature

    def adiabat(radii, pressure, mass, surface_T, cmb_mass, core_mantle_mass, *args, **kwargs):
        seen.append((cmb_mass, core_mantle_mass))
        return real(
            radii, pressure, mass, surface_T, cmb_mass, core_mantle_mass, *args, **kwargs
        )

    monkeypatch.setattr(zs, 'compute_adiabatic_temperature', adiabat)
    r = zs.main(cfg, load_material_dictionaries(), melt, INPUT)
    fractions = (
        cfg['core_mass_fraction'],
        cfg['core_mass_fraction'] + cfg['mantle_mass_fraction'],
    )
    assert len(seen) >= 4  # the blend ramp ran
    assert set(seen) == {tuple(f * cfg['planet_mass'] for f in fractions)}
    assert r['accepted_on'] == 'strict' and r['adiabat_blend'] == 1.0
    assert r['mass_error_at_accept'] < r['accept_mass_limit'] == 1e-3


def _oscillating_solve(
    monkeypatch,
    caplog,
    factor,
    temperature_function=None,
    cut_on=None,
    cut_before=None,
    **kwargs,
):
    """Run _solve on 20 layers with the mass of outer pass n scaled to factor(n) Earth
    masses; the pass index is keyed on the outer radius. The test clock jumps past
    wall_timeout at the first structure solve of pass cut_on, or at the temperature profile
    of pass cut_before, before its first structure solve. Returns the result and the outer
    radii of the passes."""
    real, index, cut = zs.solve_structure, {}, [False]
    real_time, real_profile = time.time, zs.calculate_temperature_profile

    def solve(*args, **kw):
        m, g, P = real(*args, **kw)
        n = index.setdefault(float(args[3][-1]), len(index))
        cut[0] |= n == cut_on
        f = factor(n) * earth_mass / m[-1]
        return m * f, g * f, P

    def profile(*args, **kw):
        cut[0] |= len(index) == cut_before
        return real_profile(*args, **kw)

    monkeypatch.setattr(zs, 'solve_structure', solve)
    monkeypatch.setattr(zs, 'calculate_temperature_profile', profile)
    monkeypatch.setattr(zs, 'time', SimpleNamespace(time=lambda: real_time() + 1e9 * cut[0]))
    opts = {'max_iterations_inner': 5, 'num_layers': 20, **kwargs}
    cfg = _cfg(outer_solver='picard', wall_timeout=WALL, **opts)
    with caplog.at_level(logging.WARNING, logger='fwl.zalmoxis.solver'):
        r = zs._solve(
            cfg,
            load_material_dictionaries(),
            None,
            INPUT,
            temperature_function=temperature_function,
        )
    return r, list(index)


def _first_then_alternating(first, later):
    """Pass 0 at 1 + first, later passes at 1 +- later with alternating sign, shrinking by
    1e-6 per pass: far above round-off, far below the 1 % slack of the oscillation count."""
    return lambda n: 1 + first if n == 0 else 1 + (1 if n % 2 else -1) * later * (1 - 1e-6 * n)


STOP = 'Accepting best solution after'
PALEOS = {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'}


def _need_paleos():
    from tests.test_spider_coupling_convergence import _has_paleos_data

    if not _has_paleos_data():
        pytest.skip('PALEOS tables not on disk')


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.parametrize('inner', [5, 1])
def test_ten_mass_sign_changes_stop_the_outer_loop(monkeypatch, caplog, inner):
    """Pass 0 ends 0.2 % heavy, later passes 2 % off with alternating sign: they never beat
    pass 0, so after 10 sign changes the stop returns it. Adiabatic mode on the Analytic EOS
    has no blend ramp. A converged density accepts on 3 tolerance_outer, an unconverged
    one (1 inner iteration) on the 3 % limit of a timeout."""
    r, radius = _oscillating_solve(
        monkeypatch,
        caplog,
        _first_then_alternating(0.002, 0.02),
        max_iterations_outer=14,
        max_iterations_inner=inner,
        tolerance_outer=1e-3,
        temperature_mode='adiabatic',
    )
    assert r['accepted_on'] == 'oscillation', (r['accepted_on'], len(radius))
    assert 'Accepting best solution after 10 oscillations' in caplog.text
    assert r['mass_error_at_accept'] == pytest.approx(0.002, rel=1e-6)
    assert r['converged'] and r['converged_mass']
    assert r['converged_density'] == (inner == 5)
    assert r['accept_mass_limit'] == pytest.approx(3e-3 if inner == 5 else 0.03)
    assert len(radius) == 12  # the best pass, one more on its side, 10 sign changes


@pytest.mark.smoke
@pytest.mark.accept_solves
def test_no_oscillation_stop_while_the_best_pass_is_above_3_tolerance_outer(
    monkeypatch, caplog
):
    """Pass 0 at 5 %, later passes 6 % with alternating sign, tolerance_outer 1e-3: the
    best pass stays above 3 tolerance_outer, so the loop runs to its cap."""
    r, radius = _oscillating_solve(
        monkeypatch,
        caplog,
        _first_then_alternating(0.05, 0.06),
        max_iterations_outer=14,
        tolerance_outer=1e-3,
    )
    assert STOP not in caplog.text
    assert r['accepted_on'] is None and len(radius) == 14


@pytest.mark.unit
def test_no_oscillation_stop_above_the_3_percent_limit(monkeypatch, caplog):
    """tolerance_outer 0.015, 1 inner iteration: the best pass (3.5 %) is below 3
    tolerance_outer but above the 3 % limit of an unconverged density, so no stop fires and
    the mass is not marked converged."""
    r, radius = _oscillating_solve(
        monkeypatch,
        caplog,
        _first_then_alternating(0.035, 0.05),
        max_iterations_outer=14,
        max_iterations_inner=1,
        tolerance_outer=0.015,
    )
    assert STOP not in caplog.text and len(radius) == 14
    assert r['accepted_on'] is None and not r['converged_mass'] and not r['converged']


@pytest.mark.smoke
@pytest.mark.accept_solves
def test_a_new_best_pass_resets_the_oscillation_count(monkeypatch, caplog):
    """Pass 0 ends 0.25 % heavy, passes 1 to 8 alternate at 0.29 % (8 sign changes), pass 9
    is a new best at 0.24 % with the sign of pass 8, later passes alternate at 0.29 %: the
    new best resets the count, so it stays below 10 and the cap stops the loop."""

    def factor(n):
        if n in (0, 9):
            return 1.0025 if n == 0 else 1.0024
        return 1 + (-1) ** (n + (n > 9)) * 0.0029 * (1 - 1e-6 * n)

    r, radius = _oscillating_solve(
        monkeypatch, caplog, factor, max_iterations_outer=14, tolerance_outer=1e-3
    )
    assert STOP not in caplog.text
    assert r['accepted_on'] == 'max_iter' and len(radius) == 14


@pytest.mark.smoke
@pytest.mark.accept_solves
def test_a_contracting_oscillation_is_not_stopped(monkeypatch, caplog):
    """Pass 0 ends 0.2 % heavy, later passes 2 % x 0.85^n with alternating sign: the error
    shrinks at every sign change, so the stop waits and the mass reaches tolerance_outer."""
    r, radius = _oscillating_solve(
        monkeypatch,
        caplog,
        lambda n: 1.002 if n == 0 else 1 + (-1) ** n * 0.02 * 0.85**n,
        max_iterations_outer=24,
        tolerance_outer=1e-3,
    )
    assert STOP not in caplog.text
    assert r['mass_error_at_accept'] < 1e-3 and len(radius) == 20


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.parametrize('mode', ['adiabatic', 'adiabatic_from_cmb'])
def test_the_oscillation_stop_waits_for_the_adiabat_blend_ramp(monkeypatch, caplog, mode):
    """PALEOS: pass 0 ends 0.2 % heavy, later passes 2 % off with alternating sign, so no
    pass reaches tolerance_outer and the blend ramp never starts. The stop must not return
    the linear temperature pass; the loop ends at its cap on blend 0 as max_iter."""
    _need_paleos()
    r, radius = _oscillating_solve(
        monkeypatch,
        caplog,
        _first_then_alternating(0.002, 0.02),
        max_iterations_outer=14,
        tolerance_outer=1e-3,
        temperature_mode=mode,
        layer_eos_config=PALEOS,
    )
    assert STOP not in caplog.text
    assert r['accepted_on'] == 'max_iter' and r['adiabat_blend'] == pytest.approx(0.0)
    assert len(radius) == 14


@pytest.mark.smoke
@pytest.mark.accept_solves
def test_the_oscillation_stop_fires_after_the_adiabat_blend_ramp(monkeypatch, caplog):
    """PALEOS, adiabatic: pass 0 reaches tolerance_outer and starts the ramp, pass 4 ends on
    blend 1.0 0.2 % heavy, later passes 2 % off with alternating sign: the stop fires on
    blend 1.0 and returns pass 4."""
    _need_paleos()
    sched = {0: 1.0005, 1: 1.05, 2: 1.05, 3: 1.05, 4: 1.002}
    r, radius = _oscillating_solve(
        monkeypatch,
        caplog,
        lambda n: sched.get(n, _first_then_alternating(0, 0.02)(n)),
        max_iterations_outer=22,
        tolerance_outer=1e-3,
        temperature_mode='adiabatic',
        layer_eos_config=PALEOS,
    )
    assert r['accepted_on'] == 'oscillation', (r['accepted_on'], len(radius))
    assert r['adiabat_blend'] == pytest.approx(1.0)
    assert r['mass_error_at_accept'] == pytest.approx(0.002, rel=1e-6)
    assert len(radius) == 16


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.parametrize('mode', ['isothermal', 'adiabatic'])
def test_the_oscillation_stop_fires_on_a_tdep_eos_without_a_blend_ramp(
    monkeypatch, caplog, mode
):
    """PALEOS with no internal adiabat (isothermal, or an external temperature function):
    the schedule of test_ten_mass_sign_changes_stop_the_outer_loop stops on oscillation."""
    _need_paleos()
    r, radius = _oscillating_solve(
        monkeypatch,
        caplog,
        _first_then_alternating(0.002, 0.02),
        temperature_function=None if mode == 'isothermal' else lambda r, P: 4000.0,
        max_iterations_outer=14,
        tolerance_outer=1e-3,
        temperature_mode=mode,
        layer_eos_config=PALEOS,
    )
    assert r['accepted_on'] == 'oscillation', (r['accepted_on'], len(radius))
    assert len(radius) == 12


@pytest.mark.smoke
@pytest.mark.accept_solves
def test_a_wall_clock_cut_on_the_tenth_sign_change_stops_on_timeout(monkeypatch, caplog):
    """The schedule of test_ten_mass_sign_changes_stop_the_outer_loop with the clock cutting
    the inner loop of pass 11, the pass of the tenth sign change, after its first pressure
    solve: the stop is the wall clock's, and the best pass is returned as timeout."""
    r, radius = _oscillating_solve(
        monkeypatch,
        caplog,
        _first_then_alternating(0.002, 0.02),
        cut_on=11,
        max_iterations_outer=14,
        tolerance_outer=1e-3,
        temperature_mode='adiabatic',
    )
    assert STOP not in caplog.text and len(radius) == 12
    assert r['accepted_on'] == 'timeout' and r['stopped_on_wall_clock']
    assert r['mass_error_at_accept'] == pytest.approx(0.002, rel=1e-6)


@pytest.mark.smoke
@pytest.mark.accept_solves
def test_a_pass_cut_before_its_pressure_solve_does_not_stop_on_oscillation(monkeypatch, caplog):
    """Pass 0 ends 0.2 % light, later passes 2 % off with alternating sign, and the clock
    cuts pass 11 before its first pressure solve: its mass of zero (error -1) would be the
    tenth sign change, but the stop is the wall clock's."""
    r, radius = _oscillating_solve(
        monkeypatch,
        caplog,
        lambda n: 0.998 if n == 0 else 1 + (-1 if n % 2 else 1) * 0.02 * (1 - 1e-6 * n),
        cut_before=11,
        max_iterations_outer=14,
        tolerance_outer=1e-3,
        temperature_mode='adiabatic',
    )
    assert STOP not in caplog.text and len(radius) == 11
    assert r['accepted_on'] == 'timeout' and r['stopped_on_wall_clock']
    assert r['density_from_eos']
    assert r['mass_error_at_accept'] == pytest.approx(0.002, rel=1e-6)
