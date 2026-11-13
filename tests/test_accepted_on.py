"""The ``accepted_on`` and ``mass_error_at_accept`` fields of a structure solve result.

Each accept path of ``main`` names itself in ``accepted_on``; the returned density is
rho_EOS(P, T) at the returned nodes, and a Newton step result is converged only when the
inner solve of its returned profile reports converged pressure and that density.
"""

from __future__ import annotations

import json
import logging
import os
import time
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

import zalmoxis.solver as zs
from tests.test_failed_structure_solve import ROOT, _cfg, _count_outer_iterations, _run
from tests.test_table_band_report import _need_wb_tables
from tools.validation import grid_start_up_reference as grid_ref
from tools.validation.grid_start_up_reference import GRID_CMF95
from zalmoxis.config import load_material_dictionaries
from zalmoxis.constants import G, earth_mass, earth_radius
from zalmoxis.invariants import StructureInvariantError
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


@pytest.fixture
def canned(no_check, monkeypatch):
    """main on a canned _solve result: no physics-invariant check and no table-band report."""
    monkeypatch.setattr(zs, '_report_table_bands', lambda r, *a: r)


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

    @pytest.mark.parametrize('inner', [2, 3])
    def test_newton_with_unconverged_density_flag(self, inner):
        """The inner caps at which the Picard density flag is False: the returned density is
        the EOS density, so the result is converged. Measured flags for caps 2 to 6: False,
        False, True, True, True on macOS arm64 and Linux amd64, True from cap 4 on since the
        refined pressure root lets the density converge; without the refinement False, False,
        True, False, True on macOS and False, False, False, False, True on Linux. Caps 2 and 3
        are False with and without the refinement on both platforms."""
        r = _run(_cfg(outer_solver='newton', max_iterations_inner=inner, wall_timeout=WALL))
        assert not r['converged_density']
        assert r['converged'] is True and r['accepted_on'] == 'newton_tol'
        _assert_eos_density(r)


@pytest.mark.smoke
@pytest.mark.wolfbower
def test_wall_clock_in_the_adiabat_ramp_is_retried_to_the_full_adiabat(monkeypatch, caplog):
    """The clock runs out after the mass converged, in the adiabat blend ramp: the linear
    pass is not accepted, main retries, and the retry returns the full adiabat."""
    from zalmoxis.eos import get_solidus_liquidus_functions

    _need_wb_tables()
    caplog.set_level(logging.INFO, logger='fwl.zalmoxis.solver')
    ramp = lambda: any('activating adiabat blend' in m for m in caplog.messages)  # noqa: E731
    clock = time.time
    monkeypatch.setattr(zs, 'time', SimpleNamespace(time=lambda: clock() + 1e9 * ramp()))
    tighten = zs._tighten_solver_params  # the retry clock is 2x the 300 s default
    monkeypatch.setattr(
        zs, '_tighten_solver_params', lambda p: {**tighten(p), 'wall_timeout': 900.0}
    )
    cfg = _cfg(
        temperature_mode='adiabatic',
        layer_eos_config={'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'},
        tolerance_outer=0.02,
        max_iterations_inner=2,
        wall_timeout=WALL,
        num_layers=30,  # a short retry
        relative_tolerance=1e-5,
        absolute_tolerance=1e-6,
    )
    melt = get_solidus_liquidus_functions()
    r = zs.main(cfg, load_material_dictionaries(), melt, '/tmp')
    _assert_eos_density(r, cfg['layer_eos_config'], melt)  # T-dependent mantle, restored pass
    assert 'Wall-clock timeout (1000000s)' in caplog.text
    assert 'No accept of a pass at adiabat blend 0.00' in caplog.text
    assert 'Retrying with tighter parameters' in caplog.text
    assert r['converged'] is True and r['accepted_on'] is not None
    assert (
        r['adiabat_computed'] is True and r['adiabat_blend'] == r['adiabat_table_blend'] == 1.0
    )


@pytest.mark.smoke
@pytest.mark.wolfbower
def test_wall_clock_in_the_adiabat_ramp_returns_the_linear_pass_unaccepted(monkeypatch, caplog):
    """As above with tolerance_outer 0.05, in _solve: the clock stop returns the linear pass,
    3 to 5 % off the mass, with no accept and no mass limit, as the start of a retry."""
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
    r = zs._solve(cfg, load_material_dictionaries(), get_solidus_liquidus_functions(), '/tmp')
    assert r['stopped_on_wall_clock'] is True and r['adiabat_blend'] == 0.0
    assert r['converged'] is False and r['accepted_on'] is None
    assert r['accept_mass_limit'] is None
    assert 0.03 < r['mass_error_at_accept'] < 0.05


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
def test_a_clock_stop_returns_the_blended_pass_its_mass_test_takes(monkeypatch, caplog):
    """tolerance_outer 0.05 and blended passes 4 % off: the mass test of the clock stop takes
    the newest pass above the 3 % ceiling, so it is returned, not the linear one; at blend 0.25
    it is not accepted."""
    with caplog.at_level(logging.WARNING, logger='fwl.zalmoxis.solver'):
        r = _wb_blended(monkeypatch, 1.04, 0.05)
    assert r['stopped_on_wall_clock'] is True and r['accepted_on'] is None
    assert r['converged'] is False
    assert r['adiabat_blend'] == 0.25
    assert r['mass_error_at_accept'] == pytest.approx(0.04, rel=1e-9)
    assert 'No pass of adiabat blend' not in caplog.text


@pytest.mark.smoke
@pytest.mark.wolfbower
@pytest.mark.usefixtures('no_check')
def test_a_clock_stop_beyond_its_limit_returns_the_linear_pass_under_that_limit(monkeypatch):
    """The blended passes end 10 % off, beyond the 6 % limit of a clock stop: the linear pass
    that started the ramp is returned, not accepted, as its temperature is not the adiabat."""
    r = _wb_blended(monkeypatch, 1.10, 0.02)
    assert r['stopped_on_wall_clock'] is True and r['accepted_on'] is None
    assert r['adiabat_blend'] == 0.0 and r['converged'] is False
    assert r['mass_error_at_accept'] < 0.02


@pytest.mark.smoke
@pytest.mark.wolfbower
@pytest.mark.usefixtures('no_check')
def test_a_clock_stop_keeps_a_blended_pass_whose_inner_solves_converged(monkeypatch):
    """Blended passes 5 % off, inside the 6 % limit of a clock stop but above tolerance_outer
    0.02 and the 3 % ceiling: the first blended pass ran its inner solves to convergence, so
    the strict test of the stop takes it and no earlier pass replaces it; at blend 0.25 it is
    not accepted."""
    r = _wb_blended(monkeypatch, 1.05, 0.02, inner=100, clock_after=2)
    assert r['accepted_on'] is None and r['converged'] is False
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
    assert r['stopped_on_wall_clock'] is True and r['accepted_on'] is None
    assert r['adiabat_blend'] == 0.0
    assert r['mass_error_at_accept'] < 0.02
    assert r['best_mass_error'] == pytest.approx(r['mass_error_at_accept'], rel=1e-9)
    assert 'No pass of adiabat blend 0.25' in caplog.text
    assert (
        f'best pass of blend 0.00 on a table of blend 0.00 (mass error '
        f'{r["best_mass_error"] * 100:.4f}%)' in caplog.text
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
    blended pass is 4.5 % off and the linear pass 4 %: the linear pass is returned, not
    accepted."""
    r = _wb_blended(monkeypatch, 1.045, 0.05, outer=3, clock_after=2, linear=1.04)
    assert r['accepted_on'] is None and r['adiabat_blend'] == 0.0
    assert r['mass_error_at_accept'] == pytest.approx(0.04, rel=1e-9)


@pytest.mark.smoke
@pytest.mark.wolfbower
@pytest.mark.usefixtures('no_check')
def test_a_clock_stop_swap_returns_the_pass_its_mass_test_read(monkeypatch):
    """tolerance_outer 0.05, the clock runs out in the first blended pass, 6 % off, and the
    linear pass is 4.5 % off: the swap returns the linear pass, and the mass error of the
    result is the one of the returned pass, not of the blended pass."""
    r = _wb_blended(monkeypatch, 1.06, 0.05, inner=5, outer=2, linear=1.045)
    assert r['accepted_on'] is None and r['adiabat_blend'] == 0.0
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
    assert r['accepted_on'] is None and r['adiabat_blend'] == 0.0
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
def test_a_max_iter_stop_in_the_blend_ramp_returns_a_pass_on_the_adiabat(caplog, tmp_path):
    """PALEOS, adiabatic, 4 outer passes: the mass converges on the linear temperature, and
    the cap stops the solve in the blend ramp. The returned pass is on the newest blend the
    solver logged, not the linear pass with the smallest mass error, and it is not accepted.
    Measured newest blend: 0.25 without the root refinement on macOS and Linux; with
    it 0.5 on macOS and 0.25 on Linux."""
    from tests.test_spider_coupling_convergence import _has_paleos_data

    caplog.set_level(logging.DEBUG, logger='fwl.zalmoxis.solver')

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
        debug_profile_dir=str(tmp_path),
    )
    with patch.object(zs, 'create_pressure_density_files') as dump:
        r = zs._solve(cfg, load_material_dictionaries(), None, INPUT)
    assert dump.called  # the DEBUG profile dump of the solve ran into the stub
    T = np.asarray(r['temperature'])
    off_linear = np.max(np.abs(T - np.linspace(6000.0, 3000.0, len(T))))
    assert r['converged'] is False and r['accepted_on'] is None
    assert r['adiabat_computed'] is True and off_linear > 100.0
    steps = [float(m.rsplit('= ', 1)[1]) for m in caplog.messages if 'adiabat blend = ' in m]
    assert steps and 0.0 < steps[-1] < 1.0, steps
    assert round(r['adiabat_blend'], 2) == steps[-1]  # the log prints %.2f
    assert f'No accept of a pass at adiabat blend {steps[-1]:.2f}' in caplog.text


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
    before its first pressure solve. The linear pass that met the mass test comes back, not
    accepted, with its EOS density and adiabat_blend 0, not an empty profile."""
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
    assert r['converged'] is False and r['density_from_eos'] is True
    assert r['adiabat_blend'] == 0.0 and r['mass_error_at_accept'] < 1e-3


LABELS = (
    'strict', 'mass_only', 'oscillation', 'max_iter', 'timeout', 'retry_skip', 'newton_tol',
    'brentq_best',
)  # fmt: skip


def _adiabatic_cfg(eos='PALEOS', mode='adiabatic', **kwargs):
    return {
        'planet_mass': M_T,
        'temperature_mode': mode,
        'layer_eos_config': {'core': f'{eos}:iron', 'mantle': f'{eos}:MgSiO3'},
        **kwargs,
    }


SHORT = [  # (eos, mode, computed, blend, table, raises); SHORT[2] raises, SHORT[9] passes
    ('PALEOS', 'adiabatic', False, 0.0, 0.0, True),
    ('PALEOS', 'adiabatic_from_cmb', True, 0.5, 0.25, True),
    ('PALEOS', 'adiabatic', True, 0.75, 0.5, True),
    ('PALEOS', 'adiabatic', True, float('nan'), 1.0, True),
    ('PALEOS', 'adiabatic', False, 1.0, 1.0, True),  # no computed adiabat: blend 0
    ('PALEOS', 'adiabatic', True, 1.0, 0.75, True),  # its table is from the 0.75 pass
    ('PALEOS', 'adiabatic_from_cmb', True, 1.0, 0.75, True),
    ('PALEOS', 'adiabatic', True, 1.0, float('nan'), True),
    ('PALEOS', 'adiabatic', True, 1.0, None, True),
    ('PALEOS', 'adiabatic', True, 1.0, 1.0, False),
    ('PALEOS', 'adiabatic_from_cmb', True, 1.0, 1.0, False),
    ('PALEOS', 'linear', False, 0.0, 0.0, False),
    ('Analytic', 'adiabatic', False, 0.0, 0.0, False),  # no T-dependent component: no ramp
]
assert SHORT[2][-1] and not SHORT[9][-1]  # the rows of every label: one refused, one accepted
ARRAYS = {'temperature_arrays': (np.array([0.0, 7.0e6]), np.full(2, 3000.0))}
EXTERNAL_T = pytest.mark.parametrize(  # the numpy path ignores temperature_arrays
    'kwargs, use_jax, external',
    [
        ({'temperature_function': lambda r, P: 3000.0}, False, True),
        ({'temperature_function': lambda r, P: 3000.0}, True, True),
        (ARRAYS, True, True),
        (ARRAYS, False, False),
    ],
    ids=['function_numpy', 'function_jax', 'arrays_jax', 'arrays_numpy'],
)


@pytest.mark.unit
@pytest.mark.usefixtures('canned')
@pytest.mark.parametrize(
    'label, eos, mode, computed, blend, table, raises',
    [('max_iter', *row) for row in SHORT]
    + [(label, *SHORT[k]) for label in LABELS if label != 'max_iter' for k in (2, 9)],
)
def test_main_refuses_an_accepted_profile_short_of_the_adiabat(
    monkeypatch, label, eos, mode, computed, blend, table, raises
):
    """Every accept label of a result whose profile is not the full adiabat, or whose T table
    was built on the pressure of a pass short of it, raises, when the solve computes the
    adiabat of the temperature mode itself."""
    first = {**_inner(7.0e6, accepted_on=label), 'adiabat_computed': computed}
    first.update(adiabat_blend=blend, adiabat_table_blend=table)
    monkeypatch.setattr(zs, '_solve', lambda *a, **k: dict(first))
    cfg = _adiabatic_cfg(eos, mode)
    if not raises:
        assert zs.main(cfg, {}, None, '/tmp')['accepted_on'] == label
        return
    with pytest.raises(
        StructureInvariantError, match=f"accepted_on='{label}' at adiabat blend"
    ):
        zs.main(cfg, {}, None, '/tmp')


@pytest.mark.unit
@pytest.mark.usefixtures('canned')
@EXTERNAL_T
def test_an_external_temperature_profile_is_accepted_at_blend_0(
    monkeypatch, kwargs, use_jax, external
):
    """With the temperature of the caller no adiabat is asked for, so blend 0 is accepted;
    temperature arrays on the numpy path leave the adiabat asked, and blend 0 is refused."""
    first = {**_inner(7.0e6, accepted_on='strict'), 'adiabat_computed': False}
    first['adiabat_blend'] = 0.0
    monkeypatch.setattr(zs, '_solve', lambda *a, **k: dict(first))
    cfg = _adiabatic_cfg(use_jax=use_jax)
    if external:
        assert zs.main(cfg, {}, None, '/tmp', **kwargs)['accepted_on'] == 'strict'
        return
    with pytest.raises(StructureInvariantError, match='at adiabat blend'):
        zs.main(cfg, {}, None, '/tmp', **kwargs)


@pytest.mark.unit
@pytest.mark.usefixtures('canned')
@EXTERNAL_T
def test_newton_with_an_external_temperature_returns_the_newton_result(
    monkeypatch, kwargs, use_jax, external
):
    """With the temperature of the caller no adiabat is asked for, so main returns a converged
    Newton result of an adiabatic mode with a T-dependent EOS without the blend ramp; with the
    adiabat asked, the Newton root seeds the ramp."""
    newton = {**_inner(7.0e6, accepted_on='newton_tol'), 'adiabat_computed': False}
    newton.update(adiabat_blend=0.0, total_time=1.0, newton_n_iter=4, newton_history=[])
    newton.update(
        newton_used_brentq=False, pressure=np.array([3.6e11, 0.0]), density=np.ones(2)
    )
    picard = {**_inner(7.0e6, accepted_on='strict'), 'adiabat_computed': True}
    picard.update(adiabat_blend=1.0, adiabat_table_blend=1.0, total_time=1.0)
    ramps = []
    monkeypatch.setattr(zs, '_solve_newton_outer', lambda *a, **k: newton)
    monkeypatch.setattr(zs, '_solve', lambda *a, **k: ramps.append(1) or dict(picard))
    cfg = _adiabatic_cfg(outer_solver='newton', use_jax=use_jax)
    r = zs.main(cfg, {}, None, '/tmp', **kwargs)
    assert (r is newton, len(ramps)) == ((True, 0) if external else (False, 1))
    assert r['accepted_on'] == ('newton_tol' if external else 'strict')


@pytest.mark.unit
def test_an_unaccepted_result_short_of_the_adiabat_passes_the_check():
    result = {'converged': False, 'adiabat_computed': False, 'adiabat_blend': 0.0}
    cfg = _adiabatic_cfg()
    assert zs._check_accepted(result, cfg, cfg, {}, None, None, None, True) is result


@pytest.mark.unit
@pytest.mark.usefixtures('canned')
@pytest.mark.parametrize(
    'computed, blend, table, solves, label',
    [
        (False, 0.0, 0.0, 2, 'strict'),
        (True, 0.5, 0.25, 2, 'strict'),
        (True, 1.0, 0.75, 2, 'strict'),
        (True, 1.0, 1.0, 1, 'retry_skip'),
    ],
)
def test_a_retry_skip_takes_only_the_full_adiabat(
    monkeypatch, computed, blend, table, solves, label
):
    """An unaccepted first solve with converged density and pressure and a mass error under
    7 % skips the retry only when its profile is the full adiabat on a table of a full-adiabat
    pass; otherwise main retries."""
    first = {**_inner(7.0e6), 'converged': False, 'accepted_on': None}
    first.update(mass_error_at_accept=0.01, adiabat_computed=computed, adiabat_blend=blend)
    first['adiabat_table_blend'] = table
    second = {**_inner(7.0e6, accepted_on='strict'), 'adiabat_computed': True}
    second.update(adiabat_blend=1.0, adiabat_table_blend=1.0)
    results = [first, second]
    monkeypatch.setattr(zs, '_solve', lambda *a, **k: dict(results.pop(0)))
    r = zs.main(_adiabatic_cfg(), {}, None, '/tmp')
    assert 2 - len(results) == solves and r['accepted_on'] == label


@pytest.mark.unit
@pytest.mark.usefixtures('canned')
@pytest.mark.parametrize('wall', [1000.0, WALL])
def test_the_retry_runs_on_a_600_s_clock_whatever_the_configured_clock(monkeypatch, wall):
    """The retry takes its clock from the solver defaults, 2 x 300 s: it does not read the
    configured ``wall_timeout``, and this test fixes that current value."""
    first = {
        **_inner(7.0e6),
        'converged': False,
        'accepted_on': None,
        'mass_error_at_accept': 0.5,
    }
    second = {**_inner(7.0e6, accepted_on='strict'), 'adiabat_computed': True}
    second.update(adiabat_blend=1.0, adiabat_table_blend=1.0)
    results, clocks = [first, second], []

    def solve(cfg, *a, **k):
        clocks.append(cfg['wall_timeout'])
        return dict(results.pop(0))

    monkeypatch.setattr(zs, '_solve', solve)
    zs.main(_adiabatic_cfg(wall_timeout=wall), {}, None, '/tmp')
    assert clocks == [wall, 600.0]


@pytest.mark.unit
@pytest.mark.usefixtures('canned')
@pytest.mark.parametrize(
    'eos, converged, ramp',
    [('PALEOS', True, True), ('PALEOS', False, False), ('Analytic', True, False)],
)
def test_newton_continues_with_the_blend_ramp_from_its_profile(
    monkeypatch, eos, converged, ramp
):
    """A converged Newton result of an adiabatic mode with a T-dependent EOS holds the linear
    first-guess temperature, so main continues with the Picard blend ramp seeded from its
    profile (radius, central pressure, density and radii) at the mass tolerance
    min(tolerance_outer, newton_tol), and the result keeps the Newton bookkeeping. Otherwise
    main returns the Newton result."""
    newton = {**_inner(7.0e6, accepted_on='newton_tol'), 'converged': converged}
    newton.update(pressure=np.array([3.6e11, 0.0]), density=np.array([1.2e4, 3.0e3]))
    newton.update(adiabat_computed=False, adiabat_blend=0.0, total_time=1.0)
    books = {
        'newton_n_iter': 4,
        'newton_history': [(7.0e6, M_T, 0.0)],
        'newton_used_brentq': False,
    }
    newton.update(books)
    picard = {**_inner(7.0e6, accepted_on='strict'), 'adiabat_computed': True}
    picard.update(adiabat_blend=1.0, adiabat_table_blend=1.0, total_time=1.0)
    seeds = []

    def solve(cp, *args, **kwargs):
        seeds.append((cp, kwargs))
        return dict(picard)

    monkeypatch.setattr(zs, '_solve_newton_outer', lambda *a, **k: newton)
    monkeypatch.setattr(zs, '_solve', solve)
    cfg = _adiabatic_cfg(eos, outer_solver='newton', tolerance_outer=3e-3, newton_tol=2e-4)
    r = zs.main(cfg, {}, None, '/tmp')
    if not ramp:
        assert not seeds and r is newton
        return
    assert len(seeds) == 1 and r['accepted_on'] == 'strict' and r['adiabat_blend'] == 1.0
    cp, kwargs = seeds[0]
    assert cp['_initial_radius_guess'] == 7.0e6 and cp['tolerance_outer'] == 2e-4
    assert kwargs['p_center_hint'] == 3.6e11
    assert kwargs['initial_density'] is newton['density']
    assert kwargs['initial_radii'] is newton['radii']
    assert {k: r[k] for k in books} == books
    assert cfg['tolerance_outer'] == 3e-3 and '_initial_radius_guess' not in cfg


@pytest.mark.unit
@pytest.mark.usefixtures('canned')
@pytest.mark.parametrize(
    'tols, expected',
    [
        ({'tolerance_outer': 3e-3, 'newton_tol': 2e-4}, 2e-4),
        ({'tolerance_outer': 1e-5, 'newton_tol': 1e-4}, 1e-5),
        ({}, 1e-4),  # the defaults: tolerance_outer 1e-3, newton_tol 1e-4
    ],
)
def test_the_newton_hand_off_and_its_retry_run_at_the_newton_mass_tolerance(
    monkeypatch, tols, expected
):
    """The Picard continuation after a Newton root and its retry both run at the mass
    tolerance min(tolerance_outer, newton_tol): the retry does not tighten it again. The
    result's total_time adds the Newton time to the time of the solve that returns."""
    newton = {**_inner(7.0e6, accepted_on='newton_tol'), 'total_time': 200.0}
    newton.update(pressure=np.array([3.6e11, 0.0]), density=np.array([1.2e4, 3.0e3]))
    newton.update(newton_n_iter=4, newton_history=[], newton_used_brentq=False)
    failed = {**_inner(7.1e6), 'converged': False, 'accepted_on': None, 'total_time': 50.0}
    failed.update(converged_density=False, adiabat_computed=True, adiabat_blend=0.5)
    picard = {**_inner(7.0e6, accepted_on='strict'), 'total_time': 30.0}
    picard.update(adiabat_computed=True, adiabat_blend=1.0, adiabat_table_blend=1.0)
    results, configs = [failed, picard], []

    def solve(cp, *args, **kwargs):
        configs.append(cp)
        return dict(results.pop(0))

    monkeypatch.setattr(zs, '_solve_newton_outer', lambda *a, **k: newton)
    monkeypatch.setattr(zs, '_solve', solve)
    r = zs.main(_adiabatic_cfg(outer_solver='newton', **tols), {}, None, '/tmp')
    assert [cp['tolerance_outer'] for cp in configs] == [expected, expected]
    assert r['accepted_on'] == 'strict' and r['total_time'] == 230.0


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(2400)
def test_an_iteration_cap_whose_newest_blend_step_diverges_returns_an_earlier_best_pass(caplog):
    """3-layer PALEOS ice planet (cmf 0.3, mmf 0.4), adiabatic Picard, 20 outer passes and
    tolerance_outer 1e-2, so a linear pass starts the blend ramp also on the CI runner (whose
    linear passes end near 2.4e-3, above the default 1e-3): every pass of the newest blend step
    misses the mass by more than 3 %, so the cap returns the best pass of an earlier step (blend
    0, R about 1.2978e7 m), not accepted, and the log names the ramp start and the fallback."""
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
    with caplog.at_level(logging.INFO, logger='fwl.zalmoxis.solver'):
        r = zs._solve(cfg, load_material_dictionaries(), melt, os.path.join(ROOT, 'input'))
    assert 'activating adiabat blend' in caplog.text
    assert 'No pass of adiabat blend' in caplog.text
    assert not r['converged'] and r['accepted_on'] is None
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
    the linear temperature pass; the loop ends at its cap on blend 0, not accepted."""
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
    assert r['accepted_on'] is None and r['adiabat_blend'] == pytest.approx(0.0)
    assert len(radius) == 14


@pytest.mark.smoke
@pytest.mark.accept_solves
def test_the_oscillation_stop_fires_after_the_adiabat_blend_ramp(monkeypatch, caplog):
    """PALEOS, adiabatic: pass 0 reaches tolerance_outer and starts the ramp, pass 4 is the
    first at blend 1.0 and reads the table of the blend-0.75 pass, pass 5 is the first on a
    table of a blend-1.0 pass and ends 0.2 % heavy, later passes 2 % off with alternating
    sign: the stop fires on blend 1.0 and returns pass 5."""
    _need_paleos()
    sched = {0: 1.0005, 1: 1.05, 2: 1.05, 3: 1.05, 4: 1.05, 5: 1.002}
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
    assert r['adiabat_table_blend'] == pytest.approx(1.0)
    assert r['mass_error_at_accept'] == pytest.approx(0.002, rel=1e-6)
    assert len(radius) == 16  # passes 6 to 15 change sign 10 times


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


def _default_toml(**kwargs):
    """Zalmoxis input/default.toml (PALEOS iron core, PALEOS MgSiO3 mantle, adiabatic) with
    ``kwargs``, and its melting curves."""
    from zalmoxis.config import load_solidus_liquidus_functions, load_zalmoxis_config

    cfg = load_zalmoxis_config(os.path.join(INPUT, 'default.toml'))
    cfg.update(data_output_enabled=False, plotting_enabled=False, wall_timeout=WALL, **kwargs)
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'], cfg['rock_solidus'], cfg['rock_liquidus'], 1.0
    )
    return cfg, melt


@pytest.mark.smoke
@pytest.mark.accept_solves_long
@pytest.mark.timeout(4800)
def test_a_tight_tolerance_starts_the_blend_ramp_and_accepts_only_the_adiabat():
    """default.toml at a 2515 K surface, tolerance_outer 1e-7 and 30 outer passes: no linear
    pass reaches 1e-7, so the blend ramp starts at the 1e-3 trigger, and an accepted result is
    the full adiabat, not the linear first guess with its 6000 K centre."""
    from tests.test_spider_coupling_convergence import _has_paleos_data

    if not _has_paleos_data():
        pytest.skip('PALEOS tables not on disk')
    cfg, melt = _default_toml(
        surface_temperature=2515.0, tolerance_outer=1e-7, max_iterations_outer=30
    )
    r = zs.main(cfg, load_material_dictionaries(), melt, INPUT)
    assert r['adiabat_computed'] is True
    assert r['converged'] is True and r['adiabat_blend'] == 1.0
    assert r['temperature'][0] != cfg['center_temperature']


@pytest.mark.smoke
@pytest.mark.accept_solves_long
@pytest.mark.timeout(4800)
@pytest.mark.parametrize('case', ['all_options', 'dummy'])
def test_newton_gives_the_mantle_adiabat_of_picard_on_the_proteus_initial_structure(case):
    """The PROTEUS initial structure solve (adiabatic_from_cmb, PALEOS, 1 Earth mass, the
    PROTEUS solver settings): the Newton result has the mantle temperature of the Picard result
    at 39 mantle mass points within 5 K and its CMB pressure within 0.1 GPa."""
    from tests.test_spider_coupling_convergence import _has_paleos_data
    from tools.validation import proteus_ic_solvers as tool

    if not _has_paleos_data():
        pytest.skip('PALEOS iron and unified MgSiO3 tables not on disk')
    runs = tool.run_case(case)
    for s, rec in runs.items():
        assert rec['converged'] and rec['adiabat_computed'] and rec['blend'] == 1.0, s
    c = tool.compare(runs['picard'], runs['newton'])
    print(case, json.dumps({k: c[k] for k in ('dT', 'dP')}))
    assert c['dT'] < tool.LIMIT_T, f'mantle T differs by {c["dT"]:.3f} K'
    assert c['dP'] < tool.LIMIT_P, f'CMB pressure differs by {c["dP"] / 1e9:.4f} GPa'


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    ('case', 'solver', 'tol'),
    [
        ('dummy', 'newton', None),
        ('all_options', 'newton', None),
        ('all_options', 'picard', 1e-4),
    ],
)
def test_the_returned_mantle_temperature_is_the_adiabat_of_the_returned_pressure(
    monkeypatch, case, solver, tol
):
    """PROTEUS initial structure (CMB anchor), Picard at the mass tolerance of the Newton
    hand-off: the returned mantle T equals the adiabat recomputed on the returned pressure
    within 1 K, apart from 2 nodes with their own bounds. The CMB anchor node reads the T(P)
    table across the jump from T_surf on the core nodes to T_cmb (within 6 K of T_cmb); the
    top node can lose the low-pressure step of the adiabat (within 25 K).
    adiabat_self_consistency is the largest mantle difference, both nodes included."""
    from tests.test_spider_coupling_convergence import _has_paleos_data
    from tools.validation import proteus_ic_solvers as tool

    if not _has_paleos_data():
        pytest.skip('PALEOS iron and unified MgSiO3 tables not on disk')
    real, args = zs.compute_adiabatic_temperature, []

    def adiabat(radii, pressure, mass, *rest, **kwargs):
        args.append((rest, kwargs))
        return real(radii, pressure, mass, *rest, **kwargs)

    monkeypatch.setattr(zs, 'compute_adiabatic_temperature', adiabat)
    r = tool.solve(case, solver, **({'tolerance_outer': tol} if tol else {}))
    assert r['converged'] and r['adiabat_blend'] == 1.0
    rest, kwargs = args[-1]
    fresh = real(r['radii'], r['pressure'], r['mass_enclosed'], *rest, **kwargs)
    lag = _mantle_lag(r, fresh)
    i_cmb = int(np.searchsorted(r['mass_enclosed'], r['cmb_mass']))
    dT_cmb = abs(r['temperature'][i_cmb] - tool.PROTEUS_IC[case]['cmb_temperature'])
    print(
        case, solver, f'CMB {dT_cmb:.3f} K, top {lag[-1]:.3f} K, other {lag[1:-1].max():.3f} K'
    )
    assert lag[1:-1].max() <= 1.0
    # the core nodes carry T_surf, so the one T(P) table jumps by T_cmb - T_surf at the CMB
    assert dT_cmb <= 6.0, f'CMB anchor node {dT_cmb:.3f} K off T_cmb: core/mantle T jump'
    # the adiabat skips its step where a pressure is below 1e5 Pa
    assert lag[-1] <= 25.0, f'top node {lag[-1]:.3f} K off the adiabat: low-pressure skip'
    assert r['adiabat_table_blend'] == 1.0
    assert r['adiabat_self_consistency'] == pytest.approx(lag.max(), rel=1e-12)
    assert r['adiabat_self_consistency'] >= max(dT_cmb, lag[-1])


def _fake_adiabat(pressure, t_surf):
    """The adiabat of ``_analytic_ramp``: t_surf plus 2000 K times sqrt(P / 400 GPa)."""
    return t_surf + 2000.0 * np.sqrt(np.clip(np.asarray(pressure, float), 0.0, None) / 4e11)


def _analytic_ramp(
    linear=None, factor=None, clock_after=None, drift=0.0, mass=None, seed=None, **kwargs
):
    """_solve on analytic densities through the adiabatic path: the EOS counts as T-dependent
    and the adiabat is ``_fake_adiabat`` of the pressure. Passes before the ramp starts end
    ``linear`` times the target mass and passes after it ``factor`` times (unscaled when None;
    a callable takes the number of adiabat computations so far, 4 on the first pass at blend
    1). With ``clock_after`` the clock runs out once the adiabat was computed that many times.
    ``drift`` scales the pressure of each pass by 1 + drift times the number of adiabat
    computations so far. ``mass(R, k)`` replaces both scalings: the total mass of a pass at
    outer radius R after k adiabat computations. ``seed`` goes to ``_solve`` as
    ``adiabat_seed``. Returns the result and the number of adiabat computations."""
    adiabat, real, clock = [], zs.solve_structure, time.time

    def adi(radii, pressure, mass, t_surf, *args, **kw):
        adiabat.append(1)
        return _fake_adiabat(pressure, t_surf)

    def solve(*args, **kw):
        m, g, P = real(*args, **kw)
        scale = (factor(len(adiabat)) if callable(factor) else factor) if adiabat else linear
        f = scale * earth_mass / m[-1] if scale else 1.0
        if mass is not None:
            f = mass(args[3][-1], len(adiabat)) / m[-1]
        return m * f, g * f, P * (1.0 + drift * len(adiabat))

    cfg = _cfg(
        temperature_mode='adiabatic',
        relative_tolerance=1e-5,
        absolute_tolerance=1e-6,
        num_layers=30,
        **kwargs,
    )
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zs, 'any_component_is_tdep', lambda mixes: True)
        mp.setattr(zs, 'compute_adiabatic_temperature', adi)
        mp.setattr(zs, 'solve_structure', solve)
        if clock_after:
            stop = lambda: clock() + 1e9 * (len(adiabat) >= clock_after)  # noqa: E731
            mp.setattr(zs, 'time', SimpleNamespace(time=stop))
        seeded = {} if seed is None else {'adiabat_seed': seed}
        r = zs._solve(cfg, load_material_dictionaries(), None, '/tmp', **seeded)
    return r, len(adiabat)


@pytest.mark.smoke
def test_a_linear_floor_above_a_tight_tolerance_starts_the_ramp_unaccepted_on_a_clock_stop():
    """tolerance_outer 1e-7 and every linear pass 5e-4 off the mass: the blend ramp starts at
    the 1e-3 trigger, and a clock stop once the adiabat was computed twice ends the solve in
    the ramp, which is not accepted and keeps no met mass test."""
    r, n_adiabat = _analytic_ramp(
        linear=1.0005,
        clock_after=2,
        tolerance_outer=1e-7,
        max_iterations_outer=40,
        wall_timeout=WALL,
    )
    assert n_adiabat >= 2 and r['stopped_on_wall_clock'] is True
    assert r['adiabat_blend'] < 1.0
    assert r['converged'] is False and r['converged_mass'] is False
    assert r['accepted_on'] is None and r['accept_mass_limit'] is None


def _mantle_lag(r, adiabat):
    """|T - adiabat| at the mantle nodes of ``r`` with a positive pressure."""
    m, P = np.asarray(r['mass_enclosed']), np.asarray(r['pressure'])
    keep = (m >= r['cmb_mass']) & (P > 0)
    return np.abs(np.asarray(r['temperature']) - adiabat)[keep]


@pytest.mark.smoke
def test_the_accept_waits_for_a_table_built_on_a_full_adiabat_pass():
    """Every pass on the target mass: the first pass at blend 1 reads the T table built on the
    pressure of the blend-0.75 pass, so the solve accepts the next pass, whose table comes
    from a pass at blend 1. The adiabat is computed for the 5 passes of the ramp and once
    more on the returned structure, for adiabat_self_consistency. The pressure drifts from
    pass to pass, so the returned T, read from the table of the pass before, lags that
    adiabat."""
    r, n_adiabat = _analytic_ramp(
        linear=1.0, factor=1.0, drift=1e-4, tolerance_outer=1e-3, wall_timeout=WALL
    )
    assert n_adiabat == 6
    assert r['converged'] is True and r['accepted_on'] == 'strict'
    assert (r['adiabat_blend'], r['adiabat_table_blend']) == (1.0, 1.0)
    lag = _mantle_lag(r, _fake_adiabat(r['pressure'], _cfg()['surface_temperature']))
    assert lag.max() > 0.0
    assert r['adiabat_self_consistency'] == pytest.approx(lag.max(), rel=1e-12)


@pytest.mark.smoke
@pytest.mark.usefixtures('no_check')
def test_an_iteration_cap_on_the_first_full_adiabat_pass_is_not_accepted():
    """The iteration cap ends the solve on the first pass at blend 1, on the target mass: its
    T table comes from the blend-0.75 pass, so the pass is returned unaccepted."""
    r, _ = _analytic_ramp(
        linear=1.0, factor=1.0, tolerance_outer=1e-3, max_iterations_outer=5, wall_timeout=WALL
    )
    assert r['converged'] is False and r['accepted_on'] is None
    assert (r['adiabat_blend'], r['adiabat_table_blend']) == (1.0, 0.75)


@pytest.mark.smoke
@pytest.mark.usefixtures('no_check')
def test_a_pass_below_blend_1_reports_no_self_consistency():
    """The iteration cap ends the solve on the blend-0.75 pass, on the target mass: its T is
    not the adiabat, so adiabat_self_consistency is None."""
    r, _ = _analytic_ramp(
        linear=1.0, factor=1.0, tolerance_outer=1e-3, max_iterations_outer=4, wall_timeout=WALL
    )
    assert r['converged'] is False and r['adiabat_blend'] == 0.75
    assert np.all(np.asarray(r['pressure']) > 0) and r['adiabat_self_consistency'] is None


@pytest.mark.smoke
@pytest.mark.usefixtures('no_check')
def test_an_iteration_cap_takes_the_best_pass_on_a_full_adiabat_table():
    """The first pass at blend 1 (table of blend 0.75) ends 0.01 % off the mass, the passes on
    a table of blend 1 0.5 % off: the iteration cap accepts the best pass of the newest step,
    blend 1 on a table of blend 1, not the closer pass on the older table."""
    r, _ = _analytic_ramp(
        linear=1.0,
        factor=lambda k: 1.0001 if k == 4 else 1.005,
        tolerance_outer=1e-3,
        max_iterations_outer=8,
        wall_timeout=WALL,
    )
    assert r['converged'] is True and r['accepted_on'] == 'max_iter'
    assert (r['adiabat_blend'], r['adiabat_table_blend']) == (1.0, 1.0)
    assert r['mass_error_at_accept'] == pytest.approx(0.005, rel=1e-6)


@pytest.mark.smoke
@pytest.mark.usefixtures('no_check')
def test_an_earlier_best_pass_on_an_older_table_is_named_and_refused(caplog):
    """The passes on a table of blend 1 end 5 % off, above every accept limit of the iteration
    cap; the first pass at blend 1 ends 0.01 % off. The stop returns that pass, the warning
    names the blend and the table blend of both passes, and the pass is not accepted."""
    with caplog.at_level(logging.WARNING, logger='fwl.zalmoxis.solver'):
        r, _ = _analytic_ramp(
            linear=1.0008,
            factor=lambda k: 1.0001 if k == 4 else 1.05,
            tolerance_outer=1e-3,
            max_iterations_outer=8,
            wall_timeout=WALL,
        )
    assert 'No pass of adiabat blend 1.00 on a table of blend 1.00 passes' in caplog.text
    assert 'returning the best pass of blend 1.00 on a table of blend 0.75' in caplog.text
    assert r['converged'] is False and r['accepted_on'] is None
    assert (r['adiabat_blend'], r['adiabat_table_blend']) == (1.0, 0.75)


@pytest.mark.smoke
def test_a_newton_probe_in_an_adiabatic_mode_refuses_its_linear_pass_at_debug(caplog):
    """A Newton probe (one linear pass, near the root) is not accepted in an adiabatic mode;
    the refusal is logged at DEBUG, as the other progress lines of a probe, and no accept of
    that pass is logged as a warning."""
    with caplog.at_level(logging.DEBUG, logger='fwl.zalmoxis.solver'):
        r, n_adiabat = _analytic_ramp(
            max_iterations_outer=1,
            wall_timeout=WALL,
            _newton_probe=True,
            _initial_radius_guess=6.12e6,
        )
    assert r['converged'] is False and n_adiabat == 0
    refusals = [x for x in caplog.records if 'No accept of a pass' in x.getMessage()]
    assert refusals and all(x.levelno == logging.DEBUG for x in refusals)
    accepts = [x for x in caplog.records if x.getMessage().startswith('Accepting')]
    assert all(
        x.levelno < logging.WARNING for x in accepts
    )  # no accept warning for a refused pass


_TOLERANCES = st.one_of(
    st.floats(1e-12, 0.1),
    st.sampled_from([float('nan'), float('inf'), 0.0, -1e-3, True, False]),
)


@pytest.mark.smoke
@pytest.mark.accept_solves_long
@pytest.mark.timeout(4800)
@settings(max_examples=3, deadline=None, derandomize=True)
@given(tol=_TOLERANCES, n=st.integers(1, 9), factor=st.sampled_from([None, 1.02, 1.05, 1.1]))
@example(tol=1e-3, n=8, factor=None)  # the cap ends on blend 1 with the table of blend 0.75
@example(tol=float('nan'), n=9, factor=None)  # max_iter on a full-adiabat table
@example(tol=1e-7, n=9, factor=1.02)  # 2 % off the mass, max_iter on a full-adiabat table
def test_no_tolerance_or_pass_cap_accepts_a_profile_short_of_the_adiabat(tol, n, factor):
    """Analytic densities on the adiabatic path (the EOS counts as T-dependent and the adiabat
    is a function of the pressure), passes after the ramp starts scaled to ``factor`` times the
    mass, and no reachable wall clock: for any tolerance_outer, including NaN, inf, 0, a
    negative value and a bool, and any max_iterations_outer from 1 to 9 (unscaled passes reach
    blend 1 on a full-adiabat table at pass 9), an accepted result is the full adiabat on a
    table of a full-adiabat pass."""
    r, _ = _analytic_ramp(
        factor=factor, tolerance_outer=tol, max_iterations_outer=n, wall_timeout=WALL
    )
    full = r['adiabat_computed'] and r['adiabat_blend'] == 1.0
    assert not r['converged'] or (full and r['adiabat_table_blend'] == 1.0)


@pytest.mark.smoke
def test_a_loose_tolerance_without_a_clock_limit_accepts_the_full_adiabat():
    """tolerance_outer 1e-2, 50 passes and no reachable wall clock on analytic densities
    through the adiabatic path: the solve ramps to the adiabat and accepts it."""
    r, n_adiabat = _analytic_ramp(
        tolerance_outer=1e-2, max_iterations_outer=50, wall_timeout=WALL
    )
    assert r['converged'] is True and r['accepted_on'] == 'strict'
    assert r['adiabat_computed'] is True and r['adiabat_blend'] == 1.0 and n_adiabat >= 4


R_ROOT = 6.4e6  # [m] outer radius where the power-law mass of the secant tests meets the target


def _power_mass(R, k):
    """Total mass [kg] of a pass at outer radius R after k adiabat computations: earth_mass
    (R / 6.4e6 m)^0.9, 0.3 % lower per blend step up to blend 1. The exponent 0.9 makes the
    cube-root update shrink the mass error by 0.7 per pass, as on the cmf 0.95 PROTEUS grid."""
    return earth_mass * (R / R_ROOT) ** 0.9 * (1.0 - 0.003 * min(k, 4))


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(600)
def test_the_secant_tail_reaches_the_mass_in_3_full_adiabat_passes():
    """A ramp started by main (``_secant_tail``) on a mass that grows as R^0.9: the passes on a
    full-adiabat table reach tolerance_outer 1e-4 in 3 passes (the cube-root update alone
    takes 13 on this mass) and the solve accepts the full adiabat. The adiabat is computed
    on the 4 blend passes, on each full-table pass and once on the returned structure."""
    r, n_adiabat = _analytic_ramp(
        mass=_power_mass,
        tolerance_outer=1e-4,
        max_iterations_outer=40,
        wall_timeout=WALL,
        _secant_tail=True,
        _initial_radius_guess=R_ROOT,
    )
    assert r['converged'] is True and r['accepted_on'] == 'strict'
    assert (r['adiabat_blend'], r['adiabat_table_blend']) == (1.0, 1.0)
    assert n_adiabat - 5 <= 3 and r['mass_error_at_accept'] < 1e-4
    R_star = R_ROOT * (1.0 - 0.012) ** (-1.0 / 0.9)  # the root at blend 1
    assert r['radii'][-1] == pytest.approx(R_star, rel=1e-4 / 0.9 * 1.01)


def _secant_calls(monkeypatch, tol):
    """Record (mass error in tolerances, real secant step) of each _secant_radius call."""
    calls, real = [], zs._secant_radius

    def spy(R, M, R_prev, M_prev, M_target, R_fallback):
        got = real(R, M, R_prev, M_prev, M_target, R_fallback)
        calls.append(
            (pytest.approx(abs(M / M_target - 1.0) / tol, rel=1e-6), got != R_fallback)
        )
        return got

    monkeypatch.setattr(zs, '_secant_radius', spy)
    return calls


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.usefixtures('no_check')
@pytest.mark.timeout(900)
def test_the_secant_steps_above_2_tolerances_until_a_secant_step_does_not_shrink_the_error(
    monkeypatch,
):
    """The passes on a full-adiabat table end 5, 1.5, 5, 5.5, 4, 6 and 5 tolerances below the
    mass, whatever the radius. The first pass has no earlier full-table pass and the second is
    below the floor of 2 tolerances. Passes 3 and 4 have a mass that falls as the radius grows,
    so the secant falls back to the cube-root rule, and a fallback does not end the tail. Pass
    5 takes a real secant step; pass 6 ends 6 tolerances off, so the tail ends there."""
    tol = 1e-4
    off = {5: -5.0, 6: -1.5, 7: -5.0, 8: -5.5, 9: -4.0, 10: -6.0, 11: -5.0}
    calls = _secant_calls(monkeypatch, tol)
    r, n_adiabat = _analytic_ramp(
        linear=1.0,
        factor=lambda k: 1.0 + off.get(k, 0.0) * tol,
        tolerance_outer=tol,
        max_iterations_outer=12,
        wall_timeout=WALL,
        _secant_tail=True,
    )
    assert n_adiabat - 5 == 7 and r['adiabat_table_blend'] == 1.0
    assert calls == [(5.0, False), (5.5, False), (4.0, True)]


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.usefixtures('no_check')
@pytest.mark.timeout(900)
def test_a_ramp_without_the_secant_key_takes_no_secant_step(monkeypatch):
    """A Picard ramp that main did not start (no ``_secant_tail``) keeps the cube-root rule:
    its full-adiabat passes 5, 5 and 4 tolerances off the mass call no secant."""
    tol = 1e-4
    off = {5: -5.0, 6: -5.0, 7: -4.0}
    calls = _secant_calls(monkeypatch, tol)
    r, n_adiabat = _analytic_ramp(
        linear=1.0,
        factor=lambda k: 1.0 + off.get(k, 0.0) * tol,
        tolerance_outer=tol,
        max_iterations_outer=8,
        wall_timeout=WALL,
    )
    assert n_adiabat - 5 == 3 and r['adiabat_table_blend'] == 1.0 and calls == []


@pytest.fixture(scope='module')
def ramp_seed():
    """The structure of a converged ramp on analytic densities (every pass on the target
    mass), as an adiabat_seed."""
    r, _ = _analytic_ramp(linear=1.0, factor=1.0, tolerance_outer=1e-3, wall_timeout=WALL)
    assert r['converged'] and r['adiabat_table_blend'] == 1.0
    return {k: np.array(r[k]) for k in ('radii', 'pressure', 'mass_enclosed')}


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.timeout(600)
def test_a_seeded_solve_accepts_its_second_pass(ramp_seed):
    """Seeded from an earlier structure, the solve starts at blend 1 on the adiabat of the
    seed: its first pass is on the target mass but its table is not from a pass of this
    solve, so the second pass, on the table of the first, is accepted. The adiabat is
    computed for the 2 passes and once on the returned structure."""
    r, n_adiabat = _analytic_ramp(
        linear=1.0, factor=1.0, tolerance_outer=1e-3, wall_timeout=WALL, seed=ramp_seed
    )
    assert r['converged'] is True and r['accepted_on'] == 'strict'
    assert (r['adiabat_blend'], r['adiabat_table_blend']) == (1.0, 1.0)
    assert r['adiabat_seeded'] is True and n_adiabat == 3


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.usefixtures('no_check')
@pytest.mark.timeout(600)
def test_the_first_seeded_pass_carries_the_seed_label_and_is_not_accepted(ramp_seed):
    """An iteration cap of 1 ends a seeded solve on its first pass, on the target mass: its
    table is labelled -1.0 (from the seed), so the pass is returned unaccepted. The adiabat is
    computed for that pass and once on the returned structure."""
    r, n_adiabat = _analytic_ramp(
        linear=1.0,
        factor=1.0,
        tolerance_outer=1e-3,
        max_iterations_outer=1,
        wall_timeout=WALL,
        seed=ramp_seed,
    )
    assert r['converged'] is False and r['accepted_on'] is None and n_adiabat == 2
    assert (r['adiabat_blend'], r['adiabat_table_blend']) == (1.0, -1.0)
    assert r['adiabat_seeded'] is True


@pytest.mark.smoke
@pytest.mark.accept_solves
@pytest.mark.usefixtures('no_check')
@pytest.mark.timeout(600)
def test_the_seed_table_pass_starts_no_secant(monkeypatch, ramp_seed):
    """A seeded solve with the secant tail whose passes end 5, 4 and 3 tolerances below the
    mass: the first pass reads the seed table, so the second is the first full-adiabat pass and
    the secant starts only at the third."""
    tol = 1e-4
    off = {1: -5.0, 2: -4.0, 3: -3.0}
    calls = _secant_calls(monkeypatch, tol)
    r, n_adiabat = _analytic_ramp(
        linear=1.0,
        factor=lambda k: 1.0 + off.get(k, 0.0) * tol,
        tolerance_outer=tol,
        max_iterations_outer=3,
        wall_timeout=WALL,
        seed=ramp_seed,
        _secant_tail=True,
    )
    assert r['adiabat_seeded'] is True and n_adiabat == 4
    assert [c[0] for c in calls] == [3.0]


def _grid_main(cfg, **kwargs):
    """main on a grid setup; returns the result, the (R, M) of each pass of the solve that
    returned (consecutive solve_structure calls at one outer radius make one pass), the number
    of structure integrations, and the lag |T - adiabat(P)| at the CMB node, the largest at the
    other mantle nodes and at the top node, from the last adiabat computation (that of the
    returned structure)."""
    calls, starts, last = [], [], {}
    real_ss, real_solve, real_adi = (
        zs.solve_structure,
        zs._solve,
        zs.compute_adiabatic_temperature,
    )

    def spy(*a, **k):
        res = real_ss(*a, **k)
        calls.append((float(a[3][-1]), float(np.asarray(res[0])[-1])))
        return res

    def adi(*a, **k):
        last['T'] = np.asarray(real_adi(*a, **k))
        return last['T']

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zs, 'solve_structure', spy)
        mp.setattr(zs, 'compute_adiabatic_temperature', adi)
        mp.setattr(
            zs, '_solve', lambda *a, **k: starts.append(len(calls)) or real_solve(*a, **k)
        )
        r = grid_ref.solve(cfg, **kwargs)
    passes = []
    for R, M in calls[starts[-1] :]:
        passes = passes[:-1] + [(R, M)] if passes and passes[-1][0] == R else passes + [(R, M)]
    m, P = np.asarray(r['mass_enclosed']), np.asarray(r['pressure'])
    lag = np.abs(np.asarray(r['temperature']) - last['T'])[(m >= r['cmb_mass']) & (P > 0)]
    assert lag.max() == pytest.approx(r['adiabat_self_consistency'], rel=1e-12)
    return r, passes, len(calls), (lag[0], lag[1:-1].max(), lag[-1])


def _off_ref(r, ref):
    """Largest mantle T difference [K] on 39 mass points from the lowest to the second-highest
    mantle node (mantle nodes with P > 0 of each), the T difference at the top mantle node
    against ref interpolated in mass, and the relative differences of the outer radius, the
    central pressure, the core radius and the CMB pressure, both interpolated in mass at
    cmb_mass."""
    keep = [
        (np.asarray(x['mass_enclosed']) >= x['cmb_mass']) & (np.asarray(x['pressure']) > 0)
        for x in (r, ref)
    ]
    m = [np.asarray(x['mass_enclosed'])[k] for x, k in zip((r, ref), keep)]
    T = [np.asarray(x['temperature'])[k] for x, k in zip((r, ref), keep)]
    lo, hi = max(m[0][0], m[1][0]), min(m[0][-2], m[1][-2])
    q = lo + np.linspace(0.0, 1.0, 41)[1:-1] * (hi - lo)
    dT = float(np.max(np.abs(np.interp(q, m[0], T[0]) - np.interp(q, m[1], T[1]))))
    core = [
        [np.interp(x['cmb_mass'], x['mass_enclosed'], x[k]) for k in ('radii', 'pressure')]
        for x in (r, ref)
    ]
    return (
        dT,
        float(T[0][-1] - np.interp(m[0][-1], m[1], T[1])),
        r['radii'][-1] / ref['radii'][-1] - 1.0,
        r['pressure'][0] / ref['pressure'][0] - 1.0,
        core[0][0] / core[1][0] - 1.0,
        core[0][1] / core[1][1] - 1.0,
    )


def _ramp_figure(runs):
    """Relative mass error per outer pass of the first solve, the re-solve without and with the
    seed (``runs``: (passes, planet mass) each), with the mass tolerance and the secant floor;
    asserts that no legend or axis text box crosses a line vertex or another text box."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    for (passes, M), label, mark, ls in zip(
        runs,
        ('first solve', 're-solve without seed', 're-solve with seed'),
        ('o', 's', '^'),
        ('-', '--', '-'),
    ):
        err = [max(abs(m / M - 1.0), 1e-7) for _, m in passes]
        mfc = 'none' if mark == 's' else None
        ax.semilogy(range(1, len(err) + 1), err, ls=ls, marker=mark, ms=5, mfc=mfc, label=label)
    c = plt.rcParams['text.color']
    ax.axhline(1e-4, ls='--', color=c, alpha=0.5, label='mass tolerance')
    ax.axhline(2e-4, ls=':', color=c, alpha=0.5, label='secant floor')
    ax.minorticks_off()
    ax.set_xlabel('Outer pass')
    ax.set_ylabel('Relative mass error')
    legend = fig.legend(loc='lower center', ncol=2, fontsize=plt.rcParams['axes.labelsize'])
    fig.tight_layout(rect=(0, 0.3, 1, 1))
    fig.canvas.draw()
    rnd = fig.canvas.get_renderer()
    texts = [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels()]
    boxes = [legend.get_window_extent(rnd)] + [
        t.get_window_extent(rnd) for t in texts if t.get_text()
    ]
    pts = np.vstack([ln.get_transform().transform(ln.get_xydata()) for ln in ax.get_lines()])
    for k, b in enumerate(boxes):
        inside = (
            (pts[:, 0] > b.x0) & (pts[:, 0] < b.x1) & (pts[:, 1] > b.y0) & (pts[:, 1] < b.y1)
        )
        assert not inside.any(), f'text box {k} crosses a line vertex'
    assert not any(boxes[0].overlaps(b) for b in boxes[1:]), 'the legend crosses an axis text'
    return fig


@pytest.mark.smoke
@pytest.mark.accept_solves_long
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.timeout(7200)
def test_the_secant_tail_and_the_seed_keep_the_converged_adiabat_of_the_grid_start_up(request):
    """The first solve and the first re-solve of the PROTEUS grid case with core mass fraction
    0.95, against the same solves at a mass tolerance of 1e-6 (REF, stored in
    ``tests/data/grid_start_up_reference.json``; an EOS table with another md5 stops the test,
    and the slow reference test solves REF again): the first solve (Newton
    root, ramp with the secant tail), the
    seeded re-solve and the re-solve without the seed keep the mantle temperature within 1 K of
    REF from the lowest to the second-highest of the 12 mantle nodes, the radius, central
    pressure and core radius within 2e-4 and the CMB pressure within 2 tol / (1 - cmf) = 4e-3:
    the core mass is fixed, so the mass error sits in the mantle and moves P_cmb by
    dM / (M (1 - cmf)). The top mantle node stays within 1 K of REF in the re-solves (51.3 MPa);
    in the first solve (1 bar) it is not compared, since its adiabat step flips with round-off
    between 2 states (about -0.15 K or 28.65 to 28.81 K from REF).
    Each solve lags the adiabat of its own pressure by at most 6 K at the CMB node and 1 K at
    the other mantle nodes, and the re-solves by at most 25 K at the top node. The seeded and
    the unseeded re-solve differ by at most 2.5 K at the 39 mass points and at the top node,
    and 5e-4 in radius. The secant tail needs at most 3 passes on a full-adiabat table, and
    the seeded re-solve at most 4 passes and at most half the structure integrations of the
    unseeded one. With
    ``--validation-plots`` it writes the plot of the mass error per pass; the pass count can
    change with round-off, so a run without the option does not compare it with the committed
    plot. See ``docs/Validation/hydrostatic_structure.md``."""
    _need_paleos()
    first, p1, _, lag1 = _grid_main(GRID_CMF95)
    stored = grid_ref.load()
    ref1, ref2, resolve = stored['first'], stored['resolve'], grid_ref.CASES['resolve']
    warm = dict(
        initial_density=first['density'],
        initial_radii=first['radii'],
        p_center_hint=float(first['pressure'][0]),
    )
    seed = {k: first[k] for k in ('radii', 'pressure', 'mass_enclosed')}
    seeded, p2, n_seeded, lag2 = _grid_main(resolve, adiabat_seed=seed, **warm)
    plain, p3, n_plain, lag3 = _grid_main(resolve, **warm)
    for r, flag in (
        (first, False),
        (seeded, True),
        (plain, False),
    ):
        assert r['converged'] and r['accepted_on'] == 'strict' and r['adiabat_seeded'] is flag
        assert (r['adiabat_blend'], r['adiabat_table_blend']) == (1.0, 1.0)
    off = [
        (_off_ref(first, ref1), np.inf),  # 1-bar top node: 2-state adiabat step flip
        (_off_ref(seeded, ref2), 1.0),
        (_off_ref(plain, ref2), 1.0),
    ]
    print(
        'vs REF (dT, dT_top, dR, dPc, dRc, dPcmb):',
        off,
        'seeded vs plain:',
        _off_ref(seeded, plain),
    )
    print('lags (CMB node, mantle, top node) [K]:', lag1, lag2, lag3)
    p_cmb_bound = 2 * GRID_CMF95['newton_tol'] / (1.0 - GRID_CMF95['core_mass_fraction'])
    for (dT, dT_top, dR, dPc, dRc, dPcmb), top_bound in off:
        assert dT <= 1.0 and abs(dT_top) <= top_bound
        assert abs(dR) <= 2e-4 and abs(dPc) <= 2e-4
        assert abs(dRc) <= 2e-4 and abs(dPcmb) <= p_cmb_bound
    for lag in (lag1, lag2, lag3):
        assert lag[0] <= 6.0 and lag[1] <= 1.0
    assert lag2[2] <= 25.0 and lag3[2] <= 25.0
    dT, dT_top, dR, *_ = _off_ref(seeded, plain)
    assert dT <= 2.5 and abs(dT_top) <= 2.5 and abs(dR) <= 5e-4
    assert len(p1) - 5 <= 3  # linear pass, 4 blend passes, then the full-adiabat passes
    assert len(p2) <= 4 and n_seeded <= 0.5 * n_plain

    def figure():
        return _ramp_figure(
            [
                (p1, GRID_CMF95['planet_mass']),
                (p3, resolve['planet_mass']),
                (p2, resolve['planet_mass']),
            ]
        )

    if request.config.getoption('validation_plots') is not None:
        request.getfixturevalue('validation_plot')(figure, 'grid_start_up_ramp')


@pytest.mark.slow
@pytest.mark.accept_solves_reference
@pytest.mark.physics_invariant
@pytest.mark.timeout(3600)
def test_a_first_solve_at_a_planet_mass_2e12_higher_keeps_the_stored_reference():
    """The first solve of the grid start-up case at a planet mass 2e-12 higher, a round-off
    nudge that can flip the adiabat step of the top mantle node at 1 bar, keeps the bounds of
    the grid start-up test against the stored REF: strict on the full adiabat, mantle
    temperature within 1 K up to the second-highest mantle node (the 1-bar top node is not
    compared), radius, central pressure and core radius within 2e-4, CMB pressure within
    2 tol / (1 - cmf) = 4e-3, and lags of at most 6 K at the CMB node and 1 K at the other
    mantle nodes. It runs in the nightly job of the stored reference."""
    _need_paleos()
    nudged = {**GRID_CMF95, 'planet_mass': GRID_CMF95['planet_mass'] * (1 + 2e-12)}
    r, _, _, lag = _grid_main(nudged)
    assert r['converged'] and r['accepted_on'] == 'strict' and r['adiabat_seeded'] is False
    assert (r['adiabat_blend'], r['adiabat_table_blend']) == (1.0, 1.0)
    off = _off_ref(r, grid_ref.load()['first'])
    print('vs REF (dT, dT_top, dR, dPc, dRc, dPcmb):', off, 'lag:', lag)
    dT, _, dR, dPc, dRc, dPcmb = off  # 1-bar top node: 2-state adiabat step flip
    bound = 2 * GRID_CMF95['newton_tol'] / (1.0 - GRID_CMF95['core_mass_fraction'])
    assert dT <= 1.0
    assert max(abs(dR), abs(dPc), abs(dRc)) <= 2e-4 and abs(dPcmb) <= bound
    assert lag[0] <= 6.0 and lag[1] <= 1.0


@pytest.mark.slow
@pytest.mark.accept_solves_reference
@pytest.mark.physics_invariant
@pytest.mark.timeout(7800)
def test_the_stored_grid_start_up_reference_is_the_converged_solve_of_the_source():
    """Both structures of ``tests/data/grid_start_up_reference.json`` solved again at the mass
    tolerance 1e-6: strict on the full adiabat, the mantle temperature within 0.1 K of the
    stored one up to the second-highest mantle node, the top node within 1 K in the re-solve
    (in the first solve at 1 bar it is not compared: its adiabat step flips with round-off
    between 2 states), the radius,
    central pressure and core radius within 1e-5 and the CMB pressure within
    2 tol_REF / (1 - cmf) = 4e-5; then the sha256 of the stored list of Zalmoxis modules
    equals the stored one. A failure means the source or a table moved the reference: run
    ``python -m tools.validation.grid_start_up_reference`` and commit the file."""
    _need_paleos()
    import zalmoxis

    stored = grid_ref.load()
    for key, cfg in grid_ref.CASES.items():
        r = grid_ref.solve({**cfg, **grid_ref.REF_TOL})
        assert r['converged'] and r['accepted_on'] == 'strict', key
        assert (r['adiabat_blend'], r['adiabat_table_blend']) == (1.0, 1.0), key
        off = _off_ref(r, stored[key])
        print(key, 'vs stored (dT, dT_top, dR, dPc, dRc, dPcmb):', off)
        dT, dT_top, dR, dPc, dRc, dPcmb = off
        bound = 2 * grid_ref.REF_TOL['newton_tol'] / (1.0 - cfg['core_mass_fraction'])
        # 1-bar top node of the first solve: 2-state adiabat step flip, not compared
        assert dT <= 0.1 and (key == 'first' or abs(dT_top) <= 1.0), key
        assert max(abs(dR), abs(dPc), abs(dRc)) <= 1e-5 and abs(dPcmb) <= bound, key
    pkg = os.path.dirname(os.path.abspath(zalmoxis.__file__))
    assert grid_ref.source_hash(stored['source']['files'], pkg) == stored['source']['sha256']
