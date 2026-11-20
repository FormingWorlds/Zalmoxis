"""Solves with the core temperature table ``core_temperature_arrays``: the core nodes and the
core ODE read T(P) from the table, the mantle keeps its own source, on JAX and on numpy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tests._core_table_helpers import (
    CLOSURE,
    STEP,
    _assert_node_rule,
    _key,
    _layers,
    _mass_closure,
    _solve,
    _table,
)
from tests.test_jax_temperature_arrays import _INPUT_DIR, _chili_like_profile, _load_fixtures
from zalmoxis import solver as zs
from zalmoxis.constants import earth_mass
from zalmoxis.invariants import StructureInvariantError
from zalmoxis.structure_model import core_temperature

pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]


def test_a_table_of_none_is_the_solve_without_the_keyword():
    """core_temperature_arrays=None gives the solve without the keyword, bit for bit, with no
    core_T keys."""
    a, _ = _solve()
    b, _ = _solve(table=lambda t_cmb: None)
    for k in ('radii', 'pressure', 'mass_enclosed', 'density', 'temperature', 'gravity'):
        assert np.array_equal(a[k], b[k]), k
    assert 'core_T_used' not in a and 'core_T_used' not in b


@pytest.mark.physics_invariant
@pytest.mark.parametrize('jax', [True, False], ids=['jax', 'numpy'])
def test_core_nodes_take_the_table_and_mantle_nodes_the_mantle_source(jax):
    """T1: the first table point lies STEP K above the mantle T at the CMB; the returned core
    nodes carry table(P), the mantle nodes the mantle arrays, so the step shows at the CMB, and
    the core mass closes on the returned density (the core ODE read the table too)."""
    res, mantle = _solve(jax=jax, table=_table)
    _assert_node_rule(res, mantle, _table(_chili_like_profile()[3]))
    core, _, t = _layers(res)
    assert t[np.flatnonzero(core)[-1]] - t[np.flatnonzero(~core)[0]] > STEP
    assert abs(_mass_closure(res)) < CLOSURE
    assert res['core_T_used'] is True
    assert res['core_T_nodes_outside'] == 0 and res['core_T_max_excess'] == 0.0


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_jax_and_numpy_agree_with_the_table():
    """T3: the same inputs on 2 integrators and 2 T codes give R, P_c, P_cmb and rho_c within
    1e-4, at an ODE tolerance (1e-8) below the integration difference of the 2 paths."""
    ode = {'relative_tolerance': 1e-8, 'absolute_tolerance': 1e-9}
    a, _ = _solve(jax=True, table=_table, **ode)
    b, _ = _solve(jax=False, table=_table, **ode)
    assert _key(a) == pytest.approx(_key(b), rel=1e-4)


@pytest.mark.physics_invariant
def test_core_nodes_outside_the_table_take_the_end_slope_and_are_counted():
    """T5: a table that leaves the lowest 5 % of the core pressures out; the nodes there are
    extended with the end slope, and the result counts them and stays accepted."""
    full, _ = _solve()
    core, p, _ = _layers(full)
    lo, hi = p[core].min() + 0.05 * np.ptp(p[core]), p[core].max() * 1.01
    res, _ = _solve(table=lambda t_cmb: _table(t_cmb, lo, hi))
    core, p, t = _layers(res)
    table = _table(_chili_like_profile()[3], lo, hi)
    np.testing.assert_allclose(t[core], core_temperature(p[core], table), rtol=1e-9)
    out = core & ((p < lo) | (p > hi))
    excess = np.maximum(lo - p[out], p[out] - hi).max() / (hi - lo)
    assert out.sum() > 0 and res['core_T_nodes_outside'] == out.sum()
    assert res['core_T_max_excess'] == pytest.approx(excess, rel=1e-12)


def test_a_core_without_a_t_dependent_eos_ignores_the_table():
    """The unused table: an Analytic iron core reads no T, so the table leaves the solve
    unchanged and counts no node."""
    over = {'layer_eos_config': {'core': 'Analytic:iron', 'mantle': 'PALEOS-2phase:MgSiO3'}}
    a, _ = _solve(jax=False, **over)
    b, _ = _solve(
        jax=False, table=lambda t_cmb: (np.array([1e9, 2e9]), np.array([3e3, 3.1e3])), **over
    )
    for k in ('radii', 'pressure', 'mass_enclosed', 'density', 'temperature'):
        assert np.array_equal(a[k], b[k]), k
    assert b['core_T_used'] is False and b['core_T_nodes_outside'] == 0


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_a_table_of_the_core_t_of_an_r_indexed_run_gives_its_structure():
    """T4: r-indexed arrays that cover the core as well, continuous at the CMB of the arrays;
    a table of the core (P, T) of that run adds no T change at the nodes, so the run with it
    has the same R, P_c, P_cmb and rho_c within 1e-4, at an ODE tolerance (1e-8) below the
    integration difference of the 2 runs."""
    r_m, t_m, r_cmb, t_cmb = _chili_like_profile()
    r_c = np.linspace(0.0, r_cmb, 40, endpoint=False)
    arrays = (
        np.concatenate([r_c, r_m]),
        np.concatenate([t_cmb + 900.0 * (1.0 - (r_c / r_cmb) ** 2), t_m]),
    )
    cp, mat, mf = _load_fixtures()
    cp.update(use_jax=True, relative_tolerance=1e-8, absolute_tolerance=1e-9)
    a = zs.main(cp, mat, mf, _INPUT_DIR, temperature_arrays=arrays)
    core, p, t = _layers(a)
    table = (p[core][::-1], t[core][::-1])
    b = zs.main(
        cp, mat, mf, _INPUT_DIR, temperature_arrays=arrays, core_temperature_arrays=table
    )
    assert a['converged'] and b['converged'] and b['core_T_used'] is True
    assert _key(b) == pytest.approx(_key(a), rel=1e-4)


@pytest.mark.physics_invariant
def test_the_numpy_fallback_of_a_jax_solve_keeps_the_table(monkeypatch):
    """T6: every JAX structure solve raises, so main runs on numpy, which keeps the core table:
    the node rule holds and the core mass closes on the returned density."""
    import zalmoxis.jax_eos.wrapper as jw

    raised = []

    def fail(*args, **kwargs):
        raised.append(1)
        raise ValueError('forced JAX failure')

    monkeypatch.setattr(jw, 'solve_structure_via_jax', fail)
    res, mantle = _solve(jax=True, table=_table)
    assert raised and res['core_T_used'] is True
    _assert_node_rule(res, mantle, _table(_chili_like_profile()[3]))
    assert abs(_mass_closure(res)) < CLOSURE


def test_a_table_in_gpa_is_not_accepted(caplog):
    """A table with P in GPa puts every core node far above it; the solve converges on the
    end-slope extension but is not accepted, with a warning that names the table."""
    res, _ = _solve(table=lambda t_cmb: (_table(t_cmb)[0] / 1e9, _table(t_cmb)[1]), check=False)
    assert res['converged'] is False and res['accepted_on'] is None
    assert res['core_T_max_excess'] > zs._CORE_T_MAX_EXCESS
    assert any('Not accepted' in x.getMessage() for x in caplog.records)


def test_a_steep_table_that_goes_below_0_k_names_the_table():
    """The extension of a steep table below its range reaches T < 0 K at core nodes; the solve
    fails with an error that names the table and the node."""
    steep = (np.array([2.5e11, 3.5e11]), np.array([1000.0, 8000.0]))
    with pytest.raises(StructureInvariantError, match='core_temperature_arrays gives T ='):
        _solve(table=lambda t_cmb: steep, check=False)


@pytest.mark.physics_invariant
def test_the_core_adiabat_of_a_10_earth_mass_planet_is_accepted_as_a_table():
    """The core (P, T) of the 10 M_E planet on its own adiabat, with the mantle T of that solve
    as r-indexed arrays: the table is accepted and the core nodes carry it."""
    d = json.loads((Path(__file__).parent / 'validation/data/core_table_10me.json').read_text())
    table = (np.array(d['core_P_Pa']), np.array(d['core_T_K']))
    mantle = (np.array(d['mantle_r_m']), np.array(d['mantle_T_K']))
    cp, mat, mf = _load_fixtures()
    cp.update(use_jax=True, planet_mass=10 * earth_mass)
    res = zs.main(
        cp, mat, mf, _INPUT_DIR, temperature_arrays=mantle, core_temperature_arrays=table
    )
    assert res['converged'] and res['accepted_on'] is not None and res['core_T_used'] is True
    assert res['core_T_max_excess'] < zs._CORE_T_MAX_EXCESS
    _assert_node_rule(res, mantle, table)


@pytest.mark.physics_invariant
@pytest.mark.timeout(2400)  # Newton at rtol 1e-9: about 110 s on an Apple M1 Ultra
def test_the_newton_outer_solver_takes_the_table():
    """The Newton outer solver, the PROTEUS default, at the ODE tolerance it needs (1e-9) on
    the JAX path: the core nodes carry the table and the core mass closes."""
    ode = {'relative_tolerance': 1e-9, 'absolute_tolerance': 1e-10}
    res, mantle = _solve(table=_table, outer_solver='newton', **ode)
    assert res['core_T_used'] is True and 'newton_n_iter' in res
    _assert_node_rule(res, mantle, _table(_chili_like_profile()[3]))
    assert abs(_mass_closure(res)) < CLOSURE
