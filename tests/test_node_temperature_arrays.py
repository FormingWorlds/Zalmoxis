"""The node density update reads T from the profile the structure solve integrates."""

from __future__ import annotations

import os

import numpy as np
import pytest

import zalmoxis.solver as zs
from tests.test_failed_structure_solve import ROOT, _cfg
from zalmoxis.config import load_material_dictionaries

pytestmark = pytest.mark.unit
_SHORT = dict(max_iterations_outer=1, max_iterations_inner=1, _initial_radius_guess=6113601.77)


def _node_temperatures(monkeypatch, cfg, **main_kwargs):
    """Run main with numpy structure solves; return main's result and (r, T) of every node
    whose density was updated, with r taken from the solve that gave the node pressure."""
    solves, nodes = [], []
    real_solve, real_rho = zs.solve_structure, zs.calculate_mixed_density_batch

    def solve(*args, **kwargs):
        out = real_solve(*args, **dict(kwargs, use_jax=False))
        solves.append((np.asarray(args[3]), np.asarray(out[2])))
        return out

    def rho(pressure, temperature, *args, **kwargs):
        radii, profile = next(s for s in reversed(solves) if np.all(np.isin(pressure, s[1])))
        nodes.extend(zip(radii[np.searchsorted(-profile, -pressure)], temperature))
        return real_rho(pressure, temperature, *args, **kwargs)

    monkeypatch.setattr(zs, 'solve_structure', solve)
    monkeypatch.setattr(zs, 'calculate_mixed_density_batch', rho)
    mats = load_material_dictionaries()
    result = zs.main(cfg, mats, None, os.path.join(ROOT, 'input'), **main_kwargs)
    return result, np.array(nodes)


@pytest.mark.parametrize('outer_solver', ['picard', 'newton'])
def test_node_temperature_follows_temperature_arrays(monkeypatch, outer_solver):
    """With use_jax the solve integrates the arrays, so every node T is the arrays at r."""
    r_arr = np.linspace(0.0, 8e6, 41)
    T_arr = np.linspace(7000.0, 4000.0, 41)
    cfg = _cfg(outer_solver=outer_solver, use_jax=True, **_SHORT)
    result, nodes = _node_temperatures(monkeypatch, cfg, temperature_arrays=(r_arr, T_arr))
    assert len(nodes)
    np.testing.assert_array_equal(nodes[:, 1], np.interp(nodes[:, 0], r_arr, T_arr))
    np.testing.assert_array_equal(
        result['temperature'], np.interp(result['radii'], r_arr, T_arr)
    )


def test_node_temperature_follows_the_callable_without_jax(monkeypatch):
    """Without use_jax the solve integrates the callable, and so does the node update."""
    r_arr = np.linspace(0.0, 8e6, 41)

    def temperature(r, P):
        return 5000.0 + 1e-4 * r

    result, nodes = _node_temperatures(
        monkeypatch,
        _cfg(use_jax=False, **_SHORT),
        temperature_function=temperature,
        temperature_arrays=(r_arr, np.full(41, 4000.0)),
    )
    assert len(nodes)
    np.testing.assert_array_equal(nodes[:, 1], 5000.0 + 1e-4 * nodes[:, 0])
    np.testing.assert_array_equal(result['temperature'], 5000.0 + 1e-4 * result['radii'])


def test_temperature_arrays_replace_the_temperature_mode(monkeypatch):
    """The arrays take precedence over the mode: a missing prescribed profile is not read."""
    r_arr, T_arr = np.linspace(0.0, 8e6, 41), np.linspace(7000.0, 4000.0, 41)
    cfg = _cfg(
        use_jax=True, temperature_mode='prescribed', temp_profile_file='missing.txt', **_SHORT
    )
    result, nodes = _node_temperatures(monkeypatch, cfg, temperature_arrays=(r_arr, T_arr))
    np.testing.assert_array_equal(nodes[:, 1], np.interp(nodes[:, 0], r_arr, T_arr))


def test_temperature_arrays_skip_the_adiabat_blend(monkeypatch, caplog):
    """With the arrays giving T, mass convergence ends the outer loop without a blend ramp."""
    monkeypatch.setattr(zs, 'any_component_is_tdep', lambda *a, **k: True)
    r_arr, T_arr = np.linspace(0.0, 8e6, 41), np.linspace(7000.0, 4000.0, 41)
    cfg = _cfg(
        use_jax=True,
        temperature_mode='adiabatic',
        max_iterations_outer=6,
        max_iterations_inner=1,
        tolerance_outer=0.05,
    )
    with caplog.at_level('INFO', logger='zalmoxis'):
        _node_temperatures(monkeypatch, cfg, temperature_arrays=(r_arr, T_arr))
    assert 'Outer loop (total mass) converged' in caplog.text
    assert 'activating adiabat blend' not in caplog.text
