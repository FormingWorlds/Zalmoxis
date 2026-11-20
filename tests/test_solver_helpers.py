"""Unit tests for the small solver helpers.

``zalmoxis.solver`` exposes module-level helpers that the integration
tests cover only implicitly by running a full solve. These tests pin
their contracts directly so a regression in any one shows up at unit
tier instead of as a slow-suite divergence.

Covers:
- ``_default_solver_params(mass)`` mass-adaptive scaling: maximum_step
  scales linearly in r_est (Seager+2007 mass-radius), and
  max_center_pressure_guess is 10x the central-pressure scaling.
- ``_tighten_solver_params(params)`` doubles iteration caps + halves
  step + 10x tightens tolerances + 2x wall_timeout.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from zalmoxis.constants import earth_mass
from zalmoxis.solver import _default_solver_params, _layer_lags, _tighten_solver_params

pytestmark = pytest.mark.unit


class TestDefaultSolverParams:
    """``_default_solver_params`` mass-adaptive scaling."""

    def test_returns_required_keys(self):
        p = _default_solver_params(1.0 * earth_mass)
        for k in (
            'max_iterations_outer',
            'tolerance_outer',
            'max_iterations_inner',
            'tolerance_inner',
            'relative_tolerance',
            'absolute_tolerance',
            'maximum_step',
            'adaptive_radial_fraction',
            'max_center_pressure_guess',
            'pressure_tolerance',
            'max_iterations_pressure',
            'outer_solver',
        ):
            assert k in p, f'missing key {k!r}'

    def test_outer_solver_default_is_picard(self):
        assert _default_solver_params(1.0 * earth_mass)['outer_solver'] == 'picard'

    def test_maximum_step_scales_with_mass(self):
        """maximum_step is ~r_est * 0.004; r_est = R_e * (M/M_e)^0.282 so it
        grows with planet mass."""
        p1 = _default_solver_params(1.0 * earth_mass)
        p10 = _default_solver_params(10.0 * earth_mass)
        # 10 M_earth has roughly 10^0.282 ~= 1.92x larger radius -> step
        assert p10['maximum_step'] > p1['maximum_step']
        # and within a factor of ~3 (rules out a runaway scaling bug)
        assert p10['maximum_step'] < 3 * p1['maximum_step']

    def test_max_center_pressure_guess_scales_with_mass(self):
        """max_center_pressure_guess = 10 * earth_center_pressure * (M/M_e)^0.87."""
        p1 = _default_solver_params(1.0 * earth_mass)
        p10 = _default_solver_params(10.0 * earth_mass)
        # 10^0.87 ~= 7.4x heavier-mass scaling
        ratio = p10['max_center_pressure_guess'] / p1['max_center_pressure_guess']
        assert 5.0 < ratio < 10.0

    def test_low_mass_clamped_to_one_percent_earth(self):
        """Mass below 0.01 M_earth is clamped to that floor for scaling
        (prevents divide-by-zero in the scaling laws)."""
        p_tiny = _default_solver_params(0.001 * earth_mass)
        p_floor = _default_solver_params(0.01 * earth_mass)
        # Both should produce the same scaled values
        assert p_tiny['maximum_step'] == p_floor['maximum_step']
        assert p_tiny['max_center_pressure_guess'] == p_floor['max_center_pressure_guess']


class TestTightenSolverParams:
    """``_tighten_solver_params`` doubles iter caps + tightens tolerances."""

    def test_iteration_counts_doubled(self):
        params = _default_solver_params(1.0 * earth_mass)
        tight = _tighten_solver_params(params)
        assert tight['max_iterations_outer'] == 2 * params['max_iterations_outer']
        assert tight['max_iterations_inner'] == 2 * params['max_iterations_inner']
        assert tight['max_iterations_pressure'] == 2 * params['max_iterations_pressure']

    def test_tolerances_tightened_10x(self):
        params = _default_solver_params(1.0 * earth_mass)
        tight = _tighten_solver_params(params)
        assert tight['tolerance_outer'] == pytest.approx(0.1 * params['tolerance_outer'])
        assert tight['tolerance_inner'] == pytest.approx(0.1 * params['tolerance_inner'])
        assert tight['relative_tolerance'] == pytest.approx(0.1 * params['relative_tolerance'])
        assert tight['absolute_tolerance'] == pytest.approx(0.1 * params['absolute_tolerance'])
        assert tight['pressure_tolerance'] == pytest.approx(0.1 * params['pressure_tolerance'])

    def test_maximum_step_halved(self):
        params = _default_solver_params(1.0 * earth_mass)
        tight = _tighten_solver_params(params)
        assert tight['maximum_step'] == pytest.approx(0.5 * params['maximum_step'])

    def test_wall_timeout_doubled(self):
        params = _default_solver_params(1.0 * earth_mass)
        tight = _tighten_solver_params(params)
        # default wall_timeout is 300.0 when not present in params
        assert tight['wall_timeout'] == pytest.approx(2 * 300.0)

    def test_wall_timeout_doubled_from_explicit_value(self):
        params = _default_solver_params(1.0 * earth_mass)
        params['wall_timeout'] = 600.0
        tight = _tighten_solver_params(params)
        assert tight['wall_timeout'] == pytest.approx(1200.0)

    def test_does_not_mutate_input(self):
        """``_tighten_solver_params`` returns a new dict, leaves caller's
        dict unchanged."""
        params = _default_solver_params(1.0 * earth_mass)
        snapshot = dict(params)
        _ = _tighten_solver_params(params)
        for k in snapshot:
            assert params[k] == snapshot[k], f'mutation of key {k!r}'

    def test_keys_not_in_tighten_pass_through(self):
        """Keys not handled by the tightening logic survive unchanged."""
        params = _default_solver_params(1.0 * earth_mass)
        params['custom_thing'] = 'preserve_me'
        tight = _tighten_solver_params(params)
        assert tight['custom_thing'] == 'preserve_me'


class TestTimeoutAcceptThreshold:
    """The loose-accept mass-error ceiling and the warning that reports it.

    A structure solve that does not tighten to the requested tolerance is
    still accepted when its mass error sits below ``_TIMEOUT_ACCEPT_MASS_ERR``;
    the warning that announces this must report the same ceiling it gated on.
    Regression: the warning hard-coded 'threshold 2%' while the gate accepted
    below 3%, so a 2.5% solution was logged as if it had cleared a 2% bar. The
    gate and the message now read one shared constant so the reported threshold
    cannot drift from the gate. The accept branch lives inside the slow
    monolithic ``_solve`` driver, so this guards the gate-vs-message invariant
    at the source rather than through a multi-minute live solve.
    """

    def test_ceiling_is_a_small_positive_fraction(self):
        """The ceiling is a usable loose-accept fraction, not a fraction-vs-
        percent unit slip (3%, not 3.0 or 0.0003)."""
        from zalmoxis.solver import _TIMEOUT_ACCEPT_MASS_ERR

        assert 0.005 < _TIMEOUT_ACCEPT_MASS_ERR < 0.10
        # A wrong unit (percent written as 3.0, or per-mille as 0.0003) would
        # fall outside this band by more than an order of magnitude.
        assert _TIMEOUT_ACCEPT_MASS_ERR == pytest.approx(0.03, abs=1e-9)

    def test_gate_and_message_share_the_constant(self):
        """The accept gate and its warning both read the constant, and no
        stale numeric threshold literal survives in the message."""
        import inspect

        from zalmoxis import solver

        src = inspect.getsource(solver._solve)
        # The gate condition AND the warning format both read the constant
        # (>= 2 references: the ``elif`` test and the ``%`` argument).
        assert src.count('_TIMEOUT_ACCEPT_MASS_ERR') >= 2, src
        assert 'best_mass_error < _TIMEOUT_ACCEPT_MASS_ERR' in src
        # Discriminating guard: no baked-in 'threshold N%' literal survives in
        # the accept message that could drift from the gate (that hard-coded
        # mismatch, 'threshold 2%' against a 3% gate, is the regressed bug).
        assert 'threshold 2%' not in src
        assert 'threshold 3%' not in src


class _SolveReached(Exception):
    """Raised by the stub solve, so main stops after its checks."""


@pytest.mark.parametrize(
    ('ice', 'asked', 'fires'),
    [(True, True, True), (False, True, False), (True, False, False)],
    ids=['three_layers_adiabat', 'two_layers_adiabat', 'three_layers_no_adiabat'],
)
def test_main_warns_once_of_the_uncut_ice_boundary_on_an_adiabat(
    monkeypatch, caplog, ice, asked, fires
):
    """A solve that integrates the adiabat itself with an ice layer logs 1 warning that the
    boundary between the mantle and the ice layer is not cut, naming the known limit and its
    docs page; a 2-layer planet or a solve without the adiabat logs none."""
    from zalmoxis import solver

    def stop(*a, **k):
        raise _SolveReached

    monkeypatch.setattr(solver, '_adiabat_asked', lambda *a, **k: asked)
    monkeypatch.setattr(solver, '_solve', stop)
    monkeypatch.setattr(solver, '_solve_newton_outer', stop)
    eos = {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'}
    if ice:
        eos['ice_layer'] = 'Seager2007:H2O'
    cfg = {'layer_eos_config': eos, 'temperature_mode': 'adiabatic', 'planet_mass': 5.97e24}
    with (
        caplog.at_level(logging.WARNING, logger='zalmoxis.solver'),
        pytest.raises(_SolveReached),
    ):
        solver.main(cfg, {}, None, '/tmp')
    hits = [r for r in caplog.records if 'ice layer is not cut' in r.getMessage()]
    assert len(hits) == (1 if fires else 0)
    if fires:
        assert 'adiabat_integration.md' in hits[0].getMessage()


def test_the_layer_lags_split_the_nodes_by_the_core_rule():
    """_layer_lags gives the largest |T - adiabat| over the mantle and over the core nodes with
    P > 0. A node in the 1e-12 band below the CMB mass is a mantle node (``is_core``), as is a
    node at the CMB mass; a core node at P = 0 counts in no layer. Each node has its own
    difference, so a wrong mask returns another value."""
    cmb = 1.0e24
    m = np.array([0.0, 1e23, 2e23, cmb * (1 - 1e-13), cmb, 2 * cmb, 3 * cmb])
    p = np.array([5.0, 0.0, 4.0, 3.0, 2.0, 1.0, 0.0])
    adiabat = np.full(7, 1000.0)
    T = adiabat + np.array([1.0, -64.0, 2.0, 16.0, -8.0, 4.0, 32.0])
    assert _layer_lags(T, adiabat, m, p, cmb) == (16.0, 2.0)
    assert _layer_lags(T, adiabat, m, p, 0.0) == (16.0, None)  # no core: P > 0 nodes only
    assert _layer_lags(T, adiabat, m, np.zeros(7), cmb) == (None, None)
    T[0] = np.nan
    mantle, core = _layer_lags(T, adiabat, m, p, cmb)
    assert mantle == 16.0 and np.isnan(core)
