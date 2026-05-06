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

import pytest

from zalmoxis.constants import earth_mass
from zalmoxis.solver import _default_solver_params, _tighten_solver_params

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
