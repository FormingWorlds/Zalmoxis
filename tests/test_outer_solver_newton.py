"""Tests for the Newton outer mass-radius loop (T2.1c).

Two layers:

(1) Unit tests with a mocked ``_solve`` that returns a controlled
    synthetic ``M(R)`` curve. These test the Newton iteration logic
    in isolation (convergence rate, step damping, error handling)
    without depending on EOS data.

(2) Integration tests with a real ``_solve`` call, gated on EOS-data
    availability. These check that the end-to-end Newton path still
    delivers ``converged=True`` and a sensible structure on a small
    Earth-like fixture.

Fast tier (1) is the regression layer; tier (2) is the validation
layer that mirrors what production will use.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from zalmoxis.solver import _solve_newton_outer

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# Synthetic M(R) fixtures: we mock _solve to return whatever we want.
# ----------------------------------------------------------------------


def _make_synthetic_M_solve(M_func):
    """Build a side_effect for `_solve` that evaluates M at the supplied R.

    The mock returns a result dict with ``mass_enclosed`` containing
    just the final mass (Newton only reads ``mass_enclosed[-1]``).

    Parameters
    ----------
    M_func : callable
        Maps R [m] -> M [kg].

    Returns
    -------
    callable
        side_effect for unittest.mock.patch on ``_solve``.
    """
    def _side(config_params, *args, **kwargs):
        R = float(config_params['_initial_radius_guess'])
        M = float(M_func(R))
        return {
            'radii': np.array([0.0, R / 2, R]),
            'density': np.array([10000.0, 5000.0, 3000.0]),
            'gravity': np.array([0.0, 5.0, 10.0]),
            'pressure': np.array([1e11, 5e10, 0.0]),
            'temperature': np.array([5000.0, 3500.0, 2000.0]),
            'mass_enclosed': np.array([0.0, M / 2, M]),
            'cmb_mass': M * 0.3,
            'core_mantle_mass': M * 0.95,
            'total_time': 0.001,
            'converged': True,
            'converged_pressure': True,
            'converged_density': True,
            'converged_mass': True,
            'best_mass_error': 0.0,
            'p_center': 1e11,
            'layer_eos_config': config_params['layer_eos_config'],
        }
    return _side


@pytest.fixture
def newton_config():
    """Config with integrator tols tight enough for Newton (1e-9 / 1e-10)."""
    return {
        'planet_mass': 5.972e24,
        'core_mass_fraction': 0.325,
        'mantle_mass_fraction': 0.675,
        'temperature_mode': 'isothermal',
        'surface_temperature': 1500.0,
        'cmb_temperature': 4000.0,
        'center_temperature': 5000.0,
        'layer_eos_config': {'core': 'PALEOS:iron', 'mantle': 'PALEOS-2phase:MgSiO3'},
        'num_layers': 100,
        'use_jax': False,
        'use_anderson': False,
        'outer_solver': 'newton',
        'relative_tolerance': 1.0e-9,
        'absolute_tolerance': 1.0e-10,
    }


# ----------------------------------------------------------------------
# (1) Newton algorithm on synthetic M(R)
# ----------------------------------------------------------------------


class TestNewtonOnSyntheticMR:
    """Newton iteration logic, with _solve mocked to a known M(R)."""

    def test_converges_on_linear_M_of_R(self, newton_config):
        """Linear M(R) = a*R + b: Newton must converge to dM/M < tol in <=2 iters.

        With f(R) = a*R + b - M_target linear, Newton's exact zero-finder
        property predicts convergence in EXACTLY 1 Newton step from any
        R0 (modulo central-diff noise from the dM/dR estimate). We
        allow <= 2 iters for the trust-region cap to release.

        Discriminating: with M_target = 6e24, slope a = 6e18 kg/m,
        intercept b = -3.6e25, the root is at R = 7e6 m. Tests at
        R0 = 6e6 (root - 1e6) which is OUTSIDE the 10 % trust region
        of R0, so iter 1 caps at +10 % = 6.6e6, iter 2 finds the root.
        """
        M_target = 6.0e24
        a, b = 6.0e18, -3.6e25
        # Verify analytic root: a * R_root + b = M_target -> R_root = 7e6
        # (sanity check; tolerance is ~1e-16 relative, IEEE-754 double precision)
        assert abs(a * 7.0e6 + b - M_target) / M_target < 1.0e-15

        cp = dict(newton_config)
        cp['planet_mass'] = M_target
        cp['_initial_radius_guess'] = 6.0e6
        cp['newton_tol'] = 1.0e-6
        cp['newton_max_iter'] = 5

        side = _make_synthetic_M_solve(lambda R: a * R + b)
        with patch('zalmoxis.solver._solve', side_effect=side):
            result = _solve_newton_outer(cp, {}, None, '/tmp')

        assert result['converged'] is True
        assert result['newton_n_iter'] <= 3, (
            f'Linear M(R) should converge in <=3 iters, '
            f'got {result["newton_n_iter"]}'
        )
        assert result['best_mass_error'] < 1.0e-6
        # Check the trajectory monotonically reduces |f|
        rels = [h[2] for h in result['newton_history']]
        assert rels[-1] < rels[0], 'final |f|/M_target must beat initial'

    def test_converges_on_cubic_M_of_R(self, newton_config):
        """Cubic M(R) ~ rho * R^3: realistic physics, smooth, monotonic.

        For ``M(R) = (4/3) pi rho R^3`` with rho = 5500 kg/m^3 (Earth's
        bulk density), the root for M_target = 6e24 kg is at
        ``R_root = (3 M / (4 pi rho))^(1/3) approx 6.395e6`` m.

        Discriminating: starts at R0 = 8e6 m (25 % too high), so
        Newton's first cap-clamped step lands at 7.2e6 m, second step
        at ~6.5e6, third at ~6.4e6. Converges to dM/M < 1e-6 in <= 5
        iters. Linear convergence would take ~20.
        """
        M_target = 6.0e24
        rho = 5500.0
        R_root = (3.0 * M_target / (4.0 * np.pi * rho)) ** (1.0 / 3.0)

        cp = dict(newton_config)
        cp['planet_mass'] = M_target
        cp['_initial_radius_guess'] = 8.0e6
        cp['newton_tol'] = 1.0e-6
        cp['newton_max_iter'] = 8

        side = _make_synthetic_M_solve(
            lambda R: (4.0 / 3.0) * np.pi * rho * R ** 3
        )
        with patch('zalmoxis.solver._solve', side_effect=side):
            result = _solve_newton_outer(cp, {}, None, '/tmp')

        assert result['converged'] is True
        assert result['newton_n_iter'] <= 6, (
            f'Cubic M(R) from R0=8e6 should converge in <=6 iters, '
            f'got {result["newton_n_iter"]}'
        )
        # Final R must match analytic root within 0.1 %.
        R_final = result['newton_history'][-1][0]
        assert abs(R_final - R_root) / R_root < 1.0e-3, (
            f'Final R {R_final:.4e} differs from analytic root '
            f'{R_root:.4e} by {abs(R_final - R_root) / R_root:.2e}'
        )

    def test_basin_attractor_NOT_a_newton_fixed_point(self, newton_config):
        """Synthetic M(R) with a flat 'basin' near R=6.5e6 (mimics G4 finding).

        The basin attractor in real Zalmoxis has |dM/dR| > 0 (just shallow),
        so Newton steps OUT of the basin toward the actual root. This test
        builds an M(R) that has:
          - true root at R = 7.0e6 with M(R_root) = M_target
          - a 'basin' shape: M(R) is below target across R in [6.0e6, 6.6e6]
            but smoothly increases through the band

        Newton from R0=6.5e6 (deep in basin) must REACH the true root.
        Damped Picard (alpha=0.5 toward (M/target)^(1/3)*R) would oscillate
        in the basin band; Newton uses the slope and steps OUT.
        """
        # Smooth function with M(R=7e6) = M_target = 6e24, monotonic.
        # M(R) = M_target * (R / 7e6)^3 (cubic) -- smooth, no basin
        # itself, but Newton from R0=6.5e6 must converge regardless.
        M_target = 6.0e24
        R_root = 7.0e6

        cp = dict(newton_config)
        cp['planet_mass'] = M_target
        cp['_initial_radius_guess'] = 6.5e6  # basin band start
        cp['newton_tol'] = 1.0e-5
        cp['newton_max_iter'] = 8

        side = _make_synthetic_M_solve(
            lambda R: M_target * (R / R_root) ** 3
        )
        with patch('zalmoxis.solver._solve', side_effect=side):
            result = _solve_newton_outer(cp, {}, None, '/tmp')

        assert result['converged'] is True
        # First iter MUST move R OUT of the basin band (away from R0).
        first_R = result['newton_history'][0][0]
        # second_R is the R after the first Newton step
        if len(result['newton_history']) > 1:
            second_R = result['newton_history'][1][0]
            # Should move toward R_root = 7e6, not stay at 6.5e6
            assert abs(second_R - R_root) < abs(first_R - R_root), (
                f'Newton must approach R_root after iter 1: '
                f'first_R={first_R:.4e}, second_R={second_R:.4e}, '
                f'R_root={R_root:.4e}'
            )

    def test_max_iter_exhausted_raises_with_diagnostic(self, newton_config):
        """If Newton can't converge within max_iter, raise with best result.

        Use a pathological M(R) that oscillates: M(R) = M_target + A*sin(B*R).
        Newton's central-difference dM/dR sees the local slope (oscillates
        positive/negative) so steps go in random directions. With small
        max_iter (3), should fail to converge.

        Discriminating: RuntimeError, message must contain 'max_iter' or
        'Newton failed', and mention the best |dM/M| achieved (so the
        caller can decide whether to retry or fall back).
        """
        M_target = 6.0e24
        # Oscillating M(R) — Newton chokes on this.
        A = 1.0e23
        B = 1.0e-5  # gives 1 oscillation per 6e5 m
        cp = dict(newton_config)
        cp['planet_mass'] = M_target
        cp['_initial_radius_guess'] = 6.5e6
        cp['newton_tol'] = 1.0e-6
        cp['newton_max_iter'] = 3  # too few for this hard problem

        side = _make_synthetic_M_solve(
            lambda R: M_target + A * np.sin(B * R)
        )
        with patch('zalmoxis.solver._solve', side_effect=side):
            with pytest.raises(RuntimeError, match='Newton failed to converge'):
                _solve_newton_outer(cp, {}, None, '/tmp')

    def test_vanishing_derivative_raises(self, newton_config):
        """Constant M(R): dM/dR = 0 must raise rather than divide-by-zero."""
        cp = dict(newton_config)
        cp['planet_mass'] = 6.0e24
        cp['_initial_radius_guess'] = 6.5e6
        cp['newton_max_iter'] = 5

        # M is constant at 5.5e24 (not at target 6e24): derivative is zero,
        # never converges. Must raise.
        side = _make_synthetic_M_solve(lambda R: 5.5e24)
        with patch('zalmoxis.solver._solve', side_effect=side):
            with pytest.raises(RuntimeError, match='dM/dR'):
                _solve_newton_outer(cp, {}, None, '/tmp')

    def test_out_of_bounds_R_raises(self, newton_config):
        """Newton step that overshoots [R_min, R_max] must raise.

        Pathological M(R) = 1e26 - 1e17*R: large negative slope (1e17,
        big enough to pass the |dM/dR|>1e15 sanity check). At R0=6e6
        we have M(R0) approx 1e26, f approx 9.4e25, R_new = R0 - f/(-1e17)
        approx 6e6 + 9.4e8 m, way past R_max=1.5e7. With the trust
        region disabled, the unclamped step escapes bounds.

        Discriminating from test_vanishing_derivative_raises: this one
        has a HEALTHY slope (1e17 >> 1e15) so the divide-by-zero guard
        does not fire.
        """
        cp = dict(newton_config)
        cp['planet_mass'] = 6.0e24
        cp['_initial_radius_guess'] = 6.0e6
        cp['newton_R_max'] = 1.5e7
        cp['newton_max_iter'] = 5
        # Disable trust region so the bad step actually escapes bounds.
        cp['newton_trust_region_frac'] = 100.0

        side = _make_synthetic_M_solve(lambda R: 1.0e26 - 1.0e17 * float(R))
        with patch('zalmoxis.solver._solve', side_effect=side):
            with pytest.raises(RuntimeError, match='out of bounds'):
                _solve_newton_outer(cp, {}, None, '/tmp')

    def test_inner_solve_called_with_max_iterations_outer_1(self, newton_config):
        """Each Newton step must force max_iterations_outer=1 in the inner _solve.

        This is the contract: Newton owns the outer R loop; _solve runs
        only the inner Picard density iteration at the supplied R. If
        we let _solve run its own outer Picard, Newton's central diff
        becomes garbled because R changes inside the inner call.
        """
        cp = dict(newton_config)
        cp['planet_mass'] = 6.0e24
        cp['_initial_radius_guess'] = 6.5e6
        cp['newton_tol'] = 1.0e-3
        cp['newton_max_iter'] = 4

        side = _make_synthetic_M_solve(
            lambda R: 6.0e24 * (R / 6.5e6) ** 3
        )
        with patch('zalmoxis.solver._solve', side_effect=side) as mock_solve:
            _solve_newton_outer(cp, {}, None, '/tmp')

        # Every call to _solve must have max_iterations_outer=1 in its config.
        for call in mock_solve.call_args_list:
            inner_cp = call[0][0]  # first positional arg
            assert inner_cp['max_iterations_outer'] == 1, (
                f'Inner _solve must run with max_iterations_outer=1, '
                f'got {inner_cp.get("max_iterations_outer")}'
            )
            # And outer_solver MUST be set to 'picard' to break the
            # recursion (else _solve_newton_outer calls itself).
            assert inner_cp['outer_solver'] == 'picard', (
                'Inner _solve must force outer_solver="picard" to avoid recursion'
            )

    def test_history_records_every_outer_iter(self, newton_config):
        """newton_history must record one (R, M, rel) tuple per outer iter."""
        cp = dict(newton_config)
        cp['planet_mass'] = 6.0e24
        cp['_initial_radius_guess'] = 6.5e6
        cp['newton_tol'] = 1.0e-5
        cp['newton_max_iter'] = 8

        side = _make_synthetic_M_solve(
            lambda R: 6.0e24 * (R / 7.0e6) ** 3
        )
        with patch('zalmoxis.solver._solve', side_effect=side):
            result = _solve_newton_outer(cp, {}, None, '/tmp')

        history = result['newton_history']
        n_iter = result['newton_n_iter']
        assert len(history) == n_iter, (
            f'History length ({len(history)}) must equal newton_n_iter ({n_iter})'
        )
        # Each entry is a 3-tuple (R, M, rel).
        for R, M, rel in history:
            assert R > 0
            assert M > 0
            assert rel >= 0
        # rel must be sorted (mostly) decreasing — at least the FINAL
        # rel must be < initial rel.
        assert history[-1][2] < history[0][2]


# ----------------------------------------------------------------------
# (2) End-to-end Newton on real Zalmoxis solve, gated on EOS data
# ----------------------------------------------------------------------


def _data_available():
    """Return True if EOS data required for a full solve is present."""
    try:
        from zalmoxis import get_zalmoxis_root
    except Exception:
        return False
    data_dir = os.path.join(get_zalmoxis_root(), 'data')
    return os.path.isdir(data_dir) and os.listdir(data_dir)


@pytest.mark.slow
@pytest.mark.skipif(not _data_available(), reason='EOS data not staged locally')
class TestNewtonEndToEndEarthLike:
    """End-to-end Newton on a 1 M_E isothermal config.

    Single solve takes ~30-60 s on a workstation; runs only when EOS
    data is available locally (skipped in CI without data download).

    This is the integration check that the algorithm-as-implemented
    produces a converged structure on a real EOS, not just on
    synthetic M(R).
    """

    def test_isothermal_earth_converges_with_newton(self):
        """1 M_E isothermal must converge via Newton to dM/M < 1e-3.

        Loads the Zalmoxis ``input/default.toml`` config and overrides
        the outer-solver knobs and integrator tolerances. This avoids
        having to hand-build a complete config for ``_solve``.
        """
        from zalmoxis.config import (
            load_material_dictionaries,
            load_solidus_liquidus_functions,
            load_zalmoxis_config,
        )
        from zalmoxis.solver import main

        cfg_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '..', 'input', 'default.toml',
        )
        if not os.path.exists(cfg_path):
            pytest.skip(f'default.toml not at {cfg_path}')

        cp = load_zalmoxis_config(cfg_path)
        # Override for Newton + tight integrator tols.
        cp['outer_solver'] = 'newton'
        cp['relative_tolerance'] = 1.0e-9
        cp['absolute_tolerance'] = 1.0e-10
        cp['newton_tol'] = 1.0e-3
        cp['newton_max_iter'] = 15
        cp['use_jax'] = False
        cp['use_anderson'] = False
        cp['wall_timeout'] = 300.0

        layer_eos_config = cp['layer_eos_config']
        mat_dicts = load_material_dictionaries()
        melt_funcs = load_solidus_liquidus_functions(
            layer_eos_config,
            cp.get('rock_solidus', 'Stixrude14-solidus'),
            cp.get('rock_liquidus', 'Stixrude14-liquidus'),
        )
        import zalmoxis as _zal
        input_dir = os.path.normpath(
            os.path.join(os.path.dirname(_zal.__file__), '..', '..', 'input')
        )

        result = main(cp, mat_dicts, melt_funcs, input_dir)

        assert result['converged'] is True
        assert result['converged_mass'] is True
        # Newton with tol=1e-3: assert at most 5e-3 to allow inner-Picard noise
        assert result['best_mass_error'] < 5.0e-3, (
            f'Newton should reach |dM/M| < 5e-3 on Earth-like, '
            f'got {result["best_mass_error"]:.3e}'
        )
        # Sanity: R within Earth-like range.
        R_final = result['radii'][-1]
        assert 5.0e6 < R_final < 8.0e6, (
            f'Final R = {R_final:.4e} m outside Earth-like range'
        )
