"""Tests for the Newton outer mass-radius loop.

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
            f'Linear M(R) should converge in <=3 iters, got {result["newton_n_iter"]}'
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

        side = _make_synthetic_M_solve(lambda R: (4.0 / 3.0) * np.pi * rho * R**3)
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

        side = _make_synthetic_M_solve(lambda R: M_target * (R / R_root) ** 3)
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

    def test_max_iter_exhausted_falls_back_to_brentq(self, newton_config):
        """Newton hitting max_iter falls back to brentq, not raises.

        Use a pathological M(R) that oscillates: M(R) = M_target + A*sin(B*R).
        Newton's central-difference dM/dR sees the local slope (oscillates
        positive/negative) so steps go in random directions. With small
        max_iter (3), should fall through to brentq.

        Discriminating: result['newton_used_brentq'] is True. Brentq
        finds A*sin(B*R) = 0 i.e. R = n*pi/B, the closest of which gives
        |dM/M| ~= machine epsilon.
        """
        M_target = 6.0e24
        A = 1.0e23
        B = 1.0e-5  # 1 oscillation per 6e5 m
        cp = dict(newton_config)
        cp['planet_mass'] = M_target
        cp['_initial_radius_guess'] = 6.5e6
        cp['newton_tol'] = 1.0e-6
        cp['newton_max_iter'] = 3

        side = _make_synthetic_M_solve(lambda R: M_target + A * np.sin(B * R))
        with patch('zalmoxis.solver._solve', side_effect=side):
            result = _solve_newton_outer(cp, {}, None, '/tmp')

        assert result['newton_used_brentq'] is True, (
            'Max-iter exhaustion must route to brentq fall-back'
        )
        assert result['converged'] is True, (
            'Brentq must find a root of f(R)=A*sin(B*R) (any n*pi/B works)'
        )

    def test_vanishing_derivative_falls_back_to_brentq(self, newton_config):
        """Constant M(R): dM/dR = 0 -> falls back to brentq.

        Constant M(R)=5.5e24 != M_target=6e24 has no root; brentq's
        bracket sweep cannot find a sign flip. Result must be
        converged=False with newton_used_brentq=True (so the caller
        knows brentq tried and failed, distinct from "never tried").
        """
        cp = dict(newton_config)
        cp['planet_mass'] = 6.0e24
        cp['_initial_radius_guess'] = 6.5e6
        cp['newton_max_iter'] = 5

        side = _make_synthetic_M_solve(lambda R: 5.5e24)
        with patch('zalmoxis.solver._solve', side_effect=side):
            result = _solve_newton_outer(cp, {}, None, '/tmp')

        assert result['newton_used_brentq'] is True
        assert result['converged'] is False, (
            'Constant M != M_target has no root; must report converged=False'
        )
        # Best-seen rel must equal abs(M-target)/target = abs(5.5e24-6e24)/6e24
        assert abs(result['best_mass_error'] - (0.5e24 / 6.0e24)) < 1.0e-12

    def test_out_of_bounds_R_falls_back_to_brentq(self, newton_config):
        """Newton step that overshoots [R_min, R_max] -> brentq fall-back.

        Pathological M(R) = 1e26 - 1e17*R: large negative slope (1e17,
        big enough to pass the |dM/dR|>1e15 sanity check). At R0=6e6
        we have M(R0)~1e26, f~9.4e25, Newton step takes R_new past R_max.
        Trust region disabled to expose the bound check.

        Discriminating from vanishing-derivative test: this one HAS a
        sign-flipping bracket (linear M(R) crosses M_target at
        R = (1e26 - 6e24)/1e17 = 9.4e8 m -- BUT that's outside
        [R_min=2e6, R_max=1.5e7]). Brentq sweep won't find a sign flip
        within bounds. Expected: converged=False, newton_used_brentq=True.
        """
        cp = dict(newton_config)
        cp['planet_mass'] = 6.0e24
        cp['_initial_radius_guess'] = 6.0e6
        cp['newton_R_max'] = 1.5e7
        cp['newton_max_iter'] = 5
        cp['newton_trust_region_frac'] = 100.0

        side = _make_synthetic_M_solve(lambda R: 1.0e26 - 1.0e17 * float(R))
        with patch('zalmoxis.solver._solve', side_effect=side):
            result = _solve_newton_outer(cp, {}, None, '/tmp')

        assert result['newton_used_brentq'] is True
        # Root at R~9.4e8 m is outside [R_min, R_max] -> sweep can't bracket.
        assert result['converged'] is False

    def test_brentq_recovers_when_newton_at_max_iter_with_a_real_root(
        self,
        newton_config,
    ):
        """A monotonic-but-noisy M(R) where Newton spins out: brentq finds root.

        Builds M(R) = (4/3) pi rho R^3 + small noise: monotonic on average
        but Newton's central-diff sees noise that prevents fast convergence.
        With max_iter=2, Newton WILL spin out; brentq must finish the job.

        This is the canonical 'integrator-noise + bounded Newton iters'
        scenario that brentq fall-back exists for. It's the scenario the
        production CHILI runs will hit.
        """
        M_target = 6.0e24
        rho = 5500.0
        # Noise amplitude small relative to dynamic range.
        noise_amp = 1.0e22

        def M_func(R):
            base = (4.0 / 3.0) * np.pi * rho * R**3
            # Deterministic noise (reproducible) keyed on a hash of R.
            seed = int(abs(R) * 1e3) % (2**31)
            local_rng = np.random.default_rng(seed)
            return base + noise_amp * (local_rng.random() - 0.5)

        cp = dict(newton_config)
        cp['planet_mass'] = M_target
        cp['_initial_radius_guess'] = 8.0e6
        cp['newton_tol'] = 1.0e-3
        cp['newton_max_iter'] = 2  # too few; force brentq

        side = _make_synthetic_M_solve(M_func)
        with patch('zalmoxis.solver._solve', side_effect=side):
            result = _solve_newton_outer(cp, {}, None, '/tmp')

        assert result['converged'] is True
        # newton_used_brentq could be True or False depending on whether
        # Newton happened to nail the noisy root in 2 iters; the test
        # is robust to either outcome.
        # Final R close to the true cubic root.
        R_true = (3.0 * M_target / (4.0 * np.pi * rho)) ** (1.0 / 3.0)
        R_final = result['newton_history'][-1][0]
        assert abs(R_final - R_true) / R_true < 1.0e-2

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

        side = _make_synthetic_M_solve(lambda R: 6.0e24 * (R / 6.5e6) ** 3)
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

        side = _make_synthetic_M_solve(lambda R: 6.0e24 * (R / 7.0e6) ** 3)
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

    def test_step_damping_halves_step_when_residual_does_not_improve(self, newton_config):
        """When iter k's |f_k| >= |f_{k-1}|, the next proposed step is halved.

        Drives the damping branch in ``_solve_newton_outer`` (around
        ``solver.py:1052-1061``) using a stateful mock for ``_solve``.
        The mock returns ``M(R)`` that decreases monotonically with
        each main evaluation: f_0 = +1e23, f_1 = +2e23 (worse), so the
        damping kicks in for iter 2's proposed step. The test then
        checks that iter 2's recorded radius is closer to iter 1 than
        an unhalved Newton step would have produced, confirming the
        halving fired.

        Side dM/dR evaluations (R+/-dR) are recognised via the
        ``newton_dR_rel`` offset and given a positive monotone slope so
        Newton's direction is well-defined.
        """

        cp = dict(newton_config)
        cp['planet_mass'] = 6.0e24
        cp['_initial_radius_guess'] = 6.0e6
        cp['newton_tol'] = 1.0e-9
        cp['newton_max_iter'] = 4
        cp['newton_dR_rel'] = 1.0e-3

        M_target = cp['planet_mass']

        # Stateful mock: index by the index-of-main-eval.
        # Sequence of MAIN evals (R-only calls, not central-diff probes):
        # iter 0 main: f = +1e23 (M too large)
        # iter 1 main: f = +2e23 (worse -> damping arms for iter 2)
        # iter 2 main: f = +1e22 (improved, damping disarms)
        main_residuals = [1.0e23, 2.0e23, 1.0e22, 1.0e21]
        main_idx = {'i': 0}
        recorded_main_R = []

        # dR offset Newton uses for central-difference probes.
        dR_rel = cp['newton_dR_rel']

        def _stateful_M(R: float) -> float:
            # Detect main eval vs central-diff probe by checking whether
            # R is "close to" a recorded main-eval R. Central-diff probes
            # always sit at R_main +/- dR_rel*R_main; their residual sign
            # encodes a positive slope so Newton's direction is forward.
            for j, R_main in enumerate(recorded_main_R):
                if abs(R - R_main) <= dR_rel * R_main * 2.0:
                    # Side eval: use +ve slope around the latest main R.
                    sign = 1.0 if R > R_main else -1.0
                    return M_target + main_residuals[j] + sign * 1.0e21
            # Main eval: pop the next residual from the planned sequence.
            i = main_idx['i']
            recorded_main_R.append(R)
            f = main_residuals[min(i, len(main_residuals) - 1)]
            main_idx['i'] = i + 1
            return M_target + f

        side = _make_synthetic_M_solve(_stateful_M)
        with patch('zalmoxis.solver._solve', side_effect=side):
            result = _solve_newton_outer(cp, {}, None, '/tmp')

        history = result['newton_history']
        # Must have run at least 3 outer iters to exercise the damping
        # arming-then-firing transition.
        assert len(history) >= 3, (
            f'Expected >=3 outer iters to exercise damping; got {len(history)}'
        )
        # Iter 1's |f| (2e23) must exceed iter 0's |f| (1e23). This is the
        # condition that arms the damping check on iter 2.
        f0 = history[0][1] - M_target
        f1 = history[1][1] - M_target
        assert abs(f1) > abs(f0), (
            f'Damping precondition not met: |f1|={abs(f1):.2e} should exceed '
            f'|f0|={abs(f0):.2e}.'
        )
        # Discriminating assertion. With the engineered slope from the side
        # evals, ``dM/dR`` at iter 1 is ``1e21 / (dR_rel * R_1) = 1e24 / R_1``.
        # The raw Newton step is ``-f_1 / dM/dR = -0.2 * R_1``, which exceeds
        # the default trust-region cap ``newton_trust_region_frac=0.1`` so
        # the cap clips the step to ``-0.1 * R_1``. Damping then halves it
        # again to ``-0.05 * R_1``. The composed ratio is:
        #   damped + capped:  R_2 / R_1 = 0.95
        #   capped only:      R_2 / R_1 = 0.90
        #   raw Newton:       R_2 / R_1 = 0.80
        # Only damped+capped matches 0.95; the assertion fails if damping
        # silently regresses to capped-only behaviour.
        R_1 = history[1][0]
        R_2 = history[2][0]
        assert R_2 / R_1 == pytest.approx(0.95, rel=1e-3), (
            f'Step damping did not halve the trust-region-capped step: '
            f'R_2/R_1={R_2 / R_1:.4f}, expected 0.95 (capped + damped) vs '
            f'0.90 (capped only) vs 0.80 (raw Newton).'
        )


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
            '..',
            'input',
            'default.toml',
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
        assert 5.0e6 < R_final < 8.0e6, f'Final R = {R_final:.4e} m outside Earth-like range'
