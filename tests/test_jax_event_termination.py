"""Regression tests for the diffrax.Event pressure-zero terminal event
in the JAX structure-ODE path.

Verifies:
  (1) When pressure crosses zero mid-grid, diffrax terminates and
      the wrapper pads the post-event shells to match numpy's
      solve_structure contract (pressure = 0, mass/gravity carry
      the last-valid value).
  (2) The wrapper's post-event padding handler produces no `inf`
      or `NaN` in the returned arrays.
  (3) Physics drift between the JAX+Event path and the numpy path
      is at solver-tolerance (profile drift ~5e-4, previously ~1e-3
      with the freeze-based hack and 1e25 ratio-artifact on pressure).

Uses the full `zalmoxis.solver.main()` path (not an isolated
solve_structure call) because the pressure-zero event only fires
meaningfully when the Picard loop has converged to a self-consistent
state. An isolated solve with a hand-picked temperature profile
produces mid-shell drift that accumulates past the P=0 crossing
point and makes the Event behavior hard to reason about.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _run_main(use_jax, use_anderson=False):
    """Run zalmoxis.solver.main() on bench_performance.toml.

    Returns the result dict.
    """
    from zalmoxis.config import (
        load_material_dictionaries,
        load_solidus_liquidus_functions,
        load_zalmoxis_config,
    )
    from zalmoxis.solver import main

    cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', 'input', 'bench_performance.toml',
    )
    if not os.path.exists(cfg_path):
        pytest.skip(f'bench_performance.toml not available at {cfg_path}')

    config_params = load_zalmoxis_config(cfg_path)
    config_params['wall_timeout'] = 300.0
    if use_jax:
        config_params['use_jax'] = True
    if use_anderson:
        config_params['use_anderson'] = True

    layer_eos_config = config_params['layer_eos_config']
    mat_dicts = load_material_dictionaries()
    melt_funcs = load_solidus_liquidus_functions(
        layer_eos_config,
        config_params.get('rock_solidus', 'Stixrude14-solidus'),
        config_params.get('rock_liquidus', 'Stixrude14-liquidus'),
    )
    import zalmoxis as _zal
    input_dir = os.path.normpath(
        os.path.join(os.path.dirname(_zal.__file__), '..', '..', 'input')
    )
    return main(
        config_params,
        material_dictionaries=mat_dicts,
        melting_curves_functions=melt_funcs,
        input_dir=input_dir,
    )


def _data_available():
    try:
        from zalmoxis import get_zalmoxis_root
    except Exception:
        return False
    data_dir = os.path.join(get_zalmoxis_root(), 'data')
    return os.path.isdir(data_dir) and os.listdir(data_dir)


@pytest.mark.skipif(not _data_available(), reason='EOS data not staged locally')
class TestEventTermination:
    """Event-termination regression on the full bench_performance path.

    Each solve is ~20-30 s; class-scoped fixtures keep to two runs.
    """

    @pytest.fixture(scope='class')
    def numpy_result(self):
        return _run_main(use_jax=False)

    @pytest.fixture(scope='class')
    def jax_event_result(self):
        return _run_main(use_jax=True)

    def test_both_converge(self, numpy_result, jax_event_result):
        """Basic contract: both paths must reach all three convergence
        criteria before any other test is meaningful.
        """
        assert numpy_result['converged'] is True
        assert numpy_result['converged_pressure'] is True
        assert jax_event_result['converged'] is True
        assert jax_event_result['converged_pressure'] is True

    def test_pad_outputs_finite(self, jax_event_result):
        """After the Event fires and the wrapper pads the post-event
        shells, every returned value must be finite. Guards against
        a regression where diffrax's post-event `inf` padding leaks
        through the wrapper.
        """
        for k in ('radii', 'mass_enclosed', 'gravity', 'pressure',
                  'temperature', 'density'):
            arr = np.asarray(jax_event_result[k])
            assert np.all(np.isfinite(arr)), (
                f'{k} has non-finite values at indices: '
                f'{np.flatnonzero(~np.isfinite(arr))}'
            )

    def test_outer_pressure_zero_when_numpy_zero(self, numpy_result,
                                                   jax_event_result):
        """If numpy pads pressure to exactly 0 in the outer shells,
        JAX+Event must do the same. Matches numpy's contract at
        structure_model.solve_structure line 369.
        """
        P_np = np.asarray(numpy_result['pressure'])
        P_jx = np.asarray(jax_event_result['pressure'])

        numpy_zero_mask = (P_np == 0.0)
        if not np.any(numpy_zero_mask):
            pytest.skip('numpy did not pad (integration reached outer radius)')

        jax_at_np_zero = P_jx[numpy_zero_mask]
        assert np.all(jax_at_np_zero == 0.0), (
            f'JAX pressure at numpy-zero indices: {jax_at_np_zero} '
            f'(want all exactly 0.0)'
        )

    def test_mass_gravity_pad_carries_last_valid(self, jax_event_result):
        """On shells where the JAX path padded pressure to 0, the
        mass/gravity arrays must carry the last-valid value across
        all padded shells (no variation inside the pad region).
        """
        P_jx = np.asarray(jax_event_result['pressure'])
        mass_jx = np.asarray(jax_event_result['mass_enclosed'])
        g_jx = np.asarray(jax_event_result['gravity'])

        zero_mask = (P_jx == 0.0)
        if zero_mask.sum() < 2:
            pytest.skip('Less than 2 padded shells; can\'t test flatness.')

        pad_idx = np.flatnonzero(zero_mask)
        m0 = mass_jx[pad_idx[0]]
        g0 = g_jx[pad_idx[0]]
        assert np.all(mass_jx[pad_idx] == m0), (
            f'mass varies on padded shells: {mass_jx[pad_idx]}')
        assert np.all(g_jx[pad_idx] == g0), (
            f'gravity varies on padded shells: {g_jx[pad_idx]}')

    def test_profile_drift_at_solver_tolerance(self, numpy_result,
                                                 jax_event_result):
        """Profile drift between numpy and JAX+Event paths must be
        at solver-tolerance. Previously (with the freeze-based hack
        and no Event) pressure drift could peak at 1e25 relative due
        to the outer-shell numerical-zero ratio. With Event+pad that
        collapses to ~5e-4.
        """
        def scaled_drift(a, b):
            a_arr = np.asarray(a)
            b_arr = np.asarray(b)
            scale = max(np.abs(a_arr).max(), 1e-30)
            return float(np.abs(a_arr - b_arr).max() / scale)

        drifts = {}
        for k in ('radii', 'mass_enclosed', 'gravity', 'pressure',
                  'temperature', 'density'):
            drifts[k] = scaled_drift(numpy_result[k], jax_event_result[k])
            print(f'  {k:15s}  scaled drift = {drifts[k]:.3e}')

        # 1e-3 bound is generous: scipy-RK45 rtol=1e-5 and Tsit5
        # rtol=1e-5 with different dense-output polynomials produce
        # O(1e-4 to 5e-4) drift at the solver-tolerance floor. 1e-3
        # catches regressions while tolerating normal FP variation.
        for k, d in drifts.items():
            assert d <= 1e-3, f'{k} drift {d:.3e} > 1e-3'

    def test_scalar_endpoints_at_solver_tolerance(self, numpy_result,
                                                    jax_event_result):
        """Planet-level scalars (R_planet, M_planet, cmb_mass) must
        agree with the numpy path to within solver tolerance.
        """
        for k in ('cmb_mass', 'core_mantle_mass'):
            a = float(numpy_result[k])
            b = float(jax_event_result[k])
            rel = abs(a - b) / abs(a) if a != 0 else abs(a - b)
            print(f'  {k:18s}  np={a:.6e}  jax={b:.6e}  rel={rel:.3e}')
            assert rel <= 1e-3, f'{k} drift {rel:.3e} > 1e-3'

        # R_planet is an output of the outer mass-convergence loop (each
        # Picard iteration adjusts the grid to hit target M_planet), so
        # it can drift at solver-tolerance between the two paths.
        R_np = float(numpy_result['radii'][-1])
        R_jx = float(jax_event_result['radii'][-1])
        rel_R = abs(R_np - R_jx) / abs(R_np)
        print(f'  R_planet           np={R_np:.6e}  jax={R_jx:.6e}  rel={rel_R:.3e}')
        assert rel_R <= 1e-3, f'R_planet drift {rel_R:.3e} > 1e-3'

        # Mass at the outermost shell == total planet mass (after pad).
        M_np = float(numpy_result['mass_enclosed'][-1])
        M_jx = float(jax_event_result['mass_enclosed'][-1])
        rel_M = abs(M_np - M_jx) / abs(M_np)
        print(f'  M_planet           np={M_np:.6e}  jax={M_jx:.6e}  rel={rel_M:.3e}')
        assert rel_M <= 1e-3, f'M_planet drift {rel_M:.3e} > 1e-3'

    def test_no_spurious_zeros_in_interior(self, jax_event_result):
        """Interior shells (inner half of the profile) must have
        well-positive pressure. Guards against a bug where the Event
        fires spuriously at t0 or early in the integration, which
        would incorrectly truncate the solve to a near-empty profile.
        """
        P_jx = np.asarray(jax_event_result['pressure'])
        inner = P_jx[: len(P_jx) // 2]
        assert np.all(inner > 1e8), (
            f'interior pressure dropped below 1e8 Pa: min={inner.min():.3e}, '
            f'suggests Event fired too early.'
        )
