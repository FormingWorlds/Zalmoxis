"""Coverage tests for ``solver.solve_miscible_interior``.

The wrapped ``main()`` solver is mocked so each test runs in milliseconds and
exercises the outer mass-conservation iteration in isolation.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from zalmoxis.mixing import VolatileProfile
from zalmoxis.solver import solve_miscible_interior

pytestmark = pytest.mark.unit


def _synthetic_main_result(
    *,
    radii: np.ndarray,
    density: np.ndarray,
    pressure: np.ndarray,
    temperature: np.ndarray,
    converged: bool = True,
) -> dict:
    """Return a minimal dict with the keys ``solve_miscible_interior`` reads."""
    return {
        'converged': converged,
        'radii': radii.copy(),
        'density': density.copy(),
        'pressure': pressure.copy(),
        'temperature': temperature.copy(),
    }


def _shell_grid(n: int = 10):
    """Build a tiny radial grid with monotone profiles."""
    radii = np.linspace(1.0e6, 6.4e6, n)
    density = np.linspace(8000.0, 3000.0, n)
    pressure = np.linspace(3.0e11, 1.0e5, n)
    temperature = np.linspace(5500.0, 1500.0, n)
    return radii, density, pressure, temperature


def test_fast_path_volatile_profile_none():
    """volatile_profile is None -> single main() call, sentinel keys appended."""
    radii, density, pressure, temperature = _shell_grid()
    result_template = _synthetic_main_result(
        radii=radii,
        density=density,
        pressure=pressure,
        temperature=temperature,
    )

    with patch('zalmoxis.solver.main', return_value=result_template) as mocked_main:
        result = solve_miscible_interior(
            config_params={},
            material_dictionaries={},
            melting_curves_functions=None,
            input_dir='.',
            volatile_profile=None,
            h2_mass_targets={'Chabrier:H': 1.0e22},
        )

    assert mocked_main.call_count == 1
    assert result['miscibility_converged'] is True
    assert result['miscibility_iterations'] == 0
    assert result['solvus_radius'] is None
    assert result['solvus_temperature'] is None
    assert result['solvus_pressure'] is None
    assert result['x_interior_converged'] == {}
    assert result['h2_mass_integrated'] == {}


def test_fast_path_global_miscibility_disabled():
    """global_miscibility=False -> fast path, no iteration."""
    radii, density, pressure, temperature = _shell_grid()
    result_template = _synthetic_main_result(
        radii=radii,
        density=density,
        pressure=pressure,
        temperature=temperature,
    )
    vp = VolatileProfile(global_miscibility=False, x_interior={'Chabrier:H': 0.01})

    with patch('zalmoxis.solver.main', return_value=result_template) as mocked_main:
        result = solve_miscible_interior(
            config_params={},
            material_dictionaries={},
            melting_curves_functions=None,
            input_dir='.',
            volatile_profile=vp,
            h2_mass_targets={'Chabrier:H': 1.0e22},
        )

    assert mocked_main.call_count == 1
    assert result['miscibility_iterations'] == 0
    assert result['miscibility_converged'] is True


def test_fast_path_empty_h2_mass_targets():
    """Empty h2_mass_targets dict -> fast path even with global_miscibility=True."""
    radii, density, pressure, temperature = _shell_grid()
    result_template = _synthetic_main_result(
        radii=radii,
        density=density,
        pressure=pressure,
        temperature=temperature,
    )
    vp = VolatileProfile(global_miscibility=True, x_interior={'Chabrier:H': 0.01})

    with patch('zalmoxis.solver.main', return_value=result_template) as mocked_main:
        result = solve_miscible_interior(
            config_params={},
            material_dictionaries={},
            melting_curves_functions=None,
            input_dir='.',
            volatile_profile=vp,
            h2_mass_targets={},
        )

    assert mocked_main.call_count == 1
    assert result['miscibility_iterations'] == 0


def test_inner_main_does_not_converge_breaks_loop():
    """Inner main()['converged']=False -> log warning and break out of outer loop."""
    radii, density, pressure, temperature = _shell_grid()
    diverged = _synthetic_main_result(
        radii=radii,
        density=density,
        pressure=pressure,
        temperature=temperature,
        converged=False,
    )
    vp = VolatileProfile(global_miscibility=True, x_interior={'Unknown:Foo': 0.01})

    with patch('zalmoxis.solver.main', return_value=diverged) as mocked_main:
        result = solve_miscible_interior(
            config_params={},
            material_dictionaries={},
            melting_curves_functions=None,
            input_dir='.',
            volatile_profile=vp,
            h2_mass_targets={'Unknown:Foo': 1.0e22},
            max_iterations=5,
        )

    assert mocked_main.call_count == 1
    assert result['miscibility_converged'] is False
    assert result['miscibility_iterations'] == 1
    # Regression guard: the early-break path must still attach every
    # downstream key. Pre-loop initialisation of integrated_masses /
    # solvus_info ensures the post-loop block at solver.py:2474-2480
    # does not raise UnboundLocalError when the very first iteration
    # bails out via the converged=False break.
    assert 'solvus_radius' in result
    assert 'solvus_temperature' in result
    assert 'solvus_pressure' in result
    assert 'x_interior_converged' in result
    assert 'h2_mass_integrated' in result
    # No iterations completed -> integrated mass dict is empty and the
    # solvus probes default to None (no shells were inspected).
    assert result['solvus_radius'] is None
    assert result['solvus_temperature'] is None
    assert result['solvus_pressure'] is None
    assert result['h2_mass_integrated'] == {}


def test_unknown_species_assume_miscible_converges_first_iteration():
    """Unknown species -> _is_above_binodal returns True everywhere; with x_interior tuned
    so the integrated mass matches the target, convergence happens on iteration 1."""
    n = 50
    radii = np.linspace(1.0e5, 6.4e6, n)
    density = np.full(n, 5000.0)
    pressure = np.linspace(3.0e11, 1.0e5, n)
    temperature = np.linspace(5500.0, 1500.0, n)
    result_template = _synthetic_main_result(
        radii=radii,
        density=density,
        pressure=pressure,
        temperature=temperature,
    )

    # Compute the integrated shell mass for x_int=1 over all shells.
    total_mass_at_x1 = 0.0
    for i in range(n - 1):
        r_mid = 0.5 * (radii[i] + radii[i + 1])
        dr = radii[i + 1] - radii[i]
        rho_mid = 0.5 * (density[i] + density[i + 1])
        total_mass_at_x1 += rho_mid * 4.0 * np.pi * r_mid**2 * dr

    x_int_initial = 0.01
    target = x_int_initial * total_mass_at_x1
    vp = VolatileProfile(
        global_miscibility=True,
        x_interior={'Unknown:Foo': x_int_initial},
    )

    with patch('zalmoxis.solver.main', return_value=result_template) as mocked_main:
        result = solve_miscible_interior(
            config_params={},
            material_dictionaries={},
            melting_curves_functions=None,
            input_dir='.',
            volatile_profile=vp,
            h2_mass_targets={'Unknown:Foo': target},
            max_iterations=5,
            mass_tolerance=1e-3,
        )

    assert mocked_main.call_count == 1
    assert result['miscibility_converged'] is True
    assert result['miscibility_iterations'] == 1
    assert result['h2_mass_integrated']['Unknown:Foo'] == pytest.approx(target, rel=1e-3)
    assert result['x_interior_converged']['Unknown:Foo'] == pytest.approx(x_int_initial)


def test_secant_scaling_converges_in_two_iterations():
    """Initial x_interior wrong by 2x -> secant rescaling lands on target in iter 2."""
    n = 30
    radii = np.linspace(1.0e5, 6.4e6, n)
    density = np.full(n, 5000.0)
    pressure = np.linspace(3.0e11, 1.0e5, n)
    temperature = np.linspace(5500.0, 1500.0, n)
    result_template = _synthetic_main_result(
        radii=radii,
        density=density,
        pressure=pressure,
        temperature=temperature,
    )

    total_mass_at_x1 = 0.0
    for i in range(n - 1):
        r_mid = 0.5 * (radii[i] + radii[i + 1])
        dr = radii[i + 1] - radii[i]
        rho_mid = 0.5 * (density[i] + density[i + 1])
        total_mass_at_x1 += rho_mid * 4.0 * np.pi * r_mid**2 * dr

    target_x = 0.02
    target = target_x * total_mass_at_x1
    vp = VolatileProfile(
        global_miscibility=True,
        x_interior={'Unknown:Foo': 0.01},  # half of target_x
    )

    with patch('zalmoxis.solver.main', return_value=result_template):
        result = solve_miscible_interior(
            config_params={},
            material_dictionaries={},
            melting_curves_functions=None,
            input_dir='.',
            volatile_profile=vp,
            h2_mass_targets={'Unknown:Foo': target},
            max_iterations=5,
            mass_tolerance=1e-3,
        )

    assert result['miscibility_converged'] is True
    assert result['miscibility_iterations'] == 2
    assert result['x_interior_converged']['Unknown:Foo'] == pytest.approx(target_x, rel=1e-3)


def test_no_integrated_mass_doubles_x_interior():
    """When no shells are above the binodal, integrated mass = 0; the loop doubles x.
    Use H2O species at low T (always below binodal) so all shells fail the test."""
    n = 20
    radii = np.linspace(1.0e5, 6.4e6, n)
    density = np.full(n, 5000.0)
    pressure = np.linspace(3.0e11, 1.0e5, n)
    temperature = np.full(n, 100.0)  # Far below H2O critical point
    result_template = _synthetic_main_result(
        radii=radii,
        density=density,
        pressure=pressure,
        temperature=temperature,
    )
    vp = VolatileProfile(
        global_miscibility=True,
        x_interior={'PALEOS:H2O': 0.001},
    )

    with patch('zalmoxis.solver.main', return_value=result_template):
        result = solve_miscible_interior(
            config_params={},
            material_dictionaries={},
            melting_curves_functions=None,
            input_dir='.',
            volatile_profile=vp,
            h2_mass_targets={'PALEOS:H2O': 1.0e23},
            max_iterations=3,
            mass_tolerance=1e-3,
        )

    # No shells above binodal -> integrated mass = 0 -> x doubled each iteration.
    # 0.001 * 2 * 2 = 0.004 after 2 doublings (3rd iter would also double).
    assert result['miscibility_converged'] is False
    # Each iteration doubled x: clamp range is [1e-6, 0.5], so doubling lands within.
    assert result['x_interior_converged']['PALEOS:H2O'] == pytest.approx(0.008, rel=1e-6)


def test_x_interior_upper_clamp():
    """Secant rescaling > 0.5 should be clamped at 0.5."""
    n = 20
    radii = np.linspace(1.0e5, 6.4e6, n)
    density = np.full(n, 5000.0)
    pressure = np.linspace(3.0e11, 1.0e5, n)
    temperature = np.linspace(5500.0, 1500.0, n)
    result_template = _synthetic_main_result(
        radii=radii,
        density=density,
        pressure=pressure,
        temperature=temperature,
    )

    total_mass_at_x1 = 0.0
    for i in range(n - 1):
        r_mid = 0.5 * (radii[i] + radii[i + 1])
        dr = radii[i + 1] - radii[i]
        rho_mid = 0.5 * (density[i] + density[i + 1])
        total_mass_at_x1 += rho_mid * 4.0 * np.pi * r_mid**2 * dr

    # Target requires x ~10 -> rescaling should hit clamp at 0.5
    target = 10.0 * total_mass_at_x1 * 0.001
    vp = VolatileProfile(
        global_miscibility=True,
        x_interior={'Unknown:Foo': 0.001},
    )

    with patch('zalmoxis.solver.main', return_value=result_template):
        result = solve_miscible_interior(
            config_params={},
            material_dictionaries={},
            melting_curves_functions=None,
            input_dir='.',
            volatile_profile=vp,
            h2_mass_targets={'Unknown:Foo': target},
            max_iterations=2,
            mass_tolerance=1e-6,
        )

    assert result['x_interior_converged']['Unknown:Foo'] <= 0.5


def test_solvus_detection_for_chabrier_h():
    """Above-then-below transition for Chabrier:H records solvus radius/T/P.

    Mocks ``VolatileProfile._is_above_binodal`` to deterministically return
    True at high pressure and False at low pressure, guaranteeing exactly
    one above-then-below transition. The test then asserts that the
    detected solvus point lies inside the radial grid range and that its
    P/T match the shell where the transition happens.
    """
    n = 30
    radii = np.linspace(1.0e5, 6.4e6, n)
    density = np.full(n, 5000.0)
    pressure = np.linspace(5.0e11, 1.0e9, n)  # 500 GPa -> 1 GPa
    temperature = np.linspace(8000.0, 2000.0, n)
    result_template = _synthetic_main_result(
        radii=radii,
        density=density,
        pressure=pressure,
        temperature=temperature,
    )
    vp = VolatileProfile(
        global_miscibility=True,
        x_interior={'Chabrier:H': 0.05},
    )

    # Threshold pressure separating "above binodal" (deep) from "below
    # binodal" (shallow). Pick a value squarely in the middle of the
    # pressure profile so the solvus crossing is unambiguous.
    p_threshold = 1.0e11

    def _mock_above(self, species, P, T):  # noqa: ARG001
        return P > p_threshold

    with (
        patch('zalmoxis.solver.main', return_value=result_template),
        patch.object(VolatileProfile, '_is_above_binodal', _mock_above),
    ):
        result = solve_miscible_interior(
            config_params={},
            material_dictionaries={},
            melting_curves_functions=None,
            input_dir='.',
            volatile_profile=vp,
            h2_mass_targets={'Chabrier:H': 1.0e20},
            max_iterations=2,
            mass_tolerance=1.0,
        )

    # Deterministic crossing exists -> solvus must be set, not None.
    assert result['solvus_radius'] is not None
    assert result['solvus_temperature'] is not None
    assert result['solvus_pressure'] is not None
    # Solvus radius lies strictly inside the radial grid extent.
    assert radii[0] < result['solvus_radius'] < radii[-1]
    # P at the solvus must straddle the threshold (the crossing happens at
    # the first shell whose midpoint pressure dips below it).
    assert result['solvus_pressure'] < p_threshold
    # Temperature is positive and within the supplied profile range.
    assert temperature[-1] <= result['solvus_temperature'] <= temperature[0]


def test_zero_target_skipped():
    """Species with target <= 0 is not used to drive convergence."""
    n = 15
    radii = np.linspace(1.0e5, 6.4e6, n)
    density = np.full(n, 5000.0)
    pressure = np.linspace(3.0e11, 1.0e5, n)
    temperature = np.linspace(5500.0, 1500.0, n)
    result_template = _synthetic_main_result(
        radii=radii,
        density=density,
        pressure=pressure,
        temperature=temperature,
    )
    vp = VolatileProfile(
        global_miscibility=True,
        x_interior={'Unknown:Foo': 0.01},
    )

    with patch('zalmoxis.solver.main', return_value=result_template):
        result = solve_miscible_interior(
            config_params={},
            material_dictionaries={},
            melting_curves_functions=None,
            input_dir='.',
            volatile_profile=vp,
            h2_mass_targets={'Unknown:Foo': 0.0},  # zero target -> skip
            max_iterations=3,
            mass_tolerance=1e-3,
        )

    # With target=0 the loop sees all_converged stay True after the skip.
    assert result['miscibility_converged'] is True


def test_non_converging_exhausts_max_iterations():
    """Persistent rel_error > tolerance -> miscibility_converged=False after max."""
    n = 20
    radii = np.linspace(1.0e5, 6.4e6, n)
    density = np.full(n, 5000.0)
    pressure = np.linspace(3.0e11, 1.0e5, n)
    temperature = np.linspace(5500.0, 1500.0, n)
    result_template = _synthetic_main_result(
        radii=radii,
        density=density,
        pressure=pressure,
        temperature=temperature,
    )

    total_mass_at_x1 = 0.0
    for i in range(n - 1):
        r_mid = 0.5 * (radii[i] + radii[i + 1])
        dr = radii[i + 1] - radii[i]
        rho_mid = 0.5 * (density[i] + density[i + 1])
        total_mass_at_x1 += rho_mid * 4.0 * np.pi * r_mid**2 * dr

    vp = VolatileProfile(
        global_miscibility=True,
        x_interior={'Unknown:Foo': 0.01},
    )
    target = 0.02 * total_mass_at_x1  # achievable but tight tolerance

    with patch('zalmoxis.solver.main', return_value=result_template):
        result = solve_miscible_interior(
            config_params={},
            material_dictionaries={},
            melting_curves_functions=None,
            input_dir='.',
            volatile_profile=vp,
            h2_mass_targets={'Unknown:Foo': target},
            max_iterations=1,  # force non-convergence after 1 iter
            mass_tolerance=1e-9,
        )

    assert result['miscibility_converged'] is False
    assert result['miscibility_iterations'] == 1
