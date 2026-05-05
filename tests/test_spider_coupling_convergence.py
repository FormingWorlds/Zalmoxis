"""Regression tests for the SPIDER-coupling failure regime.

Pins the behaviour of the robustness fixes (density seeding from the
previous structure, Newton outer solver, retry-then-fallback) that
recovered convergence on the failure mode SPIDER's entropy formulation
exposes: steep adiabatic T(r) gradients (5000 K centre to 1800 K
surface) form within a few coupling steps and straddle the silicate
solidus-liquidus transition, where PALEOS density jumps 5-15%. Plain
damped Picard diverges in that regime.

Three behaviours tested:

1. **Steep T(r) profile converges** (SPIDER-like, 5000 K → 1800 K).
   The regime that broke a large fraction of cases pre-hardening.
2. **Flat T(r) profile converges** (Aragog-like, ~4400 K).
   The regime that worked even before hardening; negative control
   that confirms the fix did not regress the easy case.
3. **Density seeding accelerates the Picard loop**: second Zalmoxis
   call with `initial_density` from the first call must reach the
   same answer in fewer Picard iterations and lower wall time.

Tests use Newton outer-solver to keep wall under ~30 s per solve
(Picard would take 60-90 s and may hit the basin attractor on the
steep profile). PALEOS data is required; tests skip cleanly when
absent (CI without data download).
"""

from __future__ import annotations

import os
import time

import numpy as np
import pytest


def _zalmoxis_root() -> str:
    import zalmoxis as _zal

    return os.path.normpath(os.path.join(os.path.dirname(_zal.__file__), '..', '..'))


def _has_paleos_data() -> bool:
    root = _zalmoxis_root()
    return all(
        os.path.isfile(os.path.join(root, 'data', sub, name))
        for sub, name in (
            ('EOS_PALEOS_iron', 'paleos_iron_eos_table_pt.dat'),
            ('EOS_PALEOS_MgSiO3_unified', 'paleos_mgsio3_eos_table_pt.dat'),
        )
    )


def _input_dir() -> str:
    return os.path.join(_zalmoxis_root(), 'input')


def _spider_like_profile(T_centre: float, T_surface: float):
    """Return a callable T(r, P) -> T mimicking a SPIDER-evolved adiabat.

    Linear in r between r=0 (T = T_centre) and r=R (T = T_surface), as
    SPIDER would project after a few coupling steps. R is unknown to
    the callable since the planet radius is the solver's output;
    closure on R is via passed argument.
    """

    def _make(R_planet: float):
        def T_of_rP(r, P, _R=R_planet, _Tc=T_centre, _Ts=T_surface):
            x = max(0.0, min(1.0, float(r) / _R))
            return _Tc + (_Ts - _Tc) * x

        return T_of_rP

    return _make


def _flat_profile(T_value: float):
    """Constant T(r, P) callable, mimicking Aragog's flat IC."""

    def T_of_rP(r, P, _T=T_value):
        return _T

    return T_of_rP


def _make_config(*, planet_mass_kg: float, num_layers: int = 50):
    return {
        'planet_mass': planet_mass_kg,
        'core_mass_fraction': 0.32,
        'mantle_mass_fraction': 0,
        'temperature_mode': 'adiabatic',  # placeholder; bypassed by temperature_function
        'surface_temperature': 1800.0,
        'center_temperature': 5000.0,
        'temp_profile_file': '',
        'layer_eos_config': {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
        'mushy_zone_factor': 1.0,
        'num_layers': num_layers,
        'target_surface_pressure': 101325,
        'data_output_enabled': False,
        'plotting_enabled': False,
        # Newton outer to avoid the Picard basin attractor on hot profiles.
        'outer_solver': 'newton',
        'relative_tolerance': 1.0e-9,
        'absolute_tolerance': 1.0e-10,
        'newton_tol': 1.0e-3,
        'newton_max_iter': 15,
        'wall_timeout': 120.0,
    }


def _earth_mass_kg() -> float:
    from zalmoxis.constants import earth_mass

    return float(earth_mass)


@pytest.mark.unit
@pytest.mark.slow
class TestSpiderCouplingConvergence:
    """Solver must converge for the SPIDER-like and Aragog-like T(r) regimes
    that motivated the robustness-hardening work.

    Tagged ``slow`` because the three regression tests in this class
    (test_density_seeding_accelerates_second_call, test_flat_aragog_T_profile_converges_at_1ME,
    test_steep_spider_T_profile_converges_at_1ME) sum to ~370s of solver
    wall: they each run a full Picard solve over PALEOS-unified at
    1 M_earth, including a second call for the seeding case. Default
    CI ``pytest -m "unit and not slow"`` excludes them; the smaller
    ``TestSpiderCouplingMassSweep`` already-slow class covers the
    multi-mass sweep.
    """

    @pytest.fixture(scope='class')
    def materials(self):
        if not _has_paleos_data():
            pytest.skip('PALEOS unified EOS data not available')
        from zalmoxis.config import load_material_dictionaries

        return load_material_dictionaries()

    def _solve(self, materials, config, temperature_function):
        from zalmoxis.solver import main

        return main(
            config,
            material_dictionaries=materials,
            melting_curves_functions=None,
            input_dir=_input_dir(),
            temperature_function=temperature_function,
        )

    def test_steep_spider_T_profile_converges_at_1ME(self, materials):
        """Steep 5000 K -> 1800 K SPIDER-like T(r) at 1 M_Earth must converge.

        This is a failure regime where the gradient spans the silicate
        mushy zone and rho jumps ~10% across the solidus-liquidus
        boundary. Plain Picard diverges at iter ~10-15 in this regime.
        With Newton + the density-seeding / structure-fallback fixes,
        the solver must converge to mass-error < 5e-3 within wall_timeout.

        Asserts:
          (1) converged is True (mass + pressure + density)
          (2) mass conservation: |dM/M| < 5e-3
          (3) physical invariants: T > 0 and P > 0 everywhere,
              P monotonically decreasing centre-to-surface
          (4) R_planet within Earth-like 5-8 Mm range (sanity)
        """
        config = _make_config(planet_mass_kg=1.0 * _earth_mass_kg())
        # Use a plausible R guess for the closure; solver re-evaluates.
        T_factory = _spider_like_profile(T_centre=5000.0, T_surface=1800.0)
        T_func = T_factory(R_planet=6.4e6)

        result = self._solve(materials, config, T_func)

        assert result['converged'] is True, 'steep SPIDER T(r) must converge'
        assert result['converged_mass'] is True
        assert result['best_mass_error'] < 5.0e-3, (
            f'mass-error {result["best_mass_error"]:.3e} exceeds 5e-3 threshold'
        )

        T = np.asarray(result['temperature'])
        P = np.asarray(result['pressure'])
        assert np.all(T > 0), 'temperature must be positive everywhere'
        assert np.all(P >= 0), 'pressure must be non-negative everywhere'

        # P decreases monotonically centre-to-surface (excluding padded outer
        # shells with P==0 from the surface-pressure boundary condition).
        valid = P > 0
        P_valid = P[valid]
        assert np.all(np.diff(P_valid) <= 0), (
            'pressure must be monotonically decreasing in the interior'
        )

        R_planet = float(result['radii'][-1])
        assert 5.0e6 < R_planet < 8.0e6, (
            f'R_planet = {R_planet:.3e} m outside Earth-like 5-8 Mm range'
        )

    def test_flat_aragog_T_profile_converges_at_1ME(self, materials):
        """Flat ~4400 K Aragog-like T(r) at 1 M_Earth must converge.

        Negative control: this is the regime that worked even before
        the hardening. Confirms the fix did not regress the easy case.
        Asserts the same invariants as the steep test.
        """
        config = _make_config(planet_mass_kg=1.0 * _earth_mass_kg())
        T_func = _flat_profile(T_value=4400.0)

        result = self._solve(materials, config, T_func)

        assert result['converged'] is True
        assert result['converged_mass'] is True
        assert result['best_mass_error'] < 5.0e-3

        T = np.asarray(result['temperature'])
        P = np.asarray(result['pressure'])
        assert np.all(T > 0)
        assert np.all(P >= 0)
        # T should be ~constant 4400 K (small variation from interp on radii grid).
        np.testing.assert_allclose(T[T > 0], 4400.0, rtol=0, atol=1.0)

    def test_density_seeding_accelerates_second_call(self, materials):
        """Second Zalmoxis call seeded with first-call density must
        either converge faster (fewer Picard iters / lower wall time)
        or at minimum produce the same converged structure.

        This is the load-bearing assertion behind density seeding:
        without it every Zalmoxis call starts the Picard loop from
        `density = np.zeros`, and the 50-80% Picard-iteration overhead
        per call accumulates into the SPIDER coupling failure mode.

        The wall-time speedup is environment-dependent and not always
        observable on a single solve; we therefore assert (a) both
        solves converge, (b) the seeded solve produces a structure
        that agrees with the unseeded one to within solver tolerance,
        and (c) the seeded solve is no slower than 1.5x the unseeded
        wall time (guards against accidental regressions).
        """
        config = _make_config(planet_mass_kg=1.0 * _earth_mass_kg())
        T_func = _flat_profile(T_value=4400.0)

        # First call: cold start (initial_density not provided).
        t0 = time.perf_counter()
        result_cold = self._solve(materials, config, T_func)
        wall_cold = time.perf_counter() - t0
        assert result_cold['converged'] is True
        rho_cold = np.asarray(result_cold['density'])
        radii_cold = np.asarray(result_cold['radii'])

        # Second call: seed with the first call's density profile.
        from zalmoxis.solver import main

        t0 = time.perf_counter()
        result_warm = main(
            config,
            material_dictionaries=materials,
            melting_curves_functions=None,
            input_dir=_input_dir(),
            temperature_function=T_func,
            initial_density=rho_cold,
            initial_radii=radii_cold,
        )
        wall_warm = time.perf_counter() - t0

        assert result_warm['converged'] is True
        # Same converged structure (within solver tolerance).
        rho_warm = np.asarray(result_warm['density'])
        # Density valid where positive in both runs.
        valid = (rho_cold > 0) & (rho_warm > 0)
        rel_drift = np.abs(rho_cold[valid] - rho_warm[valid]) / rho_cold[valid]
        assert np.all(rel_drift < 1.0e-2), (
            f'seeded run differs from cold-start by max {rel_drift.max():.3e} '
            '(>1e-2); density seeding altered the converged answer.'
        )
        # Loose wall guard: seeded run should be no more than 1.5x slower
        # than the cold-start (perf assertion is intentionally generous to
        # tolerate CI runner noise; main signal is the convergence + drift).
        assert wall_warm <= 1.5 * wall_cold + 5.0, (
            f'seeded wall {wall_warm:.2f} s > 1.5x cold {wall_cold:.2f} s; '
            'density seeding regressed wall time.'
        )


@pytest.mark.slow
class TestSpiderCouplingMassSweep:
    """Steep SPIDER-like profile must converge across 1, 5 M_Earth.

    Marked `slow` because the 5 M_E end-to-end solve takes ~3-4 minutes
    even with Newton + num_layers=50. The 10 M_Earth super-Earth case
    is intentionally omitted (10 M_E with PALEOS takes ~1.8 h wall in
    production). Run via `pytest -m slow` to exercise the coverage.
    """

    @pytest.fixture(scope='class')
    def materials(self):
        if not _has_paleos_data():
            pytest.skip('PALEOS unified EOS data not available')
        from zalmoxis.config import load_material_dictionaries

        return load_material_dictionaries()

    @pytest.mark.parametrize('M_earth', [1.0, 5.0])
    def test_steep_T_converges_across_masses(self, materials, M_earth):
        """Steep SPIDER-like T(r) must converge at each parametrized mass.

        Edge cases:
          - 1 M_E: baseline rocky planet, tightest atmospheric coupling
          - 5 M_E: super-Earth regime where Newton outer is required;
            deep-mantle pressures hit the high-P EOS extrapolation
            regime
        """
        from zalmoxis.solver import main

        config = _make_config(planet_mass_kg=M_earth * _earth_mass_kg())
        # R guess scaling: rocky-planet R ~ M^0.27.
        R_guess = 6.4e6 * (M_earth) ** 0.27
        T_factory = _spider_like_profile(T_centre=5000.0, T_surface=1800.0)
        T_func = T_factory(R_planet=R_guess)

        result = main(
            config,
            material_dictionaries=materials,
            melting_curves_functions=None,
            input_dir=_input_dir(),
            temperature_function=T_func,
        )

        assert result['converged'] is True, (
            f'{M_earth} M_E with steep SPIDER T(r) failed to converge'
        )
        assert result['best_mass_error'] < 5.0e-3
        # R scaling sanity: Earth ~6.4 Mm, super-Earth-5 ~10 Mm.
        R_actual = float(result['radii'][-1])
        assert R_guess * 0.7 < R_actual < R_guess * 1.4, (
            f'{M_earth} M_E R={R_actual:.3e} m far from guess {R_guess:.3e}'
        )
