# Validation

## What this page checks

Numerical codes for planetary interior structure involve several layers of approximation: the ODE integrator introduces truncation errors, the iterative density-pressure coupling (Picard iteration) may not fully converge, the Brent root-finder for the central pressure operates within a finite tolerance, and the equation of state tables introduce interpolation error. When these components are assembled into a coupled solver, it becomes difficult to attribute any discrepancy to a specific source. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance.

First-principles verification addresses this by testing each component against problems with known analytical solutions. If the code reproduces an exact result to the expected numerical precision, that component passes the check. If it does not, the test isolates where the error is introduced.

The verification has 3 steps, each in one pytest tier:

1. **Integrator alone (unit tier):** Test the ODE integrator in isolation by patching the density function to return a constant. No EOS data files, no Picard iteration, no Brent solver. Pure mathematics.
2. **Full solver with the analytic EOS (smoke tier):** Test the full solver chain (ODE + Picard + Brent + mass-radius loop) using the Seager et al. (2007) analytic EOS, which is a closed-form formula requiring no tabulated data. Compare against known physical benchmarks (Earth's radius, gravity, central pressure) and scaling laws.
3. **Grids and tolerances (slow tier):** Check the error over several grids and solver tolerances, and the full-solver radius on 3 grids.

All tests are implemented in [`tests/test_first_principles.py`](https://github.com/FormingWorlds/Zalmoxis/blob/main/tests/test_first_principles.py) (25 pytest tests). The tests of these cases draw their own plot pairs.

## Cases by process

| Process | Case | Test id | Reference | Status |
|---|---|---|---|---|
| [Hydrostatic structure](hydrostatic_structure.md) | Uniform-density sphere | `tests/test_first_principles.py::TestUniformDensitySphere` | closed-form solution | unit tier; plot pair |
| [Hydrostatic structure](hydrostatic_structure.md) | Two-layer sphere | `tests/test_first_principles.py::TestTwoLayerSphere` | piecewise closed-form solution | unit tier; plot pair |
| [Hydrostatic structure](hydrostatic_structure.md) | Conservation diagnostics | `tests/test_first_principles.py::TestGaussLaw`, `::TestHydrostaticBalance` | Gauss's law, hydrostatic balance | unit tier; plot pair |
| [Hydrostatic structure](hydrostatic_structure.md) | Gravitational binding energy | `tests/test_first_principles.py::TestGravitationalBindingEnergy` | $-\frac{3}{5} G M_p^2 / R$ | unit tier; plot pair |
| [Hydrostatic structure](hydrostatic_structure.md) | n = 1 polytrope | `tests/test_polytrope_verification.py` | analytic Lane-Emden solution | smoke tier; plot pair |
| [Hydrostatic structure](hydrostatic_structure.md) | Grid and tolerance checks | `tests/test_first_principles.py::TestGridConvergence`, `::TestToleranceConvergence` | closed-form solution of the uniform sphere | slow tier; plot pair |
| [Hydrostatic structure](hydrostatic_structure.md) | JAX and numpy right-hand sides | `tests/test_jax_unified_mantle_parity.py::test_unified_mantle_dry_parity` and 5 more | numpy `coupled_odes` (cross-code check) | unit and integration tiers; plot pair |
| [Hydrostatic structure](hydrostatic_structure.md) | Surface boundary condition of the pressure solve | `tests/test_pressure_root.py::test_the_refined_root_meets_the_surface_boundary_condition`, `tests/test_pressure_root_solves.py::test_the_accepted_root_meets_the_surface_boundary_condition` | boundary condition $P(R) = P_t$ | unit and smoke tiers; plot pair |
| [Hydrostatic structure](hydrostatic_structure.md) | Surface boundary condition in every Newton setup | `tests/test_pressure_root_solves.py::test_every_newton_setup_meets_the_surface_boundary_condition` | boundary condition $P(R) = P_t$ | nightly job `accept_solves_newton_setups`; script figure |
| [Hydrostatic structure](hydrostatic_structure.md) | Inner-core boundary of a two-layer core | `tests/test_core_two_phase_solve.py::test_the_adiabat_of_a_converged_liquidus_solve_takes_its_icb` and 8 more | root of $T - T_\mathrm{liq}$ on the returned profile (self-consistency); regression pins | unit and integration tiers; plot pair |
| [Mass-radius relation](mass_radius.md) | Earth benchmark | `tests/test_first_principles.py::TestEarthBenchmark` | Earth radius, surface gravity, central pressure | smoke tier; plot pair |
| [Mass-radius relation](mass_radius.md) | Mass-radius scaling and CMF sensitivity | `tests/test_first_principles.py::TestMassRadiusScaling`, `::TestCMFMonotonicity`, `::TestPureIronPlanet`, `::TestConservationLaws` | Seager et al. (2007)[^cite-seager2007] | smoke tier; plot pair |
| [Mass-radius relation](mass_radius.md) | A two-layer Earth core against PREM | `tests/test_core_two_phase_solve.py::test_an_earth_core_against_prem` | PREM of Dziewonski & Anderson (1981)[^cite-dziewonski1981] (seismic model; regression pins) | integration tier; 2 plot pairs |
| [Equation of state](equation_of_state.md) | PALEOS table loader | `tests/test_eos_export_real_tables.py`, `tests/test_eos_export_highres.py` | `numpy.genfromtxt` reading of the same tables (cross-code check) | integration and slow tiers; no plot (identity of 2 readers) |
| [Equation of state](equation_of_state.md) | Liquid Fe and Fe-water members | `tests/test_core_alloy.py::TestLuoMembers::test_eq14_density_matches_an_independent_bisection` | Luo et al. (2024)[^cite-luo2024] Eq. 14 (analytical) | unit tier; plot pair |
| [Equation of state](equation_of_state.md) | Light-element core on PALEOS iron | `tests/test_core_alloy.py::TestPaleosCore::test_core_mixture_limits_on_paleos_iron` | volume-additive limits (analytical) | unit tier; plot pair |
| [Equation of state](equation_of_state.md) | Light-element cores against the outer core | `tests/test_core_alloy.py::TestHiroseCalibration` | Hirose et al. (2021)[^cite-hirose2021], p. 3 (published estimate) | unit tier; plot pair |
| [Equation of state](equation_of_state.md) | Light-element outer cores on the liquid iron side | `tests/test_core_alloy.py::TestHiroseCalibration::test_the_outer_cores_on_the_liquid_iron_side` | PALEOS liquid of Luo et al. (2024)[^cite-luo2024] and Hirose et al. (2021)[^cite-hirose2021], p. 3 (consistency check) | unit tier; plot pair |
| [Thermal state](thermal_state.md) | Initial thermal state | none | named in the script docstrings | script figure, no test, not run in CI |

## Summary of the first-principles cases

| Case | Target | Key metric | Result |
|------|--------|------------|--------|
| Uniform-density sphere | ODE integration | $\|M_\mathrm{num} - M_\mathrm{exact}\|/M_\mathrm{exact}$ | $2 \times 10^{-15}$ (test: $< 10^{-6}$) |
| Two-layer sphere | Layer transitions | $\|M_\mathrm{num} - M_\mathrm{exact}\|/M_\mathrm{exact}$ | $3 \times 10^{-9}$ (test: $< 10^{-5}$) |
| Conservation diagnostics | Gauss + hydrostatic | Pointwise residuals | $\le 1.45 \times 10^{-13}$ (test: $< 10^{-5}$, $< 10^{-4}$) |
| Binding energy | Energy integral | $\|E_\mathrm{num} - E_\mathrm{exact}\|/\|E_\mathrm{exact}\|$ | $7 \times 10^{-6}$ (test: $< 1\%$) |
| Earth benchmark | Full solver | $R_p$, $P_c$, $g_\mathrm{surf}$ against 6371 km, 365 GPa, 9.81 m/s$^2$ | 4%, 7%, 9% |
| Mass-radius scaling | Physics | $\alpha$ in $R \propto M^\alpha$, fit from 0.5 to 10 $M_\oplus$ | 0.289 (test: $0.27 \pm 0.04$) |
| Grid and tolerance checks | Numerics | Mass error at $N = 50$ to 400 | $2$ to $3 \times 10^{-15}$ (round-off) |

Together, these 7 cases establish that:

- The ODE integrator reproduces the exact uniform-sphere solution to round-off and the two-layer solution to $3 \times 10^{-9}$
- Conservation laws (Gauss's law, hydrostatic equilibrium) hold pointwise to round-off for the uniform sphere
- The binding energy agrees with the analytic value to $7 \times 10^{-6}$
- The full solver chain (Picard + Brent + mass-radius) reproduces Earth to 4-9% with a simplified EOS
- Physical scaling laws and monotonicity constraints are respected
- The uniform-sphere error stays at round-off level over every grid and tolerance tested

These tests are independent of any tabulated EOS data, melting curves, or coupled atmosphere models. They verify the numerical machinery of Zalmoxis from first principles.

## Plots written by the tests

A case with a plot pair shows the tested values against their reference in a light and a dark version. The test writes the pair with the same numbers it asserts, or with a second solve of the same case where the note says so, through the `validation_plot` fixture of proteus-mpl. This command writes all pairs to `docs/assets/validation/`:

```console
pytest -o "addopts=" -m reference_pinned tests --validation-plots
```

The first-principles cases are not `reference_pinned`; this command writes their pairs:

```console
pytest -o "addopts=" tests/test_first_principles.py tests/test_polytrope_verification.py --validation-plots
```

Without `--validation-plots` the tests write nothing: each compares the data of its plot with the committed pair, and a missing or stale file fails the test. Two kinds of script figure appear too. The sweep over 26 setups is an SVG pair in `docs/assets/validation/` (`pressure_root_sweep*.svg`), which `tools/validation/pressure_root_sweep.py` writes through `proteus_mpl.validation.save` with its 60 kB cap; its tool test checks the plotting code on stand-in records, not the committed pair. The 4 thermal-state figures are PNG files in `docs/img/validation/`, made by scripts under `tools/validation/`; no test writes or checks them, and they are not limited to 60 kB.

[^cite-dziewonski1981]: A. M. Dziewonski, D. L. Anderson, *[Preliminary reference Earth model](https://doi.org/10.1016/0031-9201(81)90046-7)*, Physics of the Earth and Planetary Interiors, 25, 297-356, 1981. [SciX](https://scixplorer.org/abs/1981PEPI...25..297D/abstract).
[^cite-hirose2021]: K. Hirose, B. Wood, L. Vočadlo, *[Light elements in the Earth's core](https://doi.org/10.1038/s43017-021-00203-6)*, Nature Reviews Earth & Environment, 2, 645-658, 2021. [SciX](https://scixplorer.org/abs/2021NRvEE...2..645H/abstract).
[^cite-luo2024]: H. Luo, C. Dorn, J. Deng, *[The interior as the dominant water reservoir in super-Earths and sub-Neptunes](https://doi.org/10.1038/s41550-024-02347-z)*, Nature Astronomy, 8, 1399-1407, 2024. [SciX](https://scixplorer.org/abs/2024NatAs...8.1399L/abstract).
[^cite-seager2007]: S. Seager, M. Kuchner, C. A. Hier-Majumder, B. Militzer, *[Mass-Radius Relationships for Solid Exoplanets](https://doi.org/10.1086/521346)*, The Astrophysical Journal, 669(2), 1279-1297, 2007. [SciX](https://scixplorer.org/abs/2007ApJ...669.1279S/abstract).
