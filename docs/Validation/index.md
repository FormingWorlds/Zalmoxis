# Validation

## What this page checks

Numerical codes for planetary interior structure involve several layers of approximation: the ODE integrator introduces truncation errors, the iterative density-pressure coupling (Picard iteration) may not fully converge, the Brent root-finder for the central pressure operates within a finite tolerance, and the equation of state tables introduce interpolation error. When these components are assembled into a coupled solver, it becomes difficult to attribute any discrepancy to a specific source. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. Each case on the pages of this tab says why the check exists, what it tests, how we know the right answer and why it matters, and ends with a box "Test details".

First-principles verification addresses this by testing each component against problems with known analytical solutions. If the code reproduces an exact result to the expected numerical precision, that component passes the check. If it does not, the test isolates where the error is introduced.

The verification strategy follows a tiered approach:

1. **Tier 1 (unit tests):** Test the ODE integrator in isolation by patching the density function to return a constant. No EOS data files, no Picard iteration, no Brent solver. Pure mathematics.
2. **Tier 2 (smoke tests):** Test the full solver chain (ODE + Picard + Brent + mass-radius loop) using the Seager et al. (2007) analytic EOS, which is a closed-form formula requiring no tabulated data. Compare against known physical benchmarks (Earth's radius, gravity, central pressure) and scaling laws.
3. **Tier 3 (convergence tests):** Check the error over several grids and solver tolerances, and the full-solver radius on two grids.

All tests are implemented in [`tests/test_first_principles.py`](https://github.com/FormingWorlds/Zalmoxis) (25 pytest tests). A standalone plotting script at `tools/validation/run_first_principles_validation.py` makes PDF versions of the first-principles figures on the [hydrostatic structure](hydrostatic_structure.md) and [mass-radius relation](mass_radius.md) pages.

## Cases by process

| Process | Case | Test id | Reference | Status |
|---|---|---|---|---|
| [Hydrostatic structure](hydrostatic_structure.md) | Uniform-density sphere | `tests/test_first_principles.py::TestUniformDensitySphere` | closed-form solution | unit tier; script figure |
| [Hydrostatic structure](hydrostatic_structure.md) | Two-layer sphere | `tests/test_first_principles.py::TestTwoLayerSphere` | piecewise closed-form solution | unit tier; script figure |
| [Hydrostatic structure](hydrostatic_structure.md) | Conservation diagnostics | `tests/test_first_principles.py::TestGaussLaw`, `::TestHydrostaticBalance` | Gauss's law, hydrostatic balance | unit tier; script figure |
| [Hydrostatic structure](hydrostatic_structure.md) | Gravitational binding energy | `tests/test_first_principles.py::TestGravitationalBindingEnergy` | $-\frac{3}{5} G M_p^2 / R$ | unit tier |
| [Hydrostatic structure](hydrostatic_structure.md) | n = 1 polytrope | `tests/test_polytrope_verification.py` | analytic Lane-Emden solution | smoke tier |
| [Hydrostatic structure](hydrostatic_structure.md) | Numerical convergence | `tests/test_first_principles.py::TestGridConvergence`, `::TestToleranceConvergence` | closed-form solution of the uniform sphere | slow tier; script figure |
| [Hydrostatic structure](hydrostatic_structure.md) | JAX and numpy right-hand sides | `tests/test_jax_unified_mantle_parity.py::test_unified_mantle_dry_parity` and 5 more | numpy `coupled_odes` (cross-code check) | unit and integration tiers; plot pair |
| [Mass-radius relation](mass_radius.md) | Earth benchmark | `tests/test_first_principles.py::TestEarthBenchmark` | Earth radius, surface gravity, central pressure | smoke tier; script figure |
| [Mass-radius relation](mass_radius.md) | Mass-radius scaling and CMF sensitivity | `tests/test_first_principles.py::TestMassRadiusScaling`, `::TestCMFMonotonicity`, `::TestPureIronPlanet`, `::TestConservationLaws` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) | smoke tier; script figures |
| [Equation of state](equation_of_state.md) | PALEOS table loader | `tests/test_eos_export_real_tables.py`, `tests/test_eos_export_highres.py` | `numpy.genfromtxt` reading of the same tables (cross-code check) | integration and slow tiers |
| [Equation of state](equation_of_state.md) | Liquid Fe and Fe-water members | `tests/test_core_alloy.py::TestLuoMembers::test_eq14_density_matches_an_independent_bisection` | [Luo et al. (2024)](https://doi.org/10.1038/s41550-024-02347-z) Eq. 14 (analytical) | unit tier; plot pair |
| [Equation of state](equation_of_state.md) | Light-element core on PALEOS iron | `tests/test_core_alloy.py::TestPaleosCore::test_core_mixture_limits_on_paleos_iron` | volume-additive limits (analytical) | unit tier; plot pair |
| [Thermal state](thermal_state.md) | Initial thermal state | none | named in the script docstrings | script figure, no test, not run in CI |

## Summary of the first-principles cases

| Test | Target | Key metric | Result |
|------|--------|------------|--------|
| 1. Uniform sphere | ODE integration | $\|M_\mathrm{num} - M_\mathrm{exact}\|/M_\mathrm{exact}$ | $3 \times 10^{-15}$ (test: $< 10^{-6}$) |
| 2. Two-layer sphere | Layer transitions | $\|M_\mathrm{num} - M_\mathrm{exact}\|/M_\mathrm{exact}$ | $3 \times 10^{-9}$ (test: $< 10^{-5}$) |
| 3. Conservation | Gauss + hydrostatic | Pointwise residuals | $\le 1.4 \times 10^{-13}$ (test: $< 10^{-5}$, $< 10^{-4}$) |
| 4. Binding energy | Energy integral | $\|E_\mathrm{num} - E_\mathrm{exact}\|/\|E_\mathrm{exact}\|$ | $7 \times 10^{-6}$ (test: $< 1\%$) |
| 5. Earth benchmark | Full solver | $R_p$, $P_c$, $g_\mathrm{surf}$ against 6371 km, 365 GPa, 9.81 m/s$^2$ | 4%, 7%, 9% |
| 6. M-R scaling | Physics | $\alpha$ in $R \propto M^\alpha$ | 0.293 (expected $\sim 0.27$) |
| 7. Convergence | Numerics | Mass error at $N = 25$ to 800 | $2$ to $3 \times 10^{-15}$ (round-off) |

Together, these seven tests establish that:

- The ODE integrator reproduces the exact uniform-sphere solution to round-off and the two-layer solution to $3 \times 10^{-9}$ (Tests 1-2).
- Conservation laws (Gauss's law, hydrostatic equilibrium) hold pointwise to round-off for the uniform sphere (Test 3).
- The binding energy agrees with the analytic value to $7 \times 10^{-6}$ (Test 4).
- The full solver chain (Picard + Brent + mass-radius) reproduces Earth to 4-9% with a simplified EOS (Test 5).
- Physical scaling laws and monotonicity constraints are respected (Test 6).
- The uniform-sphere error stays at round-off level over every grid and tolerance tested (Test 7).

These tests are independent of any tabulated EOS data, melting curves, or coupled atmosphere models. They verify the numerical machinery of Zalmoxis from first principles.

## Plots written by the tests

A case with a plot pair shows the tested values against their reference in a light and a dark version. The test writes the pair with the same numbers it asserts, through the `validation_plot` fixture of proteus-mpl. This command writes all pairs to `docs/assets/validation/`:

```console
pytest -m reference_pinned tests --validation-plots
```

Without `--validation-plots` the tests assert and write nothing. The command makes every plot again; commit only the pair of the case you changed, because the bytes match only for the same matplotlib and Pillow versions. The other figures on these pages are script figures: PNG files made by the scripts under `tools/validation/` and kept in `docs/img/validation/`. No test writes or checks them, and they are not limited to 60 kB.
