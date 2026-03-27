# Testing

[![codecov](https://codecov.io/gh/FormingWorlds/Zalmoxis/graph/badge.svg)](https://codecov.io/gh/FormingWorlds/Zalmoxis)

Zalmoxis uses [pytest](https://docs.pytest.org/) with [pytest-xdist](https://pytest-xdist.readthedocs.io/) for parallel execution. Tests are organized by speed and purpose into three categories using pytest markers.

## Prerequisites

Install the development dependencies (includes pytest, pytest-xdist, ruff, and coverage tools):

```console
pip install -e ".[develop]"
```

`ZALMOXIS_ROOT` is auto-detected by the package. If auto-detection fails, set it explicitly (see [Installation](installation.md#step-3-set-the-environment-variable)). Unit tests that use mocked EOS functions (e.g., the first-principles tests) can run without `ZALMOXIS_ROOT` being set.

## Running tests

### By marker

Tests are categorized with markers that reflect their runtime and purpose:

| Marker | Tests | Runtime | What it validates |
|---|---|---|---|
| `unit` | ~435 | ~1.5 min | EOS functions, config validation, mixing, binodal suppression, melting curves, PALEOS loading, adiabatic profiles, first-principles ODE verification (analytic solutions). No solver calls (except first-principles tests that use the Analytic EOS). |
| `integration` | ~40 | ~10 to 20 min | Full solver runs: mass-radius relations, density profiles, analytic vs. tabulated MR, T-dependent EOS convergence, PALEOS convergence, RTPress100TPa mass limits. |
| `slow` | ~4 | ~30+ min | Ternary composition grid sweep, grid/tolerance convergence studies. |

Run a specific category:

```console
pytest -m unit                # Fast feedback during development
pytest -m integration         # Full solver validation
pytest -m slow                # Composition grid sweep (pre-release)
pytest -m "not slow"          # Everything except slow
pytest -m "unit or integration"  # All except slow
```

### All tests

```console
pytest                        # Runs all tests in parallel
```

The default configuration (`pyproject.toml`) includes `-n auto --dist loadfile`, which distributes test files across CPU cores for parallel execution via pytest-xdist.

### Single test

```console
pytest tests/test_MR.py::test_mass_radius[rocky-1]
```

### Without parallelization

If you need serial execution (e.g., for debugging):

```console
pytest -o "addopts=-ra -v" -m unit
```

This overrides the default `-n auto` flag.

## Test suite overview

### First-principles verification (`test_first_principles.py`)

25 tests in 3 tiers that verify the ODE system against exact analytical solutions:

- **Tier 1 (unit, 11 tests)**: Patches `calculate_mixed_density` with constant density to isolate the ODE integrator. Uniform-density sphere (exact $M(r)$, $g(r)$, $P(r)$), two-layer sphere (layer transitions), Gauss's law, hydrostatic balance residual, gravitational binding energy.
- **Tier 2 (integration, 10 tests)**: Full solver with Seager+2007 analytic EOS (no data files needed). Earth benchmark ($R$, $P_c$, $g_{\mathrm{surf}}$), mass-radius scaling exponent, CMF monotonicity, pure iron planet, mass conservation (direct + trapezoidal), surface pressure, Gauss's law.
- **Tier 3 (slow, 4 tests)**: Grid convergence and tolerance convergence to machine precision.

Standalone plotting script: `tools/validation/run_first_principles_validation.py` (7 PDF plots).

### Unit tests

Validates EOS functions, configuration parsing, mixing logic, binodal suppression, melting curves, PALEOS table loading, adiabatic profiles, and profile plotting without invoking the full solver:

- **Analytic EOS** (`test_analytic_eos.py`): zero-pressure limit, monotonicity, iron at Earth-center pressure, cross-material density ordering, edge cases.
- **EOS functions** (`test_eos_functions.py`): fast bilinear interpolation, batch density and nabla_ad lookups, WB2018 and RTPress100TPa EOS paths, temperature profile modes, melting curve loading, PALEOS unified cache.
- **Mixing** (`test_mixing.py`): multi-material parsing, harmonic mean density, batch density calculation, nabla_ad weighted average, sigmoid suppression, binodal factor, per-component nabla_ad routing.
- **Binodal** (`test_binodal.py`): Rogers+2025 and Gupta+2025 binodal suppression, coexistence compositions, Gibbs mixing, critical temperature/pressure curves.
- **Config validation** (`test_config_validation.py`): mass fraction checks, temperature parameter validation, mushy zone factor bounds, EOS compatibility.
- **Zalmoxis core** (`test_zalmoxis_unit.py`): EOS config parsing (new and legacy formats), layer EOS validation, condensed phase parameter validation, multi-material fraction checks.
- **Melting curves** (`test_melting_curves.py`): Monteux and Stixrude solidus/liquidus, PALEOS-liquidus, per-cell clamping behavior.
- **PALEOS 2-phase** (`test_paleos.py`): table registration, loading, density lookup, nabla_ad lookup.
- **PALEOS unified** (`test_paleos_unified.py`): unified table registration, loading, density, mushy zone interpolation.
- **Adiabatic profiles** (`test_adiabatic.py`): surface temperature anchoring, monotonicity, blend convergence.
- **Profile plotting** (`test_plot_profiles.py`): smoke tests for the 6-panel profile plot.

### Integration tests

Run the full 3-level nested solver (structure ODE + density-pressure Picard iteration + mass-radius outer loop):

- **`test_MR.py`**: Mass-radius validation against Zeng et al. (2019) for rocky and water planets at 1, 5, 10, and 50 $M_\oplus$.
- **`test_Seager.py`**: Density profile validation against Seager et al. (2007) radial profiles.
- **`test_analytic_MR.py`**: Analytic vs. tabulated EOS comparison using the full solver.
- **`test_convergence_TdepEOS.py`**: Wolf & Bower (2018) T-dependent EOS convergence.
- **`test_convergence_RTPress100TPa.py`**: RTPress100TPa mass limits and convergence.
- **`test_convergence_PALEOS.py`**: PALEOS EOS convergence across planet masses and compositions.

### Slow tests

- **`test_convergence.py`**: Ternary composition grid sweep at `step=0.2` for 1 and 10 $M_\oplus$.
- **`test_first_principles.py`** (slow tier): Grid and tolerance convergence studies.

## Test file structure

```
tests/
├── conftest.py                        # Shared fixtures (lazy env, solver cache)
├── test_first_principles.py           # First-principles ODE verification (25 tests)
├── test_analytic_eos.py               # Unit: analytic EOS correctness and edge cases
├── test_eos_functions.py              # Unit: EOS functions (bilinear, batch, T-dep)
├── test_mixing.py                     # Unit: multi-material mixing
├── test_binodal.py                    # Unit: binodal suppression
├── test_config_validation.py          # Unit: config validation
├── test_zalmoxis_unit.py              # Unit: core solver helpers
├── test_melting_curves.py             # Unit: melting curves
├── test_paleos.py                     # Unit: PALEOS 2-phase
├── test_paleos_unified.py             # Unit: unified PALEOS
├── test_adiabatic.py                  # Unit: adiabatic temperature profiles
├── test_plot_profiles.py              # Unit: profile plotting
├── test_analytic_MR.py                # Integration: analytic vs. tabulated MR
├── test_MR.py                         # Integration: mass-radius vs. Zeng (2019)
├── test_Seager.py                     # Integration: density profiles vs. Seager (2007)
├── test_convergence_TdepEOS.py        # Integration: T-dependent EOS convergence
├── test_convergence_RTPress100TPa.py  # Integration: RTPress100TPa mass limits
├── test_convergence_PALEOS.py         # Integration: PALEOS convergence
└── test_convergence.py                # Slow: ternary composition grid sweep
```

Test helpers (solver wrappers, reference data loaders) are in `tools/setup/setup_tests.py`.

## Fixtures

Shared fixtures are defined in `tests/conftest.py`:

### `zalmoxis_root` (session)

Returns the Zalmoxis root path via `get_zalmoxis_root()`. Skips the test if the root cannot be determined (auto-detection fails and `ZALMOXIS_ROOT` is not set).

### `cached_solver` (session)

A session-scoped callable that wraps `run_zalmoxis_rocky_water()` with transparent caching. Identical parameter combinations (mass, config type, composition fractions, EOS override) return the same output file paths without re-running the solver.

The cache key is `(mass, config_type, cmf, immf, eos_override_tuple)`. With `--dist loadfile` (the default), all tests in a file share one xdist worker and therefore one cache instance.

## Parallelization

Tests run in parallel via pytest-xdist with `--dist loadfile`, which groups all tests from the same file onto one worker. This ensures:

- The `cached_solver` fixture works correctly (session-scoped per worker).
- Module-level imports and setup run once per file.
- Different test files run concurrently on separate CPU cores.

## Linting

Before committing, format and check all files:

```console
ruff check --fix src/ tests/ tools/
ruff format src/ tests/ tools/
```

## Coverage

```console
pytest --cov=src --cov-report=html -m "not slow"
```

Open `htmlcov/index.html` to inspect line-by-line coverage.
