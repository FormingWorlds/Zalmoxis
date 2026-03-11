# Testing

Zalmoxis uses [pytest](https://docs.pytest.org/) with [pytest-xdist](https://pytest-xdist.readthedocs.io/) for parallel execution. Tests are organized by speed and purpose into three categories using pytest markers.

## Prerequisites

Install the development dependencies (includes pytest, pytest-xdist, ruff, and coverage tools):

```console
pip install -e ".[develop]"
```

The `ZALMOXIS_ROOT` environment variable must be set (see [Installation](installation.md#step-3-set-the-environment-variable)). Tests validate this at session start and exit immediately if it is missing.

## Running tests

### By marker

Tests are categorized with markers that reflect their runtime and purpose:

| Marker | Tests | Runtime | What it validates |
|---|---|---|---|
| `unit` | 27 | < 2 s | EOS functions, edge cases, analytic vs. tabulated consistency. No solver calls. |
| `integration` | 29 | ~10--20 min | Full solver runs: mass-radius relations, density profiles, analytic vs. tabulated MR, T-dependent EOS convergence. |
| `slow` | 2 | ~30+ min | Ternary composition grid sweep across core/mantle/water fractions. |

Run a specific category:

```console
pytest -m unit                # Fast feedback during development
pytest -m integration         # Full solver validation
pytest -m slow                # Composition grid sweep (pre-release)
pytest -m "not slow"          # Everything except the slow grid sweep
pytest -m "unit or integration"  # All except slow
```

### All tests

```console
pytest                        # Runs all 58 tests in parallel
```

The default configuration (`pyproject.toml`) includes `-n auto --dist loadfile`, which distributes test files across CPU cores for parallel execution via pytest-xdist.

### Single test

```console
pytest src/tests/test_MR.py::test_mass_radius[rocky-1]
```

### Without parallelization

If you need serial execution (e.g., for debugging):

```console
pytest -o "addopts=-ra -v" -m unit
```

This overrides the default `-n auto` flag.

## Test suite overview

### Unit tests (`test_analytic_eos.py`)

Validates the Seager et al. (2007) analytic modified polytropic EOS (`get_analytic_density()`) without invoking the planetary structure solver:

- **Basic correctness**: zero-pressure limit, monotonicity, iron at Earth-center pressure, cross-material density ordering.
- **Edge cases**: invalid material key, negative/zero/NaN pressure, high-pressure warning.
- **Analytic vs. tabulated**: compares the analytic fit against tabulated EOS data for iron, MgSiO3, and H2O within 15% tolerance.

### Integration tests

Four test files run the full 3-level nested solver (structure ODE + density-pressure Picard iteration + mass-radius outer loop):

**`test_MR.py`** -- Mass-radius validation against [Zeng et al. (2019)](https://lweb.cfa.harvard.edu/~lzeng/planetmodels.html) for rocky and water planets at 1, 5, 10, and 50 $M_\oplus$. Tolerance: 3% relative error.

**`test_Seager.py`** -- Density profile validation against [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) radial profiles for rocky and water planets. Masks the core-mantle boundary discontinuity; tolerance: 24% relative error.

**`test_analytic_MR.py`** -- Compares analytic EOS against tabulated EOS using the full solver:

- Rocky planet (iron/MgSiO3): analytic vs. tabulated radii within 5%.
- Water planet (iron/MgSiO3/H2O): same comparison for 3-layer models.
- Exotic compositions (iron/SiC, iron/graphite): convergence check.
- Mixed EOS (tabulated core + analytic mantle): consistency within 5%.

Uses a session-scoped solver cache (see [Fixtures](#fixtures)) to avoid redundant solver runs when the same configuration appears in multiple test classes.

**`test_convergence_TdepEOS.py`** -- Validates convergence of the temperature-dependent Wolf & Bower (2018) EOS for 1 and 2 $M_\oplus$ planets. Checks physically plausible density ranges: iron core 8,000--50,000 kg/m$^3$, MgSiO3 mantle 2,000--8,000 kg/m$^3$.

### Slow tests (`test_convergence.py`)

Sweeps a ternary composition grid (core, mantle, water mass fractions) at `step=0.2` for 1 and 10 $M_\oplus$. This produces ~21 grid points per mass value, testing convergence across corners (pure compositions), edges (binary mixtures), and interior points of the composition triangle. Verifies that all physically valid compositions converge.

The step size is configurable via `run_ternary_grid_for_mass(mass, step=...)`. The default for the plotting code is `step=0.05` (~231 grid points); the test uses `step=0.2` for faster CI feedback while still covering the composition space.

## Test file structure

```
src/tests/
├── conftest.py                   # Shared fixtures (env validation, solver cache)
├── test_analytic_eos.py          # Unit tests (27 tests, no solver)
├── test_analytic_MR.py           # Integration: analytic vs. tabulated MR
├── test_MR.py                    # Integration: mass-radius vs. Zeng (2019)
├── test_Seager.py                # Integration: density profiles vs. Seager (2007)
├── test_convergence_TdepEOS.py   # Integration: T-dependent EOS convergence
└── test_convergence.py           # Slow: ternary composition grid sweep
```

Test helpers (solver wrappers, reference data loaders) are in `src/tools/setup_tests.py`.

## Fixtures

Shared fixtures are defined in `src/tests/conftest.py`:

### `_validate_environment` (autouse, session)

Validates that `ZALMOXIS_ROOT` is set at the start of the test session. Calls `pytest.exit()` if missing, providing a clear error message instead of per-file `RuntimeError` raises.

### `zalmoxis_root` (session)

Returns the `ZALMOXIS_ROOT` path. Used by tests that need the repository root for log file paths or data directories.

### `cached_solver` (session)

A session-scoped callable that wraps `run_zalmoxis_rocky_water()` with transparent caching. Identical parameter combinations (mass, config type, composition fractions, EOS override) return the same output file paths without re-running the solver.

This eliminates redundant solver runs when multiple test classes share a baseline configuration. For example, `TestAnalyticVsTabulatedMR` and `TestMixedEOS` both need the tabulated rocky baseline at each mass -- the cache ensures it runs once.

The cache key is `(mass, config_type, cmf, immf, eos_override_tuple)`. With `--dist loadfile` (the default), all tests in a file share one xdist worker and therefore one cache instance.

## Parallelization

Tests run in parallel via pytest-xdist with `--dist loadfile`, which groups all tests from the same file onto one worker. This ensures:

- The `cached_solver` fixture works correctly (session-scoped per worker, all tests in a file share one worker).
- Module-level imports and setup run once per file, not per test.
- Different test files run concurrently on separate CPU cores.

With 6 test files and `loadfile` distribution, the suite uses up to 6 workers. The bottleneck is typically the slowest file (`test_convergence.py`).

## Linting

Before committing, format and check all test files:

```console
ruff check --fix src/tests/
ruff format src/tests/
```

## Coverage

```console
pytest --cov=src --cov-report=html -m "not slow"
```

Open `htmlcov/index.html` to inspect line-by-line coverage.
