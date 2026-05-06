# Testing suite

[![codecov](https://codecov.io/gh/FormingWorlds/Zalmoxis/graph/badge.svg)](https://codecov.io/gh/FormingWorlds/Zalmoxis)

This page is about *running* the existing test suite. For guidance on *writing*
new tests see [How to build tests](build_tests.md).

Zalmoxis uses [pytest](https://docs.pytest.org/) with
[pytest-xdist](https://pytest-xdist.readthedocs.io/) for parallel execution.
Tests are categorised by speed and purpose into four pytest markers.

## Prerequisites

Install the development extras (pytest, pytest-xdist, pytest-cov, ruff):

```console
pip install -e ".[develop]"
```

`ZALMOXIS_ROOT` is auto-detected by the package. If auto-detection fails, set
it explicitly (see [Installation](installation.md#step-3-set-the-environment-variable)).
Tests that use mocked EOS functions, including the first-principles tier, run
without `ZALMOXIS_ROOT` being set.

## Markers

| Marker | Tests | Wall | Scope |
|---|---|---|---|
| `unit` | ~1026 | ~1.5 min | EOS helpers, config validation, mixing, binodal, melting curves, PALEOS loaders, JAX parity, structure-model branches. No real solver call (a handful of analytic-EOS smoke tests excepted). |
| `smoke` | ~23 | ~5 to 10 min | Single 1 $M_\oplus$ full-solver runs that exercise the whole code path under relaxed cost. |
| `integration` | ~2 | ~10 to 20 min | PALEOS rocky 1 + 5 $M_\oplus$ against published references. |
| `slow` | ~44 | ~30+ min each | Composition grid sweeps and grid/tolerance convergence studies. Manual only. |

Total collected: ~1077 tests. The exact counts drift as new branches are
covered; `pytest -o "addopts=" --collect-only -m <marker>` reports the live
number.

## Running tests

### By marker

```console
pytest -m unit                       # Fast feedback during development
pytest -m smoke                      # Single-mass full-solver smoke
pytest -m integration                # Published-reference comparisons
pytest -m "(unit or smoke or integration) and not slow"   # Full nightly tier
pytest -m slow                       # Pre-release composition sweeps
```

### Single test

```console
pytest tests/test_MR_rocky.py::test_rocky_1Mearth_vs_zeng_and_seager
```

### Without parallelization

The default `addopts` in `pyproject.toml` includes `-n auto --dist loadfile`,
which distributes test files across CPU cores. To force serial execution
(useful when debugging a flaky test), override `addopts`:

```console
pytest -o "addopts=-ra -v" -m unit
```

The `-o "addopts="` form replaces the default; this is also how the CI
matrix runs the unit tier without xdist contention on small runners.

## CI tiers

| Trigger | Markers | Budget | Coverage |
|---|---|---|---|
| Push / PR (`CI.yml`) | `unit and not slow` | < 5 min | None |
| Nightly cron (`nightly.yml`, 02:00 UTC) | `(unit or smoke or integration) and not slow` | < 60 min | Yes; uploaded to Codecov with the `nightly` flag |
| Manual `workflow_dispatch` | as above | < 60 min | Yes |

Push CI is intentionally unit-only because each smoke or integration test
runs a full Zalmoxis solver call (5 to 10 min on a 2-vCPU runner under
coverage instrumentation). Burning that on every push gives no bug-finding
signal that the unit tier doesn't already cover.

## Fixtures

Shared fixtures are defined in `tests/conftest.py` and the helpers package
in `tests/_paleos_helpers.py` and `tests/_paleos_mock.py`.

### `zalmoxis_root` (session)

Returns the Zalmoxis root path via `get_zalmoxis_root()`. Skips the test if
the root cannot be determined (auto-detection fails and `ZALMOXIS_ROOT` is
not set).

### `cached_solver` (session)

A session-scoped callable that wraps the rocky and water full-solver runs
with transparent caching keyed by
`(mass, config_type, cmf, immf, eos_override_tuple)`. With `--dist loadfile`
all tests in one file share an xdist worker and therefore one cache, so
identical parameter combinations re-use the same output without re-running
the solver.

## Parallelization

`--dist loadfile` groups all tests from the same file onto one worker. This
ensures (a) the `cached_solver` fixture works correctly because it is
session-scoped per worker, (b) module-level imports and setup run once per
file, (c) different test files run concurrently on separate cores.

## Coverage

```console
pytest -o "addopts=" --cov=zalmoxis --cov-report=html -m "(unit or smoke or integration) and not slow"
```

Open `htmlcov/index.html` to inspect line-by-line coverage. The nightly CI
uses `--cov-report=xml` and uploads the result to Codecov.

## Linting

Before committing, format and check all files:

```console
ruff check --fix src/ tests/ tools/
ruff format src/ tests/ tools/
```

The local ruff (often 0.12.x) and the CI ruff (0.15.x) sometimes disagree on
formatting drift; CI is canonical.
