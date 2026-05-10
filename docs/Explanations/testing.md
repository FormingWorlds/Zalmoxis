# Testing suite

[![codecov](https://img.shields.io/codecov/c/github/FormingWorlds/Zalmoxis?label=coverage&logo=codecov)](https://app.codecov.io/gh/FormingWorlds/Zalmoxis)
[![Unit Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/Zalmoxis/CI.yml?branch=main&label=Unit%20Tests)](https://github.com/FormingWorlds/Zalmoxis/actions/workflows/CI.yml)
[![Integration Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/Zalmoxis/nightly.yml?branch=main&label=Integration%20Tests)](https://github.com/FormingWorlds/Zalmoxis/actions/workflows/nightly.yml)
[![tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/FormingWorlds/Zalmoxis/main/.github/badges/tests-total.json)](https://proteus-framework.org/testing)


This page is about *running* the existing test suite. For guidance on *writing*
new tests see [How to build tests](build_tests.md).

Tests verify that the code does what was written; physical correctness is
judged by data, not by tests. A test passing tells the developer that the
implementation matches the design intent (the Picard loop converges, the
binodal sigmoid is symmetric, the helpfile schema has the expected columns);
it does not tell anyone that the planet model is right. That judgement
comes from comparing model output against published data and observations,
which is a separate workflow.

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
| `unit` | ~1119 | ~1.5 min | EOS helpers, config validation, mixing, binodal, melting curves, PALEOS loaders, JAX parity, structure-model branches, mocked-solver branch coverage. No real solver call (a handful of analytic-EOS smoke tests excepted). |
| `smoke` | ~23 | ~5 to 10 min | Single 1 $M_\oplus$ full-solver runs that exercise the whole code path under relaxed cost. |
| `integration` | ~2 | ~10 to 20 min | PALEOS rocky 1 + 5 $M_\oplus$ against published references. |
| `slow` | ~44 | ~30+ min each | Composition grid sweeps and grid/tolerance convergence studies. Manual only. |

Total collected: ~1170 tests. The exact counts drift as new branches are
covered; `pytest -o "addopts=" --collect-only -m <marker>` reports the live
number.

The four-marker scheme (`unit`, `smoke`, `integration`, `slow`) is identical
to the PROTEUS main repo's, so a developer moving between Zalmoxis and the
parent project works against one mental model. Zalmoxis additionally enforces
`--strict-markers` and `--strict-config` so a typo'd marker fails the run
instead of silently passing as an "unknown marker" warning.

### Public 2-category scheme

Public-facing badges (README, website) collapse `smoke` + `integration` +
`slow` into a single "Integration Tests" category, because a 4-way taxonomy
is confusing to non-developer readers. The 4-marker internal scheme remains
for CI infrastructure granularity (different timeouts, different schedules,
push vs nightly tier separation).

The ecosystem-wide testing standard, of which the 2-category public scheme
and the 4-marker internal scheme are part, is documented at
[proteus-framework.org/PROTEUS/Explanations/ecosystem_testing_standard/](https://proteus-framework.org/PROTEUS/Explanations/ecosystem_testing_standard/).

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
| Push / PR (`CI.yml`) | `unit and not slow` | < 5 min | None (gate pre-flight only) |
| Nightly cron (`nightly.yml`, 02:00 UTC) | `(unit or smoke or integration) and not slow` | < 60 min | Yes; gates on 95% and uploads to Codecov with the `nightly` flag |
| Manual `workflow_dispatch` | as above | < 60 min | Yes |

Push CI is intentionally unit-only because each smoke or integration test
runs a full Zalmoxis solver call (5 to 10 min on a 2-vCPU runner under
coverage instrumentation). Burning that on every push gives no bug-finding
signal that the unit tier doesn't already cover.

Push and PR CI run a **pre-flight check** on `ubuntu-latest` that compares
`[tool.coverage.report].fail_under` in the PR's `pyproject.toml` against
`origin/main` and fails the PR if the value was decreased. This guards
against accidental relaxation of the coverage gate. The check uses the same
`tomllib`-based comparison idiom as PROTEUS's
`.github/workflows/ci-pr-checks.yml` and is a no-op when the key is missing
on either side.

## Coverage gate

Zalmoxis enforces a hard `--cov-fail-under=95` in the nightly CI invocation.
The threshold lives in `[tool.coverage.report] fail_under` of
`pyproject.toml` and is **not** ratcheted: once Zalmoxis crosses 95%, the
gate is fixed at 95% and never raised again. Aim for **~97% real coverage**
so small future code additions do not trip the gate.

```toml
# pyproject.toml
[tool.coverage.report]
show_missing = true
precision = 2
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
    "if __name__ == .__main__.:",
    "if TYPE_CHECKING:",
    "if typing.TYPE_CHECKING:",
    "@abstractmethod",
    "@abc.abstractmethod",
]
fail_under = 95.0
```

The `exclude_lines` list, the `omit` list under `[tool.coverage.run]`, and
the markers list under `[tool.pytest.ini_options]` all match the PROTEUS
main repo verbatim. The intentional divergence is the threshold style:
PROTEUS uses an auto-ratcheting `fail_under` (handled by
`tools/update_coverage_threshold.py`) that only ever increases, while
Zalmoxis uses a hard gate at 95%. This matches the ecosystem-wide policy
that **95% is the maximum coverage threshold for any PROTEUS module**:
above 95% you are tracking style and pragma usage, not bug-finding signal.

## `# pragma: no cover` usage

Inline `# pragma: no cover` annotations should mark only code that is
genuinely defensive and not productively unit-tested. The exclusion list
above already captures the common cases (`def __repr__`, `if TYPE_CHECKING:`,
`@abstractmethod`, etc.); inline pragmas cover the rest:

- Numerical pathology recovery (LinAlgError on `lstsq`, RuntimeError on a
  non-finite mass evaluation, `brentq` raising on a same-sign bracket).
- Dev-gated diagnostic blocks (e.g. `if _PROFILE:` blocks behind
  `ZALMOXIS_JAX_PROFILE`).
- Heavy-data-dependent paths whose unit-test fixture would be larger than
  the implementation (RTPress100TPa irregular-grid LinearNDInterpolator
  path, for instance).

Do not mark normal-execution code paths. Every inline pragma should carry a
one-line justification:

```python
except np.linalg.LinAlgError:  # pragma: no cover - lstsq with rcond=None is robust; defensive
    return None
```

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

## Local coverage runs

To match the nightly CI measurement locally:

```console
pytest -o "addopts=" --cov=zalmoxis --cov-report=html -m "(unit or smoke or integration) and not slow"
```

Open `htmlcov/index.html` to inspect line-by-line coverage. The nightly CI
uses `--cov-report=xml --cov-fail-under=95` and uploads the result to
Codecov with the `nightly` flag.

Branch coverage (`branch = true` in `[tool.coverage.run]`) is on by default,
matching PROTEUS. The `omit` list excludes `tests/`, `test_*.py`,
`__pycache__/`, and `conftest.py` from the percentage so coverage reflects
the production code only.

## Test count badges

Three shields.io endpoint-badge JSON files live in `.github/badges/`:

- `tests-total.json`: count of all tests excluding `skip`.
- `tests-unit.json`: count of `@pytest.mark.unit` tests.
- `tests-integration.json`: combined count of `@pytest.mark.smoke`,
  `@pytest.mark.integration`, and `@pytest.mark.slow` tests.

The files are regenerated by the `Refresh test count badges` GitHub
Actions workflow on every push to `main` whose paths touch `tests/`,
`src/`, `pyproject.toml`, the workflow YAML, or
`tools/generate_test_badges.py`. Shields.io fetches them via the raw
GitHub URL and renders the badge live in the README and on the PROTEUS
framework website at
[proteus-framework.org/testing](https://proteus-framework.org/testing).

## Linting

Before committing, format and check all files:

```console
ruff check --fix src/ tests/ tools/
ruff format src/ tests/ tools/
```

The local ruff (often 0.12.x) and the CI ruff (0.15.x) sometimes disagree on
formatting drift; CI is canonical.
