# How to build tests

[![codecov](https://img.shields.io/codecov/c/github/FormingWorlds/Zalmoxis?label=coverage&logo=codecov)](https://app.codecov.io/gh/FormingWorlds/Zalmoxis)
[![Unit Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/Zalmoxis/CI.yml?branch=main&label=Unit%20Tests)](https://github.com/FormingWorlds/Zalmoxis/actions/workflows/CI.yml)
[![Integration Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/Zalmoxis/nightly.yml?branch=main&label=Integration%20Tests)](https://github.com/FormingWorlds/Zalmoxis/actions/workflows/nightly.yml)
[![tests](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/FormingWorlds/Zalmoxis/main/.github/badges/tests-total.json)](https://proteus-framework.org/testing)

This page is about *writing* a new test, by hand or with an LLM. For running
the existing suite see [Testing suite](../Explanations/testing.md).

## Decision tree: which marker?

The marker is the load-bearing decision. CI runs `-m unit` on every push and
`-m "(unit or smoke or integration) and not slow"` nightly; an unmarked test
is invisible to CI. Pick the strictest marker the test fits.

| Use ... | When ... |
|---|---|
| `@pytest.mark.unit` | < 100 ms, no real solver call. EOS helpers, config validators, mixing math, melting-curve evaluation, JAX-vs-numpy parity on a single point, isolated branch coverage with mocked density. |
| `@pytest.mark.smoke` | One full-solver call at 1 $M_\oplus$ that finishes in seconds. Verifies the whole code path runs end to end at relaxed tolerance. |
| `@pytest.mark.integration` | Full-solver call against a published-reference comparison (Zeng+2019 mass-radius, Seager+2007 density profile, PALEOS rocky 1 + 5 $M_\oplus$). |
| `@pytest.mark.slow` | Composition grids, tolerance-convergence studies, anything that takes minutes per test. Manual only. |

Tests covering more than one tier (e.g. an integration test that also
exercises a unit-tier helper) carry a single marker matching the dominant
runtime.

## Choosing a file

Naming convention: `test_<module>_<aspect>.py`, lower-snake, one module per
file where possible.

| Situation | Where the test goes |
|---|---|
| New unit test for an existing module | `test_<module>.py` if it exists, else create it. |
| Branch-coverage test for a hard-to-reach path | `test_<module>_branches.py` (matches `test_eos_dispatch_branches.py`, `test_paleos_unified_branches.py`, ...). |
| Failure-mode test (raises, validation errors) | `test_<module>_failures.py` (matches `test_eos_dispatch_failures.py`). |
| Smoke / integration solver test | `test_<scenario>.py` (matches `test_MR_rocky.py`, `test_Seager_water.py`). |
| Regression for a fixed bug | Add to the closest existing file; do not create one regression file per bug. |

## Float comparisons

Always `pytest.approx`, never `==`.

```python
assert result == pytest.approx(expected, rel=1e-6)
```

Choose the tolerance to match the physics, not the implementation: a
well-converged structure ODE matches Zeng+2019 to ~3 % at 1 $M_\oplus$, not
1e-9. State the chosen tolerance with a one-line comment naming the source
or the limiting factor.

## Fixtures

| Fixture | Use when ... |
|---|---|
| `zalmoxis_root` | The test reads or writes a path under the repo root; skip-if-unset behaviour is automatic. |
| `cached_solver` | The test triggers a full rocky or water solver run; identical parameter combinations within a file share one cache hit. |

Adding a new fixture: put it in `tests/conftest.py` if it is shared across
files; keep it module-local otherwise. Session-scope only when the fixture
is genuinely expensive to build (table loaders, solver calls).

## Reference values

Every reference value asserted by a test must come from a primary source
cited inline. Examples:

```python
# Zeng+2019 mass-radius for 1 M_E rocky at CMF=0.325 (their Eq. 2 fit).
assert R_calc == pytest.approx(R_zeng, rel=3e-2)

# Stacey 2008 Earth CMB temperature: 4500-6000 K at 136 GPa.
assert 4500.0 <= T_cmb <= 6000.0
```

Do not "guess" a tolerance from a previous run. If you do not have a
reference, the test belongs in the unit tier with a synthetic input whose
expected output you can derive analytically.

## Mocking EOS

Mocking is appropriate when:

- The test isolates a non-EOS component (the ODE integrator, the mass-radius
  solver, a config validator) and EOS evaluation is a confounder.
- The point of the test is the analytic limit (constant density, polytropic).

Mocking is not appropriate when the test is meant to verify
EOS-table-dependent physics (mushy-zone width, phase routing, binodal
suppression). For those, exercise the real PALEOS or Seager paths through
the `cached_solver` or a direct call.

The `_paleos_mock.py` helper provides a lightweight stand-in for the
upstream PALEOS Python package on CI runners that do not vendor it; use it
only when the upstream package is genuinely unavailable, not as a default.

## Comment hygiene

Inline comments and docstrings should explain *why* the test exists, never
*when it was added* or *what it used to do*. Acceptable:

```python
# Stixrude+14 cryoscopic depression sets mushy zone factor to 0.79.
# Below 0.7 the solidus crosses the core-mantle boundary at 5 M_E,
# which is unphysical. Validate the rejection.
```

Not acceptable:

```python
# Added in T2.3 to cover the basin attractor we found on 2026-04-27.
# Previously this assertion was rel=1e-3; loosened to 3e-2 in commit abc1234
# after the BLAS-noise rerun.
```

History belongs in the commit message and the PR description, not in the
test source.

## A worked example

Adding a unit test for a hypothetical helper `clamp_to_grid(P, T, table)`:

```python
# tests/test_eos_clamp_branches.py
from __future__ import annotations

import numpy as np
import pytest

from zalmoxis.eos.interpolation import clamp_to_grid


@pytest.mark.unit
def test_clamp_below_min_p_pins_to_grid_edge():
    """When P is below the table minimum, clamp_to_grid returns P_min."""
    table_P = np.array([1e5, 1e6, 1e7])
    table_T = np.array([300.0, 1000.0, 3000.0])
    P_clamped, T_clamped = clamp_to_grid(0.5e4, 1500.0, table_P, table_T)
    assert P_clamped == pytest.approx(1e5)
    assert T_clamped == pytest.approx(1500.0)


@pytest.mark.unit
def test_clamp_within_grid_is_identity():
    """Inside-grid inputs must pass through unchanged (no rounding)."""
    table_P = np.array([1e5, 1e6, 1e7])
    table_T = np.array([300.0, 1000.0, 3000.0])
    P_clamped, T_clamped = clamp_to_grid(5e6, 1500.0, table_P, table_T)
    assert P_clamped == pytest.approx(5e6)
    assert T_clamped == pytest.approx(1500.0)
```

Two unit tests, one branch each, both fast, both with a one-line docstring
stating the rationale.

## Coverage gate and pragma usage

Zalmoxis enforces a 90% coverage gate in the nightly CI (see
[Testing suite > Coverage gate](../Explanations/testing.md#coverage-gate)). Most defensive
code is already excluded automatically through the `exclude_lines` list in
`[tool.coverage.report]` (`def __repr__`, `if TYPE_CHECKING:`,
`@abstractmethod`, `raise NotImplementedError`, etc.) — you do not need
inline pragmas for those.

For inline `# pragma: no cover`, use it only on **genuinely defensive**
branches that no realistic test can reach without contrived inputs:
`LinAlgError` from `np.linalg.lstsq`, `RuntimeError` recovery on a
non-finite mass evaluation, dev-gated `if _PROFILE:` instrumentation,
heavy-data-dependent fallback paths. Always pair the pragma with a
one-line reason:

```python
except (ValueError, RuntimeError) as exc:  # pragma: no cover - stale-cache sign mismatch; defensive
    ...
```

Do not pragma normal-execution paths. The reviewer will push back; "I
couldn't be bothered to write a test" is not a defensive branch.

## Anti-patterns

- **Forgetting the marker.** Tests without `@pytest.mark.unit` (or another
  marker) are invisible to CI. Run `pytest --collect-only -m unit | tail`
  after adding a test to confirm pickup. With `--strict-markers`, a typo'd
  marker now fails the run.
- **Hardcoding paths.** Use `zalmoxis_root` or `pathlib.Path(__file__).parent`,
  never absolute paths.
- **Test ordering dependence.** Each test must pass in isolation. xdist
  reorders aggressively; relying on side-effects from a previous test is a
  bug, not a shortcut.
- **Asserting on log output.** Logs change for cosmetic reasons; assert on
  return values or state. Use `caplog` only when the log line is itself the
  contract.
- **Sleeping or polling.** If a test needs a wait, the code under test has a
  race condition; fix that first.

## Suggested LLM prompt

When asking an LLM (Claude, Cursor, Copilot) to add or modify tests, paste
the prompt below at the start of the request along with the relevant source
file. The prompt encodes the PROTEUS and Zalmoxis testing principles so the
generated test passes review without an iteration round.

````text
You are writing a pytest test for the Zalmoxis interior-structure solver
(part of the PROTEUS ecosystem). Follow these rules strictly.

MARKERS (mandatory; CI-invisible without one):
- @pytest.mark.unit: < 100 ms, no real solver call. Use for EOS helpers,
  config validators, mixing math, JAX-vs-numpy parity, branch coverage.
- @pytest.mark.smoke: one full-solver call at 1 M_E that finishes in
  seconds. Use for end-to-end smoke checks.
- @pytest.mark.integration: full-solver call against a published reference
  (Zeng+2019, Seager+2007, PALEOS rocky 1 + 5 M_E).
- @pytest.mark.slow: composition grids or tolerance studies that take
  minutes per test. Manual only.
Pick the strictest marker the test fits. Do not double-mark.

FLOAT COMPARISONS:
- Always pytest.approx, never ==.
- Choose the tolerance to match the physics, not the implementation.
- State the source or limiting factor in a one-line comment.

REFERENCE VALUES:
- Cite the primary source inline. If no source is available, the test must
  use an analytically derivable synthetic input.
- Do not infer tolerances from a previous run.

FIXTURES:
- Use `zalmoxis_root` for paths.
- Use `cached_solver` for full rocky / water solver runs.
- Add new fixtures to tests/conftest.py only if shared across files.

MOCKING:
- Mock EOS only when the test isolates a non-EOS component or exercises an
  analytic limit. For EOS-table-dependent physics, use the real PALEOS or
  Seager path via `cached_solver` or a direct call.

COVERAGE EXCLUSIONS:
- The 90% gate runs in nightly CI. Do NOT add `# pragma: no cover` just to
  pass the gate. The exclude_lines list in pyproject.toml already covers
  def __repr__, raise NotImplementedError, if TYPE_CHECKING:, @abstractmethod.
- Inline pragma is acceptable ONLY for: numerical pathology recovery
  (LinAlgError, RuntimeError on non-finite values, brentq same-sign
  bracket), dev-gated diagnostics behind env vars, heavy-data-dependent
  fallback paths.
- Every inline pragma must carry a one-line "why this branch is
  unreachable in unit tests" justification.

NAMING:
- Files: test_<module>.py, test_<module>_branches.py for hard-to-reach
  paths, test_<module>_failures.py for failure-mode tests.
- Function names: snake_case, descriptive, no test_1 / test_2.

STYLE:
- Single quotes (ruff).
- `from __future__ import annotations` at the top of every file.
- One-line docstring stating the rationale for the test.
- Comments explain WHY, never WHEN added or what the code used to do.
- No project-tracking labels (T1.x, Stage X, ISO dates, commit SHAs).

ANTI-PATTERNS:
- No bare assertions on float equality.
- No hardcoded absolute paths.
- No reliance on test execution order.
- No assertions on log content unless the log line is itself the contract.
- No sleep / poll. If you need to wait, the code under test has a race.

OUTPUT:
- Produce only the test source. Do not modify the module under test.
- Place the new test in the appropriate existing file, or name a new file
  per the naming convention.
- After the test, list (a) which marker you chose and why, (b) the
  reference source for any literature value, (c) the tolerance and its
  justification.
````
