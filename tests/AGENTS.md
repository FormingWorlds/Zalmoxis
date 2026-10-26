# Zalmoxis test instructions

<!-- fwl-tests-core:begin sha256=0d210e346c572a96 -->
## Test rules shared by the PROTEUS ecosystem

Each test file starts with a module-level tier marker and a timeout. CI selects tests by marker, so a file without one runs in no CI job; the timeout stops a hang, it is not a target.

```python
pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]
```

| tier | what it tests | timeout |
|---|---|---|
| `unit` | Python logic with the heavy physics mocked; aim for < 100 ms | 30 s |
| `smoke` | the real binaries or solvers, one step, low resolution | 60 s |
| `integration` | several modules coupled | 300 s |
| `slow` | full physics validation | 3600 s |

Each test carries exactly one tier marker, so that the tier filters select it in the one CI job meant for it; a function marker for a second tier breaks that. Split a file whose tests need different tiers. `skip` excludes a test from every CI job.

Every new test covers an edge case (a boundary value, an empty input, an extreme physical parameter), exercises the error contract (a documented exception with the check that no side effect ran, a guard, a clamp, or the limit input of the formula), and asserts values that do not follow trivially from the implementation. A pinned value of 1 that every exponent reproduces checks nothing.

`python tools/check_test_quality.py --check` fails when the count of any rule rises above `tools/test_quality_baseline.json`; the offenders it prints are the first few of that rule in the tree, not necessarily yours. The rules: a file without a tier marker, a test without a docstring, a test with one assertion or none, a weak assertion as the only one (`is None`, `is not None`, `> 0`, `len(...) > 0`, `isinstance`), `==` next to a non-zero float literal, and an optional dependency imported without `pytest.importorskip`. `bash tools/validate_test_structure.sh` runs the repository's own structure check; the repository part of this file says what it checks.

Physics tests carry markers so their coverage is tracked apart from line coverage:
- `@pytest.mark.physics_invariant` on each test function that asserts a conservation law, a bound (T > 0, fractions in [0, 1]), a monotonicity or symmetry, or a pinned value with a discrimination guard. The marker goes on the function, not the module: structural tests in the same file do not carry it.
- `@pytest.mark.reference_pinned` (together with `physics_invariant`) on a test that pins a published benchmark, an analytical limit or a cross-implementation result; cite the paper, table or figure in the docstring.
- A discrimination guard asserts that the most plausible wrong formula (a missing factor, a swapped exponent, the wrong unit) gives a result outside the tolerance of the pinned value.

Floats: compare with `pytest.approx` or `np.testing.assert_allclose` and a tolerance you can justify from the method, not the one that makes the test pass.

Mocks: mock at the narrowest scope (the one external call), and return physically plausible values, so the code under test runs its real branches. Set random seeds and write files only under `tmp_path`.

A module-level constant read from an environment variable at import time does not change with `monkeypatch.setenv`; patch the constant with `monkeypatch.setattr`.
<!-- fwl-tests-core:end -->

## Zalmoxis specifics

### Testing

This repository has no `tools/check_test_quality.py`; the marker check is `bash tools/validate_test_structure.sh`.

- 2547 tests collected with the EOS data linked: 2356 unit, 126 smoke, 19 integration, 46 slow.
- Unit tier: `pytest -o "addopts=" -m unit` (~1.5 min on the dev machine,
  ~1m54s ubuntu / ~2m29s macOS in CI).
- Nightly tier: `pytest -o "addopts=" -m "(unit or smoke or integration) and not slow" -n auto`
  with coverage; on ubuntu CI it runs as eight jobs, the `wolfbower` solves, the
  `paleos_adiabat` solves (5 M_earth with no reachable wall clock), the `accept_solves`
  (real solves of the structure invariants, the accept labels and the pressure root, one
  job per test file, `accept_solves_long` for the longest of them and
  `accept_solves_newton_setups` for the sweep over every Newton setup alone) and the rest
  (90 min ceiling each, 120 min for `accept_solves_long`, 170 min for
  `accept_solves_newton_setups`).
- The `-o "addopts="` override is needed because `pyproject.toml` defaults to
  `-n auto --dist loadfile` for parallel execution; a single-marker filter run
  is faster without the parallel split.
- Use `@pytest.mark.unit` on every new unit test; CI runs with `-m unit` /
  `-m smoke`. Tests without a marker are invisible to CI.
- Use `pytest.approx` for float comparisons, never `==`.
- First-principles verification suite: `tests/test_first_principles.py`
  (25 tests: uniform and two-layer spheres, Gauss's law, hydrostatic balance,
  binding energy, Earth benchmark, mass-radius scaling, CMF trend, pure iron,
  conservation, grid and tolerance convergence).

### Coverage gate

- Hard gate at `[tool.coverage.report] fail_under = 90.0` in
  `pyproject.toml`. Per ecosystem policy this is the maximum ratchet
  threshold for any module: it does not increase above 90% even if real
  coverage exceeds it. Real coverage targets ~92% to give 2 percentage
  points of headroom for small future additions before the gate trips.
- `fail_under` in `[tool.coverage.report]` sets the 90% floor; in
  `.github/workflows/nightly.yml` the main job repeats it with
  `--cov-fail-under=90` and the `wolfbower`, `paleos_adiabat` and the 5 `accept_solves` jobs
  turn it off with `0`.
- A pre-flight check in `.github/workflows/CI.yml` rejects PRs that lower
  `fail_under` below the value on `origin/main`, mirroring PROTEUS's
  `.github/workflows/ci-pr-checks.yml` idiom.
- Use `# pragma: no cover` only for genuinely defensive recovery paths
  (NaN fallbacks, LinAlgError handlers, dev-gated diagnostics). Never on
  feature code. Every inline pragma carries a one-line "why this branch is
  unreachable in unit tests" justification.
