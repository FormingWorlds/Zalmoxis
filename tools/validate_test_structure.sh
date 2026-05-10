#!/usr/bin/env bash
# Marker-validation gate for the Zalmoxis test suite.
#
# Walks every test_*.py file under tests/, finds every top-level
# ``def test_*`` and ``class Test*`` definition, and verifies the
# function (or its enclosing class) carries exactly one of the four
# canonical pytest markers (unit, smoke, integration, slow) or the
# ``skip`` exclusion. Files that declare a module-level
# ``pytestmark = pytest.mark.<marker>`` are accepted as marker-bearing
# for every test inside.
#
# Exit codes
# ----------
# 0  every test under tests/ carries exactly one marker
# 1  one or more tests are unmarked (printed as file:line)
#
# Run
# ---
# bash tools/validate_test_structure.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
TESTS_DIR="${REPO_ROOT}/tests"

if [ ! -d "${TESTS_DIR}" ]; then
    echo "tests/ not found at ${TESTS_DIR}" >&2
    exit 1
fi

python3 - "${TESTS_DIR}" <<'PY'
"""Validate that every test under tests/ carries a pytest marker.

A test is considered marked when:
- its module declares ``pytestmark = pytest.mark.<marker>``, OR
- the def has an immediately-preceding ``@pytest.mark.<marker>`` decorator,
  OR
- the enclosing class has an ``@pytest.mark.<marker>`` decorator.

Allowed markers: unit, smoke, integration, slow, skip.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

ALLOWED = {'unit', 'smoke', 'integration', 'slow', 'skip'}


def marker_from_decorator(dec: ast.expr) -> str | None:
    """Return the marker name from ``@pytest.mark.<x>`` or ``@pytest.mark.<x>(...)``."""
    if isinstance(dec, ast.Call):
        dec = dec.func
    if isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Attribute):
        if (
            isinstance(dec.value.value, ast.Name)
            and dec.value.value.id == 'pytest'
            and dec.value.attr == 'mark'
        ):
            return dec.attr
    return None


_REAL_TIER_MARKERS = ALLOWED - {'skip'}


def _marker_names_in_value(value: ast.expr) -> set[str]:
    """Recursively extract marker names from a ``pytestmark`` RHS.

    Accepts ``pytest.mark.<name>``, ``pytest.mark.<name>(...)``, and
    list/tuple wrappers thereof. Returns the set of marker names found.
    """
    found: set[str] = set()
    if isinstance(value, (ast.List, ast.Tuple)):
        for elt in value.elts:
            found |= _marker_names_in_value(elt)
        return found
    name = marker_from_decorator(value)
    if name:
        found.add(name)
    return found


def module_pytestmark_markers(tree: ast.Module) -> set[str]:
    """Return the set of marker names assigned to module-level ``pytestmark``.

    Empty set when ``pytestmark`` is absent or its RHS is not a
    recognisable ``pytest.mark.<x>`` expression. A module that assigns
    ``pytestmark = []``, ``pytestmark = "unit"``, or
    ``pytestmark = pytest.mark.skip`` returns a set that does not
    include any of unit/smoke/integration/slow, which the caller then
    treats as "not marker-bearing".
    """
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == 'pytestmark':
                    return _marker_names_in_value(node.value)
        if isinstance(node, ast.AnnAssign):
            tgt = node.target
            if (
                isinstance(tgt, ast.Name)
                and tgt.id == 'pytestmark'
                and node.value is not None
            ):
                return _marker_names_in_value(node.value)
    return set()


def find_unmarked(path: Path) -> list[tuple[int, str]]:
    """Return [(lineno, name), ...] of unmarked test_* defs in path.

    Returns ``[(0, '<unparseable>')]`` if the file fails to parse, so
    the caller can surface the syntax error as a marker-validation
    failure rather than crashing the script.
    """
    try:
        tree = ast.parse(path.read_text(), filename=str(path))
    except SyntaxError:
        return [(0, '<unparseable>')]

    module_markers = module_pytestmark_markers(tree)
    if module_markers & _REAL_TIER_MARKERS:
        return []

    unmarked: list[tuple[int, str]] = []

    class_markers: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
            class_markers[node.name] = {
                m for m in (marker_from_decorator(d) for d in node.decorator_list) if m
            }

    def collect(parent: ast.AST, in_class: str | None = None):
        children = getattr(parent, 'body', []) or []
        for node in children:
            if isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                collect(node, in_class=node.name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith('test_'):
                    continue
                own = {m for m in (marker_from_decorator(d) for d in node.decorator_list) if m}
                cls = class_markers.get(in_class, set()) if in_class else set()
                marks = (own | cls | module_markers) & ALLOWED
                if not marks:
                    unmarked.append((node.lineno, node.name))

    collect(tree)
    return unmarked


def main() -> int:
    tests_dir = Path(sys.argv[1])
    failures: list[str] = []
    n_files = 0
    n_tests = 0
    for path in sorted(tests_dir.rglob('test_*.py')):
        n_files += 1
        unmarked = find_unmarked(path)
        try:
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef)
                ) and node.name.startswith('test_'):
                    n_tests += 1
        except SyntaxError:
            # find_unmarked already reported this path with name '<unparseable>'.
            pass
        for lineno, name in unmarked:
            rel = path.relative_to(tests_dir.parent)
            failures.append(f'{rel}:{lineno}: {name}')

    if failures:
        print(
            f'{len(failures)} unmarked tests in {n_files} files '
            f'(of {n_tests} tests total):',
            file=sys.stderr,
        )
        for line in failures:
            print(f'  {line}', file=sys.stderr)
        print(
            "\nEvery test must carry exactly one of: @pytest.mark.unit, "
            '@pytest.mark.smoke, @pytest.mark.integration, @pytest.mark.slow.',
            file=sys.stderr,
        )
        print(
            'Or declare ``pytestmark = pytest.mark.<marker>`` at module level.',
            file=sys.stderr,
        )
        return 1

    print(f'All {n_tests} tests in {n_files} files carry a marker.')
    return 0


sys.exit(main())
PY
