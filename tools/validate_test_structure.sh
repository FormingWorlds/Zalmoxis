#!/usr/bin/env bash
# Marker-validation gate for the Zalmoxis test suite.
#
# Walks every test_*.py file under tests/, finds every ``def test_*``
# definition, and verifies that the function (or its enclosing class,
# or the module-level pytestmark) carries EXACTLY ONE of the four
# canonical pytest tier markers:
#
#     @pytest.mark.unit
#     @pytest.mark.smoke
#     @pytest.mark.integration
#     @pytest.mark.slow
#
# A test that carries ``@pytest.mark.skip`` (and no tier marker) also
# passes, since a skipped test never runs and does not need a tier.
#
# Exit codes
# ----------
# 0  every test under tests/ carries exactly one tier marker (or skip)
# 1  one or more tests violate the marker rule (printed as file:line)
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
"""Validate that every test under tests/ carries exactly one tier marker.

A test passes the validator iff it carries exactly one of
``unit / smoke / integration / slow``, OR it carries ``skip`` (and no
tier marker, since a skipped test never runs).

Markers are inherited along: module-level ``pytestmark``, enclosing
class decorator, function decorator. The merged set across these
three sources must contain exactly one tier marker, or be exactly
``{skip}``.

Reports two kinds of failure with file:line diagnostics:

- ``has multiple tier markers ['integration', 'unit']``: a test that
  inherits or declares more than one tier marker. Such a test is
  selected by both the PR-tier and the nightly-tier selectors and
  gets double-counted in the per-tier badges.
- ``carries no marker``: a test with no tier marker and no skip.
  Invisible to CI under ``-m unit`` / ``-m "(unit or smoke or ...)``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

TIER = {'unit', 'smoke', 'integration', 'slow'}
SKIP = {'skip'}
ALLOWED = TIER | SKIP


def mark_names_from_decorators(decorators) -> list[str]:
    """Return tier / skip marker names found on a list of decorator nodes."""
    out: list[str] = []
    for d in decorators:
        # @pytest.mark.NAME
        if isinstance(d, ast.Attribute) and isinstance(d.value, ast.Attribute):
            if (
                isinstance(d.value.value, ast.Name)
                and d.value.value.id == 'pytest'
                and d.value.attr == 'mark'
            ):
                out.append(d.attr)
        # @pytest.mark.NAME(...)
        elif isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute):
            f = d.func
            if (
                isinstance(f.value, ast.Attribute)
                and isinstance(f.value.value, ast.Name)
                and f.value.value.id == 'pytest'
                and f.value.attr == 'mark'
            ):
                out.append(f.attr)
    return out


def module_marks(tree: ast.Module) -> list[str]:
    """Return the marker names assigned to module-level ``pytestmark``.

    Accepts ``pytestmark = pytest.mark.<name>``, ``pytestmark = [...]``,
    and ``pytestmark = (...)`` wrappers. An assignment whose RHS is
    not a recognised ``pytest.mark.<name>`` expression returns an
    empty list, which the caller treats as "no module-level marker".
    """
    marks: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == 'pytestmark':
                    val = node.value
                    items = val.elts if isinstance(val, (ast.List, ast.Tuple)) else [val]
                    for item in items:
                        marks.extend(mark_names_from_decorators([item]))
        elif isinstance(node, ast.AnnAssign):
            tgt = node.target
            if (
                isinstance(tgt, ast.Name)
                and tgt.id == 'pytestmark'
                and node.value is not None
            ):
                val = node.value
                items = val.elts if isinstance(val, (ast.List, ast.Tuple)) else [val]
                for item in items:
                    marks.extend(mark_names_from_decorators([item]))
    return marks


def walk(tree, parent_marks: list[str]):
    """Yield ``(test_def, marks)`` for every test_* function in the tree.

    ``marks`` is the merged list of marker names along the chain
    module-level pytestmark plus enclosing class decorators plus
    function decorators.
    """
    out: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, list[str]]] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            cls_marks = parent_marks + mark_names_from_decorators(node.decorator_list)
            out.extend(walk(node, cls_marks))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith('test_'):
                fn_marks = mark_names_from_decorators(node.decorator_list)
                out.append((node, parent_marks + fn_marks))
    return out


def main() -> int:
    tests_dir = Path(sys.argv[1])
    failures: list[str] = []
    total = 0
    n_files = 0

    for path in sorted(tests_dir.rglob('test_*.py')):
        n_files += 1
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        except SyntaxError as exc:
            failures.append(f'{path}:{exc.lineno}: SyntaxError {exc.msg}')
            continue

        mod_marks = module_marks(tree)
        for fn, marks in walk(tree, mod_marks):
            total += 1
            tier_marks = [m for m in marks if m in TIER]
            has_skip = any(m in SKIP for m in marks)
            rel = path.relative_to(tests_dir.parent)
            if len(tier_marks) > 1:
                failures.append(
                    f'{rel}:{fn.lineno}: {fn.name} has multiple tier markers '
                    f'{sorted(set(tier_marks))}; exactly one of '
                    'unit / smoke / integration / slow is required.'
                )
            elif len(tier_marks) == 0 and not has_skip:
                failures.append(
                    f'{rel}:{fn.lineno}: {fn.name} carries no marker '
                    '(need one of unit / smoke / integration / slow / skip)'
                )

    if failures:
        print(
            f'Marker validation FAILED on {len(failures)} of {total} tests '
            f'in {n_files} files:',
            file=sys.stderr,
        )
        for f in failures:
            print(f'  {f}', file=sys.stderr)
        return 1

    print(f'Marker validation OK: {total} tests in {n_files} files, all carry a marker.')
    return 0


sys.exit(main())
PY
