#!/usr/bin/env python3
"""Emit shields.io endpoint-badge JSON files for the Zalmoxis test count.

Implements the public 2-category test scheme used across the PROTEUS
ecosystem: every test counts as either "unit" (mocked, fast) or
"integration" (everything else, including the internal ``smoke`` and
``slow`` markers). The internal 4-marker scheme (unit / smoke /
integration / slow) is kept inside CI for operational granularity but
is hidden from the public-facing badges to keep the surface readable to
non-developers.

Three files are always written: ``tests-total.json``, ``tests-unit.json``,
and ``tests-integration.json``. The integration count is the union of
``smoke``, ``integration``, and ``slow`` markers; pytest's ``or`` marker
expression handles the union deduplication for free.

Badges are consumed by shields.io via raw GitHub URLs of the form
``https://raw.githubusercontent.com/FormingWorlds/Zalmoxis/main/.github/badges/tests-<name>.json``
and rendered in the README and on the PROTEUS framework website.

The script never executes test bodies; it only collects, so the cost is
bounded by import time.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# (filename suffix, pytest marker expression, shields.io label).
# "integration" here is the public union of smoke + integration + slow.
_BADGE_QUERIES: tuple[tuple[str, str, str], ...] = (
    ('total', 'not skip', 'tests'),
    ('unit', 'unit and not skip', 'unit tests'),
    ('integration', '(smoke or integration or slow) and not skip', 'integration tests'),
)

# Pytest prints either ``<N> tests collected`` (all match) or
# ``<selected>/<total> tests collected (<deselected> deselected)`` (some
# filtered out). The first capture group always holds the selected count.
_COLLECTED_RE = re.compile(r'(\d+)(?:/\d+)?\s+tests?\s+collected', re.IGNORECASE)


def count_collected(marker_expr: str) -> int:
    """Return the number of tests pytest would collect for a marker expression.

    Parameters
    ----------
    marker_expr : str
        Pytest marker expression passed via ``-m``.

    Returns
    -------
    int
        Number of tests reported by ``pytest --collect-only``. Pytest exit
        code 5 ("no tests collected") is treated as a successful zero;
        any other non-zero exit raises.

    Raises
    ------
    RuntimeError
        If pytest exits non-zero with a code other than 5, or if the
        trailing summary line cannot be parsed from stdout.
    """
    cmd = [
        sys.executable,
        '-m',
        'pytest',
        '-o',
        'addopts=',
        '--collect-only',
        '-q',
        '-m',
        marker_expr,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 5:
        return 0
    if proc.returncode != 0:
        raise RuntimeError(
            f'pytest collection failed for marker {marker_expr!r} '
            f'(exit code {proc.returncode}):\n{proc.stdout}\n{proc.stderr}'
        )

    # Pytest emits the "<N> tests collected" summary as the LAST
    # non-empty line of its collection-only output. Searching only the
    # last line guards against a stray earlier match (e.g. a plugin's
    # progress line, a docstring printed during collection, or an
    # incidental "<digit>+ tests collected" string in a warning).
    last_line = next(
        (line for line in reversed(proc.stdout.splitlines()) if line.strip()),
        '',
    )
    match = _COLLECTED_RE.search(last_line)
    if match is None:
        raise RuntimeError(
            f'Could not parse "<N> tests collected" from pytest output '
            f'for marker {marker_expr!r}:\n{proc.stdout}'
        )
    return int(match.group(1))


def write_badge(out_dir: Path, suffix: str, label: str, count: int) -> Path:
    """Write a single shields.io endpoint-badge JSON file.

    Parameters
    ----------
    out_dir : pathlib.Path
        Directory where the badge file is written; created if missing.
    suffix : str
        Badge name suffix (e.g. ``"total"``, ``"unit"``). The filename
        becomes ``tests-<suffix>.json``.
    label : str
        Left-side label rendered by shields.io.
    count : int
        Right-side message rendered by shields.io.

    Returns
    -------
    pathlib.Path
        Absolute path of the written JSON file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'tests-{suffix}.json'
    payload = {
        'schemaVersion': 1,
        'label': label,
        'message': str(count),
        'color': 'blue',
    }
    path.write_text(json.dumps(payload, indent=2) + '\n')
    return path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with attribute ``out`` (pathlib.Path).
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--out',
        required=True,
        type=Path,
        help='Directory where tests-<name>.json badge files are written.',
    )
    return parser.parse_args()


def main() -> int:
    """Entry point.

    Returns
    -------
    int
        Process exit code: 0 on success, non-zero on failure.
    """
    args = parse_args()
    out_dir: Path = args.out

    for suffix, marker_expr, label in _BADGE_QUERIES:
        count = count_collected(marker_expr)
        print(f'{label}: {count}')
        write_badge(out_dir, suffix, label, count)

    return 0


if __name__ == '__main__':
    sys.exit(main())
