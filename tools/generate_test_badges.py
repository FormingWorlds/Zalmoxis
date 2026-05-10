#!/usr/bin/env python3
"""Emit shields.io endpoint-badge JSON files for the Zalmoxis test count.

Runs ``pytest --collect-only -q`` once per marker expression to obtain the
collection count, parses the trailing summary line, and writes one
endpoint-badge JSON per non-zero count to the requested output directory.
The badges are consumed by shields.io via raw GitHub URLs of the form
``https://raw.githubusercontent.com/FormingWorlds/Zalmoxis/main/.github/badges/tests-<name>.json``
and rendered on the README and the PROTEUS framework website.

The script never executes the test bodies; it only collects, so the cost
is bounded by import time.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

# Marker -> badge filename suffix and shields.io label.
# The total badge has label "tests"; per-marker badges use the marker name.
_MARKER_QUERIES: tuple[tuple[str, str, str], ...] = (
    ('total', 'not skip', 'tests'),
    ('unit', 'unit and not skip', 'unit'),
    ('integration', 'integration and not skip', 'integration'),
    ('smoke', 'smoke and not skip', 'smoke'),
    ('slow', 'slow and not skip', 'slow'),
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
    # Exit 5 = "no tests collected". Treat as a successful zero.
    if proc.returncode == 5:
        return 0
    if proc.returncode != 0:
        raise RuntimeError(
            f'pytest collection failed for marker {marker_expr!r} '
            f'(exit code {proc.returncode}):\n{proc.stdout}\n{proc.stderr}'
        )

    match = _COLLECTED_RE.search(proc.stdout)
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


def remove_stale_badge(out_dir: Path, suffix: str) -> bool:
    """Delete a stale badge file if it exists.

    Parameters
    ----------
    out_dir : pathlib.Path
        Directory holding badge files.
    suffix : str
        Badge name suffix whose file should be removed.

    Returns
    -------
    bool
        True if a file was removed, False if no file existed.
    """
    path = out_dir / f'tests-{suffix}.json'
    if path.exists():
        path.unlink()
        return True
    return False


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

    for suffix, marker_expr, label in _MARKER_QUERIES:
        count = count_collected(marker_expr)
        print(f'{label}: {count}')
        if count > 0:
            write_badge(out_dir, suffix, label, count)
        else:
            # Drop any stale badge so the website does not show an
            # outdated number for a marker that no longer matches.
            if suffix != 'total':
                remove_stale_badge(out_dir, suffix)

    return 0


if __name__ == '__main__':
    sys.exit(main())
