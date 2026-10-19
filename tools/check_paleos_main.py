"""Compare PALEOS upstream main with the commit the PALEOS-API path is tested with.

Exit 0 when main is at ``PALEOS_TESTED_SHA``, 1 when it is at another commit (test the
PALEOS-API path on it and move the pin), 2 when the source cannot be read or holds no
40-character lowercase commit SHA as the pin,
4 when GitHub does not find the repository or its main (404, 422), 3 when GitHub could
not be read for another reason. One request, no retry; with GITHUB_TOKEN set the request
is authenticated.

Usage: python tools/check_paleos_main.py
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1] / 'src' / 'zalmoxis' / 'eos' / 'paleos_api.py'
_MAIN = 'https://api.github.com/repos/maraattia/PALEOS/commits/main'


def tested_sha() -> str:
    """Return ``PALEOS_TESTED_SHA`` as ``paleos_api.py`` assigns it, without importing it.

    Raises
    ------
    LookupError, ValueError
        If the pin is not assigned, or is no 40-character lowercase commit SHA.
    OSError, SyntaxError
        If the source cannot be read or parsed.
    """
    for node in ast.parse(_SOURCE.read_text()).body:
        targets = (
            node.targets if isinstance(node, ast.Assign) else [getattr(node, 'target', None)]
        )
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and node.value is not None
            and any(getattr(t, 'id', None) == 'PALEOS_TESTED_SHA' for t in targets)
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{40}', value):
                raise ValueError(f'{value!r} is no 40-character lowercase commit SHA')
            return value
    raise LookupError(f'PALEOS_TESTED_SHA is not assigned in {_SOURCE}')


def main_sha() -> str:
    """Return the commit SHA of PALEOS upstream main (one GitHub request)."""
    headers = {'Accept': 'application/vnd.github+json'}
    if os.environ.get('GITHUB_TOKEN'):
        headers['Authorization'] = f'Bearer {os.environ["GITHUB_TOKEN"]}'
    request = urllib.request.Request(_MAIN, headers=headers)
    with urllib.request.urlopen(request, timeout=30) as reply:
        return json.load(reply)['sha']


def main(fetch=main_sha) -> int:
    """Print the verdict and return the exit status."""
    try:
        tested = tested_sha()
    except (LookupError, ValueError, OSError, SyntaxError) as exc:
        print(f'PALEOS_TESTED_SHA could not be read: {exc}')
        return 2
    try:
        upstream = str(fetch())
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 422):
            print(f'PALEOS main not found at {_MAIN}: {exc}')
            return 4
        print(f'PALEOS main could not be read: {exc}')
        return 3
    except Exception as exc:  # noqa: BLE001 -- one attempt, reported, never retried
        print(f'PALEOS main could not be read: {exc}')
        return 3
    if not re.fullmatch(r'[0-9a-f]{40}', upstream):
        print(f'PALEOS main could not be read: no commit SHA in the reply ({upstream!r})')
        return 3
    if upstream == tested:
        print(f'PALEOS main is at the tested commit {tested}')
        return 0
    print(
        f'PALEOS main is at {upstream}, not at the tested commit {tested}: run the '
        'PALEOS-API tests on it and move PALEOS_TESTED_SHA in src/zalmoxis/eos/paleos_api.py'
    )
    return 1


if __name__ == '__main__':
    sys.exit(main())
