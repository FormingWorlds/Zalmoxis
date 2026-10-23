"""Compare PALEOS upstream main with the commit the PALEOS-API path is tested with.

Exit 0 when main is at ``PALEOS_TESTED_SHA``; 6 when main is ahead of it (the tested commit
is in main's history; the workflow reports a warning and the new commits); 1 when main does
not contain it (diverged or behind, as after a rewrite of main); 2 when the source cannot
be read or holds no 40-character lowercase commit SHA as the pin; 4 when GitHub does not
find the repository, main or the tested commit (404, 422); 5 when GitHub refuses the request
(401, or 403 with requests left and no Retry-After: a token or permission fault); 3 when
GitHub could not be read for another reason (a used-up or secondary rate limit, a 5xx, the
network, a reply without a comparison). One request, no retry; with GITHUB_TOKEN set the
request is authenticated.

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
_COMPARE = 'https://api.github.com/repos/maraattia/PALEOS/compare/{}...main'
_STATUSES = ('identical', 'ahead', 'behind', 'diverged')


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


def compare(tested: str) -> dict:
    """Return GitHub's comparison of ``tested`` with PALEOS upstream main (one request)."""
    headers = {'Accept': 'application/vnd.github+json'}
    if os.environ.get('GITHUB_TOKEN'):
        headers['Authorization'] = f'Bearer {os.environ["GITHUB_TOKEN"]}'
    request = urllib.request.Request(_COMPARE.format(tested), headers=headers)
    with urllib.request.urlopen(request, timeout=30) as reply:
        return json.load(reply)


def main(fetch=compare) -> int:
    """Print the verdict and return the exit status."""
    try:
        tested = tested_sha()
    except (LookupError, ValueError, OSError, SyntaxError) as exc:
        print(f'PALEOS_TESTED_SHA could not be read: {exc}')
        return 2
    url = _COMPARE.format(tested)
    try:
        reply = fetch(tested)
    except urllib.error.HTTPError as exc:
        if exc.code in (404, 422):
            print(f'PALEOS main or the tested commit {tested} not found at {url}: {exc}')
            return 4
        headers = exc.headers or {}
        limited = headers.get('X-RateLimit-Remaining') == '0' or 'Retry-After' in headers
        if exc.code == 401 or (exc.code == 403 and not limited):
            print(f'GitHub refused the request for {url}: {exc}')
            return 5
        print(f'PALEOS main could not be read: {exc}')
        return 3
    except Exception as exc:  # noqa: BLE001 -- one attempt, reported, never retried
        print(f'PALEOS main could not be read: {exc}')
        return 3
    try:
        status = reply['status']
        ahead, behind = int(reply['ahead_by']), int(reply['behind_by'])
        if status not in _STATUSES:
            raise ValueError(status)
    except Exception as exc:  # noqa: BLE001 -- any malformed reply is an unread one
        print(f'PALEOS main could not be read: no comparison in the reply ({exc!r})')
        return 3
    if status == 'identical':
        print(f'PALEOS main is at the tested commit {tested}')
        return 0
    if status == 'ahead':
        try:
            new = [
                f'  {c["sha"][:9]} {c["commit"]["committer"]["date"][:10]} '
                + c['commit']['message'].partition('\n')[0]
                for c in reply['commits']
            ]
        except Exception as exc:  # noqa: BLE001
            print(f'PALEOS main could not be read: no commit list in the reply ({exc!r})')
            return 3
        listed = f' ({len(new)} of {ahead} listed)' if len(new) < ahead else ''
        print(
            f'PALEOS main is {ahead} commit{"s" * (ahead != 1)} ahead of the tested commit '
            f'{tested}; the pin moves after the PALEOS-API tests pass on the new commit'
            f'{listed}:\n\n```'
        )
        print(*new, '```', sep='\n')
        return 6
    base = reply.get('merge_base_commit')
    base = base.get('sha', 'unknown') if isinstance(base, dict) else 'unknown'
    cause = (
        'as after a rewrite of main: run the PALEOS-API tests on main and move '
        if status == 'diverged'
        else 'main is older than the tested commit (main moved back, or the pin is not on '
        'main yet): keep or move '
    )
    print(
        f'PALEOS main does not contain the tested commit {tested} ({status}: {ahead} '
        f'commits ahead, {behind} behind, merge base {base}), {cause}PALEOS_TESTED_SHA in '
        'src/zalmoxis/eos/paleos_api.py to a commit on main'
    )
    return 1


if __name__ == '__main__':
    sys.exit(main())
