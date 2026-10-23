"""Tests for tools/check_paleos_main.py, the scheduled check of PALEOS upstream main."""

from __future__ import annotations

import io
import json
import re
import urllib.error
from pathlib import Path

import pytest

from tools import check_paleos_main as check
from zalmoxis.eos.paleos_api import PALEOS_TESTED_SHA

pytestmark = pytest.mark.unit


def _reply():
    return io.BytesIO(json.dumps(_compare('identical')).encode())


def _compare(status, ahead=0, behind=0, commits=()):
    """A GitHub compare reply of the tested commit against main."""
    return {
        'status': status,
        'ahead_by': ahead,
        'behind_by': behind,
        'merge_base_commit': {'sha': 'e' * 40},
        'commits': [
            {
                'sha': sha,
                'commit': {
                    'committer': {'date': '2026-10-22T15:20:45Z'},
                    'message': f'Subject {sha[:4]}\n\nbody',
                },
            }
            for sha in commits
        ],
    }


def test_main_ahead_of_the_pin_warns_and_exits_six(capsys):
    """Upstream main ahead of the tested commit (the pin is in its history): exit 6, which
    the workflow turns into a warning, and the summary names every new commit."""
    new = ['a' * 40, 'b' * 40, 'f' * 40]
    assert check.main(fetch=lambda *a: _compare('ahead', 3, 0, new)) == 6
    out = capsys.readouterr().out
    assert 'PALEOS main is 3 commits ahead of the tested commit' in out
    assert all(sha[:9] in out for sha in new) and 'Subject aaaa' in out


def test_the_pin_is_read_from_the_source_without_importing_it():
    """The script reads PALEOS_TESTED_SHA from paleos_api.py as the module defines it."""
    assert check.tested_sha() == PALEOS_TESTED_SHA


def test_main_at_the_pin_exits_zero(capsys):
    """Upstream main at the tested commit: in order, exit 0."""
    assert check.main(fetch=lambda tested: _compare('identical')) == 0
    assert 'PALEOS main is at the tested commit' in capsys.readouterr().out


@pytest.mark.parametrize('status, ahead, behind', [('diverged', 8, 5), ('behind', 0, 2)])
def test_main_without_the_pin_in_its_history_fails(capsys, status, ahead, behind):
    """Upstream main that does not contain the tested commit (diverged as after the rewrite
    of main, or behind it) fails with exit 1 and names the counts and the merge base."""
    reply = _compare(status, ahead, behind)
    assert check.main(fetch=lambda tested: reply) == 1
    out = capsys.readouterr().out
    assert f'PALEOS main does not contain the tested commit {PALEOS_TESTED_SHA}' in out
    assert f'{status}: {ahead} commits ahead, {behind} behind, merge base {"e" * 40}' in out
    assert 'PALEOS_TESTED_SHA in src/zalmoxis/eos/paleos_api.py' in out


def test_an_unread_main_exits_three(capsys):
    """GitHub not answering is no evidence either way: exit 3, no retry."""
    calls = []

    def fetch(tested):
        calls.append(tested)
        raise OSError('no answer')

    assert check.main(fetch=fetch) == 3
    assert calls == [PALEOS_TESTED_SHA]
    assert 'no answer' in capsys.readouterr().out


@pytest.mark.parametrize(
    'reply',
    [{'message': 'rate limited'}, _compare('unknown'), 'not a mapping'],
    ids=['no-status', 'unknown', 'text'],
)
def test_a_reply_without_a_comparison_exits_three(capsys, reply):
    """A reply without a known comparison status is treated as unread."""
    assert check.main(fetch=lambda tested: reply) == 3
    assert capsys.readouterr().out.startswith('PALEOS main could not be read: no comparison')


def test_the_request_carries_the_token_when_one_is_set(monkeypatch):
    """With GITHUB_TOKEN set the request is authenticated, which lifts the rate limit."""
    seen = {}

    def urlopen(request, timeout):
        seen['auth'] = request.get_header('Authorization')
        seen['url'] = request.full_url
        return _reply()

    monkeypatch.setattr(check.urllib.request, 'urlopen', urlopen)
    monkeypatch.setenv('GITHUB_TOKEN', 'abc')
    assert check.compare(PALEOS_TESTED_SHA)['status'] == 'identical'
    assert seen['auth'] == 'Bearer abc'
    assert seen['url'].endswith(f'/repos/maraattia/PALEOS/compare/{PALEOS_TESTED_SHA}...main')


def test_an_annotated_pin_is_read(tmp_path, monkeypatch):
    """PALEOS_TESTED_SHA: str = '...' is read like a plain assignment."""
    src = tmp_path / 'paleos_api.py'
    src.write_text("PALEOS_TESTED_SHA: str = '" + 'd' * 40 + "'\n")
    monkeypatch.setattr(check, '_SOURCE', src)
    assert check.tested_sha() == 'd' * 40


@pytest.mark.parametrize(
    'code, exit_code, text',
    [
        (404, 4, 'PALEOS main or the tested commit'),
        (422, 4, 'PALEOS main or the tested commit'),
        (500, 3, 'PALEOS main could not be read'),
    ],
    ids=['404', '422', '500'],
)
def test_github_codes_map_to_their_exit(capsys, code, exit_code, text):
    """404 and 422 (no repository, branch or tested commit) fail with exit 4; a 5xx only
    warns with exit 3."""

    def fetch(tested):
        raise urllib.error.HTTPError(check._COMPARE, code, 'x', {}, None)

    assert check.main(fetch=fetch) == exit_code
    assert capsys.readouterr().out.startswith(text)


def test_without_a_token_no_authorization_header_is_sent(monkeypatch):
    """An unset or empty GITHUB_TOKEN sends no Authorization header."""
    seen = []
    monkeypatch.setattr(
        check.urllib.request,
        'urlopen',
        lambda request, timeout: seen.append(request.get_header('Authorization')) or _reply(),
    )
    monkeypatch.delenv('GITHUB_TOKEN', raising=False)
    check.compare(PALEOS_TESTED_SHA)
    monkeypatch.setenv('GITHUB_TOKEN', '')
    check.compare(PALEOS_TESTED_SHA)
    assert seen == [None, None]


@pytest.mark.parametrize(
    'source',
    [
        'PALEOS_TESTED_SHA: str\n',
        'OTHER = 1\n',
        "PALEOS_TESTED_SHA = os.environ['X']\n",
        "PALEOS_TESTED_SHA = '654aca9'\n",
        "PALEOS_TESTED_SHA = ''\n",
        'PALEOS_TESTED_SHA = 5\n',
        'PALEOS_TESTED_SHA = None\n',
        "PALEOS_TESTED_SHA = '" + 'A' * 40 + "'\n",
        "PALEOS_TESTED_SHA = '" + 'a' * 41 + "'\n",
        'PALEOS_TESTED_SHA = (\n',
        None,
    ],
    ids=[
        'annotation',
        'absent',
        'not-literal',
        'short',
        'empty',
        'int',
        'none',
        'upper',
        'long',
        'syntax',
        'no-file',
    ],
)
def test_a_pin_that_cannot_be_read_exits_2(tmp_path, monkeypatch, capsys, source):
    """A pin that is no 40-character lowercase SHA, or a source that cannot be parsed or
    read, is exit 2, not the exit 1 of a moved main."""
    src = tmp_path / 'paleos_api.py'
    if source is not None:
        src.write_text(source)
    monkeypatch.setattr(check, '_SOURCE', src)
    try:
        code = check.main(fetch=lambda tested: _compare('identical'))
    except Exception as exc:  # noqa: BLE001 -- an escaped error is the failure under test
        pytest.fail(f'main raised {exc!r} instead of exiting 2')
    assert code == 2
    assert capsys.readouterr().out.startswith('PALEOS_TESTED_SHA could not be read')


@pytest.mark.parametrize(
    'code, headers, exit_code, text',
    [
        (403, {'X-RateLimit-Remaining': '0'}, 3, 'PALEOS main could not be read'),
        (403, {'Retry-After': '60'}, 3, 'PALEOS main could not be read'),
        (403, {'X-RateLimit-Remaining': '42'}, 5, 'GitHub refused the request'),
        (403, {}, 5, 'GitHub refused the request'),
        (401, {}, 5, 'GitHub refused the request'),
    ],
    ids=[
        '403 rate limit',
        '403 secondary limit',
        '403 with requests left',
        '403 without the header',
        '401',
    ],
)
def test_a_refused_request_fails_unless_it_is_the_rate_limit(
    capsys, code, headers, exit_code, text
):
    """A 401, or a 403 that is not a used-up rate limit, is a token or permission fault that
    stays: exit 5 fails the workflow; a 403 with no requests left only warns with exit 3."""

    def fetch(tested):
        raise urllib.error.HTTPError(check._COMPARE, code, 'x', headers, None)

    assert check.main(fetch=fetch) == exit_code
    assert capsys.readouterr().out.startswith(text)


def test_the_workflow_warns_on_exit_3_and_6_and_fails_on_every_other_code():
    """The scheduled job turns exit 3 (GitHub not read) and exit 6 (main ahead of the tested
    commit) into a warning and a passed run, and fails with every other exit status."""
    workflow = (
        Path(check.__file__).resolve().parents[1] / '.github/workflows/paleos-main.yml'
    ).read_text()
    tests = re.findall(r'-eq (\d+) \]', workflow)
    warned = re.findall(
        r'-eq (\d+) \]; then\n\s+echo "::warning [^\n]+\n\s+exit 0\n\s+fi\n', workflow
    )
    assert tests == warned == ['3', '6'], (tests, warned)
    assert workflow.rstrip().endswith('exit "${code}"')
