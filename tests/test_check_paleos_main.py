"""Tests for tools/check_paleos_main.py, the scheduled check of PALEOS upstream main."""

from __future__ import annotations

import io
import urllib.error

import pytest

from tools import check_paleos_main as check
from zalmoxis.eos.paleos_api import PALEOS_TESTED_SHA

pytestmark = pytest.mark.unit


def _reply():
    return io.BytesIO(b'{"sha": "' + b'c' * 40 + b'"}')


def test_the_pin_is_read_from_the_source_without_importing_it():
    """The script reads PALEOS_TESTED_SHA from paleos_api.py as the module defines it."""
    assert check.tested_sha() == PALEOS_TESTED_SHA


def test_main_at_the_pin_exits_zero(capsys):
    """Upstream main at the tested commit: in order, exit 0."""
    assert check.main(fetch=lambda: PALEOS_TESTED_SHA) == 0
    assert 'PALEOS main is at the tested commit' in capsys.readouterr().out


def test_main_ahead_of_the_pin_exits_one(capsys):
    """Upstream main at another commit: the job fails and names both commits."""
    assert check.main(fetch=lambda: 'b' * 40) == 1
    out = capsys.readouterr().out
    assert 'b' * 40 in out and PALEOS_TESTED_SHA in out
    assert 'PALEOS_TESTED_SHA' in out


def test_an_unread_main_exits_three(capsys):
    """GitHub not answering is no evidence either way: exit 3, no retry."""
    calls = []

    def fetch():
        calls.append(1)
        raise OSError('no answer')

    assert check.main(fetch=fetch) == 3
    assert calls == [1]
    assert 'no answer' in capsys.readouterr().out


def test_an_answer_that_is_no_sha_exits_three(capsys):
    """A reply without a 40-character commit SHA is treated as unread."""
    assert check.main(fetch=lambda: 'rate limited') == 3
    assert 'rate limited' in capsys.readouterr().out


def test_the_request_carries_the_token_when_one_is_set(monkeypatch):
    """With GITHUB_TOKEN set the request is authenticated, which lifts the rate limit."""
    seen = {}

    def urlopen(request, timeout):
        seen['auth'] = request.get_header('Authorization')
        return _reply()

    monkeypatch.setattr(check.urllib.request, 'urlopen', urlopen)
    monkeypatch.setenv('GITHUB_TOKEN', 'abc')
    assert check.main_sha() == 'c' * 40
    assert seen['auth'] == 'Bearer abc'


def test_an_annotated_pin_is_read(tmp_path, monkeypatch):
    """PALEOS_TESTED_SHA: str = '...' is read like a plain assignment."""
    src = tmp_path / 'paleos_api.py'
    src.write_text("PALEOS_TESTED_SHA: str = '" + 'd' * 40 + "'\n")
    monkeypatch.setattr(check, '_SOURCE', src)
    assert check.tested_sha() == 'd' * 40


@pytest.mark.parametrize(
    'code, exit_code, text',
    [
        (404, 4, 'PALEOS main not found'),
        (422, 4, 'PALEOS main not found'),
        (403, 3, 'PALEOS main could not be read'),
        (500, 3, 'PALEOS main could not be read'),
    ],
    ids=['404', '422', '403', '500'],
)
def test_github_codes_map_to_their_exit(capsys, code, exit_code, text):
    """404 and 422 (no repository or branch) fail with exit 4; a 403 rate limit or a 5xx
    only warns with exit 3."""

    def fetch():
        raise urllib.error.HTTPError(check._MAIN, code, 'x', {}, None)

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
    check.main_sha()
    monkeypatch.setenv('GITHUB_TOKEN', '')
    check.main_sha()
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
        code = check.main(fetch=lambda: 'c' * 40)
    except Exception as exc:  # noqa: BLE001 -- an escaped error is the failure under test
        pytest.fail(f'main raised {exc!r} instead of exiting 2')
    assert code == 2
    assert capsys.readouterr().out.startswith('PALEOS_TESTED_SHA could not be read')
