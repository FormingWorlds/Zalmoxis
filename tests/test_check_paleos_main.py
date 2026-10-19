"""Tests for tools/check_paleos_main.py, the scheduled check of PALEOS upstream main."""

from __future__ import annotations

import pytest

from tools import check_paleos_main as check
from zalmoxis.eos.paleos_api import PALEOS_TESTED_SHA

pytestmark = pytest.mark.unit


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


def test_a_main_that_is_not_found_fails_with_exit_4(capsys):
    """A 404 (repository renamed or private) is no transient fault: exit 4 fails the job."""
    import urllib.error

    def fetch():
        raise urllib.error.HTTPError(check._MAIN, 404, 'Not Found', {}, None)

    assert check.main(fetch=fetch) == 4
    assert 'not found' in capsys.readouterr().out


def test_the_request_carries_the_token_when_one_is_set(monkeypatch):
    """With GITHUB_TOKEN set the request is authenticated, which lifts the rate limit."""
    seen = {}

    class _Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"sha": "' + b'c' * 40 + b'"}'

    def urlopen(request, timeout):
        seen['auth'] = request.get_header('Authorization')
        return _Reply()

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
    'code, exit_code', [(422, 4), (403, 3), (500, 3)], ids=['422', '403', '500']
)
def test_github_codes_map_to_their_exit(capsys, code, exit_code):
    """422 (no such branch) fails like 404; a 403 rate limit or a 5xx only warns."""
    import urllib.error

    def fetch():
        raise urllib.error.HTTPError(check._MAIN, code, 'x', {}, None)

    assert check.main(fetch=fetch) == exit_code
    assert capsys.readouterr().out.startswith('PALEOS main')


def test_without_a_token_no_authorization_header_is_sent(monkeypatch):
    """An unset or empty GITHUB_TOKEN sends no Authorization header."""
    seen = []

    class _Reply:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"sha": "' + b'c' * 40 + b'"}'

    monkeypatch.setattr(
        check.urllib.request,
        'urlopen',
        lambda request, timeout: seen.append(request.get_header('Authorization')) or _Reply(),
    )
    for value in (None, ''):
        if value is None:
            monkeypatch.delenv('GITHUB_TOKEN', raising=False)
        else:
            monkeypatch.setenv('GITHUB_TOKEN', value)
        check.main_sha()
    assert seen == [None, None]


def test_a_pin_that_cannot_be_read_exits_2(tmp_path, monkeypatch, capsys):
    """A bare annotation without a value is no pin: exit 2, not the exit 1 of a moved main."""
    src = tmp_path / 'paleos_api.py'
    src.write_text('PALEOS_TESTED_SHA: str\n')
    monkeypatch.setattr(check, '_SOURCE', src)
    assert check.main(fetch=lambda: 'c' * 40) == 2
    assert 'PALEOS_TESTED_SHA' in capsys.readouterr().out
