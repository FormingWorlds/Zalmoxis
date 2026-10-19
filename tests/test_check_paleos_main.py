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
