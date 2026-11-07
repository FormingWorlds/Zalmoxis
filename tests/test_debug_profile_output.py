"""Tests of where a DEBUG solve writes its pressure and density profile dump.

Contract: at the DEBUG log level the solver writes pressure_profiles.txt and
density_profiles.txt only into the folder named by config key debug_profile_dir;
without the key it writes no file, and never into the package tree. Commands: see
tests/AGENTS.md.
"""

from __future__ import annotations

import logging
import os

import pytest

from tests.test_failed_structure_solve import ROOT, _cfg, _run

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(120)]

DUMPS = ('pressure_profiles.txt', 'density_profiles.txt')


def _tree_dumps():
    """mtime of each dump file present in the package tree output folder."""
    out = os.path.join(ROOT, 'output')
    return {
        f: os.stat(os.path.join(out, f)).st_mtime_ns
        for f in DUMPS
        if os.path.exists(os.path.join(out, f))
    }


def test_debug_solve_without_the_key_writes_no_file(caplog, tmp_path):
    """A DEBUG solve with no debug_profile_dir leaves the package tree and tmp_path as they were."""
    before = _tree_dumps()
    with caplog.at_level(logging.DEBUG, logger='fwl.zalmoxis'):
        result = _run(_cfg())
    assert result['converged'] is True
    assert _tree_dumps() == before
    assert list(tmp_path.iterdir()) == []


def test_debug_solve_writes_the_dump_into_the_given_folder(caplog, tmp_path):
    """With debug_profile_dir the 2 dump files appear there, with one block per pressure solve."""
    before = _tree_dumps()
    with caplog.at_level(logging.DEBUG, logger='fwl.zalmoxis'):
        result = _run(_cfg(debug_profile_dir=str(tmp_path)))
    assert result['converged'] is True
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(DUMPS)
    assert '# Pressure iteration 0' in (tmp_path / 'pressure_profiles.txt').read_text()
    assert _tree_dumps() == before
