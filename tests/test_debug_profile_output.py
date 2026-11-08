"""Tests of where a DEBUG solve writes its pressure and density profile dump.

Contract: at the DEBUG log level the solver writes pressure_profiles.txt and
density_profiles.txt only into the folder named by config key debug_profile_dir;
without the key, or below DEBUG, it writes no file, and never into the package
tree or the working directory. Commands: see tests/AGENTS.md.
"""

from __future__ import annotations

import logging
import os

import pytest

from tests.test_failed_structure_solve import _cfg, _run
from zalmoxis import get_zalmoxis_root

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]

DUMPS = ('pressure_profiles.txt', 'density_profiles.txt')
FAST = {'max_iterations_outer': 1, 'relative_tolerance': 1e-6, 'absolute_tolerance': 1e-7}


def _dumps_outside():
    """mtime of each dump file in the package output folder and the working directory."""
    folders = (os.path.join(get_zalmoxis_root(), 'output'), os.getcwd())
    return {
        os.path.join(d, f): os.stat(os.path.join(d, f)).st_mtime_ns
        for d in folders
        for f in DUMPS
        if os.path.exists(os.path.join(d, f))
    }


def test_debug_solve_without_the_key_writes_no_file(caplog):
    """A DEBUG solve with no debug_profile_dir logs at DEBUG and leaves the tree and cwd as they were."""
    before = _dumps_outside()
    with caplog.at_level(logging.DEBUG, logger='fwl.zalmoxis'):
        result = _run(_cfg(**FAST))
    assert result['converged'] is True
    assert any(r.levelno == logging.DEBUG for r in caplog.records)
    assert _dumps_outside() == before


def test_debug_solve_writes_the_dump_into_the_given_folder(caplog, tmp_path):
    """With debug_profile_dir the 2 dump files appear there, one profile per pressure trial in each."""
    before = _dumps_outside()
    with caplog.at_level(logging.DEBUG, logger='fwl.zalmoxis'):
        result = _run(_cfg(debug_profile_dir=str(tmp_path), **FAST))
    assert result['converged'] is True
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(DUMPS)
    pressure = (tmp_path / 'pressure_profiles.txt').read_text()
    density = (tmp_path / 'density_profiles.txt').read_text()
    assert pressure.count('# Pressure iteration ') == density.count('# Pressure iteration ') > 1
    assert pressure.count('radius pressure') == density.count('radius density')
    assert _dumps_outside() == before


def test_info_solve_with_the_key_writes_no_file(caplog, tmp_path):
    """Below DEBUG the key alone writes nothing: the folder stays empty."""
    with caplog.at_level(logging.INFO, logger='fwl.zalmoxis'):
        result = _run(_cfg(debug_profile_dir=str(tmp_path), **FAST))
    assert result['converged'] is True
    assert list(tmp_path.iterdir()) == []
