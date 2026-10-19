"""Tests for the PALEOS commit check of the PALEOS-API path and the dirty-tree cache key."""

from __future__ import annotations

import subprocess

import pytest

from zalmoxis.eos import paleos_api, paleos_api_cache
from zalmoxis.eos.paleos_api import PaleosVersionError, check_paleos_commit

pytestmark = pytest.mark.unit


def _git(repo, *args):
    out = subprocess.run(
        ['git', '-C', str(repo), *args], capture_output=True, text=True, check=True
    )
    return out.stdout.strip()


def _commit(repo, text):
    (repo / 'eos.py').write_text(text)
    _git(repo, 'add', 'eos.py')
    _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-qm', text)
    return _git(repo, 'rev-parse', 'HEAD')


@pytest.fixture
def history(tmp_path):
    """A PALEOS-like repository with commits old -> tested -> new, and an unrelated one."""
    repo = tmp_path / 'PALEOS'
    repo.mkdir()
    _git(repo, 'init', '-q', '-b', 'main')
    old = _commit(repo, 'old')
    tested = _commit(repo, 'tested')
    new = _commit(repo, 'new')
    _git(repo, 'checkout', '-q', '--orphan', 'other')
    unrelated = _commit(repo, 'unrelated')
    _git(repo, 'checkout', '-q', 'main')
    return repo, {'old': old, 'tested': tested, 'new': new, 'unrelated': unrelated}


def test_the_tested_commit_runs(history):
    repo, c = history
    _git(repo, 'checkout', '-q', c['tested'])
    assert check_paleos_commit(repo, c['tested']) == 'tested'


def test_a_newer_commit_runs_with_a_warning(history, caplog):
    repo, c = history
    _git(repo, 'checkout', '-q', c['new'])
    assert check_paleos_commit(repo, c['tested']) == 'newer'
    assert c['tested'] in caplog.text and c['new'] in caplog.text


@pytest.mark.parametrize('which', ['old', 'unrelated'])
def test_an_older_or_unrelated_commit_stops(history, which):
    repo, c = history
    _git(repo, 'checkout', '-q', c[which])
    with pytest.raises(PaleosVersionError) as err:
        check_paleos_commit(repo, c['tested'])
    assert c[which] in str(err.value) and c['tested'] in str(err.value)
    assert 'git -C' in str(err.value)


def test_a_tested_commit_missing_from_the_clone_stops(history):
    repo, c = history
    with pytest.raises(PaleosVersionError, match='does not contain'):
        check_paleos_commit(repo, 'f' * 40)


def test_a_checkout_without_git_stops(tmp_path):
    with pytest.raises(PaleosVersionError, match='cannot tell'):
        check_paleos_commit(tmp_path, 'a' * 40)


def test_the_version_error_is_not_caught_as_a_solver_failure():
    assert not issubclass(PaleosVersionError, (RuntimeError, ValueError, OSError))


def test_a_dirty_tree_gets_its_own_sha_and_cache_name(history, monkeypatch):
    repo, c = history
    monkeypatch.setattr(paleos_api, '_paleos_repo_dir', lambda: repo)
    clean = paleos_api.paleos_installed_sha()
    assert clean == c['new']
    (repo / 'eos.py').write_text('edited')
    dirty = paleos_api.paleos_installed_sha()
    assert dirty == c['new'] + '-dirty'
    assert paleos_api_cache._sha_short(dirty) != paleos_api_cache._sha_short(clean)
    assert paleos_api_cache._sha_short(dirty) == c['new'][:10] + '-dirty'


def test_an_untracked_file_does_not_make_the_tree_dirty(history, monkeypatch):
    repo, c = history
    monkeypatch.setattr(paleos_api, '_paleos_repo_dir', lambda: repo)
    (repo / 'notes.txt').write_text('scratch')
    assert paleos_api.paleos_installed_sha() == c['new']


def test_the_pin_is_a_full_commit_sha():
    assert len(paleos_api.PALEOS_TESTED_SHA) == 40
    int(paleos_api.PALEOS_TESTED_SHA, 16)


def test_the_cache_checks_the_commit_before_it_resolves(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(paleos_api_cache, '_check_paleos_once', lambda: calls.append(1))
    monkeypatch.setattr(paleos_api_cache, 'paleos_installed_sha', lambda: 'a' * 40)
    monkeypatch.setattr(paleos_api_cache, '_cache_root', lambda: tmp_path)
    monkeypatch.setattr(paleos_api_cache, 'generate_paleos_api_unified_table', lambda **k: None)
    paleos_api_cache.resolve_paleos_api_unified('iron', paleos_api.make_default_grid_iron())
    assert calls == [1]
