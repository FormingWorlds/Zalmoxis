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
    (repo / 'paleos' / 'eos.py').write_text(text)
    _git(repo, 'add', 'paleos/eos.py')
    _git(
        repo,
        *('-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false'),
        *('-c', 'core.hooksPath=/dev/null', 'commit', '-qm', text),
    )
    return _git(repo, 'rev-parse', 'HEAD')


@pytest.fixture
def history(tmp_path):
    """A PALEOS-like repository with commits old -> tested -> new, and an unrelated one."""
    repo = tmp_path / 'PALEOS'
    (repo / 'paleos').mkdir(parents=True)
    _git(repo, 'init', '-q')
    _git(repo, 'checkout', '-q', '-b', 'main')
    (repo / 'paleos' / '__init__.py').write_text('')
    (repo / '.gitignore').write_text('.venv/\n')
    _git(repo, 'add', 'paleos/__init__.py', '.gitignore')
    old = _commit(repo, 'old')
    tested = _commit(repo, 'tested')
    new = _commit(repo, 'new')
    _git(repo, 'checkout', '-q', '--orphan', 'other')
    unrelated = _commit(repo, 'unrelated')
    _git(repo, 'checkout', '-q', 'main')
    return repo, {'old': old, 'tested': tested, 'new': new, 'unrelated': unrelated}


def test_the_tested_commit_runs(history):
    """A checkout at the tested commit runs."""
    repo, c = history
    _git(repo, 'checkout', '-q', c['tested'])
    assert check_paleos_commit(repo, c['tested']) == 'tested'


def test_a_newer_commit_runs_with_a_warning(history, caplog):
    """A descendant of the tested commit runs and the log names both commits."""
    repo, c = history
    _git(repo, 'checkout', '-q', c['new'])
    assert check_paleos_commit(repo, c['tested']) == 'newer'
    assert c['tested'] in caplog.text and c['new'] in caplog.text


@pytest.mark.parametrize('which', ['old', 'unrelated'])
def test_an_older_or_unrelated_commit_stops(history, which):
    """An ancestor or an unrelated commit stops with both SHAs and the update command."""
    repo, c = history
    _git(repo, 'checkout', '-q', c[which])
    with pytest.raises(PaleosVersionError) as err:
        check_paleos_commit(repo, c['tested'])
    assert c[which] in str(err.value) and c['tested'] in str(err.value)
    assert 'git -C' in str(err.value)


def test_a_tested_commit_missing_from_the_clone_stops(history):
    """A clone without the tested commit cannot contain it, so the check stops."""
    repo, c = history
    with pytest.raises(PaleosVersionError, match='does not contain'):
        check_paleos_commit(repo, 'f' * 40)


def test_a_checkout_without_git_stops(tmp_path, monkeypatch):
    """PALEOS outside a git checkout has no commit to check, so the check stops."""
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path.parent))
    with pytest.raises(PaleosVersionError, match='cannot tell') as err:
        check_paleos_commit(tmp_path, 'a' * 40)
    assert 'not a git repository' in str(err.value)


def test_the_version_error_is_not_caught_as_a_solver_failure():
    """PaleosVersionError passes the handlers that turn solver errors into a failed solve."""
    assert not issubclass(PaleosVersionError, (RuntimeError, ValueError, OSError))


def test_a_dirty_tree_gets_its_own_sha_and_cache_name(history, monkeypatch):
    """A change to a tracked file adds -dirty to the SHA and to the cache file name."""
    repo, c = history
    monkeypatch.setattr(paleos_api, '_paleos_repo_dir', lambda: repo)
    clean = paleos_api.paleos_installed_sha()
    assert clean == c['new']
    (repo / 'paleos' / 'eos.py').write_text('edited')
    dirty = paleos_api.paleos_installed_sha()
    assert dirty.startswith(c['new'] + '-dirty-') and len(dirty) == 40 + 7 + 8
    assert paleos_api_cache._sha_short(dirty) != paleos_api_cache._sha_short(clean)
    assert paleos_api_cache._sha_short(dirty) == c['new'][:10] + dirty[40:]


def test_a_file_outside_the_package_does_not_make_the_tree_dirty(history, monkeypatch):
    """A file outside the paleos package is no change to the PALEOS code."""
    repo, c = history
    monkeypatch.setattr(paleos_api, '_paleos_repo_dir', lambda: repo)
    (repo / 'notes.txt').write_text('scratch')
    assert paleos_api.paleos_installed_sha() == c['new']


def test_the_pin_is_a_full_commit_sha():
    """The pin is a full 40-character hex commit SHA."""
    assert len(paleos_api.PALEOS_TESTED_SHA) == 40
    assert int(paleos_api.PALEOS_TESTED_SHA, 16) > 0


def test_the_cache_checks_the_commit_before_it_resolves(monkeypatch, tmp_path):
    """Resolving a PALEOS-API table runs the commit check first."""
    calls = []
    monkeypatch.setattr(paleos_api_cache, '_check_paleos_once', lambda sha: calls.append(sha))
    monkeypatch.setattr(paleos_api_cache, 'paleos_installed_sha', lambda: 'a' * 40)
    monkeypatch.setattr(paleos_api_cache, '_cache_root', lambda: tmp_path)
    monkeypatch.setattr(
        paleos_api_cache, 'generate_paleos_api_unified_table', lambda **k: calls.append('gen')
    )
    paleos_api_cache.resolve_paleos_api_unified('iron', paleos_api.make_default_grid_iron())
    assert calls == ['a' * 40, 'gen']


def test_two_different_edits_get_two_cache_names(history, monkeypatch):
    """Each set of uncommitted changes has its own key, so one edit never reads another's tables."""
    repo, c = history
    monkeypatch.setattr(paleos_api, '_paleos_repo_dir', lambda: repo)
    (repo / 'paleos' / 'eos.py').write_text('edit A')
    a = paleos_api.paleos_installed_sha()
    (repo / 'paleos' / 'eos.py').write_text('edit B')
    b = paleos_api.paleos_installed_sha()
    assert a != b and a[:40] == b[:40] == c['new']
    assert paleos_api_cache._sha_short(a) != paleos_api_cache._sha_short(b)


def test_a_dirty_tree_runs_with_a_warning(history, caplog):
    """Changes to tracked files at or after the tested commit run, with a warning that says so."""
    repo, c = history
    _git(repo, 'checkout', '-q', c['tested'])
    (repo / 'paleos' / 'eos.py').write_text('local change')
    assert check_paleos_commit(repo, c['tested']) == 'dirty'
    assert 'uncommitted changes' in caplog.text


def test_without_git_the_check_stops(history, monkeypatch):
    """No git on PATH: the commit cannot be read, which is a PaleosVersionError."""
    repo, c = history
    monkeypatch.setenv('PATH', '/nonexistent')
    with pytest.raises(PaleosVersionError, match='cannot tell'):
        check_paleos_commit(repo, c['tested'])


def test_a_copy_outside_the_tracked_tree_stops(history):
    """A paleos copied into an ignored folder of a checkout is not that checkout's code."""
    repo, c = history
    site = repo / '.venv' / 'site-packages'
    (site / 'paleos').mkdir(parents=True)
    (site / 'paleos' / '__init__.py').write_text('')
    with pytest.raises(PaleosVersionError, match='not tracked'):
        check_paleos_commit(site, c['tested'])


def test_a_shallow_clone_without_the_tested_commit_names_the_fetch(history, tmp_path):
    """A shallow clone lacks the tested commit; the message names git fetch --unshallow."""
    repo, c = history
    shallow = tmp_path / 'shallow'
    _git(tmp_path, 'clone', '-q', '--depth', '1', f'file://{repo}', str(shallow))
    with pytest.raises(PaleosVersionError, match='fetch --unshallow'):
        check_paleos_commit(shallow, c['tested'])


def test_without_paleos_the_check_names_it(monkeypatch):
    """No paleos package: the message says PALEOS is not installed."""
    monkeypatch.setattr(paleos_api, '_paleos_repo_dir', lambda: None)
    with pytest.raises(PaleosVersionError, match='PALEOS is not installed'):
        check_paleos_commit()


def test_the_pin_is_read_when_the_check_runs(history, monkeypatch):
    """The default pin is the module constant at call time."""
    repo, c = history
    _git(repo, 'checkout', '-q', c['tested'])
    monkeypatch.setattr(paleos_api, 'PALEOS_TESTED_SHA', c['tested'])
    assert check_paleos_commit(repo) == 'tested'
    monkeypatch.setattr(paleos_api, 'PALEOS_TESTED_SHA', c['new'])
    with pytest.raises(PaleosVersionError):
        check_paleos_commit(repo)


def test_the_check_runs_again_for_another_commit(monkeypatch):
    """The once-per-process memo is per PALEOS SHA: a checkout during a run is checked again."""
    calls = []
    monkeypatch.setattr(
        paleos_api_cache, 'check_paleos_commit', lambda: calls.append(1) or 'tested'
    )
    paleos_api_cache._check_paleos_once.cache_clear()
    try:
        for sha in ('a' * 40, 'a' * 40, 'b' * 40):
            paleos_api_cache._check_paleos_once(sha)
    finally:
        paleos_api_cache._check_paleos_once.cache_clear()
    assert calls == [1, 1]


def test_the_two_phase_cache_checks_the_commit_too(monkeypatch, tmp_path):
    """Resolving the 2-phase PALEOS-API tables runs the commit check first."""
    calls = []
    monkeypatch.setattr(paleos_api_cache, '_check_paleos_once', lambda sha: calls.append(sha))
    monkeypatch.setattr(paleos_api_cache, 'paleos_installed_sha', lambda: 'a' * 40)
    monkeypatch.setattr(paleos_api_cache, '_cache_root', lambda: tmp_path)
    monkeypatch.setattr(
        paleos_api_cache, 'generate_paleos_api_2phase_mgsio3_tables', lambda **k: None
    )
    paleos_api_cache.resolve_paleos_api_2phase_mgsio3(paleos_api.make_default_grid_mgsio3())
    assert calls == ['a' * 40]


def _dirty_key(repo, monkeypatch):
    monkeypatch.setattr(paleos_api, '_paleos_repo_dir', lambda: repo)
    return paleos_api.paleos_installed_sha()


def test_an_untracked_module_in_the_package_changes_the_key(history, monkeypatch):
    """A new module not yet added to git is code: two contents of it give two keys."""
    repo, c = history
    (repo / 'paleos' / 'helper.py').write_text('K = 1\n')
    a = _dirty_key(repo, monkeypatch)
    (repo / 'paleos' / 'helper.py').write_text('K = 2\n')
    b = _dirty_key(repo, monkeypatch)
    assert a.startswith(c['new'] + '-dirty-') and b.startswith(c['new'] + '-dirty-')
    assert a != b


def test_an_edit_git_is_told_to_hide_still_changes_the_key(history, monkeypatch):
    """--assume-unchanged hides an edit from git status, not from the files on disk."""
    repo, c = history
    _git(repo, 'update-index', '--assume-unchanged', 'paleos/eos.py')
    (repo / 'paleos' / 'eos.py').write_text('hidden edit')
    assert _dirty_key(repo, monkeypatch).startswith(c['new'] + '-dirty-')
    assert check_paleos_commit(repo, c['new']) == 'dirty'


def test_a_non_utf8_edit_and_an_external_diff_tool_still_give_a_key(history, monkeypatch):
    """Bytes that are not UTF-8 and a configured external diff tool do not break or hide the key."""
    repo, c = history
    _git(repo, 'config', 'diff.external', 'true')
    (repo / 'paleos' / 'eos.py').write_bytes(b'caf\xe9\n')
    key = _dirty_key(repo, monkeypatch)
    assert key.startswith(c['new'] + '-dirty-') and len(key) == 40 + 7 + 8


def test_compiled_files_do_not_make_the_tree_dirty(history, monkeypatch):
    """__pycache__ and .pyc files are written by Python, not code changes."""
    repo, c = history
    (repo / 'paleos' / '__pycache__').mkdir()
    (repo / 'paleos' / '__pycache__' / 'eos.cpython-312.pyc').write_bytes(b'\x00\x01')
    assert _dirty_key(repo, monkeypatch) == c['new']
    assert check_paleos_commit(repo, c['new']) == 'tested'


def test_a_dirty_newer_tree_runs_as_dirty_with_two_warnings(history, caplog):
    """A descendant with an edit warns that it is newer and that it has changes."""
    repo, c = history
    (repo / 'paleos' / 'eos.py').write_text('edit')
    assert check_paleos_commit(repo, c['tested']) == 'dirty'
    assert 'newer than the commit' in caplog.text and 'uncommitted changes' in caplog.text


def test_a_key_the_check_did_not_make_is_refused(monkeypatch):
    """A fallback key such as version-<x> names no commit, so no cache is read or built for it."""
    monkeypatch.setattr(paleos_api_cache, 'check_paleos_commit', lambda: 'tested')
    paleos_api_cache._check_paleos_once.cache_clear()
    try:
        with pytest.raises(PaleosVersionError, match='version-0.1'):
            paleos_api_cache._check_paleos_once('version-0.1.0')
    finally:
        paleos_api_cache._check_paleos_once.cache_clear()


def test_a_commit_git_cannot_list_falls_back_to_the_version_and_the_check_stops(
    history, monkeypatch
):
    """If the package of the commit cannot be listed, the key is the version fallback (which
    no cache accepts) and the check stops with PaleosVersionError."""
    import sys
    import types

    repo, c = history
    fake = types.ModuleType('paleos')
    fake.__file__, fake.__version__ = str(repo / 'paleos' / '__init__.py'), '1.0'
    monkeypatch.setitem(sys.modules, 'paleos', fake)
    real = paleos_api._git

    def git(repo_dir, *args):
        if args[:1] == ('ls-tree',):
            return paleos_api._subprocess.CompletedProcess(args, 128, '', 'fatal: bad tree')
        return real(repo_dir, *args)

    monkeypatch.setattr(paleos_api, '_git', git)
    assert paleos_api.paleos_installed_sha() == 'version-1.0'
    with pytest.raises(PaleosVersionError, match='cannot list the paleos package'):
        check_paleos_commit(repo, c['new'])
