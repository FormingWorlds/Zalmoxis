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


@pytest.fixture
def allow_dirty(monkeypatch):
    """Opt in to running PALEOS package files that differ from their commit."""
    monkeypatch.setenv('ZALMOXIS_PALEOS_ALLOW_DIRTY', '1')


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


def test_an_older_checkout_names_the_pull(history):
    """A checkout behind the tested commit gets the pull, and no rewrite is named."""
    repo, c = history
    _git(repo, 'checkout', '-q', c['old'])
    with pytest.raises(PaleosVersionError) as err:
        check_paleos_commit(repo, c['tested'])
    assert f'git -C {repo} pull' in str(err.value)
    assert 'main was rewritten' not in str(err.value)


def test_a_rewritten_upstream_names_the_missing_history(history):
    """A main rewritten past the tested commit keeps it out of the history of the
    checkout, so the message says that and names a checkout instead of a pull."""
    repo, c = history
    _git(repo, 'checkout', '-q', c['unrelated'])
    with pytest.raises(PaleosVersionError) as err:
        check_paleos_commit(repo, c['tested'])
    assert 'not in the history of this checkout' in str(err.value)
    assert f'git -C {repo} checkout {c["tested"]}' in str(err.value)
    assert f'git -C {repo} pull' not in str(err.value)


def test_a_tested_commit_missing_from_a_full_clone_names_both_causes(history):
    """A clone that lacks the tested commit gets a fetch and a checkout of it, and the
    pin move for a main that no longer holds it."""
    repo, _ = history
    with pytest.raises(PaleosVersionError) as err:
        check_paleos_commit(repo, 'f' * 40)
    assert f'git -C {repo} fetch && git -C {repo} checkout {"f" * 40}' in str(err.value)
    assert f'git -C {repo} pull' not in str(err.value)
    assert 'main was rewritten' in str(err.value) and 'PALEOS_TESTED_SHA' in str(err.value)


def test_a_clone_from_before_a_rewrite_runs_after_the_named_commands(
    history, tmp_path, monkeypatch
):
    """A clone made before main was rewritten (the tested commit amended to a new SHA
    with the same tree) cannot pull the new history; the commands the message names
    fetch the rewritten commit and give a checkout the check accepts."""
    monkeypatch.setenv('GIT_CONFIG_GLOBAL', '/dev/null')
    monkeypatch.setenv('GIT_CONFIG_NOSYSTEM', '1')
    repo, c = history
    clone = tmp_path / 'clone'
    _git(tmp_path, 'clone', '-q', f'file://{repo}', str(clone))
    _git(repo, 'reset', '-q', '--hard', c['tested'])
    _git(
        repo,
        *('-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false'),
        *('-c', 'core.hooksPath=/dev/null', 'commit', '-q', '--amend', '-m', 'rewritten'),
    )
    rewritten = _git(repo, 'rev-parse', 'HEAD')
    assert _git(repo, 'rev-parse', f'{rewritten}^{{tree}}') == _git(
        repo, 'rev-parse', f'{c["tested"]}^{{tree}}'
    )
    with pytest.raises(PaleosVersionError) as err:
        check_paleos_commit(clone, rewritten)
    message = str(err.value)
    assert f'git -C {clone} pull' not in message
    fix = f'git -C {clone} fetch && git -C {clone} checkout {rewritten}'
    assert fix in message
    for command in fix.split(' && '):
        subprocess.run(command.split(), check=True, capture_output=True)
    assert check_paleos_commit(clone, rewritten) == 'tested'


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


@pytest.mark.usefixtures('allow_dirty')
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
    monkeypatch.setattr(paleos_api, 'check_paleos_commit', lambda: calls.append(1) or 'tested')
    paleos_api._check_paleos_once.cache_clear()
    try:
        for sha in ('a' * 40, 'a' * 40, 'b' * 40):
            paleos_api._check_paleos_once(sha)
    finally:
        paleos_api._check_paleos_once.cache_clear()
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


@pytest.mark.usefixtures('allow_dirty')
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


@pytest.mark.usefixtures('allow_dirty')
def test_a_dirty_newer_tree_runs_as_dirty_with_two_warnings(history, caplog):
    """A descendant with an edit warns that it is newer and that it has changes."""
    repo, c = history
    (repo / 'paleos' / 'eos.py').write_text('edit')
    assert check_paleos_commit(repo, c['tested']) == 'dirty'
    assert 'newer than the commit' in caplog.text and 'uncommitted changes' in caplog.text


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


@pytest.mark.parametrize('value', [None, '', '0', 'true', 'yes', ' 1'])
def test_a_dirty_tree_stops_unless_it_is_allowed(history, monkeypatch, value):
    """Changed package files stop the check, and the message names the opt-in; only the value
    1 opts in."""
    repo, c = history
    if value is None:
        monkeypatch.delenv('ZALMOXIS_PALEOS_ALLOW_DIRTY', raising=False)
    else:
        monkeypatch.setenv('ZALMOXIS_PALEOS_ALLOW_DIRTY', value)
    (repo / 'paleos' / 'eos.py').write_text('local change')
    with pytest.raises(PaleosVersionError, match='ZALMOXIS_PALEOS_ALLOW_DIRTY=1') as err:
        check_paleos_commit(repo, c['tested'])
    assert c['new'] in str(err.value) and 'core.autocrlf' in str(err.value)


def test_old_package_files_at_a_newer_commit_stop(history):
    """Package files of an older commit checked out at a newer HEAD are not the newer code."""
    repo, c = history
    _git(repo, 'checkout', c['old'], '--', 'paleos')
    assert (repo / 'paleos' / 'eos.py').read_text() == 'old'
    with pytest.raises(PaleosVersionError, match='differ from its commit'):
        check_paleos_commit(repo, c['tested'])


@pytest.mark.parametrize(
    'name',
    ['.DS_Store', 'mod.pyo', '__pycache__/x.txt', '__pycache__/m.cpython-312.pyc', 'm.pyx'],
)
def test_a_file_python_does_not_import_leaves_the_key_clean(history, monkeypatch, name):
    """Finder metadata, .pyo, __pycache__ content and source that is not Python are not code."""
    repo, c = history
    f = repo / 'paleos' / name
    f.parent.mkdir(exist_ok=True)
    f.write_bytes(b'\x00junk')
    assert _dirty_key(repo, monkeypatch) == c['new']
    assert check_paleos_commit(repo, c['new']) == 'tested'


@pytest.mark.parametrize('name', ['extra.pyc', 'mod.so', 'mod.pyd', 'mod.py', 'sub/mod.py'])
def test_an_untracked_file_python_imports_makes_the_key_dirty(history, monkeypatch, name):
    """A sourceless .pyc, an extension module and a module in a subfolder are code."""
    repo, c = history
    f = repo / 'paleos' / name
    f.parent.mkdir(exist_ok=True)
    f.write_bytes(b'code')
    assert _dirty_key(repo, monkeypatch).startswith(c['new'] + '-dirty-')


def test_an_ignored_module_is_code(history, monkeypatch):
    """A module git ignores is still imported, so it makes the key dirty."""
    repo, c = history
    (repo / '.gitignore').write_text('.venv/\npaleos/_gen.py\n')
    (repo / 'paleos' / '_gen.py').write_text('X = 1\n')
    assert _git(repo, 'status', '--porcelain', '--', 'paleos') == ''
    assert _dirty_key(repo, monkeypatch).startswith(c['new'] + '-dirty-')


def _commit_files(repo, files, msg):
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text)
    _git(repo, 'add', '-A')
    _git(
        repo,
        *('-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false'),
        *('-c', 'core.hooksPath=/dev/null', 'commit', '-qm', msg),
    )
    return _git(repo, 'rev-parse', 'HEAD')


def test_a_tracked_symlink_leaves_the_key_clean(history, monkeypatch):
    """A committed symlink is compared by its link text, as git stores it."""
    import os

    repo, _ = history
    os.symlink('eos.py', repo / 'paleos' / 'link.py')
    head = _commit_files(repo, {}, 'link')
    assert _dirty_key(repo, monkeypatch) == head
    os.remove(repo / 'paleos' / 'link.py')
    os.symlink('__init__.py', repo / 'paleos' / 'link.py')
    assert _dirty_key(repo, monkeypatch).startswith(head + '-dirty-')


def test_an_eol_filter_makes_a_fresh_checkout_dirty(history, monkeypatch):
    """With eol=crlf the files on disk differ from the blobs by line ends; files are hashed
    without git filters, so the checkout counts as changed."""
    repo, _ = history
    files = {'.gitattributes': '*.py text eol=crlf\n', 'paleos/eos.py': 'A = 1\n'}
    head = _commit_files(repo, files, 'eol')
    (repo / 'paleos' / 'eos.py').unlink()
    _git(repo, 'checkout', '--', 'paleos/eos.py')
    assert (repo / 'paleos' / 'eos.py').read_bytes().endswith(b'\r\n')
    assert _dirty_key(repo, monkeypatch).startswith(head + '-dirty-')


def test_a_deleted_tracked_module_makes_the_key_dirty(history, monkeypatch):
    """A committed module removed from disk is a change to the code."""
    repo, c = history
    (repo / 'paleos' / 'eos.py').unlink()
    assert _dirty_key(repo, monkeypatch).startswith(c['new'] + '-dirty-')


def test_a_module_in_a_subfolder_is_compared(history, monkeypatch):
    """A committed module in a subfolder is clean as committed and dirty when edited."""
    repo, _ = history
    head = _commit_files(repo, {'paleos/sub/mod.py': 'Y = 1\n'}, 'sub')
    assert _dirty_key(repo, monkeypatch) == head
    (repo / 'paleos' / 'sub' / 'mod.py').write_text('Y = 2\n')
    assert _dirty_key(repo, monkeypatch).startswith(head + '-dirty-')


def test_a_mode_change_alone_leaves_the_key_clean(history, monkeypatch):
    """The key compares file content; an exec bit is no code change."""
    repo, c = history
    (repo / 'paleos' / 'eos.py').chmod(0o755)
    assert _dirty_key(repo, monkeypatch) == c['new']


def test_git_variables_of_the_caller_do_not_reach_the_check(history, tmp_path, monkeypatch):
    """A package copy outside git stops even when GIT_DIR names a clone, as in a git hook."""
    import shutil

    repo, c = history
    copy = tmp_path / 'copy'
    shutil.copytree(repo / 'paleos', copy / 'paleos')
    monkeypatch.setenv('GIT_DIR', str(repo / '.git'))
    monkeypatch.setenv('GIT_WORK_TREE', str(repo))
    monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(tmp_path))
    with pytest.raises(PaleosVersionError, match='cannot tell'):
        check_paleos_commit(copy, c['tested'])


def test_an_unreadable_module_stops_the_check_and_the_key_falls_back(history, monkeypatch):
    """A module that cannot be read is a PaleosVersionError, not an OSError, and the key falls
    back to the version, which no cache accepts."""
    import sys
    import types

    repo, c = history
    fake = types.ModuleType('paleos')
    fake.__file__, fake.__version__ = str(repo / 'paleos' / '__init__.py'), '1.0'
    monkeypatch.setitem(sys.modules, 'paleos', fake)
    target = repo / 'paleos' / 'eos.py'
    target.chmod(0)
    try:
        assert paleos_api.paleos_installed_sha() == 'version-1.0'
        with pytest.raises(PaleosVersionError, match='cannot read'):
            check_paleos_commit(repo, c['new'])
    finally:
        target.chmod(0o644)


@pytest.mark.parametrize('key', ['a' * 40, 'a' * 40 + '-dirty-0123abcd'])
def test_the_cache_accepts_a_commit_key(monkeypatch, key):
    """A full commit SHA, with or without the dirty suffix, is a cache key."""
    monkeypatch.setattr(paleos_api, 'check_paleos_commit', lambda: 'tested')
    paleos_api._check_paleos_once.cache_clear()
    try:
        assert paleos_api._check_paleos_once(key) == 'tested'
    finally:
        paleos_api._check_paleos_once.cache_clear()


@pytest.mark.parametrize(
    'key',
    [
        'version-0.1.0',
        'paleos-not-installed',
        'a' * 39,
        'a' * 41,
        'A' * 40,
        'a' * 40 + '-dirty-0123abc',
        'a' * 40 + '-dirty-0123ABCD',
        'a' * 40 + '-dirty-0123abcd0',
    ],
)
def test_the_cache_refuses_a_key_that_names_no_commit(monkeypatch, key):
    """Fallback keys and malformed SHAs or suffixes are refused."""
    monkeypatch.setattr(paleos_api, 'check_paleos_commit', lambda: 'tested')
    paleos_api._check_paleos_once.cache_clear()
    try:
        with pytest.raises(PaleosVersionError, match='names no commit'):
            paleos_api._check_paleos_once(key)
    finally:
        paleos_api._check_paleos_once.cache_clear()


def test_the_key_at_the_first_import_is_kept(history, monkeypatch):
    """The first import of paleos in a process records its key; a later edit on disk does
    not move the recorded key."""
    import sys

    repo, c = history
    monkeypatch.delitem(sys.modules, 'paleos', raising=False)
    monkeypatch.syspath_prepend(str(repo))
    monkeypatch.setattr(paleos_api, '_KEY_AT_IMPORT', {})
    assert paleos_api.paleos_installed_sha() == c['new']
    (repo / 'paleos' / 'eos.py').write_text('edited later')
    assert paleos_api.paleos_installed_sha().startswith(c['new'] + '-dirty-')
    assert paleos_api._KEY_AT_IMPORT == {'key': c['new']}


def test_an_unreadable_untracked_module_stops_the_check(history):
    """An untracked module that cannot be read is a PaleosVersionError that names it."""
    repo, c = history
    extra = repo / 'paleos' / 'extra.py'
    extra.write_text('X = 1\n')
    extra.chmod(0)
    try:
        with pytest.raises(PaleosVersionError, match='cannot read paleos/extra.py'):
            check_paleos_commit(repo, c['new'])
    finally:
        extra.chmod(0o644)


def test_a_symlink_git_cannot_hash_stops_the_check(history, monkeypatch):
    """A failed hash of a symlink target is a PaleosVersionError that names the link."""
    import os

    repo, _ = history
    os.symlink('eos.py', repo / 'paleos' / 'link.py')
    head = _commit_files(repo, {}, 'link')
    real = paleos_api._git

    def git(repo_dir, *args, stdin=None):
        if args[:2] == ('hash-object', '--no-filters'):
            return paleos_api._subprocess.CompletedProcess(args, 128, '', 'fatal: no hash')
        return real(repo_dir, *args, stdin=stdin)

    monkeypatch.setattr(paleos_api, '_git', git)
    with pytest.raises(PaleosVersionError, match='cannot hash paleos/link.py'):
        check_paleos_commit(repo, head)


def test_a_replace_ref_does_not_make_an_unrelated_commit_contain_the_tested_one(history):
    """git replace can graft the tested commit under any commit; the check ignores replace
    refs, so an unrelated checkout still stops."""
    repo, c = history
    _git(repo, 'checkout', '-q', c['unrelated'])
    _git(repo, 'replace', '--graft', c['unrelated'], c['tested'])
    assert _git(repo, 'merge-base', '--is-ancestor', c['tested'], c['unrelated']) == ''
    with pytest.raises(PaleosVersionError, match='does not contain'):
        check_paleos_commit(repo, c['tested'])


def test_a_committed_path_that_is_not_utf8_gives_a_version_error(history):
    """git output with bytes that are not UTF-8 is decoded without failing, so a committed
    module with such a name, absent on disk, is a change and stops the check."""
    import subprocess

    repo, _ = history
    blob = _git(repo, 'hash-object', '-w', '--stdin', '--path', 'x.py')
    name = b'paleos/caf\xe9.py'
    subprocess.run(
        [
            b'git',
            b'-C',
            str(repo).encode(),
            b'update-index',
            b'--add',
            b'--cacheinfo',
            b'100644,' + blob.encode() + b',' + name,
        ],
        check=True,
    )
    _git(
        repo,
        *('-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false'),
        *('-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'bytes'),
    )
    head = _git(repo, 'rev-parse', 'HEAD')
    err = None
    try:
        check_paleos_commit(repo, head)
    except Exception as exc:  # noqa: BLE001 -- the error type is the assertion
        err = exc
    assert isinstance(err, PaleosVersionError), repr(err)
    assert 'differ from its commit' in str(err)


@pytest.mark.parametrize('kind', ['symlink', 'nested repository'])
@pytest.mark.usefixtures('allow_dirty')
def test_a_link_to_a_directory_or_a_nested_repository_stops_even_when_allowed(
    history, tmp_path, kind
):
    """An untracked link to a directory or a nested repository under paleos/ can hold code
    Python imports that no key can describe: the check stops, also with the opt-in."""
    repo, c = history
    ext = repo / 'paleos' / 'ext'
    if kind == 'symlink':
        (tmp_path / 'outside').mkdir()
        (tmp_path / 'outside' / 'mod.py').write_text('K = 1\n')
        ext.symlink_to(tmp_path / 'outside', target_is_directory=True)
    else:
        ext.mkdir()
        (ext / 'mod.py').write_text('K = 1\n')
        _git(ext, 'init', '-q')
    with pytest.raises(PaleosVersionError, match='no key can describe'):
        check_paleos_commit(repo, c['new'])


@pytest.mark.usefixtures('allow_dirty')
def test_a_plain_untracked_directory_without_modules_is_no_change(history):
    """An untracked directory of data files under paleos/ holds nothing Python imports."""
    repo, c = history
    (repo / 'paleos' / 'notes').mkdir()
    (repo / 'paleos' / 'notes' / 'readme.txt').write_text('notes')
    assert check_paleos_commit(repo, c['new']) == 'tested'


def test_a_package_below_the_top_level_is_compared(tmp_path):
    """With paleos/ under src/, the files are hashed by their path in the repository."""
    repo = tmp_path / 'PALEOS'
    (repo / 'src' / 'paleos').mkdir(parents=True)
    _git(repo, 'init', '-q')
    (repo / 'src' / 'paleos' / '__init__.py').write_text('')
    (repo / 'src' / 'paleos' / 'eos.py').write_text('A = 1\n')
    _git(repo, 'add', '.')
    _git(
        repo,
        *('-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false'),
        *('-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'src'),
    )
    head = _git(repo, 'rev-parse', 'HEAD')
    try:
        clean = paleos_api._changes_hash(repo / 'src', head)
    except PaleosVersionError as exc:
        pytest.fail(f'a src layout is refused: {exc}')
    assert clean is None
    (repo / 'src' / 'paleos' / 'eos.py').write_text('A = 2\n')
    assert paleos_api._changes_hash(repo / 'src', head) is not None


def test_a_clean_filter_from_the_environment_does_not_hide_an_edit(
    history, monkeypatch, tmp_path
):
    """A clean filter that git config from the environment applies cannot map edited code
    back to its commit: files are hashed without filters."""
    repo, c = history
    attributes = tmp_path / 'attributes'
    attributes.write_text('*.py filter=hide\n')
    for i, (key, value) in enumerate(
        [('filter.hide.clean', 'sed s/nex/new/'), ('core.attributesFile', str(attributes))]
    ):
        monkeypatch.setenv(f'GIT_CONFIG_KEY_{i}', key)
        monkeypatch.setenv(f'GIT_CONFIG_VALUE_{i}', value)
    monkeypatch.setenv('GIT_CONFIG_COUNT', '2')
    (repo / 'paleos' / 'eos.py').write_text('nex')
    assert _dirty_key(repo, monkeypatch).startswith(c['new'] + '-dirty-')


def test_a_paleos_imported_before_the_first_key_is_covered_from_then_on(
    history, monkeypatch, caplog
):
    """When paleos is imported before the first key, that key is recorded with a warning, so
    an edit after it is refused."""
    import sys

    repo, c = history
    monkeypatch.delitem(sys.modules, 'paleos', raising=False)
    monkeypatch.syspath_prepend(str(repo))
    monkeypatch.setattr(paleos_api, '_KEY_AT_IMPORT', {})
    import paleos  # noqa: F401

    with caplog.at_level('WARNING', logger=paleos_api.logger.name):
        assert paleos_api.paleos_installed_sha() == c['new']
    assert paleos_api._KEY_AT_IMPORT == {'key': c['new']}
    assert 'imported before' in caplog.text
    (repo / 'paleos' / 'eos.py').write_text('edited later')
    with pytest.raises(PaleosVersionError, match='changed on disk since the first key'):
        paleos_api._same_code_as_import(paleos_api.paleos_installed_sha())


def test_a_staged_new_module_is_a_change(history, monkeypatch):
    """A module added to the index but not committed is code the commit does not hold: it is
    refused without the opt-in, and keyed by its content on disk with it."""
    repo, c = history
    (repo / 'paleos' / 'helper.py').write_text('K = 1\n')
    _git(repo, 'add', 'paleos/helper.py')
    with pytest.raises(PaleosVersionError, match='differ from its commit'):
        check_paleos_commit(repo, c['new'])
    monkeypatch.setenv('ZALMOXIS_PALEOS_ALLOW_DIRTY', '1')
    a = _dirty_key(repo, monkeypatch)
    (repo / 'paleos' / 'helper.py').write_text('K = 2\n')
    b = _dirty_key(repo, monkeypatch)
    assert a.startswith(c['new'] + '-dirty-') and b.startswith(c['new'] + '-dirty-') and a != b


@pytest.mark.usefixtures('allow_dirty')
@pytest.mark.parametrize('kind', ['symlink', 'submodule'])
def test_a_committed_link_to_a_directory_or_submodule_stops_even_when_allowed(
    history, tmp_path, kind
):
    """A committed link to a directory or a committed submodule under paleos/ holds code that no
    key can describe: the check stops, also with the opt-in."""
    repo, _ = history
    if kind == 'symlink':
        (tmp_path / 'outside').mkdir()
        (tmp_path / 'outside' / 'mod.py').write_text('K = 1\n')
        (repo / 'paleos' / 'ext').symlink_to(tmp_path / 'outside', target_is_directory=True)
        _git(repo, 'add', 'paleos/ext')
    else:
        sha = _git(repo, 'rev-parse', 'HEAD')
        (repo / 'paleos' / 'sub').mkdir()
        _git(repo, 'update-index', '--add', '--cacheinfo', f'160000,{sha},paleos/sub')
    _git(
        repo,
        *('-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false'),
        *('-c', 'core.hooksPath=/dev/null', 'commit', '-qm', kind),
    )
    head = _git(repo, 'rev-parse', 'HEAD')
    with pytest.raises(PaleosVersionError, match='no key can describe'):
        check_paleos_commit(repo, head)
