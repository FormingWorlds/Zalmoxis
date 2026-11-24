"""Tests for ``tools.validation.grid_start_up_reference`` on written files, no solves."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]
_TRUE = object()  # stands for the md5 of the table on disk


def _tool():
    from tools.validation import grid_start_up_reference

    return grid_start_up_reference


def _write(tmp_path, table_bytes=b'P T rho\n1 2 3\n', stored_md5=_TRUE):
    """A table file, the material dictionaries that point to it and a reference that names it
    with ``stored_md5`` (the true md5 by default); returns (reference path, dictionaries)."""
    table = tmp_path / 'data' / 'EOS_X' / 'x.dat'
    table.parent.mkdir(parents=True)
    table.write_bytes(table_bytes)
    true = hashlib.md5(table_bytes).hexdigest()
    block = {
        'radii': [0.0, 1.0e6, 2.0e6],
        'pressure': [3.0e11, 1.0e10, 1.0e5],
        'temperature': [6000.0, 5000.0, 4000.0],
        'mass_enclosed': [0.0, 3.9e24, 4.18e24],
        'cmb_mass': 3.97e24,
    }
    ref = {
        'tables': {
            'PALEOS:iron': {
                'file': 'EOS_X/x.dat',
                'md5': true if stored_md5 is _TRUE else stored_md5,
            }
        },
        **{key: block for key in ('first', 'resolve', 'first_jax', 'resolve_jax')},
    }
    path = tmp_path / 'ref.json'
    _tool().dump_compact(ref, path)
    dicts = {'PALEOS:iron': {'eos_file': str(table), 'format': 'paleos_unified'}}
    return path, dicts


def test_a_table_whose_md5_differs_stops_the_reader(tmp_path):
    """A table that changed after the reference was solved raises with both md5s and the
    command that writes the reference again."""
    path, dicts = _write(tmp_path, stored_md5='0' * 32)
    with pytest.raises(ValueError, match='PALEOS:iron') as err:
        _tool().load(path, dicts)
    assert '0' * 32 in str(err.value)
    assert hashlib.md5(b'P T rho\n1 2 3\n').hexdigest() in str(err.value)
    assert 'python -m tools.validation.grid_start_up_reference' in str(err.value)


def test_a_missing_table_stops_the_reader(tmp_path):
    """A table file that is not on disk raises before any comparison."""
    path, dicts = _write(tmp_path)
    (tmp_path / 'data' / 'EOS_X' / 'x.dat').unlink()
    with pytest.raises(ValueError, match='not on disk'):
        _tool().load(path, dicts)


def test_a_table_the_dictionaries_do_not_name_stops_the_reader(tmp_path):
    """A stored table whose material is not in the dictionaries raises with its name."""
    path, _ = _write(tmp_path)
    with pytest.raises(ValueError, match='PALEOS:iron'):
        _tool().load(path, {})


def test_the_reader_returns_the_stored_structures(tmp_path):
    """With matching md5s the reader returns the 4 structures as arrays, cmb_mass a float."""
    path, dicts = _write(tmp_path)
    ref = _tool().load(path, dicts)
    assert set(ref) >= {'first', 'resolve', 'first_jax', 'resolve_jax'}
    assert isinstance(ref['resolve_jax']['pressure'], np.ndarray)
    np.testing.assert_array_equal(ref['first']['pressure'], [3.0e11, 1.0e10, 1.0e5])
    assert isinstance(ref['resolve']['radii'], np.ndarray)
    assert ref['resolve']['cmb_mass'] == 3.97e24


@settings(max_examples=40, deadline=None)
@given(
    st.one_of(
        st.text(alphabet='0123456789abcdef', min_size=32, max_size=32),
        st.text(),
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=True),
    )
)
def test_every_other_md5_stops_the_reader(tmp_path_factory, stored):
    """Any stored md5 other than that of the table on disk raises, also one of another type
    (null, a TOML-like bool, a number)."""
    tmp_path = tmp_path_factory.mktemp('md5')
    true = hashlib.md5(b'P T rho\n1 2 3\n').hexdigest()
    path, dicts = _write(tmp_path, stored_md5=stored)
    if stored == true:
        assert _tool().load(path, dicts)['first']['cmb_mass'] == 3.97e24
    else:
        with pytest.raises(ValueError, match='md5'):
            _tool().load(path, dicts)


def test_the_compact_file_reads_back_equal_with_one_block_per_line(tmp_path):
    """json.load of the written file equals the object; each top-level key is one line with
    no indentation inside it, and floats keep every bit."""
    obj = {'a': {'x': [0.1, 1 / 3, 6.02214076e23]}, 'b': [1, 2], 'c': 'text'}
    path = tmp_path / 'c.json'
    _tool().dump_compact(obj, path)
    assert json.loads(path.read_text()) == obj
    lines = path.read_text().splitlines()
    assert lines[0] == '{' and lines[-1] == '}' and len(lines) == 2 + len(obj)
    assert not any(line.startswith(' ') for line in lines)


def test_the_source_hash_follows_the_code_and_not_the_order(tmp_path):
    """The hash changes with one constant of one file or the name of a file, and stays with
    the order of the names."""
    for name, text in (('a.py', 'x = 1\n'), ('b.py', 'y = 2\n')):
        (tmp_path / name).write_text(text)
    h = _tool().source_hash(['a.py', 'b.py'], tmp_path)
    assert h == _tool().source_hash(['b.py', 'a.py'], tmp_path)
    (tmp_path / 'c.py').write_text('y = 2\n')
    assert _tool().source_hash(['a.py', 'c.py'], tmp_path) != h
    (tmp_path / 'b.py').write_text('y = 3\n')
    assert _tool().source_hash(['a.py', 'b.py'], tmp_path) != h


def _fake_solve(label='strict'):
    """A stand-in for ``solve``: a 3-node structure with the accept label ``label``."""

    def solve(cfg, **kwargs):
        n = np.array([0.0, 0.5, 1.0])
        return {
            'converged': True,
            'accepted_on': label,
            'adiabat_blend': 1.0,
            'adiabat_table_blend': 1.0,
            'radii': n * 6.0e6,
            'pressure': 3.0e11 * (1.0 - n) + 1.0e5,
            'temperature': 6000.0 - 2000.0 * n,
            'mass_enclosed': n * cfg['planet_mass'],
            'cmb_mass': cfg['core_mass_fraction'] * cfg['planet_mass'],
        }

    return solve


def _fake_tables(tmp_path, monkeypatch):
    """Material dictionaries of the 2 case materials on small table files in ``tmp_path``."""
    import zalmoxis.config

    dicts = {}
    for name in ('PALEOS:iron', 'PALEOS:MgSiO3'):
        f = tmp_path / f'{name.split(":")[1]}.dat'
        f.write_text(f'{name} table\n')
        dicts[name] = {'eos_file': str(f), 'format': 'paleos_unified'}
    monkeypatch.setattr(zalmoxis.config, 'load_material_dictionaries', lambda: dicts)
    return dicts


def test_the_writer_stores_the_4_structures_with_tables_and_source(tmp_path, monkeypatch):
    """write solves the 4 cases at REF_TOL and stores the structures, the table md5s, the
    commit and the hash of the loaded modules; the reader reads the file back."""
    tool = _tool()
    seen = []
    solve = _fake_solve()
    monkeypatch.setattr(tool, 'solve', lambda cfg: seen.append(cfg) or solve(cfg))
    dicts = _fake_tables(tmp_path, monkeypatch)
    ref = tool.write(tmp_path / 'ref.json')
    assert [c['tolerance_outer'] for c in seen] == [1e-6] * 4
    assert [c['planet_mass'] for c in seen] == [4.1804e24, 4.177602725330534e24] * 2
    assert [c['use_jax'] for c in seen] == [False, False, True, True]
    assert (
        ref['tables']['PALEOS:iron']['md5'] == hashlib.md5(b'PALEOS:iron table\n').hexdigest()
    )
    assert len(ref['commit']) == 40 and 'config.py' in ref['source']['files']
    pkg, files = tool.loaded_sources()
    assert ref['source']['sha256'] == tool.source_hash(ref['source']['files'], pkg)
    assert ref['source'].get('python') == tool.PYTHON
    back = tool.load(tmp_path / 'ref.json', dicts)
    np.testing.assert_array_equal(
        back['resolve']['mass_enclosed'], np.array([0.0, 0.5, 1.0]) * 4.177602725330534e24
    )
    assert back['first']['cmb_mass'] == 0.95 * 4.1804e24
    assert {'first', 'first_jax', 'resolve', 'resolve_jax'} <= set(back)


def test_the_writer_refuses_a_reference_that_is_not_strict(tmp_path, monkeypatch):
    """A reference solve accepted on any other label stops the writer and writes no file."""
    tool = _tool()
    monkeypatch.setattr(tool, 'solve', _fake_solve('mass_only'))
    _fake_tables(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="first: the reference solve accepted 'mass_only'"):
        tool.write(tmp_path / 'ref.json')
    assert not (tmp_path / 'ref.json').exists()


def test_the_command_writes_the_file_it_names(tmp_path, monkeypatch, capsys):
    """main --out writes there and prints the commit and the source hash."""
    tool = _tool()
    monkeypatch.setattr(tool, 'solve', _fake_solve())
    _fake_tables(tmp_path, monkeypatch)
    tool.main(['--out', str(tmp_path / 'cli.json')])
    ref = json.loads((tmp_path / 'cli.json').read_text())
    assert capsys.readouterr().out.split() == [ref['commit'], ref['source']['sha256']]


def test_the_reader_takes_the_zalmoxis_dictionaries_by_default(tmp_path, monkeypatch):
    """Without dictionaries the reader checks the tables that load_material_dictionaries
    names."""
    import zalmoxis.config

    path, dicts = _write(tmp_path, stored_md5='f' * 32)
    monkeypatch.setattr(zalmoxis.config, 'load_material_dictionaries', lambda: dicts)
    with pytest.raises(ValueError, match='f' * 32):
        _tool().load(path)


def test_the_solve_passes_the_grid_melting_curves_and_input_folder(monkeypatch):
    """solve calls main with the config, the PALEOS liquidus and 0.8 of it as the solidus
    (the mushy zone factor of the case), the input folder and the extra keywords."""
    import zalmoxis.solver as zs
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.melting_curves import paleos_liquidus

    got = {}
    monkeypatch.setattr(zs, 'main', lambda *a, **k: got.update(a=a, k=k) or 'result')
    cfg = {'planet_mass': 1.0}
    assert _tool().solve(cfg, adiabat_seed='seed') == 'result'
    c, _, (solidus, liquidus), inp = got['a']
    assert c is cfg and got['k'] == {'adiabat_seed': 'seed'}
    assert inp == os.path.join(get_zalmoxis_root(), 'input')
    P = np.array([1.0e9, 50.0e9])
    np.testing.assert_allclose(liquidus(P), paleos_liquidus(P))
    np.testing.assert_allclose(solidus(P), 0.8 * paleos_liquidus(P), rtol=1e-12)


def test_the_source_list_leaves_out_the_generated_version_file(monkeypatch):
    """_version.py is written at install time with the commit in it and git ignores it, so it
    is not part of the source the hash stands for; a stand-in module inside the package keeps
    the test valid in a checkout without the file."""
    import sys
    from types import SimpleNamespace

    import zalmoxis

    version = os.path.join(os.path.dirname(zalmoxis.__file__), '_version.py')
    monkeypatch.setitem(sys.modules, 'zalmoxis._version', SimpleNamespace(__file__=version))
    pkg, files = _tool().loaded_sources()
    assert '_version.py' not in files and 'config.py' in files
    assert all((pkg / f).is_file() for f in files)


def test_a_generated_file_from_another_install_is_left_out(monkeypatch):
    """A _version.py that another install serves (a checkout without its own) is left out
    before its path is read; any other zalmoxis module from outside the package raises."""
    import sys
    from types import SimpleNamespace

    monkeypatch.setitem(
        sys.modules, 'zalmoxis._version', SimpleNamespace(__file__='/elsewhere/_version.py')
    )
    pkg, files = _tool().loaded_sources()
    assert '_version.py' not in files and 'config.py' in files
    monkeypatch.setitem(
        sys.modules, 'zalmoxis.other', SimpleNamespace(__file__='/elsewhere/other.py')
    )
    with pytest.raises(ValueError, match='other.py'):
        _tool().loaded_sources()


_MOD = '''"""Module doc."""

X = 1.5  # a comment
"""Doc of X."""
NAME = 'iron'


class A:
    """Class doc."""

    def f(self, y):
        """Function doc."""
        print(NAME)
        return y * X

    def g(self):
        try:
            pass
        except ValueError:
            'except text'
        else:
            'else text'
        finally:
            'finally text'
'''


def _hash_of(tmp_path, text):
    """The source hash of one module ``a.py`` with ``text``."""
    (tmp_path / 'a.py').write_text(text)
    return _tool().source_hash(['a.py'], tmp_path)


@pytest.mark.parametrize(
    'old, new',
    [
        ('"""Module doc."""', '"""Another module doc."""'),
        ('# a comment', '# another, longer comment'),
        ('"""Doc of X."""', '"""Another doc of X."""'),
        ('"""Class doc."""', '"""Class doc\n\n    over 3 lines."""'),
        ('        """Function doc."""\n', ''),
        ('\n\nclass A', '\n\n\n# a block\n# of comments\nclass A'),
        ("'except text'", "'other text'"),
        ("'else text'", "'other text'"),
        ("'finally text'", "'other text'"),
    ],
    ids=[
        'module-doc',
        'comment',
        'attribute-doc',
        'class-doc',
        'function-doc-gone',
        'lines',
        'except-text',
        'else-text',
        'finally-text',
    ],
)
def test_a_docstring_or_comment_edit_keeps_the_source_hash(tmp_path, old, new):
    """A docstring, a string after an assignment or a comment does nothing at run time, so
    an edit of one, also one that moves every later line, keeps the hash."""
    h = _hash_of(tmp_path, _MOD)
    assert _MOD.count(old) == 1
    assert _hash_of(tmp_path, _MOD.replace(old, new)) == h


@pytest.mark.parametrize(
    'old, new',
    [
        ('X = 1.5', 'X = 1.6'),
        ('y * X', 'y / X'),
        ('def f(self, y)', 'def f(self, z, y=y)'),
        ("NAME = 'iron'", "NAME = 'rock'"),
        ('print(NAME)', 'print(X)'),
    ],
    ids=['constant', 'operator', 'signature', 'string-constant', 'call-statement'],
)
def test_a_code_edit_changes_the_source_hash(tmp_path, old, new):
    """One changed constant, operator, argument, string value or call changes the hash."""
    h = _hash_of(tmp_path, _MOD)
    assert _MOD.count(old) == 1
    assert _hash_of(tmp_path, _MOD.replace(old, new)) != h


def test_the_source_check_names_the_restamp_for_another_python_minor(tmp_path):
    """The hash is the syntax tree of one Python minor: a stored minor other than the running
    one fails with the source-only command, and a code change fails with the full write."""
    tool = _tool()
    minor = f'{sys.version_info.major}.{sys.version_info.minor}'
    stored = {'files': ['a.py'], 'sha256': _hash_of(tmp_path, _MOD), 'python': minor}
    assert tool.source_mismatch(stored, tmp_path) is None
    msg = tool.source_mismatch({**stored, 'python': '3.9'}, tmp_path)
    assert msg is not None and 'Python 3.9' in msg and f'{tool.REFRESH} --source-only' in msg
    (tmp_path / 'a.py').write_text(_MOD.replace('X = 1.5', 'X = 1.6'))
    msg = tool.source_mismatch(stored, tmp_path)
    assert msg is not None and 'source' in msg and msg.endswith(tool.REFRESH)
    assert tool.source_mismatch({k: v for k, v in stored.items() if k != 'python'}, tmp_path)


def _git(root, *args):
    """Run git in ``root`` and return its output."""
    user = ('-c', 'user.name=t', '-c', 'user.email=t@t', '-c', 'commit.gpgsign=false')
    cmd = ('git', '-C', str(root), *user, *args)
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout.strip()


def _repo_with_reference(tmp_path, edit):
    """A git repository with ``src/zalmoxis/a.py`` committed, a reference that names that
    commit with a hash of another Python minor, and a second commit with ``edit`` applied;
    returns (root, package folder, reference path)."""
    root = tmp_path / 'repo'
    pkg = root / 'src' / 'zalmoxis'
    pkg.mkdir(parents=True)
    _git(root, 'init', '-q')
    (pkg / 'a.py').write_text(_MOD)
    _git(root, 'add', '.')
    _git(root, 'commit', '-qm', 'first')
    ref = {
        'commit': _git(root, 'rev-parse', 'HEAD'),
        'source_tree_clean': True,
        'source': {'files': ['a.py'], 'sha256': 'f' * 64, 'python': '3.9'},
        'first': {'radii': [0.0, 1.0e6], 'cmb_mass': 3.97e24},
    }
    path = tmp_path / 'ref.json'
    _tool().dump_compact(ref, path)
    (pkg / 'a.py').write_text(_MOD.replace(*edit))
    _git(root, 'commit', '-qam', 'second')
    return root, pkg, path


def test_the_source_only_restamp_keeps_the_structures_after_a_docstring_edit(tmp_path):
    """With the code at the stored commit and now equal but for docstrings, the restamp
    writes the hash and minor of the running Python and keeps every other value."""
    tool = _tool()
    root, pkg, path = _repo_with_reference(tmp_path, ('"""Function doc."""', '"""New."""'))
    before = json.loads(path.read_text())
    tool.restamp_source(path, root, pkg)
    after = json.loads(path.read_text())
    assert after['source']['sha256'] == tool.source_hash(['a.py'], pkg) != 'f' * 64
    assert after['source']['python'] == f'{sys.version_info.major}.{sys.version_info.minor}'
    assert {k: v for k, v in after.items() if k != 'source'} == {
        k: v for k, v in before.items() if k != 'source'
    }
    assert after['source']['files'] == ['a.py']


def test_the_source_only_restamp_refuses_a_code_change(tmp_path):
    """A code edit since the stored commit stops the restamp, names the full write and
    leaves the file as it was; so do a stored commit that is not a sha or not in the
    repository, a listed file that is not at the stored commit or not in the package, a module
    that does not parse on either side, and an imported package outside the repository."""
    tool = _tool()
    root, pkg, path = _repo_with_reference(tmp_path, ('X = 1.5', 'X = 1.6'))
    before = path.read_bytes()
    with pytest.raises(ValueError, match='changed since') as err:
        tool.restamp_source(path, root, pkg)
    assert err.value.args[0].endswith(tool.REFRESH)
    assert path.read_bytes() == before
    ref = json.loads(before)
    tool.dump_compact({**ref, 'commit': '0' * 40}, path)
    with pytest.raises(ValueError, match='0' * 40):
        tool.restamp_source(path, root, pkg)
    tool.dump_compact({**ref, 'source': {**ref['source'], 'files': ['a.py', 'b.py']}}, path)
    with pytest.raises(ValueError, match='src/zalmoxis/b.py at commit'):
        tool.restamp_source(path, root, pkg)
    with pytest.raises(ValueError, match='the imported zalmoxis'):
        tool.restamp_source(path, root, tmp_path)
    for commit in (None, int('1' * 40)):
        tool.dump_compact({**ref, 'commit': commit}, path)
        with pytest.raises(ValueError, match='not a commit sha'):
            tool.restamp_source(path, root, pkg)
    (pkg / 'a.py').write_text('X = = 1\n')
    tool.dump_compact(ref, path)
    with pytest.raises(ValueError, match=r'a\.py.*run python'):
        tool.restamp_source(path, root, pkg)
    _git(root, 'add', 'src/zalmoxis/a.py')
    _git(root, 'commit', '-qm', 'broken')
    (pkg / 'a.py').write_text(_MOD)
    tool.dump_compact({**ref, 'commit': _git(root, 'rev-parse', 'HEAD')}, path)
    with pytest.raises(ValueError, match=r'a\.py.*run python'):
        tool.restamp_source(path, root, pkg)
    _git(root, 'rm', '-qf', 'src/zalmoxis/a.py')
    tool.dump_compact(ref, path)
    with pytest.raises(ValueError, match='a.py') as err:
        tool.restamp_source(path, root, pkg)
    assert err.value.args[0].endswith(tool.REFRESH)
    assert path.read_bytes() == before


def test_the_source_only_restamp_refuses_structures_from_a_tree_with_changes(tmp_path):
    """Structures written from a tree with uncommitted changes are not the code of the stored
    commit, so the restamp refuses them and leaves the file as it was."""
    tool = _tool()
    root, pkg, path = _repo_with_reference(tmp_path, ('"""Function doc."""', '"""New."""'))
    tool.dump_compact({**json.loads(path.read_text()), 'source_tree_clean': False}, path)
    before = path.read_bytes()
    with pytest.raises(ValueError, match='tree with changes'):
        tool.restamp_source(path, root, pkg)
    assert path.read_bytes() == before


def test_the_source_only_restamp_takes_only_a_commit_sha(tmp_path):
    """A stored commit that is not a hex sha stops the restamp before git reads it, so an
    option in its place never runs."""
    tool = _tool()
    root, pkg, path = _repo_with_reference(tmp_path, ('"""Function doc."""', '"""New."""'))
    _git(root, 'remote', 'add', 'origin', str(root))
    marker = tmp_path / 'ran'
    tool.dump_compact(
        {**json.loads(path.read_text()), 'commit': f'--upload-pack=touch {marker};'}, path
    )
    with pytest.raises(ValueError, match='not a commit sha'):
        tool.restamp_source(path, root, pkg)
    assert not marker.exists()


def test_the_source_only_restamp_fetches_a_commit_that_only_the_origin_holds(tmp_path):
    """A stored commit that is on no local branch (a squash-merged branch) is fetched by sha."""
    tool = _tool()
    root, pkg, path = _repo_with_reference(tmp_path, ('"""Function doc."""', '"""New."""'))
    up = tmp_path / 'up'
    _git(tmp_path, 'clone', '-q', str(root), str(up))
    (up / 'src' / 'zalmoxis' / 'a.py').write_text(_MOD.replace('Class doc.', 'Other doc.'))
    _git(up, 'commit', '-qam', 'third')
    commit = _git(up, 'rev-parse', 'HEAD')
    _git(root, 'remote', 'add', 'origin', str(up))
    tool.dump_compact({**json.loads(path.read_text()), 'commit': commit}, path)
    tool.restamp_source(path, root, pkg)
    assert json.loads(path.read_text())['source']['sha256'] == tool.source_hash(['a.py'], pkg)


def test_the_source_hash_reads_the_encoding_a_module_declares(tmp_path):
    """The hash parses the bytes of each module, so a module in its declared encoding hashes
    like the same code in UTF-8, whatever the locale."""
    for folder, data in (
        ('l1', "# -*- coding: latin-1 -*-\nX = '\u00e9'\n".encode('latin-1')),
        ('u8', "X = '\u00e9'\n".encode()),
    ):
        (tmp_path / folder).mkdir()
        (tmp_path / folder / 'a.py').write_bytes(data)
    tool = _tool()
    assert tool.source_hash(['a.py'], tmp_path / 'l1') == tool.source_hash(
        ['a.py'], tmp_path / 'u8'
    )


def test_the_source_only_command_restamps_the_file_it_names(tmp_path, monkeypatch, capsys):
    """main --source-only --out restamps that file without a solve and prints the commit and
    the source hash."""
    tool = _tool()
    calls = []
    monkeypatch.setattr(tool, 'write', lambda path: pytest.fail('the command solved'))
    monkeypatch.setattr(
        tool,
        'restamp_source',
        lambda path: calls.append(path) or {'commit': 'c', 'source': {'sha256': 's'}},
    )
    tool.main(['--source-only', '--out', str(tmp_path / 'r.json')])
    assert calls == [str(tmp_path / 'r.json')]
    assert capsys.readouterr().out.split() == ['c', 's']


@pytest.mark.parametrize(
    'name, pattern, repl, kept',
    [
        ('eos_vinet.py', '"""', '"""Edited. ', True),
        ('constants.py', r'\Z', '# an added comment\n', True),
        ('constants.py', r'(?m)^(\w+ = )(?=\d)', r'\g<1>2 * ', False),
    ],
    ids=['module-docstring', 'comment', 'constant'],
)
def test_the_hash_of_the_stored_modules_follows_their_code(tmp_path, name, pattern, repl, kept):
    """On a copy of the modules the stored reference lists, a docstring or comment edit of one
    keeps the hash and doubling its first numeric constant changes it; the live package and
    the stored hash are the slow test's to compare, so a code edit in a pull request does not
    fail this one."""
    import zalmoxis

    tool = _tool()
    shutil.copytree(os.path.dirname(zalmoxis.__file__), tmp_path, dirs_exist_ok=True)
    stored = json.loads(tool.REFERENCE.read_text())['source']['files']
    files = [f for f in stored if (tmp_path / f).is_file()]
    h = tool.source_hash(files, tmp_path)
    text, n = re.subn(pattern, repl, (tmp_path / name).read_text(), count=1)
    assert name in files and n == 1
    (tmp_path / name).write_text(text)
    assert (tool.source_hash(files, tmp_path) == h) is kept
