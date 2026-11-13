"""Tests for ``tools.validation.grid_start_up_reference`` on written files, no solves."""

from __future__ import annotations

import hashlib
import json
import os

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
        'first': block,
        'resolve': block,
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
    """With matching md5s the reader returns both structures as arrays, cmb_mass a float."""
    path, dicts = _write(tmp_path)
    ref = _tool().load(path, dicts)
    assert set(ref) >= {'first', 'resolve'}
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


def test_the_source_hash_follows_the_bytes_and_not_the_order(tmp_path):
    """The hash changes with one byte of one file and stays with the order of the names."""
    for name, text in (('a.py', 'x = 1\n'), ('b.py', 'y = 2\n')):
        (tmp_path / name).write_text(text)
    h = _tool().source_hash(['a.py', 'b.py'], tmp_path)
    assert h == _tool().source_hash(['b.py', 'a.py'], tmp_path)
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


def test_the_writer_stores_both_structures_with_tables_and_source(tmp_path, monkeypatch):
    """write solves both cases at REF_TOL and stores the structures, the table md5s, the
    commit and the hash of the loaded modules; the reader reads the file back."""
    tool = _tool()
    seen = []
    solve = _fake_solve()
    monkeypatch.setattr(tool, 'solve', lambda cfg: seen.append(cfg) or solve(cfg))
    dicts = _fake_tables(tmp_path, monkeypatch)
    ref = tool.write(tmp_path / 'ref.json')
    assert [c['tolerance_outer'] for c in seen] == [1e-6, 1e-6]
    assert [c['planet_mass'] for c in seen] == [4.1804e24, 4.177602725330534e24]
    assert (
        ref['tables']['PALEOS:iron']['md5'] == hashlib.md5(b'PALEOS:iron table\n').hexdigest()
    )
    assert len(ref['commit']) == 40 and 'config.py' in ref['source']['files']
    pkg, files = tool.loaded_sources()
    assert ref['source']['sha256'] == tool.source_hash(ref['source']['files'], pkg)
    back = tool.load(tmp_path / 'ref.json', dicts)
    np.testing.assert_array_equal(
        back['resolve']['mass_enclosed'], np.array([0.0, 0.5, 1.0]) * 4.177602725330534e24
    )
    assert back['first']['cmb_mass'] == 0.95 * 4.1804e24


def test_the_writer_refuses_a_reference_that_is_not_strict(tmp_path, monkeypatch):
    """A reference solve accepted on any other label stops the writer and writes no file."""
    tool = _tool()
    monkeypatch.setattr(tool, 'solve', _fake_solve('mass_only'))
    _fake_tables(tmp_path, monkeypatch)
    with pytest.raises(AssertionError, match='first'):
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


def test_the_source_list_leaves_out_the_generated_version_file():
    """_version.py is written at install time with the commit in it and git ignores it, so it
    is not part of the source the hash stands for."""
    import zalmoxis._version  # noqa: F401

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
