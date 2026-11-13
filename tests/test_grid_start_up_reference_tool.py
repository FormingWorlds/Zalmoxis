"""Tests for ``tools.validation.grid_start_up_reference`` on written files, no solves."""

from __future__ import annotations

import hashlib
import json

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
