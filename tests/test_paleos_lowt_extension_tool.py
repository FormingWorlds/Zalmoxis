"""Tests for ``tools.converters.paleos_lowt_extension``, with a stub in place of PALEOS."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import numpy as np
import pytest

from tests.test_eos_lowt_extension import _DLOGT, _P, _T, _paleos, _props, _write
from tools.converters import paleos_lowt_extension as tool
from zalmoxis import eos_export

pytestmark = pytest.mark.unit


class _StubEoS:
    """The synthetic properties of the merge tests; NaN at one point, or an exception raised
    by one method at one point, if asked."""

    def __init__(self, nan_at=None, raise_at=None, method='density', exc=RuntimeError):
        self.nan_at, self.raise_at, self.method, self.exc = nan_at, raise_at, method, exc

    def _hit(self, at, P, T):
        return at is not None and np.isclose(P, at[0]) and np.isclose(T, at[1])

    def _get(self, i, P, T):
        if self._hit(self.raise_at, P, T) and tool._METHODS[i] == self.method:
            raise self.exc('no convergence')
        if self._hit(self.nan_at, P, T):
            return np.nan
        return _props(P, T)[i]

    def __getattr__(self, name):
        if name in tool._METHODS:
            return lambda P, T: self._get(tool._METHODS.index(name), P, T)
        raise AttributeError(name)

    def phase(self, P, T):
        if self._hit(self.raise_at, P, T) and self.method == 'phase':
            raise self.exc('no convergence')
        return 'solid-lpcen'


def test_grid_continues_the_table_step_down_to_t_min(tmp_path):
    table = _paleos(tmp_path, [6, 6, 6, 6])
    P, T = tool.extension_grid(table, t_min=180.0, p_max=5.0e9)
    np.testing.assert_allclose(P, _P[_P < 5.0e9], rtol=1e-8)
    assert T[0] == 300.0
    np.testing.assert_allclose(np.diff(np.log10(T)), -_DLOGT, rtol=1e-6)
    assert T[-1] <= 180.0 < T[-2]


def test_grid_drops_the_zero_pressure_row(tmp_path):
    table = _paleos(tmp_path, [6, 6, 6, 6])
    table.write_text(table.read_text() + f'0.00000000e+00 {300.0:.8e} ' + '1 ' * 7 + 'solid\n')
    P, _ = tool.extension_grid(table, t_min=200.0)
    assert P.min() > 0.0


def test_written_extension_merges_under_the_table(tmp_path):
    """The tool output passes the seam check and fills every cell below 300 K."""
    table = _paleos(tmp_path, [6, 6, 6, 6])
    out = tmp_path / 'ext.dat'
    n, _ = tool.write_extension(_StubEoS(), table, out, ['test header'], t_min=150.4)
    assert n == _P.size * 7
    assert out.read_text().startswith('# test header\n')
    merged = eos_export.load_paleos_with_lowt_extension(table, out)
    assert np.isfinite(merged['s']).all()
    np.testing.assert_allclose(10.0 ** merged['unique_log_t'], _T, rtol=1e-8)


def test_point_with_a_non_finite_property_is_left_out(tmp_path):
    table = _paleos(tmp_path, [6, 6, 6, 6])
    out = tmp_path / 'ext.dat'
    n, _ = tool.write_extension(_StubEoS(nan_at=(_P[1], _T[2])), table, out, [], t_min=150.4)
    assert n == _P.size * 7 - 1
    assert 'nan' not in out.read_text()


@pytest.mark.parametrize('method', ['density', 'specific_entropy', 'phase'])
@pytest.mark.parametrize('exc', [RuntimeError, ArithmeticError, ValueError])
def test_point_where_paleos_raises_is_left_out(tmp_path, method, exc):
    """A PALEOS point that does not converge (an exception from a property or the phase) is
    left out, as a non-finite one is, and the other points are written."""
    table = _paleos(tmp_path, [6, 6, 6, 6])
    out = tmp_path / 'ext.dat'
    eos = _StubEoS(raise_at=(_P[1], _T[2]), method=method, exc=exc)
    try:
        n, _ = tool.write_extension(eos, table, out, [], t_min=150.4)
    except Exception as err:  # the point must be left out, not stop the converter
        pytest.fail(f'a non-converging point stopped the converter: {err!r}')
    assert n == _P.size * 7 - 1
    data = [x for x in out.read_text().splitlines() if not x.startswith('#')]
    assert not any(x.startswith(f'{_P[1]:.8e} {_T[2]:.8e} ') for x in data)


def test_left_out_points_are_returned_and_named_in_the_header(tmp_path):
    """write_extension returns the left-out (P, T) points; the header gives their number and
    each point, and says 0 points when none is left out."""
    table = _paleos(tmp_path, [6, 6, 6, 6])
    out, full = tmp_path / 'ext.dat', tmp_path / 'full.dat'
    n, left = tool.write_extension(
        _StubEoS(raise_at=(_P[1], _T[2])), table, out, ['h'], t_min=150.4
    )
    assert (n, len(left)) == (_P.size * 7 - 1, 1)
    np.testing.assert_allclose(left[0], (_P[1], _T[2]), rtol=1e-8)
    head = [x for x in out.read_text().splitlines() if x.startswith('#')]
    assert '# Left out: 1 point (no convergence or not finite)' in head
    assert f'# Left out at P T: {left[0][0]:.8e} {left[0][1]:.8e}' in head
    assert tool.write_extension(_StubEoS(), table, full, [], t_min=150.4)[1] == []
    assert '# Left out: 0 points (no convergence or not finite)' in full.read_text()


def test_a_hole_inside_a_row_stops_the_merge_and_one_at_the_lowest_node_does_not(tmp_path):
    """The merge refuses a left-out point between two written ones and accepts one at the
    lowest node of a row, where the row then starts one node higher."""
    table = _paleos(tmp_path, [6, 6, 6, 6])
    mid, low = tmp_path / 'mid.dat', tmp_path / 'low.dat'
    tool.write_extension(_StubEoS(raise_at=(_P[1], _T[2])), table, mid, [], t_min=150.4)
    with pytest.raises(ValueError, match='a missing or non-finite extension cell at P = '):
        eos_export.load_paleos_with_lowt_extension(table, mid)
    tool.write_extension(_StubEoS(raise_at=(_P[1], _T[0])), table, low, [], t_min=150.4)
    s = eos_export.load_paleos_with_lowt_extension(table, low)['s']
    assert np.isnan(s[1, 0]) and np.isfinite(s[1, 1:]).all()
    assert np.isfinite(np.delete(s, 1, axis=0)).all()


def test_a_programming_error_in_paleos_is_not_hidden(tmp_path):
    table = _paleos(tmp_path, [6, 6, 6, 6])
    eos = _StubEoS(raise_at=(_P[1], _T[2]), exc=TypeError)
    with pytest.raises(TypeError, match='no convergence'):
        tool.write_extension(eos, table, tmp_path / 'ext.dat', [], t_min=150.4)


def test_command_line_needs_a_commit_outside_a_git_checkout(tmp_path, capsys):
    """A plain folder as --paleos-src needs --paleos-commit."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    table = _paleos(tmp_path, [6, 6, 6, 6])
    with pytest.raises(SystemExit):
        tool.main(
            ['--paleos-src', str(src), '--table', str(table), '--out', str(tmp_path / 'o')]
        )
    err = capsys.readouterr().err
    assert 'is not the top of a git checkout' in err
    assert 'give --paleos-commit with --allow-unverified-paleos' in err
    assert not (tmp_path / 'o').exists()


def _git(path, *args):
    import subprocess

    return subprocess.run(
        ['git', '-C', str(path), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_repo(path, commit=True):
    """A git repository at ``path``, with one commit unless ``commit`` is False; returns
    HEAD."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, 'init', '-q')
    (path / 'README').write_text('paleos\n')
    if not commit:
        return ''
    _git(path, 'add', 'README')
    _git(path, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'init')
    return _git(path, 'rev-parse', 'HEAD')


def _stub_paleos(src, ds=0.0, raise_at=None):
    """A paleos package under ``src`` whose MgSiO3EoS returns the synthetic properties, with
    ``ds`` added to s; its density raises RuntimeError at the (P, T) ``raise_at``."""
    (src / 'paleos').mkdir()
    (src / 'paleos' / '__init__.py').write_text('')
    stub = ['from tests.test_eos_lowt_extension import _props', '', 'class MgSiO3EoS:']
    for i, m in enumerate(tool._METHODS):
        add = f' + {ds!r}' if m == 'specific_entropy' else ''
        stub.append(f'    def {m}(self, P, T):')
        if raise_at is not None and m == 'density':
            P0, T0 = (float(x) for x in raise_at)
            stub.append(
                f'        if abs(P / {P0!r} - 1) < 1e-8 and abs(T / {T0!r} - 1) < 1e-8:'
            )
            stub.append("            raise RuntimeError('no convergence')")
        stub.append(f'        return _props(P, T)[{i}]{add}')
    stub.append("    def phase(self, P, T):\n        return 'solid-lpcen'")
    (src / 'paleos' / 'mgsio3_eos.py').write_text('\n'.join(stub) + '\n')


@pytest.fixture
def clean_import(monkeypatch):
    """main() inserts --paleos-src into sys.path and imports paleos: restore both afterwards,
    including the removal of paleos modules that the test added."""
    import sys

    before = {k for k in sys.modules if k.split('.')[0] == 'paleos'}
    monkeypatch.setattr(sys, 'path', list(sys.path))
    for name in ('paleos', 'paleos.mgsio3_eos'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield monkeypatch
    for k in [k for k in sys.modules if k.split('.')[0] == 'paleos' and k not in before]:
        del sys.modules[k]


_GIVEN = ('--paleos-commit', '654aca9', '--allow-unverified-paleos')


def _files(folder):
    """Names in ``folder`` except the PALEOS source, hidden files included."""
    return sorted(p.name for p in folder.iterdir() if p.name != 'paleos_src')


def _main(src, table, out, *extra):
    tool.main(['--paleos-src', str(src), '--table', str(table), '--out', str(out), *extra])


def test_command_line_does_not_take_the_commit_of_an_enclosing_repository(tmp_path, capsys):
    """A source folder inside another checkout needs --paleos-commit; the outer HEAD is not
    taken."""
    _git_repo(tmp_path / 'outer')
    src = tmp_path / 'outer' / 'paleos_src'
    src.mkdir()
    with pytest.raises(SystemExit):
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), tmp_path / 'o')
    assert 'is not the top of a git checkout' in capsys.readouterr().err
    assert not (tmp_path / 'o').exists()


def test_command_line_refuses_a_checkout_without_a_commit(tmp_path, capsys):
    """A checkout with no commit gives no commit to record."""
    src = tmp_path / 'paleos_src'
    _git_repo(src, commit=False)
    with pytest.raises(BaseException) as exc:
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), tmp_path / 'o')
    assert exc.type is SystemExit, repr(exc.value)
    assert 'has no commit' in capsys.readouterr().err
    assert not (tmp_path / 'o').exists()


@pytest.mark.parametrize(
    'commit', ['', 'abc', 'abcdef', 'a' * 65, 'abcdef1\n# T 1', 'g000000', 'abcdef1 x']
)
def test_command_line_refuses_a_commit_that_is_not_a_hash(tmp_path, capsys, commit):
    """--paleos-commit must be 7 to 64 hex digits, so it cannot break the header."""
    with pytest.raises(SystemExit):
        _main(
            tmp_path, _paleos(tmp_path, [6, 6, 6, 6]), tmp_path / 'o', '--paleos-commit', commit
        )
    assert 'is not a commit hash' in capsys.readouterr().err
    assert not (tmp_path / 'o').exists()


def test_command_line_refuses_a_paleos_imported_from_elsewhere(tmp_path, capsys, clean_import):
    """The PALEOS module that is used must come from --paleos-src, also when it has a file."""
    import sys
    import types

    src = tmp_path / 'paleos_src'
    _git_repo(src)
    other = types.ModuleType('paleos.mgsio3_eos')
    other.__file__ = str(tmp_path / 'elsewhere' / 'paleos' / 'mgsio3_eos.py')
    other.MgSiO3EoS = object
    clean_import.setitem(sys.modules, 'paleos', types.ModuleType('paleos'))
    clean_import.setitem(sys.modules, 'paleos.mgsio3_eos', other)
    with pytest.raises(BaseException) as exc:
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), tmp_path / 'o')
    assert exc.type is SystemExit, repr(exc.value)
    assert 'paleos was imported from' in capsys.readouterr().err
    assert not (tmp_path / 'o').exists()


@pytest.mark.parametrize('dirty', [True, False])
def test_command_line_writes_the_commit_and_the_dirty_state(tmp_path, clean_import, dirty):
    """The PALEOS HEAD goes in the header, marked when the checkout has uncommitted changes
    (accepted only with --allow-unverified-paleos)."""
    src = tmp_path / 'paleos_src'
    _git_repo(src)
    _stub_paleos(src)
    if not dirty:
        _git(src, 'add', 'paleos')
        _git(src, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'stub')
    table, out = _paleos(tmp_path, [6, 6, 6, 6]), tmp_path / 'ext.dat'
    _main(
        src, table, out, '--t-min', '150.4', *(['--allow-unverified-paleos'] if dirty else [])
    )
    expect = (
        '# PALEOS commit: ' + _git(src, 'rev-parse', 'HEAD') + dirty * ' (uncommitted changes)'
    )
    assert expect in out.read_text().splitlines()[:9]
    assert np.isfinite(eos_export.load_paleos_with_lowt_extension(table, out)['s']).all()


@pytest.mark.parametrize(
    ('given', 'written'), [('654aca9', '654aca9'), ('ABCDEF1', 'abcdef1'), ('a' * 64, 'a' * 64)]
)
def test_command_line_takes_a_given_commit_outside_a_checkout(
    tmp_path, clean_import, given, written
):
    """--paleos-commit names the source of a PALEOS tree that is not a git checkout; hex
    digits of either case, 7 to 64 of them, are written in lower case."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)
    table, out = _paleos(tmp_path, [6, 6, 6, 6]), tmp_path / 'ext.dat'
    _main(
        src,
        table,
        out,
        '--t-min',
        '150.4',
        '--paleos-commit',
        given,
        '--allow-unverified-paleos',
    )
    assert (
        f'# PALEOS commit: {written} (given, not verified)' in out.read_text().splitlines()[:9]
    )
    assert out.read_text().count('solid-lpcen') == len(_P) * 7


def test_command_line_refuses_a_failed_git_status(tmp_path, capsys, clean_import, monkeypatch):
    """A git status that fails leaves the dirty state unknown, so nothing is written."""
    src = tmp_path / 'paleos_src'
    _git_repo(src)
    _stub_paleos(src)
    real = tool._git
    monkeypatch.setattr(
        tool, '_git', lambda path, *a: None if a[0] == 'status' else real(path, *a)
    )
    with pytest.raises(SystemExit):
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), tmp_path / 'o', '--t-min', '150.4')
    assert 'git status failed' in capsys.readouterr().err
    assert not (tmp_path / 'o').exists()


def test_command_line_without_git_needs_a_commit(tmp_path, capsys, clean_import, monkeypatch):
    """Without a git program the source counts as no checkout."""
    src = tmp_path / 'paleos_src'
    _git_repo(src)

    def no_git(*a, **k):
        raise FileNotFoundError('git')

    monkeypatch.setattr(tool.subprocess, 'run', no_git)
    with pytest.raises((SystemExit, FileNotFoundError)):
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), tmp_path / 'o')
    assert 'is not the top of a git checkout' in capsys.readouterr().err
    assert not (tmp_path / 'o').exists()


def test_command_line_refuses_an_extension_with_no_rows(tmp_path, capsys, clean_import):
    """A p_max below every table row gives no extension rows, which is an error, not a file."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)
    with pytest.raises(SystemExit):
        _main(
            src,
            _paleos(tmp_path, [6, 6, 6, 6]),
            tmp_path / 'o',
            '--t-min',
            '150.4',
            *_GIVEN,
            '--p-max',
            '1e5',
        )
    assert 'no extension rows' in capsys.readouterr().err
    assert not (tmp_path / 'o').exists()


def test_command_line_refuses_a_missing_output_folder_first(tmp_path, capsys, clean_import):
    """The output folder is checked before PALEOS is evaluated."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)
    out = tmp_path / 'absent' / 'ext.dat'
    with pytest.raises(BaseException) as exc:
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), out, *_GIVEN)
    assert exc.type is SystemExit, repr(exc.value)
    assert 'output folder' in capsys.readouterr().err
    assert not out.parent.exists()


def test_command_line_refuses_an_extension_that_does_not_merge(tmp_path, capsys, clean_import):
    """The writer loads its result under the table; a seam mismatch stops the tool and leaves
    no output."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src, ds=1.0)
    out = tmp_path / 'ext.dat'
    with pytest.raises(SystemExit):
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), out, '--t-min', '150.4', *_GIVEN)
    assert 'does not merge under the table' in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize('t_min', [0.0, -10.0, 300.0, 350.0, float('nan')])
def test_grid_refuses_a_lowest_temperature_outside_the_extension_range(tmp_path, t_min):
    """t_min must lie strictly between 0 K and the lowest temperature of the table."""
    table = _paleos(tmp_path, [6, 6, 6, 6])
    with pytest.raises(ValueError, match='must lie between 0 K and the table minimum'):
        tool.extension_grid(table, t_min=t_min)
    assert tool.extension_grid(table, t_min=299.0)[1].size == 2


@pytest.mark.parametrize('cells', [[(1.0e6, 300.0)], [(P, 300.0) for P in _P]])
def test_grid_refuses_a_table_with_one_temperature(tmp_path, cells):
    """A table with one temperature, also one with a single data row, has no log-T step."""
    one = _write(tmp_path / 'one.dat', cells)
    with pytest.raises((ValueError, IndexError), match='needs at least two temperatures'):
        tool.extension_grid(one, t_min=100.0)
    assert tool.extension_grid(_paleos(tmp_path, [6, 6, 6, 6]), t_min=299.0)[1].size == 2


def test_command_line_header_holds_the_input_hashes_and_versions(tmp_path, clean_import):
    """The header names sha256 of the data lines of the extended table, of the PALEOS
    mgsio3_eos.py and of the converter, the numpy, scipy, sympy and Python versions, HEAD of
    the Zalmoxis checkout of the converter and the version of the Zalmoxis package."""
    import hashlib
    import platform
    import re

    import scipy
    import sympy

    import zalmoxis

    conv = Path(tool.__file__).resolve()
    root = conv.parents[2]
    top = tool._git(root, 'rev-parse', '--show-toplevel')
    if top is None or Path(top).resolve() != root:
        pytest.skip('the converter is not in a git checkout of Zalmoxis')
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)
    table, out = _paleos(tmp_path, [6, 6, 6, 6]), tmp_path / 'ext.dat'
    _main(src, table, out, '--t-min', '150.4', *_GIVEN)
    data = ''.join(x for x in table.read_text().splitlines(True) if not x.startswith('#'))
    head = [x for x in out.read_text().splitlines() if x.startswith('#')]
    assert f'# Table data sha256: {hashlib.sha256(data.encode()).hexdigest()}' in head
    eos_py = (src / 'paleos' / 'mgsio3_eos.py').read_bytes()
    assert f'# paleos/mgsio3_eos.py sha256: {hashlib.sha256(eos_py).hexdigest()}' in head
    versions = f'numpy {np.__version__}, scipy {scipy.__version__}, sympy {sympy.__version__}'
    assert f'# {versions}, Python {platform.python_version()}' in head
    assert f'# Converter sha256: {hashlib.sha256(conv.read_bytes()).hexdigest()}' in head
    commit = tool._git(root, 'rev-parse', 'HEAD')
    dirty = tool._git(root, 'status', '--porcelain')
    want = (
        f'# Zalmoxis commit of the converter: {commit}' + bool(dirty) * ' (uncommitted changes)'
    )
    assert re.fullmatch(r'[0-9a-f]{40}([0-9a-f]{24})?', commit) and want in head
    pkg = f'# Zalmoxis package of the merge check: {zalmoxis.__version__}, from that checkout'
    assert pkg in head


def _repo(path):
    """A git repository at ``path`` with one commit; returns its HEAD."""
    _git_repo(path)
    return _git(path, 'rev-parse', 'HEAD')


def test_zalmoxis_commit_is_head_of_the_checkout_whose_top_is_the_root(tmp_path):
    """HEAD of a clean checkout; marked for a tracked change and for an untracked file."""
    head = _repo(tmp_path / 'zal')
    assert tool._zalmoxis_commit(tmp_path / 'zal') == head
    (tmp_path / 'zal' / 'new.py').write_text('x = 1\n')
    assert tool._zalmoxis_commit(tmp_path / 'zal') == head + ' (uncommitted changes)'
    (tmp_path / 'zal' / 'new.py').unlink()
    (tmp_path / 'zal' / 'README').write_text('changed\n')
    assert tool._zalmoxis_commit(tmp_path / 'zal') == head + ' (uncommitted changes)'


def test_zalmoxis_commit_is_unknown_outside_its_own_checkout(tmp_path, monkeypatch):
    """A folder inside another repository, a folder in no repository, and a failed git status
    give 'unknown'."""
    _repo(tmp_path / 'outer')
    (tmp_path / 'outer' / 'zal').mkdir()
    (tmp_path / 'plain').mkdir()
    unknown = 'unknown (not a git checkout)'
    assert tool._zalmoxis_commit(tmp_path / 'outer' / 'zal') == unknown
    assert tool._zalmoxis_commit(tmp_path / 'plain') == unknown
    real = tool._git
    monkeypatch.setattr(tool, '_git', lambda p, *a: None if a[0] == 'status' else real(p, *a))
    assert tool._zalmoxis_commit(tmp_path / 'outer') == unknown


def test_command_line_names_a_left_out_point_that_stops_the_merge(
    tmp_path, capsys, clean_import
):
    """A point left out inside a row makes the merge check refuse the file: the command line
    removes it and exits with an error that names the point."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src, raise_at=(_P[1], _T[2]))
    out = tmp_path / 'ext.dat'
    with pytest.raises(SystemExit) as exc:
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), out, '--t-min', '150.4', *_GIVEN)
    assert exc.value.code not in (0, None)
    err = capsys.readouterr().err
    assert 'does not merge under the table' in err
    assert f'left-out points: P = {_P[1]:.6e} Pa, T = {_T[2]:.6e} K' in err
    assert not out.exists()


def test_command_line_reports_a_left_out_lowest_node_and_keeps_the_file(
    tmp_path, capsys, clean_import
):
    """A point left out at the lowest node of a row merges; the header and the output give
    it."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src, raise_at=(_P[1], _T[0]))
    table, out = _paleos(tmp_path, [6, 6, 6, 6]), tmp_path / 'ext.dat'
    _main(src, table, out, '--t-min', '150.4', *_GIVEN)
    assert '# Left out: 1 point (no convergence or not finite)' in out.read_text().splitlines()
    printed = capsys.readouterr().out
    assert f'left out 1 point: P = {_P[1]:.6e} Pa, T = {_T[0]:.6e} K' in printed
    assert np.isnan(eos_export.load_paleos_with_lowt_extension(table, out)['s'][1, 0])


def test_command_line_refuses_a_stale_pinned_table_and_writes_no_file(
    tmp_path, clean_import, monkeypatch
):
    """A --table under a registry name whose md5 is not the pinned one raises StaleTableError
    before PALEOS is imported, so no output is written."""
    from zalmoxis.eos import registry_guard
    from zalmoxis.eos.registry_guard import StaleTableError

    monkeypatch.setenv('FWL_IO_OFFLINE', '1')
    src = tmp_path / 'paleos_src'
    _git_repo(src)
    _stub_paleos(src)
    (src / 'paleos' / '__init__.py').write_text("raise AssertionError('PALEOS was imported')\n")
    _git(src, 'add', 'paleos')
    _git(src, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'stub')
    table = _paleos(tmp_path, [6, 6, 6, 6], name='paleos_mgsio3_tables_pt_proteus_solid.dat')
    pins = {table.name: {'interior.eos.paleos_mgsio3': '0' * 32}}
    monkeypatch.setattr(registry_guard, '_pins', lambda: pins)
    registry_guard._HASHES.clear()
    out = tmp_path / 'ext.dat'
    with pytest.raises(StaleTableError, match=table.name):
        _main(src, table, out, '--t-min', '150.4')
    assert not out.exists()


def test_command_line_removes_its_temporary_file_on_any_error(
    tmp_path, clean_import, monkeypatch
):
    """An error of any type from the merge check (here an interrupt) leaves no file."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)

    def broken(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(eos_export, 'load_paleos_with_lowt_extension', broken)
    out = tmp_path / 'ext.dat'
    with pytest.raises(KeyboardInterrupt):
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), out, '--t-min', '150.4', *_GIVEN)
    assert _files(tmp_path) == ['paleos.dat']


def test_command_line_refuses_a_dirty_checkout_by_default(tmp_path, capsys, clean_import):
    """Uncommitted changes in --paleos-src stop the tool unless --allow-unverified-paleos."""
    src = tmp_path / 'paleos_src'
    _git_repo(src)
    _stub_paleos(src)
    out = tmp_path / 'ext.dat'
    with pytest.raises(SystemExit) as exc:
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), out, '--t-min', '150.4')
    assert exc.value.code not in (0, None)
    assert 'uncommitted or untracked files' in capsys.readouterr().err
    assert not out.exists()


def test_command_line_refuses_a_given_commit_by_default(tmp_path, capsys, clean_import):
    """A --paleos-commit that cannot be checked stops the tool unless
    --allow-unverified-paleos."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)
    out = tmp_path / 'ext.dat'
    with pytest.raises(SystemExit) as exc:
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), out, '--paleos-commit', '654aca9')
    assert exc.value.code not in (0, None)
    assert '--paleos-commit cannot be checked' in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize('mask', [0o022, 0o002])
def test_a_failed_run_keeps_an_earlier_output_and_leaves_no_temporary_file(
    tmp_path, capsys, clean_import, mask
):
    """A run that fails after the write keeps the output of an earlier run unchanged; the
    written file has the mode the umask gives a new file."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)
    table, out = _paleos(tmp_path, [6, 6, 6, 6]), tmp_path / 'ext.dat'
    old = os.umask(mask)
    try:
        _main(src, table, out, '--t-min', '150.4', *_GIVEN)
    finally:
        os.umask(old)
    before = out.read_text()
    assert stat.S_IMODE(out.stat().st_mode) == 0o666 & ~mask
    with pytest.raises(SystemExit):
        _main(src, table, out, '--t-min', '150.4', '--p-max', '1e5', *_GIVEN)
    assert 'no extension rows' in capsys.readouterr().err
    assert out.read_text() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ['ext.dat', 'paleos.dat', 'paleos_src']


def test_an_error_during_the_write_leaves_no_partial_output(
    tmp_path, clean_import, monkeypatch
):
    """A write that stops part way (a full disk) leaves no output and no temporary file."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)

    def partial(eos, table_file, out_file, *args):
        out_file.write_text('# partial\n')
        raise OSError(28, 'No space left on device')

    monkeypatch.setattr(tool, 'write_extension', partial)
    out = tmp_path / 'ext.dat'
    with pytest.raises(OSError, match='No space left'):
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), out, '--t-min', '150.4', *_GIVEN)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['paleos.dat', 'paleos_src']


@pytest.mark.parametrize('form', ['link', 'dotdot'])
def test_command_line_refuses_an_output_that_is_the_table(tmp_path, capsys, clean_import, form):
    """--out naming the --table file through a link or a '..' path stops the tool before
    anything is written."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)
    table = _paleos(tmp_path, [6, 6, 6, 6])
    (tmp_path / 'sub').mkdir()
    out = tmp_path / 'sub' / '..' / table.name
    if form == 'link':
        out = tmp_path / 'sub' / 'link.dat'
        out.symlink_to(table)
    before = table.read_text()
    with pytest.raises(SystemExit):
        _main(src, table, out, *_GIVEN)
    assert '--out must not be the --table file' in capsys.readouterr().err
    assert table.read_text() == before


def test_command_line_refuses_a_missing_table(tmp_path, capsys, clean_import):
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)
    with pytest.raises(BaseException) as exc:
        _main(src, tmp_path / 'absent.dat', tmp_path / 'ext.dat', *_GIVEN)
    assert exc.type is SystemExit, repr(exc.value)
    assert 'absent.dat is not a file' in capsys.readouterr().err
    assert not (tmp_path / 'ext.dat').exists()


@pytest.mark.parametrize('case', ['link', 'table', 'user'])
def test_a_file_at_the_old_temporary_name_is_never_touched(tmp_path, clean_import, case):
    """A link to the table at <out>.tmp, a table named <out>.tmp, and a file of the user at
    that name keep their content; the tool uses a temporary name of its own."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)
    out, old = tmp_path / 'ext.dat', tmp_path / 'ext.dat.tmp'
    table = _paleos(tmp_path, [6, 6, 6, 6], name=old.name if case == 'table' else 'paleos.dat')
    if case == 'link':
        old.symlink_to(table)
    elif case == 'user':
        old.write_text('user data\n')
    keep = {p.name: p.read_bytes() for p in (table, old)}
    try:
        _main(src, table, out, '--t-min', '150.4', *_GIVEN)
    except Exception as err:  # a file at the old name must not stop the run
        pytest.fail(f'a file at the old temporary name stopped the run: {err!r}')
    assert out.is_file() and not out.is_symlink()
    assert {p.name: p.read_bytes() for p in (table, old)} == keep
    assert old.is_symlink() == (case == 'link')


def test_command_line_refuses_an_output_that_is_a_folder(tmp_path, capsys, clean_import):
    """An --out that is a folder stops the tool before PALEOS is imported."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)
    (src / 'paleos' / '__init__.py').write_text("raise AssertionError('PALEOS was imported')\n")
    (tmp_path / 'ext.dat').mkdir()
    with pytest.raises(SystemExit):
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), tmp_path / 'ext.dat', *_GIVEN)
    assert 'is a folder' in capsys.readouterr().err


def test_an_output_link_is_followed_and_kept(tmp_path, clean_import):
    """An --out that is a link keeps the link; the file it points to gets the extension."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)
    (tmp_path / 'store').mkdir()
    real = tmp_path / 'store' / 'real.dat'
    real.write_text('old\n')
    out = tmp_path / 'ext.dat'
    out.symlink_to(real)
    _main(src, _paleos(tmp_path, [6, 6, 6, 6]), out, '--t-min', '150.4', *_GIVEN)
    assert out.is_symlink() and out.resolve() == real.resolve()
    assert real.read_text().startswith('# PALEOS MgSiO3 solid low-temperature extension')
    assert _files(tmp_path / 'store') == ['real.dat']


def test_a_merge_error_names_the_output_not_the_temporary_file(tmp_path, capsys, clean_import):
    """A seam mismatch names --out in the error, not the temporary file, and leaves no file."""
    import re

    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src, ds=1.0)
    out = tmp_path / 'ext.dat'
    with pytest.raises(SystemExit):
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), out, '--t-min', '150.4', *_GIVEN)
    err = capsys.readouterr().err
    assert 'does not merge under the table' in err and str(out) in err
    assert not re.search(r'\.ext\.dat\.\w+\.tmp', err)
    assert _files(tmp_path) == ['paleos.dat']


@pytest.mark.parametrize('where', ['site', 'venv', 'build'])
def test_command_line_refuses_a_zalmoxis_package_from_another_tree(
    tmp_path, capsys, clean_import, monkeypatch, where
):
    """The merge check must run src/zalmoxis of the converter's checkout, whose commit the
    header names; an installed copy elsewhere or inside the checkout is refused before the
    table check and before PALEOS is imported."""
    import zalmoxis
    from zalmoxis.eos import registry_guard

    root = Path(tool.__file__).resolve().parents[2]
    pkg = {
        'site': tmp_path / 'site',
        'venv': root / '.venv' / 'lib' / 'python3.12' / 'site-packages',
        'build': root / 'build' / 'lib',
    }[where]
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)
    (src / 'paleos' / '__init__.py').write_text("raise AssertionError('PALEOS was imported')\n")

    def no_verify(*args):
        raise AssertionError('the table check ran before the tree check')

    monkeypatch.setattr(registry_guard, 'verify', no_verify)
    monkeypatch.setattr(zalmoxis, '__file__', str(pkg / 'zalmoxis' / '__init__.py'))
    out = tmp_path / 'ext.dat'
    with pytest.raises(SystemExit):
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), out, *_GIVEN)
    assert 'zalmoxis was imported from' in capsys.readouterr().err
    assert not out.exists()


def test_command_line_refuses_an_output_link_into_a_missing_folder(
    tmp_path, capsys, clean_import
):
    """An --out link whose target folder does not exist stops the tool before PALEOS is
    imported."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)
    (src / 'paleos' / '__init__.py').write_text("raise AssertionError('PALEOS was imported')\n")
    out = tmp_path / 'ext.dat'
    out.symlink_to(tmp_path / 'nodir' / 'real.dat')
    with pytest.raises(SystemExit):
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), out, *_GIVEN)
    assert 'nodir does not exist' in capsys.readouterr().err
