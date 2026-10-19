"""Tests for ``tools.converters.paleos_lowt_extension``, with a stub in place of PALEOS."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_eos_lowt_extension import _DLOGT, _P, _T, _paleos, _props, _write
from tools.converters import paleos_lowt_extension as tool
from zalmoxis import eos_export

pytestmark = pytest.mark.unit


class _StubEoS:
    """The synthetic properties of the merge tests; NaN at one point if asked."""

    def __init__(self, nan_at=None):
        self.nan_at = nan_at

    def _get(self, i, P, T):
        if (
            self.nan_at is not None
            and np.isclose(P, self.nan_at[0])
            and np.isclose(T, self.nan_at[1])
        ):
            return np.nan
        return _props(P, T)[i]

    def __getattr__(self, name):
        if name in tool._METHODS:
            return lambda P, T: self._get(tool._METHODS.index(name), P, T)
        raise AttributeError(name)

    def phase(self, P, T):
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
    n = tool.write_extension(_StubEoS(), table, out, ['test header'], t_min=150.4)
    assert n == _P.size * 7
    assert out.read_text().startswith('# test header\n')
    merged = eos_export.load_paleos_with_lowt_extension(table, out)
    assert np.isfinite(merged['s']).all()
    np.testing.assert_allclose(10.0 ** merged['unique_log_t'], _T, rtol=1e-8)


def test_point_with_a_non_finite_property_is_left_out(tmp_path):
    table = _paleos(tmp_path, [6, 6, 6, 6])
    out = tmp_path / 'ext.dat'
    n = tool.write_extension(_StubEoS(nan_at=(_P[1], _T[2])), table, out, [], t_min=150.4)
    assert n == _P.size * 7 - 1
    assert 'nan' not in out.read_text()


def test_command_line_needs_a_commit_outside_a_git_checkout(tmp_path):
    src = tmp_path / 'paleos_src'
    src.mkdir()
    with pytest.raises(SystemExit):
        tool.main(['--paleos-src', str(src), '--table', 'x', '--out', str(tmp_path / 'o')])


def _git(path, *args):
    import subprocess

    return subprocess.run(
        ['git', '-C', str(path), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _git_repo(path, commit=True):
    """A git repository at ``path``, with one commit unless ``commit`` is False; returns HEAD."""
    path.mkdir(parents=True, exist_ok=True)
    _git(path, 'init', '-q')
    (path / 'README').write_text('paleos\n')
    if not commit:
        return ''
    _git(path, 'add', 'README')
    _git(path, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'init')
    return _git(path, 'rev-parse', 'HEAD')


def _stub_paleos(src):
    """A paleos package under ``src`` whose MgSiO3EoS returns the synthetic properties."""
    (src / 'paleos').mkdir()
    (src / 'paleos' / '__init__.py').write_text('')
    stub = ['from tests.test_eos_lowt_extension import _props', '', 'class MgSiO3EoS:']
    for i, m in enumerate(tool._METHODS):
        stub.append(f'    def {m}(self, P, T):\n        return _props(P, T)[{i}]')
    stub.append("    def phase(self, P, T):\n        return 'solid-lpcen'")
    (src / 'paleos' / 'mgsio3_eos.py').write_text('\n'.join(stub) + '\n')


@pytest.fixture
def clean_import(monkeypatch):
    """main() inserts --paleos-src into sys.path and imports paleos: restore both afterwards."""
    import sys

    monkeypatch.setattr(sys, 'path', list(sys.path))
    for name in ('paleos', 'paleos.mgsio3_eos'):
        monkeypatch.delitem(sys.modules, name, raising=False)
    return monkeypatch


def _main(src, table, out, *extra):
    tool.main(['--paleos-src', str(src), '--table', str(table), '--out', str(out), *extra])


def test_command_line_does_not_take_the_commit_of_an_enclosing_repository(tmp_path, capsys):
    """A source folder inside another checkout needs --paleos-commit; the outer HEAD is not taken."""
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
    with pytest.raises(SystemExit):
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), tmp_path / 'o')
    assert 'has no commit' in capsys.readouterr().err
    assert not (tmp_path / 'o').exists()


@pytest.mark.parametrize('commit', ['', 'abc', 'abcdef1\n# T 1', 'g000000', 'abcdef1 x'])
def test_command_line_refuses_a_commit_that_is_not_a_hash(tmp_path, capsys, commit):
    """--paleos-commit must be 7 to 40 hex digits, so it cannot break the header."""
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
    with pytest.raises(SystemExit):
        _main(src, _paleos(tmp_path, [6, 6, 6, 6]), tmp_path / 'o')
    assert 'paleos was imported from' in capsys.readouterr().err
    assert not (tmp_path / 'o').exists()


def test_command_line_writes_the_commit_and_the_dirty_state(tmp_path, clean_import):
    """Uncommitted changes in the PALEOS checkout are marked next to its HEAD."""
    src = tmp_path / 'paleos_src'
    sha = _git_repo(src)
    _stub_paleos(src)
    table, out = _paleos(tmp_path, [6, 6, 6, 6]), tmp_path / 'ext.dat'
    _main(src, table, out, '--t-min', '150.4')
    assert f'# PALEOS commit: {sha} (uncommitted changes)' in out.read_text().splitlines()[:5]
    assert np.isfinite(eos_export.load_paleos_with_lowt_extension(table, out)['s']).all()


def test_command_line_writes_a_clean_commit(tmp_path, clean_import):
    """A clean PALEOS checkout gives its HEAD alone."""
    src = tmp_path / 'paleos_src'
    _git_repo(src)
    _stub_paleos(src)
    _git(src, 'add', 'paleos')
    _git(src, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-q', '-m', 'stub')
    sha = _git(src, 'rev-parse', 'HEAD')
    out = tmp_path / 'ext.dat'
    _main(src, _paleos(tmp_path, [6, 6, 6, 6]), out, '--t-min', '150.4')
    head = out.read_text().splitlines()[:5]
    assert f'# PALEOS commit: {sha}' in head
    assert not any('uncommitted' in line for line in head)


def test_command_line_takes_a_given_commit_outside_a_checkout(tmp_path, clean_import):
    """--paleos-commit names the source of a PALEOS tree that is not a git checkout."""
    src = tmp_path / 'paleos_src'
    src.mkdir()
    _stub_paleos(src)
    out = tmp_path / 'ext.dat'
    _main(
        src,
        _paleos(tmp_path, [6, 6, 6, 6]),
        out,
        '--t-min',
        '150.4',
        '--paleos-commit',
        '654aca9',
    )
    assert '# PALEOS commit: 654aca9' in out.read_text().splitlines()[:5]
    assert out.read_text().count('solid-lpcen') == len(_P) * 7


@pytest.mark.parametrize('t_min', [0.0, -10.0, 300.0, 350.0, float('nan')])
def test_grid_refuses_a_lowest_temperature_outside_the_extension_range(tmp_path, t_min):
    """t_min must lie strictly between 0 K and the lowest temperature of the table."""
    table = _paleos(tmp_path, [6, 6, 6, 6])
    with pytest.raises(ValueError, match='must lie between 0 K and the table minimum'):
        tool.extension_grid(table, t_min=t_min)
    assert tool.extension_grid(table, t_min=299.0)[1].size == 2


def test_grid_refuses_a_table_with_one_temperature(tmp_path):
    """A table with one temperature has no log-T step to continue."""
    one = _write(tmp_path / 'one.dat', [(P, 300.0) for P in _P])
    with pytest.raises(ValueError, match='needs at least two temperatures'):
        tool.extension_grid(one, t_min=100.0)
    assert one.read_text().count('300') >= len(_P)
