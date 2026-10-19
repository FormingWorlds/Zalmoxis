"""Tests for ``tools.converters.paleos_lowt_extension``, with a stub in place of PALEOS."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_eos_lowt_extension import _DLOGT, _P, _T, _paleos, _props
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
