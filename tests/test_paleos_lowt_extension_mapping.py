"""Tests for column-to-method mapping in PALEOS low-temperature extension converter."""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_eos_lowt_extension import _paleos, _props
from tools.converters import paleos_lowt_extension as tool
from zalmoxis import eos_export

pytestmark = pytest.mark.unit

_INDEPENDENT_METHODS = {
    'density': 0,
    'specific_internal_energy': 1,
    'specific_entropy': 2,
    'isobaric_heat_capacity': 3,
    'isochoric_heat_capacity': 4,
    'thermal_expansion': 5,
    'adiabatic_gradient': 6,
}


class _IndependentStubEoS:
    """Stub whose method dispatch is anchored to an independent literal mapping."""

    def __getattr__(self, name: str):
        if name in _INDEPENDENT_METHODS:
            idx = _INDEPENDENT_METHODS[name]
            return lambda P, T: _props(P, T)[idx]
        raise AttributeError(name)

    def phase(self, P, T):
        return 'solid-lpcen'


def test_methods_tuple_matches_exact_paleos_methods():
    """tool._METHODS lists the PALEOS methods in the column order of the independent literal."""
    assert tool._METHODS == tuple(sorted(_INDEPENDENT_METHODS, key=_INDEPENDENT_METHODS.get))


def test_written_extension_columns_match_independent_mapping(tmp_path):
    """Verify extension table columns match independent mapping via seam check."""
    table = _paleos(tmp_path, [6, 6, 6, 6])
    out = tmp_path / 'ext.dat'
    tool.write_extension(_IndependentStubEoS(), table, out, ['test header'], t_min=150.4)
    try:
        merged = eos_export.load_paleos_with_lowt_extension(table, out)
    except ValueError as exc:
        pytest.fail(f'Seam mismatch under independent mapping stub: {exc}')
    assert np.isfinite(merged['s']).all()


def test_paleos_mgsio3_eos_has_all_method_names():
    """Verify paleos.mgsio3_eos.MgSiO3EoS implements all required methods when available."""
    eos_cls = pytest.importorskip('paleos.mgsio3_eos').MgSiO3EoS
    for name in tool._METHODS:
        assert hasattr(eos_cls, name), f'MgSiO3EoS is missing method {name}'
    assert hasattr(eos_cls, 'phase'), 'MgSiO3EoS is missing phase method'
