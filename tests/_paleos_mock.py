"""In-process mock of the upstream ``paleos`` package for CI test environments.

The CI runner does not vendor the upstream PALEOS Python package, so without
this mock ``tests/test_paleos_api.py`` would skip on every CI push and drag
``src/zalmoxis/eos/paleos_api.py`` coverage on CI well below the >95 % the
PALEOS-installed developer machines see.

This mock injects a minimal ``paleos`` package into ``sys.modules`` whenever
the real package cannot be imported. The injection happens once per process
at conftest collection time, before any test imports ``zalmoxis.eos.paleos_api``.
With the real package available (local dev), this module is a no-op.

Surface mocked:

- ``paleos`` (top-level): ``__file__``, ``__version__``.
- ``paleos.iron_eos.IronEoS``: 7 EoS scalar properties + ``phase`` method.
- ``paleos.mgsio3_eos`` module:
  - ``MgSiO3EoS``: same 7 properties + ``phase``, plus ``_phase_eos_map``
    dict keyed by the 5 polymorph labels the picker can return.
  - ``Wolf18``: same 7 properties (no ``phase`` method; PALEOS's real
    Wolf18 is a polymorph-level class with phase as a string attribute).
  - ``_P_HPCEN_BRG`` constant (12 GPa).
  - Phase boundary functions ``P_brg_ppv``, ``P_en_hpcen``,
    ``P_lpcen_hpcen``, ``P_lpcen_en``: each takes T [K] and returns P [Pa].
- ``paleos.water_eos.WaterEoS``: same 7 properties; accepts an optional
  ``table_path=...`` constructor kwarg (the real ``WaterEoS`` does too).

Phase boundary function values are constants chosen so the picker
``_get_mgsio3_solid_phase`` dispatches correctly at every probe point used
in the tests. See the comment block in ``_register_mgsio3_module`` for the
full audit. Where a test asserts physical plausibility (iron density in a
realistic range, S_liquid != S_solid, ...), the mock returns realistic
constants, not 0 or 1, so the assertion would still catch a real bug.

References
----------
- ``src/zalmoxis/eos/paleos_api.py``: the consumer.
- ``tests/test_paleos_api.py``: the tests.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

_REGISTERED = False


def install_if_missing() -> bool:
    """Install the mock ``paleos`` into ``sys.modules`` iff the real one is absent.

    Returns
    -------
    bool
        True if the mock was installed (real PALEOS not importable),
        False otherwise. Safe to call repeatedly; re-installation is a no-op.
    """
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        import paleos  # noqa: F401

        return False
    except ImportError:
        pass

    _register_paleos_root()
    _register_iron_module()
    _register_mgsio3_module()
    _register_water_module()
    _REGISTERED = True
    return True


def _register_paleos_root() -> None:
    """Build and register the top-level ``paleos`` mock package."""
    paleos = types.ModuleType('paleos')
    paleos.__file__ = '/tmp/paleos-mock-not-a-real-package/__init__.py'
    paleos.__version__ = '0.0.0-mock'
    paleos.__path__ = []  # marks it as a package, not a module
    sys.modules['paleos'] = paleos


class _BaseFakeEoS:
    """Minimum surface every PALEOS EoS class exposes.

    The 7 scalar property methods are the contract ``_evaluate_eos`` exercises.
    Subclasses override the constants so different materials and polymorphs
    return distinguishable values; this lets tests like
    ``test_liquid_entropy_differs_from_solid_at_matched_p_t`` discriminate
    a swap between the two output files.

    Default constants are deliberately physically plausible (kg/m^3 for
    density, J/(kg K) for entropy) so any test asserting "in a realistic
    range" continues to catch real bugs.
    """

    _RHO = 5500.0
    _U = 1.0e6
    _S = 1500.0
    _CP = 1200.0
    _CV = 900.0
    _ALPHA = 1.5e-5
    _NABLA_AD = 0.27

    def density(self, P, T):  # noqa: ARG002
        return self._RHO

    def specific_internal_energy(self, P, T):  # noqa: ARG002
        return self._U

    def specific_entropy(self, P, T):  # noqa: ARG002
        return self._S

    def isobaric_heat_capacity(self, P, T):  # noqa: ARG002
        return self._CP

    def isochoric_heat_capacity(self, P, T):  # noqa: ARG002
        return self._CV

    def thermal_expansion(self, P, T):  # noqa: ARG002
        return self._ALPHA

    def adiabatic_gradient(self, P, T):  # noqa: ARG002
        return self._NABLA_AD


class _FakeIronEoS(_BaseFakeEoS):
    """Mock iron EoS with values inside the realistic Earth-core range.

    Density = 8000 kg/m^3 falls in [4e3, 1.5e4] which the round-trip test
    (``test_iron_table_round_trips_through_load_paleos_all_properties``)
    asserts. The ``phase`` method returns a stable string label.
    """

    _RHO = 8000.0
    _S = 200.0

    def phase(self, P, T):  # noqa: ARG002
        return 'hcp-Fe'


class _FakeWolf18(_BaseFakeEoS):
    """Mock liquid MgSiO3 EoS (Wolf18 RTpress)."""

    _RHO = 3500.0
    _S = 4500.0


def _register_iron_module() -> None:
    """Build and register ``paleos.iron_eos`` with ``IronEoS``."""
    iron = types.ModuleType('paleos.iron_eos')
    iron.IronEoS = _FakeIronEoS
    sys.modules['paleos.iron_eos'] = iron
    sys.modules['paleos'].iron_eos = iron


# Polymorph-level mock EoS classes. Each carries a distinct entropy so the
# 2-phase generator's per-row dispatch through ``_phase_eos_map`` produces
# distinguishable rows in the solid-side file. Wolf18 (liquid) sits at
# 4500 J/(kg K), polymorphs sit at 100-300; the gap is far larger than the
# 1 J/(kg K) margin in the entropy-differs test.
class _FakeSolidLpcen(_BaseFakeEoS):
    _RHO = 3200.0
    _S = 100.0


class _FakeSolidEn(_BaseFakeEoS):
    _RHO = 3300.0
    _S = 150.0


class _FakeSolidHpcen(_BaseFakeEoS):
    _RHO = 3500.0
    _S = 200.0


class _FakeSolidBrg(_BaseFakeEoS):
    _RHO = 4100.0
    _S = 250.0


class _FakeSolidPpv(_BaseFakeEoS):
    _RHO = 4400.0
    _S = 300.0


class _FakeMgSiO3EoS(_BaseFakeEoS):
    """Top-level MgSiO3 EoS with a polymorph dispatch map.

    The five polymorph keys match the labels the production picker
    ``_get_mgsio3_solid_phase`` can return.
    """

    _RHO = 3700.0
    _S = 350.0

    def __init__(self):
        self._phase_eos_map = {
            'solid-lpcen': _FakeSolidLpcen(),
            'solid-en': _FakeSolidEn(),
            'solid-hpcen': _FakeSolidHpcen(),
            'solid-brg': _FakeSolidBrg(),
            'solid-ppv': _FakeSolidPpv(),
        }

    def phase(self, P, T):  # noqa: ARG002
        return 'liquid'


# Phase-boundary function values are calibrated against every probe in
# tests/test_paleos_api.py::TestGetMgsio3SolidPhase. The picker logic is:
#
#     if P >= P_brg_ppv(T):    return 'solid-ppv'
#     if P >= _P_HPCEN_BRG:    return 'solid-brg'
#     if P >= P_en_hpcen(T):
#         if P >= P_lpcen_hpcen(T): return 'solid-hpcen'
#         return 'solid-lpcen'
#     if T > 750.0 and P < P_lpcen_en(T): return 'solid-en'
#     return 'solid-lpcen'
#
# Probe-by-probe verification (P [Pa], T [K] -> expected):
#   (2e11,  2000) -> ppv     : P >= 1.265e11 (P_brg_ppv).
#   (5e10,  2000) -> brg     : P < 1.265e11; P >= 1.2e10 (_P_HPCEN_BRG).
#   (1e8,    700) -> lpcen   : P < 1.2e10; P < 6.3e9 (P_en_hpcen);
#                              T <= 750 -> en branch skipped -> final lpcen.
#   (1e8,   1500) -> en      : P < 1.2e10; P < 6.3e9; T > 750;
#                              P < 5e9 (P_lpcen_en).
#   (5e9,   1000) -> lpcen   : P < 1.2e10; P < 6.3e9; T > 750;
#                              P NOT < 5e9 -> final lpcen.
#   (6.4e9, 1000) -> lpcen   : P >= 6.3e9 (enters inner block);
#                              P < 6.54e9 (P_lpcen_hpcen) -> inner lpcen.
#   (2e11,  3000) -> ppv     : same as first.
#   (1e11,  6000) -> brg     : starts with 'solid-' (any polymorph).
#   (5e10,  2500) -> brg     : same as second.
_P_HPCEN_BRG_VALUE = 1.2e10  # 12 GPa
_P_BRG_PPV = 1.265e11
_P_EN_HPCEN = 6.30e9
_P_LPCEN_HPCEN = 6.54e9
_P_LPCEN_EN = 5.0e9


def _P_brg_ppv(T):  # noqa: N802, ARG001
    return _P_BRG_PPV


def _P_en_hpcen(T):  # noqa: N802, ARG001
    return _P_EN_HPCEN


def _P_lpcen_hpcen(T):  # noqa: N802, ARG001
    return _P_LPCEN_HPCEN


def _P_lpcen_en(T):  # noqa: N802, ARG001
    return _P_LPCEN_EN


def _register_mgsio3_module() -> None:
    """Build and register ``paleos.mgsio3_eos``."""
    mod = types.ModuleType('paleos.mgsio3_eos')
    mod.MgSiO3EoS = _FakeMgSiO3EoS
    mod.Wolf18 = _FakeWolf18
    mod._P_HPCEN_BRG = _P_HPCEN_BRG_VALUE
    mod.P_brg_ppv = _P_brg_ppv
    mod.P_en_hpcen = _P_en_hpcen
    mod.P_lpcen_hpcen = _P_lpcen_hpcen
    mod.P_lpcen_en = _P_lpcen_en
    sys.modules['paleos.mgsio3_eos'] = mod
    sys.modules['paleos'].mgsio3_eos = mod


class _FakeWaterEoS(_BaseFakeEoS):
    """Mock water EoS. Accepts an optional ``table_path`` kwarg like real WaterEoS."""

    _RHO = 1500.0
    _S = 3000.0

    def __init__(self, table_path=None):
        self.table_path = table_path

    def phase(self, P, T):  # noqa: ARG002
        return 'liquid'


def _register_water_module() -> None:
    """Build and register ``paleos.water_eos``."""
    mod = types.ModuleType('paleos.water_eos')
    mod.WaterEoS = _FakeWaterEoS
    sys.modules['paleos.water_eos'] = mod
    sys.modules['paleos'].water_eos = mod


# Resolve symbols on import for callers that import the classes directly,
# e.g. ``from tests._paleos_mock import _FakeIronEoS`` for fixture composition.
__all__ = [
    '_BaseFakeEoS',
    '_FakeIronEoS',
    '_FakeMgSiO3EoS',
    '_FakeWaterEoS',
    '_FakeWolf18',
    'install_if_missing',
]


# Path() smoke-check: keep ``paleos.__file__`` valid as a Path string even
# though the file does not exist. Path(...).resolve() does not require
# existence on POSIX in non-strict mode, which is the default.
_ = Path('/tmp/paleos-mock-not-a-real-package').resolve()
