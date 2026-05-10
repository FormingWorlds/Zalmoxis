"""Branch tests for ``phase_columns`` paths the existing suite does not cover.

Targets:
- ``phase_for_component`` returning ``PHASE_UNKNOWN`` for empty ``eos_name``.
- ``phase_for_component`` returning ``PHASE_UNKNOWN`` for ``paleos_api`` format.
- ``phase_for_component`` Seager2007 fallthrough when entry exists but lacks
  recognised format flags.
- ``compute_layer_phase_columns`` skipping a layer whose EOS string fails to
  parse (``parse_layer_components`` raises ``ValueError``).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from zalmoxis.phase_columns import (
    PHASE_UNKNOWN,
    compute_layer_phase_columns,
    phase_for_component,
)

pytestmark = pytest.mark.unit


def test_phase_for_component_empty_eos_name_returns_unknown():
    """Empty string short-circuits before any registry / EOS lookup."""
    assert phase_for_component('', P=1.0e10, T=2000.0) == PHASE_UNKNOWN


def test_phase_for_component_paleos_api_format_returns_unknown():
    """A ``paleos_api`` registry entry returns PHASE_UNKNOWN (no live lookup)."""
    fake_registry = {
        'PALEOS-API:MgSiO3': {'format': 'paleos_api', 'eos_file': '/dev/null'},
    }
    with patch.object(
        __import__('zalmoxis.eos_properties', fromlist=['EOS_REGISTRY']),
        'EOS_REGISTRY',
        fake_registry,
    ):
        result = phase_for_component('PALEOS-API:MgSiO3', P=1.0e10, T=2000.0)
    assert result == PHASE_UNKNOWN


def test_phase_for_component_unrecognised_format_returns_unknown():
    """Registry entry that is a dict with no recognised format/keys -> UNKNOWN."""
    fake_registry = {
        'Custom:thing': {'format': 'something_else'},
    }
    with patch.object(
        __import__('zalmoxis.eos_properties', fromlist=['EOS_REGISTRY']),
        'EOS_REGISTRY',
        fake_registry,
    ):
        result = phase_for_component('Custom:thing', P=1.0e10, T=2000.0)
    assert result == PHASE_UNKNOWN


def test_phase_for_component_seager_in_registry_returns_solid():
    """Seager2007:* in the registry but without paleos_unified format ->
    falls through to the bottom-level startswith check and returns 'solid'."""
    fake_registry = {
        'Seager2007:H2O': {'format': 'seager_table'},
    }
    with patch.object(
        __import__('zalmoxis.eos_properties', fromlist=['EOS_REGISTRY']),
        'EOS_REGISTRY',
        fake_registry,
    ):
        result = phase_for_component('Seager2007:H2O', P=1.0e10, T=2000.0)
    assert result == 'solid'


def test_compute_layer_phase_columns_skips_unparseable_layer(caplog):
    """A layer whose EOS string fails to parse is logged and skipped without
    aborting the per-shell phase assignment for the other layers."""
    n = 6
    pressure = np.linspace(3.0e11, 1.0e5, n)
    temperature = np.linspace(5500.0, 1500.0, n)
    mass_enclosed = np.linspace(0.0, 5.972e24, n)
    cmb_mass = 0.325 * 5.972e24
    core_mantle_mass = 5.972e24

    # Mantle string is malformed: a multi-component string with a negative
    # fraction (rejected by parse_layer_components -> ValueError).
    layer_eos_config = {
        'core': 'PALEOS:iron',
        'mantle': 'PALEOS:MgSiO3:-0.3+PALEOS:H2O:1.3',
    }

    components, phases = compute_layer_phase_columns(
        pressure=pressure,
        temperature=temperature,
        mass_enclosed=mass_enclosed,
        cmb_mass=cmb_mass,
        core_mantle_mass=core_mantle_mass,
        layer_eos_config=layer_eos_config,
    )

    # Both columns produced for every shell, no exception bubble-up.
    assert len(components) == n
    assert len(phases) == n
    # Mantle shells (mass > cmb_mass) should be PHASE_UNKNOWN since parse
    # failed; core shells should still resolve.
    for i in range(n):
        if mass_enclosed[i] >= cmb_mass:
            assert components[i] == PHASE_UNKNOWN
            assert phases[i] == PHASE_UNKNOWN
