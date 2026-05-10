"""Coverage tests for ``plots.plot_profiles._lookup_phases``.

The unified-EOS cache is mocked so the test runs in milliseconds and does not
require ~600 MB of EOS data to be present. ``EOS_REGISTRY`` is patched with a
synthetic two-material registry that exercises every branch.
"""

from __future__ import annotations

from unittest.mock import patch

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

pytestmark = pytest.mark.unit


def _fake_unified_cache(*_args, **_kw):
    """Return a synthetic cache dict matching what ``_ensure_unified_cache`` builds.

    Phase grid:
    - low-pressure half (ip < 4) -> 'liquid' (ambiguous, tagged with material name)
    - high-pressure half (ip >= 4) -> 'solid-brg' (unambiguous, no tag)
    - phase_grid[0, 0] = '' -> falsy, becomes 'unknown'
    """
    unique_log_p = np.linspace(5.0, 12.0, 8)  # log10 Pa
    unique_log_t = np.linspace(2.0, 4.0, 8)  # log10 K
    phase_grid = np.full((8, 8), 'solid-brg', dtype=object)
    phase_grid[:4, :] = 'liquid'
    phase_grid[0, 0] = ''
    return {
        'p_min': 10 ** unique_log_p[0],
        'p_max': 10 ** unique_log_p[-1],
        'unique_log_p': unique_log_p,
        'unique_log_t': unique_log_t,
        'phase_grid': phase_grid,
    }


def _earth_like(n=24):
    """Synthetic monotone Earth-like profile."""
    radii = np.linspace(0.0, 6.378e6, n)
    pressure = np.linspace(360e9, 1.0e5, n)
    density = np.linspace(13000.0, 3000.0, n)
    temperature = np.linspace(6000.0, 1500.0, n)
    mass_enclosed = np.linspace(0.0, 5.972e24, n)
    cmb_mass = 1.94e24  # ~32% of Earth mass
    return radii, pressure, density, temperature, mass_enclosed, cmb_mass


def test_lookup_phases_paleos_path_with_ambiguous_tagging(monkeypatch):
    """All shells routed to PALEOS:iron / PALEOS:MgSiO3 -> non-empty labels.

    The synthetic cache puts low-(P,T) cells in 'liquid' (ambiguous, tagged
    with material) and high-(P,T) cells in 'solid-brg' (unambiguous).
    """
    from zalmoxis.plots import plot_profiles as mod

    radii, pressure, density, temperature, mass_enclosed, cmb_mass = _earth_like()

    fake_registry = {
        'PALEOS:iron': {'format': 'paleos_unified', 'eos_file': 'fake_iron.txt'},
        'PALEOS:MgSiO3': {'format': 'paleos_unified', 'eos_file': 'fake_mgsio3.txt'},
    }

    layer_eos_config = {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'}

    with (
        patch(
            'zalmoxis.eos._ensure_unified_cache',
            side_effect=_fake_unified_cache,
        ),
        patch.object(
            __import__('zalmoxis.eos_properties', fromlist=['EOS_REGISTRY']),
            'EOS_REGISTRY',
            fake_registry,
        ),
    ):
        labels = mod._lookup_phases(
            radii,
            pressure,
            density,
            temperature,
            mass_enclosed,
            layer_eos_config,
            cmb_mass,
        )

    assert len(labels) == len(radii)
    # Outer (low P) mantle shells -> 'liquid:MgSiO3' (ambiguous, tagged).
    assert any(label.startswith('liquid:') for label in labels)
    # Inner (high P) shells -> 'solid-brg' (unambiguous, untagged).
    assert any(label == 'solid-brg' for label in labels)
    # No 'none' labels because P>0 and density>0 throughout.
    assert 'none' not in labels


def test_lookup_phases_zero_pressure_yields_none():
    """P<=0 or density<=0 -> 'none' label without touching the EOS cache."""
    from zalmoxis.plots import plot_profiles as mod

    radii = np.array([0.0, 1.0e6, 2.0e6])
    pressure = np.array([0.0, 1.0e9, -1.0])  # zero, valid, negative
    density = np.array([5000.0, 0.0, 5000.0])  # valid, zero, valid (but P<0)
    temperature = np.array([3000.0, 2000.0, 1500.0])
    mass_enclosed = np.array([0.0, 1.0e22, 5.0e23])
    cmb_mass = 1.0e22

    labels = mod._lookup_phases(
        radii,
        pressure,
        density,
        temperature,
        mass_enclosed,
        layer_eos_config={'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
        cmb_mass=cmb_mass,
    )

    # All three shells hit the early-continue 'none' branch.
    assert labels == ['none', 'none', 'none']


def test_lookup_phases_eos_not_in_registry_returns_unknown():
    """eos_name not in EOS_REGISTRY -> 'unknown' label."""
    from zalmoxis.plots import plot_profiles as mod

    radii, pressure, density, temperature, mass_enclosed, cmb_mass = _earth_like(n=10)

    # Empty registry (real registry might have these names but we patch to empty)
    with patch.object(
        __import__('zalmoxis.eos_properties', fromlist=['EOS_REGISTRY']),
        'EOS_REGISTRY',
        {},
    ):
        labels = mod._lookup_phases(
            radii,
            pressure,
            density,
            temperature,
            mass_enclosed,
            layer_eos_config={'core': 'NOT_REAL:x', 'mantle': 'ALSO_FAKE:y'},
            cmb_mass=cmb_mass,
        )

    assert all(label == 'unknown' for label in labels)


def test_lookup_phases_non_paleos_format_returns_unknown():
    """EOS exists but format != 'paleos_unified' -> 'unknown' label."""
    from zalmoxis.plots import plot_profiles as mod

    radii, pressure, density, temperature, mass_enclosed, cmb_mass = _earth_like(n=8)

    fake_registry = {
        'Seager:iron': {'format': 'seager_table', 'eos_file': 'whatever.txt'},
    }

    with patch.object(
        __import__('zalmoxis.eos_properties', fromlist=['EOS_REGISTRY']),
        'EOS_REGISTRY',
        fake_registry,
    ):
        labels = mod._lookup_phases(
            radii,
            pressure,
            density,
            temperature,
            mass_enclosed,
            layer_eos_config={'core': 'Seager:iron', 'mantle': 'Seager:iron'},
            cmb_mass=cmb_mass,
        )

    assert all(label == 'unknown' for label in labels)


def test_lookup_phases_empty_layer_config():
    """layer_eos_config without 'core'/'mantle' keys -> primary is None -> 'unknown'."""
    from zalmoxis.plots import plot_profiles as mod

    radii, pressure, density, temperature, mass_enclosed, cmb_mass = _earth_like(n=8)

    labels = mod._lookup_phases(
        radii,
        pressure,
        density,
        temperature,
        mass_enclosed,
        layer_eos_config={},  # no 'core' or 'mantle' keys
        cmb_mass=cmb_mass,
    )

    assert all(label == 'unknown' for label in labels)


def test_lookup_phases_unambiguous_phase_no_tag(monkeypatch):
    """Unambiguous phase (e.g. 'solid-brg') is returned as-is without ':material' tag."""
    from zalmoxis.plots import plot_profiles as mod

    # All-high P-T so we land in the solid-brg quadrant of the cache grid.
    n = 8
    radii = np.linspace(0.0, 6.378e6, n)
    pressure = np.full(n, 1.0e11)  # 100 GPa, high
    density = np.full(n, 5000.0)
    temperature = np.full(n, 5000.0)  # high T
    mass_enclosed = np.linspace(0.0, 5.972e24, n)
    cmb_mass = 1.94e24

    fake_registry = {
        'PALEOS:iron': {'format': 'paleos_unified', 'eos_file': 'fake_iron.txt'},
        'PALEOS:MgSiO3': {'format': 'paleos_unified', 'eos_file': 'fake_mgsio3.txt'},
    }

    with (
        patch(
            'zalmoxis.eos._ensure_unified_cache',
            side_effect=_fake_unified_cache,
        ),
        patch.object(
            __import__('zalmoxis.eos_properties', fromlist=['EOS_REGISTRY']),
            'EOS_REGISTRY',
            fake_registry,
        ),
    ):
        labels = mod._lookup_phases(
            radii,
            pressure,
            density,
            temperature,
            mass_enclosed,
            layer_eos_config={'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
            cmb_mass=cmb_mass,
        )

    # All shells should be 'solid-brg' from the upper quadrant of the cache.
    assert all(label == 'solid-brg' for label in labels)
