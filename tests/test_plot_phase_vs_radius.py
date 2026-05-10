"""Smoke tests for ``plots.plot_phase_vs_radius``.

Drive the public ``plot_PT_with_phases`` helper with synthetic profiles and
assert the PDF is created. Two scenarios cover the only branch in the module:
``cmb_radius`` set vs ``cmb_radius=None``.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')

pytestmark = pytest.mark.unit


def _synthetic_mantle_profile(n=80):
    """Return a tiny synthetic mantle P-T-r profile with mixed phase labels."""
    pressure = np.linspace(135.0e9, 1.0e5, n)  # CMB to surface in Pa
    temperature = np.linspace(4500.0, 1500.0, n)
    radii = np.linspace(3.48e6, 6.378e6, n)  # CMB radius to Earth radius

    third = n // 3
    mantle_phases = (
        ['solid_mantle'] * third
        + ['mixed_mantle'] * third
        + ['melted_mantle'] * (n - 2 * third)
    )
    return pressure, temperature, radii, mantle_phases


def test_plot_PT_with_phases_writes_pdf_with_cmb(tmp_path, monkeypatch):
    """All three phase masks populated and CMB axhline path taken."""
    import zalmoxis
    import zalmoxis.plots.plot_phase_vs_radius as mod

    monkeypatch.setattr(zalmoxis, '_zalmoxis_root', str(tmp_path))
    os.makedirs(tmp_path / 'output', exist_ok=True)

    pressure, temperature, radii, mantle_phases = _synthetic_mantle_profile()
    cmb_radius = float(radii[len(radii) // 3])

    mod.plot_PT_with_phases(pressure, temperature, radii, mantle_phases, cmb_radius)

    out = tmp_path / 'output' / 'mantle_PT_profile.pdf'
    assert out.exists(), 'Expected mantle_PT_profile.pdf to be created'
    # PDFs from matplotlib are typically several KB at minimum.
    assert out.stat().st_size > 1000, 'PDF suspiciously small'


def test_plot_PT_with_phases_no_cmb_branch(tmp_path, monkeypatch):
    """cmb_radius=None skips the axhline ('if cmb_radius is not None' False)."""
    import zalmoxis
    import zalmoxis.plots.plot_phase_vs_radius as mod

    monkeypatch.setattr(zalmoxis, '_zalmoxis_root', str(tmp_path))
    os.makedirs(tmp_path / 'output', exist_ok=True)

    pressure, temperature, radii, mantle_phases = _synthetic_mantle_profile()

    mod.plot_PT_with_phases(pressure, temperature, radii, mantle_phases, None)

    out = tmp_path / 'output' / 'mantle_PT_profile.pdf'
    assert out.exists()
    assert out.stat().st_size > 1000


def test_plot_PT_with_phases_all_one_phase(tmp_path, monkeypatch):
    """Edge case: all shells solid -> only 'Solid mantle' line is non-empty.

    Verifies the function does not crash when one or two phase masks are
    empty (a regression guard against ``ax.plot`` failing on length-0 arrays).
    """
    import zalmoxis
    import zalmoxis.plots.plot_phase_vs_radius as mod

    monkeypatch.setattr(zalmoxis, '_zalmoxis_root', str(tmp_path))
    os.makedirs(tmp_path / 'output', exist_ok=True)

    n = 30
    pressure = np.linspace(135.0e9, 1.0e5, n)
    temperature = np.linspace(2500.0, 1500.0, n)
    radii = np.linspace(3.48e6, 6.378e6, n)
    mantle_phases = ['solid_mantle'] * n

    mod.plot_PT_with_phases(pressure, temperature, radii, mantle_phases, None)

    out = tmp_path / 'output' / 'mantle_PT_profile.pdf'
    assert out.exists()
