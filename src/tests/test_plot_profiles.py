"""Smoke tests for the planet profile plotting module.

Tests cover:
- 6-panel profile figure generation with synthetic Earth-like data
- Output file creation and cleanup
- Both code paths: without layer_eos_config (fallback text) and with
  a mocked phase lookup (legend patches)

References:
    - docs/testing.md
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np
import pytest

matplotlib.use('Agg')


@pytest.mark.unit
class TestPlotPlanetProfileSingle:
    """Smoke tests for plot_planet_profile_single."""

    @staticmethod
    def _make_earth_like_profiles(n=200):
        """Create synthetic Earth-like interior profiles.

        Parameters
        ----------
        n : int
            Number of radial shells.

        Returns
        -------
        dict
            Arrays keyed by argument name for plot_planet_profile_single.
        """
        from zalmoxis.constants import earth_mass, earth_radius

        radii = np.linspace(0, earth_radius, n)
        # Density: decreasing from center to surface
        density = 13000 - 10000 * (radii / earth_radius) ** 0.6
        # Pressure: decreasing from ~360 GPa to ~1 atm
        pressure = 360e9 * (1 - (radii / earth_radius) ** 1.8) + 1e5
        # Temperature: decreasing from ~6000 K to ~300 K
        temperature = 6000 - 5700 * (radii / earth_radius) ** 0.7
        # Gravity: zero at center, peaks around half-radius, drops at surface
        t = radii / earth_radius
        gravity = 10.0 * np.sin(np.pi * t) * (1 - 0.15 * t)
        gravity[0] = 0.0
        # Mass enclosed: monotonically increasing
        mass_enclosed = earth_mass * (radii / earth_radius) ** 3

        cmb_radius = 3.48e6  # m
        cmb_mass = 0.325 * earth_mass
        average_density = 5515.0  # kg/m^3

        return {
            'radii': radii,
            'density': density,
            'gravity': gravity,
            'pressure': pressure,
            'temperature': temperature,
            'cmb_radius': cmb_radius,
            'cmb_mass': cmb_mass,
            'average_density': average_density,
            'mass_enclosed': mass_enclosed,
        }

    def test_profile_plot_no_eos(self, tmp_path, monkeypatch):
        """Generate a profile plot without phase lookup (layer_eos_config=None).

        All phase labels become 'unknown', so the phase panel shows the
        fallback italic text instead of colored patches.
        """
        import zalmoxis.plots.plot_profiles as mod

        monkeypatch.setattr(mod, 'ZALMOXIS_ROOT', str(tmp_path))
        os.makedirs(tmp_path / 'output_files', exist_ok=True)

        profiles = self._make_earth_like_profiles()
        mod.plot_planet_profile_single(
            **profiles,
            id_mass=None,
            layer_eos_config=None,
        )

        out = tmp_path / 'output_files' / 'planet_profile.png'
        assert out.exists(), 'Expected planet_profile.png to be created'
        assert out.stat().st_size > 1000, 'Output file suspiciously small'

    def test_profile_plot_with_id_mass(self, tmp_path, monkeypatch):
        """Filename includes id_mass when it is not None."""
        import zalmoxis.plots.plot_profiles as mod

        monkeypatch.setattr(mod, 'ZALMOXIS_ROOT', str(tmp_path))
        os.makedirs(tmp_path / 'output_files', exist_ok=True)

        profiles = self._make_earth_like_profiles()
        mod.plot_planet_profile_single(
            **profiles,
            id_mass=7,
            layer_eos_config=None,
        )

        out = tmp_path / 'output_files' / 'planet_profile7.png'
        assert out.exists(), 'Expected planet_profile7.png to be created'

    def test_profile_plot_with_phase_lookup(self, tmp_path, monkeypatch):
        """Exercise the phase lookup branch via a mocked _lookup_phases.

        Uses two distinct phases so the legend-patch branch is taken
        (covers the 'if patches' path and Patch construction).
        """
        import zalmoxis.plots.plot_profiles as mod

        monkeypatch.setattr(mod, 'ZALMOXIS_ROOT', str(tmp_path))
        os.makedirs(tmp_path / 'output_files', exist_ok=True)

        profiles = self._make_earth_like_profiles(n=200)
        n = len(profiles['radii'])

        # First half 'liquid', second half 'solid-brg'
        fake_phases = ['liquid'] * (n // 2) + ['solid-brg'] * (n - n // 2)

        monkeypatch.setattr(mod, '_lookup_phases', lambda *_a, **_kw: fake_phases)

        mod.plot_planet_profile_single(
            **profiles,
            id_mass=None,
            layer_eos_config={'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
        )

        out = tmp_path / 'output_files' / 'planet_profile.png'
        assert out.exists()
        assert out.stat().st_size > 1000
