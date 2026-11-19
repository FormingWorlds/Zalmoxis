"""Tests for adiabatic temperature mode.

Tests cover:
- Adiabatic temperature profile integration via native dT/dP tables
- PALEOS phase-aware adiabat (solid + liquid nabla_ad)
- Adiabat blend convergence (0 -> 0.5 -> 1.0 transition)
- Thick-mantle divergence regression test
- Adiabatic mode in calculate_temperature_profile()

References:
    - docs/testing.md
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests._plot_checks import assert_text_clear, legend_below
from zalmoxis.eos import (
    calculate_temperature_profile,
    compute_adiabatic_temperature,
)
from zalmoxis.eos.temperature import ADIABAT_P_FLOOR as P_FLOOR  # Pa; T held below it
from zalmoxis.mixing import parse_all_layer_mixtures


@pytest.mark.unit
class TestComputeAdiabaticTemperature:
    """Tests for compute_adiabatic_temperature() using native dT/dP tables."""

    def test_surface_temperature_exact(self):
        """T at the surface should equal surface_temperature exactly."""
        import os

        grad_file = os.path.join(
            os.environ.get('ZALMOXIS_ROOT', ''),
            'data',
            'EOS_WolfBower2018_1TPa',
            'adiabat_temp_grad_melt.dat',
        )
        if not os.path.isfile(grad_file):
            pytest.skip('WolfBower2018 adiabat gradient table not found')

        n = 50
        radii = np.linspace(0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e5, n)  # center to surface
        mass_enclosed = np.linspace(0, 5.972e24, n)

        T_surface = 3500.0
        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'}
        from zalmoxis.config import load_material_dictionaries

        material_dicts = load_material_dictionaries()

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=T_surface,
            cmb_mass=0.325 * 5.972e24,
            core_mantle_mass=5.972e24,
            layer_mixtures=parse_all_layer_mixtures(layer_eos_config),
            material_dictionaries=material_dicts,
        )
        assert T[-1] == pytest.approx(T_surface), f'Surface temperature {T[-1]} != {T_surface}'

    def test_core_is_isothermal(self):
        """The iron core (T-independent EOS) should be isothermal.

        Only the mantle has a dT/dP table; the core should hold T constant
        at whatever temperature the CMB reaches.
        """
        import os

        grad_file = os.path.join(
            os.environ.get('ZALMOXIS_ROOT', ''),
            'data',
            'EOS_WolfBower2018_1TPa',
            'adiabat_temp_grad_melt.dat',
        )
        if not os.path.isfile(grad_file):
            pytest.skip('WolfBower2018 adiabat gradient table not found')

        n = 100
        radii = np.linspace(0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e5, n)
        mass_enclosed = np.linspace(0, 5.972e24, n)

        T_surface = 3500.0
        CMF = 0.325
        cmb_mass = CMF * 5.972e24
        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'}
        from zalmoxis.config import load_material_dictionaries

        material_dicts = load_material_dictionaries()

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=T_surface,
            cmb_mass=cmb_mass,
            core_mantle_mass=5.972e24,
            layer_mixtures=parse_all_layer_mixtures(layer_eos_config),
            material_dictionaries=material_dicts,
        )
        # Core shells: mass_enclosed < cmb_mass
        core_mask = mass_enclosed < cmb_mass
        core_T = T[core_mask]
        # All core temperatures should be identical (isothermal)
        np.testing.assert_allclose(core_T, core_T[0], rtol=1e-10)

    def test_monotonic_increase_with_Tdep_eos(self):
        """T should increase from surface toward center for a T-dependent mantle EOS.

        The adiabatic gradient dT/dP > 0 means T increases with pressure
        (i.e., inward toward the center).
        """
        import os

        grad_file = os.path.join(
            os.environ.get('ZALMOXIS_ROOT', ''),
            'data',
            'EOS_WolfBower2018_1TPa',
            'adiabat_temp_grad_melt.dat',
        )
        if not os.path.isfile(grad_file):
            pytest.skip('WolfBower2018 adiabat gradient table not found')

        from zalmoxis.config import load_material_dictionaries

        material_dicts = load_material_dictionaries()
        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'}

        n = 100
        radii = np.linspace(0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e5, n)
        mass_enclosed = np.linspace(0, 5.972e24, n)

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=3500.0,
            cmb_mass=0.325 * 5.972e24,
            core_mantle_mass=5.972e24,
            layer_mixtures=parse_all_layer_mixtures(layer_eos_config),
            material_dictionaries=material_dicts,
        )
        # In the mantle (T-dependent EOS), T should increase inward.
        cmb_idx = np.argmax(mass_enclosed >= 0.325 * 5.972e24)
        mantle_T = T[cmb_idx:]
        diffs = np.diff(mantle_T)
        assert np.all(diffs <= 0), (
            f'Mantle T profile is not monotonically decreasing from CMB to surface. '
            f'Max upward step: {np.max(diffs):.1f} K at index {np.argmax(diffs)}'
        )

    def test_missing_adiabat_grad_file_holds_T_constant(self):
        """If adiabat_grad_file is missing, T should be held constant (isothermal).

        The mixing-based adiabat gracefully degrades: if nabla_ad cannot be
        computed for a component, it returns None and the temperature is held
        constant at that shell (rather than raising ValueError).
        """
        import copy

        from zalmoxis.config import load_material_dictionaries

        material_dicts = copy.deepcopy(load_material_dictionaries())
        # Remove the adiabat_grad_file key from WolfBower2018 melted_mantle
        material_dicts['WolfBower2018:MgSiO3']['melted_mantle'].pop('adiabat_grad_file', None)

        n = 50
        radii = np.linspace(0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e5, n)
        mass_enclosed = np.linspace(0, 5.972e24, n)

        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'}

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=3500.0,
            cmb_mass=0.325 * 5.972e24,
            core_mantle_mass=5.972e24,
            layer_mixtures=parse_all_layer_mixtures(layer_eos_config),
            material_dictionaries=material_dicts,
        )
        # Without gradient data, T should be isothermal at surface_temperature
        np.testing.assert_allclose(T, 3500.0, rtol=1e-10)

    def test_rejects_T_independent_mantle_eos(self):
        """Should raise ValueError if mantle EOS is T-independent (e.g. Seager2007)."""
        from zalmoxis.config import load_material_dictionaries

        material_dicts = load_material_dictionaries()

        n = 50
        radii = np.linspace(0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e5, n)
        mass_enclosed = np.linspace(0, 5.972e24, n)

        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'}

        with pytest.raises(ValueError, match='T-dependent EOS'):
            compute_adiabatic_temperature(
                radii=radii,
                pressure=pressure,
                mass_enclosed=mass_enclosed,
                surface_temperature=3500.0,
                cmb_mass=0.325 * 5.972e24,
                core_mantle_mass=5.972e24,
                layer_mixtures=parse_all_layer_mixtures(layer_eos_config),
                material_dictionaries=material_dicts,
            )


@pytest.mark.unit
class TestNoDivergenceThickMantle:
    """Regression test for adiabat divergence with thick mantles (CMF <= 0.325).

    A dT/dr = -α·g·T/Cp formulation with finite-difference α is unstable
    here: in the mixed zone, phase-averaged density inflates α by ~100×
    through the latent-heat density jump, driving exponential T runaway
    across a thick mantle. The native dT/dP table path used by
    ``compute_adiabatic_temperature`` avoids this.
    """

    def test_no_divergence_thick_mantle(self):
        """Adiabat for an Earth-mass planet with CMF=0.325 should not diverge.

        The temperature profile should contain no NaN/Inf values and
        stay below 10,000 K for an Earth-mass planet.
        """
        import os

        grad_file = os.path.join(
            os.environ.get('ZALMOXIS_ROOT', ''),
            'data',
            'EOS_WolfBower2018_1TPa',
            'adiabat_temp_grad_melt.dat',
        )
        if not os.path.isfile(grad_file):
            pytest.skip('WolfBower2018 adiabat gradient table not found')

        from zalmoxis.config import load_material_dictionaries

        material_dicts = load_material_dictionaries()
        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'}

        # Earth-mass planet with CMF=0.325 (thick mantle ~4300 km)
        M_earth = 5.972e24
        R_earth = 6.371e6
        CMF = 0.325

        n = 200
        radii = np.linspace(0, R_earth, n)
        pressure = np.linspace(360e9, 1e5, n)
        mass_enclosed = np.linspace(0, M_earth, n)

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=3500.0,
            cmb_mass=CMF * M_earth,
            core_mantle_mass=M_earth,
            layer_mixtures=parse_all_layer_mixtures(layer_eos_config),
            material_dictionaries=material_dicts,
        )

        # No NaN or Inf values
        assert np.all(np.isfinite(T)), (
            f'Adiabat contains non-finite values: '
            f'NaN count={np.sum(np.isnan(T))}, Inf count={np.sum(np.isinf(T))}'
        )

        # Temperature should be physically reasonable (not diverge)
        assert np.max(T) < 10000.0, (
            f'Adiabat T_max={np.max(T):.0f} K is unreasonably high '
            f'(expected < 10,000 K for 1 M_earth)'
        )

    def test_no_divergence_cmf_01(self):
        """Adiabat for CMF=0.1 (very thick mantle ~5700 km) should not diverge.

        Most extreme case: the unstable α-based formulation diverges to
        ~6e7 K here, so the assertion T_max < 10000 K is a strong guard.
        """
        import os

        grad_file = os.path.join(
            os.environ.get('ZALMOXIS_ROOT', ''),
            'data',
            'EOS_WolfBower2018_1TPa',
            'adiabat_temp_grad_melt.dat',
        )
        if not os.path.isfile(grad_file):
            pytest.skip('WolfBower2018 adiabat gradient table not found')

        from zalmoxis.config import load_material_dictionaries

        material_dicts = load_material_dictionaries()
        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'}

        M_earth = 5.972e24
        R_earth = 6.371e6
        CMF = 0.1

        n = 200
        radii = np.linspace(0, R_earth, n)
        pressure = np.linspace(360e9, 1e5, n)
        mass_enclosed = np.linspace(0, M_earth, n)

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=3500.0,
            cmb_mass=CMF * M_earth,
            core_mantle_mass=M_earth,
            layer_mixtures=parse_all_layer_mixtures(layer_eos_config),
            material_dictionaries=material_dicts,
        )

        assert np.all(np.isfinite(T))
        assert np.max(T) < 10000.0, (
            f'CMF=0.1 adiabat T_max={np.max(T):.0f} K diverged (expected < 10,000 K)'
        )


@pytest.mark.unit
class TestCalculateTemperatureProfileAdiabatic:
    """Tests for 'adiabatic' mode in calculate_temperature_profile()."""

    def test_adiabatic_returns_linear_initial_guess(self):
        """Adiabatic mode should return a linear profile as the initial guess."""
        radii = np.linspace(0, 6.371e6, 50)
        T_surface = 3500.0
        T_center = 6000.0

        func = calculate_temperature_profile(
            radii=radii,
            temperature_mode='adiabatic',
            surface_temperature=T_surface,
            center_temperature=T_center,
            input_dir='.',
            temp_profile_file=None,
        )
        T = func(radii)

        # Should match the linear profile
        T_linear = T_surface + (T_center - T_surface) * (1 - radii / radii[-1])
        np.testing.assert_allclose(T, T_linear, rtol=1e-10)

    def test_adiabatic_invalid_mode_raises(self):
        """An unknown temperature mode should raise ValueError."""
        radii = np.linspace(0, 6.371e6, 50)
        with pytest.raises(ValueError, match='Unknown temperature mode'):
            calculate_temperature_profile(
                radii=radii,
                temperature_mode='nonexistent',
                surface_temperature=3500.0,
                center_temperature=6000.0,
                input_dir='.',
                temp_profile_file=None,
            )


def _paleos_data_available():
    """Check if PALEOS data files are available."""
    import os

    root = os.environ.get('ZALMOXIS_ROOT', '')
    solid = os.path.join(
        root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
    )
    liquid = os.path.join(
        root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_liquid.dat'
    )
    return os.path.isfile(solid) and os.path.isfile(liquid)


@pytest.mark.slow
class TestPALEOSAdiabaticProfile:
    """Tests for PALEOS phase-aware adiabat using nabla_ad from solid and liquid tables.

    Tagged ``slow``: each test rebuilds the full PALEOS adiabat
    profile across many radii (~2 s each on a fast Mac, projected
    ~10-15 s each on cold CI with cov instrumentation). The fast
    PALEOS adiabat behaviour is covered by the smaller tests in
    ``TestComputeAdiabaticTemperatureCmbAnchor``.
    """

    def test_paleos_adiabat_surface_temperature(self):
        """T at the surface should equal surface_temperature for PALEOS adiabat."""
        if not _paleos_data_available():
            pytest.skip('PALEOS data files not found')

        from zalmoxis.config import load_material_dictionaries
        from zalmoxis.eos import get_solidus_liquidus_functions

        material_dicts = load_material_dictionaries()
        solidus_func, liquidus_func = get_solidus_liquidus_functions()
        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'PALEOS-2phase:MgSiO3'}

        n = 50
        radii = np.linspace(0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e5, n)
        mass_enclosed = np.linspace(0, 5.972e24, n)
        T_surface = 3500.0

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=T_surface,
            cmb_mass=0.325 * 5.972e24,
            core_mantle_mass=5.972e24,
            layer_mixtures=parse_all_layer_mixtures(layer_eos_config),
            material_dictionaries=material_dicts,
            solidus_func=solidus_func,
            liquidus_func=liquidus_func,
        )
        assert T[-1] == pytest.approx(T_surface)

    def test_paleos_adiabat_no_nans(self):
        """PALEOS adiabat should contain no NaN or Inf values."""
        if not _paleos_data_available():
            pytest.skip('PALEOS data files not found')

        from zalmoxis.config import load_material_dictionaries
        from zalmoxis.eos import get_solidus_liquidus_functions

        material_dicts = load_material_dictionaries()
        solidus_func, liquidus_func = get_solidus_liquidus_functions()
        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'PALEOS-2phase:MgSiO3'}

        n = 100
        radii = np.linspace(0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e5, n)
        mass_enclosed = np.linspace(0, 5.972e24, n)

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=3500.0,
            cmb_mass=0.325 * 5.972e24,
            core_mantle_mass=5.972e24,
            layer_mixtures=parse_all_layer_mixtures(layer_eos_config),
            material_dictionaries=material_dicts,
            solidus_func=solidus_func,
            liquidus_func=liquidus_func,
        )

        assert np.all(np.isfinite(T)), (
            f'PALEOS adiabat has non-finite values: '
            f'NaN={np.sum(np.isnan(T))}, Inf={np.sum(np.isinf(T))}'
        )
        # Should stay below 10,000 K for 1 M_earth
        assert np.max(T) < 10000.0, f'PALEOS adiabat T_max={np.max(T):.0f} K too high'

    def test_paleos_adiabat_monotonic_mantle(self):
        """PALEOS mantle adiabat should increase monotonically toward the center."""
        if not _paleos_data_available():
            pytest.skip('PALEOS data files not found')

        from zalmoxis.config import load_material_dictionaries
        from zalmoxis.eos import get_solidus_liquidus_functions

        material_dicts = load_material_dictionaries()
        solidus_func, liquidus_func = get_solidus_liquidus_functions()
        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'PALEOS-2phase:MgSiO3'}

        n = 100
        radii = np.linspace(0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e5, n)
        mass_enclosed = np.linspace(0, 5.972e24, n)

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=3500.0,
            cmb_mass=0.325 * 5.972e24,
            core_mantle_mass=5.972e24,
            layer_mixtures=parse_all_layer_mixtures(layer_eos_config),
            material_dictionaries=material_dicts,
            solidus_func=solidus_func,
            liquidus_func=liquidus_func,
        )
        # Mantle T should decrease from center toward surface (radii ascending)
        cmb_idx = np.argmax(mass_enclosed >= 0.325 * 5.972e24)
        mantle_T = T[cmb_idx:]
        diffs = np.diff(mantle_T)
        assert np.all(diffs <= 0), (
            f'PALEOS mantle T not monotonically decreasing. '
            f'Max upward step: {np.max(diffs):.1f} K'
        )


@pytest.mark.unit
class TestAdiabaticBlendMechanism:
    """Tests for the adiabat blend convergence loop (0 -> 0.25 -> 0.5 -> 0.75 -> 1.0)."""

    def test_blend_step_constant(self):
        """The blend step should be 0.25 (transitions: 0 -> 0.25 -> 0.5 -> 0.75 -> 1.0).

        The blend step is a local variable inside main(), not directly
        importable. This test verifies the design invariant by checking
        the source code text.
        """
        import inspect

        from zalmoxis.solver import _solve

        source = inspect.getsource(_solve)
        assert '_ADIABAT_BLEND_STEP = 0.25' in source


# ---------------------------------------------------------------------------
# compute_adiabatic_temperature(anchor='cmb', ...) — outward integration
# ---------------------------------------------------------------------------


class _StubTdepMixture:
    """Minimal LayerMixture-like stub: ``has_tdep()`` is True (mantle-like)."""

    def has_tdep(self):
        return True


class _StubInertMixture:
    """Mixture stub with ``has_tdep()`` returning False (e.g. core-like)."""

    def has_tdep(self):
        return False


@pytest.mark.unit
class TestComputeAdiabaticTemperatureCmbAnchor:
    """Tests for ``compute_adiabatic_temperature(anchor='cmb', cmb_temperature=...)``.

    These exercise the outward-integration block (mantle from CMB to surface)
    that is invoked by ``main()`` once the converged structure exposes the
    CMB index. Existing tests in this file cover only the default
    ``anchor='surface'`` path. Stubs replace ``get_layer_mixture`` and
    ``get_mixed_nabla_ad`` so the integration logic is exercised
    deterministically without requiring the WolfBower2018 EOS data file.

    Coverage targets ``src/zalmoxis/eos/temperature.py`` lines 118-176:
    the ``if anchor == 'cmb':`` branch, the cmb_temperature validation,
    the cmb_index lookup, the core carry-through, and the upward integration
    loop including the low-pressure short-circuit and the
    nabla<=0 / dP>=0 ``else`` branch.
    """

    @staticmethod
    def _patch_helpers(monkeypatch, *, mixture, nabla):
        """Replace the in-function imports with stubs at the module level.

        ``compute_adiabatic_temperature`` does ``from ..mixing import
        get_mixed_nabla_ad`` and ``from ..structure_model import
        get_layer_mixture`` lazily inside its body, so the patch must hit
        the source modules' attribute table (Python's ``from`` re-fetches
        on each call).

        ``any_component_is_tdep`` is patched to return True so the early
        ValueError is bypassed; the stubbed mixture's ``has_tdep()``
        already covers the per-shell branch.
        """
        import zalmoxis.mixing as _mix
        import zalmoxis.structure_model as _struct

        monkeypatch.setattr(_struct, 'get_layer_mixture', lambda *a, **k: mixture)
        monkeypatch.setattr(_mix, 'get_mixed_nabla_ad', lambda *a, **k: nabla)
        monkeypatch.setattr(_mix, 'any_component_is_tdep', lambda *a, **k: True)

    def test_anchor_cmb_pins_temperature_at_cmb_index(self, monkeypatch):
        """``T[cmb_index] == cmb_temperature`` exactly; ``T[i<cmb_index] == surface_temperature``.

        Discriminating: pinning the wrong shell index (off-by-one in
        ``searchsorted``) would surface as a CMB-temperature mismatch.
        """
        self._patch_helpers(monkeypatch, mixture=_StubTdepMixture(), nabla=0.3)
        n = 21
        radii = np.linspace(0.0, 6.371e6, n)
        # Pressure monotonically decreases from center to surface.
        pressure = np.linspace(360e9, 1e6, n)
        mass_enclosed = np.linspace(0.0, 5.972e24, n)
        cmb_mass = 0.325 * 5.972e24

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=2500.0,
            cmb_mass=cmb_mass,
            core_mantle_mass=5.972e24,
            layer_mixtures={'core': _StubInertMixture(), 'mantle': _StubTdepMixture()},
            material_dictionaries={},
            anchor='cmb',
            cmb_temperature=4500.0,
        )

        cmb_index = int(np.searchsorted(mass_enclosed, cmb_mass))
        cmb_index = max(1, min(cmb_index, n - 1))
        assert T[cmb_index] == pytest.approx(4500.0)
        # All shells strictly below cmb_index carry the surface anchor.
        np.testing.assert_allclose(T[:cmb_index], 2500.0, rtol=0, atol=1e-12)
        # Mantle shells above the CMB are positive and finite.
        assert np.all(T[cmb_index:] > 0)
        assert np.all(np.isfinite(T))

    def test_anchor_cmb_cools_outward_from_cmb_to_surface(self, monkeypatch):
        """In the cooling regime (positive nabla, decreasing P outward), mantle T
        is monotonically non-increasing from CMB to surface.

        Discriminating: a wrong sign on ``dtdp * dP`` (heating instead of
        cooling outward) would flip monotonicity and fail.
        """
        self._patch_helpers(monkeypatch, mixture=_StubTdepMixture(), nabla=0.3)
        n = 30
        radii = np.linspace(0.0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e6, n)
        mass_enclosed = np.linspace(0.0, 5.972e24, n)
        cmb_mass = 0.325 * 5.972e24

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=2500.0,
            cmb_mass=cmb_mass,
            core_mantle_mass=5.972e24,
            layer_mixtures={'core': _StubInertMixture(), 'mantle': _StubTdepMixture()},
            material_dictionaries={},
            anchor='cmb',
            cmb_temperature=4500.0,
        )

        cmb_index = int(np.searchsorted(mass_enclosed, cmb_mass))
        cmb_index = max(1, min(cmb_index, n - 1))
        mantle_T = T[cmb_index:]
        # CMB shell hotter than surface shell.
        assert mantle_T[0] > mantle_T[-1]
        # Monotone non-increasing across the mantle.
        assert np.all(np.diff(mantle_T) <= 0)
        # Adiabat clamp at the upper bound: T must stay <= 100000 K and >= 100 K.
        assert mantle_T.min() >= 100.0
        assert mantle_T.max() <= 100000.0

    @pytest.mark.parametrize('bad_cmb_T', [None, 0.0, -100.0, -1e-6])
    def test_anchor_cmb_requires_positive_cmb_temperature(self, monkeypatch, bad_cmb_T):
        """``anchor='cmb'`` with a missing or non-positive ``cmb_temperature`` raises.

        Edge case + physically unreasonable input: includes None (missing),
        0 (boundary), small negative (numerical), and large negative
        (physically meaningless).
        """
        self._patch_helpers(monkeypatch, mixture=_StubTdepMixture(), nabla=0.3)
        n = 5
        radii = np.linspace(0.0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e5, n)
        mass_enclosed = np.linspace(0.0, 5.972e24, n)

        with pytest.raises(ValueError, match='cmb_temperature'):
            compute_adiabatic_temperature(
                radii=radii,
                pressure=pressure,
                mass_enclosed=mass_enclosed,
                surface_temperature=2500.0,
                cmb_mass=0.325 * 5.972e24,
                core_mantle_mass=5.972e24,
                layer_mixtures={'mantle': _StubTdepMixture()},
                material_dictionaries={},
                anchor='cmb',
                cmb_temperature=bad_cmb_T,
            )

    def test_anchor_cmb_short_circuits_at_low_pressure_shells(self, monkeypatch):
        """Below 1e5 Pa T is held: a segment with both ends below it keeps T, and a segment
        that crosses it integrates only down to 1e5 Pa, so every T stays finite."""
        self._patch_helpers(monkeypatch, mixture=_StubTdepMixture(), nabla=0.3)
        n = 6
        radii = np.linspace(0.0, 6.371e6, n)
        # Drop into the low-P short-circuit at the surface boundary.
        pressure = np.array([360e9, 100e9, 30e9, 10e9, 1e3, 1e2])
        mass_enclosed = np.linspace(0.0, 5.972e24, n)
        cmb_mass = 0.05 * 5.972e24  # cmb_index falls early in the array

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=2500.0,
            cmb_mass=cmb_mass,
            core_mantle_mass=5.972e24,
            layer_mixtures={'mantle': _StubTdepMixture()},
            material_dictionaries={},
            anchor='cmb',
            cmb_temperature=4500.0,
        )

        assert T[-1] == T[-2]  # 1e3 to 1e2 Pa: both ends below the floor
        assert T[-2] == pytest.approx(max(T[-3] * (1e5 / 10e9) ** 0.3, 100.0), rel=1e-12)
        assert np.all(np.isfinite(T))

    def test_anchor_cmb_holds_temperature_when_dp_is_nonnegative(self, monkeypatch):
        """When ``dP >= 0`` (non-monotone pressure), the integrator's ``else``
        branch holds T at the prior shell value rather than integrating.

        Discriminating: passes a deliberately non-monotone pressure profile
        (a single uphill step in the mantle) and verifies the integrator
        does NOT step T forward at that shell. Without the dP<0 guard at
        line 169, the integrator would still apply the formula and produce
        a heating step in the wrong direction.
        """
        self._patch_helpers(monkeypatch, mixture=_StubTdepMixture(), nabla=0.3)
        n = 6
        radii = np.linspace(0.0, 6.371e6, n)
        # All pressures stay above 1e5 (no low-P short-circuit). Index 4 has
        # pressure[4] > pressure[3], producing dP > 0 at i=4.
        pressure = np.array([360e9, 100e9, 30e9, 10e9, 50e9, 1e7])
        mass_enclosed = np.linspace(0.0, 5.972e24, n)
        cmb_mass = 0.05 * 5.972e24  # cmb_index = 1

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=2500.0,
            cmb_mass=cmb_mass,
            core_mantle_mass=5.972e24,
            layer_mixtures={'mantle': _StubTdepMixture()},
            material_dictionaries={},
            anchor='cmb',
            cmb_temperature=4500.0,
        )

        # At i=4, dP = pressure[4] - pressure[3] = +40 GPa > 0 -> T held.
        assert T[4] == pytest.approx(T[3])

    def test_anchor_cmb_skips_non_tdep_mantle_shell(self, monkeypatch):
        """A mantle shell whose mixture lacks tdep components carries T forward.

        Edge case: the per-shell ``if not mixture.has_tdep()`` branch
        (lines 143-145) fires when a shell maps to a layer without tdep
        components. Stubs ``get_layer_mixture`` to return an inert mixture
        for the second mantle shell and verifies T is held there.
        """
        import zalmoxis.mixing as _mix
        import zalmoxis.structure_model as _struct

        # Distinguish shells: mantle index 3 returns inert mixture; everything
        # else is tdep. Stub get_layer_mixture by mass_enclosed; here the
        # caller's mass_enclosed[i] uniquely identifies index i.
        n = 6
        radii = np.linspace(0.0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e6, n)
        mass_enclosed = np.linspace(0.0, 5.972e24, n)
        cmb_mass = 0.05 * 5.972e24
        target_mass = mass_enclosed[3]

        def stub_layer_mixture(m_at_i, *args, **kwargs):  # noqa: ARG001
            if m_at_i == target_mass:
                return _StubInertMixture()
            return _StubTdepMixture()

        monkeypatch.setattr(_struct, 'get_layer_mixture', stub_layer_mixture)
        monkeypatch.setattr(_mix, 'get_mixed_nabla_ad', lambda *a, **k: 0.3)
        monkeypatch.setattr(_mix, 'any_component_is_tdep', lambda *a, **k: True)

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=2500.0,
            cmb_mass=cmb_mass,
            core_mantle_mass=5.972e24,
            layer_mixtures={'mantle': _StubTdepMixture()},
            material_dictionaries={},
            anchor='cmb',
            cmb_temperature=4500.0,
        )

        # Shell 3 inherits shell 2's temperature unchanged.
        assert T[3] == pytest.approx(T[2])


# ---------------------------------------------------------------------------
# Midpoint integration in ln P and the low-pressure rule
# ---------------------------------------------------------------------------

M_E = 5.972e24
H_DEFAULT = 0.0025  # largest ln P change of one substep of a default call


def _stub_adiabat(monkeypatch, nabla, calls=None):
    """Make every layer T-dependent and route ``get_mixed_nabla_ad`` to ``nabla(P, T)``."""
    import zalmoxis.mixing as _mix
    import zalmoxis.structure_model as _struct

    def stub(P, T, *a, **k):
        if calls is not None:
            calls.append(P)
        return nabla(P, T)

    monkeypatch.setattr(_struct, 'get_layer_mixture', lambda *a, **k: _StubTdepMixture())
    monkeypatch.setattr(_mix, 'get_mixed_nabla_ad', stub)
    monkeypatch.setattr(_mix, 'any_component_is_tdep', lambda *a, **k: True)


def _profile(n=31, p_center=3.6e11, p_top=2e5):
    """Pressure of an n-node stub planet, geometric (equal ln P steps) from centre to top."""
    return np.geomspace(p_center, p_top, n)


def _call(P, anchor, T_anchor, **kw):
    """compute_adiabatic_temperature on pressure ``P`` with the anchor temperature ``T_anchor``;
    the CMB anchor sits at node 1 (cmb_mass just above the first node's mass)."""
    r, m = np.linspace(0.0, 6.4e6, len(P)), np.linspace(0, M_E, len(P))
    return compute_adiabatic_temperature(
        radii=r,
        pressure=P,
        mass_enclosed=m,
        surface_temperature=kw.pop(
            'surface_temperature', T_anchor if anchor == 'surface' else 2000.0
        ),
        cmb_mass=0.5 * m[1],
        core_mantle_mass=M_E,
        layer_mixtures={'mantle': _StubTdepMixture()},
        material_dictionaries={},
        anchor=anchor,
        cmb_temperature=T_anchor if anchor == 'cmb' else None,
        **kw,
    )


def _anchor_node(anchor, n):
    """Index of the anchor node of ``_call``: the top node or the CMB node 1."""
    return n - 1 if anchor == 'surface' else 1


def _integrated(anchor):
    """Nodes on the adiabat of ``_call``: all for the surface anchor, the CMB node outward for
    the CMB anchor (the nodes below it carry the surface temperature)."""
    return slice(None) if anchor == 'surface' else slice(1, None)


@pytest.mark.unit
class TestMidpointAdiabat:
    """The adiabat d ln T / d ln P = nabla, integrated per node segment by the midpoint rule in
    ln P with ceil(|d ln P| / h) substeps, and held below 1e5 Pa and at a padded top (P = 0)."""

    @pytest.mark.physics_invariant
    @pytest.mark.reference_pinned
    @pytest.mark.parametrize('anchor', ['surface', 'cmb'])
    def test_constant_nabla_is_exact(self, monkeypatch, anchor):
        """Constant nabla 0.1 gives T = T_a (P / P_a)^0.1 to round-off at every node (analytical
        law; the midpoint rule in ln P is exact for a constant right-hand side). A forward
        Euler step in P per node, the plausible wrong scheme, misses it by more than 10 %."""
        _stub_adiabat(monkeypatch, lambda P, T: 0.1)
        P = _profile()
        T = _call(P, anchor, 3000.0)
        a = _anchor_node(anchor, len(P))
        mantle = _integrated(anchor)
        exact = 3000.0 * (P / P[a]) ** 0.1
        np.testing.assert_allclose(T[mantle], exact[mantle], rtol=1e-12, atol=0)
        if anchor == 'surface':  # forward Euler step in P per node, inward from the top node
            euler, end = 3000.0 * np.prod(1 + 0.1 * (P[:-1] / P[1:] - 1)), exact[0]
        else:  # outward from the CMB node 1 to the top node
            euler, end = 3000.0 * np.prod(1 + 0.1 * (P[2:] / P[1:-1] - 1)), exact[-1]
        assert abs(euler / end - 1) > 0.1

    @pytest.mark.physics_invariant
    @pytest.mark.reference_pinned
    def test_midpoint_is_second_order(self, monkeypatch, validation_plot):
        """On 2 smooth nabla with closed-form adiabats the error falls as h^2 (order 1.9 to 2.1
        between successive h from 0.04 to 0.0025), both anchors: nabla = a exp(-(ln T -
        ln T_a)/2) with T = T_a (1 + a ln(P/P_a)/2)^2, and nabla = a + b cos(ln P) with
        ln T = ln T_a + a ln(P/P_a) + b (sin ln P - sin ln P_a); a = 0.1, b = 0.05. A first
        order scheme gives order 1, outside the band."""
        hs = (0.04, 0.02, 0.01, 0.005, 0.0025)
        P = _profile()
        T_a = 3000.0
        rows = {}
        for anchor in ('surface', 'cmb'):
            a_i = _anchor_node(anchor, len(P))
            x_a = math.log(P[a_i])
            cases = {
                'T-dependent': (
                    lambda P_, T_: 0.1 * math.sqrt(T_a / T_),
                    T_a * (1 + 0.1 * np.log(P / P[a_i]) / 2) ** 2,
                ),
                'P-dependent': (
                    lambda P_, T_: 0.1 + 0.05 * math.cos(math.log(P_)),
                    T_a
                    * np.exp(
                        0.1 * (np.log(P) - x_a) + 0.05 * (np.sin(np.log(P)) - math.sin(x_a))
                    ),
                ),
            }
            for name, (nabla, exact) in cases.items():
                _stub_adiabat(monkeypatch, nabla)
                mantle = _integrated(anchor)
                err = []
                for h in hs:
                    T = _call(P, anchor, T_a, max_dlnp=h)
                    err.append(float(np.max(np.abs(T[mantle] / exact[mantle] - 1))))
                seg = abs(math.log(P[0] / P[1]))
                dx = [seg / math.ceil(seg / h) for h in hs]
                order = [
                    math.log(err[k] / err[k + 1]) / math.log(dx[k] / dx[k + 1])
                    for k in range(len(hs) - 1)
                ]
                assert all(1.9 <= q <= 2.1 for q in order), (anchor, name, order)
                rows[(anchor, name)] = (dx, err)

        def figure():
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            for (anchor, name), (dx, err) in rows.items():
                ls = '-' if anchor == 'surface' else '--'
                mk = 'o' if name == 'T-dependent' else 's'
                where = 'surface' if anchor == 'surface' else 'CMB'
                ax.loglog(dx, err, ls=ls, marker=mk, label=f'{name}, {where} anchor')
            d = np.array(rows[('surface', 'T-dependent')][0])
            ref = rows[('surface', 'T-dependent')][1][0] * (d / d[0]) ** 2
            ax.loglog(
                d, ref / 4, ls=':', color=plt.rcParams['text.color'], alpha=0.5, label='slope 2'
            )
            ax.minorticks_off()
            ax.set_xlabel(r'Substep $\Delta \ln P$')
            ax.set_ylabel('Largest relative error in T')
            legend_below(fig, ncol=2)
            return assert_text_clear(fig)

        validation_plot(figure, 'adiabat_midpoint_order')

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize('anchor', ['surface', 'cmb'])
    def test_default_call_uses_the_0p0025_substeps(self, monkeypatch, anchor):
        """A default call evaluates nabla twice per substep, with ceil(|d ln P| / 0.0025)
        substeps per segment above 1e5 Pa: the count pins the default h, because the 1 K bound
        of the real profiles also admits a larger h (0.005 gives 0.50 K, 0.0055 gives 1.29 K)."""
        calls = []
        _stub_adiabat(monkeypatch, lambda P, T: 0.1, calls)  # T stays inside the node caps
        P = np.geomspace(3.6e11, 3e4, 40)  # the top segment crosses 1e5 Pa
        _call(P, anchor, 3000.0)
        lnp = np.log(np.maximum(P, P_FLOOR))
        first = _anchor_node(anchor, len(P))
        segs = range(first) if anchor == 'surface' else range(first, len(P) - 1)
        expected = sum(2 * math.ceil(abs(lnp[i] - lnp[i + 1]) / H_DEFAULT) for i in segs)
        assert np.sum(P < P_FLOOR) >= 2 and len(calls) == expected
        assert min(calls) >= P_FLOOR  # no nabla read below the floor

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize('anchor', ['surface', 'cmb'])
    def test_low_pressure_rule_follows_the_floored_adiabat(self, monkeypatch, anchor):
        """With a top pressure from 0.5e5 to 2e5 Pa the profile is the adiabat with T held below
        1e5 Pa (analytical for constant nabla 0.1), so it is continuous in the top pressure.
        A rule that skips the whole top segment once an end is below 1e5 Pa jumps by
        more than 10 % at the next node."""
        _stub_adiabat(monkeypatch, lambda P, T: 0.1)
        P0 = _profile(n=21, p_top=2e5)
        a = _anchor_node(anchor, len(P0))
        prev = None
        for p_top in np.linspace(0.5e5, 2e5, 31):
            P = P0.copy()
            P[-1] = p_top
            T = _call(P, anchor, 3000.0)
            Pf = np.maximum(P, P_FLOOR)
            exact = 3000.0 * (Pf / Pf[a]) ** 0.1
            done = _integrated(anchor)
            np.testing.assert_allclose(T[done], exact[done], rtol=1e-12, atol=0)
            if prev is not None:
                assert abs(T[-2] - prev) / prev < 0.02  # 31 samples: no jump
            prev = T[-2]
            if p_top < P_FLOOR:  # the top segment still takes its step down to the floor
                assert abs(T[-2] / T[-1] - 1) > 0.1

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize('anchor', ['surface', 'cmb'])
    def test_padded_top_integrates_like_the_floor(self, monkeypatch, anchor):
        """A padded top (P = 0 at the outer node) integrates like a top at 1e5 Pa: the profile
        equals the one with the top at 1 Pa and at 1e5 Pa, bit for bit, and with the surface
        anchor it is the constant-nabla law from 1e5 Pa (analytical), not an isothermal
        shell."""
        _stub_adiabat(monkeypatch, lambda P, T: 0.1)
        P = _profile(n=21)
        runs = {}
        for top in (0.0, 1.0, 1e5):
            P[-1] = top
            runs[top] = _call(P, anchor, 3000.0)
        np.testing.assert_array_equal(runs[0.0], runs[1.0])
        np.testing.assert_array_equal(runs[0.0], runs[1e5])
        T = runs[0.0]
        if anchor == 'surface':
            assert T[-1] == 3000.0 and T[-2] > 3000.0
            np.testing.assert_allclose(T[:-1], 3000.0 * (P[:-1] / 1e5) ** 0.1, rtol=1e-12)
        else:
            assert T[-1] < T[-2] < T[1]

    @pytest.mark.physics_invariant
    def test_negative_nabla_holds_the_rest_of_the_segment(self, monkeypatch):
        """nabla <= 0 or None at a substep holds T for the rest of that segment; the next
        segment integrates again, so T never falls inward."""
        _stub_adiabat(monkeypatch, lambda P, T: None if 1e9 < P < 1e10 else 0.1)
        P = _profile()
        T = _call(P, 'surface', 3000.0)
        assert np.all(np.diff(T) <= 0)
        held = (P[:-1] < 1e10) & (P[1:] > 1e9)  # segments read only inside the gap
        assert np.any(held) and np.all(T[:-1][held] == T[1:][held])
        assert T[0] > T[np.argmax(P < 1e9)]

    @pytest.mark.parametrize(
        'bad', [True, False, 'x', None, 0.0, 1e-300, 9e-5, -1e-3, np.nan, np.inf, -np.inf, 1.5]
    )
    def test_max_dlnp_rejects_values_outside_its_range(self, monkeypatch, bad):
        """``max_dlnp`` must be a real number from 1e-4 to 1; a bool, a string, None, 0, a value
        below 1e-4, a negative value, NaN, an infinity or a value above 1 raises ValueError
        before any nabla read."""
        calls = []
        _stub_adiabat(monkeypatch, lambda P, T: 0.3, calls)
        P = _profile(n=5)
        with pytest.raises(ValueError, match='max_dlnp'):
            _call(P, 'surface', 3000.0, max_dlnp=bad)
        assert calls == []

    @pytest.mark.physics_invariant
    def test_broken_step_raises_the_profile_invariant(self, monkeypatch):
        """A segment result that is not finite or cools inward violates the profile invariant
        (T finite, positive and non-increasing outward from the anchor) and raises."""
        import zalmoxis.eos.temperature as zt
        from zalmoxis.invariants import StructureInvariantError

        _stub_adiabat(monkeypatch, lambda P, T: 0.1)
        P = _profile(n=11)
        good = _call(P, 'surface', 3000.0)
        assert np.all(np.diff(good) < 0)
        monkeypatch.setattr(zt, '_segment_temperature', lambda T0, *a, **k: 0.5 * T0)
        with pytest.raises(StructureInvariantError, match='non-increasing'):
            _call(P, 'surface', 3000.0)
        monkeypatch.setattr(zt, '_segment_temperature', lambda T0, *a, **k: T0 - 1e-6)
        with pytest.raises(StructureInvariantError, match='non-increasing'):
            _call(P, 'surface', 3000.0)  # a rise of 1e-6 K outward
        monkeypatch.setattr(zt, '_segment_temperature', lambda T0, *a, **k: np.nan)
        with pytest.raises(StructureInvariantError, match='finite'):
            _call(P, 'cmb', 3000.0)

    @pytest.mark.parametrize('anchor', ['surface', 'cmb'])
    def test_anchor_temperature_property_over_its_domain(self, monkeypatch, anchor):
        """Hypothesis over floats, bools, None and the range edges: a real number from above 0
        (surface) or from 100 K (CMB) to 1e5 K runs and the anchor node keeps it; anything else,
        a TOML true included, raises ValueError that names the field, before any nabla read."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        calls = []
        _stub_adiabat(monkeypatch, lambda P, T: 0.1, calls)
        P, lo = _profile(n=5), 0.0 if anchor == 'surface' else 100.0
        edges = [lo, np.nextafter(lo, 1.0), np.nextafter(lo, -1.0), 1e5, np.nextafter(1e5, 2e5)]

        @settings(max_examples=200, deadline=None, derandomize=True)
        @given(st.one_of(st.floats(), st.booleans(), st.none(), st.sampled_from(edges)))
        def check(v):
            calls.clear()
            ok = (
                isinstance(v, float)
                and (lo < v if anchor == 'surface' else lo <= v)
                and v <= 1e5
            )
            if ok:
                assert _call(P, anchor, v)[_anchor_node(anchor, 5)] == v
            else:
                with pytest.raises(ValueError, match=f'{anchor}_temperature'):
                    _call(P, anchor, v)
                assert calls == []

        check()

    @pytest.mark.parametrize('bad', ['CMB', 'Surface', 'top', 'bottom', 'bogus', '', None])
    def test_an_unknown_anchor_raises_value_error(self, monkeypatch, bad):
        """``anchor`` is 'surface' or 'cmb'; any other value raises ValueError before any nabla
        read, instead of running the surface anchor."""
        calls = []
        _stub_adiabat(monkeypatch, lambda P, T: 0.1, calls)
        with pytest.raises(ValueError, match='anchor'):
            _call(_profile(n=5), bad, 3000.0, surface_temperature=3000.0)
        assert calls == []

    def test_the_binodal_scale_reaches_the_nabla_read(self, monkeypatch):
        """``binodal_T_scale`` is passed on to ``get_mixed_nabla_ad`` at every read, also when it
        differs from the default of that function."""
        import inspect

        import zalmoxis.mixing as _mix

        sig, scales = inspect.signature(_mix.get_mixed_nabla_ad), []

        def nabla(P, T, *a, **k):
            bound = sig.bind(P, T, *a, **k)
            bound.apply_defaults()
            scales.append(bound.arguments['binodal_T_scale'])
            return 0.1

        _stub_adiabat(monkeypatch, lambda P, T: 0.1)
        monkeypatch.setattr(_mix, 'get_mixed_nabla_ad', nabla)
        _call(_profile(n=5), 'surface', 3000.0, binodal_T_scale=75.0)
        assert scales and set(scales) == {75.0}

    def test_a_subnormal_anchor_runs_without_a_warning(self, monkeypatch):
        """The smallest positive anchor temperature (a subnormal float) is in range: the call
        runs with no RuntimeWarning (an overflow in T_lim / T0) and the anchor node keeps it."""
        import warnings

        _stub_adiabat(monkeypatch, lambda P, T: 0.1)
        t = np.nextafter(0.0, 1.0)
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            T = _call(_profile(n=5), 'surface', t)
        assert T[-1] == t and np.all(T > 0)

    @pytest.mark.physics_invariant
    def test_a_tiny_anchor_follows_the_closed_form_or_stops_at_the_cap_property(self):
        """Hypothesis over anchors T0 from 5e-324 K to 1e-290 K and a constant nabla from 0.1 to
        inf, inward from 1e5 to 4e11 Pa: no OverflowError and a positive result; ln T equals
        ln T0 + nabla ln(4e6) within 1e-6 where T is a normal float, also where that needs the
        log-space product (nabla 47 and 48), within 0.2 for a subnormal T (one rounding to a
        multiple of 5e-324 at T >= 2.3e-323), and T is exactly 1e5 K where the closed form
        exceeds the cap by more than 1e-6 in ln T."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from zalmoxis.eos.temperature import _segment_temperature

        @settings(max_examples=300, deadline=None, derandomize=True)
        @given(
            st.floats(-323.3, -290.0),
            st.sampled_from([0.1, 3.0, 47.0, 48.0, 50.0, 1e6, math.inf]),
        )
        def check(log10_T0, nabla):
            T0 = 10.0**log10_T0
            try:
                t = _segment_temperature(T0, 1e5, 4e11, lambda P, T: nabla, 0.0025, True)
            except OverflowError as e:
                pytest.fail(f'T0 {T0}, nabla {nabla}: {e}')
            y = math.log(T0) + nabla * math.log(4e6)  # ln of the closed form
            assert t > 0, (T0, nabla, t)
            if y > math.log(1e5) + 1e-6:
                assert t == 1e5, (T0, nabla, t)
            else:  # a subnormal T keeps few bits
                tol = 1e-6 if t >= np.finfo(float).tiny else 0.2
                assert abs(math.log(t) - min(y, math.log(1e5))) <= tol, (T0, nabla, t)

        check()

    def test_segment_end_rule(self):
        """A NaN or +inf end returns T0 exactly; a -inf, negative or zero end gives the result
        of an end at 1e5 Pa, inward and outward; a segment against the anchor direction returns
        T0."""
        from zalmoxis.eos.temperature import _segment_temperature

        def seg(P0, P1, inward):
            try:
                return _segment_temperature(3000.0, P0, P1, lambda P, T: 0.1, 0.0025, inward)
            except Exception as e:  # noqa: BLE001
                pytest.fail(f'{P0} -> {P1}: {e!r}')

        for end in (math.nan, math.inf):
            assert seg(end, 1e9, True) == seg(1e9, end, False) == 3000.0, end
            assert seg(1e9, end, True) == seg(end, 1e9, False) == 3000.0, end
        assert seg(1e9, 1e10, False) == seg(1e10, 1e9, True) == 3000.0
        for end in (-math.inf, -5.0, 0.0):
            assert seg(end, 1e9, True) == seg(1e5, 1e9, True) != 3000.0, end
            assert seg(1e9, end, False) == seg(1e9, 1e5, False) != 3000.0, end

    def test_non_finite_surface_temperature_with_the_cmb_anchor_raises(self, monkeypatch):
        """With the CMB anchor the core nodes carry the surface temperature, so it too must be
        finite and positive: NaN and True raise ValueError."""
        _stub_adiabat(monkeypatch, lambda P, T: 0.1)
        for v in (np.nan, True):
            with pytest.raises(ValueError, match='surface_temperature'):
                _call(_profile(n=5), 'cmb', 3000.0, surface_temperature=v)

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize('nabla', [1e6, np.inf])
    def test_a_huge_nabla_stops_at_the_node_caps(self, monkeypatch, nabla):
        """A nabla far above any table value (1e6, inf) gives no overflow: inward T stops at the
        1e5 K cap and outward (CMB anchor) at the 100 K floor, in every substep and every read
        of nabla, and the anchor keeps its value."""
        from zalmoxis.eos.temperature import _segment_temperature

        reads = []
        _stub_adiabat(monkeypatch, lambda P, T: reads.append(T) or nabla)
        P = _profile(n=11)

        def call(anchor):
            try:
                return _call(P, anchor, 3000.0)
            except OverflowError as e:
                pytest.fail(f'nabla {nabla}, anchor {anchor}: {e}')

        T = call('surface')
        assert T[-1] == 3000.0 and max(reads) <= 1e5
        np.testing.assert_array_equal(T[:-1], 1e5)
        reads.clear()
        T = call('cmb')
        assert T[1] == 3000.0 and min(reads) >= 100.0
        np.testing.assert_array_equal(T[2:], 100.0)
        for T0, P0, P1, inward, lim in [
            (3000.0, 1e9, 1e12, True, 1e5),
            (3000.0, 1e10, 1e6, False, 100.0),
        ]:
            reads.clear()
            try:
                t = _segment_temperature(
                    T0, P0, P1, lambda P, T: reads.append(T) or nabla, 0.0025, inward
                )
            except OverflowError as e:
                pytest.fail(f'nabla {nabla}, inward {inward}: {e}')
            assert t == lim and len(reads) == 2, (inward, t, len(reads))
            assert all(min(lim, T0) <= r <= max(lim, T0) for r in reads)

    @pytest.mark.physics_invariant
    def test_a_segment_stops_exactly_at_its_limit_property(self):
        """Hypothesis over the anchor T0 from 100 K to 1e5 K, both directions and nabla 0.3, 50,
        1e6 and inf: a segment that reaches the 1e5 K cap (inward) or the 100 K floor (outward)
        returns that limit exactly, after the reads up to the substep that reaches it; one that
        does not reach it gives T0 (P1/P0)^nabla; every read of nabla lies in [100, 1e5] K."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from zalmoxis.eos.temperature import _segment_temperature

        @settings(max_examples=300, deadline=None, derandomize=True)
        @given(
            st.floats(100.0, 1e5),
            st.booleans(),
            st.sampled_from([0.3, 50.0, 1e6, math.inf]),
        )
        def check(T0, inward, nabla):
            reads = []
            (P0, P1), lim = ((1e9, 1e12), 1e5) if inward else ((1e10, 1e6), 100.0)
            t = _segment_temperature(
                T0, P0, P1, lambda P, T: reads.append(T) or nabla, 0.0025, inward
            )
            assert all(100.0 <= r <= 1e5 for r in reads), (T0, inward, nabla)
            n = math.ceil(abs(math.log(P1 / P0)) / 0.0025)
            dx, gap = math.log(P1 / P0) / n, math.log(lim / T0)  # ln T change to the limit
            if abs(nabla * math.log(P1 / P0)) > abs(gap) * (1 + 1e-9):  # the limit is reached
                assert t == lim, (T0, inward, nabla, t)
                assert abs(len(reads) - 2 * math.ceil(gap / (nabla * dx))) <= 2
            elif abs(nabla * math.log(P1 / P0)) < abs(gap) * (1 - 1e-9):
                assert t == pytest.approx(T0 * (P1 / P0) ** nabla, rel=1e-12)

        check()
        reads = []  # dy 1 ulp below the cap, where T0 exp(dy) rounds above 1e5 K on macOS libm
        k = iter([1e-9, 400.11264147477243])
        _segment_temperature(
            36789.5385, 1e9, 1e12, lambda P, T: reads.append(T) or next(k, 1e-9), 0.0025, True
        )
        assert len(reads) > 2 and max(reads) <= 1e5

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize(
        'bad', [None, 0.0, -1.0, np.nan], ids=['none', 'zero', 'neg', 'nan']
    )
    def test_a_segment_holds_from_the_first_bad_nabla(self, bad):
        """_segment_temperature holds T from the first substep with a nabla that is missing or
        not positive: a bad first read keeps T0, also when the reads after it are good; a bad
        midpoint read after 1 good substep keeps the T of that substep (T0 exp(0.1 dx)); a
        non-finite end pressure returns T0 unread."""
        from zalmoxis.eos.temperature import _segment_temperature

        calls = []

        def nabla(P, T):  # good for substep 1 and the start of substep 2, bad after
            calls.append(P)
            return 0.1 if len(calls) <= 3 else bad

        assert _segment_temperature(3000.0, 1e9, 1e10, lambda P, T: bad, 0.0025, True) == 3000.0
        dx = math.log(10.0) / math.ceil(math.log(10.0) / 0.0025)
        T = _segment_temperature(3000.0, 1e9, 1e10, nabla, 0.0025, True)
        assert T == pytest.approx(3000.0 * math.exp(0.1 * dx), rel=1e-12) and len(calls) == 4
        assert _segment_temperature(3000.0, 1e9, np.nan, nabla, 0.0025, True) == 3000.0
        assert len(calls) == 4
        first = iter([bad])  # bad at the first read only, good after
        assert (
            _segment_temperature(3000.0, 1e9, 1e10, lambda P, T: next(first, 0.1), 0.0025, True)
            == 3000.0
        )

    @pytest.mark.physics_invariant
    def test_max_dlnp_property_over_its_domain(self, monkeypatch):
        """Hypothesis over floats: from 1e-4 to 1 the profile is finite, positive and
        non-increasing outward and within 1e-12 of the constant-nabla law; outside it, and for
        NaN and the infinities, the call raises ValueError."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        _stub_adiabat(monkeypatch, lambda P, T: 0.1)
        P = _profile(n=6, p_center=1e7, p_top=2e5)
        exact = 3000.0 * (P / P[-1]) ** 0.1

        @settings(max_examples=200, deadline=None)
        @given(
            st.one_of(
                st.floats(allow_nan=True, allow_infinity=True),
                st.sampled_from([1e-4, np.nextafter(1e-4, 0.0), 1.0, np.nextafter(1.0, 2.0)]),
            )
        )
        def check(h):
            if 1e-4 <= h <= 1.0:
                try:
                    T = _call(P, 'surface', 3000.0, max_dlnp=h)
                except ValueError as e:
                    pytest.fail(f'max_dlnp {h!r} is in range: {e}')
                assert np.all(np.isfinite(T)) and np.all(np.diff(T) <= 0)
                np.testing.assert_allclose(T, exact, rtol=1e-12)
            else:
                with pytest.raises(ValueError, match='max_dlnp'):
                    _call(P, 'surface', 3000.0, max_dlnp=h)

        check()


@pytest.mark.slow
@pytest.mark.paleos_adiabat
@pytest.mark.physics_invariant
@pytest.mark.timeout(1800)
def test_padded_top_shell_lookup_follows_the_adiabat(monkeypatch):
    """A padded default solve (P = 0 at the outer node): the T(P) lookup that the last density
    solve reads gives T_surf at 1e5 Pa and lies within 0.3 K of an independent explicit Euler
    adiabat (4000 steps in ln P) from there up to the last positive node of the pass that built
    the table, instead of holding the T of that node over the top shell. Above that node the
    lookup joins the returned top node, which carries the one-pass table lag."""
    import os

    import zalmoxis.solver as zs
    from tests.test_surface_node import euler_adiabat, spy_temperature_function
    from tools.validation import adiabat_profiles as ap
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import load_material_dictionaries

    root = get_zalmoxis_root()
    if not os.path.isdir(os.path.join(root, 'data', 'EOS_PALEOS_MgSiO3_unified')):
        pytest.skip('PALEOS tables not on disk')
    seen, tables, table = spy_temperature_function(monkeypatch), [], zs._adiabat_table

    def spy_table(pressure, *args, **kwargs):
        tables.append(np.asarray(pressure, float).copy())
        return table(pressure, *args, **kwargs)

    monkeypatch.setattr(zs, '_adiabat_table', spy_table)
    cfg = ap._default_config(0)
    melt = ap._melt('config', cfg)
    res = zs.main(cfg, load_material_dictionaries(), melt, os.path.join(root, 'input'))
    r, P, T = (np.asarray(res[k], float) for k in ('radii', 'pressure', 'temperature'))
    assert res['padded'] and P[-1] == 0.0
    assert T[-1] == pytest.approx(cfg['surface_temperature'])
    top = int(np.flatnonzero(P > 0)[-1])
    assert tables, 'the solve built no T(P) table'
    q, t_path = euler_adiabat(cfg, melt, P_FLOOR, T[-1], np.min(tables[-1][tables[-1] > 0]))
    look = np.array([seen[-1](r[top], p) for p in q])
    assert look[0] == pytest.approx(T[-1], abs=1e-9)
    assert np.max(np.abs(look - t_path)) < 0.3


@pytest.mark.unit
class TestAdiabatTable:
    """``_adiabat_table``: the sorted log10 P, T lookup of the density solve, with the adiabat
    over the top cell: from 1e5 Pa up to the last positive node of a padded top, from the top
    node (at least 1e5 Pa) up to the next node of an unpadded one."""

    @staticmethod
    def _shell(calls):
        def top_shell(q, T_top, m):
            calls.append((q.copy(), T_top, m))
            return T_top * (q / q[0]) ** 0.1

        return top_shell

    @staticmethod
    def _padded_rule(pressure, T, mass, top_shell):
        """The table with a shell on a padded top only, at the mass of the lowest positive
        node."""
        from zalmoxis.solver import _TOP_SHELL_POINTS

        s = np.argsort(pressure)
        keep = pressure[s] > 0
        P, Ts = pressure[s][keep], T[s][keep]
        if pressure[-1] <= 0 and P.size and P[0] > P_FLOOR * (1 + 1e-9):
            q = np.geomspace(P_FLOOR, P[0], _TOP_SHELL_POINTS + 1)[:-1]
            shell = top_shell(q, T[-1], mass[s[keep][0]])
            P, Ts = np.concatenate([q, P]), np.concatenate([shell, Ts])
        return np.log10(P), Ts

    @pytest.mark.parametrize(
        'P',
        [
            [1e10, 9e4, 5e4],
            [1e10, 1e8, P_FLOOR, 0.0],
            [1e10, 1e8, 5e4, 0.0],
            [1e10, P_FLOOR * (1 + 1e-10), 0.0],
            np.zeros(3),
        ],
        ids=[
            'unpadded_2_below_floor',
            'last_at_floor',
            'last_below_floor',
            'last_within_1e-9',
            'all_padded',
        ],
    )
    def test_no_shell_without_a_top_cell_above_the_floor(self, P):
        """An unpadded top whose next node also lies below 1e5 Pa, a padded top whose last
        positive node lies at, below or within 1e-9 of 1e5 Pa, or no node at P > 0: the nodes
        at P > 0 sorted by P, bit for bit, and no shell call."""
        from zalmoxis.solver import _adiabat_table

        calls, P = [], np.asarray(P, float)
        T = 3000.0 + np.arange(P.size, dtype=float)[::-1]
        lp, t = _adiabat_table(P, T, np.arange(P.size), self._shell(calls))
        np.testing.assert_array_equal(lp, np.log10(P[P > 0][::-1]))
        np.testing.assert_array_equal(t, T[P > 0][::-1])
        assert calls == []

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize('p_top', [1.2e5, 9.09e4])
    def test_an_unpadded_top_cell_gains_the_shell(self, p_top):
        """An unpadded top: the cell from max(P_top, 1e5 Pa) to the next node gains the shell
        points, from the T of the top node, at the mass of the next node; a top node below
        1e5 Pa stays in the table with its T, which the adiabat holds below 1 bar, and the
        lookup over the cell is the constant-nabla law from the start of the cell."""
        from zalmoxis.solver import _TOP_SHELL_POINTS, _adiabat_table

        calls, P = [], np.concatenate([np.geomspace(3.6e11, 1.1e9, 9), [p_top]])
        lo = max(p_top, P_FLOOR)
        T = 3000.0 * (np.maximum(P, P_FLOOR) / P_FLOOR) ** 0.1
        lp, t = _adiabat_table(P, T, np.arange(P.size), self._shell(calls))
        ((q, T_top, i),) = calls
        assert T_top == T[-1] and i == 8 and q.size == _TOP_SHELL_POINTS and q[0] == lo
        below = p_top < P_FLOOR  # the node below 1 bar keeps its place in the table
        assert lp.size == q.size + 9 + below and np.all(np.diff(lp) > 0)
        assert (lp[0], t[0]) == (math.log10(p_top), T[-1])  # lo is p_top at or above the floor
        x = np.geomspace(lo, 1.1e9, 200)
        np.testing.assert_allclose(
            np.interp(np.log10(x), lp, t), T[-1] * (x / lo) ** 0.1, rtol=1e-4
        )

    @pytest.mark.parametrize(
        'p_top',
        [
            P_FLOOR,
            101325.0,
            P_FLOOR * (1 + 1e-9),
            P_FLOOR * (1 - 1e-9),
            np.nextafter(P_FLOOR, 0),
        ],
    )
    def test_an_unpadded_top_at_or_near_the_floor(self, p_top):
        """A top node at, just above or just below 1e5 Pa (down to 1 ULP): one shell from
        max(P_top, 1e5 Pa), the top node first in the table (as the first shell point, or kept
        below the floor when its log10 P lies below 5), and log10 P strictly increasing."""
        from zalmoxis.solver import _adiabat_table

        calls, P = [], np.array([1e10, 1e8, p_top])
        lp, t = _adiabat_table(
            P, np.array([5000.0, 3500.0, 3000.0]), np.arange(3), self._shell(calls)
        )
        assert len(calls) == 1 and calls[0][0][0] == max(p_top, P_FLOOR)
        assert lp[0] == math.log10(p_top) and t[0] == 3000.0
        assert np.all(np.diff(lp) > 0)

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize(
        ('n_pad', 'outward', 'node'),
        [(0, False, 8), (0, True, 9), (1, False, 8), (1, True, 9), (2, False, 8), (2, True, 9)],
    )
    def test_the_shell_takes_the_mass_of_the_node_of_the_adiabat_over_the_cell(
        self, n_pad, outward, node
    ):
        """The adiabat gives node i its T from node j with the mixture of node i: integrated
        inward, the cell takes the node at its upper end (the next node of an unpadded top, the
        lowest positive node of a padded one); integrated outward from the CMB, the node outside
        that one (the top node, or the first padded node)."""
        from zalmoxis.solver import _adiabat_table

        nodes = [np.geomspace(3.6e11, 1.1e9, 9), [1.2e5] * (n_pad == 0), np.zeros(n_pad)]
        calls, P = [], np.concatenate(nodes)
        mass = np.linspace(1e24, 6e24, P.size)
        _adiabat_table(P, np.full(P.size, 3000.0), mass, self._shell(calls), outward)
        assert [c[2] for c in calls] == [mass[node]]

    def test_a_padded_top_keeps_the_table_of_the_padded_rule(self):
        """Hypothesis over padded profiles (P = 0 at the outer node), inward (surface anchor):
        the table is bit for bit the one of the rule that adds the shell to a padded top only,
        so padded solves with the surface anchor keep their temperatures."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        from zalmoxis.solver import _adiabat_table

        node = st.one_of(st.floats(1.0, 1e13), st.sampled_from([P_FLOOR, 2e5]))

        @settings(max_examples=200, deadline=None)
        @given(st.lists(node, min_size=1, max_size=12, unique=True), st.integers(1, 3))
        def check(nodes, n_pad):
            P = np.concatenate([np.sort(nodes)[::-1], np.zeros(n_pad)])
            T = 1000.0 + np.arange(P.size, dtype=float)
            got = _adiabat_table(P, T, np.arange(P.size), self._shell([]))
            want = self._padded_rule(P, T, np.arange(P.size), self._shell([]))
            for a, b in zip(got, want):
                np.testing.assert_array_equal(a, b)

        check()

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize('n_pad', [1, 2])
    def test_padded_top_shell_follows_the_adiabat(self, n_pad):
        """P = 0 at the outer node(s): the table gains the shell points from 1e5 Pa (T of the
        padded node there) up to below the last positive node, geometric, and the lookup is
        the constant-nabla law over the shell instead of the T of that node."""
        from zalmoxis.solver import _TOP_SHELL_POINTS, _adiabat_table

        calls, P = [], np.concatenate([np.geomspace(3.6e11, 1.1e9, 9), np.zeros(n_pad)])
        T = np.where(P > 0, 3000.0 * (np.maximum(P, P_FLOOR) / P_FLOOR) ** 0.1, 3000.0)
        lp, t = _adiabat_table(P, T, np.arange(P.size), self._shell(calls))
        assert len(calls) == 1
        ((q, T_top, i),) = calls
        assert T_top == pytest.approx(3000.0) and i == 8
        assert q.size == _TOP_SHELL_POINTS and q[0] == P_FLOOR
        np.testing.assert_allclose(q, np.geomspace(P_FLOOR, 1.1e9, q.size + 1)[:-1], rtol=1e-15)
        assert lp.size == q.size + 9 and np.all(np.diff(lp) > 0)
        np.testing.assert_array_equal(t[q.size :], T[:9][::-1])
        x = np.geomspace(P_FLOOR, 1.1e9, 200)
        np.testing.assert_allclose(
            np.interp(np.log10(x), lp, t), 3000.0 * (x / P_FLOOR) ** 0.1, rtol=1e-4
        )

    @pytest.mark.physics_invariant
    def test_the_shell_points_bound_the_lookup_error_of_a_steep_wide_shell(self):
        """A shell over 6 decades (1e5 to 1e11 Pa) with nabla 0.3: T linear in log10 P between
        the 64 shell points errs by at most (nabla h)^2 / 8 = 5.2e-4 (relative), h the ln P
        step; the window 4e-4 to 6e-4 holds from 60 to 72 points (6.9e-4 at 56, 1.3e-4 at 128).
        """
        from zalmoxis.solver import _adiabat_table

        P = np.concatenate([np.geomspace(3.6e11, 1e11, 4), [0.0]])
        T = np.where(P > 0, 3000.0 * (np.maximum(P, P_FLOOR) / P_FLOOR) ** 0.3, 3000.0)
        lp, t = _adiabat_table(
            P, T, np.arange(P.size), lambda q, T_top, m: T_top * (q / P_FLOOR) ** 0.3
        )
        x = np.geomspace(P_FLOOR, 1e11, 4001)
        err = np.interp(np.log10(x), lp, t) / (3000.0 * (x / P_FLOOR) ** 0.3) - 1.0
        assert 4e-4 < np.max(np.abs(err)) <= 6e-4, np.max(np.abs(err))

    @pytest.mark.physics_invariant
    def test_the_shell_adiabat_runs_inward_from_the_top_in_the_layer_of_its_node(self):
        """``_shell_adiabat``: the adiabat gets the shell pressures centre first, each point at
        the enclosed mass of the node below the shell, and the top T at the outer point; the T
        comes back in the order of q. With a constant-nabla adiabat (nabla 0.1) on a padded
        structure, the table over the shell is T_top (P / 1e5 Pa)^0.1 from the padded node."""
        from zalmoxis.solver import _adiabat_table, _shell_adiabat

        seen = []

        def adiabat(radii, pressure, mass, top):
            seen.append((radii, pressure, mass, top))
            return top * (pressure / pressure[-1]) ** 0.1

        P = np.concatenate([np.geomspace(3.6e11, 1.1e9, 9), [0.0]])
        mass = np.linspace(1e24, 6e24, P.size)
        T = np.where(P > 0, 3000.0 * (np.maximum(P, P_FLOOR) / P_FLOOR) ** 0.1, 3000.0)
        lp, t = _adiabat_table(
            P, T, mass, lambda q, T_top, m: _shell_adiabat(adiabat, q, T_top, m)
        )
        ((radii, pressure, m, top),) = seen
        assert pressure[0] > pressure[-1] == P_FLOOR and top == 3000.0
        np.testing.assert_array_equal(radii, np.arange(pressure.size))
        assert np.all(m == mass[8])  # the last node at P > 0
        x = np.geomspace(P_FLOOR, 1.1e9, 200)
        np.testing.assert_allclose(
            np.interp(np.log10(x), lp, t), 3000.0 * (x / P_FLOOR) ** 0.1, rtol=1e-4
        )
        np.testing.assert_allclose(
            t[: pressure.size], 3000.0 * (pressure[::-1] / P_FLOOR) ** 0.1
        )

    def test_table_property_over_profiles(self):
        """Hypothesis over node pressures (zero, negative, NaN, positive, just above 1e5 Pa)
        with a padded or an unpadded top: log10 P finite and not decreasing, strictly
        increasing over the shell points, T paired with its node, and shell points exactly when
        the top cell (padded: 1e5 Pa to the lowest positive node; unpadded: the lowest positive
        node, at least 1e5 Pa, to the next) is wider than 1e-9, starting at the T of the padded
        or the top node, at the mass of the deeper node of the cell or, ``outward``, of the next
        node in the node order (the last node at most); a top node below 1e5 Pa stays in the
        table."""
        from hypothesis import example, given, settings
        from hypothesis import strategies as st

        from zalmoxis.solver import _adiabat_table

        node = st.one_of(st.floats(1.0, 1e13), st.sampled_from([0.0, -1.0, np.nan, P_FLOOR]))

        @settings(max_examples=200, deadline=None)
        @given(st.lists(node, min_size=1, max_size=12, unique=True), st.booleans())
        @example([1e10, float(np.nextafter(P_FLOOR, np.inf)), 0.0], False)
        @example([1e10, P_FLOOR * (1 + 1e-14), 0.0], False)
        @example([1e10, P_FLOOR * (1 + 1e-7), 0.0], False)
        @example([1e13, 9999999999999.984], False)
        @example([1e10, 1e8, 9.09e4], False)
        @example([1e10, 9e4, 5e4], False)
        @example([1e10, 1e8, P_FLOOR * (1 - 1e-9)], False)
        @example([1e10, 1e8, 2e5, 3e5], True)  # not in node order: the outer node is the last
        def check(P, outward):
            P = np.array(P)
            T = 1000.0 + np.arange(P.size, dtype=float)
            calls = []
            lp, t = _adiabat_table(P, T, np.arange(P.size), self._shell(calls), outward)
            assert np.all(np.isfinite(lp)) and np.all(np.diff(lp) >= 0)
            order = np.argsort(P)
            srt, t_srt = P[order][P[order] > 0], T[order][P[order] > 0]
            padded = P[-1] <= 0
            lo = P_FLOOR if padded else max(srt[0], P_FLOOR) if srt.size else 0.0
            hi = srt[0] if padded and srt.size else srt[1] if srt.size > 1 else 0.0
            shell = hi > lo * (1 + 1e-9)
            assert len(calls) == int(shell)
            if not shell:
                np.testing.assert_array_equal(t, t_srt)
                return
            q, t0, node = calls[0]
            assert q[0] == lo and t0 == (T[-1] if padded else t_srt[0])
            deeper = order[P[order] > 0][0 if padded else 1]  # deeper node of the cell
            assert node == min(deeper + outward, P.size - 1)
            kept = int(np.log10(srt[0]) < np.log10(P_FLOOR))  # a top node below 1 bar
            tail = t_srt if padded else t_srt[1:]
            assert lp.size == q.size + tail.size + kept
            assert np.all(np.diff(lp[kept : kept + q.size + 1]) > 0)
            np.testing.assert_array_equal(t[kept + q.size :], tail)

        check()


@pytest.mark.unit
class TestLayerTables:
    """The T(P) table of a pass split at its CMB (``_layer_tables``) and its lookup
    (``_table_lookup``): each layer reads its own nodes, and past an end the adiabat of the end
    interval goes on (``_table_end``) instead of a constant T."""

    LP, TS = np.log10([1e11, 2e11]), np.array([4000.0, 5000.0])

    @pytest.mark.physics_invariant
    @pytest.mark.reference_pinned
    def test_the_ends_follow_the_power_law_of_their_end_interval(self):
        """T = T_end (P / P_end)^n with n of the end interval (analytical law): 3200 K at
        5e10 Pa below (1e11 Pa, 4000 K), (2e11 Pa, 5000 K), 6250 K at 4e11 Pa above it. A table of
        1 point, or with a flat end interval, holds its end T."""
        from zalmoxis.solver import _table_end

        assert _table_end(self.LP, self.TS, top=False)(math.log10(5e10)) == pytest.approx(
            3200.0, rel=1e-12
        )
        assert _table_end(self.LP, self.TS)(math.log10(4e11)) == pytest.approx(
            6250.0, rel=1e-12
        )
        assert _table_end(self.LP[:1], self.TS[:1])(13.0) == 4000.0
        assert _table_end(np.array([11.0, 11.0]), self.TS)(13.0) == 5000.0

    def test_the_lookup_continues_past_the_bottom_only_when_asked(self):
        """Inside the table T is linear in log10 P; past the top the end adiabat goes on; past
        the bottom T holds the bottom value, or with ``low`` follows the end adiabat; T stays in
        100 K to 1e5 K."""
        from zalmoxis.solver import _table_lookup

        mid = 10 ** np.mean(self.LP)
        assert _table_lookup(self.LP, self.TS)(mid) == pytest.approx(4500.0, rel=1e-12)
        assert _table_lookup(self.LP, self.TS)(4e11) == pytest.approx(6250.0, rel=1e-12)
        assert _table_lookup(self.LP, self.TS)(5e10) == 4000.0
        assert _table_lookup(self.LP, self.TS, low=True)(5e10) == pytest.approx(
            3200.0, rel=1e-12
        )
        assert _table_lookup(self.LP, self.TS, low=True)(1.0) == 100.0
        assert _table_lookup(self.LP, self.TS)(1e30) == 100000.0

    def test_each_layer_reads_its_own_nodes_past_a_moved_cmb(self):
        """The mantle table holds the mantle nodes only, the core part the core nodes sorted by
        P; a mantle query at a pressure of the core of the table's pass reads the mantle end
        adiabat, not a core temperature. With fewer than 2 nodes in a layer the core part is
        None and the mantle part is the table of all nodes."""
        from zalmoxis.solver import _adiabat_table, _layer_tables, _table_lookup

        p = np.geomspace(3.6e11, 1e5, 20)
        m = np.linspace(1e23, M_E, 20)
        t = np.where(np.arange(20) < 8, 9000.0, 3000.0) * (p / 1e5) ** 0.01
        cmb = 0.5 * (m[7] + m[8])
        shell = lambda q, t_top, m_: t_top * (q / q[0]) ** 0.1  # noqa: E731
        mantle, core = _layer_tables(p, t, m, cmb, shell)
        assert mantle[1].max() < 9000.0 and core[1].min() >= 9000.0
        np.testing.assert_array_equal(core[0], np.log10(p[:8][::-1]))
        np.testing.assert_array_equal(core[1], t[:8][::-1])
        lp, ts = mantle
        q = p[3]  # a core pressure of the table's pass
        n = math.log(ts[-1] / ts[-2]) / (math.log(10) * (lp[-1] - lp[-2]))
        assert _table_lookup(*mantle)(q) == pytest.approx(
            ts[-1] * (q / 10 ** lp[-1]) ** n, rel=1e-12
        )
        whole = _adiabat_table(p, t, m, shell)
        for c in (0.5 * m[0], 0.5 * (m[-2] + m[-1])):
            mantle, core = _layer_tables(p, t, m, c, shell)
            assert core is None
            for a, b in zip(mantle, whole):
                np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_the_structure_odes_read_the_core_table_of_a_layered_t_in_the_core(monkeypatch):
    """A T function with a core table (``.core``) gives every ODE evaluation below the core mass
    the core table and every one above it the T function, as the node rule does."""
    import zalmoxis.structure_model as sm
    from zalmoxis.mixing import LayerMixture

    calls, odes = [], sm.coupled_odes

    def spy(r, y, *a, **k):
        calls.append((y[0], a[5]))
        return odes(r, y, *a, **k)

    def t_mantle(r, p):
        return 1111.0

    t_mantle.core = lambda r, p: 9999.0
    monkeypatch.setattr(
        sm, 'calculate_mixed_density', lambda p, *a, **k: 5000.0 if p > 0 else None
    )
    monkeypatch.setattr(sm, 'coupled_odes', spy)
    mix = LayerMixture(['mock:uniform'], [1.0])
    radii, cmb = np.linspace(0.0, 4e6, 40), 3e23
    sm.solve_structure(
        {'core': mix, 'mantle': mix},
        cmb,
        1e30,
        radii,
        0.98,
        1e-8,
        1e-10,
        4e5,
        {},
        {},
        [0.0, 0.0, 3e11],
        None,
        None,
        temperature_function=t_mantle,
    )
    core = np.array([sm.is_core(m_, cmb) for m_, _ in calls])
    temps = np.array([t for _, t in calls])
    assert core.any() and (~core).any()
    assert np.all(temps[core] == 9999.0) and np.all(temps[~core] == 1111.0)
