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

import numpy as np
import pytest

from zalmoxis.eos import (
    calculate_temperature_profile,
    compute_adiabatic_temperature,
)
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

    Previously, the adiabat was computed via dT/dr = -α·g·T/Cp with
    finite-difference α. In the mixed zone, phase-averaged density inflated
    α by ~100× via the latent-heat density jump, causing exponential T
    runaway through thick mantles.

    The native dT/dP table approach eliminates this entirely.
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

        This is the most extreme case: previously diverged to 58,000,000 K.
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
        """When the local or next-shell pressure is below 1e5 Pa, T is held constant.

        Discriminating: the adiabat scaling ``T*nabla/P`` blows up at low P;
        the short-circuit at lines 151-153 is what prevents NaN propagation
        in the upper atmosphere. Place the second-to-last shell below 1e5
        Pa and verify T[n-1] (computed with P_eval = pressure[n-2] < 1e5)
        falls back to T[n-2] rather than integrating with the divergent
        ratio.
        """
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

        # The shell whose own pressure is below 1e5 Pa must inherit the prior
        # shell's temperature, not integrate ``T_eval * nabla / P_eval`` which
        # would diverge or overflow the [100, 100000] clamp.
        assert T[-1] == pytest.approx(T[-2])
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
