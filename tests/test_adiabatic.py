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

from tests._plot_checks import assert_text_clear, legend_below
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
P_FLOOR = 1e5  # Pa; below it the adiabat holds T


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
    """Radius, pressure (geometric, equal ln P steps) and mass of an n-node stub planet."""
    return np.linspace(0.0, 6.4e6, n), np.geomspace(p_center, p_top, n), np.linspace(0, M_E, n)


def _call(P, anchor, T_anchor, **kw):
    """compute_adiabatic_temperature on pressure ``P`` with the anchor temperature ``T_anchor``;
    the CMB anchor sits at node 1 (cmb_mass just above the first node's mass)."""
    r, m = np.linspace(0.0, 6.4e6, len(P)), np.linspace(0, M_E, len(P))
    return compute_adiabatic_temperature(
        radii=r,
        pressure=P,
        mass_enclosed=m,
        surface_temperature=T_anchor if anchor == 'surface' else 2000.0,
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
        _, P, _ = _profile()
        T = _call(P, anchor, 3000.0)
        a = _anchor_node(anchor, len(P))
        mantle = _integrated(anchor)
        exact = 3000.0 * (P / P[a]) ** 0.1
        np.testing.assert_allclose(T[mantle], exact[mantle], rtol=1e-12, atol=0)
        euler = np.empty(len(P))
        euler[-1] = 3000.0
        for i in range(len(P) - 2, -1, -1):
            euler[i] = euler[i + 1] * (1 + 0.1 * (P[i] - P[i + 1]) / P[i + 1])
        if anchor == 'surface':
            assert abs(euler[0] / exact[0] - 1) > 0.1

    @pytest.mark.physics_invariant
    @pytest.mark.reference_pinned
    def test_midpoint_is_second_order(self, monkeypatch, validation_plot):
        """On 2 smooth nabla with closed-form adiabats the error falls as h^2 (order 1.9 to 2.1
        between successive h from 0.04 to 0.0025), both anchors: nabla = a exp(-(ln T -
        ln T_a)/2) with T = T_a (1 + a ln(P/P_a)/2)^2, and nabla = a + b cos(ln P) with
        ln T = ln T_a + a ln(P/P_a) + b (sin ln P - sin ln P_a); a = 0.1, b = 0.05. A first
        order scheme gives order 1, outside the band."""
        import math

        hs = (0.04, 0.02, 0.01, 0.005, 0.0025)
        _, P, _ = _profile()
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
        substeps per segment above 1e5 Pa: the count pins the default h (a call at h 0.02
        meets the 1 K bound of the real profiles too, so only the count tells them apart)."""
        import math

        calls = []
        _stub_adiabat(monkeypatch, lambda P, T: 0.3, calls)
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
        _, P0, _ = _profile(n=21, p_top=2e5)
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
    def test_padded_top_stays_isothermal(self, monkeypatch, anchor):
        """A padded top (P = 0 at the outer node) is a stop: the top shell is isothermal, so the
        surface-anchored profile below it equals the one of a top at the next node."""
        _stub_adiabat(monkeypatch, lambda P, T: 0.1)
        _, P, _ = _profile(n=21)
        P[-1] = 0.0
        T = _call(P, anchor, 3000.0)
        assert T[-1] == T[-2]
        if anchor == 'surface':
            assert T[-2] == 3000.0
            np.testing.assert_allclose(T[:-1], 3000.0 * (P[:-1] / P[-2]) ** 0.1, rtol=1e-12)
        else:
            assert T[-2] < T[1]

    @pytest.mark.physics_invariant
    def test_negative_nabla_holds_the_rest_of_the_segment(self, monkeypatch):
        """nabla <= 0 or None at a substep holds T for the rest of that segment; the next
        segment integrates again, so T never falls inward."""
        _stub_adiabat(monkeypatch, lambda P, T: None if 1e9 < P < 1e10 else 0.1)
        _, P, _ = _profile()
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
        _, P, _ = _profile(n=5)
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
        _, P, _ = _profile(n=11)
        good = _call(P, 'surface', 3000.0)
        assert np.all(np.diff(good) < 0)
        monkeypatch.setattr(zt, '_segment_temperature', lambda T0, *a, **k: 0.5 * T0)
        with pytest.raises(StructureInvariantError, match='non-increasing'):
            _call(P, 'surface', 3000.0)
        monkeypatch.setattr(zt, '_segment_temperature', lambda T0, *a, **k: np.nan)
        with pytest.raises(StructureInvariantError, match='finite'):
            _call(P, 'cmb', 3000.0)

    @pytest.mark.parametrize(
        ('anchor', 'value'),
        [
            ('surface', 0.0),
            ('surface', -10.0),
            ('surface', np.nan),
            ('surface', np.inf),
            ('surface', 2e5),
            ('cmb', 50.0),
            ('cmb', 99.9),
            ('cmb', 2e5),
            ('cmb', np.nan),
        ],
    )
    def test_anchor_temperature_outside_its_range_raises_value_error(
        self, monkeypatch, anchor, value
    ):
        """The anchor temperature must be finite, positive and at most 1e5 K, and with the CMB
        anchor at least 100 K (the node caps); outside that range the call raises ValueError
        before any nabla read, not a profile invariant error from a capped node."""
        calls = []
        _stub_adiabat(monkeypatch, lambda P, T: 0.1, calls)
        P = _profile(n=5)[1]
        with pytest.raises(ValueError, match='temperature'):
            _call(P, anchor, value)
        assert calls == []

    def test_non_finite_surface_temperature_with_the_cmb_anchor_raises(self, monkeypatch):
        """With the CMB anchor the core nodes carry the surface temperature, so it must be finite
        and positive too; NaN raises ValueError, while the edges 100 K and 1e5 K run."""
        _stub_adiabat(monkeypatch, lambda P, T: 0.1)
        P = _profile(n=5)[1]
        r, m = np.linspace(0.0, 6.4e6, 5), np.linspace(0, M_E, 5)
        kw = dict(
            radii=r,
            pressure=P,
            mass_enclosed=m,
            cmb_mass=0.5 * m[1],
            core_mantle_mass=M_E,
            layer_mixtures={'mantle': _StubTdepMixture()},
            material_dictionaries={},
            anchor='cmb',
        )
        with pytest.raises(ValueError, match='surface_temperature'):
            compute_adiabatic_temperature(
                surface_temperature=np.nan, cmb_temperature=3000.0, **kw
            )
        for t in (100.0, 1e5):
            T = compute_adiabatic_temperature(
                surface_temperature=2000.0, cmb_temperature=t, **kw
            )
            assert np.all(np.isfinite(T)) and T[1] == t

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize('nabla', [1e6, np.inf])
    def test_a_huge_nabla_caps_the_profile_at_the_table_top(self, monkeypatch, nabla):
        """A nabla far above any table value (1e6, inf) drives T to the 1e5 K cap of each node
        instead of an overflow; the anchor keeps its value and no node exceeds the cap."""
        _stub_adiabat(monkeypatch, lambda P, T: nabla)
        P = _profile(n=11)[1]
        T = _call(P, 'surface', 3000.0)
        assert T[-1] == 3000.0
        np.testing.assert_array_equal(T[:-1], 1e5)

    @pytest.mark.physics_invariant
    def test_max_dlnp_property_over_its_domain(self, monkeypatch):
        """Hypothesis over floats: from 1e-4 to 1 the profile is finite, positive and
        non-increasing outward and within 1e-12 of the constant-nabla law; outside it, and for
        NaN and the infinities, the call raises ValueError."""
        from hypothesis import given, settings
        from hypothesis import strategies as st

        _stub_adiabat(monkeypatch, lambda P, T: 0.1)
        _, P, _ = _profile(n=6, p_center=1e7, p_top=2e5)
        exact = 3000.0 * (P / P[-1]) ** 0.1

        @settings(max_examples=200, deadline=None)
        @given(st.floats(allow_nan=True, allow_infinity=True))
        def check(h):
            if 1e-4 <= h <= 1.0:
                T = _call(P, 'surface', 3000.0, max_dlnp=h)
                assert np.all(np.isfinite(T)) and np.all(np.diff(T) <= 0)
                np.testing.assert_allclose(T, exact, rtol=1e-12)
            else:
                with pytest.raises(ValueError, match='max_dlnp'):
                    _call(P, 'surface', 3000.0, max_dlnp=h)

        check()
