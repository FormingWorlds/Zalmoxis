"""Tests for adiabatic temperature mode.

Tests cover:
- Adiabatic temperature profile integration via native dT/dP tables
- Thick-mantle divergence regression test
- Adiabatic mode in calculate_temperature_profile()

References:
    - docs/test_infrastructure.md
    - docs/test_categorization.md
"""

from __future__ import annotations

import numpy as np
import pytest

from zalmoxis.eos_functions import (
    calculate_temperature_profile,
    compute_adiabatic_temperature,
)


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
        from zalmoxis.zalmoxis import load_material_dictionaries

        material_dicts = load_material_dictionaries()

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=T_surface,
            cmb_mass=0.325 * 5.972e24,
            core_mantle_mass=5.972e24,
            layer_eos_config=layer_eos_config,
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
        from zalmoxis.zalmoxis import load_material_dictionaries

        material_dicts = load_material_dictionaries()

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            mass_enclosed=mass_enclosed,
            surface_temperature=T_surface,
            cmb_mass=cmb_mass,
            core_mantle_mass=5.972e24,
            layer_eos_config=layer_eos_config,
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

        from zalmoxis.zalmoxis import load_material_dictionaries

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
            layer_eos_config=layer_eos_config,
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

    def test_requires_adiabat_grad_file(self):
        """Should raise ValueError if T-dep EOS has no adiabat_grad_file."""
        import copy

        from zalmoxis.zalmoxis import load_material_dictionaries

        material_dicts = copy.deepcopy(load_material_dictionaries())
        # Remove the adiabat_grad_file key
        material_dicts[1]['melted_mantle'].pop('adiabat_grad_file', None)

        n = 50
        radii = np.linspace(0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e5, n)
        mass_enclosed = np.linspace(0, 5.972e24, n)

        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'}

        with pytest.raises(ValueError, match='adiabat_grad_file'):
            compute_adiabatic_temperature(
                radii=radii,
                pressure=pressure,
                mass_enclosed=mass_enclosed,
                surface_temperature=3500.0,
                cmb_mass=0.325 * 5.972e24,
                core_mantle_mass=5.972e24,
                layer_eos_config=layer_eos_config,
                material_dictionaries=material_dicts,
            )

    def test_rejects_T_independent_mantle_eos(self):
        """Should raise ValueError if mantle EOS is T-independent (e.g. Seager2007)."""
        from zalmoxis.zalmoxis import load_material_dictionaries

        material_dicts = load_material_dictionaries()

        n = 50
        radii = np.linspace(0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e5, n)
        mass_enclosed = np.linspace(0, 5.972e24, n)

        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'}

        with pytest.raises(ValueError, match='T-dependent mantle EOS'):
            compute_adiabatic_temperature(
                radii=radii,
                pressure=pressure,
                mass_enclosed=mass_enclosed,
                surface_temperature=3500.0,
                cmb_mass=0.325 * 5.972e24,
                core_mantle_mass=5.972e24,
                layer_eos_config=layer_eos_config,
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

        from zalmoxis.zalmoxis import load_material_dictionaries

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
            layer_eos_config=layer_eos_config,
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

        from zalmoxis.zalmoxis import load_material_dictionaries

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
            layer_eos_config=layer_eos_config,
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
