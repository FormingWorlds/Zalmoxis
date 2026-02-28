"""Tests for adiabatic temperature mode.

Tests cover:
- Thermal expansivity computation via finite differences
- Adiabatic temperature profile integration
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
    compute_thermal_expansivity,
)


@pytest.mark.unit
class TestComputeThermalExpansivity:
    """Tests for compute_thermal_expansivity()."""

    def test_positive_alpha_for_melt(self):
        """Thermal expansivity should be positive for MgSiO3 melt at moderate P, T.

        At 10 GPa and 3000 K, MgSiO3 melt has well-constrained EOS
        with alpha ~ 1e-5 to 1e-4 1/K.
        """
        from zalmoxis.zalmoxis import (
            load_material_dictionaries,
            load_solidus_liquidus_functions,
        )

        material_dicts = load_material_dictionaries()
        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'}
        melting = load_solidus_liquidus_functions(layer_eos_config)
        solidus_func, liquidus_func = melting

        alpha = compute_thermal_expansivity(
            pressure=10e9,
            temperature=3000.0,
            material_dictionaries=material_dicts,
            layer_eos='WolfBower2018:MgSiO3',
            solidus_func=solidus_func,
            liquidus_func=liquidus_func,
        )
        assert alpha > 0, f'Expected positive alpha for MgSiO3 melt, got {alpha}'
        # Typical values for silicate melts: 1e-5 to 5e-4 1/K
        assert 1e-6 < alpha < 1e-3, f'Alpha {alpha} outside expected range'

    def test_zero_alpha_for_T_independent_eos(self):
        """Thermal expansivity should be zero for T-independent EOS (Seager2007 iron).

        Seager2007 iron density depends only on pressure, so drho/dT = 0.
        """
        from zalmoxis.zalmoxis import load_material_dictionaries

        material_dicts = load_material_dictionaries()

        alpha = compute_thermal_expansivity(
            pressure=100e9,
            temperature=4000.0,
            material_dictionaries=material_dicts,
            layer_eos='Seager2007:iron',
            solidus_func=None,
            liquidus_func=None,
        )
        assert alpha == pytest.approx(0.0), (
            f'Expected zero alpha for T-independent EOS, got {alpha}'
        )


@pytest.mark.unit
class TestComputeAdiabaticTemperature:
    """Tests for compute_adiabatic_temperature()."""

    def test_surface_temperature_exact(self):
        """T at the surface should equal surface_temperature exactly."""
        n = 50
        radii = np.linspace(0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e5, n)  # center to surface
        gravity = np.full(n, 9.8)
        mass_enclosed = np.linspace(0, 5.972e24, n)

        T_surface = 3500.0
        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'}
        from zalmoxis.zalmoxis import load_material_dictionaries

        material_dicts = load_material_dictionaries()

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            gravity=gravity,
            mass_enclosed=mass_enclosed,
            surface_temperature=T_surface,
            Cp=1200.0,
            cmb_mass=0.325 * 5.972e24,
            core_mantle_mass=5.972e24,
            layer_eos_config=layer_eos_config,
            material_dictionaries=material_dicts,
            solidus_func=None,
            liquidus_func=None,
        )
        assert T[-1] == pytest.approx(T_surface), f'Surface temperature {T[-1]} != {T_surface}'

    def test_isothermal_for_T_independent_eos(self):
        """For a fully T-independent EOS planet, the adiabat should be isothermal.

        When alpha = 0 everywhere, dT/dr = 0, so T(r) = T_surface.
        """
        n = 50
        radii = np.linspace(0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e5, n)
        gravity = np.full(n, 9.8)
        mass_enclosed = np.linspace(0, 5.972e24, n)

        T_surface = 3500.0
        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'}
        from zalmoxis.zalmoxis import load_material_dictionaries

        material_dicts = load_material_dictionaries()

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            gravity=gravity,
            mass_enclosed=mass_enclosed,
            surface_temperature=T_surface,
            Cp=1200.0,
            cmb_mass=0.325 * 5.972e24,
            core_mantle_mass=5.972e24,
            layer_eos_config=layer_eos_config,
            material_dictionaries=material_dicts,
            solidus_func=None,
            liquidus_func=None,
        )
        np.testing.assert_allclose(T, T_surface, rtol=1e-10)

    def test_monotonic_increase_with_Tdep_eos(self):
        """T should increase from surface toward center for a T-dependent mantle EOS.

        The adiabatic gradient dT/dr = -alpha*g*T/Cp with alpha > 0, g > 0
        means T increases inward (decreasing r).
        """
        from zalmoxis.zalmoxis import (
            load_material_dictionaries,
            load_solidus_liquidus_functions,
        )

        material_dicts = load_material_dictionaries()
        layer_eos_config = {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'}
        melting = load_solidus_liquidus_functions(layer_eos_config)
        solidus_func, liquidus_func = melting

        n = 100
        radii = np.linspace(0, 6.371e6, n)
        pressure = np.linspace(360e9, 1e5, n)
        gravity = np.linspace(0, 9.8, n)
        mass_enclosed = np.linspace(0, 5.972e24, n)

        T = compute_adiabatic_temperature(
            radii=radii,
            pressure=pressure,
            gravity=gravity,
            mass_enclosed=mass_enclosed,
            surface_temperature=3500.0,
            Cp=1200.0,
            cmb_mass=0.325 * 5.972e24,
            core_mantle_mass=5.972e24,
            layer_eos_config=layer_eos_config,
            material_dictionaries=material_dicts,
            solidus_func=solidus_func,
            liquidus_func=liquidus_func,
        )
        # In the mantle (T-dependent EOS), T should increase inward
        cmb_idx = np.argmax(mass_enclosed >= 0.325 * 5.972e24)
        mantle_T = T[cmb_idx:]
        # Mantle goes from CMB (lower index) to surface (higher index)
        # T should decrease from CMB to surface (i.e. increase going inward)
        assert mantle_T[0] > mantle_T[-1], (
            f'Mantle T at CMB ({mantle_T[0]:.0f} K) should be > surface ({mantle_T[-1]:.0f} K)'
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
