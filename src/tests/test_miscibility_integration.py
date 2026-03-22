"""Integration tests for global miscibility (binodal-aware structure).

These tests run the full Zalmoxis solver with solve_miscible_interior
to verify end-to-end mass conservation, solvus detection, and density
profiles for sub-Neptune-like planets.

Requires EOS data in ZALMOXIS_ROOT/data/ (downloaded via get_zalmoxis.sh).
"""

from __future__ import annotations

import numpy as np
import pytest

from zalmoxis.eos_properties import EOS_REGISTRY
from zalmoxis.mixing import VolatileProfile
from zalmoxis.zalmoxis import (
    load_material_dictionaries,
    load_zalmoxis_config,
    main,
    solve_miscible_interior,
)

M_EARTH = 5.972e24  # kg
R_EARTH = 6.371e6  # m


def _make_sub_neptune_config(
    mass_earth=6.0, T_surface=4000.0, T_center=6000.0, temp_mode='isothermal'
):
    """Create a Zalmoxis config for a sub-Neptune test case."""
    config = load_zalmoxis_config('input/default.toml')
    config['planet_mass'] = mass_earth * M_EARTH
    config['core_mass_fraction'] = 0.325
    config['num_layers'] = 80
    config['surface_temperature'] = T_surface
    config['center_temperature'] = T_center
    config['temperature_mode'] = temp_mode
    config['max_iterations_outer'] = 50
    config['tolerance_outer'] = 5e-3
    config['max_iterations_inner'] = 50
    config['tolerance_inner'] = 1e-3
    config['layer_eos_config'] = {
        'core': 'PALEOS:iron',
        'mantle': 'PALEOS:MgSiO3:0.97+Chabrier:H:0.03',
    }
    return config


def _make_volatile_profile(x_H2=0.03):
    """Create a VolatileProfile with global miscibility enabled."""
    return VolatileProfile(
        w_liquid={'Chabrier:H': x_H2},
        w_solid={'Chabrier:H': 0.0},
        primary_component='PALEOS:MgSiO3',
        x_interior={'Chabrier:H': x_H2},
        global_miscibility=True,
    )


# ═════════════════════════════════════════════════════════════════════
# Baseline: no miscibility
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestBaselineNoMiscibility:
    """Verify standard Zalmoxis solve works for sub-Neptune mass."""

    def test_6_mearth_converges(self):
        """6 M_Earth planet at 4000 K converges and gives reasonable radius."""
        config = _make_sub_neptune_config(mass_earth=6.0, T_surface=4000.0)
        result = main(
            config,
            material_dictionaries=load_material_dictionaries(),
            melting_curves_functions=None,
            input_dir='output_files',
        )
        assert result['converged']
        R = result['radii'][-1] / R_EARTH
        # 6 M_Earth rocky planet: expect ~1.5-2.0 R_Earth
        assert 1.2 < R < 2.5, f'Radius {R:.3f} R_Earth out of expected range'


# ═════════════════════════════════════════════════════════════════════
# Miscibility: isothermal (fully miscible, no solvus)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestMiscibilityIsothermal:
    """Isothermal 4000 K: fully miscible, no solvus crossing."""

    def test_converges(self):
        """solve_miscible_interior converges."""
        config = _make_sub_neptune_config(mass_earth=6.0, T_surface=4000.0)
        vp = _make_volatile_profile(x_H2=0.03)
        M_mantle = 0.675 * config['planet_mass']
        target = 0.03 * M_mantle

        result = solve_miscible_interior(
            config,
            material_dictionaries=load_material_dictionaries(),
            melting_curves_functions=None,
            input_dir='output_files',
            volatile_profile=vp,
            h2_mass_targets={'Chabrier:H': target},
            max_iterations=8,
            mass_tolerance=0.05,
        )
        assert result['converged']
        assert result['miscibility_converged']

    def test_no_solvus_at_4000K(self):
        """At 4000 K isothermal, entire planet is above binodal."""
        config = _make_sub_neptune_config(mass_earth=6.0, T_surface=4000.0)
        vp = _make_volatile_profile(x_H2=0.03)
        M_mantle = 0.675 * config['planet_mass']
        target = 0.03 * M_mantle

        result = solve_miscible_interior(
            config,
            material_dictionaries=load_material_dictionaries(),
            melting_curves_functions=None,
            input_dir='output_files',
            volatile_profile=vp,
            h2_mass_targets={'Chabrier:H': target},
            max_iterations=8,
            mass_tolerance=0.05,
        )
        # No solvus: fully miscible
        assert result['solvus_radius'] is None

    def test_mass_conservation(self):
        """Integrated H2 mass matches target within tolerance."""
        config = _make_sub_neptune_config(mass_earth=6.0, T_surface=4000.0)
        vp = _make_volatile_profile(x_H2=0.03)
        M_mantle = 0.675 * config['planet_mass']
        target = 0.03 * M_mantle

        result = solve_miscible_interior(
            config,
            material_dictionaries=load_material_dictionaries(),
            melting_curves_functions=None,
            input_dir='output_files',
            volatile_profile=vp,
            h2_mass_targets={'Chabrier:H': target},
            max_iterations=8,
            mass_tolerance=0.05,
        )
        integrated = result['h2_mass_integrated']['Chabrier:H']
        rel_err = abs(integrated - target) / target
        assert rel_err < 0.05, f'Mass conservation error: {rel_err:.4f}'

    def test_miscible_radius_smaller_than_standard(self):
        """Dissolved H2 should make the interior denser (smaller radius).

        Rogers+2025 Fig. 5 shows MISCIBLE models are more compact than
        STANDARD at early times because the interior stores H2, making
        the gravitational well deeper.
        """
        config = _make_sub_neptune_config(mass_earth=6.0, T_surface=4000.0)

        # Standard (no miscibility)
        result_std = main(
            config,
            material_dictionaries=load_material_dictionaries(),
            melting_curves_functions=None,
            input_dir='output_files',
        )

        # Miscible
        vp = _make_volatile_profile(x_H2=0.03)
        M_mantle = 0.675 * config['planet_mass']
        target = 0.03 * M_mantle

        result_misc = solve_miscible_interior(
            config,
            material_dictionaries=load_material_dictionaries(),
            melting_curves_functions=None,
            input_dir='output_files',
            volatile_profile=vp,
            h2_mass_targets={'Chabrier:H': target},
            max_iterations=8,
            mass_tolerance=0.05,
        )

        R_std = result_std['radii'][-1]
        R_misc = result_misc['radii'][-1]
        assert R_misc < R_std, (
            f'Miscible radius ({R_misc / R_EARTH:.3f} R_E) should be '
            f'< standard ({R_std / R_EARTH:.3f} R_E)'
        )


# ═════════════════════════════════════════════════════════════════════
# Miscibility: linear T profile (solvus crossing)
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestMiscibilityLinearT:
    """Linear T(r) from 6000 K to 2000 K: should cross binodal."""

    def test_solvus_detected(self):
        """Solvus detected at intermediate radius."""
        config = _make_sub_neptune_config(
            mass_earth=6.0,
            T_surface=2000.0,
            T_center=6000.0,
            temp_mode='linear',
        )
        vp = _make_volatile_profile(x_H2=0.03)
        M_mantle = 0.675 * config['planet_mass']
        target = 0.03 * M_mantle

        result = solve_miscible_interior(
            config,
            material_dictionaries=load_material_dictionaries(),
            melting_curves_functions=None,
            input_dir='output_files',
            volatile_profile=vp,
            h2_mass_targets={'Chabrier:H': target},
            max_iterations=8,
            mass_tolerance=0.05,
        )
        assert result['converged']
        assert result['solvus_radius'] is not None
        # Solvus should be within the planet, not at center or surface
        R_planet = result['radii'][-1]
        R_solvus = result['solvus_radius']
        assert 0.5 * R_planet < R_solvus < 0.999 * R_planet

    def test_solvus_temperature_is_physical(self):
        """Solvus temperature should be in the binodal range (~2000-4000 K)."""
        config = _make_sub_neptune_config(
            mass_earth=6.0,
            T_surface=2000.0,
            T_center=6000.0,
            temp_mode='linear',
        )
        vp = _make_volatile_profile(x_H2=0.03)
        M_mantle = 0.675 * config['planet_mass']
        target = 0.03 * M_mantle

        result = solve_miscible_interior(
            config,
            material_dictionaries=load_material_dictionaries(),
            melting_curves_functions=None,
            input_dir='output_files',
            volatile_profile=vp,
            h2_mass_targets={'Chabrier:H': target},
            max_iterations=8,
            mass_tolerance=0.05,
        )
        T_sol = result['solvus_temperature']
        assert T_sol is not None
        # Binodal temperature at GPa pressures: ~2000-4000 K
        assert 1500 < T_sol < 4500, f'Solvus T={T_sol:.0f} K unexpected'

    def test_mass_conservation_with_solvus(self):
        """Mass conservation holds when solvus truncates the miscible region."""
        config = _make_sub_neptune_config(
            mass_earth=6.0,
            T_surface=2000.0,
            T_center=6000.0,
            temp_mode='linear',
        )
        vp = _make_volatile_profile(x_H2=0.03)
        M_mantle = 0.675 * config['planet_mass']
        target = 0.03 * M_mantle

        result = solve_miscible_interior(
            config,
            material_dictionaries=load_material_dictionaries(),
            melting_curves_functions=None,
            input_dir='output_files',
            volatile_profile=vp,
            h2_mass_targets={'Chabrier:H': target},
            max_iterations=8,
            mass_tolerance=0.05,
        )
        integrated = result['h2_mass_integrated']['Chabrier:H']
        rel_err = abs(integrated - target) / target
        assert rel_err < 0.05, f'Mass conservation error: {rel_err:.4f}'


# ═════════════════════════════════════════════════════════════════════
# Backward compatibility
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestBackwardCompatibility:
    """solve_miscible_interior without targets = identical to main()."""

    def test_no_targets_matches_main(self):
        """Without h2_mass_targets, results match a direct main() call."""
        config = _make_sub_neptune_config(mass_earth=6.0, T_surface=4000.0)
        mat = load_material_dictionaries()

        result_main = main(
            config,
            material_dictionaries=mat,
            melting_curves_functions=None,
            input_dir='output_files',
        )

        result_misc = solve_miscible_interior(
            config,
            material_dictionaries=mat,
            melting_curves_functions=None,
            input_dir='output_files',
            volatile_profile=None,
            h2_mass_targets=None,
        )

        assert result_main['radii'][-1] == pytest.approx(
            result_misc['radii'][-1], rel=1e-6
        )
        assert result_misc['miscibility_converged'] is True
        assert result_misc['miscibility_iterations'] == 0
