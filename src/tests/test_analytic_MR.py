"""
Integration tests for the Seager+2007 analytic EOS with the full Zalmoxis solver.

Tests mass-radius relations for various material combinations including
Earth-like, water-world, and carbon planet configurations.

References:
    - Seager et al. (2007), ApJ 669:1279
    - docs/test_infrastructure.md
    - docs/test_categorization.md
"""

from __future__ import annotations

import os

import pytest

from tools.setup_tests import load_model_output, run_zalmoxis_rocky_water

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')


class TestAnalyticVsTabulatedMR:
    """Compare analytic iron/MgSiO3 mass-radius against tabulated iron/silicate."""

    @pytest.mark.parametrize('mass', [1, 5, 10])
    def test_rocky_planet_radius_agreement(self, mass):
        """Analytic iron/MgSiO3 should produce radii within ~5% of tabulated iron/silicate."""
        # Run tabulated
        output_tab, _ = run_zalmoxis_rocky_water(mass, 'rocky', cmf=0.325, immf=0)
        mass_tab, radius_tab = load_model_output(output_tab)

        # Run analytic with same composition
        analytic_mats = {'core': 'iron', 'mantle': 'MgSiO3'}
        output_ana, _ = run_zalmoxis_rocky_water(
            mass,
            'rocky',
            cmf=0.325,
            immf=0,
            eos_override='Analytic:Seager2007',
            analytic_materials=analytic_mats,
        )
        mass_ana, radius_ana = load_model_output(output_ana)

        rel_diff = abs(radius_ana - radius_tab) / radius_tab
        assert rel_diff < 0.05, (
            f'At {mass} M_earth: analytic R={radius_ana:.4f} vs tabulated R={radius_tab:.4f} '
            f'R_earth, rel_diff={rel_diff:.2%}'
        )


class TestAnalyticWaterPlanetMR:
    """Compare analytic 3-layer iron/MgSiO3/H2O against tabulated water planets."""

    @pytest.mark.parametrize('mass', [1, 5, 10])
    def test_water_planet_radius_agreement(self, mass):
        """Analytic iron/MgSiO3/H2O should produce radii within ~5% of tabulated water."""
        # Run tabulated water
        output_tab, _ = run_zalmoxis_rocky_water(mass, 'water', cmf=0.065, immf=0.485)
        mass_tab, radius_tab = load_model_output(output_tab)

        # Run analytic with same composition
        analytic_mats = {'core': 'iron', 'mantle': 'MgSiO3', 'water_ice_layer': 'H2O'}
        output_ana, _ = run_zalmoxis_rocky_water(
            mass,
            'water',
            cmf=0.065,
            immf=0.485,
            eos_override='Analytic:Seager2007',
            analytic_materials=analytic_mats,
        )
        mass_ana, radius_ana = load_model_output(output_ana)

        rel_diff = abs(radius_ana - radius_tab) / radius_tab
        assert rel_diff < 0.05, (
            f'At {mass} M_earth: analytic R={radius_ana:.4f} vs tabulated R={radius_tab:.4f} '
            f'R_earth, rel_diff={rel_diff:.2%}'
        )


class TestExoticPlanetConvergence:
    """Test convergence for exotic material combinations only possible with the analytic EOS."""

    def test_carbon_planet_SiC_converges(self):
        """Iron core + SiC mantle planet at 1 M_earth should converge."""
        analytic_mats = {'core': 'iron', 'mantle': 'SiC'}
        output_file, _ = run_zalmoxis_rocky_water(
            1,
            'rocky',
            cmf=0.325,
            immf=0,
            eos_override='Analytic:Seager2007',
            analytic_materials=analytic_mats,
        )
        mass_out, radius_out = load_model_output(output_file)
        # Physically plausible: smaller than Earth (SiC is denser than MgSiO3)
        assert 0.5 < radius_out < 1.5, (
            f'SiC planet radius {radius_out:.3f} R_earth outside plausible range'
        )

    def test_carbon_planet_graphite_converges(self):
        """Iron core + graphite mantle planet at 1 M_earth should converge."""
        analytic_mats = {'core': 'iron', 'mantle': 'graphite'}
        output_file, _ = run_zalmoxis_rocky_water(
            1,
            'rocky',
            cmf=0.325,
            immf=0,
            eos_override='Analytic:Seager2007',
            analytic_materials=analytic_mats,
        )
        mass_out, radius_out = load_model_output(output_file)
        assert 0.5 < radius_out < 1.5, (
            f'Graphite planet radius {radius_out:.3f} R_earth outside plausible range'
        )
