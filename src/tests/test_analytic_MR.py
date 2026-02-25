"""
Integration tests for the Seager+2007 analytic EOS with the full Zalmoxis solver.

Tests mass-radius relations for various material combinations including
Earth-like, water-world, and carbon planet configurations. Also tests
per-layer mixed EOS (tabulated + analytic).

References:
    - Seager et al. (2007), ApJ 669:1279
    - docs/test_infrastructure.md
    - docs/test_categorization.md
"""

from __future__ import annotations

import pytest

from tools.setup_tests import load_model_output


@pytest.mark.integration
class TestAnalyticVsTabulatedMR:
    """Compare analytic iron/MgSiO3 mass-radius against tabulated iron/silicate."""

    @pytest.mark.parametrize('mass', [1, 5, 10])
    def test_rocky_planet_radius_agreement(self, mass, cached_solver):
        """Analytic iron/MgSiO3 should produce radii within ~5% of tabulated iron/silicate."""
        # Run tabulated
        output_tab, _ = cached_solver(mass, 'rocky', cmf=0.325, immf=0)
        mass_tab, radius_tab = load_model_output(output_tab)

        # Run analytic with same composition
        output_ana, _ = cached_solver(
            mass,
            'rocky',
            cmf=0.325,
            immf=0,
            layer_eos_override={'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'},
        )
        mass_ana, radius_ana = load_model_output(output_ana)

        rel_diff = abs(radius_ana - radius_tab) / radius_tab
        assert rel_diff < 0.05, (
            f'At {mass} M_earth: analytic R={radius_ana:.4f} vs tabulated R={radius_tab:.4f} '
            f'R_earth, rel_diff={rel_diff:.2%}'
        )


@pytest.mark.integration
class TestAnalyticWaterPlanetMR:
    """Compare analytic 3-layer iron/MgSiO3/H2O against tabulated water planets."""

    @pytest.mark.parametrize('mass', [1, 5, 10])
    def test_water_planet_radius_agreement(self, mass, cached_solver):
        """Analytic iron/MgSiO3/H2O should produce radii within ~5% of tabulated water."""
        # Run tabulated water
        output_tab, _ = cached_solver(mass, 'water', cmf=0.065, immf=0.485)
        mass_tab, radius_tab = load_model_output(output_tab)

        # Run analytic with same composition
        output_ana, _ = cached_solver(
            mass,
            'water',
            cmf=0.065,
            immf=0.485,
            layer_eos_override={
                'core': 'Analytic:iron',
                'mantle': 'Analytic:MgSiO3',
                'ice_layer': 'Analytic:H2O',
            },
        )
        mass_ana, radius_ana = load_model_output(output_ana)

        rel_diff = abs(radius_ana - radius_tab) / radius_tab
        assert rel_diff < 0.05, (
            f'At {mass} M_earth: analytic R={radius_ana:.4f} vs tabulated R={radius_tab:.4f} '
            f'R_earth, rel_diff={rel_diff:.2%}'
        )


@pytest.mark.integration
class TestExoticPlanetConvergence:
    """Test convergence for exotic material combinations only possible with the analytic EOS."""

    def test_carbon_planet_SiC_converges(self, cached_solver):
        """Iron core + SiC mantle planet at 1 M_earth should converge."""
        output_file, _ = cached_solver(
            1,
            'rocky',
            cmf=0.325,
            immf=0,
            layer_eos_override={'core': 'Analytic:iron', 'mantle': 'Analytic:SiC'},
        )
        mass_out, radius_out = load_model_output(output_file)
        # Physically plausible: smaller than Earth (SiC is denser than MgSiO3)
        assert 0.5 < radius_out < 1.5, (
            f'SiC planet radius {radius_out:.3f} R_earth outside plausible range'
        )

    def test_carbon_planet_graphite_converges(self, cached_solver):
        """Iron core + graphite mantle planet at 1 M_earth should converge."""
        output_file, _ = cached_solver(
            1,
            'rocky',
            cmf=0.325,
            immf=0,
            layer_eos_override={'core': 'Analytic:iron', 'mantle': 'Analytic:graphite'},
        )
        mass_out, radius_out = load_model_output(output_file)
        assert 0.5 < radius_out < 1.5, (
            f'Graphite planet radius {radius_out:.3f} R_earth outside plausible range'
        )


@pytest.mark.integration
class TestMixedEOS:
    """Test per-layer mixing of tabulated and analytic EOS."""

    @pytest.mark.parametrize('mass', [1, 5, 10])
    def test_tabulated_core_analytic_mantle(self, mass, cached_solver):
        """Seager2007:iron core + Analytic:MgSiO3 mantle should converge
        and produce radii within ~5% of pure tabulated."""
        # Run pure tabulated (may hit cache from TestAnalyticVsTabulatedMR)
        output_tab, _ = cached_solver(mass, 'rocky', cmf=0.325, immf=0)
        _, radius_tab = load_model_output(output_tab)

        # Run mixed: tabulated core + analytic mantle
        output_mix, _ = cached_solver(
            mass,
            'rocky',
            cmf=0.325,
            immf=0,
            layer_eos_override={
                'core': 'Seager2007:iron',
                'mantle': 'Analytic:MgSiO3',
            },
        )
        _, radius_mix = load_model_output(output_mix)

        rel_diff = abs(radius_mix - radius_tab) / radius_tab
        assert rel_diff < 0.05, (
            f'At {mass} M_earth: mixed R={radius_mix:.4f} vs tabulated R={radius_tab:.4f} '
            f'R_earth, rel_diff={rel_diff:.2%}'
        )
