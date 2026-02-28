from __future__ import annotations

import os

import pytest

from zalmoxis.plots.plot_ternary import run_ternary_grid_for_mass


@pytest.mark.slow
@pytest.mark.parametrize(
    'mass', [1, 10]
)  # 1 and 10 Earth masses (keep it simple for CI tests; works up to 50)
def test_all_compositions_converge(mass, zalmoxis_root):
    """
    Test the convergence of the Zalmoxis model across all possible compositions
    by varying the core, mantle, and water mass fractions for a given planet mass.
    Parameters:
        mass (int): Mass of the planet in Earth masses.
    """
    print(f'Running test for mass = {mass}')

    # Delete composition_radius_log file if it exists
    custom_log_file = os.path.join(
        zalmoxis_root, 'output_files', f'composition_radius_log{mass}.txt'
    )
    if os.path.exists(custom_log_file):
        os.remove(custom_log_file)

    # Run the ternary grid with coarse step for CI (step=0.05 for full validation)
    results = run_ternary_grid_for_mass(mass, step=0.2)

    # Filter out any failed convergence
    failed_cases = [
        (core_frac, mantle_frac, water_frac)
        for (core_frac, mantle_frac, water_frac, converged) in results
        if not converged
    ]

    if failed_cases:
        failed_str = ', '.join(
            [
                f'(core={core_frac:.2f}, mantle={mantle_frac:.2f}, water={water_frac:.2f})'
                for core_frac, mantle_frac, water_frac in failed_cases
            ]
        )
        pytest.fail(
            f'The following compositions did not converge for mass {mass}: {failed_str}'
        )
