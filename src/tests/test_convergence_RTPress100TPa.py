"""Tests for the RTPress100TPa:MgSiO3 T-dependent EOS.

The RTPress100TPa melt table extends to 100 TPa (P: 1e3–1e14 Pa, T: 400–50000 K),
enabling modeling of much more massive rocky planets than WolfBower2018 (limited
to 7 M_earth). The solid table is still WolfBower2018 (1 TPa, clamped at boundary).

See also:
- docs/test_infrastructure.md
- docs/test_categorization.md
- docs/test_building.md
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tools.setup_tests import run_zalmoxis_RTPress100TPa


@pytest.mark.unit
def test_RTPress100TPa_mass_limit_raises():
    """Requesting > 50 M_earth with RTPress100TPa must raise ValueError."""
    with pytest.raises(ValueError, match='RTPress100TPa'):
        run_zalmoxis_RTPress100TPa(51)


@pytest.mark.integration
@pytest.mark.parametrize(
    'mass', [1, 5]
)  # RTPress100TPa valid up to ~50 M_earth; test low and moderate masses
def test_RTPress100TPa_converges(mass, zalmoxis_root):
    """Test that the RTPress100TPa:MgSiO3 EOS model converges.

    Verifies convergence and checks that the resulting density profiles are
    physically consistent: iron core densities >= 8000 kg/m3, mantle densities
    in the expected range for T-dependent MgSiO3.
    """
    print(f'Running RTPress100TPa test for mass = {mass}')

    # Delete log file if it exists
    custom_log_file = os.path.join(
        zalmoxis_root, 'output_files', 'composition_RTPress100TPa_mass_log.txt'
    )
    if os.path.exists(custom_log_file):
        os.remove(custom_log_file)

    # Run Zalmoxis with RTPress100TPa EOS
    results = run_zalmoxis_RTPress100TPa(mass)

    # Filter out any failed convergence
    failed_cases = [(id_mass) for (id_mass, converged, _) in results if not converged]

    if failed_cases:
        failed_str = ', '.join([f'mass={id_mass:.2f}' for id_mass in failed_cases])
        pytest.fail(f'The following masses did not converge {mass}: {failed_str}')

    # Check density profiles for physical correctness
    for id_mass, converged, model_results in results:
        if not converged:
            continue

        density = model_results['density']
        mass_enclosed = model_results['mass_enclosed']
        cmb_mass = model_results['cmb_mass']

        # Find core-mantle boundary index
        cmb_index = np.argmax(mass_enclosed >= cmb_mass)
        if cmb_index == 0:
            cmb_index = 1  # Ensure at least one core point

        # Iron core density check (8000-50000 kg/m3)
        core_densities = density[1:cmb_index]  # Skip r=0 (may be zero initially)
        if len(core_densities) > 0:
            max_core_rho = np.max(core_densities)
            min_core_rho = np.min(core_densities[core_densities > 0])
            assert min_core_rho >= 8000, (
                f'Core density {min_core_rho:.0f} kg/m3 below iron minimum (8000) '
                f'for {id_mass} M_earth'
            )
            assert max_core_rho <= 50000, (
                f'Core density {max_core_rho:.0f} kg/m3 above iron maximum (50000) '
                f'for {id_mass} M_earth'
            )

        # MgSiO3 mantle density check (2000-15000 kg/m3)
        # Upper bound higher than WB2018 because RTPress100TPa covers higher
        # pressures, yielding denser mantle material at depth for larger planets.
        mantle_densities = density[cmb_index:]
        if len(mantle_densities) > 0:
            mantle_nonzero = mantle_densities[mantle_densities > 0]
            if len(mantle_nonzero) > 0:
                max_mantle_rho = np.max(mantle_nonzero)
                min_mantle_rho = np.min(mantle_nonzero)
                assert min_mantle_rho >= 2000, (
                    f'Mantle density {min_mantle_rho:.0f} kg/m3 below MgSiO3 minimum (2000) '
                    f'for {id_mass} M_earth'
                )
                assert max_mantle_rho <= 15000, (
                    f'Mantle density {max_mantle_rho:.0f} kg/m3 above MgSiO3 maximum (15000) '
                    f'for {id_mass} M_earth'
                )
