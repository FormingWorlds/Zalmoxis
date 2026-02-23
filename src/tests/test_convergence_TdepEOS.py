from __future__ import annotations

import os

import numpy as np
import pytest

from tools.setup_tests import run_zalmoxis_TdepEOS


@pytest.mark.integration
@pytest.mark.parametrize(
    'mass', [1, 2]
)  # WolfBower2018 EOS tables are limited to ~1 TPa; only valid for <= 2 M_earth
def test_all_compositions_converge(mass, zalmoxis_root):
    """Test that the T-dependent EOS model converges for low-mass planets.

    The WolfBower2018:MgSiO3 EOS tables cover pressures up to ~1 TPa,
    which limits their applicability to planets <= 2 M_earth. For higher
    masses, deep-mantle pressures near the CMB exceed the table boundary.

    Verifies convergence and checks that the resulting density profiles are
    physically consistent: iron core densities >= 8000 kg/m3 (catches the
    old bug where MgSiO3 density was used for the core), mantle densities
    in the expected range for T-dependent MgSiO3.
    """
    print(f'Running test for mass = {mass}')

    # Delete composition_TdepEOS_mass_log file if it exists
    custom_log_file = os.path.join(
        zalmoxis_root, 'output_files', 'composition_TdepEOS_mass_log.txt'
    )
    if os.path.exists(custom_log_file):
        os.remove(custom_log_file)

    # Run Zalmoxis for a given mass with temperature-dependent EOS
    results = run_zalmoxis_TdepEOS(mass)

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
        # The center of the planet is iron; density depends on pressure.
        # Seager2007 iron EOS gives ~18000 kg/m3 at 1 TPa and ~37000 at
        # 10 TPa. The lower bound catches the old bug where MgSiO3 density
        # (~5000 kg/m3) was accidentally used for the core.
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

        # MgSiO3 mantle density check (2000-8000 kg/m3)
        # Mantle starts at CMB and extends to surface. Lower bound accounts
        # for molten MgSiO3 near the surface at T_surface=3500K and low P
        # (WolfBower2018 melt density ~2200 kg/m3 at surface conditions).
        # Upper bound allows for high-pressure compressed MgSiO3 near CMB.
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
                assert max_mantle_rho <= 8000, (
                    f'Mantle density {max_mantle_rho:.0f} kg/m3 above MgSiO3 maximum (8000) '
                    f'for {id_mass} M_earth'
                )
