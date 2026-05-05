"""Tests for the T-dependent EOS (WolfBower2018:MgSiO3) convergence.

Mirrors the RTPress100TPa test layout: the default-CI regression check
runs at 1 M_earth (``integration`` marker), while the heavier high-mass
sweep at 5 and 7 M_earth is gated behind ``slow`` and runs only with
``pytest -m slow``. The WolfBower2018 EOS tables natively cover ~1 TPa
(sufficient for ~2 M_earth); above that the Brent surface-pressure solver
clamps to the table edge so the structure still converges. The 7 M_earth
case is the validity-edge regime; 5 M_earth is the mid-mass clamping
regime; 1 M_earth is the in-table regime.

Trimmed from the previous [1, 2, 5, 7] sweep: the 2 M_earth cell shared
the same code path as the 1 M_earth cell (both fully in-table) without
adding a regime, so it was redundant cost-per-bug-caught.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tools.setup.setup_tests import run_zalmoxis_TdepEOS


def _check_TdepEOS_results(mass, results):
    """Shared physical-correctness assertions for a TdepEOS solver result.

    Verifies convergence and checks that the resulting density profiles are
    physically consistent: iron core densities >= 8000 kg/m^3 (catches the
    old bug where MgSiO3 density was used for the core), mantle densities
    in the expected range for T-dependent MgSiO3.
    """
    failed_cases = [(id_mass) for (id_mass, converged, _) in results if not converged]

    if failed_cases:
        failed_str = ', '.join([f'mass={id_mass:.2f}' for id_mass in failed_cases])
        pytest.fail(f'The following masses did not converge {mass}: {failed_str}')

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

        # Iron core density check (8000-50000 kg/m^3). Lower bound catches
        # the old bug where MgSiO3 density (~5000 kg/m^3) was accidentally
        # used for the core. Upper bound covers compressed iron at ~10 TPa.
        core_densities = density[1:cmb_index]
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

        # MgSiO3 mantle density check (2000-12000 kg/m^3). Lower bound
        # accounts for molten MgSiO3 near the surface at T_surface=3500K
        # and low P (WolfBower2018 melt density ~2200 kg/m^3). Upper bound
        # allows for high-pressure compressed MgSiO3 near CMB at higher
        # masses (5-7 M_earth) where pressures approach 1 TPa and clamped
        # WolfBower2018 densities can reach ~10000 kg/m^3.
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
                assert max_mantle_rho <= 12000, (
                    f'Mantle density {max_mantle_rho:.0f} kg/m3 above MgSiO3 maximum (12000) '
                    f'for {id_mass} M_earth'
                )


@pytest.mark.integration
def test_TdepEOS_converges_1Mearth(zalmoxis_root):
    """Default-CI regression check: T-dependent EOS converges at 1 M_earth.

    1 M_earth is the in-table regime (WolfBower2018 covers up to ~1 TPa
    natively, sufficient for Earth-mass pressures without clamping).
    Catches core/mantle density-routing regressions cheaply.
    """
    custom_log_file = os.path.join(zalmoxis_root, 'output', 'composition_TdepEOS_mass_log.txt')
    if os.path.exists(custom_log_file):
        os.remove(custom_log_file)

    results = run_zalmoxis_TdepEOS(1)
    _check_TdepEOS_results(1, results)


@pytest.mark.slow
@pytest.mark.parametrize('mass', [5, 7])
def test_TdepEOS_converges_high_mass(mass, zalmoxis_root):
    """High-mass T-dependent EOS convergence (5 and 7 M_earth).

    Gated behind ``slow`` because each parametrized cell adds ~50-65s of
    solver wall time and the regimes (boundary clamping at 5 M_earth,
    EOS-validity edge at 7 M_earth) are exercised by default only when
    the user opts in via ``pytest -m slow``. The 1 M_earth in-table
    regression check stays in the default ``integration`` suite above.
    """
    custom_log_file = os.path.join(zalmoxis_root, 'output', 'composition_TdepEOS_mass_log.txt')
    if os.path.exists(custom_log_file):
        os.remove(custom_log_file)

    results = run_zalmoxis_TdepEOS(mass)
    _check_TdepEOS_results(mass, results)
