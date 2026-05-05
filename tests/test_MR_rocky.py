"""Smoke test for the rocky-planet code path at 1 M_earth.

One test, one solve: cross-verifies both the mass-radius relation
against Zeng+2019 and the density profile against Seager+2007 from
a single 1-M_earth rocky Seager solve. The two reference checks were
previously split across ``test_MR_rocky`` and ``test_Seager_rocky``;
they exercised identical solver state, so they were merged in the
2026-05-05 CI-trim pass to avoid testing the same code path twice.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import interp1d

from tools.setup.setup_tests import (
    load_model_output,
    load_profile_output,
    load_Seager_data,
    load_zeng_curve,
)


@pytest.mark.smoke
def test_rocky_1Mearth_vs_zeng_and_seager(cached_solver):
    """Rocky 1 M_earth Seager: R(M) within 3 % of Zeng+2019 AND rho(r) within 10 % of Seager+2007."""
    mass = 1
    # cmf=0.325, immf=0 matches the rocky-branch hardcoded composition in
    # ``run_zalmoxis_rocky_water`` and the cache key used by
    # ``test_analytic_MR``; one cached solve serves both files.
    output_file, profile_output_file = cached_solver(mass, 'rocky', cmf=0.325, immf=0)

    model_mass, model_radius = load_model_output(output_file)
    zeng_masses, zeng_radii = load_zeng_curve('massradiusEarthlikeRocky.txt')
    zeng_radius_interp = 10 ** np.interp(
        np.log10(model_mass), np.log10(zeng_masses), np.log10(zeng_radii)
    )
    assert np.isclose(model_radius, zeng_radius_interp, rtol=0.03), (
        f'rocky planet mass {model_mass} Earth masses: '
        f'model radius = {model_radius}, '
        f'Zeng radius = {zeng_radius_interp} (within 3 %)'
    )

    data_by_mass = load_Seager_data('radiusdensitySeagerEarthbymass.txt')
    seager_radii = np.array(data_by_mass[mass]['radius'])
    seager_densities = np.array(data_by_mass[mass]['density'])
    sorted_indices = np.argsort(seager_radii)
    seager_radii = seager_radii[sorted_indices]
    seager_densities = seager_densities[sorted_indices]

    model_radii, model_densities = load_profile_output(profile_output_file)
    model_radii = np.array(model_radii) / 1e3  # m -> km
    model_densities = np.array(model_densities)

    interp_func = interp1d(
        seager_radii, seager_densities, bounds_error=False, fill_value='extrapolate'
    )
    seager_density_interp = interp_func(model_radii)

    # Mask radius points around large jumps in modelled density (CMB
    # discontinuity); the CMB may fall between reference grid points,
    # so a +/-3-index mask avoids spurious failures from interpolation
    # across the sharp jump.
    density_jumps = np.abs(np.diff(model_densities))
    jump_indices = np.where(density_jumps > 2000)[0]
    mask = np.ones_like(model_densities, dtype=bool)
    for idx in jump_indices:
        mask[max(0, idx - 3) : min(len(mask), idx + 4)] = False
    # Drop solver vacuum-padding shells (density=0 outside the converged solution)
    mask &= model_densities > 100.0

    assert np.allclose(
        model_densities[mask], seager_density_interp[mask], rtol=0.10, atol=300
    ), (
        f'Density profile for rocky config at {mass} M_earth deviates too much '
        f'from Seager+2007 reference (ignoring CMB discontinuity).'
    )
