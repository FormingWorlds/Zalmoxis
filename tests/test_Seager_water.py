"""Density-profile integration tests for water-world planets vs Seager+2007.

Split from a former combined ``test_Seager.py`` (rocky+water) so xdist's
``--dist loadfile`` runs rocky and water on different workers.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import interp1d

from tools.setup.setup_tests import (
    load_profile_output,
    load_Seager_data,
    run_zalmoxis_rocky_water,
)


@pytest.mark.integration
@pytest.mark.parametrize('mass', [1, 5, 10, 50])
def test_density_profile_water(mass):
    """50% H2O planet rho(r) must match Seager+2007 within 10% (excluding CMB jump)."""
    data_by_mass = load_Seager_data('radiusdensitySeagerwaterbymass.txt')
    seager_radii = np.array(data_by_mass[mass]['radius'])
    seager_densities = np.array(data_by_mass[mass]['density'])
    sorted_indices = np.argsort(seager_radii)
    seager_radii = seager_radii[sorted_indices]
    seager_densities = seager_densities[sorted_indices]

    _, profile_output_file = run_zalmoxis_rocky_water(mass, 'water', cmf=0.065, immf=0.485)

    model_radii, model_densities = load_profile_output(profile_output_file)
    model_radii = np.array(model_radii) / 1e3  # m -> km
    model_densities = np.array(model_densities)

    interp_func = interp1d(
        seager_radii, seager_densities, bounds_error=False, fill_value='extrapolate'
    )
    seager_density_interp = interp_func(model_radii)

    density_jumps = np.abs(np.diff(model_densities))
    jump_indices = np.where(density_jumps > 2000)[0]
    mask = np.ones_like(model_densities, dtype=bool)
    for idx in jump_indices:
        mask[max(0, idx - 3) : min(len(mask), idx + 4)] = False

    assert np.allclose(
        model_densities[mask], seager_density_interp[mask], rtol=0.10, atol=300
    ), (
        f'Density profile for water config at {mass} M_earth deviates too much '
        f'from Seager model (ignoring discontinuity)'
    )
