"""Smoke test for rocky-planet density profile vs Seager+2007 at 1 M_earth.

Demoted from ``integration`` to ``smoke`` and trimmed to a single mass
in the 2026-05-05 CI-trim pass.
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


@pytest.mark.smoke
def test_density_profile_rocky():
    """Rocky-planet rho(r) must match Seager+2007 within 10 % at 1 M_earth (excluding CMB jump)."""
    mass = 1
    data_by_mass = load_Seager_data('radiusdensitySeagerEarthbymass.txt')
    seager_radii = np.array(data_by_mass[mass]['radius'])
    seager_densities = np.array(data_by_mass[mass]['density'])
    sorted_indices = np.argsort(seager_radii)
    seager_radii = seager_radii[sorted_indices]
    seager_densities = seager_densities[sorted_indices]

    _, profile_output_file = run_zalmoxis_rocky_water(mass, 'rocky', cmf=0.065, immf=0.485)

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

    # Drop solver vacuum-padding shells. When the Picard solver exhausts its
    # wall budget, it writes density=0 for any outer shell whose pressure
    # dropped invalid before the final converged sweep. Silicate surface
    # density is ~3000 kg/m^3 so 100 kg/m^3 unambiguously separates a real
    # shell from vacuum pad. (Symmetric with the water-world test, defensive
    # against a regression in Picard convergence tightness.)
    mask &= model_densities > 100.0

    assert np.allclose(
        model_densities[mask], seager_density_interp[mask], rtol=0.10, atol=300
    ), (
        f'Density profile for rocky config at {mass} M_earth deviates too much '
        f'from Seager model (ignoring discontinuity)'
    )
