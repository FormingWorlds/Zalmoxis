import os
import pytest
import numpy as np
from scipy.interpolate import interp1d
from tools.setup_tests import run_zalmoxis_rocky_water, load_Seager_data, load_profile_output

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

@pytest.mark.parametrize("config_type,seager_file", [
    ("rocky", "radiusdensitySeagerEarthbymass.txt"),
    ("water", "radiusdensitySeagerwaterbymass.txt"),
])
@pytest.mark.parametrize("mass", [1, 5, 10, 50])  # 1, 5, 10, and 50 Earth masses
def test_density_profile(config_type, seager_file, mass):
    """
    Test the density profile for rocky and water planets using Zalmoxis model.
    This test compares the model output with Seager et al. (2007) density profile data.
    Parameters:
        config_type (str): Type of planet configuration ('rocky' or 'water').
        seager_file (str): Filename for Seager et al. (2007) radius and density data.
        mass (int): Mass of the planet in Earth masses (1, 5, 10, or 50).
    """

    # Load Seager et al. (2007) radius and density data by mass and order them by radius
    data_by_mass = load_Seager_data(seager_file)
    seager_radii = np.array(data_by_mass[mass]["radius"])
    seager_densities = np.array(data_by_mass[mass]["density"])
    sorted_indices = np.argsort(seager_radii)
    seager_radii = seager_radii[sorted_indices]
    seager_densities = seager_densities[sorted_indices]

    # Run the Zalmoxis model for the specified mass and configuration type
    output_file, profile_output_file = run_zalmoxis_rocky_water(mass, config_type, cmf=0.065, immf=0.485)

    # Load the model output for the density profile 
    model_radii, model_densities = load_profile_output(profile_output_file)
    model_radii = np.array(model_radii)/1e3  # Convert from meters to kilometers
    model_densities = np.array(model_densities) 

    # Interpolate Seager densities onto model radii
    interp_func = interp1d(seager_radii, seager_densities, bounds_error=False, fill_value="extrapolate")
    seager_density_interp = interp_func(model_radii)

    # Detect large jump in model densities (likely the core–mantle boundary)
    density_jumps = np.abs(np.diff(model_densities))
    jump_threshold = 2000  # Threshold for detecting a jump in density
    jump_indices = np.where(density_jumps > jump_threshold)[0]

    # Mask out radius points near the jump
    mask = np.ones_like(model_densities, dtype=bool)
    for idx in jump_indices:
        mask[max(0, idx-1):min(len(mask), idx+2)] = False  # mask a few points around the jump

    # Compare only the smooth parts
    assert np.allclose(model_densities[mask], seager_density_interp[mask], rtol=0.24, atol=1000), \
        f"Density profile for {config_type} config at {mass} M_earth deviates too much from Seager model (ignoring discontinuity)"
