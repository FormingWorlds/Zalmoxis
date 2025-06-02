import os
import pytest
import numpy as np
from tools.setup_tests import run_zalmoxis_rocky_water, load_zeng_curve, load_model_output

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

@pytest.mark.parametrize("config_type,zeng_file", [
    ("rocky", "massradiusEarthlikeRockyZeng.txt"),
    ("water", "massradiuswaterZeng.txt"),
])
@pytest.mark.parametrize("mass", range(1, 51))  # 1 to 50 Earth masses
def test_mass_radius(config_type, zeng_file, mass):

    # Load Zeng et al. (2019) mass-radius data
    zeng_masses, zeng_radii = load_zeng_curve(zeng_file)

    # Run the Zalmoxis model for the specified mass and configuration type
    output_file, profile_output_file = run_zalmoxis_rocky_water(mass, config_type, cmf=0.1625, immf=0.3375)

    # Load the model output for mass and radius
    model_mass, model_radius = load_model_output(output_file)

    # Interpolate Zeng radius for the model mass
    zeng_radius_interp = np.interp(model_mass, zeng_masses, zeng_radii)

    # Compare model output with Zeng et al. (2019) data
    assert np.isclose(model_radius, zeng_radius_interp, rtol=0.03), (
        f"{config_type} planet mass {model_mass} Earth masses: "
        f"model radius = {model_radius}, "
        f"Zeng radius = {zeng_radius_interp} (within 3%)"
    )
