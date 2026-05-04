"""Mass-radius integration tests for rocky planets vs Zeng+2019.

Split from a former combined ``test_MR.py`` (rocky+water) so xdist's
``--dist loadfile`` runs rocky and water on different workers; mass=50
on both was the integration suite's largest single concentration of
solver wall time.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.setup.setup_tests import load_model_output, load_zeng_curve, run_zalmoxis_rocky_water


@pytest.mark.integration
@pytest.mark.parametrize('mass', [1, 5, 10, 50])
def test_mass_radius_rocky(mass):
    """Rocky-planet R(M) must agree with Zeng+2019 within 3% at 1, 5, 10, 50 M_earth."""
    zeng_masses, zeng_radii = load_zeng_curve('massradiusEarthlikeRocky.txt')

    output_file, _ = run_zalmoxis_rocky_water(mass, 'rocky', cmf=0.1625, immf=0.3375)
    model_mass, model_radius = load_model_output(output_file)

    # Log-log interpolation of Zeng curve (matches compare_zeng2019.py)
    zeng_radius_interp = 10 ** np.interp(
        np.log10(model_mass), np.log10(zeng_masses), np.log10(zeng_radii)
    )

    assert np.isclose(model_radius, zeng_radius_interp, rtol=0.03), (
        f'rocky planet mass {model_mass} Earth masses: '
        f'model radius = {model_radius}, '
        f'Zeng radius = {zeng_radius_interp} (within 3%)'
    )
