"""Smoke test for water-world mass-radius vs Zeng+2019 at 1 M_earth.

Demoted from ``integration`` to ``smoke`` and trimmed to a single mass
in the 2026-05-05 CI-trim pass (only PALEOS rocky 1+5 M_earth retained
as ``integration``).
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.setup.setup_tests import load_model_output, load_zeng_curve, run_zalmoxis_rocky_water


@pytest.mark.smoke
def test_mass_radius_water():
    """50 % H2O planet R(M) must agree with Zeng+2019 within 3 % at 1 M_earth."""
    zeng_masses, zeng_radii = load_zeng_curve('massradius_50percentH2O_300K_1mbar.txt')

    output_file, _ = run_zalmoxis_rocky_water(1, 'water', cmf=0.1625, immf=0.3375)
    model_mass, model_radius = load_model_output(output_file)

    zeng_radius_interp = 10 ** np.interp(
        np.log10(model_mass), np.log10(zeng_masses), np.log10(zeng_radii)
    )

    assert np.isclose(model_radius, zeng_radius_interp, rtol=0.03), (
        f'water planet mass {model_mass} Earth masses: '
        f'model radius = {model_radius}, '
        f'Zeng radius = {zeng_radius_interp} (within 3%)'
    )
