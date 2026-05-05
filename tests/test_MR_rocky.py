"""Smoke test for rocky-planet mass-radius vs Zeng+2019 at 1 M_earth.

Demoted from ``integration`` to ``smoke`` and trimmed to a single mass
in the 2026-05-05 CI-trim pass. The full real-physics integration set
is now restricted to the two PALEOS Earth-like-rocky cases (1 M_earth
+ 5 M_earth) in ``test_convergence_PALEOS.py``; everything else lives
under ``smoke``: same code path, lower wall-time cost.
"""

from __future__ import annotations

import numpy as np
import pytest

from tools.setup.setup_tests import load_model_output, load_zeng_curve, run_zalmoxis_rocky_water


@pytest.mark.smoke
def test_mass_radius_rocky():
    """Rocky-planet R(M) must agree with Zeng+2019 within 3 % at 1 M_earth.

    Mass cells [5, 10] dropped in the CI-trim pass; the 1-M_earth cell
    exercises the same Picard solve path and catches regressions. The
    PALEOS integration tier (``test_convergence_PALEOS.py``) covers
    the super-Earth regime end-to-end.
    """
    zeng_masses, zeng_radii = load_zeng_curve('massradiusEarthlikeRocky.txt')

    output_file, _ = run_zalmoxis_rocky_water(1, 'rocky', cmf=0.1625, immf=0.3375)
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
