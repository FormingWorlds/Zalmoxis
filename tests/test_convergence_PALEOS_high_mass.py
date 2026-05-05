"""Smoke tests for high-mass adiabatic PALEOS-2phase:MgSiO3 convergence.

Demoted from ``integration`` to ``smoke`` and trimmed in the 2026-05-05
CI-trim pass: only the 5 M_earth adiabatic case is retained from the
prior [5, 10] parametrize. The integration tier (PALEOS rocky linear
1+5 M_earth) covers the published-reference comparison; this file
catches regressions in the T(P) parameterisation fix without re-paying
the 10 M_earth wall-time hit.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._paleos_helpers import _paleos_data_available, _run_paleos
from zalmoxis.constants import earth_mass, earth_radius


@pytest.mark.smoke
def test_PALEOS_adiabatic_5Mearth_converges():
    """PALEOS adiabatic mode should converge for a 5 M_earth super-Earth."""
    mass = 5
    if not _paleos_data_available():
        pytest.skip('PALEOS data files not found')

    results = _run_paleos(float(mass), temperature_mode='adiabatic')

    assert results['converged'], f'PALEOS adiabatic did not converge for {mass} M_earth'

    R = results['radii'][-1] / earth_radius
    T = results['temperature']
    M_calc = results['mass_enclosed'][-1]
    mass_error = abs(M_calc - mass * earth_mass) / (mass * earth_mass)

    # Mass error should be < 1% (the tolerance_outer is 0.3%)
    assert mass_error < 0.01, f'{mass} M_earth mass error {mass_error * 100:.2f}% exceeds 1%'

    # Physical checks
    assert np.all(np.isfinite(T)), f'{mass} M_earth has non-finite T values'
    assert T[0] > T[-1], f'{mass} M_earth center T should exceed surface T'

    # Radius should be in a reasonable range
    assert 0.5 < R < 3.0, f'{mass} M_earth radius {R:.3f} R_earth out of range'


@pytest.mark.smoke
def test_PALEOS_adiabatic_radius_increases_with_mass():
    """Planet radius should increase with mass for Earth-like composition.

    Trimmed to compare 1 and 5 M_earth (10 M_earth dropped in 2026-05-05
    CI-trim pass).
    """
    if not _paleos_data_available():
        pytest.skip('PALEOS data files not found')

    radii = []
    for mass in [1.0, 5.0]:
        results = _run_paleos(mass, temperature_mode='adiabatic')
        assert results['converged'], f'{mass} M_earth did not converge'
        radii.append(results['radii'][-1])

    assert radii[1] > radii[0], (
        f'Radius did not increase: R(1)={radii[0] / earth_radius:.3f} '
        f'>= R(5)={radii[1] / earth_radius:.3f}'
    )
