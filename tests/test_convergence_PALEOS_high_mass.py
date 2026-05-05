"""High-mass adiabatic PALEOS-2phase:MgSiO3 convergence (3-10 M_earth).

Split out from ``test_convergence_PALEOS.py`` so xdist's
``--dist loadfile`` runs this heavy block on its own worker; the
single-worker wall otherwise dominated the integration suite.

Regression tests for the T(P) parameterisation fix. Previously, the
Brent pressure solver's bracket search created unphysical (low P, high T)
queries that hit NaN gaps in the PALEOS tables, causing 14%+ mass errors
and non-convergence above ~2.8 M_earth.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._paleos_helpers import _paleos_data_available, _run_paleos
from zalmoxis.constants import earth_mass, earth_radius


@pytest.mark.integration
@pytest.mark.parametrize('mass', [5, 10])
def test_PALEOS_adiabatic_high_mass_converges(mass):
    """PALEOS adiabatic mode should converge for higher-mass planets."""
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


@pytest.mark.integration
def test_PALEOS_adiabatic_radius_increases_with_mass():
    """Planet radius should increase with mass for Earth-like composition.

    Co-located with ``test_PALEOS_adiabatic_high_mass_converges`` so the
    5 and 10 M_earth solver runs land in the same lru_cache and this test
    only adds the 1 M_earth call as fresh work.
    """
    if not _paleos_data_available():
        pytest.skip('PALEOS data files not found')

    radii = []
    for mass in [1.0, 5.0, 10.0]:
        results = _run_paleos(mass, temperature_mode='adiabatic')
        assert results['converged'], f'{mass} M_earth did not converge'
        radii.append(results['radii'][-1])

    for i in range(1, len(radii)):
        assert radii[i] > radii[i - 1], (
            f'Radius did not increase: R({[1, 5, 10][i - 1]})={radii[i - 1] / earth_radius:.3f} '
            f'>= R({[1, 5, 10][i]})={radii[i] / earth_radius:.3f}'
        )
