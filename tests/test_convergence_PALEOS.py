"""Integration tests for the PALEOS-2phase:MgSiO3 EOS at <= 5 M_earth.

The light convergence checks live here; the heavy adiabatic mass scan
(3-10 M_earth) is in ``test_convergence_PALEOS_high_mass.py`` so xdist's
``--dist loadfile`` runs the two tiers on different workers.

Linear and adiabatic modes are both exercised. Adiabatic uses T(P)
parameterisation to avoid PALEOS table NaN gaps in the Brent pressure
solver's bracket search.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests._paleos_helpers import _paleos_data_available, _run_paleos
from zalmoxis.constants import earth_radius

# ── Linear mode convergence ────────────────────────────────────────────


@pytest.mark.integration
def test_PALEOS_converges_1Mearth():
    """PALEOS-2phase:MgSiO3 should converge for a 1 M_earth planet (linear T mode)."""
    if not _paleos_data_available():
        pytest.skip('PALEOS data files not found')

    results = _run_paleos(1.0, temperature_mode='linear')

    assert results['converged'], 'PALEOS model did not converge for 1 M_earth'

    R = results['radii'][-1] / earth_radius
    assert 0.8 < R < 1.3, f'PALEOS 1 M_earth radius {R:.3f} R_earth out of expected range'


@pytest.mark.integration
def test_PALEOS_converges_5Mearth():
    """PALEOS-2phase:MgSiO3 should converge for a 5 M_earth super-Earth."""
    if not _paleos_data_available():
        pytest.skip('PALEOS data files not found')

    results = _run_paleos(5.0, temperature_mode='linear')

    assert results['converged'], 'PALEOS model did not converge for 5 M_earth'

    R = results['radii'][-1] / earth_radius
    assert 1.2 < R < 2.0, f'PALEOS 5 M_earth radius {R:.3f} R_earth out of expected range'


# ── Adiabatic mode: issue #55 fix ─────────────────────────────────────


@pytest.mark.integration
def test_PALEOS_adiabatic_differs_from_linear():
    """Adiabatic mode should produce different R and T_center than linear mode.

    This is the key test for issue #55: previously adiabatic mode gave
    identical results to linear because the convergence loop broke before
    the adiabat could activate.
    """
    if not _paleos_data_available():
        pytest.skip('PALEOS data files not found')

    results_linear = _run_paleos(1.0, temperature_mode='linear')
    results_adiabatic = _run_paleos(1.0, temperature_mode='adiabatic')

    assert results_linear['converged'], 'Linear mode did not converge'
    assert results_adiabatic['converged'], 'Adiabatic mode did not converge'

    R_linear = results_linear['radii'][-1]
    R_adiabatic = results_adiabatic['radii'][-1]
    T_center_linear = results_linear['temperature'][0]
    T_center_adiabatic = results_adiabatic['temperature'][0]

    R_diff = abs(R_adiabatic - R_linear) / R_linear
    assert R_diff > 1e-4, (
        f'Adiabatic and linear radii are too similar: '
        f'R_linear={R_linear / earth_radius:.5f}, '
        f'R_adiabatic={R_adiabatic / earth_radius:.5f}, '
        f'relative diff={R_diff:.2e}'
    )

    T_diff = abs(T_center_adiabatic - T_center_linear)
    assert T_diff > 10, (
        f'Adiabatic and linear center temperatures too similar: '
        f'T_linear={T_center_linear:.1f} K, T_adiabatic={T_center_adiabatic:.1f} K'
    )


@pytest.mark.integration
def test_PALEOS_adiabatic_physically_reasonable():
    """Adiabatic T profile from PALEOS should have physically reasonable properties."""
    if not _paleos_data_available():
        pytest.skip('PALEOS data files not found')

    results = _run_paleos(1.0, temperature_mode='adiabatic')
    assert results['converged'], 'Adiabatic mode did not converge'

    T = results['temperature']

    assert np.all(np.isfinite(T)), 'Temperature profile has non-finite values'
    assert T[0] > T[-1], f'Center T ({T[0]:.0f} K) should exceed surface T ({T[-1]:.0f} K)'
    assert T[0] < 15000, f'Center temperature {T[0]:.0f} K unreasonably high'
    assert T[0] > 3000, f'Center temperature {T[0]:.0f} K unreasonably low'
