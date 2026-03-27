"""Integration tests for the PALEOS-2phase:MgSiO3 EOS convergence.

Tests that the full solver converges with PALEOS tables for both linear
and adiabatic temperature modes across a range of planetary masses.
The adiabatic mode uses T(P) parameterization to avoid PALEOS table
NaN gaps during the Brent pressure solver's bracket search.

See also:
- docs/test_infrastructure.md
- docs/test_categorization.md
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from zalmoxis.constants import earth_mass, earth_radius


def _paleos_data_available():
    """Check if PALEOS data files are available."""
    root = os.environ.get('ZALMOXIS_ROOT', '')
    solid = os.path.join(
        root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
    )
    liquid = os.path.join(
        root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_liquid.dat'
    )
    return os.path.isfile(solid) and os.path.isfile(liquid)


def _run_paleos(mass_earth, temperature_mode='linear'):
    """Run Zalmoxis with PALEOS-2phase:MgSiO3 EOS for a given mass.

    Parameters
    ----------
    mass_earth : float
        Planet mass in Earth masses.
    temperature_mode : str
        Temperature mode ('linear' or 'adiabatic').

    Returns
    -------
    dict
        Model results dictionary.
    """
    from src.zalmoxis import zalmoxis
    from zalmoxis.zalmoxis import load_solidus_liquidus_functions

    root = os.environ['ZALMOXIS_ROOT']
    default_config_path = os.path.join(root, 'input', 'default.toml')
    config_params = zalmoxis.load_zalmoxis_config(default_config_path)

    config_params['planet_mass'] = mass_earth * earth_mass
    config_params['layer_eos_config'] = {
        'core': 'Seager2007:iron',
        'mantle': 'PALEOS-2phase:MgSiO3',
    }
    config_params['temperature_mode'] = temperature_mode
    config_params['data_output_enabled'] = False
    config_params['plotting_enabled'] = False
    config_params['verbose'] = False

    layer_eos_config = config_params['layer_eos_config']

    model_results = zalmoxis.main(
        config_params,
        material_dictionaries=zalmoxis.load_material_dictionaries(),
        melting_curves_functions=load_solidus_liquidus_functions(
            layer_eos_config,
            config_params.get('rock_solidus', 'Stixrude14-solidus'),
            config_params.get('rock_liquidus', 'Stixrude14-liquidus'),
        ),
        input_dir=os.path.join(root, 'input'),
    )
    return model_results


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


# ── Higher-mass adiabatic convergence (T(P) parameterization) ──────────


@pytest.mark.integration
@pytest.mark.parametrize('mass', [3, 5, 10])
def test_PALEOS_adiabatic_high_mass_converges(mass):
    """PALEOS adiabatic mode should converge for higher-mass planets.

    Regression test for the T(P) parameterization fix. Previously, the
    Brent pressure solver's bracket search created unphysical (low P, high T)
    queries that hit NaN gaps in the PALEOS tables, causing 14%+ mass errors
    and non-convergence for planets above ~2.8 M_earth.
    """
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
    """Planet radius should increase with mass for Earth-like composition."""
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
