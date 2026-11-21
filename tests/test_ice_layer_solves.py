"""A solved 3-layer planet with a T-independent H2O layer on the adiabat cut at the mantle and
ice boundary.

1 M_earth, core 0.325, ice 0.1, Seager2007:H2O, PALEOS iron and MgSiO3, surface 300 K (below
the H2O critical point, as ``validate_config`` requires), 150 nodes.
See ``docs/Validation/thermal_state.md`` (case C).
"""

from __future__ import annotations

import inspect
import os

import numpy as np
import pytest

from tests._paleos_helpers import WALL_TIMEOUT
from tests._plot_checks import assert_text_clear, legend_below

pytestmark = [
    pytest.mark.slow,
    pytest.mark.paleos_adiabat,
    pytest.mark.timeout(1800),  # a 150 s solve on macOS arm64
]


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_a_solved_ice_planet_is_isothermal_in_the_ice_and_meets_the_cut_reference(
    monkeypatch, request
):
    """Case C: every ice node of the accepted planet holds 300 K to round-off, and the last
    adiabat of the solve, on the accepted pressures, stays within 0.005 K of the DOP853
    reference with the ice cut; the cut at ln P linear in m (0.04 K) and no cut fail."""
    from tests.test_adiabat_real_profiles import _need_paleos, _reference
    from zalmoxis import get_zalmoxis_root
    from zalmoxis import solver as zs
    from zalmoxis.config import (
        load_material_dictionaries,
        load_solidus_liquidus_functions,
        load_zalmoxis_config,
        validate_config,
    )

    _need_paleos()
    real, calls, n = zs.compute_adiabatic_temperature, [], 150
    sig = inspect.signature(real)

    def spy(*a, **k):
        T, kw = real(*a, **k), sig.bind(*a, **k).arguments
        if len(kw['pressure']) == n:  # a full profile, not the top shell
            calls.append((kw, np.array(T)))
        return T

    monkeypatch.setattr(zs, 'compute_adiabatic_temperature', spy)
    inp = os.path.join(get_zalmoxis_root(), 'input')
    cfg = load_zalmoxis_config(os.path.join(inp, 'default.toml'))
    cfg.update(
        data_output_enabled=False,
        plotting_enabled=False,
        surface_temperature=300.0,
        core_mass_fraction=0.325,
        mantle_mass_fraction=0.575,
        num_layers=n,
        wall_timeout=WALL_TIMEOUT,
        layer_eos_config={
            'core': 'PALEOS:iron',
            'mantle': 'PALEOS:MgSiO3',
            'ice_layer': 'Seager2007:H2O',
        },
    )
    validate_config(cfg)
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        'Stixrude14-solidus',
        'Stixrude14-liquidus',
        cfg['mushy_zone_factor'],
    )
    res = zs.main(cfg, load_material_dictionaries(), melt, inp)
    got = (res['converged'], res['adiabat_blend'], res['accepted_on'])
    assert got[:2] == (True, 1.0) and got[2] != 'timeout', got
    m, T = np.asarray(res['mass_enclosed']), np.asarray(res['temperature'], float)
    ice = m >= res['core_mantle_mass'] * (1 - 1e-12)
    assert ice.sum() > 5 and np.max(np.abs(T[ice] - 300.0)) <= 1e-9, T[ice]
    assert calls, 'no full-profile adiabat call recorded'
    kw, T_last = calls[-1]
    P = np.asarray(kw['pressure'], float)
    assert np.array_equal(P, np.asarray(res['pressure'], float)), 'not the accepted structure'
    ref, uncut = (_reference(kw, cut_ice=c)[0] for c in (True, False))
    err, off = (float(np.max(np.abs(T_last - x))) for x in (ref, uncut))
    print(f'last adiabat: {err:.3g} K from the cut reference, uncut reference {off:.3g} K off')
    assert err <= 0.005 < off, (err, off)
    if request.config.getoption('validation_plots') is None:
        return

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        on, floor = P > 0, 1e-6  # K: shows a node that meets the reference to round-off
        gpa = P[on] / 1e9
        ax.plot(gpa, np.maximum(np.abs(uncut - ref)[on], floor), '--', label='no cut')
        ax.plot(gpa, np.maximum(np.abs(T_last - ref)[on], floor), label='Zalmoxis')
        p_b = np.interp(res['core_mantle_mass'], m, P) / 1e9
        ax.axvline(p_b, alpha=0.4, lw=1.0, label='mantle and ice boundary')
        ax.set(
            xscale='log', yscale='log', xlabel='Pressure [GPa]', ylabel='|T - reference| [K]'
        )
        ax.minorticks_off()
        legend_below(fig, ncol=3)
        return assert_text_clear(fig)

    request.getfixturevalue('validation_plot')(figure, 'adiabat_ice_planet')
