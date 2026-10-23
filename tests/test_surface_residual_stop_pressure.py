"""The surface residual across a stop above zero pressure.

A solve whose right-hand side turns non-finite at a pressure P_stop > 0, at most the target
surface pressure, stops there and is padded as the surface. Its residual P_stop - P_target -
k (R - r_stop) meets P(R) - P_target where the stop reaches R, since P(R) falls to P_stop
there; a residual that ignored P_stop would jump by P_stop at that central pressure. The
profile is the PALEOS setup of ``test_jax_rhs_parity`` with T at 0.6 of its values; T is
undefined below P_FLOOR, which stands for an EOS or temperature failure near the surface.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_jax_rhs_parity import _stage1b_setup  # noqa: E402

pytestmark = [pytest.mark.slow, pytest.mark.timeout(900)]

M_EARTH = 5.972e24
P_TARGET = 1e7  # a thick atmosphere: stops up to P_TARGET count as the surface [Pa]
P_FLOOR = 5e6  # T is undefined below this pressure [Pa]
RADIUS = 6.37e6
DP_CENTER = 1e5  # scan step in P_c; it moves f by about S DP_CENTER, well below P_stop [Pa]


@pytest.fixture(scope='module')
def junction_scan():
    """Residuals over central pressures across the one where the stop reaches R."""
    from zalmoxis.mixing import LayerMixture
    from zalmoxis.solver import _surface_residual
    from zalmoxis.structure_model import solve_structure

    setup = _stage1b_setup()
    eos = setup['config_params']['layer_eos_config']
    t_grid, t_values = setup['T_axis_grid'], 0.6 * np.asarray(setup['T_values'])

    def temperature(r, p):
        return np.nan if p < P_FLOOR else float(np.interp(np.log10(p), t_grid, t_values))

    radii = np.linspace(100.0, RADIUS, 50)
    kwargs = dict(
        layer_mixtures={k: LayerMixture([eos[k]], [1.0]) for k in ('core', 'mantle')},
        cmb_mass=setup['cmb_mass'],
        core_mantle_mass=setup['cmb_mass'] + 0.675 * M_EARTH,
        radii=radii,
        adaptive_radial_fraction=0.98,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-11,
        maximum_step=RADIUS * 0.004,
        material_dictionaries={
            eos['core']: setup['core_mat'],
            eos['mantle']: setup['mantle_mat'],
        },
        interpolation_cache=setup['interp_cache'],
        solidus_func=setup['sol_func'],
        liquidus_func=setup['liq_func'],
        temperature_function=temperature,
        mushy_zone_factors=setup['mushy_zone_factors'],
        surface_pressure=P_TARGET,
    )
    k = 2e4  # rho g of this profile near R [Pa/m]

    def residual(p_center):
        stop = {}
        p = solve_structure(**kwargs, y0=np.array([0.0, 0.0, p_center]), stop=stop)[2]
        return _surface_residual(np.asarray(p), stop, P_TARGET, RADIUS, k), dict(stop)

    coarse = np.linspace(2.5e11, 3.5e11, 21)
    reached = [not residual(p)[1] for p in coarse]
    i = reached.index(True)
    lo, hi = coarse[i - 1], coarse[i]
    while hi - lo > DP_CENTER:  # bisect to the central pressure where the stop reaches R
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if not residual(mid)[1] else (mid, hi)
    x = hi + DP_CENTER * np.arange(-20, 21)
    out = [residual(p) for p in x]
    return x, np.array([f for f, _ in out]), [s for _, s in out]


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_residual_has_no_jump_at_a_stop_above_zero_pressure(junction_scan, validation_plot):
    """Across the central pressure where a stop at P_stop > 0 reaches R, no step of f exceeds
    a tenth of P_stop, the jump a residual without P_stop would have there.

    Reference: the closed form of the stopped branch, f = P_stop - P_target - k (R - r_stop),
    which equals P(R) - P_target at r_stop = R.
    """
    x, y, stops = junction_scan
    stopped = np.array([bool(s) for s in stops])
    assert stopped.any() and not stopped.all(), 'the scan does not cross the stop at R'
    p_stop = np.array([s['p'] for s in stops if s])
    assert np.all((p_stop >= P_FLOOR) & (p_stop <= P_TARGET)), 'a stop below the floor'
    d = np.diff(y)
    j = int(np.flatnonzero(stopped[:-1] & ~stopped[1:])[0])
    assert abs(d[j]) < 0.1 * P_FLOOR, f'junction step {d[j]:.3e} Pa'
    assert np.abs(d).max() < 0.1 * P_FLOOR, f'largest step {np.abs(d).max():.3e} Pa'

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot((x[stopped] - x[0]) / 1e6, y[stopped] / 1e6, 'o', ms=3, label='stopped below R')
        ax.plot((x[~stopped] - x[0]) / 1e6, y[~stopped] / 1e6, 's', ms=3, label='reaches R')
        ax.axhline(
            -P_TARGET / 1e6, color=plt.rcParams['text.color'], ls='--', label='without P_stop'
        )
        ax.set(
            xlabel='Central pressure from the scan start [MPa]', ylabel='Surface residual [MPa]'
        )
        ax.legend()
        fig.tight_layout()
        return fig

    validation_plot(figure, 'surface_residual_stop_pressure')
