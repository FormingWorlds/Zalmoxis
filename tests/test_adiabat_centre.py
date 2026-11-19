"""The adiabat of a solve against independent DOP853 integrations of nabla_ad: just above the
CMB, at the top core node and at the centre."""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tests.test_jax_temperature_arrays import _INPUT_DIR, _load_fixtures
from zalmoxis import solver as zs
from zalmoxis.config import (
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
)
from zalmoxis.constants import earth_mass
from zalmoxis.mixing import get_mixed_nabla_ad, parse_all_layer_mixtures
from zalmoxis.structure_model import is_core

pytestmark = [pytest.mark.integration, pytest.mark.timeout(3000)]

# K, from the 4 cases of the test (macOS arm64): 2.5 to 7 x the largest miss
MANTLE_BOUND = 1.5  # first mantle node (largest 0.55 K)
# top core node: the adiabat recomputed on the returned structure (node discretization up to
# 0.57 K); core lag: 0.18 to 6.2 K after a strict accept at 1 to 10 M_E, 31 K without the wait
LAG_BOUND, CORE_MARGIN = 10.0, 1.0
CENTRE_BOUND = 0.5  # centre (largest 0.07 K)


def _config(case):
    if case == '10me':
        cp, mat, mf = _load_fixtures()
        cp.update(planet_mass=10 * earth_mass, surface_temperature=3000.0)
        return cp, mat, mf
    cp = load_zalmoxis_config(os.path.join(_INPUT_DIR, 'default.toml'))
    mf = load_solidus_liquidus_functions(
        cp['layer_eos_config'], cp['rock_solidus'], cp['rock_liquidus'], 1.0
    )
    return cp, load_material_dictionaries(), mf


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize(('case', 'k'), [('10me', 0), ('10me', 1), ('10me', 2), ('default', 0)])
def test_the_adiabat_meets_independent_integrations_at_the_cmb_and_the_centre(
    case, k, request, monkeypatch
):
    """A solve on its own adiabat (the 10 M_E planet of tests/data/bench_performance.toml from
    3000 K, numpy, planet mass x (1 + k 1e-12); the 1 M_E default) against DOP853 integrations
    of d ln T / d ln P = nabla_ad with the same tables (a cache of its own; rtol 1e-12 and atol
    1e-14, since at rtol 1e-10 DOP853 steps over the kink of nabla_ad at the edge of the iron
    mush band and misses the converged answer by 0.7 K on a 2515 K default core): of the mantle from
    1e11 Pa on the returned profile to the first mantle node above the CMB (MANTLE_BOUND) and
    on to the CMB, then of the core to the top core node (the adiabat recomputed on the
    returned structure there, CORE_MARGIN; ``adiabat_self_consistency_core``, the largest lag
    of the returned core T behind it, at most LAG_BOUND), and of the core from the top core
    node to the centre (CENTRE_BOUND); the centre nodes do not share one T. The T(P) table
    of a pass is built on the pass before, so the returned core T lags the adiabat of the
    returned structure by up to the reported core lag; the checks need the lookup by own
    layer, the accept on a table of a pass within the mass tolerance, the end-slope
    continuation above the table top and the split of the node segment at the CMB."""
    cp, mat, mf = _config(case)
    cp['planet_mass'] *= 1.0 + k * 1e-12
    real, calls = zs.compute_adiabatic_temperature, []
    monkeypatch.setattr(
        zs,
        'compute_adiabatic_temperature',
        lambda *a, **kw: calls.append((a, real(*a, **kw))) or calls[-1][1],
    )
    res = zs.main(cp, mat, mf, _INPUT_DIR)
    assert res['converged'] and res['adiabat_blend'] == 1.0
    m, p, t = (np.asarray(res[x], float) for x in ('mass_enclosed', 'pressure', 'temperature'))
    (_, p_adi, m_adi, *_), adi = calls[-1]  # the adiabat recomputed on the returned structure
    assert np.array_equal(p_adi, p) and np.array_equal(m_adi, m)
    cmb = res['cmb_mass']
    top = int(np.flatnonzero(is_core(m, cmb))[-1])
    mixes, cache = parse_all_layer_mixtures(cp['layer_eos_config']), {}

    def dop(layer, p0, p1, t0, at=None):
        def rhs(x, y):
            nabla = get_mixed_nabla_ad(
                math.exp(x), math.exp(y[0]), mixes[layer], mat, cache, *(mf or (None, None))
            )
            return [nabla]

        ivp = solve_ivp(
            rhs,
            (math.log(p0), math.log(p1)),
            [math.log(t0)],
            method='DOP853',
            rtol=1e-12,
            atol=1e-14,
            t_eval=None if at is None else np.log(at),
        )
        assert ivp.status == 0
        return np.exp(ivp.y[0]) if at is not None else math.exp(ivp.y[0, -1])

    ok = p > 0
    t_1e11 = math.exp(np.interp(math.log(1e11), np.log(p[ok][::-1]), np.log(t[ok][::-1])))
    p_cmb = math.exp(np.interp(cmb, m[top : top + 2], np.log(p[top : top + 2])))
    t_first = dop('mantle', 1e11, p[top + 1], t_1e11)
    t_top = dop('core', p_cmb, p[top], dop('mantle', 1e11, p_cmb, t_1e11))
    t_centre = dop('core', p[top], p[0], t[top])
    assert abs(t[top + 1] - t_first) < MANTLE_BOUND, (t[top + 1], t_first)
    keys = {
        x: res[x] for x in ('accepted_on', 'adiabat_self_consistency', 'mass_error_at_accept')
    }
    lag, lag_top = res['adiabat_self_consistency_core'], abs(t[top] - adi[top])
    assert lag == np.abs(t - adi)[is_core(m, cmb) & (p > 0)].max(), (lag, keys)
    assert lag <= LAG_BOUND, (lag, keys)
    assert abs(adi[top] - t_top) <= CORE_MARGIN, (adi[top], t_top, keys)
    assert abs(t[0] - t_centre) < CENTRE_BOUND, (t[0], t_centre)
    assert len(set(t[:3])) == 3, t[:3]
    off = (t[top + 1] - t_first, t[top] - t_top, adi[top] - t_top, t[0] - t_centre)
    print(case, k, 'K off:', *off, 'lag', lag, lag_top)
    if (case, k) == ('10me', 0) and request.config.getoption('validation_plots') is not None:
        request.getfixturevalue('validation_plot')(
            lambda: _figure(p, t, top, p_cmb, t_1e11, dop, t_top), 'adiabat_cmb_centre'
        )


def _figure(p, t, top, p_cmb, t_1e11, dop, t_top):
    """Returned T minus DOP853: mantle nodes below 1e11 Pa against the mantle chain from 1e11 Pa,
    core nodes against the core chain from the returned top core node; the 3 checked nodes are
    marked, the top core node against the chain through the exact CMB."""
    import matplotlib.pyplot as plt

    mant = np.arange(top + 1, len(p))[p[top + 1 :] >= 1e11][::-1]  # inward: rising P
    core = np.arange(top + 1)[::-1]
    d_m = t[mant] - dop('mantle', 1e11, p[mant[-1]], t_1e11, at=p[mant])
    d_c = t[core] - dop('core', p[top], p[0], t[top], at=p[core])
    c = plt.rcParams['axes.prop_cycle'].by_key()['color']
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    ax.plot(p[mant] / 1e9, d_m, '-', color=c[0], label='mantle')
    ax.plot(p[core] / 1e9, d_c, '-', color=c[1], label='core')
    ax.plot(p[top + 1] / 1e9, d_m[-1], 'o', color=c[0], label='first mantle node')
    ax.plot(p[top] / 1e9, t[top] - t_top, 's', color=c[1], label='top core node')
    ax.plot(p[0] / 1e9, d_c[-1], 'D', color=c[1], mfc='none', label='centre')
    ax.axvline(p_cmb / 1e9, color=plt.rcParams['text.color'], lw=0.8, ls=':', label='CMB')
    ax.set(xlabel='Pressure [GPa]', ylabel='$\\Delta T$ [K]', xscale='log')
    ax.set_ylim(-2.0, 1.0)
    ax.grid(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
    fig.tight_layout()
    fig.canvas.draw()
    assert ax.get_legend().get_window_extent().x0 >= ax.get_window_extent().x1
    return fig
