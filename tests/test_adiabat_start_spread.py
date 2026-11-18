"""Round-off starts of input/default.toml with the midpoint adiabat.

The 4 starts differ in planet mass by k 1e-12 (k = 0 to 3); they all end in one padding state,
and the adiabats of their returned structures have centre temperatures within 12 K. See
``docs/Validation/thermal_state.md``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests._paleos_helpers import WALL_TIMEOUT
from tools.validation import adiabat_profiles as ap

pytestmark = [
    pytest.mark.slow,
    pytest.mark.paleos_adiabat,
    pytest.mark.timeout(4200),  # inside the 90 min paleos_adiabat CI job, setup included
]


@pytest.mark.physics_invariant
def test_round_off_starts_keep_the_centre_temperature_within_12_k(monkeypatch, request):
    """4 starts of input/default.toml at 2515 K and a mass tolerance of 1e-3, planet mass x
    (1 + k 1e-12), wall clock 2400 s: all are accepted at blend 1 within the mass tolerance, not
    on the wall clock and without the retry with tighter parameters (a start sent there by the
    wall clock returns another structure, 19.4 K off at the centre), in the same padding state,
    and the adiabats of their returned structures have centre temperatures within 12 K (a
    padding flip moves the centre by tens of K, so the state is asserted first). The returned
    core T, read from the T(P) table of the pass before, is printed and not bounded: it lies up
    to 26 K off at a core node near the centre on macOS arm64."""
    import zalmoxis.solver as zs
    from tests.test_adiabat_real_profiles import _need_paleos
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import load_material_dictionaries

    _need_paleos()
    native, last = zs.compute_adiabatic_temperature, {}

    def hook(*a, **kw):
        last.update(a=a, kw=kw)
        return native(*a, **kw)

    monkeypatch.setattr(zs, 'compute_adiabatic_temperature', hook)
    tight, retries = zs._tighten_solver_params, []
    monkeypatch.setattr(zs, '_tighten_solver_params', lambda p: retries.append(p) or tight(p))
    md = load_material_dictionaries()
    inp = os.path.join(get_zalmoxis_root(), 'input')
    T_c, T_ret, padded = [], [], []
    for k in ap.DEFAULT_STARTS:
        cfg = dict(ap._default_config(k), wall_timeout=WALL_TIMEOUT)
        res = zs.main(cfg, md, ap._melt('config', cfg), inp)
        got = (res['converged'], res['adiabat_blend'], res['accepted_on'])
        assert got[:2] == (True, 1.0) and got[2] != 'timeout', (k, got)
        assert not retries, (k, 'retry with tighter parameters')
        err = res['mass_error_at_accept']
        assert err is not None and err <= cfg['tolerance_outer'], (k, err)
        own = native(
            *(res[x] for x in ('radii', 'pressure', 'mass_enclosed')),
            *last['a'][3:],
            **last['kw'],
        )
        T_c.append(float(own[0]))
        T_ret.append(float(res['temperature'][0]))
        padded.append(bool(res['padded']))
        print(k, 'returned T_c', float(res['temperature'][0]), 'own adiabat T_c', T_c[-1])
    print('centre temperatures [K]:', T_c, 'padded:', padded)
    assert len(set(padded)) == 1, padded
    assert np.ptp(T_c) <= 12.0, T_c
    assert all(5000.0 < t < 7000.0 for t in T_c)  # scale: an Earth-mass rocky centre

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8.0, 3.4))
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        k = np.arange(len(T_c))
        ax.axhspan(min(T_c), min(T_c) + 12.0, color=c[2], alpha=0.15, lw=0, label='12 K bound')
        ax.plot(k, T_c, 'o', color=c[0], label='adiabat of the returned structure')
        ax.plot(k, T_ret, 's', color=c[1], mfc='none', label='returned core T (not bounded)')
        lo, hi = min(T_c + T_ret), max(max(T_c + T_ret), min(T_c) + 12.0)
        ax.set(xlabel='Start $k$', ylabel='$T_\\mathrm{centre}$ [K]', xticks=k)
        ax.set(xlim=(-0.5, k[-1] + 0.5), ylim=(lo - 0.4 * (hi - lo), hi + 0.4 * (hi - lo)))
        ax.grid(False)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
        fig.tight_layout()
        fig.canvas.draw()
        assert ax.get_legend().get_window_extent().x0 >= ax.get_window_extent().x1
        return fig

    # the values move with round-off on each CPU, so the plot is made only with the option
    if request.config.getoption('validation_plots') is not None:
        request.getfixturevalue('validation_plot')(figure, 'adiabat_start_spread')
