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
    pytest.mark.timeout(4 * WALL_TIMEOUT),
]


@pytest.mark.physics_invariant
def test_round_off_starts_keep_the_centre_temperature_within_12_k(monkeypatch):
    """4 starts of input/default.toml at 2515 K and a mass tolerance of 1e-3, planet mass x
    (1 + k 1e-12), wall clock 2400 s: all are accepted at blend 1 within the mass tolerance, not
    on the wall clock, in the same padding state, and the adiabats of their returned structures
    have centre temperatures within 12 K (a padding flip moves the centre by tens of K, so the
    state is asserted first). The returned core T, read from the T(P) table of the pass before,
    is printed and not bounded: it can lie 20 K off."""
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
    md = load_material_dictionaries()
    inp = os.path.join(get_zalmoxis_root(), 'input')
    T_c, padded = [], []
    for k in ap.DEFAULT_STARTS:
        cfg = dict(ap._default_config(k), wall_timeout=WALL_TIMEOUT)
        res = zs.main(cfg, md, ap._melt('config', cfg), inp)
        got = (res['converged'], res['adiabat_blend'], res['accepted_on'])
        assert got[:2] == (True, 1.0) and got[2] != 'timeout', (k, got)
        err = res['mass_error_at_accept']
        assert err <= cfg['tolerance_outer'], (k, err)
        own = native(
            *(res[x] for x in ('radii', 'pressure', 'mass_enclosed')),
            *last['a'][3:],
            **last['kw'],
        )
        T_c.append(float(own[0]))
        padded.append(bool(res['padded']))
        print(k, 'returned T_c', float(res['temperature'][0]), 'own adiabat T_c', T_c[-1])
    print('centre temperatures [K]:', T_c, 'padded:', padded)
    assert len(set(padded)) == 1, padded
    assert np.ptp(T_c) <= 12.0, T_c
    assert all(5000.0 < t < 7000.0 for t in T_c)  # scale: an Earth-mass rocky centre
