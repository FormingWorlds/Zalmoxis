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

# K, from 3 round-off starts at 10 M_E and 3 of the 1 M_E default: about 3 x the largest miss
MANTLE_BOUND = 1.5  # first mantle node (largest 0.50 K)
CORE_BOUND = 3.0  # top core node (largest 1.40 K)
CENTRE_BOUND = 0.5  # centre (largest 0.15 K)


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
def test_the_adiabat_meets_independent_integrations_at_the_cmb_and_the_centre(case, k):
    """A solve on its own adiabat (the 10 M_E planet of tests/data/bench_performance.toml from
    3000 K, numpy, planet mass x (1 + k 1e-12); the 1 M_E default) against DOP853 integrations
    of d ln T / d ln P = nabla_ad with the same tables (a cache of its own; rtol 1e-12 and atol
    1e-14, since at rtol 1e-10 DOP853 steps over the kink of nabla_ad at the edge of the iron
    mush band and misses the converged answer by 0.7 K on a 2515 K default core): of the mantle from
    1e11 Pa on the returned profile to the first mantle node above the CMB (MANTLE_BOUND) and
    on to the CMB, then of the core to the top core node (CORE_BOUND), and of the core from
    the top core node to the centre (CENTRE_BOUND); the centre nodes do not share one T.
    The T(P) table of a pass is built on the pass before; the checks need the lookup by own
    layer, the accept on a table of a pass within the mass tolerance, the end-slope
    continuation above the table top and the split of the node segment at the CMB."""
    cp, mat, mf = _config(case)
    cp['planet_mass'] *= 1.0 + k * 1e-12
    res = zs.main(cp, mat, mf, _INPUT_DIR)
    assert res['converged'] and res['adiabat_blend'] == 1.0
    m, p, t = (np.asarray(res[x], float) for x in ('mass_enclosed', 'pressure', 'temperature'))
    cmb = res['cmb_mass']
    top = int(np.flatnonzero(is_core(m, cmb))[-1])
    mixes, cache = parse_all_layer_mixtures(cp['layer_eos_config']), {}

    def dop(layer, p0, p1, t0):
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
        )
        assert ivp.status == 0
        return math.exp(ivp.y[0, -1])

    ok = p > 0
    t_1e11 = math.exp(np.interp(math.log(1e11), np.log(p[ok][::-1]), np.log(t[ok][::-1])))
    p_cmb = math.exp(np.interp(cmb, m[top : top + 2], np.log(p[top : top + 2])))
    t_first = dop('mantle', 1e11, p[top + 1], t_1e11)
    t_top = dop('core', p_cmb, p[top], dop('mantle', 1e11, p_cmb, t_1e11))
    t_centre = dop('core', p[top], p[0], t[top])
    assert abs(t[top + 1] - t_first) < MANTLE_BOUND, (t[top + 1], t_first)
    assert abs(t[top] - t_top) < CORE_BOUND, (t[top], t_top)
    assert abs(t[0] - t_centre) < CENTRE_BOUND, (t[0], t_centre)
    assert len(set(t[:3])) == 3, t[:3]
