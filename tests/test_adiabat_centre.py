"""The centre of an adiabatic solve against an independent integration of its core adiabat."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from tests.test_jax_temperature_arrays import _INPUT_DIR, _load_fixtures
from zalmoxis import solver as zs
from zalmoxis.constants import earth_mass
from zalmoxis.mixing import get_mixed_nabla_ad, parse_all_layer_mixtures
from zalmoxis.structure_model import is_core

pytestmark = [pytest.mark.integration, pytest.mark.timeout(3000)]

CENTRE_BOUND = 0.5  # K: 3 x the largest miss over 5 round-off starts (0.153 K)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_centre_of_a_10_earth_mass_core_lies_on_its_adiabat():
    """The 10 M_E planet of tests/data/bench_performance.toml on its own adiabat from 3000 K
    (numpy): a DOP853 integration of d ln T / d ln P = nabla_ad of the core mixture (the same
    tables, a cache of its own) from the top core node of the returned profile to its central
    pressure gives the centre T within CENTRE_BOUND, and the centre nodes do not share one T.
    The T(P) table of a pass is built on the pass before, so the central pressure can lie
    above the table top; there the lookup follows the adiabat slope of its last interval."""
    cp, mat, mf = _load_fixtures()
    cp.update(planet_mass=10 * earth_mass, surface_temperature=3000.0)
    res = zs.main(cp, mat, mf, _INPUT_DIR)
    assert res['converged'] and res['adiabat_blend'] == 1.0
    m, p, t = (np.asarray(res[k], float) for k in ('mass_enclosed', 'pressure', 'temperature'))
    top = int(np.flatnonzero(is_core(m, res['cmb_mass']))[-1])
    mix = parse_all_layer_mixtures(cp['layer_eos_config'])['core']
    cache = {}

    def rhs(x, y):
        return [get_mixed_nabla_ad(math.exp(x), math.exp(y[0]), mix, mat, cache, *mf)]

    ivp = solve_ivp(
        rhs, (math.log(p[top]), math.log(p[0])), [math.log(t[top])], method='DOP853', rtol=1e-10
    )
    assert ivp.status == 0
    assert abs(t[0] - math.exp(ivp.y[0, -1])) < CENTRE_BOUND, (t[0], math.exp(ivp.y[0, -1]))
    assert len(set(t[:3])) == 3, t[:3]
