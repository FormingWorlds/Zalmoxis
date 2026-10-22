"""The density that ``_solve`` gives the padded surface nodes of an accepted structure.

``zalmoxis.solver._surface_density`` looks the padded nodes up at the target surface pressure
and their own temperatures, in the layer of the stop mass, with the volatile profile only in
the mantle; a density that is not finite stops the solve.
"""

from __future__ import annotations

import numpy as np
import pytest

import zalmoxis.solver as zs

pytestmark = pytest.mark.unit

CMB, TOP = 2.0e24, 4.0e24
EOS_ARGS = ('md', 'sol', 'liq', 'cache', 'mzf', 'rho_min', 'rho_scale', 'binodal')
VP = object()


def _lookup(monkeypatch, rho=lambda P, T: 1000.0 + T):
    """Replace the batch EOS by a T-dependent stub that records each call."""
    calls = []

    def batch(P, T, mix, *args, volatile_profile=None):
        calls.append(
            dict(P=np.array(P), T=np.array(T), mix=mix, args=args, vp=volatile_profile)
        )
        return rho(np.asarray(P), np.asarray(T))

    monkeypatch.setattr(zs, 'calculate_mixed_density_batch', batch)
    return calls


@pytest.mark.parametrize(
    'm_stop, layers, layer, vp',
    [
        (3.0e24, ('core', 'mantle'), 'mantle', VP),
        (5.0e24, ('core', 'mantle', 'ice_layer'), 'ice_layer', None),
        (1.0e24, ('core', 'mantle'), 'core', None),
    ],
)
def test_the_lookup_reads_the_target_pressure_the_node_temperatures_and_the_stop_layer(
    monkeypatch, m_stop, layers, layer, vp
):
    calls = _lookup(monkeypatch)
    mixes = {name: f'mix-{name}' for name in layers}
    T = np.array([2900.0, 3100.0])
    rho = zs._surface_density(T, m_stop, 101325.0, CMB, TOP, mixes, EOS_ARGS, VP)
    assert len(calls) == 1
    (c,) = calls
    assert np.array_equal(c['P'], [101325.0, 101325.0]) and np.array_equal(c['T'], T)
    assert c['mix'] == f'mix-{layer}' and c['args'] == EOS_ARGS and c['vp'] is vp
    assert np.array_equal(rho, 1000.0 + T)


@pytest.mark.parametrize('bad', [np.nan, np.inf, -np.inf])
def test_a_surface_density_that_is_not_finite_stops_the_solve(monkeypatch, bad):
    _lookup(monkeypatch, rho=lambda P, T: np.where(T > 3000.0, bad, 2300.0))
    with pytest.raises(zs.StructureSolveError, match='surface density not finite'):
        zs._surface_density(
            np.array([2900.0, 3100.0]),
            3.0e24,
            101325.0,
            CMB,
            TOP,
            {'core': 1, 'mantle': 2},
            EOS_ARGS,
            None,
        )
