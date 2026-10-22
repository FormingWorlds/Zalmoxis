"""The density that ``_solve`` gives the padded surface nodes of an accepted structure.

``zalmoxis.solver._surface_density`` looks the padded nodes up at the target surface pressure
and their own temperatures, in the layer of the stop mass, with the volatile profile only in
the mantle. ``_adiabat_to_surface`` continues a CMB-anchored adiabat from the last node with
P > 0 to the target surface pressure.
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
def test_a_surface_density_that_is_not_finite_is_returned_for_the_invariants(monkeypatch, bad):
    """An accepted result with it breaks positive; an unaccepted trial keeps it, as other
    failed profiles do, so a Newton trial does not stop the solve."""
    _lookup(monkeypatch, rho=lambda P, T: np.where(T > 3000.0, bad, 2300.0))
    rho = zs._surface_density(
        np.array([2900.0, 3100.0]),
        3.0e24,
        101325.0,
        CMB,
        TOP,
        {'core': 1, 'mantle': 2},
        EOS_ARGS,
        None,
    )
    assert rho[0] == 2300.0 and (np.isnan(rho[1]) if np.isnan(bad) else rho[1] == bad)


class _Mix:
    def __init__(self, tdep=True):
        self.tdep = tdep

    def has_tdep(self):
        return self.tdep


# get_mixed_nabla_ad takes the interpolation cache before the melting curves
NABLA_ARGS = ('md', 'cache', 'sol', 'liq', 'mzf', 'rho_min', 'rho_scale', 'binodal')


def _nabla(monkeypatch, f):
    calls = []

    def nabla(P, T, mix, *args):
        calls.append((P, T, mix, args))
        return f(P, T)

    monkeypatch.setattr(zs, 'get_mixed_nabla_ad', nabla)
    return calls


@pytest.mark.parametrize('a, b', [(0.03, 0.0), (0.05, -1e-3)])
def test_the_surface_continuation_integrates_d_ln_t_d_ln_p_exactly(monkeypatch, a, b):
    """For nabla = a + b ln P, ln T = ln T0 + a ln(p / P0) + b / 2 (ln^2 p - ln^2 P0), which
    the midpoint rule in ln P integrates exactly."""
    calls = _nabla(monkeypatch, lambda P, T: a + b * np.log(P))
    P0, T0, ps = 3.77e9, 2493.6, 101325.0
    T = zs._adiabat_to_surface(P0, T0, ps, _Mix(), EOS_ARGS)
    exact = T0 * np.exp(a * np.log(ps / P0) + b / 2 * (np.log(ps) ** 2 - np.log(P0) ** 2))
    assert T == pytest.approx(exact, rel=1e-12) and T < T0
    assert calls and all(c[3] == NABLA_ARGS for c in calls)
    assert min(c[0] for c in calls) >= ps and max(c[0] for c in calls) <= P0 * (1 + 1e-12)


@pytest.mark.parametrize(
    'P0, ps, mix, f',
    [
        (3.77e9, 101325.0, _Mix(tdep=False), lambda P, T: 0.03),
        (3.77e9, 101325.0, _Mix(), lambda P, T: None),
        (3.77e9, 101325.0, _Mix(), lambda P, T: -0.01),
        (1.0e5, 101325.0, _Mix(), lambda P, T: 0.03),
        (3.77e9, 0.0, _Mix(), lambda P, T: 0.03),
    ],
)
def test_the_surface_continuation_holds_t_without_a_gradient_or_a_pressure_drop(
    monkeypatch, P0, ps, mix, f
):
    _nabla(monkeypatch, f)
    assert zs._adiabat_to_surface(P0, 2493.6, ps, mix, EOS_ARGS) == 2493.6


def test_the_surface_continuation_predicts_the_midpoint_temperature(monkeypatch):
    """For nabla = a T / T0, T = T0 / (1 - a ln(p / P0)); the midpoint step needs the
    predicted temperature to reach it within 1e-4 (without it the error is about 1e-3)."""
    a, P0, T0, ps = 0.05, 3.77e9, 2493.6, 101325.0
    _nabla(monkeypatch, lambda P, T: a * T / T0)
    T = zs._adiabat_to_surface(P0, T0, ps, _Mix(), EOS_ARGS)
    assert T == pytest.approx(T0 / (1 - a * np.log(ps / P0)), rel=1e-4)


def test_the_surface_continuation_holds_t_when_the_midpoint_gradient_is_missing(monkeypatch):
    _nabla(monkeypatch, lambda P, T: 0.03 if P >= 3.76e9 else None)  # valid at P0 only
    assert zs._adiabat_to_surface(3.77e9, 2493.6, 101325.0, _Mix(), EOS_ARGS) == 2493.6


def test_the_surface_continuation_returns_a_non_finite_start_temperature_as_it_is(monkeypatch):
    calls = _nabla(monkeypatch, lambda P, T: 0.03)
    T = zs._adiabat_to_surface(3.77e9, np.nan, 101325.0, _Mix(), EOS_ARGS)
    assert np.isnan(T) and not calls
