"""The adiabat across the boundary between the mantle and an ice layer, on stub planets with a
constant nabla per layer, where the midpoint rule in ln P is exact.

See ``docs/Explanations/adiabat_integration.md`` (how the boundary is cut) and
``docs/Validation/thermal_state.md`` (case A).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from zalmoxis.eos.temperature import compute_adiabatic_temperature

M_E = 5.972e24
N = 31


class _Layer:
    """Mixture stub with a constant nabla, or T-independent (``nabla=None``)."""

    def __init__(self, nabla):
        self.nabla = nabla

    def has_tdep(self):
        return self.nabla is not None


@pytest.fixture
def stub_nabla(monkeypatch):
    """Route ``get_mixed_nabla_ad`` to the constant nabla of the layer it reads."""
    import zalmoxis.mixing as _mix

    monkeypatch.setattr(
        _mix, 'get_mixed_nabla_ad', lambda P, T, mixture, *a, **k: mixture.nabla
    )
    monkeypatch.setattr(_mix, 'any_component_is_tdep', lambda *a, **k: True)


def _planet(k, f, p_top=2e5):
    """Pressure [Pa] (geometric from 3.6e11 Pa to ``p_top``), mass [kg] (linear) and the ice
    boundary mass at the fraction ``f`` of the k-th node segment from the top, from its outer
    node."""
    P, m = np.geomspace(3.6e11, p_top, N), np.linspace(0.0, M_E, N)
    hi, lo = N - k, N - 1 - k
    return P, m, m[hi] + f * (m[lo] - m[hi])


def _solve(P, m, cmm, anchor, ice, mantle=0.2, **kw):
    layers = {'core': _Layer(None), 'mantle': _Layer(mantle), 'ice_layer': _Layer(ice)}
    return compute_adiabatic_temperature(
        radii=np.linspace(0.0, 6.4e6, len(P)),
        pressure=P,
        mass_enclosed=m,
        surface_temperature=1000.0 if anchor == 'surface' else 300.0,
        cmb_mass=kw.pop('cmb_mass', 0.5 * m[1]),
        core_mantle_mass=cmm,
        layer_mixtures=kw.pop('layer_mixtures', layers),
        material_dictionaries={},
        anchor=anchor,
        cmb_temperature=3000.0 if anchor == 'cmb' else None,
        **kw,
    )


def _closed_form(P, m, cmm, anchor, ice, mantle=0.2):
    """Piecewise power law T = T_a (P / P_a)^nabla per layer (anchor T_a 1000 K at the top or
    3000 K at the CMB node), the boundary at P linear in m
    between its 2 nodes; a T-independent ice layer holds T. Nodes below the CMB node 1 are
    left out (the CMB anchor carries the surface T there, the surface anchor holds the core)."""
    ice_node = m >= cmm * (1 - 1e-12)  # the layer rule of get_layer_mixture
    k = int(np.flatnonzero(~ice_node)[-1])  # deepest-up last mantle node
    p_b = float(np.interp(cmm, (m[k], m[k + 1]), (P[k], P[k + 1])))
    n_ice, T_anchor = (0.0 if ice is None else ice), (1000.0 if anchor == 'surface' else 3000.0)
    T = np.empty(len(P))
    if anchor == 'surface':
        t_b = T_anchor * (p_b / P[-1]) ** n_ice
        T[ice_node] = T_anchor * (P[ice_node] / P[-1]) ** n_ice
        T[~ice_node] = t_b * (P[~ice_node] / p_b) ** mantle
    else:
        t_b = T_anchor * (p_b / P[1]) ** mantle
        T[~ice_node] = T_anchor * (P[~ice_node] / P[1]) ** mantle
        T[ice_node] = t_b * (P[ice_node] / p_b) ** n_ice
    return T


@pytest.mark.unit
@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize('anchor', ['surface', 'cmb'])
@pytest.mark.parametrize('ice', [0.1, None], ids=['tdep_ice', 'inert_ice'])
@pytest.mark.parametrize('k', [1, 2, 5, 20])
def test_the_adiabat_follows_each_layer_across_the_ice_boundary(stub_nabla, anchor, ice, k):
    """Case A: the boundary in the middle of the k-th node segment from the top. Every node
    above the CMB node meets the piecewise power law to 1e-12 relative (mantle 0.2, ice 0.1 or
    T-independent; anchors 1000 K at the top and 3000 K at the CMB, so no node reaches a cap)."""
    P, m, cmm = _planet(k, 0.5)
    T = _solve(P, m, cmm, anchor, ice)
    np.testing.assert_allclose(T[1:], _closed_form(P, m, cmm, anchor, ice)[1:], rtol=1e-12)


@pytest.mark.unit
@pytest.mark.physics_invariant
@settings(max_examples=80, deadline=None)
@given(
    k=st.integers(1, N - 3),
    f=st.one_of(st.floats(0.0, 1.0), st.sampled_from([0.0, 1.0, 1e-12, 1 - 1e-12])),
    anchor=st.sampled_from(['surface', 'cmb']),
    ice=st.sampled_from([0.1, 0.2, None]),
    p_top=st.sampled_from([2e5, 1.2e5]),
)
def test_the_cut_meets_the_closed_form_anywhere_in_a_segment(k, f, anchor, ice, p_top):
    """The ice boundary at any fraction of any node segment above the CMB node, a boundary on a
    node included, both anchors: every node above the CMB node meets the closed form to 1e-12
    relative, with the boundary node in the layer of ``get_layer_mixture``."""
    import zalmoxis.mixing as _mix

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_mix, 'get_mixed_nabla_ad', lambda P, T, mixture, *a, **kw: mixture.nabla)
        mp.setattr(_mix, 'any_component_is_tdep', lambda *a, **kw: True)
        P, m, cmm = _planet(k, f, p_top)
        T = _solve(P, m, cmm, anchor, ice)
    np.testing.assert_allclose(T[1:], _closed_form(P, m, cmm, anchor, ice)[1:], rtol=1e-12)


@pytest.mark.unit
@settings(max_examples=60, deadline=None)
@given(
    frac=st.one_of(st.floats(0.0, 1.0), st.sampled_from([0.0, 1.0])),
    anchor=st.sampled_from(['surface', 'cmb']),
)
@example(frac=2.220446049250313e-16, anchor='surface')  # within 1e-9 of the CMB mass
def test_a_two_layer_profile_ignores_core_mantle_mass(frac, anchor):
    """Without an ice layer the profile is bit-equal for every ``core_mantle_mass`` from the
    CMB mass to the planet mass (a 2-layer caller may pass a mantle mass fraction above 0)."""
    import zalmoxis.mixing as _mix

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_mix, 'get_mixed_nabla_ad', lambda P, T, mixture, *a, **kw: mixture.nabla)
        mp.setattr(_mix, 'any_component_is_tdep', lambda *a, **kw: True)
        P, m, _ = _planet(1, 0.5)
        layers = {'core': _Layer(None), 'mantle': _Layer(0.2)}
        cmb = 0.5 * m[1]
        T = [
            _solve(P, m, c, anchor, None, layer_mixtures=layers)
            for c in (cmb, cmb + frac * (M_E - cmb))
        ]
    assert np.array_equal(T[0], T[1])


@pytest.mark.unit
@pytest.mark.physics_invariant
@settings(max_examples=60, deadline=None)
@given(
    eps=st.one_of(st.floats(1e-15, 0.5), st.sampled_from([1e-15, 1e-12, 1e-9])),
    anchor=st.sampled_from(['surface', 'cmb']),
)
def test_a_thin_mantle_inside_one_segment_keeps_the_profile_monotone(eps, anchor):
    """The CMB and the ice boundary inside one node segment (a mantle thinner than one segment,
    ``core_mantle_mass`` at the fraction ``eps`` of the segment above the CMB): the profile is
    finite and falls outward at every integrated node, no piece runs against the pressure
    order, and both cuts lie at P linear in m, which holds their order (with ln P linear in m
    for the CMB, the 2 cuts would swap for a mantle up to H/8 of the segment mass, H its ln P
    width)."""
    import zalmoxis.eos.temperature as _t
    import zalmoxis.mixing as _mix

    pieces, step = [], _t._segment_temperature

    def spy(T0, p_from, p_to, nabla_of, h, inward):
        pieces.append((p_from, p_to, inward))
        return step(T0, p_from, p_to, nabla_of, h, inward)

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(_mix, 'get_mixed_nabla_ad', lambda P, T, mixture, *a, **kw: mixture.nabla)
        mp.setattr(_mix, 'any_component_is_tdep', lambda *a, **kw: True)
        mp.setattr(_t, '_segment_temperature', spy)
        P, m = np.geomspace(3.6e11, 2e5, N), np.linspace(0.0, M_E, N)
        cmb = m[5] + 0.3 * (m[6] - m[5])
        cmm = cmb + eps * (m[6] - cmb)
        T = _solve(P, m, cmm, anchor, 0.1, cmb_mass=cmb)
    on = slice(None) if anchor == 'surface' else slice(6, None)
    assert np.all(np.isfinite(T)) and np.all(np.diff(T[on]) <= 0), T
    assert pieces and all((b >= a) if inward else (b <= a) for a, b, inward in pieces), pieces
    if anchor == 'surface':  # the CMB anchor starts above this segment
        ends = np.array([p for a, b, _ in pieces for p in (a, b)])
        for want in np.interp([cmb, cmm], m[5:7], P[5:7]):
            assert np.min(np.abs(ends / want - 1)) < 1e-12, (want, pieces)


@pytest.mark.unit
@pytest.mark.physics_invariant
def test_a_t_independent_layer_must_hold_one_temperature():
    """``_check_profile`` raises when a layer without a T-dependent component changes T over the
    integrated nodes, and keeps 2 such layers apart (each holds its own T)."""
    from zalmoxis.eos.temperature import _check_profile
    from zalmoxis.invariants import StructureInvariantError

    core, ice, mantle = _Layer(None), _Layer(None), _Layer(0.2)
    T = np.array([900.0, 900.0, 600.0, 300.0, 300.0])
    _check_profile(T, 0, [core, core, mantle, ice, ice])
    with pytest.raises(StructureInvariantError, match='changes in a T-independent layer'):
        _check_profile(T, 0, [core, core, mantle, core, core])
    with pytest.raises(StructureInvariantError, match='changes in a T-independent layer'):
        _check_profile(
            np.array([900.0, 900.0, 600.0, 300.0, 299.0]), 0, [core] * 2 + [mantle, ice, ice]
        )
