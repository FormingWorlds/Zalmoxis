"""Two-layer core: the iron side tables in the registry and the configuration of an inner
core."""

from __future__ import annotations

import math
import numbers
import sys
import types

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests import _paleos_mock
from zalmoxis.config import parse_eos_config, validate_config, validate_layer_eos
from zalmoxis.constants import G
from zalmoxis.core_profile import core_arrays, icb_from_profile
from zalmoxis.eos import core_alloy as ca
from zalmoxis.eos.paleos_api import make_default_grid_iron_2phase
from zalmoxis.eos_properties import EOS_REGISTRY
from zalmoxis.mixing import LayerMixture
from zalmoxis.phase_columns import compute_layer_phase_columns

pytestmark = pytest.mark.unit

LIQ, SOL = 'PALEOS-API-2phase:iron-liquid', 'PALEOS-API-2phase:iron-solid'


def _params(**over):
    """Valid config_params of a 1 M_E two-layer core with the ICB from a liquidus."""
    p = {
        'planet_mass': 5.972e24,
        'core_mass_fraction': 0.325,
        'mantle_mass_fraction': 0.0,
        'temperature_mode': 'adiabatic',
        'surface_temperature': 3000.0,
        'center_temperature': 6000.0,
        'temp_profile_file': '',
        'layer_eos_config': {
            'core': f'{LIQ}:0.8615+CoreAlloy:S:0.1385',
            'inner_core': SOL,
            'mantle': 'PALEOS:MgSiO3',
        },
        'rock_solidus': 'Monteux16-solidus',
        'rock_liquidus': 'Monteux16-liquidus-A-chondritic',
        'mushy_zone_factor': 1.0,
        'mushy_zone_factors': {'PALEOS:iron': 1.0, 'PALEOS:MgSiO3': 1.0, 'PALEOS:H2O': 1.0},
        'condensed_rho_min': 300.0,
        'condensed_rho_scale': 50.0,
        'binodal_T_scale': 50.0,
        'num_layers': 150,
        'target_surface_pressure': 101325.0,
        'data_output_enabled': False,
        'plotting_enabled': False,
        'core_liquidus': 'Anzellini13-iron',
    }
    p.update(over)
    return p


@pytest.mark.parametrize(('name', 'side'), [(LIQ, 'liquid'), (SOL, 'solid')])
def test_each_iron_side_is_a_registry_entry_on_the_unified_iron_box(name, side):
    """Each side resolves through the iron pair at 150 points per decade over 1e5-1e14 Pa and
    300-1e5 K, the box of the shipped unified iron table (whose T axis has 380 nodes)."""
    grid = make_default_grid_iron_2phase()
    assert (grid.p_lo, grid.p_hi, grid.t_lo, grid.t_hi) == (1e5, 1e14, 300.0, 1e5)
    assert (grid.n_p, grid.n_t) == (1351, 379)
    assert EOS_REGISTRY[name] == {
        'format': 'paleos_api_2phase_side',
        'material': 'iron',
        'side': side,
        'grid_spec': grid,
    }


@pytest.mark.parametrize('base', [LIQ, SOL])
def test_an_iron_side_is_a_light_element_base(base):
    assert ca.split_alloy(LayerMixture([base, 'CoreAlloy:S'], [0.9, 0.1])) == (base, {'S': 0.1})


def test_other_bases_stay_refused():
    with pytest.raises(ca.CoreAlloyError, match='iron base'):
        ca.split_alloy(LayerMixture(['PALEOS-API:iron', 'CoreAlloy:S'], [0.9, 0.1]))


def test_the_eos_section_carries_the_inner_core():
    eos = {'core': LIQ, 'mantle': 'PALEOS:MgSiO3', 'inner_core': SOL}
    assert parse_eos_config(eos) == eos
    assert 'inner_core' not in parse_eos_config({**eos, 'inner_core': ''})


def test_an_inner_core_takes_light_elements():
    validate_layer_eos(
        {'core': f'{LIQ}:0.9+CoreAlloy:S:0.1', 'inner_core': f'{SOL}:0.95+CoreAlloy:S:0.05'}
        | {'mantle': 'PALEOS:MgSiO3'}
    )


def test_the_icb_rule_is_the_liquidus_or_the_mass_never_both_nor_none():
    validate_config(_params())
    validate_config(_params(core_liquidus='', inner_core_mass_fraction=0.04))
    with pytest.raises(ValueError, match='exactly one'):
        validate_config(_params(inner_core_mass_fraction=0.04))
    with pytest.raises(ValueError, match='exactly one'):
        validate_config(_params(core_liquidus=''))
    one_layer = {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'}
    with pytest.raises(ValueError, match='inner_core'):
        validate_config(_params(layer_eos_config=one_layer))
    validate_config(_params(layer_eos_config=one_layer, core_liquidus=''))


@settings(max_examples=60, deadline=None)
@given(
    st.one_of(
        st.floats(allow_nan=True, allow_infinity=True),
        st.booleans(),
        st.integers(-3, 3),
        st.text(max_size=4),
        st.none(),
    )
)
def test_inner_core_mass_fraction_takes_a_float_in_0_1(value):
    """Wrong types (a TOML true, a string), NaN, inf, negative and above 1 are refused."""
    params = _params(core_liquidus='', inner_core_mass_fraction=value)
    ok = (
        isinstance(value, numbers.Real)
        and not isinstance(value, (bool, np.bool_))
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )
    if value is None:
        with pytest.raises(ValueError, match='exactly one'):
            validate_config(params)
    elif ok:
        validate_config(params)
    else:
        with pytest.raises(ValueError, match='inner_core_mass_fraction'):
            validate_config(params)


@settings(max_examples=40, deadline=None)
@given(st.one_of(st.text(max_size=20), st.booleans(), st.floats(), st.integers()))
def test_core_liquidus_takes_an_iron_melting_curve_name(value):
    params = _params(core_liquidus=value)
    if value in ('Anzellini13-iron', 'Sinmyo19-iron'):
        validate_config(params)
    else:
        with pytest.raises(ValueError, match='core_liquidus|exactly one'):
            validate_config(params)


def test_core_liquidus_may_be_a_callable():
    """A caller passes the liquidus of the outer composition as T_liq(P)."""
    validate_config(_params(core_liquidus=lambda p: 4000.0 + 1e-8 * p))
    with pytest.raises(ValueError, match='exactly one'):
        validate_config(_params(core_liquidus=lambda p: 4000.0, inner_core_mass_fraction=0.1))


def _profile(t_of_p, n=41):
    """Centre-first nodes of a core of 1e24 kg with P from 360 to 136 GPa, mass linear in ln P
    (the ICB rule interpolates in ln P)."""
    p = np.linspace(360e9, 136e9, n)
    m = 1e24 * np.log(360e9 / p) / np.log(360 / 136)
    return m * (1 - 1e-9), p, t_of_p(p)  # the last node just below the core mass


def _liq(p):
    return 4000.0 + 1e-8 * (p - 136e9)  # 4000 K at 136 GPa, 6240 K at 360 GPa


@pytest.mark.physics_invariant
def test_the_icb_is_where_the_profile_meets_the_liquidus():
    """An isothermal 5000 K core meets the linear liquidus at 236 GPa; the ICB mass is the
    enclosed mass there, and no solid lies above it."""
    m, p, t = _profile(lambda p: 0 * p + 5000.0)
    m_icb, p_icb, solid = icb_from_profile(m, p, t, 1e24, _liq)
    assert p_icb == pytest.approx(236e9, rel=1e-12)
    assert m_icb == pytest.approx(
        1e24 * math.log(360 / 236) / math.log(360 / 136) * (1 - 1e-9), rel=1e-12
    )
    assert solid == 0.0


def test_the_icb_misses_are_the_distances_to_the_root_and_to_the_adiabat_icb():
    """The layer ICB against the liquidus root of the profile and against the ICB that the
    adiabat took; without an adiabat the second distance is 0."""
    from zalmoxis.core_profile import icb_misses

    m, p, t = _profile(lambda p: 0 * p + 5000.0)
    root = icb_from_profile(m, p, t, 1e24, _liq)[0]
    miss = icb_misses(m, p, t, 1e24, _liq, root + 3e20, root - 2e20)
    assert miss == (pytest.approx(3e20, rel=1e-9), pytest.approx(5e20, rel=1e-9))
    assert icb_misses(m, p, t, 1e24, _liq, root + 3e20, None)[1] == 0.0


def test_a_core_above_or_below_the_liquidus_everywhere():
    """A centre at or above the liquidus gives no inner core; a core below it everywhere is
    the inner core, with the pressure of its last core node."""
    m, p, t = _profile(lambda p: 0 * p + 7000.0)
    assert icb_from_profile(m, p, t, 1e24, _liq)[:2] == (0.0, p[0])
    m, p, t = _profile(lambda p: 0 * p + 3000.0)
    assert icb_from_profile(m, p, t, 1e24, _liq) == (1e24, p[-1], 0.0)


def test_solid_above_a_liquid_centre_is_the_mass_of_its_segments():
    """A centre above the liquidus with a cold top: the solid mass above the ICB is the mass of
    the node segments whose 2 ends lie below the liquidus."""
    m, p, t = _profile(lambda p: np.where(p > 200e9, 7000.0, 3000.0))
    m_icb, _, solid = icb_from_profile(m, p, t, 1e24, _liq)
    below = p <= 200e9
    seg = below[:-1] & below[1:]
    assert m_icb == 0.0 and solid == pytest.approx(float(np.sum(np.diff(m)[seg])), rel=1e-15)
    assert 0.0 < solid < 1e24


def test_the_liquidus_rule_is_refused_with_the_newton_outer_solver():
    """Each Newton probe solves one outer pass, so the liquidus rule would have no profile of a
    previous pass; the mass rule needs none."""
    with pytest.raises(ValueError, match="core_liquidus needs outer_solver = 'picard'"):
        validate_config(_params(outer_solver='newton'))
    validate_config(
        _params(outer_solver='newton', core_liquidus='', inner_core_mass_fraction=0.05)
    )


def test_the_liquidus_rule_is_refused_with_the_cmb_anchored_adiabat():
    """The CMB-anchored adiabat carries the surface temperature through the core, so the
    liquidus rule would read it as the core temperature; the mass rule reads no core T."""
    from zalmoxis.config import _check_icb_rule

    cmb = {'temperature_mode': 'adiabatic_from_cmb', 'cmb_temperature': 4000.0}
    for check in (validate_config, _check_icb_rule):  # main calls _check_icb_rule itself
        with pytest.raises(ValueError, match='core_liquidus needs a core temperature'):
            check(_params(**cmb))
        try:
            check(_params(**cmb, core_liquidus='', inner_core_mass_fraction=0.05))
        except ValueError as err:
            pytest.fail(f'the mass rule with the CMB-anchored adiabat is refused: {err}')


@pytest.mark.parametrize(
    ('layers', 'match'),
    [
        ({'core': f'CoreAlloy:S:0.1385+{LIQ}:0.8615', 'inner_core': SOL}, None),
        ({'core': LIQ, 'inner_core': f'CoreAlloy:Si:0.01+{SOL}:0.99'}, None),
        ({'core': LIQ, 'inner_core': LIQ}, 'inner_core'),
        ({'core': SOL, 'inner_core': SOL}, "'core'"),
        ({'core': SOL, 'inner_core': LIQ}, "belongs in 'inner_core'"),
        ({'core': 'PALEOS:iron', 'inner_core': SOL}, "'core'"),
        ({'core': 'PALEOS:iron', 'inner_core': 'PALEOS:iron'}, "'core'"),
        ({'core': 'Analytic:iron', 'inner_core': 'Analytic:iron'}, "'core'"),
        ({'core': LIQ}, "needs an 'inner_core'"),
        ({'core': 'PALEOS:iron', 'mantle_extra': LIQ}, "belongs in 'core'"),
        ({'core': LIQ, 'inner_core': 'PALEOS:iron:0.9+CoreAlloy:S:0.1'}, 'inner_core'),
        ({'core': f'PALEOS:iron:0.5+{LIQ}:0.5'}, "needs an 'inner_core'"),
        ({'core': f'PALEOS:iron:0.5+{LIQ}:0.5', 'inner_core': SOL}, 'base alone'),
        ({'core': f'{LIQ}:0.5+PALEOS:iron:0.5', 'inner_core': SOL}, 'base alone'),
    ],
)
def test_each_side_belongs_to_its_core_layer_in_any_order(layers, match):
    """The liquid side is the iron base of the outer core of a two-layer core and the solid side
    that of its inner core, wherever it stands; the loader check, validate_config and main
    refuse a liquid side without an inner core, a second iron base or the wrong side."""
    from zalmoxis.solver import main

    layers = {'mantle': 'PALEOS:MgSiO3', **layers}
    frac = 0.05 if 'inner_core' in layers else None
    params = _params(layer_eos_config=layers, core_liquidus='', inner_core_mass_fraction=frac)
    checks = [lambda: validate_layer_eos(layers), lambda: validate_config(params)]
    if match is None:
        for check in checks:
            try:
                check()
            except ValueError as err:
                pytest.fail(f'{layers} is refused: {err}')
        return
    for check in [*checks, lambda: main(params, None, None, '')]:
        with pytest.raises(ValueError, match=match):
            check()


def test_the_liquidus_rule_needs_a_core_temperature():
    """With no T-dependent EOS in any layer and no temperature function the liquidus rule has
    no core temperature: _solve refuses it before the first pass."""
    from zalmoxis import solver as zs
    from zalmoxis.config import load_material_dictionaries
    from zalmoxis.mixing import parse_all_layer_mixtures

    layers = {
        'core': 'Analytic:iron',
        'inner_core': 'Analytic:iron',
        'mantle': 'Analytic:MgSiO3',
    }
    params = _params(
        layer_eos_config=layers, temperature_mode='isothermal', surface_temperature=5000.0
    )
    mixes = parse_all_layer_mixtures(layers)
    with pytest.raises(ValueError, match='core_liquidus needs a core temperature'):
        zs._solve(params, load_material_dictionaries(), None, '', layer_mixtures=mixes)


@pytest.mark.parametrize('bad', ['liquidus', 'temperature'])
def test_a_non_finite_liquidus_or_temperature_is_refused(bad):
    """A NaN in the liquidus or the profile would read as below the liquidus everywhere and make
    the whole core the inner core."""
    m, p, t = _profile(lambda p: 0 * p + 7000.0)
    liq = (lambda p: np.nan) if bad == 'liquidus' else _liq
    if bad == 'temperature':
        t[5] = np.nan
    with pytest.raises(ValueError, match='finite'):
        icb_from_profile(m, p, t, 1e24, liq)


def test_the_icb_on_a_profile_that_falls_outward():
    """T and the liquidus linear in ln P (exact under the interpolation of the rule): T from
    5500 K at the centre to 4500 K at the top, the liquidus from 6240 to 4000 K, so they meet
    at P = 136 GPa (360/136)^(500/1240): below the liquidus at the centre, above it at the top,
    no solid above the ICB."""
    k = math.log(360 / 136)
    m, p, t = _profile(lambda p: 4500.0 + 1000.0 / k * np.log(p / 136e9))
    m_icb, p_icb, solid = icb_from_profile(
        m, p, t, 1e24, lambda p: 4000.0 + 2240.0 / k * math.log(p / 136e9)
    )
    assert p_icb == pytest.approx(136e9 * (360 / 136) ** (500 / 1240), rel=1e-12)
    assert m_icb == pytest.approx(1e24 * (1 - 500 / 1240) * (1 - 1e-9), rel=1e-12)
    assert solid == 0.0


def _past_cmb(t_of_p):
    """Nodes from 360 to 130 GPa, mass linear in ln P with the core mass 1e24 kg at 136 GPa:
    the last core node is at 141.5 GPa, the next node past the CMB."""
    p = np.linspace(360e9, 130e9, 41)
    return 1e24 * np.log(360e9 / p) / np.log(360 / 136), p, t_of_p(p)


def test_the_icb_can_lie_between_the_last_core_node_and_the_cmb():
    """An isothermal 4030 K core meets the liquidus at 139 GPa, past the last core node: the
    rule finds the ICB on the segment to the CMB, not a core solid to the CMB."""
    m, p, t = _past_cmb(lambda p: 0 * p + 4030.0)
    m_icb, p_icb, solid = icb_from_profile(m, p, t, 1e24, _liq)
    assert p_icb == pytest.approx(139e9, rel=1e-12) and solid == 0.0
    assert m_icb == pytest.approx(1e24 * math.log(360 / 139) / math.log(360 / 136), rel=1e-12)


def test_a_cold_top_counts_its_segment_to_the_cmb():
    """A core below the liquidus to the CMB is the inner core at the CMB pressure; with a hot
    centre, the cold mass above the ICB reaches the core mass."""
    m, p, t = _past_cmb(lambda p: 0 * p + 3000.0)
    m_icb, p_icb, solid = icb_from_profile(m, p, t, 1e24, _liq)
    assert (m_icb, solid) == (1e24, 0.0) and p_icb == pytest.approx(136e9, rel=1e-12)
    m, p, t = _past_cmb(lambda p: np.where(p > 200e9, 7000.0, 3000.0))
    cold = m[np.argmax(p <= 200e9)]
    assert icb_from_profile(m, p, t, 1e24, _liq)[2] == pytest.approx(1e24 - cold, rel=1e-12)


def _sphere_result(n_nodes, cmb_scale):
    """A uniform 10000 kg/m^3 sphere on n_nodes nodes, its rhs, and a core mass of cmb_scale
    times the mass of node n_nodes // 2."""
    rho, r = 1.0e4, np.linspace(0.0, 4.0e6, n_nodes)
    m = 4 / 3 * np.pi * rho * r**3
    g = np.divide(G * m, r**2, out=np.zeros_like(r), where=r > 0)
    p = 2 / 3 * np.pi * G * rho**2 * (r[-1] ** 2 - r**2)
    res = dict(radii=r, mass_enclosed=m, gravity=g, pressure=p, density=np.full(n_nodes, rho))
    res.update(temperature=np.full(n_nodes, 5000.0), cmb_mass=cmb_scale * m[n_nodes // 2])

    def rhs(r_, y):
        return [4 * np.pi * r_**2 * rho, 4 * np.pi * G * rho - 2 * y[1] / r_, -rho * y[1]]

    return res, rhs


def test_a_node_just_below_the_core_mass_still_gives_a_cmb_node_at_the_core_mass():
    """With a node within 1e-12 below the core mass, the CMB node holds the core mass and the
    radii keep rising."""
    res, rhs = _sphere_result(41, 1.0 / (1.0 - 5e-13))
    mix = {'core': LayerMixture(['PALEOS:iron'], [1.0])}
    core = core_arrays(res, mix, rhs, lambda r, p: 5000.0, 0.0, 0.0)
    assert abs(core['m'][-1] / res['cmb_mass'] - 1) < 1e-12
    assert np.all(np.diff(core['r']) > 0)


def test_core_arrays_without_a_node_past_the_core_mass_are_refused():
    """A profile that ends below the core mass has no CMB node to build."""
    res, rhs = _sphere_result(41, 10.0)
    mix = {'core': LayerMixture(['PALEOS:iron'], [1.0])}
    with pytest.raises(ValueError, match='core mass'):
        core_arrays(res, mix, rhs, lambda r, p: 5000.0, 0.0, 0.0)


def test_the_phase_columns_label_the_inner_core_by_its_side():
    """Shells below the ICB mass take the inner-core EOS: the solid side is solid, the liquid
    side liquid; the shell at the ICB mass is outer core."""
    m = np.linspace(0.0, 10.0, 11)
    _, phases = compute_layer_phase_columns(
        pressure=np.full(11, 2e11),
        temperature=np.full(11, 5000.0),
        mass_enclosed=m,
        cmb_mass=6.0,
        core_mantle_mass=10.0,
        icb_mass=2.5,
        layer_eos_config={'core': LIQ, 'inner_core': SOL, 'mantle': 'Seager2007:MgSiO3'},
    )
    assert phases[:3] == ['solid'] * 3 and phases[3:6] == ['liquid'] * 3


def test_a_profile_without_core_nodes_is_refused():
    """The ICB rule needs core nodes of positive pressure."""
    m, p, t = _profile(lambda p: 0 * p + 5000.0)
    with pytest.raises(ValueError, match='no core node'):
        icb_from_profile(m, 0 * p, t, 1e24, _liq)


def test_core_arrays_refuse_a_core_mass_out_of_reach():
    """An rhs 100 times lighter than the profile cannot reach the core mass within 2
    segments."""
    res, rhs = _sphere_result(41, 1.0)
    mix = {'core': LayerMixture(['PALEOS:iron'], [1.0])}

    def light(r_, y):
        return [d / 100 if k == 0 else d for k, d in enumerate(rhs(r_, y))]

    with pytest.raises(ValueError, match='not reached'):
        core_arrays(res, mix, light, lambda r, p: 5000.0, 0.0, 0.0)


@pytest.mark.parametrize('frac', [1.0, 0.37])
def test_p_icb_is_the_pressure_of_the_core_arrays_at_m_ic(frac):
    """P_icb meets the analytic pressure of the uniform sphere at M_ic to 1e-3, which both
    neighbour nodes miss at 0.37 of the core mass; at the core mass it is the CMB node."""
    res, rhs = _sphere_result(41, 1.0)
    mix = {'core': LayerMixture([LIQ], [1.0]), 'inner_core': LayerMixture([SOL], [1.0])}
    core = core_arrays(res, mix, rhs, lambda r, p: 5000.0, frac * res['cmb_mass'], 0.0)
    rho, big_r = 1.0e4, 4.0e6
    r = (3 * core['M_ic'] / (4 * np.pi * rho)) ** (1 / 3)
    want = 2 / 3 * np.pi * G * rho**2 * (big_r**2 - r**2)
    assert core['P_icb'] == pytest.approx(want, rel=1e-3)
    i = int(np.searchsorted(core['m'], core['M_ic']))
    if frac < 1.0:
        assert all(abs(core['P'][k] / want - 1) > 5e-3 for k in (i - 1, i))
    else:
        assert core['P_icb'] == pytest.approx(core['P'][-1], rel=1e-12)


@pytest.mark.parametrize('frac', [0.01, 0.1, 0.5, 4.0])
def test_r_icb_and_p_icb_of_a_small_inner_core_meet_the_uniform_sphere(frac):
    """R_icb and P_icb at M_ic of frac times the first node mass (4.0: in the 2nd segment) meet
    the uniform sphere to 1e-9: r^3 is linear in m on a segment, P linear in m^(2/3)."""
    res, rhs = _sphere_result(41, 1.0)
    mix = {'core': LayerMixture([LIQ], [1.0]), 'inner_core': LayerMixture([SOL], [1.0])}
    m_ic = frac * res['mass_enclosed'][1]
    core = core_arrays(res, mix, rhs, lambda r, p: 5000.0, m_ic, 0.0)
    rho, big_r = 1.0e4, 4.0e6
    r = (3 * m_ic / (4 * np.pi * rho)) ** (1 / 3)
    assert core['R_icb'] == pytest.approx(r, rel=1e-9)
    assert core['P_icb'] == pytest.approx(
        2 / 3 * np.pi * G * rho**2 * (big_r**2 - r**2), rel=1e-9
    )


def test_a_given_icb_mass_reports_its_pressure_and_the_solid_above_it():
    """With a given ICB mass the rule reports it, the pressure there (to node resolution of the
    exponential P(m) of _profile) and the mass of the cold segments above it."""
    m, p, t = _profile(lambda p: np.where(p > 200e9, 7000.0, 3000.0))
    m_icb = 0.4e24
    got = icb_from_profile(m, p, t, 1e24, _liq, m_icb)
    p_true = 360e9 * (136 / 360) ** (m_icb / 1e24)
    below = p <= 200e9
    seg = below[:-1] & below[1:] & (m[:-1] >= m_icb)
    assert got[0] == m_icb and got[1] == pytest.approx(p_true, rel=1e-3)
    assert got[2] == pytest.approx(float(np.sum(np.diff(m)[seg])), rel=1e-15) and got[2] > 0


@pytest.mark.parametrize(
    ('value', 'ok'),
    [
        (np.float32(0.05), True),
        (np.int64(0), True),
        (np.float64(1.0), True),
        (np.bool_(True), False),
        (np.float32('nan'), False),
        (np.float32(1.5), False),
        (np.float32(-0.01), False),
    ],
)
def test_the_icb_rule_takes_numpy_numbers_in_0_1(value, ok):
    """inner_core_mass_fraction may be a NumPy real in [0, 1]; a NumPy bool, NaN or a value
    outside the range is refused."""
    params = _params(core_liquidus='', inner_core_mass_fraction=value)
    if ok:
        validate_config(params)
    else:
        with pytest.raises(ValueError, match='inner_core_mass_fraction'):
            validate_config(params)


def test_the_icb_rule_needs_no_layer_config():
    """A config without layers has no inner core and no rule to check."""
    from zalmoxis.config import _check_icb_rule

    assert _check_icb_rule({'layer_eos_config': None}) is None


def test_main_returns_an_unconverged_icb_after_a_failed_retry(monkeypatch, caplog):
    """When the ICB stays off the liquidus root in the first solve and in the retry, main
    neither takes the first result back by its retry skip nor hides the second: it returns the
    retry result, not converged, and says so."""
    from zalmoxis import solver as zs
    from zalmoxis.mixing import parse_all_layer_mixtures

    calls = []

    def fake_solve(params, *args, **kwargs):
        calls.append(params)
        return {
            'converged': False,
            'icb_converged': False,
            'converged_density': True,
            'converged_pressure': True,
            'converged_mass': False,
            'mass_error_at_accept': 0.01,
            'radii': np.linspace(0.0, 6.4e6, 5),
            'density': np.full(5, 5e3),
            'call': len(calls),
        }

    monkeypatch.setattr(zs, '_solve', fake_solve)
    monkeypatch.setattr(zs, '_report_table_bands', lambda r, *a: r)
    monkeypatch.setattr(zs, '_warn_adiabat', lambda *a: None)
    monkeypatch.setattr(zs, '_check_accepted', lambda r, *a: r)
    cfg = _params()
    caplog.set_level('WARNING', logger='fwl.zalmoxis.solver')
    res = zs.main(
        cfg, None, None, '', layer_mixtures=parse_all_layer_mixtures(cfg['layer_eos_config'])
    )
    assert len(calls) == 2 and res['call'] == 2 and res['converged'] is False
    assert 'did not converge after retry' in caplog.text


def test_the_jax_path_refuses_an_inner_core():
    """The JAX structure kernel holds one core table, so a two-layer core is refused before any
    table is read."""
    pytest.importorskip('jax')
    from zalmoxis.jax_eos.wrapper import solve_structure_via_jax

    mixes = {'core': LayerMixture([LIQ], [1.0]), 'inner_core': LayerMixture([SOL], [1.0])}
    with pytest.raises(ValueError, match='one-layer core'):
        solve_structure_via_jax(mixes, *([None] * 20))


def test_outside_side_tables_flags_nodes_off_the_table(tmp_path):
    """Nodes beyond the pressure box, at a non-positive temperature or clamped in their row are
    flagged; a node inside the table and a mixture without a side table are not."""
    _paleos_mock.skip_unless_real()
    from zalmoxis.core_profile import outside_side_tables
    from zalmoxis.eos import paleos_api as pa

    grid = pa.GridSpec(p_lo=5e10, p_hi=2e11, n_p=4, t_lo=2000.0, t_hi=6000.0, n_t=5)
    out = pa.generate_paleos_api_2phase_iron_tables(
        tmp_path / 's.dat', tmp_path / 'l.dat', grid
    )
    md = {LIQ: {'eos_file': out['liquid']['path']}}
    p, t = np.array([1e11, 1e12, 1e11, 1e11]), np.array([4000.0, 4000.0, 0.0, 9000.0])
    got = outside_side_tables(p, t, LayerMixture([LIQ], [1.0]), md, {})
    assert got.tolist() == [False, True, True, True]
    none = outside_side_tables(p, t, LayerMixture(['PALEOS:iron'], [1.0]), md, {})
    assert not none.any()


def test_a_test_of_the_real_paleos_skips_on_the_mock(monkeypatch):
    """With the PALEOS mock of a runner without the package in place, a test that needs the
    real package skips instead of failing in the import machinery."""
    mock = types.ModuleType('paleos')
    mock.__version__ = '0.0.0-mock'
    monkeypatch.setitem(sys.modules, 'paleos', mock)
    with pytest.raises(pytest.skip.Exception):
        test_the_solid_picker_keeps_the_delta_boundary_of_paleos()


def test_solid_above_counts_only_segments_cold_at_both_ends():
    """A hot centre, a cold band, a hot band and a cold top: only the segments with both ends
    below the liquidus count, not those that end in the hot band."""

    def bands(p):
        return np.where((p > 300e9) | ((p > 200e9) & (p <= 250e9)), 7000.0, 3000.0)

    m, p, t = _profile(bands)
    m_icb, _, solid = icb_from_profile(m, p, t, 1e24, _liq)
    cold = t < _liq(p)
    seg = cold[:-1] & cold[1:]
    assert m_icb == 0.0 and solid == pytest.approx(float(np.sum(np.diff(m)[seg])), rel=1e-15)
    assert (cold[:-1] & ~cold[1:]).any()  # the profile has cold-to-hot segments to leave out


def test_a_given_icb_mass_counts_no_solid_below_it():
    """On a core below the liquidus everywhere, the solid above a given ICB mass is the mass of
    the segments above it, not of the whole core."""
    m, p, t = _profile(lambda p: 0 * p + 3000.0)
    m_icb = 0.4e24
    solid = icb_from_profile(m, p, t, 1e24, _liq, m_icb)[2]
    assert solid == pytest.approx(float(np.sum(np.diff(m)[m[:-1] >= m_icb])), rel=1e-15)
    assert solid < m[-1] - m_icb + 0.1e24


def test_the_solid_picker_keeps_the_delta_boundary_of_paleos():
    """PALEOS takes bcc delta iron up to and including 5.2 GPa above T_delta_gamma (1998 K
    there): the picker gives delta at exactly 5.2 GPa and gamma just above it."""
    _paleos_mock.skip_unless_real()
    from zalmoxis.eos.paleos_api import _get_iron_solid_phase

    assert _get_iron_solid_phase(5.2e9, 2000.0) == 'solid-delta'
    assert _get_iron_solid_phase(5.21e9, 2000.0) == 'solid-gamma'


def test_the_start_dependence_scan_of_the_liquidus_rule(validation_plot):
    """The scan of tools/core_icb_scan.py stored with its commit (the Fe-S core of the ICB
    validation case; the test runs no solve): every converged step meets both ICB tests, the
    walks agree outside 2500 to 2525 K and inside it give the table of the case."""
    import csv
    from pathlib import Path

    path = Path(__file__).parent / 'validation' / 'data' / 'core_icb_scan.csv'
    with open(path) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith('#')))
    walk = {
        d: {float(r['surface_temperature_K']): r for r in rows if r['direction'] == d}
        for d in ('down', 'up')
    }
    temps = np.arange(2300.0, 2701.0, 25.0)
    order = [float(r['surface_temperature_K']) for r in rows]
    assert order == list(temps[::-1]) + list(temps)  # walk down, then walk up
    frac = {
        d: {t: float(r['inner_core_mass_fraction']) for t, r in w.items()}
        for d, w in walk.items()
    }
    ok = {d: {t: r['converged'] == '1' for t, r in w.items()} for d, w in walk.items()}
    assert all(r['icb_converged'] == '1' for r in rows if r['converged'] == '1')
    for t in temps[(temps < 2500.0) | (temps > 2525.0)]:
        assert (
            ok['down'][t]
            and ok['up'][t]
            and frac['down'][t] == pytest.approx(frac['up'][t], abs=1e-3)
        )
    table = {2475.0: (1.0, 1.0), 2500.0: (0.882, 1.0), 2525.0: (0.0, None), 2550.0: (0.0, 0.0)}
    for t, (down, up) in table.items():  # the table of the case; None: not converged
        assert ok['down'][t] and frac['down'][t] == pytest.approx(down, abs=5e-4)
        assert ok['up'][t] == (up is not None)
        assert up is None or frac['up'][t] == pytest.approx(up, abs=5e-4)

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        ax.axvspan(2487.5, 2537.5, color=c[2], alpha=0.15, lw=0, label='start-dependent')
        for d, colour, style in (('down', c[0], 'o-'), ('up', c[1], 's--')):
            y = [frac[d][t] if ok[d][t] else np.nan for t in temps]
            ax.plot(temps, y, style, color=colour, ms=4, mfc='none', label=f'walk {d}')
        bad = [t for t in temps if not ok['up'][t]]
        y_bad = [frac['up'][t] for t in bad]
        ax.plot(bad, y_bad, 'x', color=c[1], ms=8, mew=1.5, label='up, not converged')
        ax.set(xlabel='Surface temperature [K]', ylabel=r'$M_\mathrm{ic} / M_\mathrm{core}$')
        ax.set(ylim=(-0.05, 1.05), xticks=temps[::8], yticks=(0.0, 0.5, 1.0))
        ax.grid(False)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
        fig.tight_layout()
        fig.canvas.draw()
        assert ax.get_legend().get_window_extent().x0 >= ax.get_window_extent().x1
        return fig

    validation_plot(figure, 'core_two_phase_icb_start_dependence')
