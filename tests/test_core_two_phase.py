"""Two-layer core: the iron side tables in the registry and the configuration of an inner
core."""

from __future__ import annotations

import math
import numbers
import re
import sys
import types

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from tests import _paleos_mock
from zalmoxis.config import parse_eos_config, validate_config, validate_layer_eos
from zalmoxis.constants import G
from zalmoxis.core_profile import core_arrays, icb_from_profile, icb_superheat
from zalmoxis.eos import core_alloy as ca
from zalmoxis.eos.paleos_api import make_default_grid_iron_2phase
from zalmoxis.eos_properties import EOS_REGISTRY
from zalmoxis.mixing import LayerMixture
from zalmoxis.phase_columns import compute_layer_phase_columns

pytestmark = pytest.mark.unit

LIQ, SOL = 'PALEOS-API-2phase:iron-liquid', 'PALEOS-API-2phase:iron-solid'


def _refused(call, match):
    """``call`` raises a ValueError that matches ``match``; any other outcome fails by
    assertion, also another exception."""
    with pytest.raises(Exception) as err:
        call()
    assert isinstance(err.value, ValueError) and re.search(match, str(err.value)), err.value


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


def test_the_solid_side_refuses_the_luo_carrier_and_warns_once_on_the_other_laws(
    monkeypatch, caplog
):
    """The light-element laws are calibrated on liquid iron: the Luo et al. (2024) carrier, a
    liquid of iron and water, is refused on the solid side; the others warn once per process."""
    from zalmoxis.eos import core_alloy as ca

    monkeypatch.setattr(ca, '_SOLID_WARNED', [False], raising=False)
    wet = {
        'core': LIQ,
        'inner_core': f'{SOL}:0.99+CoreAlloy:H-Luo:0.01',
        'mantle': 'PALEOS:MgSiO3',
    }
    from zalmoxis.solver import main

    params = _params(layer_eos_config=wet, core_liquidus='', inner_core_mass_fraction=0.05)
    for check in (lambda: validate_layer_eos(wet), lambda: validate_config(params)):
        with pytest.raises(ValueError, match='H-Luo'):
            check()
    with pytest.raises(ValueError, match='H-Luo'):  # before the first solve, with no tables
        main(params, None, None, '')
    caplog.set_level('WARNING', logger='fwl.zalmoxis.eos.core_alloy')
    for _ in range(2):
        ca.split_alloy(LayerMixture([SOL, 'CoreAlloy:S'], [0.95, 0.05]))
    warned = [r for r in caplog.records if 'calibrated on liquid iron' in r.getMessage()]
    assert len(warned) == 1
    assert ca.split_alloy(LayerMixture([LIQ, 'CoreAlloy:H-Luo'], [0.99, 0.01]))[0] == LIQ


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
        ({'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3:0.9+CoreAlloy:S:0.1'}, 'in the core'),
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
        _refused(check, match)


@settings(max_examples=300, deadline=None)
@given(
    place=st.tuples(*[st.sampled_from(('core', 'inner_core', 'mantle'))] * 3),
    element=st.sampled_from(('S', 'Si', 'O', 'H-Luo')),
    flip=st.booleans(),
)
@example(place=('core', 'inner_core', 'mantle'), element='S', flip=False)
@example(place=('core', 'inner_core', 'inner_core'), element='H-Luo', flip=True)
@example(place=('inner_core', 'core', 'core'), element='Si', flip=False)
def test_the_layer_check_follows_the_place_of_each_component(place, element, flip):
    """A dict with the liquid side, the solid side and one light element, each in any layer and
    in either component order: the loader check and validate_config accept it exactly when the
    liquid side is in the core, the solid side in the inner core and the light element in a
    core layer, with H-Luo not on the solid side; every other dict is refused by a ValueError."""
    comps = {'mantle': ['PALEOS:MgSiO3']}
    for name, layer in zip((LIQ, SOL, f'CoreAlloy:{element}'), place):
        comps.setdefault(layer, []).append(name)
    layers = {}
    for layer, names in comps.items():
        bases = [n for n in names if not n.startswith('CoreAlloy')]
        w = {n: 0.005 if n.startswith('CoreAlloy') else 0.0 for n in names}
        for n in bases:
            w[n] = (1.0 - 0.005 * (len(names) - len(bases))) / len(bases)
        order = names[::-1] if flip else names
        layers[layer] = order[0] if len(names) == 1 else '+'.join(f'{n}:{w[n]}' for n in order)
    sides_ok = place[:2] == ('core', 'inner_core')
    match = '.' if not sides_ok else {'mantle': 'belong in the core'}.get(place[2])
    if sides_ok and element == 'H-Luo' and place[2] == 'inner_core':
        match = 'H-Luo is refused on'
    params = _params(layer_eos_config=layers, core_liquidus='Anzellini13-iron')
    for check in (lambda: validate_layer_eos(layers), lambda: validate_config(params)):
        if match is None:
            check()
        else:
            _refused(check, match)


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
    md = load_material_dictionaries()
    _refused(
        lambda: zs._solve(params, md, None, '', layer_mixtures=mixes),
        'needs a core temperature',
    )


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
    """A core at 4030 K at 139 GPa, isothermal or falling outward in ln P, meets the liquidus
    there, past the last core node: the rule finds the ICB on the segment to the CMB, not a
    core solid to the CMB."""
    for t_of_p in (lambda p: 0 * p + 4030.0, lambda p: 4030.0 + 500.0 * np.log(p / 139e9)):
        m, p, t = _past_cmb(t_of_p)  # the 2nd T is linear in ln P, as the rule interpolates
        m_icb, p_icb, solid = icb_from_profile(m, p, t, 1e24, _liq)
        assert p_icb == pytest.approx(139e9, rel=1e-12) and solid == 0.0
        want = 1e24 * math.log(360 / 139) / math.log(360 / 136)
        assert m_icb == pytest.approx(want, rel=1e-12)


def test_a_failed_node_past_the_core_mass_adds_no_cmb_point():
    """Nodes past the core mass filled with NaN by a failed stop give no CMB point: the rule
    reads the core nodes alone, as for a profile that ends at the last core node."""
    m, p, t = _past_cmb(lambda p: 0 * p + 3000.0)
    m[39:], p[39:] = np.nan, np.nan
    try:
        got = icb_from_profile(m, p, t, 1e24, _liq)
    except ValueError as err:
        pytest.fail(f'the NaN nodes past the core mass are read: {err}')
    assert got == (1e24, p[38], 0.0)


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


@pytest.mark.parametrize('k', [-5.0, -0.5, 0.0, 0.5])
def test_the_phase_column_at_the_icb_follows_the_layer_rule_of_the_eos(k):
    """A shell within 1e-12 (relative) below the ICB mass takes the outer core, as in
    get_layer_mixture, which picks its EOS; 5e-12 below it the inner core."""
    from zalmoxis.structure_model import get_layer_mixture

    icb = 2.5e24
    m = np.array([icb * (1.0 + k * 1e-12)])
    layers = {'core': LIQ, 'inner_core': SOL, 'mantle': 'Seager2007:MgSiO3'}
    _, phases = compute_layer_phase_columns(
        pressure=np.full(1, 2e11),
        temperature=np.full(1, 5000.0),
        mass_enclosed=m,
        cmb_mass=6e24,
        core_mantle_mass=10e24,
        icb_mass=icb,
        layer_eos_config=layers,
    )
    side = get_layer_mixture(m[0], 6e24, 10e24, layers, icb)
    assert phases == ['solid' if side == SOL else 'liquid'] and (side == SOL) is (k < -1.0)


@pytest.mark.parametrize(
    ('edge', 'k', 'icb'),
    [('cmb', k, icb) for k in (-5.0, -0.5, 0.5) for icb in (2.5e24, 6e24)]
    + [('ice', k, 2.5e24) for k in (-5.0, -0.5, 0.5)],
)
def test_the_phase_column_at_the_cmb_and_the_ice_boundary_follows_the_layer_rule(edge, k, icb):
    """A shell within 1e-12 (relative) below the core mass or the core+mantle mass takes the
    layer above, as in get_layer_mixture, also with a full inner core (icb = core mass); an
    iron shell has the phase of its side."""
    from zalmoxis.structure_model import get_layer_mixture

    cmb, cm = 6e24, 10e24
    m = np.array([(cmb if edge == 'cmb' else cm) * (1.0 + k * 1e-12)])
    layers = {
        'core': LIQ,
        'inner_core': SOL,
        'mantle': 'Seager2007:MgSiO3',
        'ice_layer': 'Seager2007:H2O',
    }
    comp, phases = compute_layer_phase_columns(
        pressure=np.full(1, 1e11),
        temperature=np.full(1, 3000.0),
        mass_enclosed=m,
        cmb_mass=cmb,
        core_mantle_mass=cm,
        icb_mass=icb,
        layer_eos_config=layers,
    )
    side = get_layer_mixture(m[0], cmb, cm, layers, icb)
    want = {LIQ: ('Fe', 'liquid'), SOL: ('Fe', 'solid'), layers['mantle']: ('MgSiO3', 'solid')}
    want[layers['ice_layer']] = ('H2O', 'solid')  # Seager2007 tables at 300 K
    assert (comp, phases) == ([want[side][0]], [want[side][1]]), side


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


class _Layer:
    """A T-dependent layer mixture with a constant adiabatic gradient ``n``."""

    def __init__(self, n):
        self.n = n

    def has_tdep(self):
        return True


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize(
    'where', [None, 0.3, 0.9999], ids=['between_core_nodes', 'cmb_segment', 'next_to_cmb']
)
def test_the_adiabat_cuts_its_segments_at_the_pressures_of_the_icb_readers(monkeypatch, where):
    """On the uniform sphere with a constant nabla per layer (0.1 inner core, 0.4 outer core,
    0.2 mantle), every node meets the piecewise power law T = T_a (P / P_a)^n to round-off
    (analytical law). The law breaks at the CMB with ln P linear in the enclosed mass, and at
    the ICB at the one pressure that core_arrays, icb_superheat and icb_from_profile give for
    the same mass (at this CMB position the CMB point of the readers lies above the CMB node,
    so core_arrays does not move P_icb). between_core_nodes: the ICB lies between nodes 9 and
    10, at the closed-form pressure of the sphere. cmb_segment, next_to_cmb: the ICB lies at
    0.3 and 0.9999 of the way from the last core node to the CMB, between the CMB and that
    node in pressure."""
    import zalmoxis.mixing as _mix
    from zalmoxis.eos.temperature import compute_adiabatic_temperature

    res, rhs = _sphere_result(41, 1.1)  # CMB at 0.65 of the segment above node 20
    m, p, cmb = res['mass_enclosed'], res['pressure'], res['cmb_mass']
    k = int(np.searchsorted(m, cmb))
    p_cmb = math.exp(np.interp(cmb, m[k - 1 : k + 1], np.log(p[k - 1 : k + 1])))
    icb = 0.4 * m[9] + 0.6 * m[10] if where is None else m[k - 1] + where * (cmb - m[k - 1])
    layers = {'inner_core': _Layer(0.1), 'core': _Layer(0.4), 'mantle': _Layer(0.2)}
    monkeypatch.setattr(_mix, 'get_mixed_nabla_ad', lambda P, T, mix, *a, **k: mix.n)
    monkeypatch.setattr(_mix, 'any_component_is_tdep', lambda *a, **k: True)
    t = compute_adiabatic_temperature(
        res['radii'], p, m, 3000.0, cmb, m[-1], layers, {}, icb_mass=icb
    )
    mix = {'core': LayerMixture([LIQ], [1.0]), 'inner_core': LayerMixture([SOL], [1.0])}
    p_icb = core_arrays(res, mix, rhs, lambda r, p_: 5000.0, icb, 0.0)['P_icb']
    liq = lambda p_: p_ / 1e8  # noqa: E731
    s = icb_superheat(m, p, np.full(len(p), 5000.0), cmb, liq, icb)[0]
    assert (5000.0 - s) * 1e8 == pytest.approx(p_icb, rel=1e-12)
    assert icb_from_profile(m, p, t, cmb, liq, icb_mass=icb)[1] == p_icb  # one call path
    if where is None:  # between 2 core nodes of the uniform sphere the pressure is exact
        assert p[10] < p_icb < p[9]
        r_icb = (3 * icb / (4 * np.pi * 1e4)) ** (1 / 3)
        assert p_icb == pytest.approx(2 / 3 * np.pi * G * 1e8 * (4e6**2 - r_icb**2), rel=1e-12)
    else:
        assert p_cmb < p_icb < p[k - 1]
    t_cmb = 3000.0 * (p_cmb / 1e5) ** 0.2  # the top node (P = 0) counts as 1e5 Pa
    t_icb = t_cmb * (p_icb / p_cmb) ** 0.4
    law = np.where(
        m >= cmb,
        3000.0 * (np.maximum(p, 1e5) / 1e5) ** 0.2,
        np.where(m > icb, t_cmb * (p / p_cmb) ** 0.4, t_icb * (p / p_icb) ** 0.1),
    )
    np.testing.assert_allclose(t, law, rtol=1e-12, atol=0)


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('n', 'scale', 'frac'),
    [(41, 1.1, 1.0), (41, 1.1, 0.999), (41, 1.9, 1.0), (11, 1.1, 1.0), (11, 1.9, 0.999)],
)
def test_p_icb_in_the_cmb_segment_lies_at_or_above_the_cmb_node(n, scale, frac):
    """On the uniform sphere with the ICB in the node segment of the CMB, P_icb of the core
    arrays meets the closed form P = 2/3 pi G rho^2 (R^2 - r^2) within the accuracy of the
    CMB point with ln P linear in m at these CMB positions (r/R 0.52 to 0.62: 5.5e-5 at 41
    nodes, 1.0e-3 at 11 nodes; the bounds 1.1e-4 and 2e-3), never lies below the pressure of
    the CMB node (scale 1.9: the reader value is 3.9e-5 and 5.9e-4 below it), and takes that
    pressure for a full inner core."""
    res, rhs = _sphere_result(n, scale)
    mix = {'core': LayerMixture([LIQ], [1.0]), 'inner_core': LayerMixture([SOL], [1.0])}
    icb = frac * res['cmb_mass']
    core = core_arrays(res, mix, rhs, lambda r, p_: 5000.0, icb, 0.0)
    r_icb = (3 * icb / (4 * np.pi * 1e4)) ** (1 / 3)
    exact = 2 / 3 * np.pi * G * 1e8 * (4e6**2 - r_icb**2)
    assert core['P_icb'] >= core['P'][-1]
    assert core['P_icb'] == pytest.approx(exact, rel=1.1e-4 if n == 41 else 2e-3)
    if frac == 1.0:  # an ICB at the CMB has the pressure of the CMB node
        assert core['P_icb'] == core['P'][-1]


@pytest.mark.parametrize(
    ('n', 'scale', 'frac', 'warns'),
    [(11, 3.0, 0.999, True), (41, 1.9, 1.0, False), (11, 1.1, 1.0, False)],
)
def test_a_p_icb_moved_by_more_than_1e_3_logs_a_warning(caplog, n, scale, frac, warns):
    """core_arrays logs 1 warning when it moves P_icb off the reader value by more than a
    relative 1e-3, as on 11 nodes with the CMB at r/R 0.72 (the reader CMB point lies 5.3e-3
    below the CMB node); a smaller move, or none, logs nothing."""
    import logging

    res, rhs = _sphere_result(n, scale)
    mix = {'core': LayerMixture([LIQ], [1.0]), 'inner_core': LayerMixture([SOL], [1.0])}
    with caplog.at_level(logging.WARNING, logger='fwl.zalmoxis.core_profile'):
        core_arrays(res, mix, rhs, lambda r, p_: 5000.0, frac * res['cmb_mass'], 0.0)
    hits = [r for r in caplog.records if 'P_icb' in r.getMessage()]
    assert len(hits) == (1 if warns else 0)
    assert all(r.name == 'fwl.zalmoxis.core_profile' for r in hits)  # reaches the PROTEUS log


_P_TP, _T_TP, _Q = 98.5e9, 3712.0, 1 / 1.72  # triple point and 1/c of Anzellini et al. (2013)
_RHO, _R = 1.0e4, 4.0e6  # the uniform sphere of _sphere_result


def _simon(p):
    """Eq. 3 of Anzellini et al. (2013), (P - P_TP)/a = (T/T_TP)^c - 1, with a = P_TP: the
    power law T_TP (P/P_TP)^(1/c)."""
    return _T_TP * (np.asarray(p) / _P_TP) ** _Q


def _liquid(t0, n, p):
    """Liquid branch at constant nabla_ad = dln T/dln P = n through T0 at 200 GPa."""
    return t0 * (np.asarray(p) / 200e9) ** n


def _sphere_nodes(extra_r=()):
    """Mass and pressure of the uniform sphere on 41 radii and the radii extra_r; core mass the
    mass at 2000 km (node 20). P is linear in m^(2/3), so P at a given mass is exact."""
    r = np.unique(np.r_[np.linspace(0.0, _R, 41), extra_r])
    m = 4 / 3 * np.pi * _RHO * r**3
    p = 2 / 3 * np.pi * G * _RHO**2 * (_R**2 - r**2)
    return m, p, 4 / 3 * np.pi * _RHO * 2.0e6**3


def _r_at(p):
    return math.sqrt(_R**2 - p / (2 / 3 * np.pi * G * _RHO**2))


def _root(t0, n):
    """P_B of T0 (P/P0)^n = T_TP (P/P_TP)^q in closed form."""
    return (_T_TP * 200e9**n / (t0 * _P_TP**_Q)) ** (1 / (n - _Q))


@pytest.mark.physics_invariant
def test_the_icb_superheat_is_0_at_the_closed_form_root():
    """Case 2 of the ICB Validation. A liquid branch with constant nabla_ad = n meets the power
    law liquidus at P_B = (T_TP P0^n / (T0 P_TP^q))^(1/(n - q)), q = 1/1.72. With a node at P_B
    and the layer ICB at its mass, S and the miss are 0 to 1e-8 of T_liq(P_B)."""
    p_b = _root(5500.0, 0.4)
    m, p, cmb = _sphere_nodes([_r_at(p_b)])
    m_b = float(m[np.argmin(abs(p - p_b))])
    assert 0.0 < m_b < cmb
    assert _liquid(5500.0, 0.4, p_b) == pytest.approx(_simon(p_b), rel=1e-12)
    s, miss = icb_superheat(m, p, _liquid(5500.0, 0.4, p), cmb, _simon, m_b)
    assert miss == abs(s) <= 1e-8 * float(_simon(p_b))


@pytest.mark.physics_invariant
@settings(max_examples=60, deadline=None)
@given(n=st.floats(0.1, 0.5), frac=st.floats(0.02, 0.98))
@example(n=0.4, frac=0.678)
def test_the_icb_superheat_is_0_at_the_rules_own_root(n, frac):
    """The ICB update of an adiabat takes the first liquidus root of the liquid branch and its
    mass with m^(2/3) linear in P (``m23``), the inverse of the pressure that the superheat test
    and the core arrays take at a mass; at the rule's own root on the same nodes the miss is 0
    to 1e-9 of T. T0 puts the root of the power law at frac of the core mass."""
    m, p, cmb = _sphere_nodes()
    r_b = (3 * frac * cmb / (4 * np.pi * _RHO)) ** (1 / 3)
    p_b = 2 / 3 * np.pi * G * _RHO**2 * (_R**2 - r_b**2)
    t0 = float(_simon(p_b)) * (200e9 / p_b) ** n
    t = _liquid(t0, n, p)
    icb = icb_from_profile(m, p, t, cmb, _simon, m23=True)[0]
    assert 0.0 < icb < cmb
    assert icb_superheat(m, p, t, cmb, _simon, icb)[1] <= 1e-9 * t0


@pytest.mark.physics_invariant
@settings(max_examples=60, deadline=None)
@given(n=st.floats(0.1, 0.5), frac=st.floats(-0.5, 1.5), t0=st.floats(4500.0, 6500.0))
@example(n=0.4, frac=0.97, t0=5500.0)  # the layer ICB in the top core segment
def test_the_icb_superheat_off_the_root_meets_the_closed_form(n, frac, t0):
    """At a layer ICB inside the core, S is T - T_liq of the liquid branch at the pressure of
    the ICB mass (P linear in m^(2/3), as the core arrays); T is linear in ln P, so S may differ
    from the power law by at most (dln P)^2 / 8 n^2 T on the node segment of the ICB."""
    m, p, cmb = _sphere_nodes()
    s, miss = icb_superheat(m, p, _liquid(t0, n, p), cmb, _simon, frac * cmb)
    assert miss >= 0.0 and math.isfinite(miss)
    if 0.0 < frac < 1.0:
        p_icb = icb_from_profile(m, p, _liquid(t0, n, p), cmb, _simon, frac * cmb)[1]
        want = float(_liquid(t0, n, p_icb) - _simon(p_icb))
        k = min(max(int(np.searchsorted(-p, -p_icb)), 1), 20)
        bound = math.log(p[k - 1] / p[k]) ** 2 / 8 * n**2 * t0 * (p[0] / 200e9) ** n
        assert abs(s - want) <= bound + 1e-9 * t0 and miss == abs(s), (s, want, bound)


@pytest.mark.parametrize('bad', [np.nan, np.inf])
def test_a_non_finite_liquid_branch_at_the_icb_misses_by_inf(bad):
    """A liquid branch that is not finite at the ICB fails the test: the miss is inf, not a NaN
    that every bound comparison would pass."""
    m, p, cmb = _sphere_nodes()
    t = _liquid(5500.0, 0.4, p)
    t[5:7] = bad
    assert icb_superheat(m, p, t, cmb, _simon, float(m[6]))[1] == math.inf


@pytest.mark.parametrize(('t0', 'hot'), [(4500.0, False), (9000.0, True)])
def test_the_icb_superheat_of_no_or_a_full_inner_core_counts_the_wrong_side_only(t0, hot):
    """No inner core: S at the centre, and the miss is the undercooling there, 0 for a hot
    core. A full inner core: S at the core mass, and the miss is the superheat there, 0 for a
    cold core."""
    m, p, cmb = _sphere_nodes()
    t = _liquid(t0, 0.4, p)
    s0, none = icb_superheat(m, p, t, cmb, _simon, 0.0)
    s1, full = icb_superheat(m, p, t, cmb, _simon, cmb)
    k = int(np.argmin(abs(m - cmb)))  # node 20 holds the core mass
    assert s0 == pytest.approx(t[0] - float(_simon(p[0])), rel=1e-12)
    assert s1 == pytest.approx(t[k] - float(_simon(p[k])), rel=1e-12)
    assert (none, full) == ((0.0, s1) if hot else (-s0, 0.0)) and s0 * s1 > 0.0


def test_a_given_icb_mass_in_the_first_segment_has_the_pressure_of_the_uniform_sphere():
    """With a given ICB mass the rule takes P linear in m^(2/3), as core_arrays does: exact on
    the uniform sphere, also inside the first node segment."""
    res, _ = _sphere_result(41, 1.0)
    m, p = res['mass_enclosed'], res['pressure']
    m_ic = 0.01 * m[1]
    r = (3 * m_ic / (4 * np.pi * 1.0e4)) ** (1 / 3)
    want = 2 / 3 * np.pi * G * 1.0e8 * (4.0e6**2 - r**2)
    got = icb_from_profile(m, p, res['temperature'], res['cmb_mass'], _liq, m_ic)[1]
    assert got == pytest.approx(want, rel=1e-9)


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


@pytest.mark.parametrize('node', [0, 10])
def test_the_solid_above_an_icb_at_a_node_counts_the_segment_from_that_node(node):
    """On a core below the liquidus everywhere, an ICB mass at the centre (0) or at a node
    counts every segment from that node up: the solid is the mass from it to the top node."""
    m, p, t = _profile(lambda p: 0 * p + 3000.0)
    solid = icb_from_profile(m, p, t, 1e24, _liq, float(m[node]))[2]
    assert solid == pytest.approx(m[-1] - m[node], rel=1e-14)


def test_the_solid_picker_keeps_the_delta_boundary_of_paleos():
    """PALEOS takes bcc delta iron up to and including 5.2 GPa above T_delta_gamma (1998 K
    there): the picker gives delta at exactly 5.2 GPa and gamma just above it."""
    _paleos_mock.skip_unless_real()
    from zalmoxis.eos.paleos_api import _get_iron_solid_phase

    assert _get_iron_solid_phase(5.2e9, 2000.0) == 'solid-delta'
    assert _get_iron_solid_phase(5.21e9, 2000.0) == 'solid-gamma'


_SCAN_START_LIMIT = 0.0164  # 3 x the largest down-up difference of 5 round-off scans at 2475 K


def test_the_start_dependence_scan_of_the_liquidus_rule(validation_plot):
    """The scan of tools/core_icb_scan.py stored with its commit (the Fe-S core of the ICB
    validation case; the test runs no solve): every step converges and meets both ICB tests,
    and the walk down from a liquid core and the walk up from a solid core give the same
    inner-core mass fraction to 1e-3 at every step but 2475 K, the documented known limit:
    there the scan scatters with round-off (5 pairs of walks, down and up up to 0.0055 apart),
    and the bound is 3 times that difference."""
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
    assert all(r['converged'] == '1' and r['icb_converged'] == '1' for r in rows)
    for t in temps:
        bound = _SCAN_START_LIMIT if t == 2475.0 else 1e-3
        assert frac['down'][t] == pytest.approx(frac['up'][t], abs=bound), t
    assert frac['down'][temps[0]] == 1.0 and frac['down'][temps[-1]] == 0.0

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6.8, 3.2))
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        for d, colour, style in (('down', c[0], 'o-'), ('up', c[1], 's--')):
            y = [frac[d][t] for t in temps]
            ax.plot(temps, y, style, color=colour, ms=4, mfc='none', label=f'walk {d}')
        ax.set(xlabel='Surface temperature [K]', ylabel=r'$M_\mathrm{ic} / M_\mathrm{core}$')
        ax.set(ylim=(-0.05, 1.05), xticks=temps[::8], yticks=(0.0, 0.5, 1.0))
        ax.grid(False)
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
        fig.tight_layout()
        fig.canvas.draw()
        assert ax.get_legend().get_window_extent().x0 >= ax.get_window_extent().x1
        return fig

    validation_plot(figure, 'core_two_phase_icb_start_dependence')


def test_a_one_layer_adiabat_reports_no_icb_keys():
    """A one-layer core on an adiabat (the analytic ramp of tests/test_accepted_on.py on its
    closed-form structure) has no ICB test: icb_superheat and icb_adiabat are None."""
    from tests.test_accepted_on import _analytic_ramp, _canned_structure

    res, _ = _analytic_ramp(structure=_canned_structure, max_iterations_outer=6)
    assert res['icb_superheat'] is None and res['icb_adiabat'] is None


def test_a_full_inner_core_takes_the_cmb_node_pressure_above_the_reader():
    """A full inner core takes the pressure of the CMB node of the core arrays also where the
    CMB point of the ICB readers lies above it (the first mantle node raised to the pressure of
    the last core node)."""
    res, rhs = _sphere_result(41, 1.9)
    m, cmb = res['mass_enclosed'], res['cmb_mass']
    n = int(np.count_nonzero(m < cmb))
    res['pressure'] = res['pressure'].copy()
    res['pressure'][n] = res['pressure'][n - 1]
    mix = {'core': LayerMixture([LIQ], [1.0]), 'inner_core': LayerMixture([SOL], [1.0])}
    core = core_arrays(res, mix, rhs, lambda r, p_: 5000.0, cmb, 0.0)
    reader = icb_from_profile(m, res['pressure'], res['temperature'], cmb, _liq, cmb)[1]
    assert reader > core['P'][-1] * (1.0 + 1e-6)  # the case that max(reader, P[-1]) gets wrong
    assert core['P_icb'] == core['P'][-1]
