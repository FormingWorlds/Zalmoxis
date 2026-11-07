"""Two-layer core: the iron side tables in the registry and the configuration of an inner core."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from zalmoxis.config import parse_eos_config, validate_config, validate_layer_eos
from zalmoxis.eos import core_alloy as ca
from zalmoxis.eos.paleos_api import make_default_grid_iron_2phase
from zalmoxis.eos_properties import EOS_REGISTRY
from zalmoxis.mixing import LayerMixture

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


@pytest.mark.parametrize(
    ('layers', 'match'),
    [
        ({'core': LIQ, 'inner_core': LIQ}, 'inner_core'),
        ({'core': SOL, 'inner_core': SOL}, "'core'"),
        ({'core': 'PALEOS:iron', 'inner_core': SOL}, "'core'"),
        ({'core': LIQ}, 'inner_core'),
        ({'core': 'PALEOS:iron', 'mantle_extra': LIQ}, 'core'),
        ({'core': LIQ, 'inner_core': 'PALEOS:iron:0.9+CoreAlloy:S:0.1'}, 'inner_core'),
    ],
)
def test_each_side_belongs_to_its_core_layer(layers, match):
    """The liquid side is the outer core of a two-layer core, the solid side its inner core."""
    with pytest.raises(ValueError, match=match):
        validate_layer_eos({'mantle': 'PALEOS:MgSiO3', **layers})


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
        isinstance(value, (int, float))
        and not isinstance(value, bool)
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
    import numpy as np

    p = np.linspace(360e9, 136e9, n)
    m = 1e24 * np.log(360e9 / p) / np.log(360 / 136)
    return m * (1 - 1e-9), p, t_of_p(p)  # the last node just below the core mass


def _liq(p):
    return 4000.0 + 1e-8 * (p - 136e9)  # 4000 K at 136 GPa, 6240 K at 360 GPa


@pytest.mark.physics_invariant
def test_the_icb_is_where_the_profile_meets_the_liquidus():
    """An isothermal 5000 K core meets the linear liquidus at 236 GPa; the ICB mass is the
    enclosed mass there, and no solid lies above it."""
    from zalmoxis.core_profile import icb_from_profile

    m, p, t = _profile(lambda p: 0 * p + 5000.0)
    m_icb, p_icb, solid = icb_from_profile(m, p, t, 1e24, _liq)
    assert p_icb == pytest.approx(236e9, rel=1e-12)
    assert m_icb == pytest.approx(
        1e24 * math.log(360 / 236) / math.log(360 / 136) * (1 - 1e-9), rel=1e-12
    )
    assert solid == 0.0


def test_a_core_above_or_below_the_liquidus_everywhere():
    """A centre at or above the liquidus gives no inner core; a core below it everywhere is
    the inner core, with the pressure of its last core node."""
    from zalmoxis.core_profile import icb_from_profile

    m, p, t = _profile(lambda p: 0 * p + 7000.0)
    assert icb_from_profile(m, p, t, 1e24, _liq)[:2] == (0.0, p[0])
    m, p, t = _profile(lambda p: 0 * p + 3000.0)
    assert icb_from_profile(m, p, t, 1e24, _liq) == (1e24, p[-1], 0.0)


def test_solid_above_a_liquid_centre_is_the_mass_of_its_segments():
    """A centre above the liquidus with a cold top: the solid mass above the ICB is the mass of
    the node segments whose 2 ends lie below the liquidus."""
    import numpy as np

    from zalmoxis.core_profile import icb_from_profile

    m, p, t = _profile(lambda p: np.where(p > 200e9, 7000.0, 3000.0))
    m_icb, _, solid = icb_from_profile(m, p, t, 1e24, _liq)
    below = p <= 200e9
    seg = below[:-1] & below[1:]
    assert m_icb == 0.0 and solid == pytest.approx(float(np.sum(np.diff(m)[seg])), rel=1e-15)
    assert 0.0 < solid < 1e24


def test_the_liquidus_rule_is_refused_with_the_newton_outer_solver():
    """Each Newton probe solves one outer pass, so the liquidus rule would have no profile of a
    previous pass; the mass rule needs none."""
    with pytest.raises(ValueError, match='core_liquidus needs the picard'):
        validate_config(_params(outer_solver='newton'))
    validate_config(
        _params(outer_solver='newton', core_liquidus='', inner_core_mass_fraction=0.05)
    )


@pytest.mark.parametrize(
    ('layers', 'ok'),
    [
        ({'core': f'CoreAlloy:S:0.1385+{LIQ}:0.8615', 'inner_core': SOL}, True),
        ({'core': f'PALEOS:iron:0.5+{LIQ}:0.5'}, False),
        ({'core': f'PALEOS:iron:0.5+{LIQ}:0.5', 'inner_core': SOL}, False),
        ({'core': LIQ, 'inner_core': f'CoreAlloy:Si:0.01+{SOL}:0.99'}, True),
    ],
)
def test_the_side_of_a_core_layer_is_its_iron_base_in_any_order(layers, ok):
    """The base of a core layer is its iron component wherever it stands; a liquid side in a
    core without an inner core, or a second iron base next to it, is refused."""
    layers = {'mantle': 'PALEOS:MgSiO3', **layers}
    if ok:
        validate_layer_eos(layers)
    else:
        with pytest.raises(ValueError):
            validate_layer_eos(layers)


@pytest.mark.parametrize('bad', ['liquidus', 'temperature'])
def test_a_non_finite_liquidus_or_temperature_is_refused(bad):
    """A NaN in the liquidus or the profile would read as below the liquidus everywhere and make
    the whole core the inner core."""
    import numpy as np

    from zalmoxis.core_profile import icb_from_profile

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
    import numpy as np

    from zalmoxis.core_profile import icb_from_profile

    k = math.log(360 / 136)
    m, p, t = _profile(lambda p: 4500.0 + 1000.0 / k * np.log(p / 136e9))
    m_icb, p_icb, solid = icb_from_profile(
        m, p, t, 1e24, lambda p: 4000.0 + 2240.0 / k * math.log(p / 136e9)
    )
    assert p_icb == pytest.approx(136e9 * (360 / 136) ** (500 / 1240), rel=1e-12)
    assert m_icb == pytest.approx(1e24 * (1 - 500 / 1240) * (1 - 1e-9), rel=1e-12)
    assert solid == 0.0


def _sphere_result(n_nodes, cmb_scale):
    """A uniform 10000 kg/m^3 sphere on n_nodes nodes, its rhs, and a core mass of cmb_scale
    times the mass of node n_nodes // 2."""
    import numpy as np

    from zalmoxis.constants import G

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
    import numpy as np

    from zalmoxis.core_profile import core_arrays

    res, rhs = _sphere_result(41, 1.0 / (1.0 - 5e-13))
    mix = {'core': LayerMixture(['PALEOS:iron'], [1.0])}
    core = core_arrays(res, mix, rhs, lambda r, p: 5000.0, 0.0, 0.0, 0.0)
    assert abs(core['m'][-1] / res['cmb_mass'] - 1) < 1e-12
    assert np.all(np.diff(core['r']) > 0)


def test_core_arrays_without_a_node_past_the_core_mass_are_refused():
    from zalmoxis.core_profile import core_arrays

    res, rhs = _sphere_result(41, 10.0)
    mix = {'core': LayerMixture(['PALEOS:iron'], [1.0])}
    with pytest.raises(ValueError, match='core mass'):
        core_arrays(res, mix, rhs, lambda r, p: 5000.0, 0.0, 0.0, 0.0)


def test_the_phase_columns_label_the_inner_core_by_its_side():
    """Shells below the ICB mass take the inner-core EOS: the solid side is solid, the liquid
    side liquid."""
    import numpy as np

    from zalmoxis.phase_columns import compute_layer_phase_columns

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
