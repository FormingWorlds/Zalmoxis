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
    300-1e5 K, the box and resolution of the shipped unified iron table."""
    grid = make_default_grid_iron_2phase()
    assert (grid.p_lo, grid.p_hi, grid.t_lo, grid.t_hi) == (1e5, 1e14, 300.0, 1e5)
    assert (grid.n_p, grid.n_t) == (1351, 380)
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
