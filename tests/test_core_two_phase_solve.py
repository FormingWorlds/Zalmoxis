"""Full structure solves with a two-layer core: a solid inner core on the PALEOS solid iron side
and a liquid outer core on the liquid side, with the inner-core boundary (ICB) from an iron
melting curve or from a given inner-core mass.

The side tables are generated from the real PALEOS on a small grid over core conditions, in a
scratch cache root, so no test writes the data cache.
"""

from __future__ import annotations

import copy
import os

import numpy as np
import pytest
from scipy.optimize import brentq

pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]

LIQ, SOL = 'PALEOS-API-2phase:iron-liquid', 'PALEOS-API-2phase:iron-solid'


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    """Material dictionaries with the 2 iron sides on a 40 pts/decade grid of 1e9-3e12 Pa and
    1000-2e4 K, generated once into a scratch cache root, and the default config."""
    paleos = pytest.importorskip('paleos')
    if getattr(paleos, '__version__', '') == '0.0.0-mock':
        pytest.skip('needs the real PALEOS')
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import load_material_dictionaries, load_zalmoxis_config
    from zalmoxis.eos import paleos_api, paleos_api_cache

    md = load_material_dictionaries()
    if not os.path.isfile(md['PALEOS:iron'].get('eos_file', '')):
        pytest.skip('PALEOS iron table not linked under data/')
    grid = paleos_api.make_grid_at_resolution(1e9, 3e12, 1000.0, 2.0e4, 40)
    for name in (LIQ, SOL):
        md[name] = dict(copy.deepcopy(md[name]), grid_spec=grid)
    with pytest.MonkeyPatch.context() as mp:
        root = tmp_path_factory.mktemp('paleos_api_cache')
        mp.setattr(paleos_api_cache, '_cache_root', lambda: root)
        mp.setattr(paleos_api_cache, '_check_paleos_once', lambda sha: 'tested')
        paleos_api_cache.resolve_paleos_api_2phase_iron(grid, n_workers=4)
        for name in (LIQ, SOL):
            paleos_api_cache.resolve_registry_entry(md[name])
    root_dir = get_zalmoxis_root()
    cfg = load_zalmoxis_config(os.path.join(root_dir, 'input', 'default.toml'))
    return md, cfg, os.path.join(root_dir, 'input')


def _solve(setup, inner='', core=LIQ, **over):
    """Solve the default 1 M_E config with the given core layers and overrides."""
    from zalmoxis.config import load_solidus_liquidus_functions, validate_config
    from zalmoxis.mixing import parse_all_layer_mixtures
    from zalmoxis.solver import main

    md, cfg, input_dir = setup
    cfg = dict(cfg, **over)
    cfg['layer_eos_config'] = dict(cfg['layer_eos_config'], core=core)
    if inner:
        cfg['layer_eos_config']['inner_core'] = inner
    validate_config(cfg)
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'], cfg['rock_solidus'], cfg['rock_liquidus'], 1.0
    )
    mixes = parse_all_layer_mixtures(cfg['layer_eos_config'])
    res = main(cfg, md, melt, input_dir, layer_mixtures=mixes)
    assert res['converged']
    return res


def _anzellini_inverse(t):
    from zalmoxis.melting_curves import iron_melting_anzellini13

    return brentq(
        lambda p: iron_melting_anzellini13(p) - t, 1.2e11, 2e12, xtol=1e-3, rtol=1e-15
    )


@pytest.mark.physics_invariant
def test_an_isothermal_core_has_its_icb_where_iron_melts(setup):
    """At 5000 K everywhere the ICB lies at the pressure where the Anzellini curve is 5000 K
    (206 GPa), inner-core nodes above it and outer-core nodes below it in pressure, and the ICB
    mass at that pressure on the final profile to the outer-loop tolerance."""
    res = _solve(
        setup,
        inner=SOL,
        temperature_mode='isothermal',
        surface_temperature=5000.0,
        core_liquidus='Anzellini13-iron',
    )
    core = res['core']
    p_icb = _anzellini_inverse(5000.0)
    assert core['P_icb'] == pytest.approx(p_icb, rel=1e-12)
    inner = core['phase'] == 1
    assert inner[0] and not inner[-1]
    assert np.all(core['P'][inner] > p_icb * (1 - 1e-3))
    assert np.all(core['P'][~inner] < p_icb * (1 + 1e-3))
    # the ICB mass comes from the previous pass; the outer loop stops when it moves by less
    # than tolerance_outer of the core mass
    from zalmoxis.solver import _default_solver_params

    tol = _default_solver_params(res['mass_enclosed'][-1])['tolerance_outer']
    m_at = np.interp(np.log(p_icb), np.log(core['P'][::-1]), core['m'][::-1])
    assert abs(core['M_ic'] - m_at) <= tol * res['cmb_mass']
    assert core['solid_above_icb_mass_fraction'] == 0.0


def test_a_given_inner_core_mass_sets_the_icb(setup):
    res = _solve(setup, inner=SOL, core_liquidus='', inner_core_mass_fraction=0.05)
    core = res['core']
    assert core['M_ic'] == 0.05 * res['cmb_mass']
    inner = core['phase'] == 1
    assert np.all(core['m'][inner] < core['M_ic'])
    assert np.all(core['m'][~inner] >= core['M_ic'] * (1 - 1e-12))


@pytest.mark.parametrize(('t', 'all_solid'), [(8000.0, False), (3000.0, True)])
def test_a_core_wholly_above_or_below_the_liquidus(setup, t, all_solid):
    """At 8000 K the centre is above the liquidus: no inner core; at 3000 K the whole core is
    below it: all inner core."""
    res = _solve(
        setup,
        inner=SOL,
        temperature_mode='isothermal',
        surface_temperature=t,
        core_liquidus='Anzellini13-iron',
    )
    core = res['core']
    assert core['M_ic'] == (res['cmb_mass'] if all_solid else 0.0)
    assert np.all(core['phase'][:-1] == int(all_solid))


def test_solid_above_a_liquid_centre_is_reported(setup):
    """A linear profile from 7000 K at the centre puts the centre above the liquidus and the
    core top below it: no inner core, the solid mass above the ICB reported and flagged."""
    res = _solve(
        setup,
        inner=SOL,
        temperature_mode='linear',
        center_temperature=7000.0,
        surface_temperature=300.0,
        core_liquidus='Anzellini13-iron',
    )
    core = res['core']
    assert core['M_ic'] == 0.0
    assert 0.0 < core['solid_above_icb_mass_fraction'] < 1.0
    assert core['solid_above_icb']


@pytest.mark.physics_invariant
def test_the_core_arrays_of_a_two_layer_core(setup):
    """The core arrays hold r, m, P, rho, g, T, phase and w_S per node from the centre to a CMB
    node at the core mass; the outer core carries its S, the inner core none."""
    res = _solve(
        setup,
        inner=SOL,
        core=f'{LIQ}:0.8615+CoreAlloy:S:0.1385',
        core_liquidus='Anzellini13-iron',
        temperature_mode='isothermal',
        surface_temperature=5000.0,
    )
    core = res['core']
    for k in ('r', 'm', 'P', 'rho', 'g', 'T', 'phase', 'w_S'):
        assert core[k].shape == core['r'].shape, k
    assert core['r'][0] == 0.0 and np.all(np.diff(core['r']) > 0)
    assert core['m'][-1] == pytest.approx(res['cmb_mass'], rel=1e-10)
    inner = core['phase'] == 1
    assert np.all(core['w_S'][inner] == 0.0) and np.all(core['w_S'][~inner] == 0.1385)
    assert core['R_icb'] > 0.0 and core['P_icb'] > core['P'][-1]
