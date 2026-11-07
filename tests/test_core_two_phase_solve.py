"""Full structure solves with a two-layer core: a solid inner core on the PALEOS solid iron side
and a liquid outer core on the liquid side, with the inner-core boundary (ICB) from an iron
melting curve or from a given inner-core mass.

The side tables are generated from the real PALEOS on a small grid over core conditions, in a
scratch cache root, so no test writes the data cache.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

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
def test_an_isothermal_core_has_its_icb_where_iron_melts(setup, caplog):
    """At 5000 K everywhere the ICB lies at the pressure where the Anzellini curve is 5000 K
    (206 GPa), inner-core nodes above it and outer-core nodes below it in pressure, and the ICB
    mass at that pressure on the final profile to the outer-loop tolerance."""
    caplog.set_level('WARNING', logger='fwl.zalmoxis.solver')
    res = _solve(
        setup,
        inner=SOL,
        temperature_mode='isothermal',
        surface_temperature=5000.0,
        core_liquidus='Anzellini13-iron',
    )
    core = res['core']
    p_icb = _anzellini_inverse(5000.0)
    # the ICB scalars belong to the solved layers: P_icb is the pressure at M_ic
    assert core['P_icb'] == pytest.approx(
        np.interp(core['M_ic'], core['m'], core['P']), rel=1e-12
    )
    # M_ic is within tolerance_outer (1e-3 of the core mass) of the root and dlnP/d(m/M_core) is
    # about 1 here, so P_icb is within about 1e-3 of the root: 5e-3 keeps a factor 5
    assert abs(core['P_icb'] / p_icb - 1) < 5e-3
    assert 'pure-iron curve' not in caplog.text  # a pure-iron outer core: no warning
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


def test_main_refuses_the_liquidus_rule_with_the_newton_outer_solver(setup):
    """main itself refuses the pair, also when the caller skips validate_config."""
    from zalmoxis.mixing import parse_all_layer_mixtures
    from zalmoxis.solver import main

    md, cfg, input_dir = setup
    cfg = dict(cfg, core_liquidus='Anzellini13-iron', outer_solver='newton')
    cfg.update(relative_tolerance=1e-9, absolute_tolerance=1e-10)  # what Newton needs
    cfg['layer_eos_config'] = dict(cfg['layer_eos_config'], core=LIQ, inner_core=SOL)
    with pytest.raises(ValueError, match="core_liquidus needs outer_solver = 'picard'"):
        main(
            cfg,
            md,
            None,
            input_dir,
            layer_mixtures=parse_all_layer_mixtures(cfg['layer_eos_config']),
        )


def test_a_two_layer_core_keeps_the_mass_limit_of_the_paleos_tables(setup):
    """The side tables span the box of the unified iron table, so they take its 50 M_E limit;
    the mantle has no limit here, so the refusal names a side table."""
    from zalmoxis.mixing import parse_all_layer_mixtures
    from zalmoxis.solver import main

    md, cfg, input_dir = setup
    cfg = dict(cfg, inner_core_mass_fraction=0.05, planet_mass=60 * 5.972e24)
    cfg['layer_eos_config'] = {'core': LIQ, 'inner_core': SOL, 'mantle': 'Seager2007:MgSiO3'}
    with pytest.raises(ValueError, match='PALEOS-API-2phase:iron-.* EOS is limited to planets'):
        main(
            cfg,
            md,
            None,
            input_dir,
            layer_mixtures=parse_all_layer_mixtures(cfg['layer_eos_config']),
        )


def test_an_accept_with_an_icb_off_the_liquidus_is_not_converged(setup, caplog):
    """2 outer passes end on the max_iter accept while the ICB still moves: _solve returns the
    result as not converged and says why, and main does not take it back by its retry skip."""
    from zalmoxis import solver as zs
    from zalmoxis.mixing import parse_all_layer_mixtures

    md, cfg, input_dir = setup
    cfg = dict(
        cfg,
        temperature_mode='isothermal',
        surface_temperature=5000.0,
        core_liquidus='Anzellini13-iron',
        max_iterations_outer=2,
    )
    cfg['layer_eos_config'] = dict(cfg['layer_eos_config'], core=LIQ, inner_core=SOL)
    mixes = parse_all_layer_mixtures(cfg['layer_eos_config'])
    res = zs._solve(cfg, md, None, input_dir, layer_mixtures=mixes)
    assert res['converged'] is False and res['icb_converged'] is False
    assert 'inner-core boundary' in caplog.text
    assert res['converged_mass'] is False and res['accept_mass_limit'] is None
    res = zs.main(cfg, md, None, input_dir, layer_mixtures=mixes)
    assert (
        res['accepted_on'] != 'retry_skip' and 'Retrying with tighter parameters' in caplog.text
    )
    assert res['converged'] and res['icb_converged']


def test_main_refuses_an_icb_rule_without_an_inner_core(setup):
    """main itself refuses core_liquidus without an inner core; the rule check takes None as
    unset and an empty inner_core as no inner core."""
    from zalmoxis.mixing import parse_all_layer_mixtures
    from zalmoxis.solver import main

    md, cfg, input_dir = setup
    cfg = dict(cfg, core_liquidus='Anzellini13-iron')
    with pytest.raises(ValueError, match='need an inner_core layer'):
        main(
            cfg,
            md,
            None,
            input_dir,
            layer_mixtures=parse_all_layer_mixtures(cfg['layer_eos_config']),
        )
    from zalmoxis.config import _check_icb_rule

    assert _check_icb_rule(dict(cfg, core_liquidus=None)) is None  # None counts as unset
    one_layer = {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3', 'inner_core': ''}
    assert _check_icb_rule({'layer_eos_config': one_layer}) is None  # '' is no inner core


def test_solve_ignores_a_liquidus_without_an_inner_core(setup):
    """_solve, called past the checks of main, solves a one-layer core with no inner core."""
    from zalmoxis import solver as zs
    from zalmoxis.mixing import parse_all_layer_mixtures

    md, cfg, input_dir = setup
    cfg = dict(
        cfg,
        core_liquidus='Anzellini13-iron',
        temperature_mode='isothermal',
        surface_temperature=5000.0,
    )
    res = zs._solve(
        cfg,
        md,
        None,
        input_dir,
        layer_mixtures=parse_all_layer_mixtures(cfg['layer_eos_config']),
    )
    assert res['converged'] and res['core']['M_ic'] == 0.0


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
    assert np.all(core['phase'] == int(all_solid))  # the node at the core mass too


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
def test_the_core_arrays_of_a_two_layer_core(setup, caplog):
    """The core arrays hold r, m, P, rho, g, T, phase and w_S per node from the centre to a CMB
    node at the core mass; the outer core carries its S, the inner core none."""
    caplog.set_level('WARNING', logger='fwl.zalmoxis.solver')
    res = _solve(
        setup,
        inner=SOL,
        core=f'{LIQ}:0.8615+CoreAlloy:S:0.1385',
        core_liquidus='Anzellini13-iron',
        temperature_mode='isothermal',
        surface_temperature=5000.0,
    )
    core = res['core']
    assert 'pure-iron curve' in caplog.text  # a named curve with an alloy outer core
    assert core['w_layers'] == {'core': {'S': 0.1385}, 'inner_core': {}}
    for k in ('r', 'm', 'P', 'rho', 'g', 'T', 'phase', 'w_S'):
        assert core[k].shape == core['r'].shape, k
    assert core['r'][0] == 0.0 and np.all(np.diff(core['r']) > 0)
    assert core['m'][-1] == pytest.approx(res['cmb_mass'], rel=1e-10)
    inner = core['phase'] == 1
    assert np.all(core['w_S'][inner] == 0.0) and np.all(core['w_S'][~inner] == 0.1385)
    assert core['R_icb'] > 0.0 and core['P_icb'] > core['P'][-1]
    # the node at the core mass takes the outer-core EOS, not the mantle one
    from zalmoxis.mixing import LayerMixture, calculate_mixed_density

    outer = LayerMixture([LIQ, 'CoreAlloy:S'], [0.8615, 0.1385])
    rho_cmb = calculate_mixed_density(
        core['P'][-1], core['T'][-1], outer, setup[0], None, None, {}
    )
    assert core['rho'][-1] == pytest.approx(rho_cmb, rel=1e-6) and core['phase'][-1] == 0


PREM_DATA = Path(__file__).parent / 'validation' / 'data'
# Hirose et al. 2021, PDF p. 3: midpoints of the single-element outer cores [wt%]
HIROSE_MID = {'S': 13.85, 'Si': 9.55, 'O': 8.65, 'H': 0.975}


def _prem_core(n=40001):
    """PREM core (Dziewonski & Anderson 1981): Table I density, and pressure and gravity by
    hydrostatics inward from the Table II CMB pressure; ``region`` 0 inner, 1 outer core."""
    from scipy.integrate import cumulative_trapezoid

    from zalmoxis.constants import G

    read = dict(delimiter=',', names=True, dtype=None, encoding=None, skip_header=2)
    poly = np.genfromtxt(PREM_DATA / 'prem_core_polynomials.csv', **read)
    tab = np.genfromtxt(PREM_DATA / 'prem_core_table2.csv', **read)
    r_icb, r_cmb = poly['r_hi_km'] * 1e3
    r = np.r_[np.linspace(0.0, r_icb, n), np.linspace(r_icb, r_cmb, n)]
    region = np.repeat([0, 1], n)
    rho = 1e3 * sum(poly[f'a{i}'][region] * (r / 6371e3) ** i for i in range(4))
    m = cumulative_trapezoid(4 * np.pi * r**2 * rho, r, initial=0.0)
    g = np.divide(G * m, r**2, out=np.zeros_like(r), where=r > 0)
    w = cumulative_trapezoid(rho * g, r, initial=0.0)
    p = tab['pressure_kbar'][-1] * 1e8 + w[-1] - w
    return dict(r=r, rho=rho, P=p, g=g, m=m, region=region, tab=tab)


def _against_prem(core, prem, by_phase):
    """Median and largest rho / rho_PREM - 1 per region at equal pressure, and the node count;
    the region of a node is its phase (two-layer core) or its side of the PREM ICB pressure."""
    P, rho = core['P'], core['rho']
    p_icb = prem['P'][prem['region'] == 0][-1]
    region = np.where(core['phase'] == 1, 0, 1) if by_phase else np.where(P > p_icb, 0, 1)
    out = {}
    for k, name in enumerate(('inner', 'outer')):
        sel = prem['region'] == k
        p_k, rho_k = prem['P'][sel][::-1], prem['rho'][sel][::-1]
        hit = (region == k) & (P >= p_k[0]) & (P <= p_k[-1])
        d = rho[hit] / np.interp(P[hit], p_k, rho_k) - 1.0
        out[name] = (float(np.median(d)), float(d[np.argmax(np.abs(d))]), int(hit.sum()))
    return out


# median and largest rho / rho_PREM - 1 per region at equal pressure, macOS arm64 reference
PREM_PINS = {
    'Fe, 1 layer': {'inner': (0.020196, 0.021164), 'outer': (0.080497, 0.098991)},
    'Fe, 2 layers': {'inner': (0.030165, 0.030997), 'outer': (0.080461, 0.098956)},
    'Fe-S, 2 layers': {'inner': (0.031505, 0.034530), 'outer': (0.004332, 0.021273)},
}


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_an_earth_core_against_prem(setup, validation_plot):
    """PREM (Dziewonski & Anderson 1981): the hydrostatic PREM core meets Table II, and the core
    of the default 1 M_E planet differs from it at equal pressure by the pinned medians and
    largest values per region. The 3 cores: one-layer PALEOS:iron, a two-layer pure-iron core
    and a two-layer core with the 13.85 wt% S outer core of Hirose et al. (2021), both with the
    PREM inner-core mass fraction. The Si, O and H midpoint cores have the S density by the
    calibration of the linear law, so they give the same planet."""
    prem = _prem_core()
    tab, inner = prem['tab'], prem['region'] == 0
    np.testing.assert_allclose(
        np.interp(tab['radius_km'] * 1e3, prem['r'], prem['rho'])[[0, 3]],
        tab['density_g_cm3'][[0, 3]] * 1e3,
        rtol=1e-5,
    )
    for got, want in (
        (prem['P'][0], tab['pressure_kbar'][0] * 1e8),
        (prem['P'][inner][-1], tab['pressure_kbar'][1] * 1e8),
        (prem['g'][-1], tab['gravity_cm_s2'][3] / 100),
    ):
        assert abs(got / want - 1) < 1e-3
    f_ic = prem['m'][inner][-1] / prem['m'][-1]
    s_core = f'{LIQ}:{1 - HIROSE_MID["S"] / 100:.4f}+CoreAlloy:S:{HIROSE_MID["S"] / 100:.4f}'
    cores = {
        'Fe, 1 layer': _solve(setup, core='PALEOS:iron')['core'],
        'Fe, 2 layers': _solve(setup, inner=SOL, inner_core_mass_fraction=f_ic)['core'],
        'Fe-S, 2 layers': _solve(setup, inner=SOL, core=s_core, inner_core_mass_fraction=f_ic)[
            'core'
        ],
    }
    for name, core in cores.items():
        got = _against_prem(core, prem, by_phase=name != 'Fe, 1 layer')
        for region, (median, largest, n) in got.items():
            assert n >= 5, (name, region, n)
            np.testing.assert_allclose(
                (median, largest), PREM_PINS[name][region], atol=1e-3, err_msg=name
            )

    def figure(x, scale, label):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.6, 3.6))
        for k in (0, 1):
            sel = prem['region'] == k
            ax.plot(
                prem[x][sel] * scale,
                prem['rho'][sel],
                '--',
                lw=1.2,
                color=plt.rcParams['axes.edgecolor'],
                label='PREM' if k == 0 else None,
            )
        colours = plt.rcParams['axes.prop_cycle'].by_key()['color'][1:]
        for (name, core), colour in zip(cores.items(), colours):
            ax.plot(core[x][:-1] * scale, core['rho'][:-1], lw=1.2, color=colour, label=name)
        ax.set(xlabel=label, ylabel='Density [kg/m$^3$]')
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.grid(False)
        if x == 'P':  # centre on the left, as by radius
            ax.invert_xaxis()
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
        fig.tight_layout()
        fig.canvas.draw()
        assert ax.get_legend().get_window_extent().x0 >= ax.get_window_extent().x1
        return fig

    validation_plot(lambda: figure('P', 1e-9, 'Pressure [GPa]'), 'core_two_phase_prem_pressure')
    validation_plot(lambda: figure('r', 1e-3, 'Radius [km]'), 'core_two_phase_prem_radius')


@pytest.mark.reference_pinned
def test_the_bulk_s_of_the_core_follows_the_icb(setup):
    """The layer compositions are inputs, so the bulk S of a Fe-S core changes with the ICB: an
    isothermal core at 5000 and 5500 K puts 0.480 and 0.159 of the core mass in the inner core
    (pins to 0.01, 3 times the spread over 5 perturbed starts at 5000 K), and the bulk S falls
    from 13.85 wt% of the outer core to 7.20 and 11.65 wt% of the core."""
    w_oc = 0.1385
    for t, f_pin, s_pin in ((5000.0, 0.480, 0.0720), (5500.0, 0.159, 0.1165)):
        res = _solve(
            setup,
            inner=SOL,
            core=f'{LIQ}:{1 - w_oc:.4f}+CoreAlloy:S:{w_oc:.4f}',
            temperature_mode='isothermal',
            surface_temperature=t,
            core_liquidus='Anzellini13-iron',
        )
        core = res['core']
        f = core['M_ic'] / res['cmb_mass']
        assert f == pytest.approx(f_pin, abs=0.01)
        assert (1 - f) * w_oc == pytest.approx(s_pin, abs=0.01 * w_oc)
