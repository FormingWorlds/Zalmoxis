"""The outer node of a structure whose integration stops just below the outer radius.

The analytic iron/MgSiO3 planet of ``tests.test_failed_structure_solve._cfg`` converges with
P = 0 at the outer node: the pressure-zero event fires inside the last shell and the node is
padded as the surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.test_failed_structure_solve import _cfg, _run
from zalmoxis.config import load_material_dictionaries
from zalmoxis.constants import G
from zalmoxis.mixing import calculate_mixed_density, parse_all_layer_mixtures

pytestmark = pytest.mark.smoke


@pytest.fixture(scope='module')
def padded():
    """An accepted solve whose outer node is padded."""
    r = _run(_cfg())
    assert r['converged'] and np.asarray(r['pressure'])[-1] == 0.0
    return r


@pytest.mark.timeout(600)
def test_a_padded_surface_node_has_the_eos_density_at_the_target_pressure(padded):
    cfg = _cfg()
    mix = parse_all_layer_mixtures(cfg['layer_eos_config'])['mantle']
    ref = calculate_mixed_density(
        cfg['target_surface_pressure'],
        padded['temperature'][-1],
        mix,
        load_material_dictionaries(),
        None,
        None,
        {},
        None,
    )
    assert ref > 0
    assert padded['density'][-1] == pytest.approx(ref, rel=1e-8)


@pytest.mark.timeout(600)
def test_a_padded_surface_node_has_the_gravity_of_the_planet_mass_at_its_radius(padded):
    m, r, g = (np.asarray(padded[k]) for k in ('mass_enclosed', 'radii', 'gravity'))
    assert g[-1] == pytest.approx(G * m[-1] / r[-1] ** 2, rel=1e-12)


@pytest.mark.timeout(900)
def test_a_padded_node_of_a_cmb_anchored_adiabat_takes_the_top_of_the_adiabat(monkeypatch):
    """With the adiabat integrated from the CMB, the configured surface_temperature is not the
    temperature at the surface: the padded node takes the adiabat continued from the node
    below to the target pressure (checked against a 4000-step integration), and its EOS
    density there."""
    import os

    import zalmoxis.solver as zs
    from tests.test_spider_coupling_convergence import _has_paleos_data
    from zalmoxis.config import load_solidus_liquidus_functions, load_zalmoxis_config

    if not _has_paleos_data():
        pytest.skip('PALEOS tables not on disk')
    root = os.path.normpath(os.path.join(os.path.dirname(zs.__file__), '..', '..'))
    cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    cfg.update(
        temperature_mode='adiabatic_from_cmb',
        cmb_temperature=4000.0,
        num_layers=50,
        data_output_enabled=False,
        plotting_enabled=False,
    )
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        cfg['rock_solidus'],
        cfg['rock_liquidus'],
        cfg['mushy_zone_factor'],
    )
    md, spies = load_material_dictionaries(), {}
    for name in ('_surface_density', '_adiabat_to_surface'):
        real = getattr(zs, name)

        def spy(*args, _real=real, _name=name):
            spies[_name] = args
            return _real(*args)

        monkeypatch.setattr(zs, name, spy)
    r = zs.main(cfg, md, melt, os.path.join(root, 'input'))
    P, T, rho = (np.asarray(r[k], float) for k in ('pressure', 'temperature', 'density'))
    assert r['converged'] and r['adiabat_blend'] == 1.0 and P[-1] == 0.0 and P[-2] > 0.0
    # both lookups get the arguments of calculate_mixed_density_batch after the mixture
    eos = (md, *(melt or (None, None)), zs._interpolation_cache, *zs._eos_params(cfg))
    for name, i in (('_surface_density', 6), ('_adiabat_to_surface', 4)):
        assert len(spies[name][i]) == len(eos)
        assert all(x is y or x == y for x, y in zip(spies[name][i], eos)), name
    from zalmoxis.mixing import get_mixed_nabla_ad

    mixes = parse_all_layer_mixtures(cfg['layer_eos_config'])
    mzf, rho_min, rho_scale, bts = zs._eos_params(cfg)
    md, sol_liq, cache = load_material_dictionaries(), (melt or (None, None)), {}
    t_ref, lnp = T[-2], np.linspace(np.log(P[-2]), np.log(cfg['target_surface_pressure']), 4001)
    for lo, hi in zip(lnp[:-1], lnp[1:]):  # explicit Euler in ln P
        nab = get_mixed_nabla_ad(
            np.exp(lo),
            t_ref,
            mixes['mantle'],
            md,
            cache,
            *sol_liq,
            mzf,
            rho_min,
            rho_scale,
            bts,
        )
        t_ref *= np.exp(nab * (hi - lo))
    assert t_ref < T[-2] - 50.0
    assert T[-1] == pytest.approx(t_ref, abs=1.0)
    mix = parse_all_layer_mixtures(cfg['layer_eos_config'])['mantle']
    ref = calculate_mixed_density(
        cfg['target_surface_pressure'],
        T[-1],
        mix,
        load_material_dictionaries(),
        *(melt or (None, None)),
        {},
        cfg['mushy_zone_factor'],
    )
    assert rho[-1] == pytest.approx(ref, rel=1e-8)


@pytest.mark.timeout(600)
@pytest.mark.parametrize('outer, blend', [(2, 0.25), (3, 0.5), (4, 0.75)])
def test_a_padded_node_of_a_cmb_run_accepted_mid_ramp_keeps_the_linear_profile_at_r(
    outer, blend
):
    """An accept before the adiabat blend reaches 1 is not a CMB-anchored adiabat: the padded
    node keeps the linear profile at R (the configured surface_temperature), not a continuation
    of the blended temperature below it."""
    import os

    import zalmoxis.solver as zs
    from tests.test_spider_coupling_convergence import _has_paleos_data
    from zalmoxis.config import load_solidus_liquidus_functions, load_zalmoxis_config

    if not _has_paleos_data():
        pytest.skip('PALEOS tables not on disk')
    root = os.path.normpath(os.path.join(os.path.dirname(zs.__file__), '..', '..'))
    cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    cfg.update(
        temperature_mode='adiabatic_from_cmb',
        cmb_temperature=4000.0,
        num_layers=50,
        max_iterations_outer=outer,
        wall_timeout=1e6,
        data_output_enabled=False,
        plotting_enabled=False,
    )
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        cfg['rock_solidus'],
        cfg['rock_liquidus'],
        cfg['mushy_zone_factor'],
    )
    r = zs.main(cfg, load_material_dictionaries(), melt, os.path.join(root, 'input'))
    P, T = (np.asarray(r[k], float) for k in ('pressure', 'temperature'))
    assert r['converged'] and r['adiabat_blend'] == blend and r['adiabat_computed']
    assert P[-1] == 0.0 and T[-1] == cfg['surface_temperature'] != T[-2]


@pytest.mark.timeout(900)
def test_both_padded_lookups_get_the_eos_arguments_of_the_solve_in_order(monkeypatch):
    """On a PALEOS 2-phase mantle with its melting curves and condensed_rho_scale apart from
    binodal_T_scale, every entry of the argument tuple differs, so a swap of two of them in
    _solve breaks the comparison."""
    import os

    import zalmoxis.solver as zs
    from tests.test_spider_coupling_convergence import _has_paleos_data
    from zalmoxis.config import load_solidus_liquidus_functions, load_zalmoxis_config

    if not _has_paleos_data():
        pytest.skip('PALEOS tables not on disk')
    root = os.path.normpath(os.path.join(os.path.dirname(zs.__file__), '..', '..'))
    cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    cfg['layer_eos_config'] = {**cfg['layer_eos_config'], 'mantle': 'PALEOS-2phase:MgSiO3'}
    cfg.update(
        temperature_mode='adiabatic_from_cmb',
        cmb_temperature=4000.0,
        num_layers=50,
        condensed_rho_scale=60.0,
        data_output_enabled=False,
        plotting_enabled=False,
    )
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        cfg['rock_solidus'],
        cfg['rock_liquidus'],
        cfg['mushy_zone_factor'],
    )
    md, spies = load_material_dictionaries(), {}
    for name in ('_surface_density', '_adiabat_to_surface'):
        real = getattr(zs, name)

        def spy(*args, _real=real, _name=name):
            spies[_name] = args
            return _real(*args)

        monkeypatch.setattr(zs, name, spy)
    r = zs.main(cfg, md, melt, os.path.join(root, 'input'))
    eos = (md, *melt, zs._interpolation_cache, *zs._eos_params(cfg))
    assert r['converged'] and r['adiabat_blend'] == 1.0 and r['pressure'][-1] == 0.0
    assert melt[0] is not melt[1] and eos[-2] != eos[-1]
    for name, i in (('_surface_density', 6), ('_adiabat_to_surface', 4)):
        got = spies[name][i]
        assert len(got) == len(eos) and all(x is y for x, y in zip(got[:4], eos[:4])), name
        assert list(got[4:]) == list(eos[4:]), name
