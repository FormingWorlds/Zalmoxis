"""Full structure solve with a light-element core (src/zalmoxis/eos/core_alloy.py through main).

Contract: a 1 M_earth default-config planet whose core holds 1 wt% H solves on the numpy
path (the JAX path refuses a core with light elements), passes the accepted-structure checks
of ``main`` and gives a core lighter than pure iron at every core node. A core whose light
elements are all 0 solves on JAX exactly as pure iron.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(3300)]


@pytest.mark.physics_invariant
def test_one_wt_percent_hydrogen_core_solves(caplog, monkeypatch):
    """The core with 1 wt% H as the Luo carrier (CoreAlloy:H-Luo), which samples 4-30 GPa below
    8000 K in its first trials, solves."""
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import (
        load_material_dictionaries,
        load_solidus_liquidus_functions,
        load_zalmoxis_config,
    )
    from zalmoxis.eos import calculate_density_batch
    from zalmoxis.eos import core_alloy as ca
    from zalmoxis.mixing import parse_all_layer_mixtures
    from zalmoxis.solver import main

    root = get_zalmoxis_root()
    md = load_material_dictionaries()
    if not os.path.isfile(md['PALEOS:iron'].get('eos_file', '')):
        pytest.skip('PALEOS iron table not linked under data/')
    cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    # The full ramp on a busy 4-vCPU CI runner
    cfg['use_jax'], cfg['wall_timeout'] = True, 2300.0
    cfg['layer_eos_config'] = dict(
        cfg['layer_eos_config'], core='PALEOS:iron:0.99+CoreAlloy:H-Luo:0.01'
    )
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        cfg['rock_solidus'],
        cfg['rock_liquidus'],
        cfg['mushy_zone_factor'],
    )
    mixes = parse_all_layer_mixtures(cfg['layer_eos_config'])
    monkeypatch.setitem(ca.LUO_OUT_OF_RANGE, 'count', 0)
    with caplog.at_level('WARNING'):
        res = main(cfg, md, melt, os.path.join(root, 'input'), layer_mixtures=mixes)

    assert res['converged']
    assert any('light elements' in m for m in caplog.messages)
    m, p, t, rho = (
        np.asarray(res[k]) for k in ('mass_enclosed', 'pressure', 'temperature', 'density')
    )
    core = m < res['cmb_mass']
    iron = calculate_density_batch(p[core], t[core], md, 'PALEOS:iron', None, None, {}, 1.0)
    ratio = rho[core] / iron
    # 1 wt% H as the carrier with no core O: 0.852-0.886 of iron on the pure-iron profile
    assert np.all((0.83 < ratio) & (ratio < 0.90))
    # the default linear H law gives 0.9282 at every node (next test)
    assert ratio.mean() < 0.88 and ratio.max() < 0.89
    assert np.all(np.isfinite(rho)) and np.all(rho[core] > 0)
    # pure iron gives 6.5235e6-6.5260e6 m on numpy, by environment; a lighter core is larger
    assert np.asarray(res['radii'])[-1] > 6.52602e6
    assert ca.LUO_OUT_OF_RANGE['count'] > 0


def _solve_default(core, use_jax, caplog):
    """The default 1 M_earth config with the given core string, through main."""
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import (
        load_material_dictionaries,
        load_solidus_liquidus_functions,
        load_zalmoxis_config,
    )
    from zalmoxis.mixing import parse_all_layer_mixtures
    from zalmoxis.solver import main

    root = get_zalmoxis_root()
    md = load_material_dictionaries()
    if not os.path.isfile(md['PALEOS:iron'].get('eos_file', '')):
        pytest.skip('PALEOS iron table not linked under data/')
    cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    # The full ramp on a busy 4-vCPU CI runner
    cfg['use_jax'], cfg['wall_timeout'] = use_jax, 1600.0
    cfg['layer_eos_config'] = dict(cfg['layer_eos_config'], core=core)
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        cfg['rock_solidus'],
        cfg['rock_liquidus'],
        cfg['mushy_zone_factor'],
    )
    mixes = parse_all_layer_mixtures(cfg['layer_eos_config'])
    with caplog.at_level('WARNING'):
        return main(cfg, md, melt, os.path.join(root, 'input'), layer_mixtures=mixes)


@pytest.mark.physics_invariant
def test_one_wt_percent_default_hydrogen_core_follows_the_linear_law(caplog):
    """The default H law (CoreAlloy:H, beta 7/0.975 % per wt%) with 1 wt% H solves, and its
    core is 1 - 0.07/0.975 of PALEOS iron at every core node (measured 0.928205128205 on 80
    nodes); the Luo law (CoreAlloy:H-Luo) gives 0.852-0.886 there."""
    from zalmoxis.config import load_material_dictionaries
    from zalmoxis.eos import calculate_density_batch

    res = _solve_default('PALEOS:iron:0.99+CoreAlloy:H:0.01', True, caplog)
    assert res['converged']
    m, p, t, rho = (
        np.asarray(res[k]) for k in ('mass_enclosed', 'pressure', 'temperature', 'density')
    )
    core = m < res['cmb_mass']
    iron = calculate_density_batch(
        p[core], t[core], load_material_dictionaries(), 'PALEOS:iron', None, None, {}, 1.0
    )
    assert core.sum() > 10
    np.testing.assert_allclose(rho[core] / iron, 1.0 - 0.07 / 0.975, rtol=1e-12)


@pytest.mark.physics_invariant
def test_zero_alloy_core_on_jax_is_pure_iron(caplog):
    """PALEOS:iron:1.0 with H and O at 0 solves on JAX bit for bit as PALEOS:iron."""
    iron = _solve_default('PALEOS:iron', True, caplog)
    zero = _solve_default('PALEOS:iron:1.0+CoreAlloy:H:0.0+CoreAlloy:O:0.0', True, caplog)
    assert not any('fell back to numpy path' in m for m in caplog.messages)
    for key in ('radii', 'density', 'pressure', 'temperature', 'mass_enclosed'):
        assert np.array_equal(np.asarray(zero[key]), np.asarray(iron[key])), key
