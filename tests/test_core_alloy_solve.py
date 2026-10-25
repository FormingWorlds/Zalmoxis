"""Full structure solve with a light-element core (src/zalmoxis/eos/core_alloy.py through main).

Contract: a 1 M_earth default-config planet whose core holds 1 wt% H solves on the numpy
path (the JAX path refuses a multi-component core), passes the accepted-structure checks of
``main`` and gives a core lighter than pure iron at every core node.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]


@pytest.mark.physics_invariant
def test_one_wt_percent_hydrogen_core_solves(caplog, monkeypatch):
    """The core with 1 wt% H, which samples 4-30 GPa below 8000 K in its first trials, solves."""
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
    cfg['use_jax'] = True
    cfg['layer_eos_config'] = dict(
        cfg['layer_eos_config'], core='PALEOS:iron:0.99+CoreAlloy:H:0.01'
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
    assert any('single-component core' in m for m in caplog.messages)
    m, p, t, rho = (
        np.asarray(res[k]) for k in ('mass_enclosed', 'pressure', 'temperature', 'density')
    )
    core = m < res['cmb_mass']
    iron = calculate_density_batch(p[core], t[core], md, 'PALEOS:iron', None, None, {}, 1.0)
    ratio = rho[core] / iron
    # 1 wt% H as the carrier with no core O: 0.869-0.908 of iron on the pure-iron profile
    assert np.all((0.85 < ratio) & (ratio < 0.92))
    # the linear H law (solid beta 8.7 per unit mass fraction) gives 0.913 at every node
    assert ratio.mean() < 0.905 and ratio.max() < 0.91
    assert np.all(np.isfinite(rho)) and np.all(rho[core] > 0)
    # a lighter core gives a larger planet than pure iron (6.5266e6 m on the same config)
    assert np.asarray(res['radii'])[-1] > 6.5266e6
    assert ca.LUO_OUT_OF_RANGE['count'] > 0
