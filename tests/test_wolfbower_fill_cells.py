"""Wolf & Bower (2018) mantle tables: densities never come from their 1e-15 fill cells.

Module under test: ``zalmoxis.eos.seager.get_tabulated_eos`` on the shipped
``EOS_WolfBower2018_1TPa`` tables. Contract: a query outside the valid cells is
clamped to the band edge, so every density is finite, positive and between valid
neighbours. See docs/How-to/build_tests.md for the test tiers.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from zalmoxis.eos.seager import get_tabulated_eos
from zalmoxis.eos_properties import EOS_REGISTRY

pytestmark = [pytest.mark.integration, pytest.mark.timeout(900)]

WB = EOS_REGISTRY['WolfBower2018:MgSiO3']


@pytest.fixture(scope='module')
def cache():
    """One shared interpolation cache; skip when the tables are not installed."""
    for phase in ('melted_mantle', 'solid_mantle'):
        if not os.path.isfile(WB[phase]['eos_file']):
            pytest.skip(f'Wolf & Bower table missing: {WB[phase]["eos_file"]}')
    return {}


def test_melt_above_surface_band_lies_between_valid_neighbours(cache):
    """At 1 bar the melt band ends at 3815 K; 3830 K and 4000 K take the edge value.

    The plain bilinear value at 3830 K mixes a fill corner in (about 228 kg m-3).
    """
    edge = get_tabulated_eos(1e5, WB, 'melted_mantle', 3815.0, cache)
    inner = get_tabulated_eos(1e5, WB, 'melted_mantle', 3800.0, cache)
    for t in (3830.0, 4000.0, 9000.0):
        rho = get_tabulated_eos(1e5, WB, 'melted_mantle', t, cache)
        assert 2000.0 < rho <= edge
        assert rho == pytest.approx(edge, rel=1e-3)
    assert inner > edge  # the melt density still falls with T inside the band


def test_density_does_not_fall_with_pressure_beyond_the_band(cache):
    """At fixed temperature, inside and beyond the valid bands, the density of both tables
    never falls with pressure over the whole table range."""
    pressures = np.geomspace(1e5, 1e12, 3000)
    for phase in ('melted_mantle', 'solid_mantle'):
        for t in (300.0, 4500.0, 9000.0, 16000.0):
            rho = np.array([get_tabulated_eos(p, WB, phase, t, cache) for p in pressures])
            assert np.min(np.diff(rho)) >= 0.0, (phase, t)


def test_both_tables_finite_and_positive_over_sweep(cache):
    """A P-T sweep of both tables, including cold corners below the band, stays physical."""
    rng = np.random.default_rng(42)
    p = 10.0 ** rng.uniform(5.0, 12.0, 400)
    t = rng.uniform(100.0, 16000.0, 400)
    for phase, lo in (('melted_mantle', 2000.0), ('solid_mantle', 3000.0)):
        vals = np.array([get_tabulated_eos(pi, WB, phase, ti, cache) for pi, ti in zip(p, t)])
        assert np.all(np.isfinite(vals)), phase
        assert vals.min() > lo, (phase, vals.min())
        assert vals.max() < 1.2e4, (phase, vals.max())


@pytest.mark.wolfbower
def test_wolf_bower_mantle_on_the_jax_path_falls_back_to_numpy(tmp_path, caplog):
    """use_jax with a Wolf & Bower mantle runs on the numpy path and converges: the JAX
    wrapper refuses the non-PALEOS 2-phase mantle with ValueError, which the caller catches."""
    import logging
    import re

    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import (
        load_material_dictionaries,
        load_solidus_liquidus_functions,
        load_zalmoxis_config,
    )
    from zalmoxis.solver import main

    for phase in ('melted_mantle', 'solid_mantle'):
        if not os.path.isfile(WB[phase]['eos_file']):
            pytest.skip(f'Wolf & Bower table missing: {WB[phase]["eos_file"]}')
    root = get_zalmoxis_root()
    text = open(os.path.join(root, 'input', 'default.toml')).read()
    for key, value in (
        ('mantle', '"WolfBower2018:MgSiO3"'),
        ('data_enabled', 'false'),
        ('num_layers', '60'),
    ):
        text = re.sub(rf'(?m)^{key}\s*=.*$', f'{key} = {value}', text)
    cfg_path = tmp_path / 'wb.toml'
    cfg_path.write_text(text)
    cfg = load_zalmoxis_config(str(cfg_path))
    cfg.update(use_jax=True, wall_timeout=3600.0)  # the iteration limits end the solve
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        cfg.get('rock_solidus', 'Stixrude14-solidus'),
        cfg.get('rock_liquidus', 'Stixrude14-liquidus'),
        cfg.get('mushy_zone_factor', 1.0),
    )
    with caplog.at_level(logging.WARNING, logger='zalmoxis'):
        res = main(
            cfg,
            material_dictionaries=load_material_dictionaries(),
            melting_curves_functions=melt,
            input_dir=os.path.join(root, 'input'),
        )
    fell_back = [
        r.getMessage() for r in caplog.records if 'fell back to numpy' in r.getMessage()
    ]
    assert fell_back and 'PALEOS-2phase' in fell_back[0]
    assert not [r for r in caplog.records if 'density_grid' in r.getMessage()]
    assert res['converged'] and res['converged_pressure'] and res['converged_density']
    inside = np.asarray(res['pressure']) > 0  # the surface padding node has zero density
    assert np.all(np.isfinite(res['density'])) and res['density'][inside].min() > 1000.0
