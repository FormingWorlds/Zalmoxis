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
def test_a_padded_node_of_a_cmb_anchored_adiabat_takes_the_top_of_the_adiabat():
    """With the adiabat integrated from the CMB, the configured surface_temperature is not the
    temperature at the surface: the padded node takes the adiabat at the target pressure, close to
    the node below it, and its EOS density there."""
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
    r = zs.main(cfg, load_material_dictionaries(), melt, os.path.join(root, 'input'))
    P, T, rho = (np.asarray(r[k], float) for k in ('pressure', 'temperature', 'density'))
    assert r['converged'] and P[-1] == 0.0 and P[-2] > 0.0
    assert abs(T[-1] - T[-2]) < 0.03 * T[-2]
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
