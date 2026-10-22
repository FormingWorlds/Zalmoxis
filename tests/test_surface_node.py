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
