"""A solved 3-layer planet with a T-independent H2O layer on the adiabat cut at the mantle and
ice boundary.

1 M_earth, core 0.325, ice 0.1, Seager2007:H2O, PALEOS iron and MgSiO3, surface 300 K (below
the H2O critical point, as ``validate_config`` requires).
See ``docs/Validation/thermal_state.md`` (case C).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from tests._paleos_helpers import WALL_TIMEOUT

pytestmark = [
    pytest.mark.slow,
    pytest.mark.paleos_adiabat,
    pytest.mark.timeout(1800),  # a 150 s solve on macOS arm64
]


def _solve(ice_eos, num_layers):
    from tests.test_adiabat_real_profiles import _need_paleos
    from zalmoxis import get_zalmoxis_root
    from zalmoxis import solver as zs
    from zalmoxis.config import (
        load_material_dictionaries,
        load_solidus_liquidus_functions,
        load_zalmoxis_config,
        validate_config,
    )

    _need_paleos()
    inp = os.path.join(get_zalmoxis_root(), 'input')
    cfg = load_zalmoxis_config(os.path.join(inp, 'default.toml'))
    cfg.update(
        data_output_enabled=False,
        plotting_enabled=False,
        surface_temperature=300.0,
        core_mass_fraction=0.325,
        mantle_mass_fraction=0.575,
        num_layers=num_layers,
        wall_timeout=WALL_TIMEOUT,
        layer_eos_config={
            'core': 'PALEOS:iron',
            'mantle': 'PALEOS:MgSiO3',
            'ice_layer': ice_eos,
        },
    )
    validate_config(cfg)
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        'Stixrude14-solidus',
        'Stixrude14-liquidus',
        cfg['mushy_zone_factor'],
    )
    res = zs.main(cfg, load_material_dictionaries(), melt, inp)
    got = (res['converged'], res['adiabat_blend'], res['accepted_on'])
    assert got[:2] == (True, 1.0) and got[2] != 'timeout', got
    return res


@pytest.mark.physics_invariant
def test_a_t_independent_ice_layer_is_isothermal_at_the_surface_temperature():
    """Case C: the Seager2007:H2O planet (T-independent ice) at 150 nodes: every ice node holds
    the surface temperature of 300 K to round-off, so the mantle starts at 300 K at the
    boundary."""
    res = _solve('Seager2007:H2O', 150)
    m, T = np.asarray(res['mass_enclosed']), np.asarray(res['temperature'], float)
    ice = m >= res['core_mantle_mass'] * (1 - 1e-12)
    assert ice.sum() > 5 and np.max(np.abs(T[ice] - 300.0)) <= 1e-9, T[ice]
