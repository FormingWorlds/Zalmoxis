"""Inner-core mass of the liquidus rule against the surface temperature, walked in one direction.

``python tools/core_icb_scan.py <out> down|up`` solves the default 1 M_E planet with a two-layer
core (13.85 wt% S outer core on the liquid iron side, a pure solid iron inner core,
``core_liquidus = "Anzellini13-iron"``) on an adiabat from 2300 to 2700 K at the surface in 25 K
steps. Each solve starts from the density, radii and centre temperature of the step before, so
``down`` walks from a liquid and ``up`` from a solid core. The side tables are made on a grid of
40 points per decade (1e9 to 3e12 Pa, 1000 to 2e4 K) in ``<out>/cache40``. Each step prints one
JSON row; ``<out>/scan_<direction>.json`` holds all rows. This makes the start-dependence table of
docs/Explanations/mixing.md.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

LIQ, SOL = 'PALEOS-API-2phase:iron-liquid', 'PALEOS-API-2phase:iron-solid'


def scan(out, direction):
    """Solve each surface temperature of the walk; return the rows."""
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import (
        load_material_dictionaries,
        load_solidus_liquidus_functions,
        load_zalmoxis_config,
        validate_config,
    )
    from zalmoxis.eos import paleos_api, paleos_api_cache
    from zalmoxis.mixing import parse_all_layer_mixtures
    from zalmoxis.solver import main

    md = load_material_dictionaries()
    grid = paleos_api.make_grid_at_resolution(1e9, 3e12, 1000.0, 2.0e4, 40)
    for name in (LIQ, SOL):
        md[name] = dict(copy.deepcopy(md[name]), grid_spec=grid)
    paleos_api_cache._cache_root = lambda: Path(out, 'cache40')
    paleos_api_cache.resolve_paleos_api_2phase_iron(grid, n_workers=4)
    for name in (LIQ, SOL):
        paleos_api_cache.resolve_registry_entry(md[name])
    root = get_zalmoxis_root()
    temps = np.arange(2300.0, 2701.0, 25.0)
    prev, rows = None, []
    for ts in temps[::-1] if direction == 'down' else temps:
        cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
        cfg['layer_eos_config'].update(core=f'{LIQ}:0.8615+CoreAlloy:S:0.1385', inner_core=SOL)
        cfg.update(core_liquidus='Anzellini13-iron', surface_temperature=float(ts))
        seed = {}
        if prev is not None:
            cfg['center_temperature'] = float(prev['temperature'][0])
            seed = dict(initial_density=prev['density'], initial_radii=prev['radii'])
        validate_config(cfg)
        melt = load_solidus_liquidus_functions(
            cfg['layer_eos_config'], cfg['rock_solidus'], cfg['rock_liquidus'], 1.0
        )
        mixes = parse_all_layer_mixtures(cfg['layer_eos_config'])
        res = main(cfg, md, melt, os.path.join(root, 'input'), layer_mixtures=mixes, **seed)
        row = dict(
            surface_temperature=float(ts),
            converged=bool(res['converged']),
            icb_converged=bool(res['icb_converged']),
            inner_core_mass_fraction=float(res['core']['M_ic'] / res['cmb_mass']),
            centre_temperature=float(res['core']['T'][0]),
        )
        print(json.dumps(row), flush=True)
        rows.append(row)
        prev = res
    return rows


if __name__ == '__main__':
    out, direction = sys.argv[1], sys.argv[2]
    if direction not in ('down', 'up'):
        sys.exit('direction must be down or up')
    os.makedirs(out, exist_ok=True)
    rows = scan(out, direction)
    Path(out, f'scan_{direction}.json').write_text(json.dumps(rows))
