"""Arrays and scalars of a default-configuration solve, for the default-invariance test.

``python tools/default_solve_reference.py <tree> <out.npz>`` solves ``input/default.toml`` of the
Zalmoxis tree ``<tree>`` (its ``src`` first on the path) and writes every numeric entry of the
result, apart from the wall time, to a compressed npz. The references in ``tests/data`` come
from the main commit the test names, one per platform, since the solve is bit-identical only on
one platform and library build.
"""

from __future__ import annotations

import os
import sys

import numpy as np

SKIP = ('core', 'icb_converged', 'layer_eos_config', 'newton_history', 'total_time')


def solve_arrays():
    """Numeric entries of a default 1 M_E solve, by key."""
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import (
        load_material_dictionaries,
        load_solidus_liquidus_functions,
        load_zalmoxis_config,
    )
    from zalmoxis.mixing import parse_all_layer_mixtures
    from zalmoxis.solver import main

    root = get_zalmoxis_root()
    cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        cfg['rock_solidus'],
        cfg['rock_liquidus'],
        cfg['mushy_zone_factor'],
    )
    mixes = parse_all_layer_mixtures(cfg['layer_eos_config'])
    res = main(cfg, load_material_dictionaries(), melt, os.path.join(root, 'input'), mixes)
    out = {}
    for k, v in res.items():
        a = np.asarray(np.nan if v is None else v)
        if k not in SKIP and a.dtype.kind in 'fiub':
            out[k] = a
    return out


if __name__ == '__main__':
    sys.path.insert(0, os.path.join(sys.argv[1], 'src'))
    np.savez_compressed(sys.argv[2], **solve_arrays())
