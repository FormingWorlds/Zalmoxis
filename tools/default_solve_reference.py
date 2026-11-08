"""Arrays and scalars of a default-configuration solve, for the default-invariance test.

``python tools/default_solve_reference.py <tree> <out.npz>`` solves ``input/default.toml`` of the
Zalmoxis tree ``<tree>`` (its ``src`` first on the path) and writes every numeric entry of the
result, apart from the wall time, to a compressed npz. The references in ``tests/data`` come
from the one-layer main commit, one per platform and CPU model, since the solve is bit-identical
only on one CPU and library build.
"""

from __future__ import annotations

import os
import platform
import re
import sys
from pathlib import Path

import numpy as np

SKIP = ('core', 'icb_converged', 'layer_eos_config', 'newton_history', 'total_time')


def cpu_key(system=sys.platform, machine=platform.machine(), cpuinfo=None):
    """Platform and machine; on Linux also the CPU model, e.g. linux-x86_64-AMD-EPYC-7763."""
    key = f'{system}-{machine}'
    if system != 'linux':
        return key
    if cpuinfo is None:
        cpuinfo = Path('/proc/cpuinfo').read_text()
    names = re.findall(r'^model name\s*:\s*(.+)$', cpuinfo, flags=re.M) or ['unknown']
    model = re.sub(r'\s+\d+-Core Processor\s*$', '', names[0].strip())
    return '-'.join([key, *re.findall(r'[A-Za-z0-9]+', model)])


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
    # a cut breaks bit-identity; about 1.8 x the estimated 508 s under coverage on an EPYC 7763
    # runner (205 s alone without coverage, coverage 2.48 x slower on macOS arm64)
    cfg['wall_timeout'] = 900.0
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
