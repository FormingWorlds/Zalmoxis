"""Stored reference of the PROTEUS grid start-up case at a mass tolerance of 1e-6.

The case is the first structure solve of the PROTEUS grid test case with core mass fraction
0.95 (0.7 Earth masses, H 1000 ppmw) and its first start-up re-solve. This script solves both
at the reference tolerance and writes ``tests/data/grid_start_up_reference.json`` with the
commit, the md5 of each EOS table and the sha256 of the Zalmoxis modules the solves load. The
grid start-up test compares its solves with the stored structures and stops when a table md5
differs. The slow reference test solves both again and fails when they move or when the
source hash differs; then run this script again and commit the file.

Usage::

    python -m tools.validation.grid_start_up_reference [--out <file>]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'tests' / 'data' / 'grid_start_up_reference.json'
KEYS = ('radii', 'pressure', 'temperature', 'mass_enclosed')
REFRESH = 'python -m tools.validation.grid_start_up_reference'
GENERATED = {'_version.py'}  # written at install time with the commit in it

GRID_CMF95 = {
    'planet_mass': 4.1804e24,
    'core_mass_fraction': 0.95,
    'mantle_mass_fraction': 0.0,
    'temperature_mode': 'adiabatic_from_cmb',
    'surface_temperature': 4000.0,
    'cmb_temperature': 6000.0,
    'center_temperature': 6000.0,
    'temp_profile_file': None,
    'layer_eos_config': {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
    'mushy_zone_factor': 0.8,
    'mushy_zone_factors': {'PALEOS:MgSiO3': 0.8, 'PALEOS:iron': 0.8},
    'num_layers': 150,
    'target_surface_pressure': 101325.0,
    'tolerance_outer': 3e-3,
    'tolerance_inner': 1e-4,
    'max_iterations_outer': 100,
    'max_iterations_inner': 100,
    'use_jax': False,
    'use_anderson': True,
    'outer_solver': 'newton',
    'newton_max_iter': 30,
    'newton_tol': 1e-4,
    'relative_tolerance': 1e-9,
    'absolute_tolerance': 1e-10,
    'wall_timeout': 1e6,
}
GRID_CMF95_RESOLVE = {
    'planet_mass': 4.177602725330534e24,
    'core_mass_fraction': 0.9505316225476148,
    'target_surface_pressure': 51305541.15573382,
}
REF_TOL = {'tolerance_outer': 1e-6, 'newton_tol': 1e-6}
CASES = {'first': GRID_CMF95, 'resolve': {**GRID_CMF95, **GRID_CMF95_RESOLVE}}


def solve(cfg, **kwargs):
    """``zalmoxis.solver.main`` on one config of the case, with the melting curves and input
    folder of the PROTEUS grid; ``kwargs`` go to main (warm start, ``adiabat_seed``)."""
    import zalmoxis.solver as zs
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import load_material_dictionaries
    from zalmoxis.melting_curves import (
        derive_solidus_from_liquidus,
        get_solidus_liquidus_functions,
    )

    _, liquidus = get_solidus_liquidus_functions('Stixrude14-solidus', 'PALEOS-liquidus')
    melt = (derive_solidus_from_liquidus(liquidus, 0.8), liquidus)
    inp = os.path.join(get_zalmoxis_root(), 'input')
    return zs.main(cfg, load_material_dictionaries(), melt, inp, **kwargs)


def md5(path):
    """md5 hex digest of a file."""
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'md5').hexdigest()


def source_hash(files, root):
    """sha256 over the names (sorted) and bytes of ``files``, paths relative to ``root``."""
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(name.encode() + b'\0' + (Path(root) / name).read_bytes())
    return h.hexdigest()


def loaded_sources():
    """(package folder, sorted paths below it) of every zalmoxis module loaded now, without
    the files the install generates; a module from outside the package raises ValueError."""
    import zalmoxis

    pkg = Path(zalmoxis.__file__).resolve().parent
    files = {
        Path(m.__file__).resolve().relative_to(pkg).as_posix()
        for name, m in list(sys.modules.items())
        if name.split('.')[0] == 'zalmoxis'
        and getattr(m, '__file__', None)
        and Path(m.__file__).name not in GENERATED
    }
    return pkg, sorted(files)


def dump_compact(obj, path):
    """Write ``obj`` (a dict) as JSON with one top-level key per line and no indentation."""
    rows = (f'{json.dumps(k)}: {json.dumps(v, separators=(",", ":"))}' for k, v in obj.items())
    Path(path).write_text('{\n' + ',\n'.join(rows) + '\n}\n')


def load(path=REFERENCE, material_dictionaries=None):
    """The stored reference with each structure as arrays; raises ValueError when a stored
    EOS table is not in ``material_dictionaries``, not on disk, or has another md5."""
    if material_dictionaries is None:
        from zalmoxis.config import load_material_dictionaries

        material_dictionaries = load_material_dictionaries()
    ref = json.loads(Path(path).read_text())
    for name, table in ref['tables'].items():
        eos_file = material_dictionaries.get(name, {}).get('eos_file')
        if eos_file is None:
            raise ValueError(f'{name}: not in the material dictionaries')
        if not os.path.isfile(eos_file):
            raise ValueError(f'{name}: table {eos_file} not on disk')
        now = md5(eos_file)
        if now != table['md5']:
            raise ValueError(
                f'{name}: md5 of {eos_file} is {now}, the stored reference was solved with '
                f'{table["md5"]!r}; run {REFRESH} and commit {Path(path).name}'
            )
    for key in CASES:
        ref[key] = {k: (np.asarray(v, float) if k in KEYS else v) for k, v in ref[key].items()}
    return ref


def write(path=REFERENCE):
    """Solve both structures of the case at REF_TOL and write the reference to ``path``."""
    from zalmoxis.config import load_material_dictionaries

    blocks = {}
    for key, cfg in CASES.items():
        r = solve({**cfg, **REF_TOL})
        blends = (r['adiabat_blend'], r['adiabat_table_blend'])
        if not (r['converged'] and r['accepted_on'] == 'strict' and blends == (1.0, 1.0)):
            raise RuntimeError(
                f'{key}: the reference solve accepted {r["accepted_on"]!r} at blends {blends} '
                f'(converged {r["converged"]}); a reference must be strict on the full adiabat'
            )
        blocks[key] = {k: np.asarray(r[k], float).tolist() for k in KEYS}
        blocks[key]['cmb_mass'] = float(r['cmb_mass'])
    pkg, files = loaded_sources()
    git = ('git', '-C', str(ROOT))
    md = load_material_dictionaries()
    names = sorted({v for cfg in CASES.values() for v in cfg['layer_eos_config'].values()})
    ref = {
        'commit': subprocess.check_output((*git, 'rev-parse', 'HEAD'), text=True).strip(),
        'source_tree_clean': not subprocess.check_output(
            (*git, 'status', '--porcelain', 'src')
        ),
        'tolerances': REF_TOL,
        'platform': f'{platform.platform()} python {platform.python_version()} numpy '
        f'{np.__version__}',
        'tables': {
            n: {'file': Path(md[n]['eos_file']).name, 'md5': md5(md[n]['eos_file'])}
            for n in names
        },
        'source': {'files': files, 'sha256': source_hash(files, pkg)},
        **blocks,
    }
    dump_compact(ref, path)
    return ref


def main(argv=None):
    """Command line entry: write the reference."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', default=str(REFERENCE), help='reference file to write')
    ref = write(parser.parse_args(argv).out)
    print(ref['commit'], ref['source']['sha256'])


if __name__ == '__main__':
    main()
