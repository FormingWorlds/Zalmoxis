"""Stored reference of the PROTEUS grid start-up case at a mass tolerance of 1e-6.

The case is the first structure solve of the PROTEUS grid test case with core mass fraction
0.95 (0.7 Earth masses, H 1000 ppmw) and its first start-up re-solve, each on the numpy path
and on the JAX path (``use_jax``). This script solves the 4 at the reference tolerance and
writes ``tests/data/grid_start_up_reference.json`` with the commit, the md5 of each EOS
table and the sha256 of the syntax trees of the Zalmoxis modules the solves load, without
docstrings and comments, with the Python minor version that parsed them. The grid start-up
test compares its solves with the stored structures and stops when a table md5 differs. The
slow reference test solves the 4 again and fails when they move or when the source hash
differs; then run this script again and commit the file. Under another Python minor,
``--source-only`` writes the hash again without solving, when the structures were solved in
a clean tree and the code at the stored commit and now has the same trees.

Usage::

    python -m tools.validation.grid_start_up_reference [--out <file>] [--source-only]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'tests' / 'data' / 'grid_start_up_reference.json'
KEYS = ('radii', 'pressure', 'temperature', 'mass_enclosed')
REFRESH = 'python -m tools.validation.grid_start_up_reference'
PYTHON = f'{sys.version_info.major}.{sys.version_info.minor}'
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
CASES |= {f'{k}_jax': {**cfg, 'use_jax': True} for k, cfg in CASES.items()}


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


def _code(text):
    """``ast.dump`` of the module ``text`` (bytes, in the encoding the module declares)
    without its bare string statements, which include the docstrings; comments are not in
    the tree."""

    def text_only(s):
        value = s.value if isinstance(s, ast.Expr) else None
        return isinstance(value, ast.Constant) and isinstance(value.value, str)

    tree = ast.parse(text)
    for node in ast.walk(tree):
        for field in ('body', 'orelse', 'finalbody'):
            if isinstance(getattr(node, field, None), list):
                setattr(node, field, [s for s in getattr(node, field) if not text_only(s)])
    return ast.dump(tree).encode()


def _hash(texts):
    """sha256 over the names (sorted) and code trees of ``texts``, a dict name -> bytes."""
    h = hashlib.sha256()
    for name in sorted(texts):
        h.update(name.encode() + b'\0' + _code(texts[name]))
    return h.hexdigest()


def source_hash(files, root):
    """Hash of the code of ``files``, paths relative to ``root``, under the running Python."""
    return _hash({name: (Path(root) / name).read_bytes() for name in files})


def source_mismatch(stored, pkg):
    """None when the stored ``source`` block holds the hash of the modules in ``pkg`` under
    this Python minor, else the reason with the command that writes the reference again."""
    if stored.get('python') != PYTHON:
        return (
            f'the source hash was taken under Python {stored.get("python")} and this is '
            f'Python {PYTHON}: run {REFRESH} --source-only'
        )
    if source_hash(stored['files'], pkg) != stored['sha256']:
        return f'the code of the Zalmoxis source differs from the stored one: run {REFRESH}'
    return None


def restamp_source(path=REFERENCE, root=ROOT, pkg=None):
    """Write the source hash of ``path`` again under this Python and keep every other value,
    when the structures come from a clean tree and the stored modules at the stored commit
    and in ``pkg`` have the same code; raises ValueError otherwise."""
    if pkg is None:
        import zalmoxis

        pkg = Path(zalmoxis.__file__).parent
    ref = json.loads(Path(path).read_text())
    commit, files = ref['commit'], ref['source']['files']
    if not re.fullmatch('[0-9a-f]{40,64}', str(commit)):
        raise ValueError(f'the stored commit {commit!r} is not a commit sha: run {REFRESH}')
    if ref.get('source_tree_clean') is not True:
        raise ValueError(f'the structures were solved in a tree with changes: run {REFRESH}')
    here, top = Path(pkg).resolve(), Path(root).resolve()
    if not here.is_relative_to(top):
        raise ValueError(f'the imported zalmoxis {pkg} is not in {root}: install this tree')
    rel = here.relative_to(top).as_posix()
    git = ('git', '-C', str(root))
    if subprocess.run(
        (*git, 'cat-file', '-e', f'{commit}^{{commit}}'), capture_output=True
    ).returncode:
        subprocess.run((*git, 'fetch', '-q', 'origin', commit), capture_output=True)
    then = {}
    for f in files:
        try:
            then[f] = subprocess.check_output(
                (*git, 'show', f'{commit}:{rel}/{f}'), stderr=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError:
            raise ValueError(
                f'{rel}/{f} at commit {commit} is not in {root} nor its origin: run {REFRESH}'
            ) from None
    try:
        now = source_hash(files, pkg)
    except (OSError, SyntaxError) as err:
        raise ValueError(f'{err}: run {REFRESH}') from None
    if _hash(then) != now:
        raise ValueError(
            f'the code of the Zalmoxis source changed since {commit}: run {REFRESH}'
        )
    ref['source'] = {**ref['source'], 'sha256': now, 'python': PYTHON}
    dump_compact(ref, path)
    return ref


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
    """Solve the 4 structures of the case at REF_TOL and write the reference to ``path``."""
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
        'source': {'files': files, 'sha256': source_hash(files, pkg), 'python': PYTHON},
        **blocks,
    }
    dump_compact(ref, path)
    return ref


def main(argv=None):
    """Command line entry: write the reference, or with ``--source-only`` its source hash."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--out',
        default=str(REFERENCE),
        help='reference file to write, or to restamp with --source-only',
    )
    parser.add_argument('--source-only', action='store_true', help='write only the source hash')
    args = parser.parse_args(argv)
    ref = restamp_source(args.out) if args.source_only else write(args.out)
    print(ref['commit'], ref['source']['sha256'])


if __name__ == '__main__':
    main()
