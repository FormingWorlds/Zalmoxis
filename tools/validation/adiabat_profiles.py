"""Structure profiles for the adiabat integration check.

Each case is the structure (radius, pressure, enclosed mass) and the arguments of the last
``compute_adiabatic_temperature`` call of one real solve: 4 starts of ``input/default.toml``
at 2515 K and a mass tolerance of 1e-3 whose planet mass differs by k 1e-12 (k = 0 to 3), and
the 2 PROTEUS initial structure setups of ``tools.validation.proteus_ic_solvers`` (Newton).
``write`` checks that the call rebuilt from the file gives the temperature of the solve bit for
bit, so the file holds every input of the integration.

Usage::

    python -m tools.validation.adiabat_profiles write <out.npz>
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess

import numpy as np

DEFAULT_STARTS = (0, 1, 2, 3)
IC_CASES = ('all_options', 'dummy')
CASES = tuple(f'default_k{k}' for k in DEFAULT_STARTS) + tuple(f'ic_{c}' for c in IC_CASES)
SCALARS = (
    'surface_temperature',
    'cmb_mass',
    'core_mantle_mass',
    'condensed_rho_min',
    'condensed_rho_scale',
    'binodal_T_scale',
)


def _default_config(k):
    """config_params of default start ``k``: input/default.toml at 2515 K, tolerance 1e-3."""
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import load_zalmoxis_config, validate_config

    cfg = copy.deepcopy(
        load_zalmoxis_config(os.path.join(get_zalmoxis_root(), 'input', 'default.toml'))
    )
    cfg.update(surface_temperature=2515.0, tolerance_outer=1e-3)
    cfg['planet_mass'] *= 1.0 + k * 1e-12
    validate_config(cfg)
    return cfg


def _melt(kind, cfg):
    """(solidus, liquidus) of a case: 'config' for the default starts, 'grid' for PROTEUS."""
    if kind == 'grid':
        from zalmoxis.melting_curves import (
            derive_solidus_from_liquidus,
            get_solidus_liquidus_functions,
        )

        _, liquidus = get_solidus_liquidus_functions('Stixrude14-solidus', 'PALEOS-liquidus')
        return derive_solidus_from_liquidus(liquidus, 0.8), liquidus
    from zalmoxis.config import load_solidus_liquidus_functions

    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'], cfg['rock_solidus'], cfg['rock_liquidus'], 1.0
    )
    return melt or (None, None)


def _solve(case):
    """Run the solve of ``case``; return (record of its last adiabat call, its result)."""
    import zalmoxis.solver as zs
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import load_material_dictionaries

    last = {}
    native = zs.compute_adiabatic_temperature

    def hook(*args, **kwargs):
        T = native(*args, **kwargs)
        last.update(args=args, kwargs=kwargs, T=np.array(T, float))
        return T

    zs.compute_adiabatic_temperature = hook
    try:
        if case.startswith('default_k'):
            cfg, kind = _default_config(int(case[len('default_k') :])), 'config'
            res = zs.main(
                cfg,
                load_material_dictionaries(),
                _melt(kind, cfg),
                os.path.join(get_zalmoxis_root(), 'input'),
            )
        else:
            from tools.validation.proteus_ic_solvers import PROTEUS_IC, solve

            name = case[len('ic_') :]
            cfg, kind = {**PROTEUS_IC['common'], **PROTEUS_IC[name]}, 'grid'
            res = solve(name, 'newton')
    finally:
        zs.compute_adiabatic_temperature = native
    a, kw = last['args'], last['kwargs']
    rec = {'radii': a[0], 'pressure': a[1], 'mass_enclosed': a[2], 'T': last['T']}
    rec.update(dict(zip(SCALARS, (a[3], a[4], a[5], a[12], a[13], a[14]))))
    rec.update(
        anchor=kw['anchor'],
        cmb_temperature=kw['cmb_temperature'],
        mushy_zone_factors=a[11],
        layer_eos_config=cfg['layer_eos_config'],
        melt=kind,
        rock=[cfg.get('rock_solidus'), cfg.get('rock_liquidus')],
        accepted_on=res.get('accepted_on'),
        padded=bool(res.get('padded')),
    )
    return rec


def call_kwargs(rec):
    """Keyword arguments of ``compute_adiabatic_temperature`` rebuilt from a record of
    ``load``, with a new interpolation cache."""
    from zalmoxis.config import load_material_dictionaries
    from zalmoxis.mixing import parse_all_layer_mixtures

    cfg = {
        'layer_eos_config': rec['layer_eos_config'],
        'rock_solidus': rec['rock'][0],
        'rock_liquidus': rec['rock'][1],
    }
    solidus, liquidus = _melt(rec['melt'], cfg)
    kw = {k: rec[k] for k in ('radii', 'pressure', 'mass_enclosed', *SCALARS)}
    kw.update(
        layer_mixtures=parse_all_layer_mixtures(rec['layer_eos_config']),
        material_dictionaries=load_material_dictionaries(),
        interpolation_functions={},
        solidus_func=solidus,
        liquidus_func=liquidus,
        mushy_zone_factors=rec['mushy_zone_factors'],
        anchor=rec['anchor'],
        cmb_temperature=rec['cmb_temperature'],
    )
    return kw


def write(path):
    """Solve every case, check the rebuilt call against each solve, write ``path``."""
    from zalmoxis.eos.temperature import compute_adiabatic_temperature

    arrays, meta = {}, {}
    for case in CASES:
        rec = _solve(case)
        again = compute_adiabatic_temperature(**call_kwargs(rec))
        if not np.array_equal(again, rec['T']):
            raise RuntimeError(
                f'{case}: the rebuilt call does not give the temperature of the solve'
            )
        for k in ('radii', 'pressure', 'mass_enclosed', 'T'):
            arrays[f'{case}/{k}'] = np.asarray(rec.pop(k), float)
        meta[case] = {
            k: (float(v) if isinstance(v, np.floating) else v) for k, v in rec.items()
        }
        print(case, meta[case]['accepted_on'], 'padded' if meta[case]['padded'] else 'unpadded')
    commit = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, check=False
    ).stdout.strip()
    meta['_source'] = {'commit': commit}
    np.savez_compressed(path, meta=np.array(json.dumps(meta, separators=(',', ':'))), **arrays)


def load(path):
    """{case: record} of a file written by ``write``; the source commit is under '_source'."""
    with np.load(path) as f:
        meta = json.loads(str(f['meta']))
        out = {'_source': meta.pop('_source')}
        for case, rec in meta.items():
            out[case] = {
                **rec,
                **{k: f[f'{case}/{k}'] for k in ('radii', 'pressure', 'mass_enclosed', 'T')},
            }
    return out


def main(argv=None):
    """Command line: ``write <out.npz>``."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('command', choices=('write',))
    p.add_argument('out')
    args = p.parse_args(argv)
    write(args.out)


if __name__ == '__main__':
    main()
