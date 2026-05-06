"""Shared helpers for PALEOS-2phase:MgSiO3 integration tests.

Factored out so the high-mass tests can live in a separate file from the
light tests; xdist's ``--dist loadfile`` then puts them on different
workers and the heavy adiabatic mass scan no longer dominates wall time.

``_run_paleos`` is ``lru_cache``-decorated. With ``--dist loadfile`` the
cache is per-worker, so it dedupes calls within a single test file.
"""

from __future__ import annotations

import os
from functools import lru_cache


def _paleos_data_available():
    """Check whether the PALEOS-2phase MgSiO3 tables exist on disk."""
    root = os.environ.get('ZALMOXIS_ROOT', '')
    solid = os.path.join(
        root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
    )
    liquid = os.path.join(
        root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_liquid.dat'
    )
    return os.path.isfile(solid) and os.path.isfile(liquid)


@lru_cache(maxsize=32)
def _run_paleos(mass_earth, temperature_mode='linear'):
    """Run Zalmoxis with PALEOS-2phase:MgSiO3 mantle for one (mass, T-mode).

    Cached by ``(mass_earth, temperature_mode)`` so multiple tests sharing
    the same configuration reuse one solver run. Tests must only read
    from the returned dict, never mutate it.

    Parameters
    ----------
    mass_earth : float
        Planet mass in Earth masses.
    temperature_mode : str
        Either ``'linear'`` or ``'adiabatic'``.

    Returns
    -------
    dict
        Model results dictionary from ``zalmoxis.solver.main``.
    """
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import (
        load_material_dictionaries,
        load_solidus_liquidus_functions,
        load_zalmoxis_config,
    )
    from zalmoxis.constants import earth_mass
    from zalmoxis.solver import main

    root = get_zalmoxis_root()
    default_config_path = os.path.join(root, 'input', 'default.toml')
    config_params = load_zalmoxis_config(default_config_path)

    config_params['planet_mass'] = mass_earth * earth_mass
    config_params['layer_eos_config'] = {
        'core': 'Seager2007:iron',
        'mantle': 'PALEOS-2phase:MgSiO3',
    }
    config_params['temperature_mode'] = temperature_mode
    config_params['data_output_enabled'] = False
    config_params['plotting_enabled'] = False

    layer_eos_config = config_params['layer_eos_config']

    return main(
        config_params,
        material_dictionaries=load_material_dictionaries(),
        melting_curves_functions=load_solidus_liquidus_functions(
            layer_eos_config,
            config_params.get('rock_solidus', 'Stixrude14-solidus'),
            config_params.get('rock_liquidus', 'Stixrude14-liquidus'),
        ),
        input_dir=os.path.join(root, 'input'),
    )
