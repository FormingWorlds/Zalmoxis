"""Shared helpers for PALEOS-2phase:MgSiO3 integration tests.

Factored out so the high-mass tests can live in a separate file from the
light tests; xdist's ``--dist loadfile`` then puts them on different
workers and the heavy adiabatic mass scan no longer dominates wall time.

``_run_paleos`` is ``lru_cache``-decorated. With ``--dist loadfile`` the
cache is per-worker, so it dedupes calls within a single test file.
"""

from __future__ import annotations

import logging
import math
import os
from functools import lru_cache

import pytest


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


# A wall clock the solve cannot reach on the CI runner, so the tests read the full adiabat.
WALL_TIMEOUT = 2400.0


def _assert_full_adiabat(results, mass):
    """Assert that the solve accepted on the strict test with the full adiabat."""
    got = (results['accepted_on'], results['adiabat_blend'])
    assert got == ('strict', pytest.approx(1.0)), f'{mass} M_earth: accept label, blend {got}'


class _InnerBails(logging.Filter):
    """Records the best density diff [-] of each inner-loop bail (inf for a wall-clock exit or a
    bail record without its diff) and counts converged inner loops. Passes on only the records
    at or above ``level``, the effective level of the logger before the solve."""

    def __init__(self, level):
        super().__init__()
        self.level, self.diffs, self.converged = level, [], 0

    def filter(self, record):
        msg, args = str(record.msg), record.args or ()
        if msg.startswith(('Inner loop:', 'Maximum inner iterations')):
            self.diffs.append(float(args[-2]) if len(args) >= 2 else math.inf)
        elif msg.startswith('Wall-clock timeout in inner loop'):
            self.diffs.append(math.inf)
        elif msg.startswith('Inner loop converged'):
            self.converged += 1
        return record.levelno >= self.level


@lru_cache(maxsize=32)
def _run_paleos(mass_earth, temperature_mode='linear', wall_timeout=1000.0):
    """Run Zalmoxis with PALEOS-2phase:MgSiO3 mantle for one (mass, T-mode).

    Cached by ``(mass_earth, temperature_mode, wall_timeout)`` so multiple tests sharing
    the same configuration reuse one solver run. Tests must only read
    from the returned dict, never mutate it.

    Parameters
    ----------
    mass_earth : float
        Planet mass in Earth masses.
    temperature_mode : str
        Either ``'linear'`` or ``'adiabatic'``.
    wall_timeout : float
        Solver wall clock in s; the default is 2x the slowest CI solve of the linear
        1 M_earth case on the slowest runner CPU.

    Returns
    -------
    dict
        Model results dictionary from ``zalmoxis.solver.main``, plus ``inner_bail_diffs`` (the
        best density diff [-] of each inner loop that ended on a bail, inf for a wall-clock exit)
        and ``inner_converged`` (the number of inner loops that converged).
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
    config_params['wall_timeout'] = wall_timeout
    config_params['data_output_enabled'] = False
    config_params['plotting_enabled'] = False

    layer_eos_config = config_params['layer_eos_config']

    log = logging.getLogger('fwl.zalmoxis.solver')
    bails = _InnerBails(log.getEffectiveLevel())
    log.addFilter(bails)
    level = log.level
    log.setLevel(logging.DEBUG)
    try:
        results = main(
            config_params,
            material_dictionaries=load_material_dictionaries(),
            melting_curves_functions=load_solidus_liquidus_functions(
                layer_eos_config,
                config_params.get('rock_solidus', 'Stixrude14-solidus'),
                config_params.get('rock_liquidus', 'Stixrude14-liquidus'),
                config_params.get('mushy_zone_factor', 1.0),
            ),
            input_dir=os.path.join(root, 'input'),
        )
    finally:
        log.removeFilter(bails)
        log.setLevel(level)
    return results | {'inner_bail_diffs': bails.diffs, 'inner_converged': bails.converged}
