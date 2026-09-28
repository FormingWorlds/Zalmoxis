"""Parity of the PALEOS table loader with the genfromtxt reader on the shipped tables.

Each 150 ppd table takes a few seconds to parse, so the test is in the integration tier
and needs the tables under ``FWL_DATA`` or the data directory.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from tests.test_eos_export import _reference_arrays
from zalmoxis import eos_export

pytestmark = [
    pytest.mark.integration,
    pytest.mark.reference_pinned,
    pytest.mark.filterwarnings('error::UserWarning'),
]


@pytest.fixture(autouse=True)
def _fresh_table_cache():
    """Parse each table in the test itself and release it afterwards."""
    eos_export._parse_paleos_table.cache_clear()
    yield
    eos_export._parse_paleos_table.cache_clear()


def _real_paleos_tables(highres=False):
    """Paths of the shipped unified, solid and liquid tables (or the highres pair) found locally.

    A table present in several roots is taken from the first one only.
    """
    roots = []
    if os.environ.get('FWL_DATA'):
        roots.append(Path(os.environ['FWL_DATA']) / 'zalmoxis_eos')
    try:
        from zalmoxis import get_zalmoxis_root

        roots.append(Path(get_zalmoxis_root()) / 'data')
    except RuntimeError:
        pass
    names = [
        'EOS_PALEOS_MgSiO3/paleos_mgsio3_tables_pt_proteus_solid_highres.dat',
        'EOS_PALEOS_MgSiO3/paleos_mgsio3_tables_pt_proteus_liquid_highres.dat',
    ]
    if not highres:
        names = [
            'EOS_PALEOS_MgSiO3_unified/paleos_mgsio3_eos_table_pt.dat',
            'EOS_PALEOS_MgSiO3/paleos_mgsio3_tables_pt_proteus_solid.dat',
            'EOS_PALEOS_MgSiO3/paleos_mgsio3_tables_pt_proteus_liquid.dat',
        ]
    found = {}
    for root in roots:
        for name in names:
            if name not in found and (root / name).is_file():
                found[name] = root / name
    return list(found.values())


def _assert_table_matches_genfromtxt(path):
    """Every grid cell, phase label and empty cell of the loader equals the genfromtxt reading."""
    numeric, phase = _reference_arrays(path)
    out = eos_export.load_paleos_all_properties(path)
    keep = numeric[:, 0] > 0
    log_p = np.log10(numeric[keep, 0])
    log_t = np.log10(numeric[keep, 1])
    np.testing.assert_array_equal(out['unique_log_p'], np.unique(log_p))
    np.testing.assert_array_equal(out['unique_log_t'], np.unique(log_t))
    ip = np.searchsorted(out['unique_log_p'], log_p)
    it = np.searchsorted(out['unique_log_t'], log_t)
    names = ['rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad']
    for name, col in zip(names, range(2, 9)):
        np.testing.assert_array_equal(out[name][ip, it], numeric[keep, col])
    assert list(out['phase'][ip, it]) == [p.strip() for p in phase[keep]]
    hit = np.zeros(out['rho'].shape, dtype=bool)
    hit[ip, it] = True
    for name in names:
        assert np.isnan(out[name][~hit]).all()
    assert (out['phase'][~hit] == '').all()


@pytest.mark.parametrize('path', _real_paleos_tables(), ids=lambda p: p.name)
def test_shipped_tables_are_identical_to_the_genfromtxt_reader(path):
    """On each real table the loader returns the arrays np.genfromtxt reads from it."""
    _assert_table_matches_genfromtxt(path)
