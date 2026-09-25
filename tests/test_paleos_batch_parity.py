"""Batch and scalar PALEOS density lookups agree on the shipped tables."""

from __future__ import annotations

import os

import numpy as np
import pytest

from zalmoxis.config import load_material_dictionaries, load_solidus_liquidus_functions
from zalmoxis.eos import calculate_density, calculate_density_batch

pytestmark = pytest.mark.smoke


def _eos_files(entry):
    """All ``eos_file`` paths of a registry entry, including its phase sub-tables."""
    files = [entry['eos_file']] if 'eos_file' in entry else []
    return files + [f for v in entry.values() if isinstance(v, dict) for f in _eos_files(v)]


@pytest.mark.parametrize('mzf', [0.8, 1.0])
@pytest.mark.parametrize(
    'eos', ['PALEOS:iron', 'PALEOS:MgSiO3', 'PALEOS:H2O', 'PALEOS-2phase:MgSiO3']
)
def test_batch_matches_scalar_on_shipped_tables(eos, mzf):
    """Random (P, T) from 1e4 to 3e13 Pa and 300 to 5e4 K, inside and outside the mushy
    zone: the two paths differ only by the rounding of their bilinear kernels."""
    mats = load_material_dictionaries()
    if not all(os.path.isfile(f) for f in _eos_files(mats[eos])):
        pytest.skip('PALEOS data files not found')
    curves = load_solidus_liquidus_functions(
        {'mantle': eos}, liquidus_id='PALEOS-liquidus', mushy_zone_factor=mzf
    )
    sol, liq = curves if curves else (None, None)
    rng = np.random.default_rng(1)
    ps, ts = 10.0 ** rng.uniform(4.0, 13.5, 3000), rng.uniform(300.0, 5e4, 3000)
    cache = {}
    batch = calculate_density_batch(ps, ts, mats, eos, sol, liq, cache, mzf)
    scalar = np.array(
        [calculate_density(p, mats, eos, t, sol, liq, cache, mzf) for p, t in zip(ps, ts)],
        dtype=float,
    )
    assert np.isfinite(scalar).mean() > 0.99
    np.testing.assert_allclose(batch, scalar, rtol=1e-7)
