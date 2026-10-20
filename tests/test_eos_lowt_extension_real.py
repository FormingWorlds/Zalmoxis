"""The low-temperature PALEOS solid extension on the real PALEOS tables."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from zalmoxis import eos_export, get_zalmoxis_root
from zalmoxis.melting_curves import derive_solidus_from_liquidus, paleos_liquidus

pytestmark = pytest.mark.slow

_PHASES = ('solid', 'solid_lowT', 'liquid')


def test_real_paleos_table_lowt_extension_numbers():
    """The real PALEOS tables with the extension give the documented S_min, n_add, coldest T and
    conductive gradient bias of the two lowest entropy intervals."""
    names = [f'EOS_PALEOS_MgSiO3/paleos_mgsio3_tables_pt_proteus_{n}.dat' for n in _PHASES]
    names.append('EOS_PALEOS_MgSiO3_unified/paleos_mgsio3_eos_table_pt.dat')
    roots = [Path(get_zalmoxis_root()) / 'data']
    if os.environ.get('FWL_DATA'):
        roots.append(Path(os.environ['FWL_DATA']) / 'zalmoxis_eos')
    base = next((r for r in roots if all((r / n).is_file() for n in names)), None)
    if base is None:
        pytest.skip(f'Real PALEOS tables with the extension missing under {roots}')
    solid, ext, liquid, uni = (base / n for n in names)

    kw = dict(
        eos_file=str(uni),
        solidus_func=derive_solidus_from_liquidus(paleos_liquidus, 0.8),
        liquidus_func=paleos_liquidus,
        P_range=(1.0e5, 3.5e11),
        n_P=1350,
        n_S=280,
        solid_eos_file=str(solid),
        liquid_eos_file=str(liquid),
    )

    base = eos_export.generate_spider_eos_tables(**kw)
    ext_out = eos_export.generate_spider_eos_tables(**kw, solid_lowt_file=str(ext))

    n_add = ext_out['S_solid'].size - base['S_solid'].size
    s_min = float(ext_out['S_solid'][0])
    coldest_t = float(ext_out['solid']['temperature'][ext_out['valid']['solid']].min())

    assert n_add == 1
    assert abs(s_min - 25.99) < 0.01
    assert abs(coldest_t - 67.08) < 0.01

    # Bias of (T/cp) against the T(S) slope of an interval, in %, at its nodes (linear in S)
    S, T, cp = ext_out['S_solid'], ext_out['solid']['temperature'], ext_out['solid']['cp']
    valid, P = ext_out['valid']['solid'], ext_out['P_Pa']

    def bias(k, cols):
        slope = (T[k + 1, cols] - T[k, cols]) / (S[k + 1] - S[k])
        ends = np.stack([T[j, cols] / cp[j, cols] for j in (k, k + 1)]) / slope - 1.0
        return 100 * ends.min(), 100 * ends.max()

    both, nxt = valid[0] & valid[1], valid[1] & valid[2]
    assert P[both].max() < 6.75e9 < 7.0e9 < P[~valid[0]].min()
    assert bias(0, both) == pytest.approx((-16.01, 39.71), abs=0.05)
    assert bias(1, nxt & both) == pytest.approx((-3.43, 5.44), abs=0.05)
    assert bias(1, nxt & ~valid[0]) == pytest.approx((-8.02, 13.01), abs=0.05)
