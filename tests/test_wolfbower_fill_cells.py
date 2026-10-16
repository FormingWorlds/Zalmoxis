"""Wolf & Bower (2018) mantle tables: densities never come from their 1e-15 fill cells.

Module under test: ``zalmoxis.eos.seager.get_tabulated_eos`` on the shipped
``EOS_WolfBower2018_1TPa`` tables. Contract: a query outside the valid cells is
clamped to the band edge, so every density is finite, positive and between valid
neighbours. See docs/How-to/build_tests.md for the test tiers.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from zalmoxis.eos import seager
from zalmoxis.eos.seager import get_tabulated_eos
from zalmoxis.eos_properties import EOS_REGISTRY

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

WB = EOS_REGISTRY['WolfBower2018:MgSiO3']


@pytest.fixture(scope='module')
def cache():
    """One shared interpolation cache; skip when the tables are not installed."""
    for phase in ('melted_mantle', 'solid_mantle'):
        if not os.path.isfile(WB[phase]['eos_file']):
            pytest.skip(f'Wolf & Bower table missing: {WB[phase]["eos_file"]}')
    return {}


def test_melt_above_surface_band_lies_between_valid_neighbours(cache):
    """At 1 bar the melt band ends at 3815 K; 3830 K and 4000 K take the edge value.

    The plain bilinear value at 3830 K mixes a fill corner in (about 228 kg m-3).
    """
    seager._tdep_clamp_warned.discard(WB['melted_mantle']['eos_file'])
    edge = get_tabulated_eos(1e5, WB, 'melted_mantle', 3815.0, cache)
    inner = get_tabulated_eos(1e5, WB, 'melted_mantle', 3800.0, cache)
    for t in (3830.0, 4000.0, 9000.0):
        rho = get_tabulated_eos(1e5, WB, 'melted_mantle', t, cache)
        assert 2000.0 < rho <= edge
        assert rho == pytest.approx(edge, rel=1e-3)
    assert inner > edge  # the melt density still falls with T inside the band


def test_both_tables_finite_and_positive_over_sweep(cache):
    """A P-T sweep of both tables, including cold corners below the band, stays physical."""
    rng = np.random.default_rng(42)
    p = 10.0 ** rng.uniform(5.0, 12.0, 400)
    t = rng.uniform(100.0, 16000.0, 400)
    for phase, lo in (('melted_mantle', 2000.0), ('solid_mantle', 3000.0)):
        vals = np.array([get_tabulated_eos(pi, WB, phase, ti, cache) for pi, ti in zip(p, t)])
        assert np.all(np.isfinite(vals)), phase
        assert vals.min() > lo, (phase, vals.min())
        assert vals.max() < 1.2e4, (phase, vals.max())
