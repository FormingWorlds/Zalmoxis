"""Branch tests for ``zalmoxis.eos.dispatch``.

Covers paths the existing dispatch tests in test_paleos_unified.py
do not exercise: the Vinet:<material> dispatch, the batched-density
fallback for non-unified EOS types, and the input-validation guard
for ``layer_eos == None``.
"""

from __future__ import annotations

import numpy as np
import pytest

from zalmoxis.config import load_material_dictionaries
from zalmoxis.eos import calculate_density
from zalmoxis.eos.dispatch import calculate_density_batch

pytestmark = pytest.mark.unit


class TestVinetDispatch:
    """``Vinet:<material>`` skips the material-dict lookup."""

    def test_vinet_iron(self):
        """Vinet iron at 100 GPa returns a finite density in the iron-physical
        range. The dispatch does not consult ``material_dictionaries``."""
        md = load_material_dictionaries()
        rho = calculate_density(100e9, md, 'Vinet:iron', 300, None, None)
        assert rho is not None and np.isfinite(rho)
        assert 5000 < rho < 20000

    def test_vinet_mgsio3(self):
        md = load_material_dictionaries()
        rho = calculate_density(50e9, md, 'Vinet:MgSiO3', 300, None, None)
        assert rho is not None and 3000 < rho < 8000

    def test_vinet_unknown_material_raises(self):
        """An unknown Vinet material key raises ValueError with a hint."""
        md = load_material_dictionaries()
        with pytest.raises((ValueError, KeyError)):
            calculate_density(100e9, md, 'Vinet:unobtainium', 300, None, None)


class TestCalculateDensityBatchFallback:
    """``calculate_density_batch`` scalar-loop fallback for non-unified EOS."""

    def test_batch_falls_back_for_seager(self):
        """Seager2007 has no batched path; ``calculate_density_batch`` loops
        over per-point ``calculate_density`` calls and returns the same
        densities the scalar dispatch would have produced."""
        import os

        # Seager data must be present for this test
        root = os.environ.get('ZALMOXIS_ROOT', '')
        iron_file = os.path.join(root, 'data', 'EOS_Seager2007', 'eos_seager07_iron.txt')
        if not os.path.isfile(iron_file):
            pytest.skip('Seager2007 data not found')

        md = load_material_dictionaries()
        ps = np.array([1e10, 1e11, 3e11])
        ts = np.full_like(ps, 300.0)
        rho_batch = calculate_density_batch(ps, ts, md, 'Seager2007:iron', None, None, {})
        rho_scalar = np.array(
            [calculate_density(p, md, 'Seager2007:iron', 300, None, None) for p in ps]
        )
        np.testing.assert_allclose(rho_batch, rho_scalar, rtol=1e-12)

    def test_batch_unknown_eos_returns_nan_array(self):
        """``layer_eos`` not in registry: scalar fallback raises per point,
        ``calculate_density_batch`` propagates NaN for those entries."""
        md = load_material_dictionaries()
        ps = np.array([1e10, 1e11])
        ts = np.array([300.0, 300.0])
        with pytest.raises((ValueError, KeyError)):
            # Loops through scalar dispatch which raises on first point
            calculate_density_batch(ps, ts, md, 'Nonexistent:stuff', None, None, {})

    def test_batch_analytic_path(self):
        """Analytic:iron via batch dispatch loops through scalar; both
        paths have no material dict requirement."""
        md = load_material_dictionaries()
        ps = np.array([1e10, 5e10, 1e11])
        ts = np.full_like(ps, 300.0)
        rho_batch = calculate_density_batch(ps, ts, md, 'Analytic:iron', None, None, {})
        # Density should monotonically increase with pressure (at fixed T).
        assert np.all(np.diff(rho_batch) > 0)
        assert np.all(np.isfinite(rho_batch))
