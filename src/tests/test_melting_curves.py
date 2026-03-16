"""Tests for the melting curve dispatcher and Monteux+2016 analytic functions.

Tests cover:
- Analytic solidus/liquidus evaluation at known pressures
- Dispatcher for all valid curve identifiers
- Invalid identifier raises ValueError
- Analytic curves are defined for all P >= 0 (no NaN)
- Tabulated curves return NaN outside their range

References:
    - docs/testing.md
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.unit
class TestMonteux16Solidus:
    """Tests for the Monteux+2016 analytic solidus."""

    def test_zero_pressure(self):
        """Solidus at P=0 should equal the zero-pressure anchor (1661.2 K)."""
        from zalmoxis.melting_curves import monteux16_solidus

        T = monteux16_solidus(0.0)
        assert T == pytest.approx(1661.2, rel=1e-4)

    def test_scalar_and_array(self):
        """Should return scalar for scalar input, array for array input."""
        from zalmoxis.melting_curves import monteux16_solidus

        T_scalar = monteux16_solidus(10e9)
        assert isinstance(T_scalar, float)

        T_array = monteux16_solidus(np.array([10e9, 50e9]))
        assert isinstance(T_array, np.ndarray)
        assert len(T_array) == 2

    def test_monotonically_increasing(self):
        """Solidus should increase with pressure."""
        from zalmoxis.melting_curves import monteux16_solidus

        P = np.linspace(0, 200e9, 100)
        T = monteux16_solidus(P)
        assert np.all(np.diff(T) > 0)

    def test_no_nan_at_any_pressure(self):
        """Analytic solidus must never return NaN."""
        from zalmoxis.melting_curves import monteux16_solidus

        P = np.logspace(0, 14, 500)  # 1 Pa to 100 TPa
        T = monteux16_solidus(P)
        assert np.all(np.isfinite(T))

    def test_continuity_at_20_GPa(self):
        """Solidus should be approximately continuous at the 20 GPa transition."""
        from zalmoxis.melting_curves import monteux16_solidus

        T_below = monteux16_solidus(19.99e9)
        T_above = monteux16_solidus(20.01e9)
        # Allow up to 50 K discontinuity (piecewise fit)
        assert abs(T_above - T_below) < 50, (
            f'Solidus discontinuity at 20 GPa: {T_below:.1f} vs {T_above:.1f} K'
        )


@pytest.mark.unit
class TestMonteux16Liquidus:
    """Tests for the Monteux+2016 analytic liquidus."""

    def test_liquidus_above_solidus(self):
        """Liquidus should be above solidus in the experimentally constrained range.

        The piecewise parameterization crosses at ~500 GPa (beyond the
        experimental data range of ~140 GPa). We test up to 400 GPa.
        """
        from zalmoxis.melting_curves import monteux16_liquidus, monteux16_solidus

        P = np.logspace(5, np.log10(400e9), 200)  # up to 400 GPa
        T_sol = monteux16_solidus(P)
        T_liq = monteux16_liquidus(P, model='A-chondritic')
        assert np.all(T_liq > T_sol), 'Liquidus must exceed solidus up to 400 GPa'

    def test_both_models_available(self):
        """Both A-chondritic and F-peridotitic models should return valid T."""
        from zalmoxis.melting_curves import monteux16_liquidus

        P = 50e9
        T_A = monteux16_liquidus(P, model='A-chondritic')
        T_F = monteux16_liquidus(P, model='F-peridotitic')
        assert np.isfinite(T_A)
        assert np.isfinite(T_F)
        # They should differ (different high-P parameterizations)
        assert T_A != T_F

    def test_invalid_model_raises(self):
        """Unknown model name should raise ValueError."""
        from zalmoxis.melting_curves import monteux16_liquidus

        with pytest.raises(ValueError, match='Unknown model'):
            monteux16_liquidus(50e9, model='nonexistent')


@pytest.mark.unit
class TestMeltingCurveDispatcher:
    """Tests for get_melting_curve_function() dispatcher."""

    def test_all_solidus_ids_valid(self):
        """All VALID_SOLIDUS identifiers should return a callable."""
        from zalmoxis.melting_curves import VALID_SOLIDUS, get_melting_curve_function

        for sid in VALID_SOLIDUS:
            func = get_melting_curve_function(sid)
            assert callable(func), f'{sid} did not return a callable'
            T = func(50e9)
            assert np.isfinite(T), f'{sid} returned non-finite T at 50 GPa'

    def test_all_liquidus_ids_valid(self):
        """All VALID_LIQUIDUS identifiers should return a callable."""
        from zalmoxis.melting_curves import VALID_LIQUIDUS, get_melting_curve_function

        for lid in VALID_LIQUIDUS:
            func = get_melting_curve_function(lid)
            assert callable(func), f'{lid} did not return a callable'
            T = func(50e9)
            assert np.isfinite(T), f'{lid} returned non-finite T at 50 GPa'

    def test_invalid_id_raises(self):
        """Unknown identifier should raise ValueError."""
        from zalmoxis.melting_curves import get_melting_curve_function

        with pytest.raises(ValueError, match='Unknown melting curve'):
            get_melting_curve_function('nonexistent')

    def test_get_solidus_liquidus_functions_returns_pair(self):
        """get_solidus_liquidus_functions should return a 2-tuple of callables."""
        from zalmoxis.melting_curves import get_solidus_liquidus_functions

        sol, liq = get_solidus_liquidus_functions()
        assert callable(sol)
        assert callable(liq)
        assert sol(50e9) < liq(50e9), 'Solidus should be below liquidus'

    def test_tabulated_returns_nan_outside_range(self):
        """Tabulated curves should return NaN at pressures outside their range."""
        import os

        from zalmoxis.melting_curves import get_melting_curve_function

        root = os.environ.get('ZALMOXIS_ROOT', '')
        tab_file = os.path.join(root, 'data', 'melting_curves_Monteux-600', 'solidus.dat')
        if not os.path.isfile(tab_file):
            pytest.skip('Monteux-600 tabulated data not found')

        func = get_melting_curve_function('Monteux600-solidus-tabulated')
        # At very high pressure beyond the table range, should return NaN
        T = func(1e15)  # 1000 TPa, well beyond table range
        assert np.isnan(T), f'Expected NaN at 1000 TPa, got {T}'

    def test_analytic_defined_beyond_table_range(self):
        """Analytic Monteux16 should return finite T even at extreme pressures."""
        from zalmoxis.melting_curves import get_melting_curve_function

        func = get_melting_curve_function('Monteux16-solidus')
        T = func(1e15)  # 1000 TPa
        assert np.isfinite(T), 'Analytic solidus returned non-finite at 1000 TPa'
        assert T > 0


@pytest.mark.unit
class TestPerCellClamping:
    """Tests for PALEOS per-cell temperature clamping and NN fallback."""

    def test_cache_has_per_cell_bounds(self):
        """PALEOS cache should contain per-pressure valid T bounds."""
        import os

        root = os.environ.get('ZALMOXIS_ROOT', '')
        liq_file = os.path.join(
            root,
            'data',
            'EOS_PALEOS_MgSiO3',
            'paleos_mgsio3_tables_pt_proteus_liquid.dat',
        )
        if not os.path.isfile(liq_file):
            pytest.skip('PALEOS data not found')

        from zalmoxis.eos_functions import load_paleos_table

        cache = load_paleos_table(liq_file)
        assert 'logt_valid_min' in cache
        assert 'logt_valid_max' in cache
        assert 'unique_log_p' in cache
        assert 'density_nn' in cache
        assert 'nabla_ad_nn' in cache

    def test_nn_fallback_returns_finite(self):
        """NN fallback should return finite density at NaN grid cells."""
        import os

        root = os.environ.get('ZALMOXIS_ROOT', '')
        liq_file = os.path.join(
            root,
            'data',
            'EOS_PALEOS_MgSiO3',
            'paleos_mgsio3_tables_pt_proteus_liquid.dat',
        )
        if not os.path.isfile(liq_file):
            pytest.skip('PALEOS data not found')

        from zalmoxis.eos_functions import load_paleos_table

        cache = load_paleos_table(liq_file)

        # Query a point that we know is in the NaN zone:
        # P=8 GPa, T=7000 K (outside liquid table valid range at this P)
        logp = np.log10(8e9)
        logt = np.log10(7000.0)

        rho_linear = float(cache['density_interp']((logp, logt)))
        rho_nn = float(cache['density_nn']((logp, logt)))

        assert np.isnan(rho_linear), 'Expected NaN from linear interp at (8 GPa, 7000 K)'
        assert np.isfinite(rho_nn), f'NN fallback returned non-finite: {rho_nn}'
        assert rho_nn > 1000, f'NN density {rho_nn:.0f} unreasonably low'
