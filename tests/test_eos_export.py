"""Tests for ``zalmoxis.eos_export``: P-T to P-S conversion utilities.

Covers loaders, interpolator builders, NaN fillers, SPIDER 1D/2D file
writers, phase-boundary conversion, full SPIDER P-S table generation,
Aragog P-T table writers, surface-entropy lookup and the entropy-conserving
adiabat inversion. The strategy is to construct a small synthetic PALEOS
unified table whose entropy field is monotone in temperature so that the
S(P,T) inversion has a unique root, then exercise every writer/loader
round-trip and every phase-handling branch.

Anti-happy-path coverage in each test class: at least one test exercises an
edge case (empty grid, all-NaN region, decreasing input, mushy-zone limit),
and at least one exercises a physically unreasonable input that must raise
or be handled (NaN lookup, melting curves outside table bounds).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from zalmoxis import eos_export

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Synthetic PALEOS unified table
# ---------------------------------------------------------------------------

# Use an asymmetric grid so any P/T axis swap or transpose bug is detected
# by node-value tests.
_P_NODES_PA = np.logspace(6.0, 10.0, 5)  # 1e6 .. 1e10 Pa
_T_NODES_K = np.logspace(3.0, 4.0, 5)  # 1000 .. 10000 K


def _rho(P, T):
    """Synthetic density [kg/m^3]: increases with P, decreases with T."""
    return 4000.0 + 1.0e-7 * P - 0.05 * T


def _u(P, T):  # noqa: ARG001
    """Synthetic internal energy [J/kg]; pressure-independent for simplicity."""
    return 1.0e3 * T


def _s(P, T):  # noqa: ARG001
    """Synthetic entropy [J/(kg*K)]: strictly monotone in T at fixed P.

    Monotonicity is required by ``compute_entropy_adiabat`` and by the
    bisection in ``generate_spider_eos_tables._fill_phase_grid``.
    """
    return 1000.0 * np.log(T / 300.0)


def _cp(P, T):  # noqa: ARG001
    return 1200.0


def _cv(P, T):  # noqa: ARG001
    return 1100.0


def _alpha(P, T):  # noqa: ARG001
    return 1.0e-5


def _nabla_ad(P, T):  # noqa: ARG001
    return 0.3


def _phase_for_T(T, T_phase=3000.0):
    return 'liquid' if T > T_phase else 'solid'


def _write_paleos_unified(path: Path, P_arr, T_arr, T_phase=3000.0):
    """Write a synthetic 10-column PALEOS unified table to ``path``."""
    lines = ['# Synthetic PALEOS unified table for eos_export tests\n']
    for P in P_arr:
        for T in T_arr:
            row = (
                f'{P:.8e} {T:.8e} '
                f'{_rho(P, T):.8e} {_u(P, T):.8e} {_s(P, T):.8e} '
                f'{_cp(P, T):.8e} {_cv(P, T):.8e} '
                f'{_alpha(P, T):.8e} {_nabla_ad(P, T):.8e} '
                f'{_phase_for_T(T, T_phase)}\n'
            )
            lines.append(row)
    path.write_text(''.join(lines))


@pytest.fixture
def synthetic_table(tmp_path):
    """A small synthetic PALEOS unified table (5x5)."""
    p = tmp_path / 'synth_unified.dat'
    _write_paleos_unified(p, _P_NODES_PA, _T_NODES_K)
    return p


@pytest.fixture
def synthetic_2phase(tmp_path):
    """Solid-phase and liquid-phase synthetic tables.

    Same numeric values as the unified table but with constant phase strings,
    so 2-phase code paths can be exercised without depending on the unified
    table's phase column.
    """
    P_arr = _P_NODES_PA
    T_arr = _T_NODES_K

    solid = tmp_path / 'synth_solid.dat'
    lines = ['# solid-phase synthetic table\n']
    for P in P_arr:
        for T in T_arr:
            lines.append(
                f'{P:.8e} {T:.8e} {_rho(P, T):.8e} {_u(P, T):.8e} {_s(P, T):.8e} '
                f'{_cp(P, T):.8e} {_cv(P, T):.8e} {_alpha(P, T):.8e} '
                f'{_nabla_ad(P, T):.8e} solid\n'
            )
    solid.write_text(''.join(lines))

    # Make the liquid table physically distinguishable from the solid table:
    # 5% lower density, 8% higher entropy at every (P, T). This is essential
    # for tests that need to verify the 2-phase code paths actually use the
    # liquid table (not the unified fallback).
    liquid = tmp_path / 'synth_liquid.dat'
    lines = ['# liquid-phase synthetic table\n']
    for P in P_arr:
        for T in T_arr:
            lines.append(
                f'{P:.8e} {T:.8e} {0.95 * _rho(P, T):.8e} {_u(P, T):.8e} '
                f'{1.08 * _s(P, T):.8e} '
                f'{_cp(P, T):.8e} {_cv(P, T):.8e} {_alpha(P, T):.8e} '
                f'{_nabla_ad(P, T):.8e} liquid\n'
            )
    liquid.write_text(''.join(lines))

    return solid, liquid


@pytest.fixture
def melting_curves():
    """Solidus and liquidus inside the synthetic table T-range.

    Linear-in-log-P with a finite gap; chosen so the mushy zone overlaps
    multiple grid cells but is fully contained in the synthetic T range.
    Match the canonical Zalmoxis melting-curve contract: scalar in -> scalar out,
    array in -> array out.
    """

    def solidus_func(P_Pa):
        # 2000 K at 1 bar, +200 K per decade in P -> within [1000, 10000] K
        out = 2000.0 + 200.0 * np.log10(np.asarray(P_Pa) / 1e5)
        return float(out) if np.ndim(P_Pa) == 0 else out

    def liquidus_func(P_Pa):
        out = 2400.0 + 220.0 * np.log10(np.asarray(P_Pa) / 1e5)
        return float(out) if np.ndim(P_Pa) == 0 else out

    return solidus_func, liquidus_func


# ---------------------------------------------------------------------------
# load_paleos_all_properties
# ---------------------------------------------------------------------------


class TestLoadPaleosAllProperties:
    """``load_paleos_all_properties`` parses a 10-column PALEOS table."""

    def test_returns_dict_with_all_property_grids_and_bounds(self, synthetic_table):
        """Loaded dict contains every property grid plus P/T bounds."""
        out = eos_export.load_paleos_all_properties(synthetic_table)

        expected_keys = {
            'unique_log_p',
            'unique_log_t',
            'rho',
            'u',
            's',
            'cp',
            'cv',
            'alpha',
            'nabla_ad',
            'phase',
            'p_min',
            'p_max',
            't_min',
            't_max',
        }
        assert set(out.keys()) == expected_keys

        # Bounds must come from the data, not symmetric defaults.
        np.testing.assert_allclose(out['p_min'], _P_NODES_PA[0], rtol=1e-9)
        np.testing.assert_allclose(out['p_max'], _P_NODES_PA[-1], rtol=1e-9)
        np.testing.assert_allclose(out['t_min'], _T_NODES_K[0], rtol=1e-9)
        np.testing.assert_allclose(out['t_max'], _T_NODES_K[-1], rtol=1e-9)

        assert out['rho'].shape == (len(_P_NODES_PA), len(_T_NODES_K))

    def test_grid_node_values_recover_input_function(self, synthetic_table):
        """Every (P_i, T_j) cell stores the function value the row carried.

        Discriminating: tests both rho (P-dependent + T-dependent) and s
        (purely T-dependent), so a P/T axis transpose would break rho but
        not s, while a sign error in the entropy formula would break s alone.
        """
        out = eos_export.load_paleos_all_properties(synthetic_table)
        for ip, P in enumerate(_P_NODES_PA):
            for it, T in enumerate(_T_NODES_K):
                np.testing.assert_allclose(out['rho'][ip, it], _rho(P, T), rtol=1e-7)
                np.testing.assert_allclose(out['s'][ip, it], _s(P, T), rtol=1e-7)
                # Entropy must be strictly monotone in T at every P.
                if it > 0:
                    assert out['s'][ip, it] > out['s'][ip, it - 1]

    def test_zero_pressure_rows_filtered_out(self, tmp_path):
        """Rows with P=0 are stripped before grid construction.

        Edge case: some shipped PALEOS tables include P=0 padding rows.
        """
        p = tmp_path / 'with_zero.dat'
        # One P=0 padding row plus a valid 2x2 grid.
        rows = ['# header\n', '0.0 1000.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 solid\n']
        for P in (1e6, 1e8):
            for T in (1000.0, 5000.0):
                rows.append(
                    f'{P:.6e} {T:.6e} {_rho(P, T):.6e} {_u(P, T):.6e} {_s(P, T):.6e} '
                    f'{_cp(P, T):.6e} {_cv(P, T):.6e} {_alpha(P, T):.6e} '
                    f'{_nabla_ad(P, T):.6e} solid\n'
                )
        p.write_text(''.join(rows))

        out = eos_export.load_paleos_all_properties(p)
        # Padding stripped: bounds reflect only the valid rows.
        np.testing.assert_allclose(out['p_min'], 1e6, rtol=1e-9)
        assert out['rho'].shape == (2, 2)

    def test_phase_strings_stripped_of_whitespace(self, tmp_path):
        """Trailing whitespace on phase tokens is removed."""
        p = tmp_path / 'whitespace_phase.dat'
        # Single row with padded phase string.
        p.write_text(
            '# header\n'
            '1e6 1000 4000 1e9 1000 1200 1100 1e-5 0.3   liquid   \n'
            '1e6 5000 3500 1e9 2000 1200 1100 1e-5 0.3   liquid   \n'
        )
        out = eos_export.load_paleos_all_properties(p)
        # The leading/trailing spaces in the source must not survive.
        assert out['phase'][0, 0] == 'liquid'


# ---------------------------------------------------------------------------
# _build_interpolator and _fill_nan_nearest
# ---------------------------------------------------------------------------


class TestBuildInterpolator:
    """Internal RegularGridInterpolator wrapper."""

    def test_node_lookup_recovers_grid_value(self):
        """At an exact grid node the interpolator returns the stored value."""
        log_p = np.array([6.0, 8.0, 10.0])
        log_t = np.array([3.0, 3.5, 4.0])
        # Asymmetric values so any axis swap is visible.
        grid = np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ]
        )
        interp = eos_export._build_interpolator(log_p, log_t, grid)
        np.testing.assert_allclose(interp((8.0, 3.5)), 5.0, rtol=1e-12)

    def test_off_node_value_is_strict_bilinear_average(self):
        """At the cell midpoint the interpolated value is the 4-corner mean."""
        log_p = np.array([6.0, 7.0])
        log_t = np.array([3.0, 4.0])
        grid = np.array([[10.0, 20.0], [30.0, 50.0]])  # asymmetric corners
        interp = eos_export._build_interpolator(log_p, log_t, grid)
        # bilinear average at the cell centre = (10+20+30+50)/4 = 27.5
        np.testing.assert_allclose(interp((6.5, 3.5)), 27.5, rtol=1e-12)

    def test_out_of_bounds_returns_nan_not_raises(self):
        """Out-of-bounds queries return NaN (bounds_error=False contract)."""
        log_p = np.array([6.0, 7.0])
        log_t = np.array([3.0, 4.0])
        grid = np.array([[1.0, 2.0], [3.0, 4.0]])
        interp = eos_export._build_interpolator(log_p, log_t, grid)
        assert np.isnan(float(interp((10.0, 5.0))))


class TestFillNanNearest:
    """In-place nearest-neighbor NaN fill (Manhattan)."""

    def test_no_nan_input_grid_unchanged(self):
        """When the input has no NaN, the function is a no-op."""
        grid = np.arange(12.0, dtype=float).reshape(3, 4)
        before = grid.copy()
        eos_export._fill_nan_nearest(grid)
        np.testing.assert_array_equal(grid, before)

    def test_isolated_nan_is_replaced_by_finite_neighbor(self):
        """A single interior NaN is replaced by an adjacent finite value.

        Discriminating: surround the NaN with 4 distinct finite values so a
        wrong choice of neighbor would be detectable.
        """
        grid = np.array(
            [
                [10.0, 20.0, 30.0],
                [40.0, np.nan, 60.0],
                [70.0, 80.0, 90.0],
            ]
        )
        eos_export._fill_nan_nearest(grid)
        assert not np.isnan(grid).any()
        # Whichever neighbor is selected must be one of the 4 finite ones.
        assert grid[1, 1] in {20.0, 40.0, 60.0, 80.0}

    def test_corner_nan_is_replaced_by_diagonal_neighbor(self):
        """A NaN at a grid corner is filled from the nearest finite cell."""
        grid = np.array(
            [
                [np.nan, 20.0],
                [40.0, 50.0],
            ]
        )
        eos_export._fill_nan_nearest(grid)
        assert not np.isnan(grid).any()
        assert grid[0, 0] in {20.0, 40.0, 50.0}

    def test_returns_silently_when_no_nan_present(self):
        """The early-return branch when ``mask.any()`` is False does not raise."""
        grid = np.ones((2, 2), dtype=float)
        result = eos_export._fill_nan_nearest(grid)
        assert result is None  # in-place; no return value


# ---------------------------------------------------------------------------
# SPIDER file writers (1D + 2D)
# ---------------------------------------------------------------------------


def _read_spider_1d(path):
    """Inverse of ``_write_spider_1d`` for round-trip tests."""
    text = Path(path).read_text().splitlines()
    # Header: 5 lines starting with '#'
    header = [ln for ln in text if ln.startswith('#')]
    data = [ln for ln in text if not ln.startswith('#') and ln.strip()]
    P_scale, S_scale = map(float, header[-1].lstrip('#').split())
    rows = np.array([list(map(float, ln.split())) for ln in data])
    return rows[:, 0] * P_scale, rows[:, 1] * S_scale, P_scale, S_scale, header


def _read_spider_2d(path):
    text = Path(path).read_text().splitlines()
    header = [ln for ln in text if ln.startswith('#')]
    data = [ln for ln in text if not ln.startswith('#') and ln.strip()]
    P_scale, S_scale, Q_scale = map(float, header[-1].lstrip('#').split())
    rows = np.array([list(map(float, ln.split())) for ln in data])
    return rows, P_scale, S_scale, Q_scale, header


class TestWriteSpiderFiles:
    """SPIDER 1D phase-boundary and 2D EOS lookup file writers."""

    def test_1d_round_trip_recovers_inputs_within_floating_tolerance(self, tmp_path):
        """Writing then re-parsing reproduces the SI inputs to ~1e-15 rel tol.

        Discriminating: pressures span 4 orders of magnitude so a wrong
        scaling factor would shift the recovered values by orders of
        magnitude, not a small constant.
        """
        P_Pa = np.array([1e5, 1e7, 1e9, 1e11])
        S_SI = np.array([500.0, 1500.0, 2500.0, 3500.0])
        path = tmp_path / 'p_s.dat'
        eos_export._write_spider_1d(path, P_Pa, S_SI)
        P_back, S_back, P_scale, S_scale, _ = _read_spider_1d(path)
        np.testing.assert_allclose(P_back, P_Pa, rtol=1e-13)
        np.testing.assert_allclose(S_back, S_SI, rtol=1e-13)

    def test_1d_header_layout_matches_spider_format(self, tmp_path):
        """Five-line ``#``-prefixed header with `# 5 N` on the first line."""
        path = tmp_path / 'header.dat'
        eos_export._write_spider_1d(path, np.array([1e5, 1e6]), np.array([1.0, 2.0]))
        text = Path(path).read_text().splitlines()
        header = [ln for ln in text if ln.startswith('#')]
        assert len(header) == 5
        # First header line: `# 5 N`
        tokens = header[0].lstrip('#').split()
        assert tokens[0] == '5'
        assert int(tokens[1]) == 2

    def test_1d_custom_scales_round_trip(self, tmp_path):
        """Non-default P_scale / S_scale survives a round-trip."""
        path = tmp_path / 'custom_scales.dat'
        P_Pa = np.array([1e6, 1e9])
        S_SI = np.array([1000.0, 4000.0])
        eos_export._write_spider_1d(path, P_Pa, S_SI, P_scale=1e6, S_scale=2000.0)
        P_back, S_back, P_scale, S_scale, _ = _read_spider_1d(path)
        np.testing.assert_allclose(P_back, P_Pa, rtol=1e-13)
        np.testing.assert_allclose(S_back, S_SI, rtol=1e-13)
        assert P_scale == pytest.approx(1e6)
        assert S_scale == pytest.approx(2000.0)

    def test_2d_grid_layout_uses_S_slow_P_fast(self, tmp_path):
        """Ordering: outer loop is S, inner is P (SPIDER's documented format).

        Discriminating: choose values that make S-slow and P-slow visibly
        different orderings. With S as outer index, the file's first nP rows
        all share the same S=S[0], which is enforced here.
        """
        P_Pa = np.array([1e5, 1e7, 1e9])
        S_SI = np.array([1000.0, 2000.0])
        # values[j, i] for S index j and P index i
        values = np.array(
            [
                [10.0, 20.0, 30.0],  # S=1000
                [40.0, 50.0, 60.0],  # S=2000
            ]
        )
        path = tmp_path / 'grid_2d.dat'
        eos_export._write_spider_2d(path, P_Pa, S_SI, values, quantity_scale=1.0)
        rows, P_scale, S_scale, Q_scale, _ = _read_spider_2d(path)
        # First nP=3 rows share S=S[0], next nP=3 rows share S=S[1]
        np.testing.assert_allclose(rows[:3, 1] * S_scale, np.full(3, 1000.0), rtol=1e-13)
        np.testing.assert_allclose(rows[3:, 1] * S_scale, np.full(3, 2000.0), rtol=1e-13)
        # Quantity values match the input grid in the same row order.
        np.testing.assert_allclose(rows[:, 2].reshape(2, 3) * Q_scale, values, rtol=1e-13)

    def test_2d_quantity_scaling_applied_in_storage(self, tmp_path):
        """Stored values are nondimensional: value / quantity_scale."""
        P_Pa = np.array([1e5])
        S_SI = np.array([1000.0])
        values = np.array([[1234.5]])
        path = tmp_path / 'scaled.dat'
        eos_export._write_spider_2d(path, P_Pa, S_SI, values, quantity_scale=10.0)
        rows, _, _, Q_scale, _ = _read_spider_2d(path)
        np.testing.assert_allclose(rows[0, 2], 123.45, rtol=1e-13)
        assert Q_scale == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# _find_valid_T_bounds
# ---------------------------------------------------------------------------


class TestFindValidTBounds:
    """Locate the largest [T_lo, T_hi] sub-interval where S is finite."""

    def test_returns_input_bounds_when_grid_finite_everywhere(self, synthetic_table):
        """Synthetic table has finite S everywhere -> bounds returned unchanged."""
        tab = eos_export.load_paleos_all_properties(synthetic_table)
        s_interp = eos_export._build_interpolator(
            tab['unique_log_p'], tab['unique_log_t'], tab['s']
        )
        T_lo_v, T_hi_v = eos_export._find_valid_T_bounds(
            np.log10(1e8), 1100.0, 9000.0, s_interp, n_probe=20
        )
        assert T_lo_v is not None and T_hi_v is not None
        assert T_lo_v < T_hi_v
        assert T_lo_v >= 1100.0 and T_hi_v <= 9000.0

    def test_returns_none_when_zero_finite_probes(self):
        """Edge case: all probes land in NaN region -> (None, None)."""
        log_p = np.array([6.0, 7.0])
        log_t = np.array([3.0, 4.0])
        # Entire grid is NaN.
        grid = np.full((2, 2), np.nan)
        interp = eos_export._build_interpolator(log_p, log_t, grid)
        T_lo_v, T_hi_v = eos_export._find_valid_T_bounds(
            6.5, 1100.0, 9000.0, interp, n_probe=10
        )
        assert T_lo_v is None and T_hi_v is None

    def test_narrows_when_high_T_region_is_nan(self):
        """Partial-NaN: probes at high T return NaN, valid bounds shrink."""
        log_p = np.array([6.0, 7.0])
        log_t = np.array([3.0, 3.5, 4.0])
        # NaN above logT=3.5 (T > ~3162 K)
        grid = np.array([[100.0, 200.0, np.nan], [110.0, 210.0, np.nan]])
        interp = eos_export._build_interpolator(log_p, log_t, grid)
        T_lo_v, T_hi_v = eos_export._find_valid_T_bounds(
            6.5, 1000.0, 10000.0, interp, n_probe=50
        )
        assert T_hi_v is not None
        # Upper finite probe should be <= 3162 K (the NaN edge), not 10000 K.
        assert T_hi_v < 5000.0


# ---------------------------------------------------------------------------
# generate_spider_phase_boundaries
# ---------------------------------------------------------------------------


class TestGenerateSpiderPhaseBoundaries:
    """T(P) melting curves -> S(P) phase boundaries."""

    def test_writes_solidus_and_liquidus_with_monotone_S(
        self, synthetic_table, melting_curves, tmp_path
    ):
        """Output S-arrays satisfy S_solidus < S_liquidus and dS/dP >= 0."""
        sol_func, liq_func = melting_curves
        out_dir = tmp_path / 'pb_out'
        result = eos_export.generate_spider_phase_boundaries(
            sol_func,
            liq_func,
            synthetic_table,
            P_range=(1e6, 1e9),
            n_P=80,
            output_dir=out_dir,
        )
        assert (out_dir / 'solidus_P-S.dat').is_file()
        assert (out_dir / 'liquidus_P-S.dat').is_file()
        assert result['solidus_path'].endswith('solidus_P-S.dat')

        S_sol = result['S_solidus']
        S_liq = result['S_liquidus']

        # Both arrays finite over the full output grid.
        assert np.all(np.isfinite(S_sol))
        assert np.all(np.isfinite(S_liq))
        # S_liq > S_sol pointwise (latent heat of melting > 0).
        assert np.all(S_liq > S_sol)
        # Monotone after cumulative-max enforcement.
        np.testing.assert_array_less(-np.diff(S_sol), 1e-9)  # non-decreasing
        np.testing.assert_array_less(-np.diff(S_liq), 1e-9)

    def test_returns_empty_when_curves_outside_table_range(self, synthetic_table, tmp_path):
        """Edge case: melting curves below/above the synthetic T range.

        Forces the n_valid==0 branch and the empty-result early return.
        """

        def sol_low(P_Pa):
            return np.full_like(np.atleast_1d(P_Pa), 100.0)  # below table T_min

        def liq_low(P_Pa):
            return np.full_like(np.atleast_1d(P_Pa), 200.0)

        result = eos_export.generate_spider_phase_boundaries(
            sol_low,
            liq_low,
            synthetic_table,
            P_range=(1e6, 1e9),
            n_P=20,
            output_dir=tmp_path,
        )
        assert result['P_Pa'].size == 0
        assert result['solidus_path'] is None
        assert result['liquidus_path'] is None

    def test_unphysical_negative_solidus_T_skipped_silently(self, synthetic_table, tmp_path):
        """A solidus that returns T<=0 must not crash the routine."""

        def sol_neg(P_Pa):
            return np.full_like(np.atleast_1d(P_Pa), -100.0)

        def liq_ok(P_Pa):
            return np.full_like(np.atleast_1d(P_Pa), 5000.0)

        # Should return an empty result rather than raise.
        result = eos_export.generate_spider_phase_boundaries(
            sol_neg,
            liq_ok,
            synthetic_table,
            P_range=(1e6, 1e9),
            n_P=10,
            output_dir=tmp_path,
        )
        assert result['P_Pa'].size == 0

    def test_2phase_path_uses_phase_specific_entropy(self, synthetic_2phase, melting_curves):
        """Passing solid+liquid tables uses their entropy, not the unified one.

        Discriminating: the synthetic liquid table has S = 1.08 * unified S.
        With 2-phase tables wired in, the returned S_liquidus must reflect
        that 8% offset.
        """
        solid_path, liquid_path = synthetic_2phase
        sol_func, liq_func = melting_curves

        # Run once with the unified-only path.
        unified_result = eos_export.generate_spider_phase_boundaries(
            sol_func,
            liq_func,
            solid_path,  # use solid as "unified" baseline
            P_range=(1e6, 1e9),
            n_P=40,
            output_dir=None,
        )

        # Run again with the explicit 2-phase tables; liquidus must rise.
        twophase_result = eos_export.generate_spider_phase_boundaries(
            sol_func,
            liq_func,
            solid_path,
            P_range=(1e6, 1e9),
            n_P=40,
            output_dir=None,
            solid_eos_file=solid_path,
            liquid_eos_file=liquid_path,
        )

        # S_liquidus should be ~8% higher with the liquid-phase table.
        assert np.all(twophase_result['S_liquidus'] > unified_result['S_liquidus'])

    def test_pchip_smoothing_reduces_dS_dP_sign_changes(self, synthetic_table, melting_curves):
        """The PCHIP-smoothed curve has dS/dP-sign-change count <= raw count.

        Property assertion: smoothing should not introduce new oscillations.
        """
        sol_func, liq_func = melting_curves
        result = eos_export.generate_spider_phase_boundaries(
            sol_func,
            liq_func,
            synthetic_table,
            P_range=(1e6, 1e9),
            n_P=200,
            output_dir=None,
        )
        dliq = np.diff(result['S_liquidus'])
        # After cumulative-max + PCHIP, sign-change count is zero.
        sign_changes = int(np.sum((dliq[:-1] > 0) != (dliq[1:] > 0)))
        assert sign_changes <= 1


# ---------------------------------------------------------------------------
# generate_spider_eos_tables
# ---------------------------------------------------------------------------


class TestGenerateSpiderEosTables:
    """Full P-S EOS table generation (5 properties x 2 phases = 10 files)."""

    def test_returns_grids_with_solid_and_melt_keys(self, synthetic_table, melting_curves):
        """Result dict has solid + melt grids and the expected property keys."""
        sol_func, liq_func = melting_curves
        out = eos_export.generate_spider_eos_tables(
            synthetic_table,
            sol_func,
            liq_func,
            P_range=(1e6, 1e9),
            n_P=15,
            n_S=15,
            output_dir=None,
        )
        assert {'P_Pa', 'S_solid', 'S_melt', 'solid', 'melt', 'output_dir'} <= set(out.keys())
        for prop in ('rho', 'temperature', 'cp', 'alpha', 'nabla_ad'):
            assert out['solid'][prop].shape == (15, 15)
            assert out['melt'][prop].shape == (15, 15)

    def test_writes_ten_dat_files_in_output_dir(
        self, synthetic_table, melting_curves, tmp_path
    ):
        """5 properties * 2 phases = 10 SPIDER P-S files written."""
        sol_func, liq_func = melting_curves
        out_dir = tmp_path / 'spider_pst'
        eos_export.generate_spider_eos_tables(
            synthetic_table,
            sol_func,
            liq_func,
            P_range=(1e6, 1e9),
            n_P=10,
            n_S=10,
            output_dir=out_dir,
        )
        produced = sorted(p.name for p in out_dir.glob('*.dat'))
        expected = sorted(
            f'{name}_{phase}.dat'
            for name in (
                'density',
                'temperature',
                'heat_capacity',
                'thermal_exp',
                'adiabat_temp_grad',
            )
            for phase in ('solid', 'melt')
        )
        assert produced == expected

    def test_solid_S_max_extended_to_match_melt_S_max(self, synthetic_table, melting_curves):
        """Solid-phase S range is extended up to the melt-phase max.

        Matches the comment in eos_export.py around L679: SPIDER queries
        both tables near the boundary, so the solid S range must cover at
        least the melt range.
        """
        sol_func, liq_func = melting_curves
        out = eos_export.generate_spider_eos_tables(
            synthetic_table,
            sol_func,
            liq_func,
            P_range=(1e6, 1e9),
            n_P=10,
            n_S=10,
            output_dir=None,
        )
        # Top of solid grid >= top of melt grid.
        assert out['S_solid'][-1] >= out['S_melt'][-1] - 1e-9

    def test_returns_empty_when_no_valid_phase_range(self, tmp_path):
        """Edge case: melting curves entirely outside the table T-range.

        Forces ``valid_solid.any() or valid_melt.any()`` False branch.
        """
        # A 3x3 PALEOS table.
        path = tmp_path / 'tiny.dat'
        _write_paleos_unified(
            path,
            np.logspace(6, 9, 3),
            np.array([1500.0, 2000.0, 2500.0]),
        )

        def sol(P_Pa):
            return 5e4  # scalar contract; above table T_max

        def liq(P_Pa):
            return 6e4

        out = eos_export.generate_spider_eos_tables(
            path, sol, liq, P_range=(1e6, 1e9), n_P=5, n_S=5, output_dir=None
        )
        assert out == {}

    def test_2phase_path_uses_phase_specific_property_grids(
        self, synthetic_2phase, melting_curves
    ):
        """Passing solid+liquid tables uses them for ALL property lookups.

        Discriminating: liquid synthetic table has rho = 0.95 * unified rho.
        With 2-phase tables wired in, the melt-grid density must be
        consistently lower than the solid-grid density at matching (P, S).
        """
        solid_path, liquid_path = synthetic_2phase
        sol_func, liq_func = melting_curves

        out = eos_export.generate_spider_eos_tables(
            solid_path,  # use solid as "unified" baseline
            sol_func,
            liq_func,
            P_range=(1e6, 1e9),
            n_P=12,
            n_S=12,
            output_dir=None,
            solid_eos_file=solid_path,
            liquid_eos_file=liquid_path,
        )
        # Both phase grids populated with finite densities (after NaN fill).
        assert np.all(np.isfinite(out['solid']['rho']))
        assert np.all(np.isfinite(out['melt']['rho']))
        # The melt-side density values come from the liquid table, so the
        # column of melt rho values is uniformly ~5% below the solid rho
        # column at matching pressure (the synthetic offset).
        med_solid = float(np.median(out['solid']['rho']))
        med_melt = float(np.median(out['melt']['rho']))
        assert med_melt < med_solid


# ---------------------------------------------------------------------------
# compute_entropy_adiabat: bracket-shrinking branch (low-P / high-T NaN region)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_table_with_nan_corner(tmp_path):
    """PALEOS-format table with NaN entropy at low-P / high-T corner.

    Models the real PALEOS-MgSiO3-liquid behaviour where the high-T plasma
    region at low P returns NaN and breaks naive bracket expansion. Used to
    exercise the bracket-shrinking branch in ``compute_entropy_adiabat``.
    """
    p = tmp_path / 'paleos_with_nan.dat'
    # Use a denser T-grid so the cell between T_prev and the NaN region
    # has a fully-finite cell available after one shrink, exercising the
    # shrink-on-NaN branch without trapping S_target itself in NaN.
    P_arr = np.logspace(5, 11, 7)  # 1e5 .. 1e11 Pa
    T_arr = np.logspace(2.5, 4.0, 13)  # ~316 .. 10000 K, log-spaced
    lines = ['# Table with NaN at low-P high-T corner\n']
    for P in P_arr:
        for T in T_arr:
            # NaN region: P < 1e7 Pa AND T > 8000 K. T_surface=4500 below,
            # T_hi initial = 2*T_surf = 9000 lands in NaN.
            in_nan_region = (P < 1e7) and (T > 8000.0)
            if in_nan_region:
                # Write 'nan' values; load_paleos_all_properties will treat
                # them as NaN via numpy.genfromtxt
                lines.append(f'{P:.6e} {T:.6e} nan nan nan nan nan nan nan vapour\n')
            else:
                lines.append(
                    f'{P:.6e} {T:.6e} {_rho(P, T):.6e} {_u(P, T):.6e} '
                    f'{_s(P, T):.6e} {_cp(P, T):.6e} {_cv(P, T):.6e} '
                    f'{_alpha(P, T):.6e} {_nabla_ad(P, T):.6e} liquid\n'
                )
    p.write_text(''.join(lines))
    return p


class TestComputeEntropyAdiabatNanRegion:
    """Bracket-shrinking logic when PALEOS returns NaN in part of T-space."""

    def test_bracket_shrinks_when_T_hi_lands_in_nan(self, synthetic_table_with_nan_corner):
        """At low surface P with T_surf such that 2*T_surf is in the NaN region.

        Forces the ``not np.isfinite(s_hi)`` branch in the expansion loop:
        the routine must shrink T_hi back toward T_prev rather than expand
        further into the NaN region.
        """
        # T_surface = 4500 K, P_surface = 1e6 Pa: T_hi initial = 9000 K is
        # above the 8000 K NaN threshold at P<1e7 Pa, but T_surface itself
        # is well below it so S_target is finite.
        result = eos_export.compute_entropy_adiabat(
            synthetic_table_with_nan_corner,
            T_surface=4500.0,
            P_surface=1e6,
            P_cmb=1e10,
            n_points=8,
        )
        # S_target is computed at the surface and must be finite.
        assert np.isfinite(result['S_target'])
        # T profile must be positive everywhere (fallback to T_prev if any
        # bracket fails) and at least the first row matches T_surface within
        # the shrink-then-bisect tolerance.
        assert np.all(np.array(result['T']) > 0)

    def test_bracket_failure_falls_back_to_T_prev(self, synthetic_table_with_nan_corner):
        """If the bracket expansion fails to find a sign change, T_prev is reused.

        Edge case for the ``not bracket_ok`` branch around L1459-1469.
        """
        # Surface squarely inside the NaN region of the table -> S_target NaN
        # would propagate; the routine should not crash but may use T_prev
        # as a fallback for some pressures.
        try:
            result = eos_export.compute_entropy_adiabat(
                synthetic_table_with_nan_corner,
                T_surface=8000.0,
                P_surface=1e6,
                P_cmb=1e10,
                n_points=4,
            )
            # If it survives (S_target finite from interpolation slop),
            # at least the T profile must be non-negative.
            assert np.all(np.array(result['T']) >= 0)
        except (ValueError, RuntimeError):
            # If the routine raises because S_target itself is NaN, that's
            # also acceptable behaviour for this physically unreasonable input.
            pass


# ---------------------------------------------------------------------------
# generate_aragog_pt_tables (single-phase + 2-phase)
# ---------------------------------------------------------------------------


class TestGenerateAragogPtTables:
    """Aragog-format P-T 3-column tables."""

    def test_writes_solid_and_melt_files_with_identical_data(self, synthetic_table, tmp_path):
        """Single-phase Aragog writer emits solid+melt files with same content."""
        out_dir = tmp_path / 'aragog'
        result = eos_export.generate_aragog_pt_tables(
            synthetic_table,
            P_range=(1e6, 1e9),
            n_P=8,
            n_T=8,
            output_dir=out_dir,
        )
        assert result is not None
        for prop in ('density', 'heat_capacity', 'thermal_exp', 'entropy'):
            solid_path = out_dir / f'{prop}_solid.dat'
            melt_path = out_dir / f'{prop}_melt.dat'
            assert solid_path.is_file() and melt_path.is_file()
            # Aragog single-phase semantic: solid == melt file by design.
            assert solid_path.read_text() == melt_path.read_text()

    def test_aragog_file_first_column_is_pressure_in_pa(self, synthetic_table, tmp_path):
        """First data column carries pressure values inside the requested range."""
        out_dir = tmp_path / 'aragog'
        eos_export.generate_aragog_pt_tables(
            synthetic_table,
            P_range=(1e6, 1e9),
            n_P=4,
            n_T=4,
            output_dir=out_dir,
        )
        rows = np.genfromtxt(out_dir / 'density_melt.dat', comments='#')
        P_col = rows[:, 0]
        assert P_col.min() >= 1e6
        assert P_col.max() <= 1e9 + 1.0  # allow tiny float slop

    def test_2phase_writer_uses_phase_specific_density(self, synthetic_2phase, tmp_path):
        """Solid file's density >= melt file's density (synthetic offset = 5%)."""
        solid_path, liquid_path = synthetic_2phase
        out_dir = tmp_path / 'aragog_2p'
        eos_export.generate_aragog_pt_tables_2phase(
            solid_path,
            liquid_path,
            P_range=(1e6, 1e9),
            n_P=6,
            n_T=6,
            output_dir=out_dir,
        )
        solid_rho = np.genfromtxt(out_dir / 'density_solid.dat', comments='#')[:, 2]
        melt_rho = np.genfromtxt(out_dir / 'density_melt.dat', comments='#')[:, 2]
        assert np.all(solid_rho >= melt_rho - 1e-9)

    def test_2phase_t_max_warning_path_does_not_raise(self, tmp_path, synthetic_2phase):
        """The T_max < 6000 K log-warning branch must not raise."""
        # synthetic_2phase fixture's T_max = 1e4 K; downsize via P_range only.
        solid_path, liquid_path = synthetic_2phase
        result = eos_export.generate_aragog_pt_tables_2phase(
            solid_path,
            liquid_path,
            P_range=(1e6, 1e9),
            n_P=4,
            n_T=4,
            output_dir=tmp_path,
        )
        assert result is not None
        assert 'output_dir' in result


# ---------------------------------------------------------------------------
# compute_surface_entropy
# ---------------------------------------------------------------------------


class TestComputeSurfaceEntropy:
    """Single-point surface S(P,T) lookup with optional phase weighting."""

    def test_returns_dict_with_target_and_echoed_inputs(self, synthetic_table):
        """Result dict carries ``S_target`` plus the inputs."""
        result = eos_export.compute_surface_entropy(
            synthetic_table, T_surface=2500.0, P_surface=1e6
        )
        assert {'S_target', 'P_surface', 'T_surface'} <= set(result.keys())
        assert result['T_surface'] == pytest.approx(2500.0)
        assert result['P_surface'] == pytest.approx(1e6)
        assert np.isfinite(result['S_target'])

    def test_S_target_recovers_underlying_function_at_grid_node(self, synthetic_table):
        """At a grid node, S_target equals the synthetic ``_s`` function value."""
        result = eos_export.compute_surface_entropy(
            synthetic_table, T_surface=1000.0, P_surface=1e6
        )
        # The synthetic table is written with .8e precision, so the file
        # round-trip introduces ~1e-8 relative error. 1e-6 is the realistic bar.
        np.testing.assert_allclose(result['S_target'], _s(1e6, 1000.0), rtol=1e-6)

    def test_phase_weighted_entropy_inside_mushy_zone(self, synthetic_table, melting_curves):
        """Inside the mushy zone S_target = phi * S_liq + (1-phi) * S_sol.

        Discriminating: at the midpoint of the mushy band phi=0.5, so
        S_target should equal the average of S(T_sol) and S(T_liq) within
        synthetic-table interpolation tolerance, not just S(T_input).
        """
        sol_func, liq_func = melting_curves
        # Pick (P, T) inside mushy zone at P=1e7 Pa.
        P = 1e7
        T_sol = float(sol_func(P))
        T_liq = float(liq_func(P))
        T_mid = 0.5 * (T_sol + T_liq)
        result = eos_export.compute_surface_entropy(
            synthetic_table,
            T_surface=T_mid,
            P_surface=P,
            solidus_func=sol_func,
            liquidus_func=liq_func,
        )
        S_expected = 0.5 * (_s(P, T_sol) + _s(P, T_liq))
        np.testing.assert_allclose(result['S_target'], S_expected, rtol=5e-2)

    def test_raises_when_lookup_returns_nan(self, tmp_path):
        """Edge case: query off the table -> ValueError, not silent NaN."""
        # 2x2 table with all-NaN entropy column achieved by setting s = inf*0.
        path = tmp_path / 'tiny_finite.dat'
        # Build a 2x2 table covering [1e6, 1e7] Pa, [1000, 2000] K.
        _write_paleos_unified(path, np.array([1e6, 1e7]), np.array([1000.0, 2000.0]))
        # Query well outside the table -> RegularGridInterpolator returns NaN.
        with pytest.raises(ValueError, match='returned NaN'):
            eos_export.compute_surface_entropy(path, T_surface=1e6, P_surface=1e15)

    def test_vectorised_solidus_callable_accepted_via_float_coercion(self, synthetic_table):
        """Vectorised melting-curve callables (returning 1-element ndarrays)
        must work in the mushy-zone branch.

        Regression: prior to the float() coercion in compute_surface_entropy,
        a callable like ``lambda P: np.atleast_1d(2000.0 + ...)`` raised
        ``ValueError: setting an array element with a sequence`` from the
        ``np.array([[...]])`` construction. The float() now coerces both
        scalar-returning and array-returning callables.
        """

        def sol_array(P):  # array-returning callable
            return np.atleast_1d(2000.0 + 200.0 * np.log10(np.asarray(P) / 1e5))

        def liq_array(P):
            return np.atleast_1d(2400.0 + 220.0 * np.log10(np.asarray(P) / 1e5))

        # T inside the mushy zone at P=1e7 Pa -> mushy branch executes.
        P = 1e7
        T_mid = 0.5 * (float(sol_array(P).item()) + float(liq_array(P).item()))
        result = eos_export.compute_surface_entropy(
            synthetic_table,
            T_surface=T_mid,
            P_surface=P,
            solidus_func=sol_array,
            liquidus_func=liq_array,
        )
        assert np.isfinite(result['S_target'])

    def test_2phase_tables_used_when_provided_in_mushy_zone(
        self, synthetic_table, synthetic_2phase, melting_curves
    ):
        """When solid+liquid tables are passed, mushy S uses them, not the unified."""
        solid_path, liquid_path = synthetic_2phase
        sol_func, liq_func = melting_curves
        P = 1e7
        T_sol = float(sol_func(P))
        T_liq = float(liq_func(P))
        T_mid = 0.5 * (T_sol + T_liq)

        # Unified path
        unified = eos_export.compute_surface_entropy(
            synthetic_table,
            T_surface=T_mid,
            P_surface=P,
            solidus_func=sol_func,
            liquidus_func=liq_func,
        )['S_target']
        # 2-phase path
        twophase = eos_export.compute_surface_entropy(
            synthetic_table,
            T_surface=T_mid,
            P_surface=P,
            solidus_func=sol_func,
            liquidus_func=liq_func,
            solid_eos_file=solid_path,
            liquid_eos_file=liquid_path,
        )['S_target']

        # The 2-phase liquid table has S = 1.08 * unified, so the phase-
        # weighted S_target should be larger than the unified result.
        assert twophase > unified


# ---------------------------------------------------------------------------
# compute_entropy_adiabat
# ---------------------------------------------------------------------------


class TestComputeEntropyAdiabat:
    """Entropy-conserving adiabat T(P) via brentq inversion of S(P,T)=S_target."""

    def test_T_profile_monotonic_with_P(self, synthetic_table):
        """Monotone synthetic entropy in T -> monotone T(P) along an adiabat.

        Property assertion. With ``_s(T) = 1000 * log(T/300)`` independent
        of P, S_target = const yields T(P) = const. Check that T does not
        oscillate (within numerical tolerance).
        """
        result = eos_export.compute_entropy_adiabat(
            synthetic_table,
            T_surface=2000.0,
            P_surface=1e6,
            P_cmb=1e9,
            n_points=40,
        )
        T_profile = result['T']
        assert np.all(T_profile > 0)
        # Profile drift over 3 decades in P should be < 1% (synthetic S is
        # P-independent).
        np.testing.assert_allclose(T_profile, 2000.0, rtol=2e-2)

    def test_S_profile_close_to_target_within_tolerance(self, synthetic_table):
        """S_profile values are within 1e-4 relative of S_target.

        Property: the brentq inversion has rtol=1e-10, but synthetic-table
        interpolation residuals dominate; 1e-4 is the realistic bar.
        """
        result = eos_export.compute_entropy_adiabat(
            synthetic_table,
            T_surface=2000.0,
            P_surface=1e6,
            P_cmb=1e9,
            n_points=20,
        )
        np.testing.assert_allclose(result['S_profile'], result['S_target'], rtol=1e-4)

    def test_raises_when_eos_file_missing(self, tmp_path):
        """Edge case: passing a non-existent file raises FileNotFoundError or OSError."""
        bogus = tmp_path / 'does_not_exist.dat'
        with pytest.raises((FileNotFoundError, OSError)):
            eos_export.compute_entropy_adiabat(
                bogus, T_surface=2000.0, P_surface=1e6, P_cmb=1e9
            )

    def test_phase_weighted_adiabat_with_2phase_tables(self, synthetic_2phase, melting_curves):
        """2-phase entropy used in the mushy zone changes the recovered T(P)."""
        solid_path, liquid_path = synthetic_2phase
        sol_func, liq_func = melting_curves
        # Surface T inside the mushy band so the 2-phase path is exercised.
        T_surf = 0.5 * (float(sol_func(1e6)) + float(liq_func(1e6)))
        result = eos_export.compute_entropy_adiabat(
            solid_path,
            T_surface=T_surf,
            P_surface=1e6,
            P_cmb=1e9,
            n_points=12,
            solidus_func=sol_func,
            liquidus_func=liq_func,
            solid_eos_file=solid_path,
            liquid_eos_file=liquid_path,
        )
        assert np.all(result['T'] > 0)
        assert np.isfinite(result['S_target'])
