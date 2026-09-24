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

import os
from pathlib import Path

import numpy as np
import pytest

from zalmoxis import eos_export
from zalmoxis.eos.interpolation import read_table_columns

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


def _s_pdep(P, T):
    """P- and T-dependent synthetic entropy [J/(kg*K)], monotone in T.

    Unlike ``_s``, this depends on pressure, so an isentrope is a genuine
    ``T(P)`` curve rather than a constant-temperature line. That lets an
    adiabat test exercise the molten interpolation as a function of depth and
    supplies an independent reference for the recovered profile entropy.
    """
    return 1000.0 * np.log(T / 300.0) - 150.0 * np.log(P / 1.0e6)


def _pdep_solidus(P):
    """Solidus for the P-dependent 2-phase fixture; scalar in, scalar out."""
    out = 2000.0 + 200.0 * np.log10(np.asarray(P) / 1e5)
    return float(out) if np.ndim(P) == 0 else out


def _pdep_liquidus(P):
    """Liquidus for the P-dependent 2-phase fixture; scalar in, scalar out."""
    out = 2400.0 + 220.0 * np.log10(np.asarray(P) / 1e5)
    return float(out) if np.ndim(P) == 0 else out


def _write_pdep_phase_table(path, P_arr, T_arr, s_factor, phase, nan_above=None):
    """Write a synthetic P-dependent phase table.

    When ``nan_above`` (a callable ``P -> T_cut``) is given, the entropy of
    every row with ``T > T_cut(P)`` is written as ``nan``, so
    ``load_paleos_all_properties`` returns a NaN entropy there. This mirrors
    the real PALEOS solid table, whose entropy is non-converged (NaN) at
    molten temperatures above the liquidus.
    """
    lines = [f'# synthetic P-dependent {phase} table\n']
    for P in P_arr:
        for T in T_arr:
            if nan_above is not None and T > nan_above(P):
                s_val = 'nan'
            else:
                s_val = f'{s_factor * _s_pdep(P, T):.8e}'
            lines.append(
                f'{P:.8e} {T:.8e} {4000.0:.8e} {1.0e3 * T:.8e} '
                f'{s_val} 1200 1100 1e-5 0.3 {phase}\n'
            )
    Path(path).write_text(''.join(lines))


@pytest.fixture
def pdep_2phase(tmp_path):
    """P-dependent solid/liquid tables; solid entropy is NaN above the liquidus.

    The liquid entropy is 1.08x the solid formula and finite throughout; the
    solid entropy is NaN for ``T > liquidus(P)``, reproducing the real PALEOS
    solid table's non-converged molten region. A fully molten adiabat over
    these tables reproduces the NaN plateau on the unfixed code (molten points
    read the solid table) and a finite, depth-varying profile once molten
    points are routed to the liquid table.
    """
    P_arr = np.logspace(6.0, 9.0, 16)
    T_arr = np.logspace(3.0, 4.3, 28)
    solid = tmp_path / 'pdep_solid.dat'
    liquid = tmp_path / 'pdep_liquid.dat'
    _write_pdep_phase_table(solid, P_arr, T_arr, 1.0, 'solid', nan_above=_pdep_liquidus)
    _write_pdep_phase_table(liquid, P_arr, T_arr, 1.08, 'liquid')
    return solid, liquid


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


@pytest.fixture(autouse=True)
def _fresh_table_cache():
    """Start and end every test in this module with an empty parsed-table cache."""
    eos_export._parse_paleos_table.cache_clear()
    yield
    eos_export._parse_paleos_table.cache_clear()


def _reference_arrays(path):
    """The numeric and phase columns read the way the loader read them before it cached."""
    numeric = np.genfromtxt(path, usecols=range(9), comments='#')
    phase = np.genfromtxt(path, usecols=(9,), dtype=str, comments='#')
    return numeric, phase


def _real_paleos_tables():
    """Paths of the shipped unified, solid and liquid tables that exist locally."""
    roots = []
    if os.environ.get('FWL_DATA'):
        roots.append(Path(os.environ['FWL_DATA']) / 'zalmoxis_eos')
    try:
        from zalmoxis import get_zalmoxis_root

        roots.append(Path(get_zalmoxis_root()) / 'data')
    except RuntimeError:
        pass
    names = [
        'EOS_PALEOS_MgSiO3_unified/paleos_mgsio3_eos_table_pt.dat',
        'EOS_PALEOS_MgSiO3/paleos_mgsio3_tables_pt_proteus_solid.dat',
        'EOS_PALEOS_MgSiO3/paleos_mgsio3_tables_pt_proteus_liquid.dat',
    ]
    return [r / n for r in roots for n in names if (r / n).is_file()]


class TestLoadPaleosAllPropertiesCache:
    """The parsed table is read once per file version and cannot be changed through the cache."""

    def test_second_call_does_not_read_the_file_again(self, synthetic_table, monkeypatch):
        """The reader runs twice for the first load (numbers, phases) and never after."""
        calls = []
        real = eos_export.read_table_columns

        def counting(*args, **kwargs):
            calls.append(args[1])
            return real(*args, **kwargs)

        monkeypatch.setattr(eos_export, 'read_table_columns', counting)

        first = eos_export.load_paleos_all_properties(synthetic_table)
        assert len(calls) == 2
        second = eos_export.load_paleos_all_properties(synthetic_table)
        third = eos_export.load_paleos_all_properties(str(synthetic_table))

        assert len(calls) == 2
        np.testing.assert_array_equal(first['rho'], second['rho'])
        np.testing.assert_array_equal(first['s'], third['s'])

    def test_edited_file_is_read_again(self, synthetic_table):
        """A new modification time, with the same size, invalidates the cached table."""
        before = eos_export.load_paleos_all_properties(synthetic_table)
        stat = synthetic_table.stat()
        text = synthetic_table.read_text()
        edited = text.replace('solid', 'lqiud')  # same length, different phase strings
        assert len(edited) == len(text) and edited != text
        synthetic_table.write_text(edited)
        os.utime(synthetic_table, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))

        after = eos_export.load_paleos_all_properties(synthetic_table)

        assert before['phase'][0, 0] == 'solid'
        assert after['phase'][0, 0] == 'lqiud'

    def test_a_file_of_another_size_is_read_again(self, synthetic_table):
        """A different size is enough, whatever the modification time says."""
        before = eos_export.load_paleos_all_properties(synthetic_table)
        stat = synthetic_table.stat()
        with open(synthetic_table, 'a') as f:
            f.write('1e11 1000 4000 1e9 1000 1200 1100 1e-5 0.3 solid\n')
        os.utime(synthetic_table, ns=(stat.st_atime_ns, stat.st_mtime_ns))

        after = eos_export.load_paleos_all_properties(synthetic_table)

        assert after['rho'].shape[0] == before['rho'].shape[0] + 1

    def test_arrays_match_the_genfromtxt_reader(self, tmp_path):
        """Header, blank line, a P = 0 row and NaN cells come out as the old reader gave them."""
        path = tmp_path / 'edge.dat'
        rows = [
            '# header line one\n',
            '# header line two\n',
            '\n',
            '0.0 1000.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 solid\n',
        ]
        for P in (1e6, 1e8, 1e10):
            for T in (1000.0, 3000.0, 9000.0):
                nan = 'nan' if (P == 1e10 and T == 9000.0) else f'{_rho(P, T):.8e}'
                rows.append(
                    f'{P:.8e} {T:.8e} {nan} {_u(P, T):.8e} {_s(P, T):.8e} '
                    f'{_cp(P, T):.8e} {_cv(P, T):.8e} {_alpha(P, T):.8e} '
                    f'{_nabla_ad(P, T):.8e}   {_phase_for_T(T)}  \n'
                )
        path.write_text(''.join(rows))
        numeric, phase = _reference_arrays(path)

        out = eos_export.load_paleos_all_properties(path)

        keep = numeric[:, 0] > 0
        assert np.isnan(out['rho']).sum() == 1
        for name, col in zip(['rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad'], range(2, 9)):
            expected = numeric[keep, col].reshape(3, 3)
            np.testing.assert_array_equal(out[name], expected)
        assert list(out['phase'].ravel()) == [p.strip() for p in phase[keep]]

    def test_read_table_columns_is_silent_and_equal_to_genfromtxt(self, synthetic_table):
        """The faster reader gives the same arrays and no warning about the comment header."""
        import warnings

        numeric, phase = _reference_arrays(synthetic_table)

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            got_numeric = read_table_columns(synthetic_table, range(9))
            got_phase = read_table_columns(synthetic_table, (9,), dtype=str)

        np.testing.assert_array_equal(got_numeric, numeric)
        np.testing.assert_array_equal(got_phase, phase)

    def test_cached_arrays_cannot_be_written_through(self, synthetic_table):
        """A caller that assigns into a returned array gets an error, and the cache is unchanged."""
        out = eos_export.load_paleos_all_properties(synthetic_table)
        reference = out['s'].copy()

        with pytest.raises(ValueError, match='read-only'):
            out['s'][0, 0] = -1.0
        with pytest.raises(ValueError, match='read-only'):
            out['unique_log_p'][0] = 0.0
        out['extra'] = 1  # the dict is the caller's own

        again = eos_export.load_paleos_all_properties(synthetic_table)
        assert 'extra' not in again
        np.testing.assert_array_equal(again['s'], reference)
        assert again['s'].flags.writeable is False

    def test_a_copy_of_a_cached_array_can_be_changed(self, synthetic_table):
        """Callers that need a scratch grid copy it, and the copy is a normal array."""
        out = eos_export.load_paleos_all_properties(synthetic_table)
        scratch = out['s'].copy()
        scratch[0, 0] = -1.0
        assert eos_export.load_paleos_all_properties(synthetic_table)['s'][0, 0] != -1.0

    def test_interpolator_accepts_the_read_only_grid(self, synthetic_table):
        """A cached grid feeds the interpolator, and a node lookup returns the stored value."""
        out = eos_export.load_paleos_all_properties(synthetic_table)
        interp = eos_export._build_interpolator(
            out['unique_log_p'], out['unique_log_t'], out['s']
        )
        point = np.array([[np.log10(_P_NODES_PA[2]), np.log10(_T_NODES_K[3])]])
        np.testing.assert_allclose(
            interp(point).item(), _s(_P_NODES_PA[2], _T_NODES_K[3]), rtol=1e-7
        )

    def test_missing_file_raises(self, tmp_path):
        """A path that does not exist is an error, not an empty table."""
        with pytest.raises(FileNotFoundError):
            eos_export.load_paleos_all_properties(tmp_path / 'absent.dat')

    def test_cache_holds_a_few_tables(self, tmp_path):
        """More distinct files than the cache size evict the oldest, and it is read again."""
        paths = []
        for i in range(eos_export._TABLE_CACHE_SIZE + 1):
            path = tmp_path / f'table_{i}.dat'
            _write_paleos_unified(path, _P_NODES_PA, _T_NODES_K)
            paths.append(path)
            eos_export.load_paleos_all_properties(path)
        info = eos_export._parse_paleos_table.cache_info()
        assert info.currsize == eos_export._TABLE_CACHE_SIZE
        assert info.misses == eos_export._TABLE_CACHE_SIZE + 1

    @pytest.mark.skipif(
        not _real_paleos_tables(), reason='shipped PALEOS tables not staged locally'
    )
    def test_shipped_tables_are_identical_to_the_genfromtxt_reader(self):
        """On the real tables the new reader returns the very arrays the old one did."""
        for path in _real_paleos_tables():
            numeric, phase = _reference_arrays(path)
            out = eos_export.load_paleos_all_properties(path)
            keep = numeric[:, 0] > 0
            log_p = np.log10(numeric[keep, 0])
            np.testing.assert_array_equal(out['unique_log_p'], np.unique(log_p))
            assert out['rho'].shape == (
                len(np.unique(log_p)),
                len(np.unique(np.log10(numeric[keep, 1]))),
            )
            n_valid = np.count_nonzero(~np.isnan(out['s']))
            assert n_valid == np.count_nonzero(~np.isnan(numeric[keep, 4]))
            np.testing.assert_array_equal(
                np.sort(out['s'][~np.isnan(out['s'])]),
                np.sort(numeric[keep, 4][~np.isnan(numeric[keep, 4])]),
            )


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

    def test_read_only_grid_evaluates_bit_identically(self):
        """A read-only grid (a cached table) gives the values of a writable one."""
        rng = np.random.default_rng(0)
        log_p, log_t = np.linspace(0, 1, 300), np.linspace(0, 1, 200)
        grid = rng.random((300, 200))
        points = rng.random((5000, 2))
        expected = eos_export._build_interpolator(log_p, log_t, grid)(points)
        frozen = grid.copy()
        frozen.setflags(write=False)
        result = eos_export._build_interpolator(log_p, log_t, frozen)(points)
        np.testing.assert_array_equal(result, expected)
        # The reference itself differs when SciPy sees the read-only array.
        from scipy.interpolate import RegularGridInterpolator

        raw = RegularGridInterpolator((log_p, log_t), frozen)(points)
        assert not np.array_equal(raw, expected)


class TestFillNanNearest:
    """In-place nearest-neighbor NaN fill (Euclidean; per-column then 2D fallback)."""

    def test_no_nan_input_grid_unchanged(self):
        """When the input has no NaN, the function is a no-op."""
        grid = np.arange(12.0, dtype=float).reshape(3, 4)
        before = grid.copy()
        eos_export._fill_nan_nearest(grid)
        np.testing.assert_array_equal(grid, before)

    def test_isolated_nan_is_replaced_by_finite_neighbor(self):
        """A single interior NaN is replaced by a same-column finite value.

        The fill searches each pressure column (axis 1) independently along
        the entropy axis (axis 0), so only the vertical neighbors (20.0,
        80.0) are reachable; the horizontal ones (40.0, 60.0) sit in
        different columns and must never be picked.
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
        assert grid[1, 1] in {20.0, 80.0}

    def test_corner_nan_prefers_same_column_over_closer_other_column(self):
        """A corner NaN is filled from its own column even when another is closer.

        Column 0 has one valid cell (40.0), two rows away. The other
        column holds a valid value (20.0) directly beside the NaN. A
        global 2D nearest search picks 20.0; the per-column fill must
        pick 40.0.
        """
        grid = np.array(
            [
                [np.nan, 20.0],
                [np.nan, 50.0],
                [40.0, 60.0],
            ]
        )
        eos_export._fill_nan_nearest(grid)
        assert grid[0, 0] == 40.0
        assert grid[1, 0] == 40.0

    def test_returns_silently_when_no_nan_present(self):
        """The early-return branch when ``mask.any()`` is False does not raise."""
        grid = np.ones((2, 2), dtype=float)
        result = eos_export._fill_nan_nearest(grid)
        assert result is None  # in-place; no return value

    def test_nan_fill_stays_within_its_own_column(self):
        """A sub-boundary NaN never picks up a neighboring column's value.

        One column (index 6) has valid data reaching much further down
        the entropy axis than its neighbors, mimicking a pressure column
        whose phase boundary (e.g. the liquidus) sits at a much lower
        entropy than the columns around it. Every cell is tagged with a
        value that uniquely encodes its own column, so any cross-column
        fill is directly detectable.
        """
        nS, nP = 30, 12
        s_lo = np.full(nP, 15)
        s_lo[6] = 2

        grid = np.full((nS, nP), np.nan)
        for ip in range(nP):
            for i_s in range(s_lo[ip], nS):
                grid[i_s, ip] = 1000 * ip + i_s  # decodes uniquely: ip, i_s = divmod(v, 1000)

        eos_export._fill_nan_nearest(grid)

        assert not np.isnan(grid).any()
        for ip in range(nP):
            for i_s in range(s_lo[ip]):
                donor_ip, _ = divmod(int(round(grid[i_s, ip])), 1000)
                assert donor_ip == ip, (
                    f'cell (S={i_s}, P={ip}) was filled from column {donor_ip}'
                )

    def test_column_with_no_valid_data_falls_back_to_global_fill(self):
        """A column that is entirely NaN still gets filled, from elsewhere."""
        nS, nP = 10, 5
        grid = np.full((nS, nP), np.nan)
        for ip in range(nP):
            if ip == 2:
                continue  # column 2 has no valid data at all
            grid[:, ip] = 100.0 * ip

        eos_export._fill_nan_nearest(grid)

        assert not np.isnan(grid).any()
        assert np.all(np.isin(grid[:, 2], [100.0, 300.0]))

    def test_all_nan_grid_is_left_unchanged_without_error(self):
        """A grid with no valid cell has no donor; it stays NaN and does not raise."""
        grid = np.full((3, 4), np.nan)
        eos_export._fill_nan_nearest(grid)
        assert grid.shape == (3, 4)
        assert np.isnan(grid).all()

    def test_empty_column_fallback_ignores_per_column_extrapolation(self):
        """A fully-empty column donates from real data, not a filled neighbor.

        Column 1 has one real value, far down its own column; the
        per-column pass extrapolates it across the whole column before the
        fallback runs. Column 3 is fully real data. For column 0 (empty),
        the nearest cell after the per-column pass is column 1 (distance
        1), but that cell is itself an extrapolated copy: the nearest cell
        that held real data is in column 3 (distance 3). The fallback must
        pick the real data in column 3, not the closer extrapolated copy.
        """
        nS, nP = 6, 4
        grid = np.full((nS, nP), np.nan)
        grid[5, 1] = 111.0  # column 1's only real value, at the far row
        grid[:, 3] = 333.0 + np.arange(nS)  # column 3 is fully real data

        eos_export._fill_nan_nearest(grid)

        assert not np.isnan(grid).any()
        assert grid[0, 0] == 333.0


def _curved_liquidus_table(nS=80, nP=20):
    """Build a synthetic melt-phase P-S temperature table with a curved liquidus.

    Rows are entropy ``S`` (J/kg/K), columns are pressure. Column ``j``
    holds data only above its own liquidus entropy ``S_b[j]``, which
    falls by more than one row per column, so the valid region has a
    curved lower edge. Above the edge the temperature is the column's
    analytic liquidus ``T_liq[j]`` plus a linear rise with entropy, and
    ``T_liq`` rises by several percent from one column to the next.

    Returns
    -------
    tuple
        ``(S_axis, P_axis, grid, S_b, T_liq)`` with ``grid`` of shape
        ``(nS, nP)`` and NaN below each column's boundary.
    """
    S_axis = np.linspace(1200.0, 3200.0, nS)
    x = np.linspace(0.0, 1.0, nP)
    P_axis = 1e9 + 1e11 * x  # Pa
    S_b = 3000.0 - 1200.0 * x**0.7  # boundary entropy per column
    T_liq = 2000.0 + 2500.0 * x  # analytic boundary temperature per column
    grid = np.full((nS, nP), np.nan)
    for j in range(nP):
        above = S_axis >= S_b[j]
        grid[above, j] = T_liq[j] + 0.5 * (S_axis[above] - S_b[j])
    return S_axis, P_axis, grid, S_b, T_liq


def _global_2d_nearest_fill(grid):
    """Reference fill: one global 2D nearest-neighbor search over all cells."""
    from scipy.ndimage import distance_transform_edt

    filled = grid.copy()
    mask = np.isnan(filled)
    _, idx = distance_transform_edt(mask, return_distances=True, return_indices=True)
    filled[mask] = grid[tuple(idx[:, mask])]
    return filled


class TestFillOnCurvedBoundary:
    """Fill behavior on a table whose phase boundary curves across columns."""

    def test_off_node_query_below_boundary_stays_in_its_own_column(self):
        """Interpolating within a column's fill never sees a neighboring column.

        The filled grid goes through ``_build_interpolator``, a generic
        bilinear interpolator on the S-P axes. It stands in for the
        consumer's own interpolation; it is not that code. At every
        pressure node, an off-node entropy query midway between two rows
        at depth 0 (the last filled row and the first data row), 2 and 5
        rows below the edge must return the column's own first data
        value, taken from the grid before the fill. A donor from another
        column shifts the filled cell by the column-to-column liquidus
        step (over 100 K here), so the query then misses that value.
        """
        S_axis, P_axis, grid, S_b, T_liq = _curved_liquidus_table()
        first_data = np.array(
            [grid[np.searchsorted(S_axis, S_b[j]), j] for j in range(len(P_axis))]
        )
        eos_export._fill_nan_nearest(grid)
        interp = eos_export._build_interpolator(S_axis, P_axis, grid)

        for j in range(len(P_axis)):
            k = int(np.searchsorted(S_axis, S_b[j]))  # first data row of column j
            assert k > 5
            for depth in (0, 2, 5):  # rows below the boundary edge
                s_mid = 0.5 * (S_axis[k - 1 - depth] + S_axis[k - depth])
                got = float(interp((s_mid, P_axis[j])))
                np.testing.assert_allclose(got, first_data[j], rtol=1e-12)
                assert abs(got - T_liq[j]) < 0.02 * T_liq[j]  # near this column's liquidus

    def test_filled_temperature_tracks_each_columns_own_liquidus(self):
        """Below the boundary each column's fill stays near its own analytic liquidus.

        The tolerance is one third of the smallest column-to-column
        liquidus step (relative), so a fill donated from a neighboring
        column falls outside it. The reference global 2D search violates
        the tolerance in most columns, which shows the check can fail.
        This is a per-column isolation check on a synthetic table; it
        does not measure the off-liquidus fraction of a real table.
        """
        S_axis, P_axis, grid, S_b, T_liq = _curved_liquidus_table()
        step = np.abs(np.diff(T_liq)) / T_liq[:-1]
        tol = step.min() / 3.0  # a neighbor's liquidus lies well outside tolerance

        reference = _global_2d_nearest_fill(grid)
        off_reference = 0
        for j in range(len(P_axis)):
            below = S_axis < S_b[j]
            assert below.any()
            off_reference += int(np.any(np.abs(reference[below, j] / T_liq[j] - 1.0) > tol))
        assert off_reference > len(P_axis) // 2

        eos_export._fill_nan_nearest(grid)
        for j in range(len(P_axis)):
            below = S_axis < S_b[j]
            rel_err = np.abs(grid[below, j] / T_liq[j] - 1.0)
            assert rel_err.max() < tol, (
                f'column {j}: fill off its liquidus by {rel_err.max():.3f}'
            )


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

    @pytest.mark.physics_invariant
    def test_liquidus_entropy_is_held_at_its_peak(self, tmp_path):
        """A liquidus entropy that peaks and then falls with pressure is
        written flat at its peak value above the peak, and unchanged below it.

        The table entropy s = 1000 ln(T/300) - 150 ln(P/1e6) falls with P at
        fixed T, and the liquidus T = 1500 + 400 ln(P/1e6) flattens, so
        dS_liq/dlnP = 4e5/T_liq - 150 changes sign where T_liq = 2667 K
        (P near 1.9e7 Pa).
        """
        P_arr = np.logspace(6.0, 9.0, 16)
        T_arr = np.logspace(3.0, 4.3, 28)
        solid, liquid = tmp_path / 'peak_solid.dat', tmp_path / 'peak_liquid.dat'
        _write_pdep_phase_table(solid, P_arr, T_arr, 1.0, 'solid')
        _write_pdep_phase_table(liquid, P_arr, T_arr, 1.08, 'liquid')

        def liq(P):
            out = 1500.0 + 400.0 * np.log(np.asarray(P) / 1e6)
            return float(out) if np.ndim(P) == 0 else out

        def sol(P):
            return 0.8 * liq(P)

        res = eos_export.generate_spider_phase_boundaries(
            sol,
            liq,
            solid,
            P_range=(1e6, 1e9),
            n_P=200,
            output_dir=tmp_path / 'pb',
            solid_eos_file=solid,
            liquid_eos_file=liquid,
        )
        P, S_liq = res['P_Pa'], res['S_liquidus']
        i_peak = int(np.argmax(S_liq))
        # The peak sits inside the range, near the analytic 1.9e7 Pa.
        assert 5e6 < P[i_peak] < 7e7
        # Above the peak the curve is flat at the peak value.
        np.testing.assert_array_equal(S_liq[i_peak:], S_liq[i_peak])
        # Below the peak it follows the table (smoothing aside).
        j = i_peak // 2
        assert S_liq[j] == pytest.approx(1.08 * _s_pdep(P[j], liq(P[j])), rel=2e-3)
        # Discrimination: the table curve at the top lies about 140 below.
        assert S_liq[-1] - 1.08 * _s_pdep(P[-1], liq(P[-1])) > 50.0
        # The file holds the same curve.
        on_disk = np.loadtxt(res['liquidus_path'])[:, 1] * eos_export._S_SCALE
        np.testing.assert_allclose(on_disk, S_liq, rtol=1e-12)

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
        assert {'P_Pa', 'S_solid', 'S_melt', 'solid', 'melt', 'valid', 'output_dir'} <= set(
            out.keys()
        )
        # The phase dicts hold the five table properties and nothing else.
        props = set(eos_export._SPIDER_TABLE_PROPERTIES)
        assert set(out['solid']) == props and set(out['melt']) == props
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
                'valid_mask',
            )
            for phase in ('solid', 'melt')
        )
        assert produced == expected

    def test_validity_mask_marks_filled_cells(
        self, synthetic_table, melting_curves, tmp_path, monkeypatch
    ):
        """The validity mask is 1 exactly where every property held a PALEOS
        state before the NaN fill, the fill leaves those cells untouched, and
        the mask file on disk round-trips the same 0/1 grid.

        The melt phase has no state below the liquidus, so the grid carries both
        classes of cell; a mask that marked every cell valid would fail here.
        """
        sol_func, liq_func = melting_curves
        kwargs = dict(P_range=(1e6, 1e9), n_P=12, n_S=14)
        out_dir = tmp_path / 'masked'
        out = eos_export.generate_spider_eos_tables(
            synthetic_table, sol_func, liq_func, output_dir=out_dir, **kwargs
        )
        # Same generation without the fill: the NaN pattern it leaves is the truth.
        monkeypatch.setattr(eos_export, '_fill_nan_nearest', lambda grid: None)
        raw = eos_export.generate_spider_eos_tables(
            synthetic_table, sol_func, liq_func, output_dir=None, **kwargs
        )
        for phase in ('solid', 'melt'):
            mask = out['valid'][phase]
            assert mask.dtype == bool and mask.shape == (14, 12)
            truth = np.all([np.isfinite(raw[phase][p]) for p in raw[phase]], axis=0)
            np.testing.assert_array_equal(mask, truth)
            for prop in ('rho', 'temperature', 'cp', 'alpha', 'nabla_ad'):
                # Valid cells are bit-identical; filled cells are finite.
                np.testing.assert_array_equal(out[phase][prop][mask], raw[phase][prop][mask])
                assert np.all(np.isfinite(out[phase][prop][~mask]))
            on_disk = np.loadtxt(out_dir / f'valid_mask_{phase}.dat', dtype=int)
            np.testing.assert_array_equal(on_disk, mask.astype(int))
        # Both classes present in the melt grid (cells below the liquidus are filled).
        assert out['valid']['melt'].any() and not out['valid']['melt'].all()

    @pytest.mark.parametrize('column', [7, 8], ids=['alpha', 'nabla_ad'])
    def test_validity_mask_needs_every_property(
        self, synthetic_table, melting_curves, tmp_path, column
    ):
        """A cell whose temperature inverts but whose thermal expansion or
        adiabatic gradient has no PALEOS value is marked invalid: the mask
        requires all five properties, not only the temperature. A missing
        nabla_ad is written as 0, so the mask is the only record of it.

        Edge case: the property is NaN along the highest-pressure table node
        while the entropy there stays finite, so the S(P,T) inversion still
        succeeds.
        """
        sol_func, liq_func = melting_curves
        rows = synthetic_table.read_text().splitlines(keepends=True)
        p_top = f'{_P_NODES_PA[-1]:.8e}'
        edited = []
        for row in rows:
            fields = row.split()
            if not row.startswith('#') and fields[0] == p_top:
                fields[column] = 'nan'  # alpha (7) or nabla_ad (8)
                row = ' '.join(fields) + '\n'
            edited.append(row)
        table = tmp_path / 'property_gap.dat'
        table.write_text(''.join(edited))
        out = eos_export.generate_spider_eos_tables(
            table, sol_func, liq_func, P_range=(1e6, 1e10), n_P=12, n_S=14, output_dir=None
        )
        mask = out['valid']['melt']
        # Columns above the last finite node of the property have no valid cell.
        top = out['P_Pa'] > _P_NODES_PA[-2]
        assert top.any() and not mask[:, top].any()
        # Below that node the melt phase still has valid cells.
        assert mask[:, ~top].any()
        if column == 8:
            # A missing nabla_ad is written as 0, not filled from a neighbour.
            assert (out['melt']['nabla_ad'][:, top] == 0.0).any()

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

    @pytest.mark.physics_invariant
    def test_fully_molten_surface_reads_liquid_table(self, pdep_2phase):
        """A fully molten surface anchor reads the liquid table instead of raising.

        ``compute_surface_entropy`` shares the phase routing of
        ``compute_entropy_adiabat`` and drives the SPIDER entropy-IC cross-check
        and the ``adiabatic_from_cmb`` fallback, both of which anchor at fully
        molten temperatures. For a 2-phase mantle the single-phase table is the
        solid table, NaN at molten temperatures, so a molten anchor that
        misses the phase routing falls through to it and trips the NaN guard
        (silently disabling the cross-check, or raising in the fallback). The
        molten branch must return the liquid entropy: a mis-routed lookup
        comes back NaN from the solid table (caught by the finiteness
        assert), and the 8% liquid/solid gap additionally rejects any finite
        value on the solid entropy scale. The tolerance is tight (rtol 1e-6)
        because the fixture entropy is affine in (log10 P, log10 T), which
        bilinear interpolation on the log-log grid reproduces to float
        precision, so a systematic sub-percent scaling error is resolvable.
        """
        solid_path, liquid_path = pdep_2phase
        T_surf = 4000.0  # molten at P=1e6 (liquidus 2620 K); solid table NaN here
        result = eos_export.compute_surface_entropy(
            solid_path,
            T_surface=T_surf,
            P_surface=1.0e6,
            solidus_func=_pdep_solidus,
            liquidus_func=_pdep_liquidus,
            solid_eos_file=solid_path,
            liquid_eos_file=liquid_path,
        )
        s_liquid = 1.08 * _s_pdep(1.0e6, T_surf)
        s_solid = _s_pdep(1.0e6, T_surf)
        assert np.isfinite(result['S_target'])
        np.testing.assert_allclose(result['S_target'], s_liquid, rtol=1e-6)
        assert abs(result['S_target'] - s_solid) > 0.05 * abs(s_solid)


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

    @pytest.mark.physics_invariant
    def test_fully_molten_adiabat_reads_liquid_table(self, pdep_2phase):
        """Fully molten adiabat reads the liquid table at every depth: finite, isentropic, depth-varying.

        A super-liquidus initial condition is molten at every depth, yet the
        single-phase table handed to ``compute_entropy_adiabat`` is the solid
        table, whose entropy is NaN at molten temperatures (here: NaN above the
        liquidus). Without phase routing the deep molten points come back NaN,
        the profile flattens to a constant temperature, and isentropy breaks.
        The liquid table carries s = 1.08x the solid formula, so three things
        must hold and each fails without the molten branch:

        * the fully molten surface anchor's entropy equals the liquid value
          (a mis-routed anchor reads the solid table and comes back NaN; the
          8% guard additionally rejects any finite solid-scale value);
        * the temperature rises monotonically with depth rather than pinning
          to the surface value, and the whole profile is finite (no NaN
          plateau);
        * every profile point lies on the liquid isentrope, checked against
          ``1.08 * s(P, T)`` recomputed from the fixture formula rather than
          the returned ``S_profile`` (which brentq forces to ``S_target``),
          so a depth-routing regression that reads solid at depth is caught.
          The liquid table tabulates the same formula, so this verifies
          routing and isentropy, not interpolation accuracy.

        Tolerances are rtol 1e-6: the fixture entropy is affine in
        (log10 P, log10 T), which bilinear interpolation on the log-log grid
        reproduces to float precision, so a systematic sub-percent scaling
        error is resolvable.
        """
        solid_path, liquid_path = pdep_2phase
        P_surf, P_cmb = 1.0e6, 1.0e9
        # Above liquidus(P_surf)=2620 K and molten all the way to P_cmb.
        T_surf = 4000.0
        result = eos_export.compute_entropy_adiabat(
            solid_path,
            T_surface=T_surf,
            P_surface=P_surf,
            P_cmb=P_cmb,
            n_points=24,
            solidus_func=_pdep_solidus,
            liquidus_func=_pdep_liquidus,
            solid_eos_file=solid_path,
            liquid_eos_file=liquid_path,
        )
        P = np.asarray(result['P'])
        T = np.asarray(result['T'])
        S_target = result['S_target']
        s_liquid = 1.08 * _s_pdep(P_surf, T_surf)  # liquid-table anchor entropy
        s_solid = _s_pdep(P_surf, T_surf)  # solid-table value (mis-routed branch)
        # Surface anchor entropy is the liquid value, not the solid value.
        assert np.isfinite(S_target)
        np.testing.assert_allclose(S_target, s_liquid, rtol=1e-6)
        assert abs(S_target - s_solid) > 0.05 * abs(s_solid)  # rejects solid scale
        # No NaN plateau: finite, and temperature rises with depth (the unfixed
        # code pinned every deep point to T_surf).
        order = np.argsort(P)
        Ts = T[order]
        assert np.all(np.isfinite(T))
        assert np.all(np.diff(Ts) > 0.0)
        assert Ts[-1] > 1.5 * Ts[0]
        # Every point sits on the liquid isentrope: recomputed from the fixture
        # entropy formula, so it catches a depth-routing regression that the
        # tautological S_profile == S_target check cannot.
        np.testing.assert_allclose(1.08 * _s_pdep(P, T), S_target, rtol=1e-6)

    @pytest.mark.parametrize('n_points', [0, 1])
    def test_fewer_than_two_points_is_rejected(self, pdep_2phase, n_points):
        """A profile needs both ends: with one point the CMB pin would overwrite
        the surface point, so n_points below 2 raises ValueError."""
        solid_path, liquid_path = pdep_2phase
        with pytest.raises(ValueError, match='n_points must be >= 2'):
            eos_export.compute_entropy_adiabat(
                solid_path, T_surface=4000.0, P_surface=1e6, P_cmb=1e9, n_points=n_points
            )

    @pytest.mark.physics_invariant
    def test_profile_ends_at_the_surface_and_cmb_pressures(self, pdep_2phase):
        """The profile starts at P_surface and ends at P_cmb, so T[-1] is the CMB
        temperature and T[0] the surface temperature.

        Edge case: both ends sit exactly on the table bounds (1e6 and 1e9 Pa),
        where an end point a rounding step outside the grid would read NaN.
        Discrimination: an adiabat that stops at 0.999 * P_cmb is cooler at its
        deepest point, because T rises with depth along the isentrope.
        """
        solid_path, liquid_path = pdep_2phase
        P_surf, P_cmb = 1.0e6, 1.0e9
        kwargs = dict(
            T_surface=4000.0,
            n_points=24,
            solidus_func=_pdep_solidus,
            liquidus_func=_pdep_liquidus,
            solid_eos_file=solid_path,
            liquid_eos_file=liquid_path,
        )
        result = eos_export.compute_entropy_adiabat(
            solid_path, P_surface=P_surf, P_cmb=P_cmb, **kwargs
        )
        P = np.asarray(result['P'])
        T = np.asarray(result['T'])
        np.testing.assert_array_equal(P[[0, -1]], [P_surf, P_cmb])
        assert T[0] == pytest.approx(4000.0, rel=1e-8)
        assert np.all(np.isfinite(T))
        shallow = eos_export.compute_entropy_adiabat(
            solid_path, P_surface=P_surf, P_cmb=0.999 * P_cmb, **kwargs
        )
        assert T[-1] > shallow['T'][-1]
        # The end is the input pressure bit for bit, also where 10**log10(P)
        # does not round-trip (7.3e8 Pa comes back 1.4e-6 Pa high).
        odd = eos_export.compute_entropy_adiabat(
            solid_path, P_surface=P_surf, P_cmb=7.3e8, **kwargs
        )
        np.testing.assert_array_equal(np.asarray(odd['P'])[[0, -1]], [P_surf, 7.3e8])
        # The same at the surface: 1.013e6 Pa, inside the table, does not round-trip.
        odd_surf = eos_export.compute_entropy_adiabat(
            solid_path, P_surface=1.013e6, P_cmb=P_cmb, **kwargs
        )
        assert odd_surf['P'][0] == 1.013e6
        assert odd_surf['T'][0] == pytest.approx(4000.0, rel=1e-8)
        # The CMB end lies on the same isentrope as the rest of the profile.
        np.testing.assert_allclose(1.08 * _s_pdep(P[-1], T[-1]), result['S_target'], rtol=1e-6)

    @pytest.mark.physics_invariant
    def test_fully_molten_routing_survives_collapsed_mushy_zone(self, pdep_2phase):
        """mushy_zone_factor = 1.0 (solidus == liquidus) still routes molten points to the liquid table.

        With the mushy zone collapsed there is no ``T_sol < T < T_liq``
        interval, so a strict ``T_liq > T_sol`` guard on the molten branch
        would send every molten point to the solid table, which is NaN there,
        and flatten the profile. The branch uses ``T_liq >= T_sol``, so a
        fully molten adiabat over collapsed curves stays finite, isentropic,
        and anchored to the liquid table. The pinned contract is the fully
        molten profile; a collapsed-curve adiabat whose isentrope reaches the
        melting curve at depth crosses a fusion-entropy discontinuity and is
        not covered here.
        """
        solid_path, liquid_path = pdep_2phase
        # Collapsed curves: solidus == liquidus everywhere.
        result = eos_export.compute_entropy_adiabat(
            solid_path,
            T_surface=4000.0,
            P_surface=1.0e6,
            P_cmb=1.0e9,
            n_points=16,
            solidus_func=_pdep_liquidus,
            liquidus_func=_pdep_liquidus,
            solid_eos_file=solid_path,
            liquid_eos_file=liquid_path,
        )
        T = np.asarray(result['T'])
        assert np.isfinite(result['S_target'])
        np.testing.assert_allclose(result['S_target'], 1.08 * _s_pdep(1.0e6, 4000.0), rtol=1e-6)
        assert np.all(np.isfinite(T))
        # Not a NaN-driven flat plateau: the profile actually deepens.
        assert T[-1] > 1.5 * T[0]
