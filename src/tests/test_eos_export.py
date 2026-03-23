"""Tests for EOS export: SPIDER P-S tables and Aragog P-T tables from PALEOS.

Validates table format, grid regularity, value ranges, and the specific
bugs that were fixed:
- SPIDER: uniform P spacing (was log-spaced, caused segfault)
- SPIDER: dT/dP_s units (was nabla_ad dimensionless, caused CVode failure)
- SPIDER: solid entropy range covers melt maximum
- Aragog: full rectangular grid (was phase-filtered, caused hang)
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from zalmoxis.eos_export import (
    generate_aragog_pt_tables,
    generate_spider_eos_tables,
    generate_spider_phase_boundaries,
    load_paleos_all_properties,
)

# Skip if PALEOS data not available
_PALEOS_EOS = os.path.join(
    os.environ.get('ZALMOXIS_ROOT', ''),
    'data', 'EOS_PALEOS_MgSiO3_unified', 'paleos_mgsio3_eos_table_pt.dat',
)
_HAS_PALEOS = os.path.isfile(_PALEOS_EOS)

# Also check FWL_DATA path
if not _HAS_PALEOS:
    _fwl = os.environ.get('FWL_DATA', '')
    _PALEOS_EOS = os.path.join(
        _fwl, 'zalmoxis_eos', 'EOS_PALEOS_MgSiO3_unified',
        'paleos_mgsio3_eos_table_pt.dat',
    )
    _HAS_PALEOS = os.path.isfile(_PALEOS_EOS)

pytestmark = pytest.mark.skipif(not _HAS_PALEOS, reason='PALEOS EOS data not available')


def _get_melting_curves():
    """Get solidus/liquidus functions for tests."""
    from zalmoxis.melting_curves import get_solidus_liquidus_functions

    _, liq = get_solidus_liquidus_functions('Stixrude14-solidus', 'PALEOS-liquidus')

    def solidus(P):
        return liq(P) * 0.8  # mushy_zone_factor = 0.8

    return solidus, liq


# ═════════════════════════════════════════════════════════════════════
# SPIDER P-S table tests
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestSpiderPSTables:
    """SPIDER P-S EOS table format and content."""

    def test_uniform_pressure_spacing(self):
        """P grid must be uniformly spaced (SPIDER uses floor((P-Pmin)/dP))."""
        sol, liq = _get_melting_curves()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_spider_eos_tables(
                eos_file=_PALEOS_EOS,
                solidus_func=sol,
                liquidus_func=liq,
                P_range=(1e8, 50e9),
                n_P=50,
                n_S=50,
                output_dir=tmpdir,
            )
            # Read density_melt.dat and check P spacing
            data = np.loadtxt(os.path.join(tmpdir, 'density_melt.dat'), comments='#')
            P_unique = np.unique(data[:, 0])
            dP = np.diff(P_unique)
            # All dP should be equal (within floating point tolerance)
            assert np.allclose(dP, dP[0], rtol=1e-6), (
                f'P grid not uniform: dP range [{dP.min():.6e}, {dP.max():.6e}]'
            )

    def test_header_format(self):
        """Header must be: # HEAD NX NY, with HEAD=5."""
        sol, liq = _get_melting_curves()
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_spider_eos_tables(
                eos_file=_PALEOS_EOS,
                solidus_func=sol,
                liquidus_func=liq,
                P_range=(1e8, 50e9),
                n_P=30,
                n_S=30,
                output_dir=tmpdir,
            )
            with open(os.path.join(tmpdir, 'density_melt.dat')) as f:
                header = f.readline().strip()
            parts = header.replace('#', '').split()
            assert len(parts) == 3
            HEAD, NX, NY = int(parts[0]), int(parts[1]), int(parts[2])
            assert HEAD == 5
            assert NX == 30
            assert NY == 30

    def test_data_row_count(self):
        """Data rows must equal NX * NY."""
        sol, liq = _get_melting_curves()
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_spider_eos_tables(
                eos_file=_PALEOS_EOS,
                solidus_func=sol,
                liquidus_func=liq,
                P_range=(1e8, 50e9),
                n_P=20,
                n_S=20,
                output_dir=tmpdir,
            )
            data = np.loadtxt(os.path.join(tmpdir, 'density_melt.dat'), comments='#')
            assert len(data) == 20 * 20

    def test_dTdPs_physical_range(self):
        """adiabat_temp_grad values must be in dT/dP_s range (~1e-9 to 1e-7 K/Pa)."""
        sol, liq = _get_melting_curves()
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_spider_eos_tables(
                eos_file=_PALEOS_EOS,
                solidus_func=sol,
                liquidus_func=liq,
                P_range=(1e8, 50e9),
                n_P=30,
                n_S=30,
                output_dir=tmpdir,
            )
            # Read header for scaling
            with open(os.path.join(tmpdir, 'adiabat_temp_grad_melt.dat')) as f:
                for _ in range(4):
                    f.readline()
                scales = [float(x) for x in f.readline().strip().replace('#', '').split()]
            data = np.loadtxt(
                os.path.join(tmpdir, 'adiabat_temp_grad_melt.dat'), comments='#'
            )
            vals_SI = data[:, 2] * scales[2]
            valid = vals_SI[vals_SI > 0]
            # dT/dP_s should be ~1e-9 to 1e-7 K/Pa (not ~0.3 dimensionless)
            assert valid.max() < 1e-5, f'dTdPs too large: {valid.max():.2e} (expected < 1e-5)'
            assert valid.min() > 1e-12, f'dTdPs too small: {valid.min():.2e}'

    def test_solid_entropy_covers_melt_range(self):
        """Solid phase S range must extend to at least melt phase S max."""
        sol, liq = _get_melting_curves()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_spider_eos_tables(
                eos_file=_PALEOS_EOS,
                solidus_func=sol,
                liquidus_func=liq,
                P_range=(1e8, 50e9),
                n_P=30,
                n_S=30,
                output_dir=tmpdir,
            )
            S_solid_max = result['S_solid'][-1]
            S_melt_max = result['S_melt'][-1]
            assert S_solid_max >= S_melt_max * 0.99, (
                f'Solid S max ({S_solid_max:.0f}) < melt S max ({S_melt_max:.0f})'
            )

    def test_all_files_produced(self):
        """All 10 SPIDER EOS files + 2 phase boundary files must be produced."""
        sol, liq = _get_melting_curves()
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_spider_eos_tables(
                eos_file=_PALEOS_EOS,
                solidus_func=sol,
                liquidus_func=liq,
                P_range=(1e8, 50e9),
                n_P=20,
                n_S=20,
                output_dir=tmpdir,
            )
            expected = [
                'density_melt.dat', 'density_solid.dat',
                'temperature_melt.dat', 'temperature_solid.dat',
                'heat_capacity_melt.dat', 'heat_capacity_solid.dat',
                'thermal_exp_melt.dat', 'thermal_exp_solid.dat',
                'adiabat_temp_grad_melt.dat', 'adiabat_temp_grad_solid.dat',
            ]
            for fname in expected:
                assert os.path.isfile(os.path.join(tmpdir, fname)), f'Missing: {fname}'

    def test_no_nan_in_tables(self):
        """No NaN or Inf values in any table."""
        sol, liq = _get_melting_curves()
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_spider_eos_tables(
                eos_file=_PALEOS_EOS,
                solidus_func=sol,
                liquidus_func=liq,
                P_range=(1e8, 50e9),
                n_P=20,
                n_S=20,
                output_dir=tmpdir,
            )
            for fname in os.listdir(tmpdir):
                if fname.endswith('.dat'):
                    data = np.loadtxt(os.path.join(tmpdir, fname), comments='#')
                    assert np.all(np.isfinite(data)), f'NaN/Inf in {fname}'


# ═════════════════════════════════════════════════════════════════════
# Aragog P-T table tests
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestAragogPTTables:
    """Aragog P-T EOS table format and content."""

    def test_full_rectangular_grid(self):
        """Both melt and solid files must have n_P * n_T rows (no gaps)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_aragog_pt_tables(
                eos_file=_PALEOS_EOS,
                P_range=(1e5, 50e9),
                n_P=50,
                n_T=50,
                output_dir=tmpdir,
            )
            for phase in ('melt', 'solid'):
                data = np.loadtxt(
                    os.path.join(tmpdir, f'density_{phase}.dat'), comments='#'
                )
                assert len(data) == 50 * 50, (
                    f'density_{phase}: {len(data)} rows, expected {50*50}'
                )

    def test_melt_equals_solid(self):
        """Melt and solid files must be identical (PALEOS is phase-agnostic)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_aragog_pt_tables(
                eos_file=_PALEOS_EOS,
                P_range=(1e5, 50e9),
                n_P=30,
                n_T=30,
                output_dir=tmpdir,
            )
            melt = np.loadtxt(os.path.join(tmpdir, 'density_melt.dat'), comments='#')
            solid = np.loadtxt(os.path.join(tmpdir, 'density_solid.dat'), comments='#')
            np.testing.assert_array_equal(melt, solid)

    def test_header_column_names(self):
        """Header column names must match Aragog _ScalingsParameters attributes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_aragog_pt_tables(
                eos_file=_PALEOS_EOS,
                P_range=(1e5, 10e9),
                n_P=10,
                n_T=10,
                output_dir=tmpdir,
            )
            # thermal_exp file must have header 'thermal_expansivity' (not 'thermal_exp')
            with open(os.path.join(tmpdir, 'thermal_exp_melt.dat')) as f:
                header = f.readline().strip()
            assert 'thermal_expansivity' in header, f'Bad header: {header}'

            # density file header
            with open(os.path.join(tmpdir, 'density_melt.dat')) as f:
                header = f.readline().strip()
            assert 'density' in header

    def test_all_files_produced(self):
        """All 6 Aragog P-T files must be produced."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_aragog_pt_tables(
                eos_file=_PALEOS_EOS,
                P_range=(1e5, 10e9),
                n_P=10,
                n_T=10,
                output_dir=tmpdir,
            )
            expected = [
                'density_melt.dat', 'density_solid.dat',
                'heat_capacity_melt.dat', 'heat_capacity_solid.dat',
                'thermal_exp_melt.dat', 'thermal_exp_solid.dat',
            ]
            for fname in expected:
                assert os.path.isfile(os.path.join(tmpdir, fname)), f'Missing: {fname}'

    def test_density_physical_range(self):
        """Density values must be in physical range (1000-10000 kg/m3 for silicate)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_aragog_pt_tables(
                eos_file=_PALEOS_EOS,
                P_range=(1e9, 50e9),
                n_P=20,
                n_T=20,
                output_dir=tmpdir,
            )
            data = np.loadtxt(os.path.join(tmpdir, 'density_melt.dat'), comments='#')
            rho = data[:, 2]
            valid = rho[rho > 1]
            assert valid.min() > 500, f'Density too low: {valid.min():.0f}'
            assert valid.max() < 15000, f'Density too high: {valid.max():.0f}'

    def test_generation_speed(self):
        """Table generation must complete in < 30 seconds for 200x200."""
        import time

        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_aragog_pt_tables(
                eos_file=_PALEOS_EOS,
                P_range=(1e5, 50e9),
                n_P=200,
                n_T=200,
                output_dir=tmpdir,
            )
        elapsed = time.time() - t0
        assert elapsed < 30, f'Table generation too slow: {elapsed:.1f}s (limit 30s)'


# ═════════════════════════════════════════════════════════════════════
# Phase boundary tests
# ═════════════════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPhaseBoundaries:
    """Phase boundary P-S conversion."""

    def test_solidus_below_liquidus(self):
        """Solidus entropy must be below liquidus at all pressures."""
        sol, liq = _get_melting_curves()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_spider_phase_boundaries(
                solidus_func=sol,
                liquidus_func=liq,
                eos_file=_PALEOS_EOS,
                P_range=(1e8, 50e9),
                n_P=100,
                output_dir=tmpdir,
            )
            sol_data = np.loadtxt(result['solidus_path'], comments='#')
            liq_data = np.loadtxt(result['liquidus_path'], comments='#')
            # At same P, S_solidus < S_liquidus
            # Both files have same P grid
            assert np.all(sol_data[:, 1] < liq_data[:, 1]), (
                'Solidus entropy not below liquidus everywhere'
            )

    def test_phase_boundary_header(self):
        """Phase boundary files must have correct 1D header format."""
        sol, liq = _get_melting_curves()
        with tempfile.TemporaryDirectory() as tmpdir:
            result = generate_spider_phase_boundaries(
                solidus_func=sol,
                liquidus_func=liq,
                eos_file=_PALEOS_EOS,
                P_range=(1e8, 50e9),
                n_P=50,
                output_dir=tmpdir,
            )
            with open(result['solidus_path']) as f:
                header = f.readline().strip()
            parts = header.replace('#', '').split()
            assert len(parts) == 2  # HEAD, NX (1D format)
            HEAD, NX = int(parts[0]), int(parts[1])
            assert HEAD == 5
            assert NX == 50
