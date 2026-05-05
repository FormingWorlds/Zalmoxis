"""Branch tests for ``zalmoxis.eos.tdep`` covering edge cases that the
data-file-driven WB2018/RTPress integration tests do not reach: the
``None``-density propagation in the mushy mantle path and the
``nabla_ad`` per-cell clamp + NaN-fallback paths.
"""

from __future__ import annotations

import logging
from unittest import mock

import numpy as np
import pytest

from zalmoxis.eos import tdep as _tdep
from zalmoxis.eos.tdep import _get_paleos_nabla_ad, get_Tdep_density

pytestmark = pytest.mark.unit


def _solidus(P):
    """Synthetic solidus: T_sol = 2000 + 1e-9 * P (K, P in Pa)."""
    return 2000.0 + 1e-9 * P


def _liquidus(P):
    """Synthetic liquidus: T_liq = 3000 + 1e-9 * P (K, P in Pa)."""
    return 3000.0 + 1e-9 * P


def _outside_solidus(P):
    """Both solidus and liquidus return NaN (outside melting-curve range)."""
    return np.nan


class TestGetTdepDensityMushyNone:
    """The mushy-zone path returns ``None`` if either solid or liquid
    density lookup fails (out-of-bounds pressure)."""

    def test_solid_density_none_returns_none(self):
        material = {
            'melted_mantle': {'eos_file': '/dev/null'},
            'solid_mantle': {'eos_file': '/dev/null'},
        }
        # Force solid lookup to None, liquid to a normal value
        with mock.patch(
            'zalmoxis.eos.tdep.get_tabulated_eos',
            side_effect=lambda *a, **kw: None if a[2] == 'solid_mantle' else 5000.0,
        ):
            rho = get_Tdep_density(
                pressure=1e10,
                temperature=2500.0,  # mushy: between 2010 and 3010
                material_properties_iron_Tdep_silicate_planets=material,
                solidus_func=_solidus,
                liquidus_func=_liquidus,
            )
        assert rho is None

    def test_liquid_density_none_returns_none(self):
        material = {
            'melted_mantle': {'eos_file': '/dev/null'},
            'solid_mantle': {'eos_file': '/dev/null'},
        }
        with mock.patch(
            'zalmoxis.eos.tdep.get_tabulated_eos',
            side_effect=lambda *a, **kw: None if a[2] == 'melted_mantle' else 5000.0,
        ):
            rho = get_Tdep_density(
                pressure=1e10,
                temperature=2500.0,
                material_properties_iron_Tdep_silicate_planets=material,
                solidus_func=_solidus,
                liquidus_func=_liquidus,
            )
        assert rho is None

    def test_undefined_curve_falls_to_solid(self):
        """When solidus/liquidus return NaN, the function defaults to the
        solid lookup (handled at the top of the function)."""
        material = {'solid_mantle': {'eos_file': '/dev/null'}}
        with mock.patch(
            'zalmoxis.eos.tdep.get_tabulated_eos',
            return_value=4500.0,
        ) as mocked:
            rho = get_Tdep_density(
                pressure=1e15,
                temperature=2500.0,
                material_properties_iron_Tdep_silicate_planets=material,
                solidus_func=_outside_solidus,
                liquidus_func=_outside_solidus,
            )
        assert rho == 4500.0
        # Confirm only the solid-mantle branch was called
        assert mocked.call_args[0][2] == 'solid_mantle'


class TestGetTdepDensitySolidLiquidPure:
    """Pure solid (T <= T_sol) and pure liquid (T >= T_liq) branches."""

    def test_pure_solid_branch(self):
        material = {'solid_mantle': {'eos_file': '/dev/null'}}
        with mock.patch(
            'zalmoxis.eos.tdep.get_tabulated_eos',
            return_value=4500.0,
        ) as mocked:
            rho = get_Tdep_density(
                pressure=1e10,
                temperature=1500.0,  # below solidus T_sol=2010
                material_properties_iron_Tdep_silicate_planets=material,
                solidus_func=_solidus,
                liquidus_func=_liquidus,
            )
        assert rho == 4500.0
        assert mocked.call_args[0][2] == 'solid_mantle'

    def test_pure_liquid_branch(self):
        material = {'melted_mantle': {'eos_file': '/dev/null'}}
        with mock.patch(
            'zalmoxis.eos.tdep.get_tabulated_eos',
            return_value=4500.0,
        ) as mocked:
            rho = get_Tdep_density(
                pressure=1e10,
                temperature=4000.0,  # above liquidus T_liq=3010
                material_properties_iron_Tdep_silicate_planets=material,
                solidus_func=_solidus,
                liquidus_func=_liquidus,
            )
        assert rho == 4500.0
        assert mocked.call_args[0][2] == 'melted_mantle'

    def test_missing_curves_raises(self):
        """Solidus/liquidus must be supplied for T-dep materials."""
        material = {'solid_mantle': {'eos_file': '/dev/null'}}
        with pytest.raises(ValueError, match='solidus_func'):
            get_Tdep_density(
                pressure=1e10,
                temperature=2500.0,
                material_properties_iron_Tdep_silicate_planets=material,
                solidus_func=None,
                liquidus_func=None,
            )


def _build_paleos_cache(*, with_nan_corner: bool = False):
    """Build a tiny synthetic PALEOS cache for nabla_ad lookups."""
    n_p = n_t = 8
    log_p = np.linspace(9.0, 12.0, n_p)
    log_t = np.linspace(3.0, 4.0, n_t)
    nabla_ad_grid = 0.25 * np.ones((n_p, n_t))
    if with_nan_corner:
        nabla_ad_grid[0, 0] = np.nan

    def _nn_nabla(point):
        if hasattr(point, '__len__') and len(point) == 2 and np.ndim(point[0]) == 0:
            return 0.25  # constant fallback
        pts = np.atleast_2d(np.asarray(point, dtype=float))
        return np.full(len(pts), 0.25)

    return {
        'type': 'paleos',
        'nabla_ad_grid': nabla_ad_grid,
        'nabla_ad_nn': _nn_nabla,
        'nabla_ad_interp': lambda pt: float(
            nabla_ad_grid[
                int(np.clip(round((pt[0] - log_p[0]) / (log_p[1] - log_p[0])), 0, n_p - 1)),
                int(np.clip(round((pt[1] - log_t[0]) / (log_t[1] - log_t[0])), 0, n_t - 1)),
            ]
        ),
        'p_min': 10 ** log_p[0],
        'p_max': 10 ** log_p[-1],
        't_min': 10 ** log_t[0],
        't_max': 10 ** log_t[-1],
        'unique_log_p': log_p,
        'unique_log_t': log_t,
        'logt_valid_min': np.full(n_p, log_t[0]),
        'logt_valid_max': np.full(n_p, log_t[-1]),
        'logp_min': log_p[0],
        'logt_min': log_t[0],
        'dlog_p': log_p[1] - log_p[0],
        'dlog_t': log_t[1] - log_t[0],
        'n_p': n_p,
        'n_t': n_t,
    }


class TestPaleosNablaAdScalar:
    """``_get_paleos_nabla_ad`` per-cell clamp warning and NaN fallback."""

    def test_clamp_warning_emitted_once(self, caplog):
        eos_file = '/synthetic/nabla.dat'
        cache = {eos_file: _build_paleos_cache()}
        cache[eos_file]['logt_valid_max'] = np.full(8, 3.5)  # tighten so 9000 K clamps
        material = {'mantle': {'eos_file': eos_file}}
        # Reset the warned-set so we cleanly observe the first emission.
        _tdep._paleos_clamp_warned.discard(eos_file)
        with caplog.at_level(logging.WARNING, logger='zalmoxis.eos.tdep'):
            val = _get_paleos_nabla_ad(1e10, 9000.0, material, 'mantle', cache)
            assert any('clamping' in r.message for r in caplog.records)
        assert val == pytest.approx(0.25)

    def test_nan_fallback_returns_finite(self):
        """When ``nabla_ad_interp`` returns NaN, the NN fallback recovers."""
        eos_file = '/synthetic/nabla_nan.dat'
        cache = {eos_file: _build_paleos_cache(with_nan_corner=True)}
        material = {'mantle': {'eos_file': eos_file}}
        # Query at the NaN corner — interp returns NaN, NN returns 0.25
        val = _get_paleos_nabla_ad(
            10 ** cache[eos_file]['logp_min'],
            10 ** cache[eos_file]['logt_min'],
            material,
            'mantle',
            cache,
        )
        assert val == pytest.approx(0.25)

    def test_returns_none_when_both_fail(self):
        """If both interp and NN return NaN the function returns None."""
        eos_file = '/synthetic/nabla_dead.dat'
        cache = {eos_file: _build_paleos_cache(with_nan_corner=True)}

        # Override NN to also return NaN
        def _nan_nn(point):
            if hasattr(point, '__len__') and len(point) == 2 and np.ndim(point[0]) == 0:
                return np.nan
            pts = np.atleast_2d(np.asarray(point, dtype=float))
            return np.full(len(pts), np.nan)

        cache[eos_file]['nabla_ad_nn'] = _nan_nn
        material = {'mantle': {'eos_file': eos_file}}
        val = _get_paleos_nabla_ad(
            10 ** cache[eos_file]['logp_min'],
            10 ** cache[eos_file]['logt_min'],
            material,
            'mantle',
            cache,
        )
        assert val is None
