"""Branch tests for ``zalmoxis.jax_eos.wrapper.solve_structure_via_jax``.

The end-to-end parity test in ``test_jax_wrapper_end_to_end.py`` exercises
the happy path on a Stage-1b 2-layer config. This file pins the
validation, fallback, and post-event-padding branches that the parity
test does not discriminate against:

- core format must be ``paleos_unified`` (raises ValueError otherwise)
- mantle must be PALEOS-2phase (solid_mantle + melted_mantle)
- ``temperature_arrays`` shape validation
- ``temperature_function=None`` default-fallback path (3000 K)
- post-terminal-event padding when diffrax returns ``inf`` past zero P
"""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from zalmoxis.jax_eos import wrapper as jw
from zalmoxis.mixing import LayerMixture

pytestmark = pytest.mark.unit


def _build_min_cache_dict():
    """Tiny synthetic unified-PALEOS cache, just enough for the wrapper's
    extract-and-pass-through. The actual numerical content is irrelevant
    because we mock ``solve_structure_jax``; we only need the keys."""
    n_p = n_t = 4
    log_p = np.linspace(9.0, 12.0, n_p)
    log_t = np.linspace(3.0, 4.0, n_t)
    grid = np.full((n_p, n_t), 5000.0)
    return {
        'type': 'paleos_unified',
        'density_grid': grid,
        'unique_log_p': log_p,
        'unique_log_t': log_t,
        'logp_min': log_p[0],
        'logt_min': log_t[0],
        'dlog_p': log_p[1] - log_p[0],
        'dlog_t': log_t[1] - log_t[0],
        'n_p': n_p,
        'n_t': n_t,
        'p_min': 10.0 ** log_p[0],
        'p_max': 10.0 ** log_p[-1],
        'logt_valid_min': np.full(n_p, log_t[0]),
        'logt_valid_max': np.full(n_p, log_t[-1]),
        'liquidus_log_p': log_p,
        'liquidus_log_t': np.linspace(np.log10(2000.0), np.log10(5000.0), n_p),
    }


def _common_fixtures(*, core_format='paleos_unified', mantle_2phase=True):
    """Minimal layer_mixtures + material_dictionaries + interpolation_cache
    for invoking ``solve_structure_via_jax`` with mocked downstream calls."""
    core_eos = 'PALEOS:iron'
    mantle_eos = 'PALEOS-2phase:MgSiO3'

    layer_mixtures = {
        'core': LayerMixture([core_eos], [1.0]),
        'mantle': LayerMixture([mantle_eos], [1.0]),
    }

    core_cache_file = '/synthetic/core.dat'
    sol_cache_file = '/synthetic/sol.dat'
    liq_cache_file = '/synthetic/liq.dat'

    core_mat = {
        'eos_file': core_cache_file,
        'format': core_format,
        '_api_resolved': True,
    }
    if mantle_2phase:
        mantle_mat = {
            'format': 'paleos_2phase',
            '_api_resolved': True,
            'solid_mantle': {'eos_file': sol_cache_file, 'format': 'paleos_unified'},
            'melted_mantle': {'eos_file': liq_cache_file, 'format': 'paleos_unified'},
        }
    else:
        mantle_mat = {'format': 'paleos_unified', '_api_resolved': True}

    material_dictionaries = {core_eos: core_mat, mantle_eos: mantle_mat}
    interpolation_cache = {
        core_cache_file: _build_min_cache_dict(),
        sol_cache_file: _build_min_cache_dict(),
        liq_cache_file: _build_min_cache_dict(),
    }
    return layer_mixtures, material_dictionaries, interpolation_cache


def _liquidus_func(P):
    return 3000.0 + 1e-9 * P


def _solidus_func(P):
    return 0.7 * (3000.0 + 1e-9 * P)


def _t_func(r, P):
    return 3000.0


class TestCoreFormatValidation:
    """Core EOS must be ``paleos_unified`` for the JAX path."""

    def test_non_paleos_unified_core_raises(self):
        layer_mixtures, mds, cache = _common_fixtures(core_format='Seager2007')
        radii = np.linspace(1.0, 1e6, 20)
        with pytest.raises(ValueError, match='paleos_unified'):
            jw.solve_structure_via_jax(
                layer_mixtures=layer_mixtures,
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                radii=radii,
                adaptive_radial_fraction=0.5,
                relative_tolerance=1e-6,
                absolute_tolerance=1e-8,
                maximum_step=1e5,
                material_dictionaries=mds,
                interpolation_cache=cache,
                y0=[0.0, 0.0, 1e12],
                solidus_func=_solidus_func,
                liquidus_func=_liquidus_func,
                temperature_function=_t_func,
            )


class TestMantleFormatValidation:
    """Mantle must be PALEOS-2phase (solid_mantle + melted_mantle keys)."""

    def test_non_2phase_mantle_raises(self):
        layer_mixtures, mds, cache = _common_fixtures(mantle_2phase=False)
        radii = np.linspace(1.0, 1e6, 20)
        with pytest.raises(ValueError, match='PALEOS-2phase'):
            jw.solve_structure_via_jax(
                layer_mixtures=layer_mixtures,
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                radii=radii,
                adaptive_radial_fraction=0.5,
                relative_tolerance=1e-6,
                absolute_tolerance=1e-8,
                maximum_step=1e5,
                material_dictionaries=mds,
                interpolation_cache=cache,
                y0=[0.0, 0.0, 1e12],
                solidus_func=_solidus_func,
                liquidus_func=_liquidus_func,
                temperature_function=_t_func,
            )


class TestTemperatureArraysValidation:
    """``temperature_arrays`` must be two 1-D arrays of equal length."""

    def test_shape_mismatch_raises(self):
        layer_mixtures, mds, cache = _common_fixtures()
        radii = np.linspace(1.0, 1e6, 20)
        bad = (np.array([1.0, 2.0, 3.0]), np.array([300.0, 400.0]))  # mismatched
        with pytest.raises(ValueError, match='equal length'):
            jw.solve_structure_via_jax(
                layer_mixtures=layer_mixtures,
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                radii=radii,
                adaptive_radial_fraction=0.5,
                relative_tolerance=1e-6,
                absolute_tolerance=1e-8,
                maximum_step=1e5,
                material_dictionaries=mds,
                interpolation_cache=cache,
                y0=[0.0, 0.0, 1e12],
                solidus_func=_solidus_func,
                liquidus_func=_liquidus_func,
                temperature_arrays=bad,
            )

    def test_2d_array_raises(self):
        layer_mixtures, mds, cache = _common_fixtures()
        radii = np.linspace(1.0, 1e6, 20)
        # Both arrays 2-D and same shape; ndim != 1 must trip the guard.
        two_d = np.zeros((3, 3))
        with pytest.raises(ValueError, match='equal length'):
            jw.solve_structure_via_jax(
                layer_mixtures=layer_mixtures,
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                radii=radii,
                adaptive_radial_fraction=0.5,
                relative_tolerance=1e-6,
                absolute_tolerance=1e-8,
                maximum_step=1e5,
                material_dictionaries=mds,
                interpolation_cache=cache,
                y0=[0.0, 0.0, 1e12],
                solidus_func=_solidus_func,
                liquidus_func=_liquidus_func,
                temperature_arrays=(two_d, two_d),
            )


class TestTemperatureFallback:
    """Without ``temperature_arrays`` or ``temperature_function`` the wrapper
    uses a constant 3000 K fallback. Verify the JAX call receives a tabulated
    constant grid in that case."""

    def test_no_temperature_uses_3000K_constant(self):
        layer_mixtures, mds, cache = _common_fixtures()
        radii = np.linspace(1.0, 1e6, 20)

        captured = {}

        def fake_solve_jax(radii_arr, y0, **kwargs):
            captured['T_values'] = kwargs['T_values']
            captured['T_surface'] = kwargs['T_surface']
            captured['T_axis_is_radius'] = kwargs.get('T_axis_is_radius', False)
            return np.zeros((len(radii_arr), 3))

        with mock.patch.object(jw, 'solve_structure_jax', side_effect=fake_solve_jax):
            jw.solve_structure_via_jax(
                layer_mixtures=layer_mixtures,
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                radii=radii,
                adaptive_radial_fraction=0.5,
                relative_tolerance=1e-6,
                absolute_tolerance=1e-8,
                maximum_step=1e5,
                material_dictionaries=mds,
                interpolation_cache=cache,
                y0=[0.0, 0.0, 1e12],
                solidus_func=_solidus_func,
                liquidus_func=_liquidus_func,
                temperature_function=None,
                temperature_arrays=None,
            )
        # Constant 3000 K everywhere is the documented fallback contract
        assert np.all(captured['T_values'] == pytest.approx(3000.0))
        assert captured['T_surface'] == pytest.approx(3000.0)
        # Without arrays, the axis is the log-P grid, not radius
        assert captured['T_axis_is_radius'] is False


class TestTemperatureArraysPath:
    """When ``temperature_arrays`` is provided, T is r-indexed."""

    def test_arrays_set_T_axis_is_radius_true(self):
        layer_mixtures, mds, cache = _common_fixtures()
        radii = np.linspace(1.0, 1e6, 20)
        r_arr = np.linspace(1.0, 1e6, 30)
        T_arr = np.linspace(5000.0, 1500.0, 30)

        captured = {}

        def fake_solve_jax(radii_arr, y0, **kwargs):
            captured['T_axis_is_radius'] = kwargs.get('T_axis_is_radius', False)
            captured['T_axis_grid'] = kwargs['T_axis_grid']
            captured['T_values'] = kwargs['T_values']
            return np.zeros((len(radii_arr), 3))

        with mock.patch.object(jw, 'solve_structure_jax', side_effect=fake_solve_jax):
            jw.solve_structure_via_jax(
                layer_mixtures=layer_mixtures,
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                radii=radii,
                adaptive_radial_fraction=0.5,
                relative_tolerance=1e-6,
                absolute_tolerance=1e-8,
                maximum_step=1e5,
                material_dictionaries=mds,
                interpolation_cache=cache,
                y0=[0.0, 0.0, 1e12],
                solidus_func=_solidus_func,
                liquidus_func=_liquidus_func,
                temperature_arrays=(r_arr, T_arr),
            )
        assert captured['T_axis_is_radius'] is True
        np.testing.assert_array_equal(captured['T_axis_grid'], r_arr)
        np.testing.assert_array_equal(captured['T_values'], T_arr)


class TestPostEventPadding:
    """Diffrax returns ``inf`` for save-points past the pressure-zero terminal
    event. The wrapper rewrites that contract to numpy's: mass/gravity carry
    the last valid value, pressure is padded to 0."""

    def test_inf_past_event_replaced_with_holds(self):
        layer_mixtures, mds, cache = _common_fixtures()
        radii = np.linspace(1.0, 1e6, 10)

        ys = np.zeros((10, 3))
        ys[:5, 0] = np.linspace(0.0, 1e23, 5)
        ys[:5, 1] = np.linspace(0.0, 5.0, 5)
        ys[:5, 2] = np.linspace(1e12, 1e10, 5)
        # Past index 4 the event fired: diffrax pads with inf
        ys[5:, :] = np.inf

        with mock.patch.object(jw, 'solve_structure_jax', return_value=ys):
            mass, gravity, pressure = jw.solve_structure_via_jax(
                layer_mixtures=layer_mixtures,
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                radii=radii,
                adaptive_radial_fraction=0.5,
                relative_tolerance=1e-6,
                absolute_tolerance=1e-8,
                maximum_step=1e5,
                material_dictionaries=mds,
                interpolation_cache=cache,
                y0=[0.0, 0.0, 1e12],
                solidus_func=_solidus_func,
                liquidus_func=_liquidus_func,
                temperature_function=_t_func,
            )
        # All padded entries are finite (no leftover inf)
        assert np.all(np.isfinite(mass))
        assert np.all(np.isfinite(gravity))
        # Padded mass entries hold the last pre-event value
        assert mass[-1] == pytest.approx(mass[4])
        assert gravity[-1] == pytest.approx(gravity[4])
        # Padded pressure entries are exactly zero (numpy contract)
        assert pressure[-1] == 0.0
        assert pressure[5] == 0.0


class TestMushyZoneFactorDispatch:
    """``mushy_zone_factors`` is forwarded as the core's mzf, accepting both
    a per-EOS dict and a single float."""

    def test_dict_mzf_picks_core_value(self):
        layer_mixtures, mds, cache = _common_fixtures()
        radii = np.linspace(1.0, 1e6, 10)
        captured = {}

        def fake_solve_jax(radii_arr, y0, **kwargs):
            captured['mzf'] = kwargs['mushy_zone_factor_core']
            return np.zeros((len(radii_arr), 3))

        with mock.patch.object(jw, 'solve_structure_jax', side_effect=fake_solve_jax):
            jw.solve_structure_via_jax(
                layer_mixtures=layer_mixtures,
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                radii=radii,
                adaptive_radial_fraction=0.5,
                relative_tolerance=1e-6,
                absolute_tolerance=1e-8,
                maximum_step=1e5,
                material_dictionaries=mds,
                interpolation_cache=cache,
                y0=[0.0, 0.0, 1e12],
                solidus_func=_solidus_func,
                liquidus_func=_liquidus_func,
                temperature_function=_t_func,
                mushy_zone_factors={'PALEOS:iron': 0.42, 'other': 0.99},
            )
        assert captured['mzf'] == pytest.approx(0.42)

    def test_scalar_mzf_applies_to_core(self):
        layer_mixtures, mds, cache = _common_fixtures()
        radii = np.linspace(1.0, 1e6, 10)
        captured = {}

        def fake_solve_jax(radii_arr, y0, **kwargs):
            captured['mzf'] = kwargs['mushy_zone_factor_core']
            return np.zeros((len(radii_arr), 3))

        with mock.patch.object(jw, 'solve_structure_jax', side_effect=fake_solve_jax):
            jw.solve_structure_via_jax(
                layer_mixtures=layer_mixtures,
                cmb_mass=2e23,
                core_mantle_mass=4e23,
                radii=radii,
                adaptive_radial_fraction=0.5,
                relative_tolerance=1e-6,
                absolute_tolerance=1e-8,
                maximum_step=1e5,
                material_dictionaries=mds,
                interpolation_cache=cache,
                y0=[0.0, 0.0, 1e12],
                solidus_func=_solidus_func,
                liquidus_func=_liquidus_func,
                temperature_function=_t_func,
                mushy_zone_factors=0.55,
            )
        assert captured['mzf'] == pytest.approx(0.55)
