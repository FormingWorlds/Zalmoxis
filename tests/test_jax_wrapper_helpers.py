"""Helper-function and error-path tests for ``zalmoxis.jax_eos.wrapper``.

The end-to-end JAX path is covered by ``test_jax_wrapper_end_to_end.py``.
This file targets the pure-Python helpers (``_extract_sub_args``,
``_extract_liquidus``, ``_tabulate_adiabat``) plus the early-error paths
in ``solve_structure_via_jax`` that raise before the JIT solve is reached.

Anti-happy-path: each test class includes ≥ 1 edge case and ≥ 1
physically unreasonable input.
"""

from __future__ import annotations

import numpy as np
import pytest

from zalmoxis.jax_eos.wrapper import (
    _extract_liquidus,
    _extract_sub_args,
    _tabulate_adiabat,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Cache-extraction helpers
# ---------------------------------------------------------------------------


def _make_cached(prefix=''):
    """Build a fully-populated paleos_unified-style cache dict."""
    return {
        'density_grid': np.linspace(3000.0, 13000.0, 12).reshape(3, 4),
        'unique_log_p': np.array([5.0, 8.0, 11.0]),
        'unique_log_t': np.array([3.0, 3.5, 4.0, 4.5]),
        'logp_min': 5.0,
        'logt_min': 3.0,
        'dlog_p': 3.0,
        'dlog_t': 0.5,
        'n_p': 3,
        'n_t': 4,
        'p_min': 1e5,
        'p_max': 1e11,
        'logt_valid_min': np.array([3.0, 3.0, 3.0]),
        'logt_valid_max': np.array([4.5, 4.5, 4.5]),
        'liquidus_log_p': np.array([5.0, 8.0, 11.0]),
        'liquidus_log_t': np.array([3.3, 3.7, 4.0]),
    }


class TestExtractSubArgs:
    """``_extract_sub_args`` flattens a paleos cache entry into JAX kwargs."""

    def test_keys_are_prefixed(self):
        """Every output key carries the supplied ``prefix`` token."""
        cached = _make_cached()
        out = _extract_sub_args(cached, 'core')
        for key in out:
            assert key.startswith('core_')
        # Must include the canonical density grid + axes.
        assert 'core_density_grid' in out
        assert 'core_unique_log_p' in out
        assert 'core_unique_log_t' in out

    def test_scalar_fields_are_python_floats(self):
        """Numpy scalars are coerced to Python ``float`` for jax.Tracer compat.

        Discriminating: the cache stores np.float64 values; the extractor
        must return native floats so the JAX tracer doesn't accidentally
        treat them as jnp arrays.
        """
        cached = _make_cached()
        cached['logp_min'] = np.float64(5.0)  # force np scalar
        out = _extract_sub_args(cached, 'core')
        assert isinstance(out['core_logp_min'], float)
        assert isinstance(out['core_n_p'], int)

    def test_repeat_call_returns_cached_dict(self):
        """Second call hits the in-place cache and returns the same dict object.

        Edge: ensures the cached extraction is reused (hot path optimisation).
        """
        cached = _make_cached()
        out1 = _extract_sub_args(cached, 'mantle')
        out2 = _extract_sub_args(cached, 'mantle')
        assert out1 is out2  # identity, not equality

    def test_distinct_prefixes_get_distinct_caches(self):
        """Edge: switching ``prefix`` must produce a fresh dict, not the old one."""
        cached = _make_cached()
        out_core = _extract_sub_args(cached, 'core')
        out_sol = _extract_sub_args(cached, 'sol')
        # Different keys, different identities.
        assert out_core is not out_sol
        assert any(k.startswith('core_') for k in out_core)
        assert any(k.startswith('sol_') for k in out_sol)


class TestExtractLiquidus:
    """``_extract_liquidus`` flattens liquidus curve data into JAX kwargs."""

    def test_populated_liquidus_writes_min_max_and_flag(self):
        """When liquidus arrays are non-empty, min/max/flag fields are set.

        Discriminating: log_p endpoints are 5.0 and 11.0, asymmetric, so
        any swap of min/max is detectable.
        """
        cached = _make_cached()
        out = _extract_liquidus(cached, 'core')
        assert out['core_liquidus_min_log_p'] == pytest.approx(5.0)
        assert out['core_liquidus_max_log_p'] == pytest.approx(11.0)
        assert out['core_has_liquidus_f'] == pytest.approx(1.0)

    def test_missing_liquidus_uses_zero_sentinels(self):
        """Edge: cache without liquidus arrays returns has_liquidus_f=0.

        Forces the empty-array branch in the helper.
        """
        cached = _make_cached()
        cached['liquidus_log_p'] = np.array([])
        cached['liquidus_log_t'] = np.array([])
        out = _extract_liquidus(cached, 'core')
        assert out['core_has_liquidus_f'] == pytest.approx(0.0)
        assert out['core_liquidus_min_log_p'] == pytest.approx(0.0)
        assert out['core_liquidus_max_log_p'] == pytest.approx(0.0)

    def test_completely_absent_keys_handled_via_get_default(self):
        """Edge: cache that never set 'liquidus_log_p' at all -> empty defaults."""
        cached = _make_cached()
        del cached['liquidus_log_p']
        del cached['liquidus_log_t']
        out = _extract_liquidus(cached, 'core')
        assert out['core_has_liquidus_f'] == pytest.approx(0.0)

    def test_repeat_call_returns_cached_dict(self):
        """Second call hits the cache (mirrors _extract_sub_args)."""
        cached = _make_cached()
        out1 = _extract_liquidus(cached, 'core')
        out2 = _extract_liquidus(cached, 'core')
        assert out1 is out2


# ---------------------------------------------------------------------------
# _tabulate_adiabat
# ---------------------------------------------------------------------------


class TestTabulateAdiabat:
    """Sample temperature_function on a log-P grid for the JAX RHS."""

    def test_returns_log_p_grid_and_T_values_with_matching_length(self):
        """Default n_pts=4000 yields two same-length 1-D arrays."""

        def temperature_function(r, P):
            # Realistic adiabat: T = 2000 K * (P / 1e5)^0.25
            return 2000.0 * (max(P, 1.0) / 1e5) ** 0.25

        radii = np.linspace(0.0, 6.371e6, 200)
        log_p_grid, T_values = _tabulate_adiabat(radii, temperature_function)
        assert log_p_grid.shape == (4000,)
        assert T_values.shape == (4000,)
        assert log_p_grid[0] == pytest.approx(5.0)
        assert log_p_grid[-1] == pytest.approx(13.0)
        # Discriminating values: T at end-points must match the closed form.
        np.testing.assert_allclose(T_values[0], 2000.0, rtol=1e-12)
        np.testing.assert_allclose(T_values[-1], 2000.0 * (1e13 / 1e5) ** 0.25, rtol=1e-12)

    def test_explicit_n_pts_respected(self):
        """Edge: ``n_pts=10`` produces 10 grid points, not the default."""

        def temperature_function(r, P):
            return 1500.0

        radii = np.linspace(0.0, 6.371e6, 50)
        log_p_grid, T_values = _tabulate_adiabat(radii, temperature_function, n_pts=10)
        assert log_p_grid.shape == (10,)
        assert T_values.shape == (10,)

    def test_temperature_function_called_at_midpoint_radius(self):
        """The sampler uses r_mid = 0.5 * (r[0] + r[-1]).

        Discriminating: a temperature_function that is r-dependent must
        evaluate at the midpoint, not at r=0 or r=R.
        """
        captured = []

        def temperature_function(r, P):
            captured.append(r)
            return 1500.0

        radii = np.array([0.0, 1e7])  # midpoint = 5e6
        _tabulate_adiabat(radii, temperature_function, n_pts=3)
        # All three samples carried r = 5e6.
        for r in captured:
            assert r == pytest.approx(5e6, rel=1e-12)

    def test_constant_temperature_function_returns_constant_T_values(self):
        """Edge: T_func returning a constant gives a constant ``T_values`` array."""

        def temperature_function(r, P):
            return 2500.0  # not 1.0 (per discriminating-value rule)

        radii = np.linspace(0.0, 6.371e6, 50)
        _, T_values = _tabulate_adiabat(radii, temperature_function, n_pts=20)
        np.testing.assert_allclose(T_values, 2500.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# solve_structure_via_jax pre-JIT format-rejection raises
# ---------------------------------------------------------------------------


class _FakeMixture:
    """Minimum LayerMixture surface for the wrapper's `core_lm.components[0]`."""

    def __init__(self, eos_id):
        self.components = [eos_id]


class TestSolveStructureViaJaxFormatChecks:
    """Pre-JIT-solve raises in ``solve_structure_via_jax``.

    The wrapper rejects a non-paleos_unified core or a mantle that lacks
    the 2-phase sub-tables before instantiating any caches or calling
    diffrax. These guards keep PROTEUS' coupled run from silently routing
    a Tdep / Seager / Stixrude config through the JAX path (where the
    specialised RHS would either crash or return junk) — they are the
    only typed contract enforcement on the JAX side.
    """

    @staticmethod
    def _common_args(*, core_format='paleos_unified'):
        """Smallest valid arg set that reaches the format check.

        ``core_mat`` carries the parametrised format so a wrong value
        triggers the line-198 ValueError before any cache is built. The
        rest is dummy data the wrapper does not read until past the raise.
        """
        from zalmoxis.jax_eos.wrapper import solve_structure_via_jax

        radii = np.linspace(0.0, 6.371e6, 5)
        layer_mixtures = {
            'core': _FakeMixture('Tdep:iron'),
            'mantle': _FakeMixture('Tdep:MgSiO3'),
        }
        material_dictionaries = {
            'Tdep:iron': {'format': core_format, 'eos_file': '/dev/null'},
            'Tdep:MgSiO3': {'format': 'paleos_unified', 'eos_file': '/dev/null'},
        }
        kwargs = dict(
            cmb_mass=0.325 * 5.972e24,
            core_mantle_mass=5.972e24,
            radii=radii,
            adaptive_radial_fraction=1.0,
            relative_tolerance=1e-6,
            absolute_tolerance=1e-6,
            maximum_step=1.0,
            material_dictionaries=material_dictionaries,
            interpolation_cache={},
            y0=[0.0, 0.0, 360e9],
            solidus_func=lambda P: 1500.0,
            liquidus_func=lambda P: 4500.0,
        )
        return solve_structure_via_jax, layer_mixtures, kwargs

    def test_non_paleos_unified_core_format_raises_value_error(self):
        """A core material whose format is not ``paleos_unified`` is rejected.

        Discriminating: passes ``'Tdep'`` (a real Zalmoxis format) so the
        guard is exercised against a plausible mis-routing rather than a
        nonsense string. The error message must contain the offending
        format token so the operator can find it in logs.
        """
        solve, layer_mixtures, kwargs = self._common_args(core_format='Tdep')
        with pytest.raises(ValueError, match='paleos_unified'):
            solve(layer_mixtures=layer_mixtures, **kwargs)

    def test_unknown_core_format_string_is_rejected(self):
        """Edge: an obviously-unknown format string still trips the same raise.

        Anti-happy-path: empty string + arbitrary token both must raise,
        so the guard is `format != 'paleos_unified'` (strict equality)
        rather than substring or whitelist drift.
        """
        for bad_format in ('', 'unknown_format', 'PALEOS_UNIFIED'):
            solve, layer_mixtures, kwargs = self._common_args(core_format=bad_format)
            with pytest.raises(ValueError, match='paleos_unified'):
                solve(layer_mixtures=layer_mixtures, **kwargs)
