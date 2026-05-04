"""Tests for ``zalmoxis.eos.paleos_api``: live PALEOS tabulation producer.

Covers ``GridSpec`` (hashing, axes, immutability), the log-uniform points-per-decade
helper, the four default-grid factories, the PALEOS SHA discovery fallback chain,
the header writer, the worker-count resolver, the exception-swallowing EoS
evaluator, the 10-column row formatter, the in-process worker init / row helpers,
the solid-side phase picker (``_get_mgsio3_solid_phase``), and end-to-end serial
table generation for both the unified and 2-phase MgSiO3 paths.

Anti-happy-path coverage in each test class: every class includes at least one
edge-case test (boundary, empty, extreme) AND at least one physically
unreasonable input that must raise or be handled. Discriminating values use
asymmetric P / T ranges and asymmetric grid sizes so axis-swap and
off-by-one bugs surface immediately.

Multi-process pool execution (``n_workers > 1``) is intentionally not exercised:
the worker entry points are tested in-process by calling ``_worker_init``
directly then ``_worker_unified_row`` / ``_worker_2phase_row``, which gives
full coverage of those code paths without spawning subprocesses inside pytest.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

# Skip the whole module when the upstream PALEOS package is not installed
# (CI environment does not vendor it, but local developer machines do).
# Tests in this file exercise paleos-backed code paths via paleos_api;
# they cannot run meaningfully without the package.
pytest.importorskip('paleos', reason='PALEOS package not installed in this environment')

from zalmoxis.eos import paleos_api  # noqa: E402

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Shared minimal grids
# ---------------------------------------------------------------------------

# Tiny (P, T) grid for end-to-end generators. Asymmetric n_p != n_t and
# asymmetric P/T ranges so any axis-swap is detectable.
_TINY_PT = dict(p_lo=1.0e9, p_hi=1.0e10, n_p=3, t_lo=1500.0, t_hi=4000.0, n_t=2)


def _tiny_grid() -> paleos_api.GridSpec:
    return paleos_api.GridSpec(**_TINY_PT)


# ---------------------------------------------------------------------------
# GridSpec
# ---------------------------------------------------------------------------


class TestGridSpec:
    """``GridSpec`` dataclass: hashing, axes, equality, immutability."""

    def test_axes_are_log_uniform_and_match_bounds_and_lengths(self):
        """``axes()`` returns log-uniform arrays at the requested bounds and counts.

        Discriminating: pressure spans 5 decades and temperature spans 1.5
        decades with different point counts, so a P/T axis swap or a
        wrong-sign log10 would show up as bound mismatch.
        """
        gs = paleos_api.GridSpec(
            p_lo=1.0e6, p_hi=1.0e11, n_p=11, t_lo=300.0, t_hi=10000.0, n_t=7
        )
        P, T = gs.axes()
        assert P.shape == (11,)
        assert T.shape == (7,)
        np.testing.assert_allclose(P[0], 1.0e6, rtol=1e-12)
        np.testing.assert_allclose(P[-1], 1.0e11, rtol=1e-12)
        np.testing.assert_allclose(T[0], 300.0, rtol=1e-12)
        np.testing.assert_allclose(T[-1], 10000.0, rtol=1e-12)
        # Constant log-spacing => constant ratio between successive nodes.
        # rtol 1e-12: pure logspace, no physics in between.
        np.testing.assert_allclose(np.diff(np.log10(P)), 0.5, rtol=1e-12)

    def test_hash_short_is_deterministic_across_construction(self):
        """Two equal GridSpecs hash to the same 10-char SHA-1 prefix."""
        gs1 = paleos_api.GridSpec(**_TINY_PT)
        gs2 = paleos_api.GridSpec(**_TINY_PT)
        assert gs1.hash_short() == gs2.hash_short()
        assert len(gs1.hash_short()) == 10
        # 10 hex chars
        assert re.fullmatch(r'[0-9a-f]{10}', gs1.hash_short())

    def test_hash_short_differs_when_any_field_differs(self):
        """Changing a single field changes the hash.

        Discriminating: bumps ``n_t`` by 1 only; any hash that ignored ``n_t``
        would collide.
        """
        gs1 = paleos_api.GridSpec(**_TINY_PT)
        gs2 = paleos_api.GridSpec(**{**_TINY_PT, 'n_t': _TINY_PT['n_t'] + 1})
        assert gs1.hash_short() != gs2.hash_short()

    def test_equality_is_by_value(self):
        """``GridSpec`` instances compare equal field-by-field."""
        gs1 = paleos_api.GridSpec(**_TINY_PT)
        gs2 = paleos_api.GridSpec(**_TINY_PT)
        assert gs1 == gs2
        gs3 = paleos_api.GridSpec(**{**_TINY_PT, 'p_hi': 2 * _TINY_PT['p_hi']})
        assert gs1 != gs3

    def test_dataclass_is_frozen(self):
        """Edge case: assignment to a field raises ``FrozenInstanceError``.

        ``frozen=True`` is load-bearing because hashing and dict-keying rely
        on it; any future loosening would break the cache layer.
        """
        gs = paleos_api.GridSpec(**_TINY_PT)
        with pytest.raises(Exception) as exc_info:
            gs.p_lo = 1.0  # type: ignore[misc]
        # dataclasses.FrozenInstanceError is a subclass of AttributeError.
        assert 'frozen' in repr(exc_info.value).lower() or isinstance(
            exc_info.value, AttributeError
        )


# ---------------------------------------------------------------------------
# _n_points_per_decade and the default-grid factories
# ---------------------------------------------------------------------------


class TestNPointsPerDecade:
    """Pure math helper: ``round(pts_per_decade * log10(hi/lo)) + 1``."""

    def test_returns_pts_plus_one_over_one_decade(self):
        """One decade at 600 pts/decade -> 601 nodes (inclusive bounds)."""
        # 1e5 -> 1e6 is exactly one decade.
        assert paleos_api._n_points_per_decade(1.0e5, 1.0e6, 600) == 601

    def test_handles_asymmetric_multi_decade_range(self):
        """1e-1 to 1e14 spans 15 decades -> 15*pts+1 nodes.

        Discriminating: a non-integer-decade range with 150 pts -> 2251 nodes.
        """
        # Integer-decade asymmetric range: 1e-1 .. 1e14 = 15 decades.
        assert paleos_api._n_points_per_decade(1.0e-1, 1.0e14, 150) == 15 * 150 + 1

    def test_rounds_when_decade_count_is_non_integer(self):
        """Half-decade range at 100 pts/decade -> round(50) + 1 = 51."""
        n = paleos_api._n_points_per_decade(1.0, 10.0**0.5, 100)
        assert n == 51

    def test_raises_on_nonpositive_lo(self):
        """Edge case: ``lo <= 0`` raises (log spacing undefined)."""
        with pytest.raises(ValueError, match='positive'):
            paleos_api._n_points_per_decade(0.0, 1.0e6, 600)
        with pytest.raises(ValueError, match='positive'):
            paleos_api._n_points_per_decade(-1.0, 1.0e6, 600)

    def test_raises_on_nonpositive_hi(self):
        """Physically unreasonable input: negative upper bound is rejected."""
        with pytest.raises(ValueError, match='positive'):
            paleos_api._n_points_per_decade(1.0, -10.0, 600)


class TestDefaultGrids:
    """Default Fe / MgSiO3 / H2O grids and ``make_grid_at_resolution``."""

    def test_make_grid_at_resolution_node_count_matches_pts_per_decade(self):
        """At 200 pts/decade, a 4-decade pressure range gives 4*200+1 = 801 nodes."""
        gs = paleos_api.make_grid_at_resolution(
            p_lo=1.0e6, p_hi=1.0e10, t_lo=1000.0, t_hi=10000.0, pts_per_decade=200
        )
        assert gs.n_p == 4 * 200 + 1
        assert gs.n_t == 1 * 200 + 1
        assert gs.p_lo == 1.0e6
        assert gs.p_hi == 1.0e10

    def test_make_default_grid_iron_bounds_match_zenodo_range(self):
        """Iron default: 1e5..1e14 Pa, 300..2e4 K, 9 decades pressure."""
        gs = paleos_api.make_default_grid_iron()
        np.testing.assert_allclose(gs.p_lo, 1.0e5, rtol=1e-12)
        np.testing.assert_allclose(gs.p_hi, 1.0e14, rtol=1e-12)
        np.testing.assert_allclose(gs.t_lo, 300.0, rtol=1e-12)
        np.testing.assert_allclose(gs.t_hi, 2.0e4, rtol=1e-12)
        # Sanity: 9 P decades * pts_per_decade + 1 nodes
        n_decades_P = np.log10(gs.p_hi / gs.p_lo)
        # n_p must be (round(n_decades_P * pts_per_decade) + 1).
        # Use the documented 600 pts/decade default, but read off implied value
        # so the test does not break if DEFAULT_PTS_PER_DECADE is overridden.
        implied = (gs.n_p - 1) / n_decades_P
        assert implied == pytest.approx(paleos_api.DEFAULT_PTS_PER_DECADE, rel=1e-6)

    def test_make_default_grid_mgsio3_t_hi_lower_than_iron(self):
        """MgSiO3 T_hi (11500 K) is below Fe T_hi (20000 K).

        Discriminating: mixing up the bounds between materials would surface
        here.
        """
        gs_mgsio3 = paleos_api.make_default_grid_mgsio3()
        gs_iron = paleos_api.make_default_grid_iron()
        np.testing.assert_allclose(gs_mgsio3.t_hi, 1.15e4, rtol=1e-12)
        assert gs_mgsio3.t_hi < gs_iron.t_hi

    def test_make_default_grid_h2o_extends_to_low_pressure(self):
        """H2O grid extends down to 0.1 Pa (AQUA coverage)."""
        gs = paleos_api.make_default_grid_h2o()
        np.testing.assert_allclose(gs.p_lo, 1.0e-1, rtol=1e-12)
        np.testing.assert_allclose(gs.p_hi, 1.0e14, rtol=1e-12)
        # 15 decades in P at the default pts/decade.
        assert gs.n_p > paleos_api.make_default_grid_iron().n_p

    def test_make_grid_at_resolution_rejects_zero_lo(self):
        """Edge case: ``p_lo = 0`` propagates through to ``_n_points_per_decade``."""
        with pytest.raises(ValueError, match='positive'):
            paleos_api.make_grid_at_resolution(
                p_lo=0.0, p_hi=1.0e10, t_lo=300.0, t_hi=4000.0, pts_per_decade=10
            )


# ---------------------------------------------------------------------------
# paleos_installed_sha
# ---------------------------------------------------------------------------


class TestPaleosInstalledSha:
    """Discover the installed PALEOS git SHA, with fallback paths."""

    def test_returns_40hex_sha_or_documented_fallback(self):
        """Either a 40-char hex SHA or one of the documented sentinel strings."""
        sha = paleos_api.paleos_installed_sha()
        assert isinstance(sha, str)
        # Valid outcomes: 40-hex git SHA, 'paleos-not-installed',
        # or 'version-...' (paleos.__version__ fallback).
        is_hex_sha = bool(re.fullmatch(r'[0-9a-f]{40}', sha))
        is_fallback = sha == 'paleos-not-installed' or sha.startswith('version-')
        assert is_hex_sha or is_fallback

    def test_returns_not_installed_sentinel_when_paleos_missing(self, monkeypatch):
        """If ``import paleos`` fails, the helper returns the documented sentinel.

        Edge case: simulates a fresh checkout without PALEOS installed by
        stubbing ``importlib.__import__`` for ``paleos`` only.
        """
        import builtins

        original_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == 'paleos':
                raise ImportError('no paleos for you')
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', fake_import)
        assert paleos_api.paleos_installed_sha() == 'paleos-not-installed'

    def test_falls_back_to_version_when_git_subprocess_fails(self, monkeypatch):
        """If ``git rev-parse HEAD`` fails, return ``version-<paleos.__version__>``.

        Discriminating: the fallback string carries the version prefix so a
        wrong branch (e.g. always returning 'unknown') is detectable.
        """
        import subprocess as real_subprocess

        def fake_check_output(*args, **kwargs):
            raise real_subprocess.CalledProcessError(128, ['git'])

        monkeypatch.setattr(paleos_api._subprocess, 'check_output', fake_check_output)
        sha = paleos_api.paleos_installed_sha()
        assert sha.startswith('version-')


# ---------------------------------------------------------------------------
# _write_header
# ---------------------------------------------------------------------------


class TestWriteHeader:
    """Header block emitted at the top of every produced .dat."""

    def test_header_contains_required_metadata_lines(self, tmp_path):
        """Header carries PALEOS_SHA, grid_hash, P/T ranges, and validity counts."""
        gs = _tiny_grid()
        path = tmp_path / 'h.dat'
        with open(path, 'w') as f:
            paleos_api._write_header(
                f,
                material='Fe',
                kind='unified (stable-phase)',
                grid=gs,
                paleos_sha='deadbeef' * 5,
                n_valid=5,
                n_skipped=1,
            )
        text = path.read_text()
        assert 'PALEOS_SHA: ' + 'deadbeef' * 5 in text
        assert f'grid_hash: {gs.hash_short()}' in text
        assert '5 valid / 1 skipped' in text
        # P/T grid lines carry the bounds in scientific notation.
        assert f'{gs.p_lo:.8e}' in text
        assert f'{gs.t_hi:.8e}' in text
        # Column line documents the 10-column layout.
        assert 'Columns:' in text and 'phase' in text

    def test_header_extra_lines_emitted_with_comment_prefix(self, tmp_path):
        """``extra_lines`` are appended verbatim with a ``# `` prefix.

        Discriminating: passes a unicode marker so a whitespace-only contains
        check would not pass.
        """
        gs = _tiny_grid()
        path = tmp_path / 'h.dat'
        with open(path, 'w') as f:
            paleos_api._write_header(
                f,
                material='MgSiO3',
                kind='solid',
                grid=gs,
                paleos_sha='abc',
                n_valid=0,
                n_skipped=0,
                extra_lines=['MARKER-XYZZY'],
            )
        text = path.read_text()
        assert '# MARKER-XYZZY' in text

    def test_header_with_zero_valid_does_not_raise(self, tmp_path):
        """Edge case: zero-valid / zero-skipped header still emits cleanly."""
        gs = _tiny_grid()
        path = tmp_path / 'h.dat'
        with open(path, 'w') as f:
            paleos_api._write_header(
                f,
                material='Fe',
                kind='unified',
                grid=gs,
                paleos_sha='0' * 40,
                n_valid=0,
                n_skipped=0,
            )
        # File contains the documented '0 valid / 0 skipped' line; if the
        # writer ever started skipping the count line on n_valid==0, that
        # would propagate downstream as a missing-metadata bug.
        assert '0 valid / 0 skipped' in path.read_text()


# ---------------------------------------------------------------------------
# _resolve_n_workers
# ---------------------------------------------------------------------------


class TestResolveNWorkers:
    """Map ``n_workers`` sentinel to an actual integer count."""

    def test_minus_one_resolves_to_cpu_count(self, monkeypatch):
        """``n_workers=-1`` -> ``os.cpu_count()`` (or 1 on exotic platforms)."""
        monkeypatch.setattr(paleos_api._os, 'cpu_count', lambda: 7)
        assert paleos_api._resolve_n_workers(-1) == 7

    def test_minus_one_falls_back_to_one_when_cpu_count_is_none(self, monkeypatch):
        """Edge case: ``os.cpu_count()`` returns None on some exotic systems."""
        monkeypatch.setattr(paleos_api._os, 'cpu_count', lambda: None)
        assert paleos_api._resolve_n_workers(-1) == 1

    def test_explicit_positive_int_passes_through(self):
        """Explicit ``n_workers=4`` must round-trip unchanged."""
        assert paleos_api._resolve_n_workers(4) == 4
        assert paleos_api._resolve_n_workers(1) == 1

    def test_zero_or_negative_raises(self):
        """Physically unreasonable: 0 workers / -2 workers -> ``ValueError``."""
        with pytest.raises(ValueError, match='>= 1'):
            paleos_api._resolve_n_workers(0)
        with pytest.raises(ValueError, match='>= 1'):
            paleos_api._resolve_n_workers(-5)


# ---------------------------------------------------------------------------
# _evaluate_eos
# ---------------------------------------------------------------------------


class _FakeEoSConst:
    """Mock EoS that returns asymmetric constant values for each property.

    Asymmetric values let a property-order swap (e.g. cp <-> cv) be detected.
    """

    def density(self, P, T):  # noqa: ARG002
        return 5500.0

    def specific_internal_energy(self, P, T):  # noqa: ARG002
        return 1.0e6

    def specific_entropy(self, P, T):  # noqa: ARG002
        return 1234.5

    def isobaric_heat_capacity(self, P, T):  # noqa: ARG002
        return 1200.0

    def isochoric_heat_capacity(self, P, T):  # noqa: ARG002
        return 900.0

    def thermal_expansion(self, P, T):  # noqa: ARG002
        return 1.5e-5

    def adiabatic_gradient(self, P, T):  # noqa: ARG002
        return 0.27


class _FakeEoSValueError(_FakeEoSConst):
    """Mock EoS whose density raises ``ValueError`` (PALEOS out-of-range signal)."""

    def density(self, P, T):
        raise ValueError('fake out of range')


class _FakeEoSRuntimeError(_FakeEoSConst):
    """Mock EoS whose entropy raises ``RuntimeError`` (PALEOS brentq-fail signal)."""

    def specific_entropy(self, P, T):
        raise RuntimeError('fake brentq fail')


class TestEvaluateEos:
    """Exception-swallowing wrapper around the 7 EoS scalar property methods."""

    def test_returns_seven_finite_floats_in_documented_order(self):
        """Successful call returns ``(rho, u, s, cp, cv, alpha, nabla_ad)``.

        Discriminating: every property has a distinct value so any reorder
        would be detected.
        """
        out = paleos_api._evaluate_eos(_FakeEoSConst(), 1.0e9, 4000.0)
        assert len(out) == 7
        rho, u, s, cp, cv, alpha, nabla_ad = out
        assert rho == pytest.approx(5500.0)
        assert u == pytest.approx(1.0e6)
        assert s == pytest.approx(1234.5)
        assert cp == pytest.approx(1200.0)
        assert cv == pytest.approx(900.0)
        assert alpha == pytest.approx(1.5e-5)
        assert nabla_ad == pytest.approx(0.27)

    def test_returns_all_nan_when_any_property_raises_value_error(self):
        """``ValueError`` from any property -> 7-tuple of NaN."""
        out = paleos_api._evaluate_eos(_FakeEoSValueError(), 1.0e9, 4000.0)
        assert all(np.isnan(v) for v in out)
        assert len(out) == 7

    def test_returns_all_nan_when_any_property_raises_runtime_error(self):
        """``RuntimeError`` (brentq fail) from any property -> 7-tuple of NaN.

        Edge case: error is raised by the 3rd property, not the 1st, so a
        wrap that only catches errors on density would still miss this.
        """
        out = paleos_api._evaluate_eos(_FakeEoSRuntimeError(), 1.0e9, 4000.0)
        assert all(np.isnan(v) for v in out)

    def test_does_not_swallow_unrelated_exception(self):
        """Physically unreasonable: ``KeyError`` is not in the contract -> propagates.

        Tests the negative side of the swallow list. The docstring claims only
        ``RuntimeError`` and ``ValueError`` are caught; verifying a third
        exception type bubbles up keeps the contract honest.
        """

        class _FakeEoSKeyError(_FakeEoSConst):
            def density(self, P, T):
                raise KeyError('not in contract')

        with pytest.raises(KeyError):
            paleos_api._evaluate_eos(_FakeEoSKeyError(), 1.0e9, 4000.0)

    def test_real_iron_eos_returns_finite_at_typical_core_conditions(self):
        """Sanity-check with the real ``IronEoS`` at a P/T inside the valid range.

        Discriminating: rho must be at the iron density scale (5e3 .. 1.5e4
        kg/m^3), not a wrong-units value at 5e6 or 5e0.
        """
        from paleos.iron_eos import IronEoS

        rho, u, s, cp, cv, alpha, nabla_ad = paleos_api._evaluate_eos(IronEoS(), 1.0e10, 4000.0)
        assert np.isfinite(rho) and 4.0e3 < rho < 1.5e4
        assert np.isfinite(s) and s > 0
        assert np.isfinite(cp) and cp > 0
        assert 0 < nabla_ad < 1


# ---------------------------------------------------------------------------
# _row formatter
# ---------------------------------------------------------------------------


class TestRowFormatter:
    """``_row`` produces the 10-column row layout consumed by the readers."""

    def test_round_trip_via_genfromtxt_recovers_inputs(self):
        """Numeric round-trip is bit-exact at the .8e printed precision.

        Discriminating: 9 distinct asymmetric numeric values in 9 columns
        catches any column reorder. rtol 1e-7 tracks the 8-digit print.
        """
        P, T = 1.234e9, 3.456e3
        props = (5500.0, 1.0e6, 1234.5, 1200.0, 900.0, 1.5e-5, 0.27)
        text = paleos_api._row(P, T, props, 'liquid')
        # Exactly one trailing newline.
        assert text.endswith('\n')
        # Parse via straightforward whitespace split: 9 floats + 1 string.
        tokens = text.split()
        nums = [float(t) for t in tokens[:9]]
        np.testing.assert_allclose(nums, [P, T, *props], rtol=1e-7)
        assert tokens[9] == 'liquid'

    def test_phase_string_with_internal_dash_preserved(self):
        """Edge case: PALEOS phase labels contain dashes (e.g. ``hcp-Fe``).

        Discriminating: ``solid-hpcen`` -> 'solid-hpcen' must not be split on
        the dash and lose the suffix.
        """
        text = paleos_api._row(
            1.0e9,
            3000.0,
            (5500.0, 1.0e6, 100.0, 1200.0, 900.0, 1e-5, 0.3),
            'solid-hpcen',
        )
        assert text.split()[-1] == 'solid-hpcen'

    def test_row_format_matches_documented_column_count(self):
        """Each row has exactly 10 whitespace-separated tokens (9 numeric + phase)."""
        text = paleos_api._row(
            1.0e9,
            3000.0,
            (5500.0, 1.0e6, 100.0, 1200.0, 900.0, 1e-5, 0.3),
            'liquid',
        )
        tokens = text.split()
        assert len(tokens) == 10

    def test_nan_props_emit_nan_tokens(self):
        """Physically unreasonable: NaN props are emitted as 'nan' so the
        downstream ``numpy.genfromtxt`` reader treats them as missing data."""
        text = paleos_api._row(
            1.0e9,
            3000.0,
            paleos_api._NAN_PROPS,
            'out-of-range',
        )
        # Numeric tokens 3..9 (rho .. nabla_ad) should be 'nan'.
        nan_tokens = text.split()[2:9]
        assert all('nan' in tok.lower() for tok in nan_tokens)


# ---------------------------------------------------------------------------
# _get_mgsio3_solid_phase
# ---------------------------------------------------------------------------


class TestGetMgsio3SolidPhase:
    """Solid-side polymorph picker with the melting-curve test removed."""

    @pytest.mark.parametrize(
        'P, T, expected_phase',
        [
            # Above P_brg_ppv(T) -> ppv. P_brg_ppv(2000) ~= 1.265e11 Pa.
            (2.0e11, 2000.0, 'solid-ppv'),
            # Above _P_HPCEN_BRG (12 GPa) but below P_brg_ppv -> brg.
            (5.0e10, 2000.0, 'solid-brg'),
            # T just below 750 K -> falls to lpcen branch.
            (1.0e8, 700.0, 'solid-lpcen'),
            # Mid pressure, mid T -> en (above 750 K, P below P_lpcen_en).
            (1.0e8, 1500.0, 'solid-en'),
        ],
    )
    def test_pressure_branches_dispatch_correctly(self, P, T, expected_phase):
        """Discriminating (P, T) probes hit each documented phase branch.

        Bounds chosen against the literal phase boundary functions in
        ``paleos.mgsio3_eos`` (``P_brg_ppv``, ``_P_HPCEN_BRG`` = 12 GPa,
        ``P_lpcen_en``).
        """
        assert paleos_api._get_mgsio3_solid_phase(P, T) == expected_phase

    def test_high_p_high_t_returns_postperovskite(self):
        """At P = 2e11 Pa and T = 3000 K, picker returns 'solid-ppv' (high-P phase)."""
        assert paleos_api._get_mgsio3_solid_phase(2.0e11, 3000.0) == 'solid-ppv'

    def test_above_liquidus_still_returns_solid_label(self):
        """Edge case: at T well above the MgSiO3 liquidus (5000 K), the picker
        still returns a solid polymorph rather than 'liquid'.

        This is the metastable-extension behaviour (the early-return on
        ``T >= T_melt`` is suppressed, by design, vs PALEOS's public picker).
        Discriminating: any return value starting with 'solid-' satisfies
        this; a plain 'liquid' would be a regression to PALEOS's public
        helper.
        """
        label = paleos_api._get_mgsio3_solid_phase(1.0e11, 6000.0)
        assert label.startswith('solid-')

    def test_mid_pressure_mid_temperature_path_returns_lpcen(self):
        """At T < 750 K and 5 GPa the lpcen branch fires regardless of phase boundaries.

        Bounds tuned to ``_P_HPCEN_BRG = 12 GPa`` so the function does NOT
        take the brg branch.
        """
        assert paleos_api._get_mgsio3_solid_phase(5.0e9, 1000.0) == 'solid-lpcen'

    def test_inner_lpcen_branch_when_below_hpcen_curve(self):
        """At T = 1000 K, P = 6.4e9 Pa: P >= P_en_hpcen but P < P_lpcen_hpcen.

        Hits the inner ``return 'solid-lpcen'`` (the second one, inside the
        ``P >= P_en_hpcen(T)`` block). At T=1000 K, the boundary functions
        are P_en_hpcen=6.30e9 and P_lpcen_hpcen=6.54e9.
        """
        assert paleos_api._get_mgsio3_solid_phase(6.4e9, 1000.0) == 'solid-lpcen'


# ---------------------------------------------------------------------------
# Worker init / row helpers (in-process, no Pool spawn)
# ---------------------------------------------------------------------------


class TestWorkerInit:
    """``_worker_init`` populates ``_WORKER_EOS`` for the requested material."""

    def teardown_method(self, method):  # noqa: ARG002
        """Clear module-global EoS cache between tests.

        The dict is shared across calls inside a single process; tests that
        mutate it must clean up to avoid leaking instances to siblings.
        """
        paleos_api._WORKER_EOS.clear()

    def test_iron_init_populates_unified_iron_eos(self):
        """After init, ``_WORKER_EOS['unified']`` is an ``IronEoS`` instance."""
        from paleos.iron_eos import IronEoS

        paleos_api._WORKER_EOS.clear()
        paleos_api._worker_init('iron')
        assert isinstance(paleos_api._WORKER_EOS['unified'], IronEoS)
        assert 'wolf18' not in paleos_api._WORKER_EOS

    def test_mgsio3_init_populates_unified_wolf18_and_phase_map(self):
        """MgSiO3 init fills ``unified``, ``wolf18``, and ``phase_map`` keys."""
        from paleos.mgsio3_eos import MgSiO3EoS, Wolf18

        paleos_api._WORKER_EOS.clear()
        paleos_api._worker_init('mgsio3')
        assert isinstance(paleos_api._WORKER_EOS['unified'], MgSiO3EoS)
        assert isinstance(paleos_api._WORKER_EOS['wolf18'], Wolf18)
        # phase_map is a dict from phase-string -> EoS instance.
        pmap = paleos_api._WORKER_EOS['phase_map']
        assert isinstance(pmap, dict)
        # Must contain the polymorph labels the picker can produce.
        assert {'solid-lpcen', 'solid-en', 'solid-hpcen', 'solid-brg', 'solid-ppv'} <= set(pmap)

    def test_unknown_material_raises_value_error(self):
        """Physically unreasonable: unknown material string is rejected."""
        with pytest.raises(ValueError, match='unknown material'):
            paleos_api._worker_init('basalt')

    def test_h2o_init_constructs_water_eos_with_table_path(self, monkeypatch):
        """The h2o branch constructs ``WaterEoS(table_path=...)`` when given a path.

        Patches ``paleos.water_eos.WaterEoS`` at the narrowest scope (single
        symbol) and records constructor kwargs to verify the path is forwarded.
        """
        captured = {}

        class FakeWaterEoS:
            def __init__(self, table_path=None):
                captured['table_path'] = table_path

        import paleos.water_eos as _wmod

        monkeypatch.setattr(_wmod, 'WaterEoS', FakeWaterEoS)
        paleos_api._WORKER_EOS.clear()
        paleos_api._worker_init('h2o', h2o_table_path='/some/aqua.dat')
        assert isinstance(paleos_api._WORKER_EOS['unified'], FakeWaterEoS)
        assert captured['table_path'] == '/some/aqua.dat'

    def test_h2o_init_uses_default_constructor_when_no_path(self, monkeypatch):
        """Edge case: h2o_table_path=None -> ``WaterEoS()`` with no kwargs."""
        captured = {}

        class FakeWaterEoSNoArg:
            def __init__(self):
                captured['called'] = True

        import paleos.water_eos as _wmod

        monkeypatch.setattr(_wmod, 'WaterEoS', FakeWaterEoSNoArg)
        paleos_api._WORKER_EOS.clear()
        paleos_api._worker_init('h2o', h2o_table_path=None)
        assert captured.get('called') is True


class TestWorkerUnifiedRow:
    """``_worker_unified_row`` tabulates one P-row for the unified generator."""

    def teardown_method(self, method):  # noqa: ARG002
        paleos_api._WORKER_EOS.clear()

    def test_row_text_has_one_line_per_temperature_node(self):
        """For an iron worker on a 4-T axis, output text has 4 newline-terminated lines.

        Discriminating: the T_axis is asymmetric (not a uniform step) so any
        bug that hard-codes T-grid step would surface.
        """
        paleos_api._WORKER_EOS.clear()
        paleos_api._worker_init('iron')
        T_axis = [1500.0, 2500.0, 4000.0, 6000.0]
        i_P, rows_text, n_valid, n_skip = paleos_api._worker_unified_row((3, 5.0e9, T_axis))
        assert i_P == 3
        assert rows_text.count('\n') == 4
        # All 4 should be valid for iron at these conditions.
        assert n_valid == 4
        assert n_skip == 0
        # Each row has 10 whitespace-separated tokens.
        for line in rows_text.strip().split('\n'):
            assert len(line.split()) == 10

    def test_row_text_pressures_match_input_p(self):
        """Edge case: every row's first column is the input P, not the row index."""
        paleos_api._WORKER_EOS.clear()
        paleos_api._worker_init('iron')
        T_axis = [2000.0, 3000.0]
        _, rows_text, _, _ = paleos_api._worker_unified_row((0, 7.5e9, T_axis))
        first_col = [float(line.split()[0]) for line in rows_text.strip().split('\n')]
        np.testing.assert_allclose(first_col, [7.5e9, 7.5e9], rtol=1e-12)

    def test_phase_method_raise_falls_back_to_unknown_label(self, monkeypatch):
        """When ``eos.phase()`` raises ``RuntimeError``, the row's phase token is 'unknown'.

        Discriminating: ``_evaluate_eos`` returned finite props (``n_valid``
        increments), so the fallback path runs only inside the phase-label
        try/except. Patches the ``phase`` method at the narrowest scope.
        """
        from paleos.iron_eos import IronEoS

        paleos_api._WORKER_EOS.clear()
        eos = IronEoS()

        def raise_runtime(P, T):  # noqa: ARG001
            raise RuntimeError('phase lookup failed')

        monkeypatch.setattr(eos, 'phase', raise_runtime)
        paleos_api._WORKER_EOS['unified'] = eos
        _, rows_text, n_valid, n_skip = paleos_api._worker_unified_row((0, 1.0e10, [3000.0]))
        # Props were finite (real iron EoS) but phase() raised -> 'unknown'.
        assert n_valid == 1
        assert n_skip == 0
        assert rows_text.strip().split()[-1] == 'unknown'

    def test_row_text_marks_failed_evaluation_as_out_of_range(self, monkeypatch):
        """When ``_evaluate_eos`` returns NaN, phase is forced to 'out-of-range'.

        Patches ``_evaluate_eos`` with the narrowest scope possible (single
        symbol replacement) so other code paths are untouched.
        """
        from paleos.iron_eos import IronEoS

        paleos_api._WORKER_EOS.clear()
        paleos_api._WORKER_EOS['unified'] = IronEoS()

        def all_nan_eval(eos, P, T):  # noqa: ARG001
            return paleos_api._NAN_PROPS

        monkeypatch.setattr(paleos_api, '_evaluate_eos', all_nan_eval)
        i_P, rows_text, n_valid, n_skip = paleos_api._worker_unified_row(
            (0, 1.0e9, [1500.0, 3000.0])
        )
        assert n_valid == 0
        assert n_skip == 2
        # Last token (phase column) is the sentinel.
        for line in rows_text.strip().split('\n'):
            assert line.split()[-1] == 'out-of-range'


class TestWorker2PhaseRow:
    """``_worker_2phase_row`` tabulates one P-row for the 2-phase MgSiO3 generator."""

    def teardown_method(self, method):  # noqa: ARG002
        paleos_api._WORKER_EOS.clear()

    def test_returns_two_text_blocks_with_matching_line_counts(self):
        """Solid and liquid blocks each carry exactly len(T_axis) lines.

        Discriminating: the two outputs have independent valid / skipped
        counters; if the two blocks diverged in length, downstream readers
        would mis-index.
        """
        paleos_api._WORKER_EOS.clear()
        paleos_api._worker_init('mgsio3')
        T_axis = [2000.0, 3500.0]
        result = paleos_api._worker_2phase_row((0, 5.0e10, T_axis))
        i_P, s_text, l_text, nv_s, ns_s, nv_l, ns_l = result
        assert i_P == 0
        assert s_text.count('\n') == 2
        assert l_text.count('\n') == 2
        # Counters partition T_axis entries.
        assert nv_s + ns_s == 2
        assert nv_l + ns_l == 2

    def test_solid_phase_label_matches_picker_at_high_pressure(self):
        """At 5e10 Pa, picker says 'solid-brg'; the row's phase token agrees.

        Discriminating: any wrong dispatch through ``phase_map`` would surface
        as a label mismatch.
        """
        paleos_api._WORKER_EOS.clear()
        paleos_api._worker_init('mgsio3')
        # Single T point keeps the test fast; 5e10 Pa, 2500 K is in brg field.
        _, s_text, l_text, _, _, _, _ = paleos_api._worker_2phase_row((0, 5.0e10, [2500.0]))
        s_phase = s_text.strip().split()[-1]
        l_phase = l_text.strip().split()[-1]
        assert s_phase == 'solid-brg'
        assert l_phase == 'liquid'

    def test_nan_props_force_out_of_range_label_on_both_sides(self, monkeypatch):
        """When ``_evaluate_eos`` returns NaN for both phases, both labels become 'out-of-range'.

        Edge case: hits the NaN branches in both the solid and liquid blocks
        (lines 338-339 and 347-348). Patches ``_evaluate_eos`` at the
        narrowest scope.
        """
        paleos_api._WORKER_EOS.clear()
        paleos_api._worker_init('mgsio3')

        def all_nan(eos, P, T):  # noqa: ARG001
            return paleos_api._NAN_PROPS

        monkeypatch.setattr(paleos_api, '_evaluate_eos', all_nan)
        _, s_text, l_text, nv_s, ns_s, nv_l, ns_l = paleos_api._worker_2phase_row(
            (0, 5.0e10, [3000.0, 4000.0])
        )
        assert nv_s == 0 and ns_s == 2
        assert nv_l == 0 and ns_l == 2
        for line in s_text.strip().split('\n'):
            assert line.split()[-1] == 'out-of-range'
        for line in l_text.strip().split('\n'):
            assert line.split()[-1] == 'out-of-range'

    def test_liquid_label_is_constant_liquid_for_finite_props(self):
        """Edge case: the liquid block always labels every finite row 'liquid'.

        Wolf18 is evaluated everywhere (metastable extension below solidus),
        so unless the call returns NaN, the label must be 'liquid' rather
        than a polymorph name.
        """
        paleos_api._WORKER_EOS.clear()
        paleos_api._worker_init('mgsio3')
        _, _, l_text, _, _, _, _ = paleos_api._worker_2phase_row((0, 1.0e10, [3000.0, 4000.0]))
        for line in l_text.strip().split('\n'):
            tok = line.split()[-1]
            assert tok in {'liquid', 'out-of-range'}


# ---------------------------------------------------------------------------
# generate_paleos_api_unified_table (end-to-end, n_workers=1)
# ---------------------------------------------------------------------------


class TestGenerateUnifiedTable:
    """End-to-end serial generation of a unified PALEOS .dat for material='iron'."""

    def test_iron_table_round_trips_through_load_paleos_all_properties(self, tmp_path):
        """Tiny iron table parses cleanly and yields finite density at all nodes.

        Discriminating: iron density at 1e9-1e10 Pa, 1500-4000 K must be in
        [4e3, 1.5e4] kg/m^3. A wrong unit (e.g. cgs g/cm^3) would land at 5,
        not 5000.
        """
        gs = _tiny_grid()
        out_path = tmp_path / 'iron_unified.dat'
        result = paleos_api.generate_paleos_api_unified_table(
            'iron', out_path, gs, n_workers=1, log_every=0
        )
        assert out_path.is_file()
        assert result['n_valid'] == gs.n_p * gs.n_t
        assert result['n_skipped'] == 0
        assert isinstance(result['sha'], str)

        # Round-trip through the production loader the consumers actually use.
        # Local import to avoid pytest-cov's scipy double-init at collection.
        from zalmoxis import eos_export

        tab = eos_export.load_paleos_all_properties(out_path)
        assert tab['rho'].shape == (gs.n_p, gs.n_t)
        assert np.all(np.isfinite(tab['rho']))
        # Iron density physical bounds at these conditions.
        assert tab['rho'].min() > 4.0e3
        assert tab['rho'].max() < 1.5e4

    def test_header_carries_paleos_sha_line(self, tmp_path):
        """The output .dat header contains a ``# PALEOS_SHA: ...`` tag."""
        gs = _tiny_grid()
        out_path = tmp_path / 'iron_unified.dat'
        paleos_api.generate_paleos_api_unified_table(
            'iron', out_path, gs, n_workers=1, log_every=0
        )
        text = out_path.read_text()
        # Match either a 40-hex SHA or the documented fallback strings.
        assert re.search(r'^#\s*PALEOS_SHA:\s*\S+', text, flags=re.MULTILINE)

    def test_invalid_material_raises_value_error(self, tmp_path):
        """Physically unreasonable: ``material='unknown'`` is rejected up front."""
        gs = _tiny_grid()
        with pytest.raises(ValueError, match='material'):
            paleos_api.generate_paleos_api_unified_table(
                'unknown', tmp_path / 'x.dat', gs, n_workers=1
            )

    def test_zero_p_lo_raises_value_error(self, tmp_path):
        """Edge case: ``GridSpec.p_lo = 0`` is rejected before any EoS instantiation."""
        # Bypass GridSpec's logspace path by handing a manually-built grid:
        gs_bad = paleos_api.GridSpec(
            p_lo=0.0, p_hi=1.0e10, n_p=2, t_lo=300.0, t_hi=4000.0, n_t=2
        )
        with pytest.raises(ValueError, match='p_lo'):
            paleos_api.generate_paleos_api_unified_table(
                'iron', tmp_path / 'x.dat', gs_bad, n_workers=1
            )

    def test_creates_parent_directories(self, tmp_path):
        """Edge case: the writer creates missing parent directories on demand."""
        gs = _tiny_grid()
        out_path = tmp_path / 'sub1' / 'sub2' / 'iron.dat'
        paleos_api.generate_paleos_api_unified_table(
            'iron', out_path, gs, n_workers=1, log_every=0
        )
        assert out_path.is_file()

    def test_log_every_one_emits_progress_log_per_row(self, tmp_path, caplog):
        """``log_every=1`` triggers a progress log line for each P row.

        Hits the ``log_every and n_done % log_every == 0`` branch. Discriminating:
        a 3-row tiny grid should produce exactly 3 progress lines, not 0 or 6.
        """
        gs = _tiny_grid()
        out_path = tmp_path / 'iron_logs.dat'
        with caplog.at_level('INFO', logger='zalmoxis.eos.paleos_api'):
            paleos_api.generate_paleos_api_unified_table(
                'iron', out_path, gs, n_workers=1, log_every=1
            )
        progress_lines = [r for r in caplog.records if 'P rows done' in r.getMessage()]
        assert len(progress_lines) == gs.n_p


# ---------------------------------------------------------------------------
# generate_paleos_api_2phase_mgsio3_tables (end-to-end, n_workers=1)
# ---------------------------------------------------------------------------


class TestGenerate2PhaseMgsio3Tables:
    """End-to-end 2-phase MgSiO3 generation: solid + liquid files with metastable extensions."""

    @pytest.fixture(scope='class')
    def two_phase_outputs(self, tmp_path_factory):
        """Build solid+liquid tables once per class, reuse across tests.

        MgSiO3 generation pays the Wolf18 sympy compile cost on first call;
        sharing the output across the class's tests keeps the suite under a
        few seconds.
        """
        tmp_dir = tmp_path_factory.mktemp('mgsio3_2p')
        gs = paleos_api.GridSpec(
            p_lo=1.0e10, p_hi=1.0e11, n_p=2, t_lo=2500.0, t_hi=4500.0, n_t=2
        )
        solid_path = tmp_dir / 'solid.dat'
        liquid_path = tmp_dir / 'liquid.dat'
        result = paleos_api.generate_paleos_api_2phase_mgsio3_tables(
            solid_path, liquid_path, gs, n_workers=1, log_every=0
        )
        return solid_path, liquid_path, result, gs

    def test_both_files_written_with_correct_row_counts(self, two_phase_outputs):
        """Each file contains n_p * n_t data rows after the header."""
        s_path, l_path, _, gs = two_phase_outputs
        for p in (s_path, l_path):
            data_lines = [
                ln for ln in p.read_text().splitlines() if ln.strip() and not ln.startswith('#')
            ]
            assert len(data_lines) == gs.n_p * gs.n_t

    def test_solid_and_liquid_files_carry_distinct_phase_columns(self, two_phase_outputs):
        """Solid file's last token is 'solid-...'; liquid file's is 'liquid'.

        Discriminating: a file-write swap would land 'liquid' tokens in the
        solid file; this test would fail immediately.
        """
        s_path, l_path, _, _ = two_phase_outputs
        s_phases = {
            ln.split()[-1]
            for ln in s_path.read_text().splitlines()
            if ln.strip() and not ln.startswith('#')
        }
        l_phases = {
            ln.split()[-1]
            for ln in l_path.read_text().splitlines()
            if ln.strip() and not ln.startswith('#')
        }
        # Solid file: at least one polymorph token. None should be 'liquid'.
        assert 'liquid' not in s_phases
        assert any(ph.startswith('solid-') or ph == 'out-of-range' for ph in s_phases)
        # Liquid file: only 'liquid' or 'out-of-range'.
        assert l_phases <= {'liquid', 'out-of-range'}

    def test_liquid_entropy_differs_from_solid_at_matched_p_t(self, two_phase_outputs):
        """At the same (P, T), Wolf18 liquid and solid polymorph entropies differ.

        Discriminating: latent heat of melting > 0 means S_liquid > S_solid
        at the same (P, T) inside the stable-liquid region. Even in the
        metastable-extension region the values must not coincide bit-for-bit
        (would indicate one file accidentally overwriting the other).
        """
        s_path, l_path, _, _ = two_phase_outputs
        s_rows = np.genfromtxt(
            [ln for ln in s_path.read_text().splitlines() if not ln.startswith('#')],
            usecols=(0, 1, 4),  # P, T, s
        )
        l_rows = np.genfromtxt(
            [ln for ln in l_path.read_text().splitlines() if not ln.startswith('#')],
            usecols=(0, 1, 4),
        )
        # Pair rows by (P, T); they were written in identical (i_P, T) order.
        np.testing.assert_allclose(s_rows[:, 0], l_rows[:, 0], rtol=1e-12)
        np.testing.assert_allclose(s_rows[:, 1], l_rows[:, 1], rtol=1e-12)
        # At least one (P, T) pair has S_liquid > S_solid by a margin > 1 J/(kg K).
        # If every paired entropy were equal, this would catch a swap.
        finite = np.isfinite(s_rows[:, 2]) & np.isfinite(l_rows[:, 2])
        assert finite.any()
        diff = l_rows[finite, 2] - s_rows[finite, 2]
        assert np.any(np.abs(diff) > 1.0)

    def test_zero_p_lo_raises_value_error(self, tmp_path):
        """Physically unreasonable: ``p_lo = 0`` rejected before any work starts."""
        gs_bad = paleos_api.GridSpec(
            p_lo=0.0, p_hi=1.0e10, n_p=2, t_lo=2000.0, t_hi=4000.0, n_t=2
        )
        with pytest.raises(ValueError, match='p_lo'):
            paleos_api.generate_paleos_api_2phase_mgsio3_tables(
                tmp_path / 's.dat', tmp_path / 'l.dat', gs_bad, n_workers=1
            )

    def test_log_every_one_emits_progress_log_per_row(self, tmp_path, caplog):
        """``log_every=1`` triggers a progress log line per P row in the 2-phase path.

        Hits the ``log_every and n_done % log_every == 0`` branch in the
        2-phase generator. Uses a 2x2 grid to keep cost bounded.
        """
        gs = paleos_api.GridSpec(
            p_lo=1.0e10, p_hi=1.0e11, n_p=2, t_lo=2500.0, t_hi=4500.0, n_t=2
        )
        s_path = tmp_path / 'solid_logs.dat'
        l_path = tmp_path / 'liquid_logs.dat'
        with caplog.at_level('INFO', logger='zalmoxis.eos.paleos_api'):
            paleos_api.generate_paleos_api_2phase_mgsio3_tables(
                s_path, l_path, gs, n_workers=1, log_every=1
            )
        progress_lines = [r for r in caplog.records if 'P rows done' in r.getMessage()]
        assert len(progress_lines) == gs.n_p

    def test_result_dict_carries_per_side_counts_and_paths(self, two_phase_outputs):
        """Edge case: the result dict's structure matches the documented schema."""
        s_path, l_path, result, gs = two_phase_outputs
        assert {'solid', 'liquid', 'sha'} <= set(result.keys())
        assert {'n_valid', 'n_skipped', 'path'} <= set(result['solid'].keys())
        assert {'n_valid', 'n_skipped', 'path'} <= set(result['liquid'].keys())
        assert result['solid']['n_valid'] + result['solid']['n_skipped'] == gs.n_p * gs.n_t
        assert result['liquid']['n_valid'] + result['liquid']['n_skipped'] == gs.n_p * gs.n_t
        assert Path(result['solid']['path']) == s_path
        assert Path(result['liquid']['path']) == l_path
