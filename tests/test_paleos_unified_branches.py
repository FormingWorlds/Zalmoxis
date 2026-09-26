"""Synthetic-cache branch tests for ``zalmoxis.eos.paleos``.

The unified-PALEOS density / nabla_ad lookups have several edge-case
branches that the data-file-driven tests in ``test_paleos_unified.py``
do not reach: pressure clamping, mushy-zone outside-liquidus fallback,
NaN-cell nearest-neighbor recovery, and the per-pressure clamp inner
branches. This file builds a small synthetic cache dict that
``_ensure_unified_cache`` returns directly when a key is already
present, then probes each branch with controlled inputs.

The synthetic cache uses a log-uniform 8x8 grid over P in [1e9, 1e12]
Pa and T in [1000, 10000] K. Density is ``rho(P, T) = 5000 * P^0.1 *
(T / 4000)^-0.05`` (no physical meaning, just a smooth analytic
function so the bilinear and nearest-neighbor paths return finite
values when the grid is intact).
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from zalmoxis.eos import paleos as _paleos
from zalmoxis.eos.paleos import (
    _get_paleos_unified_nabla_ad,
    get_paleos_unified_density,
    get_paleos_unified_density_batch,
)

pytestmark = pytest.mark.unit


def _build_cache(*, with_nan_corner: bool = False) -> dict:
    """Construct a synthetic unified-PALEOS cache.

    Parameters
    ----------
    with_nan_corner : bool
        If True, set the (0, 0) corner of the density grid to NaN so
        that bilinear queries near the corner return NaN, exercising
        the nearest-neighbor fallback paths.
    """
    n_p = n_t = 8
    log_p = np.linspace(9.0, 12.0, n_p)  # 1e9 to 1e12 Pa
    log_t = np.linspace(3.0, 4.0, n_t)  # 1000 to 10000 K

    P, T = np.meshgrid(10.0**log_p, 10.0**log_t, indexing='ij')
    density_grid = 5000.0 * (P / 1e9) ** 0.1 * (T / 4000.0) ** -0.05
    nabla_ad_grid = 0.25 * np.ones_like(density_grid)
    if with_nan_corner:
        density_grid[0, 0] = np.nan
        nabla_ad_grid[0, 0] = np.nan

    # Liquidus: synthetic monotonic curve from 2000 K at low P to 5000 K at high P
    liq_lp = log_p.copy()
    liq_lt = np.linspace(np.log10(2000.0), np.log10(5000.0), n_p)

    # Per-pressure clamp bounds: full T range is valid (no ragged edges)
    lt_min = np.full(n_p, log_t[0])
    lt_max = np.full(n_p, log_t[-1])

    # NN fallbacks: use a closure mimicking NearestNDInterpolator's __call__
    def _nn_density(point):
        # Accept tuple/array; return nearest grid value (no NaN handling here).
        if hasattr(point, '__len__') and len(point) == 2 and np.ndim(point[0]) == 0:
            lp, lt = point
            return float(_nearest(lp, lt, density_grid))
        pts = np.atleast_2d(np.asarray(point, dtype=float))
        return np.array([_nearest(p[0], p[1], density_grid) for p in pts])

    def _nn_nabla(point):
        if hasattr(point, '__len__') and len(point) == 2 and np.ndim(point[0]) == 0:
            lp, lt = point
            return float(_nearest(lp, lt, nabla_ad_grid))
        pts = np.atleast_2d(np.asarray(point, dtype=float))
        return np.array([_nearest(p[0], p[1], nabla_ad_grid) for p in pts])

    def _nearest(lp, lt, grid):
        ip = int(np.clip(round((lp - log_p[0]) / (log_p[1] - log_p[0])), 0, n_p - 1))
        it = int(np.clip(round((lt - log_t[0]) / (log_t[1] - log_t[0])), 0, n_t - 1))
        v = grid[ip, it]
        if not np.isfinite(v):
            # Walk outward until a finite cell is found
            for r in range(1, max(n_p, n_t)):
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        ii, jj = ip + di, it + dj
                        if 0 <= ii < n_p and 0 <= jj < n_t and np.isfinite(grid[ii, jj]):
                            return float(grid[ii, jj])
            return np.nan
        return float(v)

    return {
        'type': 'paleos_unified',
        'density_grid': density_grid,
        'nabla_ad_grid': nabla_ad_grid,
        'density_nn': _nn_density,
        'nabla_ad_nn': _nn_nabla,
        'p_min': 10.0 ** log_p[0],
        'p_max': 10.0 ** log_p[-1],
        't_min': 10.0 ** log_t[0],
        't_max': 10.0 ** log_t[-1],
        'unique_log_p': log_p,
        'unique_log_t': log_t,
        'logt_valid_min': lt_min,
        'logt_valid_max': lt_max,
        'liquidus_log_p': liq_lp,
        'liquidus_log_t': liq_lt,
        'logp_min': log_p[0],
        'logt_min': log_t[0],
        'dlog_p': log_p[1] - log_p[0],
        'dlog_t': log_t[1] - log_t[0],
        'n_p': n_p,
        'n_t': n_t,
    }


@pytest.fixture
def synthetic_cache():
    """Single-call cache and the matching material_dict / cache dict."""
    cache = _build_cache()
    eos_file = '/synthetic/paleos.dat'
    return {
        'mat': {'eos_file': eos_file, 'format': 'paleos_unified'},
        'cache': {eos_file: cache},
        'eos_file': eos_file,
        'entry': cache,
    }


@pytest.fixture
def synthetic_cache_with_nan():
    """Cache where the (0, 0) corner is NaN, to provoke NN fallbacks."""
    cache = _build_cache(with_nan_corner=True)
    eos_file = '/synthetic/paleos_nan.dat'
    return {
        'mat': {'eos_file': eos_file, 'format': 'paleos_unified'},
        'cache': {eos_file: cache},
        'eos_file': eos_file,
        'entry': cache,
    }


class TestPressureClamping:
    """Pressure outside ``[p_min, p_max]`` is clamped to the bound."""

    def test_below_p_min_clamps_up(self, synthetic_cache):
        """Pressure below ``p_min`` is replaced by ``p_min`` before lookup."""

        s = synthetic_cache
        rho_below = get_paleos_unified_density(
            1e8,  # 10x below p_min=1e9
            3000.0,
            s['mat'],
            mushy_zone_factor=1.0,
            interpolation_functions=s['cache'],
        )
        rho_at = get_paleos_unified_density(
            s['entry']['p_min'],
            3000.0,
            s['mat'],
            mushy_zone_factor=1.0,
            interpolation_functions=s['cache'],
        )
        assert rho_below == pytest.approx(rho_at, rel=1e-12)

    def test_above_p_max_clamps_down(self, synthetic_cache):
        """Pressure above ``p_max`` is replaced by ``p_max`` before lookup."""

        s = synthetic_cache
        rho_above = get_paleos_unified_density(
            1e15,  # 1000x above p_max=1e12
            3000.0,
            s['mat'],
            mushy_zone_factor=1.0,
            interpolation_functions=s['cache'],
        )
        rho_at = get_paleos_unified_density(
            s['entry']['p_max'],
            3000.0,
            s['mat'],
            mushy_zone_factor=1.0,
            interpolation_functions=s['cache'],
        )
        assert rho_above == pytest.approx(rho_at, rel=1e-12)


class TestMushyZoneOutsideLiquidus:
    """Mushy-zone path falls back to direct lookup when log_p is outside the
    extracted liquidus coverage range."""

    def test_below_liquidus_p_range(self, synthetic_cache):
        """Query below the liquidus log_p[0] uses the direct-lookup branch."""

        s = synthetic_cache
        # liquidus_log_p starts at 9.0; query at log_p < 9.0 by setting
        # p_min lower than the liquidus coverage start.
        s['entry']['liquidus_log_p'] = np.array([10.0, 11.0, 12.0])
        s['entry']['liquidus_log_t'] = np.log10(np.array([3000.0, 4000.0, 5000.0]))

        rho = get_paleos_unified_density(
            1e9,  # log_p = 9.0 < liquidus_log_p[0] = 10.0
            3000.0,
            s['mat'],
            mushy_zone_factor=0.8,  # mushy mode
            interpolation_functions=s['cache'],
        )
        assert rho is not None and np.isfinite(rho)

    def test_above_liquidus_p_range(self, synthetic_cache):
        """Query above the liquidus log_p[-1] uses the direct-lookup branch."""

        s = synthetic_cache
        s['entry']['liquidus_log_p'] = np.array([9.0, 10.0, 11.0])
        s['entry']['liquidus_log_t'] = np.log10(np.array([3000.0, 4000.0, 5000.0]))

        rho = get_paleos_unified_density(
            1e12,  # log_p = 12.0 > liquidus_log_p[-1] = 11.0
            3000.0,
            s['mat'],
            mushy_zone_factor=0.8,
            interpolation_functions=s['cache'],
        )
        assert rho is not None and np.isfinite(rho)


class TestEmptyLiquidusGoesDirectLookup:
    """When ``liquidus_log_p`` is empty, even the mushy path falls back to
    direct lookup (the second clause of the ``mushy_zone_factor`` guard)."""

    def test_empty_liquidus_uses_direct_lookup(self, synthetic_cache):

        s = synthetic_cache
        s['entry']['liquidus_log_p'] = np.array([])
        s['entry']['liquidus_log_t'] = np.array([])
        rho = get_paleos_unified_density(1e10, 3000.0, s['mat'], 0.8, s['cache'])
        assert rho is not None and np.isfinite(rho)


class TestNanFallback:
    """Bilinear lookup over a NaN cell falls back to nearest-neighbor."""

    def test_density_nn_fallback_at_corner(self, synthetic_cache_with_nan):
        """Querying at the NaN corner triggers the NN density fallback."""

        s = synthetic_cache_with_nan
        # Query inside the (0, 0) cell so bilinear returns NaN
        rho = get_paleos_unified_density(
            10.0 ** s['entry']['logp_min'],  # P at logp_min
            10.0 ** s['entry']['logt_min'],  # T at logt_min
            s['mat'],
            mushy_zone_factor=1.0,
            interpolation_functions=s['cache'],
        )
        assert rho is not None and np.isfinite(rho)

    def test_nabla_ad_nn_fallback_at_corner(self, synthetic_cache_with_nan):
        """Querying at the NaN corner triggers the NN nabla_ad fallback."""

        s = synthetic_cache_with_nan
        nabla = _get_paleos_unified_nabla_ad(
            10.0 ** s['entry']['logp_min'],
            10.0 ** s['entry']['logt_min'],
            s['mat'],
            s['cache'],
        )
        assert nabla is not None and np.isfinite(nabla)


class TestExceptionPath:
    """Errors inside the lookup are caught and surface as ``None``."""

    def test_lookup_returns_none_on_exception(self, synthetic_cache):
        """A torn cache (missing required key) raises inside ``_fast_bilinear``
        but the outer try/except returns None and logs the error."""

        s = synthetic_cache
        # Drop a required field so _fast_bilinear's KeyError fires.
        del s['entry']['n_p']
        rho = get_paleos_unified_density(1e10, 3000.0, s['mat'], 1.0, s['cache'])
        assert rho is None


class TestBatchPath:
    """Vectorized lookup matches the scalar lookup elementwise."""

    def test_batch_matches_scalar(self, synthetic_cache):

        s = synthetic_cache
        ps = np.array([1e9, 1e10, 1e11, 1e12])
        ts = np.array([1500.0, 3000.0, 5000.0, 7000.0])
        rho_batch = get_paleos_unified_density_batch(ps, ts, s['mat'], 1.0, s['cache'])
        rho_scalar = np.array(
            [
                get_paleos_unified_density(p, t, s['mat'], 1.0, s['cache'])
                for p, t in zip(ps, ts, strict=False)
            ]
        )
        np.testing.assert_allclose(rho_batch, rho_scalar, rtol=1e-12)

    @pytest.mark.parametrize('narrow', [True, False])
    @pytest.mark.parametrize('mzf', [0.8, 0.7])
    def test_batch_matches_scalar_in_the_mushy_zone(self, synthetic_cache, mzf, narrow):
        """With a mushy zone the batch path takes the table's own liquidus, as the scalar
        path does, on the clamped pressure, and the direct lookup outside the liquidus
        pressure range (narrowed, or the full table range with P outside the table)."""
        s = synthetic_cache
        if narrow:
            s['entry']['liquidus_log_p'] = np.linspace(9.5, 11.5, 8)
        p, t = np.meshgrid(np.logspace(8.5, 12.5, 17), np.linspace(1200.0, 6000.0, 17))
        ps, ts = p.ravel(), t.ravel()
        rho_batch = get_paleos_unified_density_batch(ps, ts, s['mat'], mzf, s['cache'])
        rho_scalar = [
            get_paleos_unified_density(p, t, s['mat'], mzf, s['cache'])
            for p, t in zip(ps, ts, strict=False)
        ]
        np.testing.assert_allclose(rho_batch, rho_scalar, rtol=1e-12)

    def test_batch_mushy_path(self, synthetic_cache):
        """With ``mushy_zone_factor < 1`` the batch path takes the mushy
        branch for shells that fall between the synthetic solidus/liquidus."""

        s = synthetic_cache
        # Table liquidus at log P = 10: about 2714 K; mzf 0.8 puts the solidus at 2171 K.
        ps = np.array([1e10, 1e10, 1e10])
        ts = np.array([1500.0, 2500.0, 6000.0])  # below / mushy / above
        rho = get_paleos_unified_density_batch(ps, ts, s['mat'], 0.8, s['cache'])
        assert np.all(np.isfinite(rho))
        # All three should be in a sensible range
        assert np.all(rho > 1000)
        assert np.all(rho < 20000)
        scalar = [
            get_paleos_unified_density(p, t, s['mat'], 0.8, s['cache'])
            for p, t in zip(ps, ts, strict=False)
        ]
        np.testing.assert_allclose(rho, scalar, rtol=1e-12)
        direct = get_paleos_unified_density_batch(ps, ts, s['mat'], 1.0, s['cache'])
        assert rho[1] != pytest.approx(
            direct[1], rel=1e-6
        )  # the mushy blend, not the direct value

    def test_batch_nan_recovery(self, synthetic_cache_with_nan):
        """NaN cells are recovered by NN fallback in the batch path."""

        s = synthetic_cache_with_nan
        ps = np.array([10.0 ** s['entry']['logp_min'], 1e11])
        ts = np.array([10.0 ** s['entry']['logt_min'], 4000.0])
        rho = get_paleos_unified_density_batch(ps, ts, s['mat'], 1.0, s['cache'])
        assert np.all(np.isfinite(rho))

    def test_batch_mushy_solid_side_nan_corner_uses_nn_fallback(self, synthetic_cache_with_nan):
        """At P=1e9 Pa the table liquidus is 2000 K (2001 K with the phase guard) and
        ``mushy_zone_factor=0.5`` puts the solidus at 1000 K, so T=2000 K is in the mushy
        zone and its solid-side lookup reads the (0, 0) NaN corner, which the NN fallback
        recovers as in the scalar path."""
        s = synthetic_cache_with_nan
        ps = np.array([10.0 ** s['entry']['logp_min']])
        ts = np.array([2000.0])
        rho = get_paleos_unified_density_batch(ps, ts, s['mat'], 0.5, s['cache'])
        scalar = get_paleos_unified_density(ps[0], ts[0], s['mat'], 0.5, s['cache'])
        assert np.isfinite(rho[0]) and rho[0] == pytest.approx(scalar, rel=1e-12)

    def test_batch_below_solidus_with_nan_recovery(self, synthetic_cache_with_nan):
        """A shell below the solidus takes the direct lookup, which can land on a NaN
        cell; the NN fallback must then recover the density. (No above-liquidus query
        reaches this fixture's NaN corner.)

        At P=1e9 Pa with ``mushy_zone_factor=0.8``, T_sol = 1600 K. A query
        at T=900 K is below T_sol, classified as ``below``, and bypasses
        the mushy branch. After per-cell clamping log_t up to
        ``logt_valid_min[0]=3.0`` the bilinear lands on the (0, 0) NaN
        corner; the NN fallback fires.
        """
        s = synthetic_cache_with_nan
        ps = np.array([1e9])
        ts = np.array([900.0])  # well below T_sol at P=1e9
        rho = get_paleos_unified_density_batch(ps, ts, s['mat'], 0.8, s['cache'])
        assert np.all(np.isfinite(rho))
        # Density should be physically reasonable (synthetic table
        # rho ~ 5000 * (P/1e9)^0.1 at table corner is ~5000 kg/m^3).
        assert 1000 < rho[0] < 20000
        scalar = get_paleos_unified_density(ps[0], ts[0], s['mat'], 0.8, s['cache'])
        assert rho[0] == pytest.approx(scalar, rel=1e-12)

    def test_batch_mushy_solid_and_liquid_side_nan_recovery(self):
        """The solid-side, liquid-side and direct NN fallbacks each recover a NaN cell.

        At P=1e10 Pa: table liquidus T_liq ≈ 2714 K, mushy_zone_factor=0.6
        gives T_sol ≈ 1629 K. log_t_sol ≈ 3.212 brackets cells
        (2,1)-(3,2); log_t_liq ≈ 3.434 brackets cells (2,3)-(3,4). NaN at
        (3, 2) lies in the solid-side bracket (and in the direct bracket of
        the mushy query at T = 2500 K, whose direct value the mushy value
        replaces); NaN at (3, 3) lies in the liquid-side bracket and in the
        direct bracket of the above-liquidus query at T = 2800 K (log_t ≈
        3.447, cells (2,3)-(3,4)), where the direct fallback sets the result.
        """
        cache = _build_cache()
        # log_p step is 3/7 ≈ 0.429, log_t step is 1/7 ≈ 0.143.
        # cell (3, 2) is at (log_p≈10.286, log_t≈3.286) -> solid bracket
        # cell (3, 3) is at (log_p≈10.286, log_t≈3.429) -> liquid bracket
        cache['density_grid'][3, 2] = np.nan
        cache['density_grid'][3, 3] = np.nan
        eos_file = '/synthetic/paleos_solliq_nan.dat'
        mat = {'eos_file': eos_file, 'format': 'paleos_unified'}
        cdict = {eos_file: cache}

        ps = np.array([10**10.0, 10**10.0])
        ts = np.array([2500.0, 2800.0])  # mushy and above the liquidus at P=1e10
        rho = get_paleos_unified_density_batch(ps, ts, mat, 0.6, cdict)
        assert np.all(np.isfinite(rho)) and np.all((1000 < rho) & (rho < 20000))
        scalar = [
            get_paleos_unified_density(p, t, mat, 0.6, cdict)
            for p, t in zip(ps, ts, strict=False)
        ]
        np.testing.assert_allclose(rho, scalar, rtol=1e-12)


class TestPerCellClampLogger:
    """The per-cell ``T`` clamp emits a one-shot warning per file."""

    def test_clamp_warning_emitted_once(self, synthetic_cache, caplog):
        """First out-of-range T triggers the warning; second is silent."""

        s = synthetic_cache
        # Tighten lt_max so that querying T near t_max triggers a clamp
        s['entry']['logt_valid_max'] = np.full_like(s['entry']['logt_valid_max'], 3.5)

        # Reset module-level warned-set so the first call definitely warns
        _paleos._paleos_clamp_warned.discard(s['eos_file'])
        with caplog.at_level(logging.WARNING, logger='zalmoxis.eos.paleos'):
            get_paleos_unified_density(1e10, 9000.0, s['mat'], 1.0, s['cache'])
            n_first = len(caplog.records)
            get_paleos_unified_density(1e10, 9000.0, s['mat'], 1.0, s['cache'])
            n_second = len(caplog.records)
        assert n_first >= 1
        assert n_second == n_first  # second call did not add a new record


class TestMushyZoneNanFallback:
    """NaN cells inside the mushy-zone solid/liquid lookups are recovered by
    nearest-neighbor or surface as None when both sides fail."""

    def test_nan_at_solid_side_recovers(self, synthetic_cache_with_nan):
        """A query in the mushy band with NaN at the solid-side grid point
        falls back via density_nn for that side, then volume-averages."""

        s = synthetic_cache_with_nan
        # Liquidus: pick T_liq=4000K at our query log_p, so mushy band is
        # T_sol = 0.6 * 4000 = 2400 K to T_liq = 4000 K. Our NaN is at
        # corner (logp=9, logt=3) -> P=1e9, T=1000 K, well below mushy.
        # We can't easily trigger NaN at the mushy lookups with this fixture
        # alone; instead, build a mushy-friendly liquidus and put NaN where
        # the solid-side query lands.
        s['entry']['liquidus_log_p'] = np.linspace(9.0, 12.0, 8)
        s['entry']['liquidus_log_t'] = np.full(8, np.log10(4000.0))
        # Place NaN near the solid-side query point (T_sol ~ 0.6*4000 = 2400)
        # The grid has log_t in [3.0, 4.0]; log10(2400) ~ 3.38
        s['entry']['density_grid'][2, 2] = np.nan  # near (logp~10, logt~3.43)
        rho = get_paleos_unified_density(
            10**10.0,  # logp = 10
            3000.0,  # in mushy zone (between 2400 and 4000)
            s['mat'],
            mushy_zone_factor=0.6,
            interpolation_functions=s['cache'],
        )
        assert rho is not None and np.isfinite(rho)

    def test_below_solidus_in_mushy_mode(self, synthetic_cache):
        """``temperature <= T_sol`` in mushy mode takes the below-solidus
        direct-lookup branch (line 155-160), distinct from the no-mushy-zone
        guard at the top of the function."""

        s = synthetic_cache
        # Force liquidus T = 5000 K at our query pressure
        s['entry']['liquidus_log_p'] = np.linspace(9.0, 12.0, 8)
        s['entry']['liquidus_log_t'] = np.full(8, np.log10(5000.0))
        # mushy_zone_factor = 0.5 -> T_sol = 2500 K. Query at T = 1500 K
        # is below T_sol, takes the below-solidus branch.
        rho = get_paleos_unified_density(
            10**10.0,
            1500.0,
            s['mat'],
            mushy_zone_factor=0.5,
            interpolation_functions=s['cache'],
        )
        assert rho is not None and np.isfinite(rho)

    def test_mushy_clamp_to_local_tmax(self, synthetic_cache):
        """When the synthetic liquidus pushes T_liq above the per-pressure
        ``logt_valid_max``, the clamp branch at lines 202-205 fires."""

        s = synthetic_cache
        # Tighten the per-P upper bound below liquidus log_t
        s['entry']['liquidus_log_p'] = np.linspace(9.0, 12.0, 8)
        s['entry']['liquidus_log_t'] = np.full(8, np.log10(8000.0))  # above logt_max
        s['entry']['logt_valid_max'] = np.full(8, np.log10(6000.0))
        rho = get_paleos_unified_density(
            10**10.0,
            5000.0,
            s['mat'],
            mushy_zone_factor=0.5,
            interpolation_functions=s['cache'],
        )
        assert rho is not None and np.isfinite(rho)

    def test_mushy_clamp_to_local_tmin(self, synthetic_cache):
        """When the synthetic solidus pushes T_sol below the per-pressure
        ``logt_valid_min``, the clamp branch at lines 196-197 fires."""

        s = synthetic_cache
        s['entry']['liquidus_log_p'] = np.linspace(9.0, 12.0, 8)
        s['entry']['liquidus_log_t'] = np.full(8, np.log10(4000.0))
        # Raise the per-P lower bound above the solidus log_t (T_sol = 2000)
        s['entry']['logt_valid_min'] = np.full(8, np.log10(2500.0))
        rho = get_paleos_unified_density(
            10**10.0,
            3000.0,
            s['mat'],
            mushy_zone_factor=0.5,  # T_sol = 0.5 * 4000 = 2000 < 2500
            interpolation_functions=s['cache'],
        )
        assert rho is not None and np.isfinite(rho)

    def test_nan_bound_fallback(self, synthetic_cache):
        """When ``logt_valid_min`` has NaN at the local pressure, the clamp
        falls through unchanged (lines 209-210)."""

        s = synthetic_cache
        s['entry']['liquidus_log_p'] = np.linspace(9.0, 12.0, 8)
        s['entry']['liquidus_log_t'] = np.full(8, np.log10(4000.0))
        # Inject NaN at the logt_valid_min entry adjacent to query pressure
        s['entry']['logt_valid_min'] = np.full(8, np.nan)
        s['entry']['logt_valid_max'] = np.full(8, np.nan)
        rho = get_paleos_unified_density(
            10**10.0,
            3000.0,
            s['mat'],
            mushy_zone_factor=0.5,
            interpolation_functions=s['cache'],
        )
        # Should still return a finite value via NN fallback
        assert rho is not None and np.isfinite(rho)


class TestNablaAdScalar:
    """Scalar nabla_ad lookup wraps the same clamp/lookup pattern."""

    def test_nabla_finite_on_synthetic_grid(self, synthetic_cache):

        s = synthetic_cache
        nabla = _get_paleos_unified_nabla_ad(1e10, 4000.0, s['mat'], s['cache'])
        assert nabla == pytest.approx(0.25, rel=1e-12)

    def test_nabla_pressure_clamping(self, synthetic_cache):
        """Pressure below ``p_min`` is clamped before lookup (matches density
        lookup behaviour). Returns the same value as a query at the boundary."""

        s = synthetic_cache
        nabla_below = _get_paleos_unified_nabla_ad(1e8, 4000.0, s['mat'], s['cache'])
        nabla_at = _get_paleos_unified_nabla_ad(
            s['entry']['p_min'], 4000.0, s['mat'], s['cache']
        )
        assert nabla_below == pytest.approx(nabla_at, rel=1e-12)

    def test_nabla_clamp_warning_once(self, synthetic_cache, caplog):
        """Clamp warning fires once per file for nabla_ad as well."""

        s = synthetic_cache
        s['entry']['logt_valid_max'] = np.full_like(s['entry']['logt_valid_max'], 3.5)
        _paleos._paleos_clamp_warned.discard(s['eos_file'])
        with caplog.at_level(logging.WARNING, logger='zalmoxis.eos.paleos'):
            _get_paleos_unified_nabla_ad(1e10, 9000.0, s['mat'], s['cache'])
            assert any('clamping' in r.message for r in caplog.records)


# Anti-happy-path: physically unreasonable inputs should not crash silently.
class TestUnreasonableInputs:
    """Negative or zero temperature is guarded by the ``log10(max(T, 1.0))`` clamp."""

    def test_zero_temperature_uses_floor(self, synthetic_cache):
        """``T = 0 K`` is clamped to 1 K before log10; lookup still returns a
        finite value (per-cell T-clamp pulls it back into the valid range)."""

        s = synthetic_cache
        rho = get_paleos_unified_density(1e10, 0.0, s['mat'], 1.0, s['cache'])
        assert rho is not None and np.isfinite(rho)
