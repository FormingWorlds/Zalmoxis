"""Shared interpolation helpers for PALEOS EOS tables.

Provides table loading, bilinear interpolation, and per-cell temperature
clamping used by all PALEOS-based EOS lookups.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import (
    NearestNDInterpolator,
    RegularGridInterpolator,
)

logger = logging.getLogger(__name__)

# Track whether the per-cell clamping warning has been issued for each file,
# to avoid flooding the log with repeated messages.
_paleos_clamp_warned = set()


def load_paleos_table(eos_file):
    """Load a PALEOS MgSiO3 table and build RegularGridInterpolator objects.

    The PALEOS tables have 10 columns (SI units):
    P, T, rho, u, s, cp, cv, alpha, nabla_ad, phase_id(string).
    The grid is log-uniform in both P and T with 150 points per decade.
    Some grid cells are missing (unconverged corners), filled with NaN.
    A P=0 row may exist and is excluded for log interpolation.

    Parameters
    ----------
    eos_file : str
        Path to the PALEOS table file.

    Returns
    -------
    dict
        Cache entry with keys:
        - ``'type'``: ``'paleos'``
        - ``'density_interp'``: RegularGridInterpolator for rho(log10P, log10T)
        - ``'nabla_ad_interp'``: RegularGridInterpolator for nabla_ad(log10P, log10T)
        - ``'p_min'``, ``'p_max'``: pressure bounds in Pa
        - ``'t_min'``, ``'t_max'``: temperature bounds in K
    """
    # Read only numeric columns (0-8), skipping the string phase_id column (9)
    data = np.genfromtxt(eos_file, usecols=range(9), comments='#')

    pressures = data[:, 0]
    temps = data[:, 1]
    densities = data[:, 2]
    nabla_ad = data[:, 8]

    # Filter out P=0 rows (log10(0) is undefined)
    valid = pressures > 0
    pressures = pressures[valid]
    temps = temps[valid]
    densities = densities[valid]
    nabla_ad = nabla_ad[valid]

    # Work in log10 space for the grid axes
    log_p = np.log10(pressures)
    log_t = np.log10(temps)

    unique_log_p = np.unique(log_p)
    unique_log_t = np.unique(log_t)

    n_p = len(unique_log_p)
    n_t = len(unique_log_t)

    # Build 2D grids filled with NaN for missing cells
    density_grid = np.full((n_p, n_t), np.nan)
    nabla_ad_grid = np.full((n_p, n_t), np.nan)

    # Map log_p and log_t values to grid indices
    p_idx_map = {v: i for i, v in enumerate(unique_log_p)}
    t_idx_map = {v: i for i, v in enumerate(unique_log_t)}

    for k in range(len(pressures)):
        ip = p_idx_map[log_p[k]]
        it = t_idx_map[log_t[k]]
        density_grid[ip, it] = densities[k]
        nabla_ad_grid[ip, it] = nabla_ad[k]

    density_interp = RegularGridInterpolator(
        (unique_log_p, unique_log_t),
        density_grid,
        bounds_error=False,
        fill_value=np.nan,
    )
    nabla_ad_interp = RegularGridInterpolator(
        (unique_log_p, unique_log_t),
        nabla_ad_grid,
        bounds_error=False,
        fill_value=np.nan,
    )

    # Per-pressure valid T bounds for per-cell clamping.
    # At each pressure row, find the min and max log10(T) with finite density.
    # This lets us clamp queries into the valid domain when the grid has NaN
    # holes in the corners (e.g. high T at low P in the liquid table).
    logt_valid_min = np.full(n_p, np.nan)
    logt_valid_max = np.full(n_p, np.nan)
    for ip in range(n_p):
        finite_mask = np.isfinite(density_grid[ip, :])
        if finite_mask.any():
            valid_indices = np.where(finite_mask)[0]
            logt_valid_min[ip] = unique_log_t[valid_indices[0]]
            logt_valid_max[ip] = unique_log_t[valid_indices[-1]]

    # Nearest-neighbor fallback interpolators built from valid cells only.
    # Used when bilinear interpolation returns NaN near the ragged domain
    # boundary (a valid cell neighboring a NaN cell poisons bilinear output).
    valid_cells = np.isfinite(density_grid)
    ip_valid, it_valid = np.where(valid_cells)
    coords_valid = np.column_stack([unique_log_p[ip_valid], unique_log_t[it_valid]])

    density_nn = NearestNDInterpolator(coords_valid, density_grid[valid_cells])
    nabla_ad_nn = NearestNDInterpolator(
        coords_valid[np.isfinite(nabla_ad_grid[valid_cells])],
        nabla_ad_grid[valid_cells][np.isfinite(nabla_ad_grid[valid_cells])],
    )

    # Precompute grid spacing for O(1) bilinear interpolation.
    dlog_p = (unique_log_p[-1] - unique_log_p[0]) / (n_p - 1) if n_p > 1 else 1.0
    dlog_t = (unique_log_t[-1] - unique_log_t[0]) / (n_t - 1) if n_t > 1 else 1.0

    return {
        'type': 'paleos',
        'density_interp': density_interp,
        'nabla_ad_interp': nabla_ad_interp,
        'density_nn': density_nn,
        'nabla_ad_nn': nabla_ad_nn,
        'p_min': 10.0 ** unique_log_p[0],
        'p_max': 10.0 ** unique_log_p[-1],
        't_min': 10.0 ** unique_log_t[0],
        't_max': 10.0 ** unique_log_t[-1],
        'unique_log_p': unique_log_p,
        'unique_log_t': unique_log_t,
        'logt_valid_min': logt_valid_min,
        'logt_valid_max': logt_valid_max,
        # Fast bilinear interpolation data
        'density_grid': density_grid,
        'nabla_ad_grid': nabla_ad_grid,
        'logp_min': unique_log_p[0],
        'logp_max': unique_log_p[-1],
        'logt_min': unique_log_t[0],
        'logt_max': unique_log_t[-1],
        'dlog_p': dlog_p,
        'dlog_t': dlog_t,
        'n_p': n_p,
        'n_t': n_t,
    }


def load_paleos_unified_table(eos_file):
    """Load a unified PALEOS table (single file per material) and build interpolators.

    The unified tables use the same 10-column format as the 2-phase tables
    (P, T, rho, u, s, cp, cv, alpha, nabla_ad, phase_id) but contain all
    stable phases for a material in a single file. The thermodynamically
    stable phase at each (P, T) is encoded in the phase column.

    In addition to density and nabla_ad interpolators, this function
    extracts the liquidus boundary from the phase column: for each pressure
    row, it finds the lowest temperature where the phase is 'liquid'.

    Parameters
    ----------
    eos_file : str
        Path to the unified PALEOS table file.

    Returns
    -------
    dict
        Cache entry with keys:
        - ``'type'``: ``'paleos_unified'``
        - ``'density_interp'``: RegularGridInterpolator for rho(log10P, log10T)
        - ``'nabla_ad_interp'``: RegularGridInterpolator for nabla_ad(log10P, log10T)
        - ``'density_nn'``, ``'nabla_ad_nn'``: NearestNDInterpolator fallbacks
        - ``'p_min'``, ``'p_max'``: pressure bounds in Pa
        - ``'t_min'``, ``'t_max'``: temperature bounds in K
        - ``'unique_log_p'``: unique log10(P) grid values
        - ``'logt_valid_min'``, ``'logt_valid_max'``: per-pressure T bounds
        - ``'liquidus_log_p'``: log10(P) array for the extracted liquidus
        - ``'liquidus_log_t'``: log10(T_liquidus) array at each pressure
        - ``'phase_grid'``: 2D string array of phase identifiers
    """
    # Single-pass read: parse numeric columns and phase string together.
    # Avoids the 2x penalty of calling genfromtxt twice on 50-140 MB files.
    numeric_rows = []
    phase_list = []
    with open(eos_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            numeric_rows.append([float(x) for x in parts[:9]])
            phase_list.append(parts[9] if len(parts) > 9 else '')

    data_numeric = np.array(numeric_rows)
    phase_strings = np.array(phase_list, dtype=str)

    pressures = data_numeric[:, 0]
    temps = data_numeric[:, 1]
    densities = data_numeric[:, 2]
    nabla_ad = data_numeric[:, 8]

    # Filter out P=0 rows
    valid = pressures > 0
    pressures = pressures[valid]
    temps = temps[valid]
    densities = densities[valid]
    nabla_ad = nabla_ad[valid]
    phase_strings = np.char.strip(phase_strings[valid])

    # Work in log10 space
    log_p = np.log10(pressures)
    log_t = np.log10(temps)

    unique_log_p = np.unique(log_p)
    unique_log_t = np.unique(log_t)

    n_p = len(unique_log_p)
    n_t = len(unique_log_t)

    # Build 2D grids
    density_grid = np.full((n_p, n_t), np.nan)
    nabla_ad_grid = np.full((n_p, n_t), np.nan)
    phase_grid = np.full((n_p, n_t), '', dtype=object)

    p_idx_map = {v: i for i, v in enumerate(unique_log_p)}
    t_idx_map = {v: i for i, v in enumerate(unique_log_t)}

    for k in range(len(pressures)):
        ip = p_idx_map[log_p[k]]
        it = t_idx_map[log_t[k]]
        density_grid[ip, it] = densities[k]
        nabla_ad_grid[ip, it] = nabla_ad[k]
        phase_grid[ip, it] = phase_strings[k]

    density_interp = RegularGridInterpolator(
        (unique_log_p, unique_log_t),
        density_grid,
        bounds_error=False,
        fill_value=np.nan,
    )
    nabla_ad_interp = RegularGridInterpolator(
        (unique_log_p, unique_log_t),
        nabla_ad_grid,
        bounds_error=False,
        fill_value=np.nan,
    )

    # Per-pressure valid T bounds
    logt_valid_min = np.full(n_p, np.nan)
    logt_valid_max = np.full(n_p, np.nan)
    for ip in range(n_p):
        finite_mask = np.isfinite(density_grid[ip, :])
        if finite_mask.any():
            valid_indices = np.where(finite_mask)[0]
            logt_valid_min[ip] = unique_log_t[valid_indices[0]]
            logt_valid_max[ip] = unique_log_t[valid_indices[-1]]

    # Nearest-neighbor fallback interpolators
    valid_cells = np.isfinite(density_grid)
    ip_valid, it_valid = np.where(valid_cells)
    coords_valid = np.column_stack([unique_log_p[ip_valid], unique_log_t[it_valid]])

    density_nn = NearestNDInterpolator(coords_valid, density_grid[valid_cells])

    nabla_valid = np.isfinite(nabla_ad_grid[valid_cells])
    nabla_ad_nn = NearestNDInterpolator(
        coords_valid[nabla_valid],
        nabla_ad_grid[valid_cells][nabla_valid],
    )

    # Extract liquidus boundary from the phase column.
    # For each pressure row, find the lowest T where phase == 'liquid'.
    liquidus_log_p = []
    liquidus_log_t = []
    for ip in range(n_p):
        liquid_mask = phase_grid[ip, :] == 'liquid'
        if liquid_mask.any():
            first_liquid_idx = np.where(liquid_mask)[0][0]
            liquidus_log_p.append(unique_log_p[ip])
            liquidus_log_t.append(unique_log_t[first_liquid_idx])

    liquidus_log_p = np.array(liquidus_log_p)
    liquidus_log_t = np.array(liquidus_log_t)

    # Precompute grid spacing for O(1) bilinear interpolation.
    # Both PALEOS and Chabrier grids are log-uniform (constant dlogP, dlogT).
    dlog_p = (unique_log_p[-1] - unique_log_p[0]) / (n_p - 1) if n_p > 1 else 1.0
    dlog_t = (unique_log_t[-1] - unique_log_t[0]) / (n_t - 1) if n_t > 1 else 1.0

    # Verify grid is log-uniform (required for O(1) bilinear interpolation)
    if n_p > 2:
        p_spacings = np.diff(unique_log_p)
        if not np.allclose(p_spacings, dlog_p, rtol=1e-3):
            logger.warning(
                f'P grid is not log-uniform: spacing range '
                f'[{p_spacings.min():.6f}, {p_spacings.max():.6f}] '
                f'vs mean {dlog_p:.6f}. Bilinear interpolation may be inaccurate.'
            )
    if n_t > 2:
        t_spacings = np.diff(unique_log_t)
        if not np.allclose(t_spacings, dlog_t, rtol=1e-3):
            logger.warning(
                f'T grid is not log-uniform: spacing range '
                f'[{t_spacings.min():.6f}, {t_spacings.max():.6f}] '
                f'vs mean {dlog_t:.6f}. Bilinear interpolation may be inaccurate.'
            )

    return {
        'type': 'paleos_unified',
        'density_interp': density_interp,
        'nabla_ad_interp': nabla_ad_interp,
        'density_nn': density_nn,
        'nabla_ad_nn': nabla_ad_nn,
        'p_min': 10.0 ** unique_log_p[0],
        'p_max': 10.0 ** unique_log_p[-1],
        't_min': 10.0 ** unique_log_t[0],
        't_max': 10.0 ** unique_log_t[-1],
        'unique_log_p': unique_log_p,
        'unique_log_t': unique_log_t,
        'logt_valid_min': logt_valid_min,
        'logt_valid_max': logt_valid_max,
        'liquidus_log_p': liquidus_log_p,
        'liquidus_log_t': liquidus_log_t,
        'phase_grid': phase_grid,
        # Fast bilinear interpolation data
        'density_grid': density_grid,
        'nabla_ad_grid': nabla_ad_grid,
        'logp_min': unique_log_p[0],
        'logp_max': unique_log_p[-1],
        'logt_min': unique_log_t[0],
        'logt_max': unique_log_t[-1],
        'dlog_p': dlog_p,
        'dlog_t': dlog_t,
        'n_p': n_p,
        'n_t': n_t,
    }


def _fast_bilinear(log_p, log_t, grid, cached):
    """O(1) bilinear interpolation on a nominally log-uniform grid.

    Replaces scipy's RegularGridInterpolator for scalar lookups. The
    index computation uses O(1) arithmetic with a rounding correction
    to handle floating-point spacing jitter in the grid axes, then
    computes the fractional position from the actual stored grid
    coordinates for exact interpolation.

    Parameters
    ----------
    log_p : float
        log10 of pressure, already clamped to grid bounds.
    log_t : float
        log10 of temperature, already clamped to valid range.
    grid : numpy.ndarray
        2D array of values (density or nabla_ad), shape (n_p, n_t).
    cached : dict
        Cache entry with grid metadata (logp_min, dlog_p, unique_log_p,
        unique_log_t, etc.).

    Returns
    -------
    float
        Interpolated value. NaN if any corner is NaN.
    """
    # Guard: grids with fewer than 2 points cannot be interpolated
    if cached['n_p'] < 2 or cached['n_t'] < 2:
        return grid[0, 0]

    ulp = cached['unique_log_p']
    ult = cached['unique_log_t']
    n_p = cached['n_p']
    n_t = cached['n_t']

    # O(1) index estimation for nominally log-uniform grid.
    # Use round() to get the nearest node, then determine the lower
    # bounding index. This handles floating-point spacing jitter that
    # causes int() to land on the wrong cell.
    fp = (log_p - cached['logp_min']) / cached['dlog_p']
    ft = (log_t - cached['logt_min']) / cached['dlog_t']

    # Nearest node via rounding, then find lower bounding index
    ip_near = min(max(int(fp + 0.5), 0), n_p - 1)
    it_near = min(max(int(ft + 0.5), 0), n_t - 1)

    # Determine lower bounding index: if the query is below the
    # nearest node, step back by one
    if ip_near > 0 and log_p < ulp[ip_near]:
        ip = ip_near - 1
    else:
        ip = ip_near
    ip = max(0, min(ip, n_p - 2))

    if it_near > 0 and log_t < ult[it_near]:
        it = it_near - 1
    else:
        it = it_near
    it = max(0, min(it, n_t - 2))

    # Fractional parts computed from actual grid coordinates
    span_p = ulp[ip + 1] - ulp[ip]
    span_t = ult[it + 1] - ult[it]

    dp = (log_p - ulp[ip]) / span_p if span_p > 0 else 0.0
    dt = (log_t - ult[it]) / span_t if span_t > 0 else 0.0

    # Clamp to [0, 1] for boundary safety
    dp = max(0.0, min(dp, 1.0))
    dt = max(0.0, min(dt, 1.0))

    # Four corner values
    v00 = grid[ip, it]
    v01 = grid[ip, it + 1]
    v10 = grid[ip + 1, it]
    v11 = grid[ip + 1, it + 1]

    # Bilinear interpolation
    return v00 * (1 - dp) * (1 - dt) + v01 * (1 - dp) * dt + v10 * dp * (1 - dt) + v11 * dp * dt


def fast_bilinear_batch(log_p_arr, log_t_arr, grid, cached):
    """Vectorized O(1) bilinear interpolation on a log-uniform grid.

    Batch version of ``_fast_bilinear`` for arrays of query points.
    Uses numpy vectorized operations instead of Python loops, giving
    5-10x speedup over calling ``_fast_bilinear`` in a loop or using
    scipy ``RegularGridInterpolator`` for small-to-medium batches.

    Parameters
    ----------
    log_p_arr : numpy.ndarray
        1D array of log10(pressure) values.
    log_t_arr : numpy.ndarray
        1D array of log10(temperature) values (same length as log_p_arr).
    grid : numpy.ndarray
        2D array of values, shape (n_p, n_t).
    cached : dict
        Cache entry with grid metadata.

    Returns
    -------
    numpy.ndarray
        Interpolated values. NaN where any corner is NaN.
    """
    n_p = cached['n_p']
    n_t = cached['n_t']

    if n_p < 2 or n_t < 2:
        return np.full(len(log_p_arr), grid[0, 0])

    logp_min = cached['logp_min']
    logt_min = cached['logt_min']
    dlog_p = cached['dlog_p']
    dlog_t = cached['dlog_t']

    # O(1) index computation for log-uniform grids
    fp = (np.asarray(log_p_arr) - logp_min) / dlog_p
    ft = (np.asarray(log_t_arr) - logt_min) / dlog_t

    # Lower bounding indices (clamp to valid range)
    ip = np.clip(np.floor(fp).astype(int), 0, n_p - 2)
    it = np.clip(np.floor(ft).astype(int), 0, n_t - 2)

    # Fractional positions within cell
    ulp = cached['unique_log_p']
    ult = cached['unique_log_t']
    span_p = ulp[ip + 1] - ulp[ip]
    span_t = ult[it + 1] - ult[it]

    dp = np.where(span_p > 0, (np.asarray(log_p_arr) - ulp[ip]) / span_p, 0.0)
    dt = np.where(span_t > 0, (np.asarray(log_t_arr) - ult[it]) / span_t, 0.0)
    dp = np.clip(dp, 0.0, 1.0)
    dt = np.clip(dt, 0.0, 1.0)

    # Four corners
    v00 = grid[ip, it]
    v01 = grid[ip, it + 1]
    v10 = grid[ip + 1, it]
    v11 = grid[ip + 1, it + 1]

    # Bilinear blend
    return v00 * (1 - dp) * (1 - dt) + v01 * (1 - dp) * dt + v10 * dp * (1 - dt) + v11 * dp * dt


def _paleos_clamp_temperature(log_p, log_t, cached):
    """Clamp log10(T) to the per-pressure valid range of a PALEOS table.

    Parameters
    ----------
    log_p : float
        log10 of pressure in Pa (already clamped to table bounds).
    log_t : float
        log10 of temperature in K.
    cached : dict
        PALEOS cache entry from ``load_paleos_table()``.

    Returns
    -------
    float
        Clamped log10(T), guaranteed to fall within the valid data region
        at the given pressure.
    bool
        True if clamping was applied, False if the original value was valid.
    """
    ulp = cached['unique_log_p']
    lt_min = cached['logt_valid_min']
    lt_max = cached['logt_valid_max']

    # Interpolate per-pressure T bounds at the query pressure
    local_tmin = float(np.interp(log_p, ulp, lt_min))
    local_tmax = float(np.interp(log_p, ulp, lt_max))

    # Guard: if the pressure is near an all-NaN row, the interpolated
    # bounds are NaN. Return unclamped and let the NN fallback handle it.
    if not (np.isfinite(local_tmin) and np.isfinite(local_tmax)):
        return log_t, False

    if log_t < local_tmin:
        return local_tmin, True
    elif log_t > local_tmax:
        return local_tmax, True
    return log_t, False


def _ensure_unified_cache(eos_file, interpolation_functions):
    """Ensure a unified PALEOS table is loaded into the interpolation cache.

    Tries loading from a binary pickle cache first (fast), then falls back
    to the text table (slow). Saves a pickle cache after the first text load.

    Parameters
    ----------
    eos_file : str
        Path to the unified PALEOS table file.
    interpolation_functions : dict
        Shared interpolation cache.

    Returns
    -------
    dict
        The cache entry for this file.
    """
    if eos_file not in interpolation_functions:
        import pickle

        cache_path = eos_file.replace('.dat', '.pkl')
        try:
            with open(cache_path, 'rb') as f:
                interpolation_functions[eos_file] = pickle.load(f)
            logger.debug('Loaded PALEOS cache from %s', cache_path)
        except (FileNotFoundError, EOFError, pickle.UnpicklingError):
            logger.info('Loading PALEOS table from text: %s', eos_file)
            interpolation_functions[eos_file] = load_paleos_unified_table(eos_file)
            # Save binary cache for next time
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(interpolation_functions[eos_file], f, protocol=4)
                logger.info('Saved PALEOS binary cache: %s', cache_path)
            except OSError:
                logger.debug('Could not save cache to %s', cache_path)

    return interpolation_functions[eos_file]
