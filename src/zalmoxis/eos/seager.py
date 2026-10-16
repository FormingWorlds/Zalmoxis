"""Seager2007 1D P-rho table lookups and related tabulated EOS routines.

Handles loading and interpolating tabulated EOS data for core, mantle,
and ice layer materials, including both 1D (P-rho) and 2D (P-T-rho)
table formats.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from scipy.interpolate import (
    LinearNDInterpolator,
    RegularGridInterpolator,
    interp1d,
)

from .interpolation import (
    _fast_bilinear,
    _paleos_clamp_temperature,
    _paleos_clamp_warned,
    load_paleos_table,
    require_eos_file,
)

logger = logging.getLogger(__name__)

_tdep_edge_warned: set[str] = set()


def _fill_invalid_cells(density_grid, unique_temps):
    """Fill each pressure row's invalid cells with the row's nearest valid value along T.

    Cells that are not finite or at or below 1e-10 are table fill (the Wolf & Bower
    tables use 1e-15). Below and above a row's valid band the fill becomes the band
    edge, and inside a gap the nearest valid node in T (the lower one on a tie), so a
    bilinear query only ever combines valid densities. A row with no valid cell stays NaN.

    Parameters
    ----------
    density_grid : numpy.ndarray
        Densities on the (pressure, temperature) grid, shape (n_p, n_t).
    unique_temps : numpy.ndarray
        Temperature axis, in K.

    Returns
    -------
    tuple of numpy.ndarray
        The filled grid, and the lowest and highest valid T of each row (NaN for a
        row without a valid cell).
    """
    ok = np.isfinite(density_grid) & (density_grid > 1e-10)
    filled = np.full(density_grid.shape, np.nan)
    t_lo = np.full(len(density_grid), np.nan)
    t_hi = np.full(len(density_grid), np.nan)
    for i, row in enumerate(ok):
        j = np.flatnonzero(row)
        if len(j) == 0:
            continue
        k = np.arange(row.size)
        right = np.clip(np.searchsorted(j, k), 0, len(j) - 1)
        left = np.clip(right - 1, 0, len(j) - 1)
        t = unique_temps
        nearest = np.where(
            np.abs(t[j[left]] - t[k]) <= np.abs(t[j[right]] - t[k]), j[left], j[right]
        )
        filled[i] = density_grid[i, nearest]
        t_lo[i], t_hi[i] = unique_temps[j[0]], unique_temps[j[-1]]
    return filled, t_lo, t_hi


def _warn_beyond_row_edge(cached, eos_file, pressure, temperature):
    """Warn once per table when T lies outside the valid band of a pressure row that the query uses.

    A row is used when its bilinear weight is not zero, so a query on a pressure node
    looks only at that node's row; a row without a valid cell is not a band.

    Parameters
    ----------
    cached : dict
        Regular-grid cache entry with 'unique_pressures', 't_lo' and 't_hi'.
    eos_file : str
        Table path, named in the warning.
    pressure : float
        Pressure in Pa, already inside the table range.
    temperature : float
        Temperature in K.
    """
    if eos_file in _tdep_edge_warned:
        return
    up = cached['unique_pressures']
    i = min(max(int(np.searchsorted(up, pressure, side='right')) - 1, 0), len(up) - 2)
    rows = [r for r, w in ((i, up[i + 1] - pressure), (i + 1, pressure - up[i])) if w > 0]
    for r in rows:
        lo, hi = cached['t_lo'][r], cached['t_hi'][r]
        if np.isfinite(lo) and not lo <= temperature <= hi:
            _tdep_edge_warned.add(eos_file)
            logger.warning(
                f'{os.path.basename(eos_file)}: T={temperature:.0f} K at P={pressure:.2e} Pa is '
                f'outside the valid band [{lo:.0f}, {hi:.0f}] K of the pressure row at '
                f'{up[r]:.2e} Pa; the density is taken at the band edge. Further queries '
                f'outside a band are not reported.'
            )
            return


def get_tabulated_eos(
    pressure, material_dictionary, material, temperature=None, interpolation_functions=None
):
    """Retrieve density from tabulated EOS data for a given material.

    Parameters
    ----------
    pressure : float
        Pressure at which to evaluate the EOS, in Pa.
    material_dictionary : dict
        Dictionary containing material properties and EOS file paths.
    material : str
        Material type, for example ``"core"``, ``"mantle"``,
        ``"ice_layer"``, ``"melted_mantle"``, or ``"solid_mantle"``.
    temperature : float, optional
        Temperature at which to evaluate the EOS, in K. Required for
        temperature-dependent materials such as ``"melted_mantle"`` and
        ``"solid_mantle"``.
    interpolation_functions : dict, optional
        Cache of interpolation functions used to avoid reloading and
        rebuilding interpolators for EOS tables.

    Returns
    -------
    float or None
        Density in kg/m^3 if the interpolation succeeds, otherwise ``None``.
    """
    if interpolation_functions is None:
        interpolation_functions = {}
    props = material_dictionary[material]
    eos_file = props['eos_file']
    is_paleos = props.get('format') == 'paleos'
    try:
        if eos_file not in interpolation_functions:
            require_eos_file(eos_file)
            if is_paleos:
                # PALEOS 10-column format with log-log grid
                interpolation_functions[eos_file] = load_paleos_table(eos_file)
            elif material == 'melted_mantle' or material == 'solid_mantle':
                # Load P-T-rho file
                data = np.loadtxt(eos_file, delimiter='\t', skiprows=1)
                pressures = data[:, 0]  # in Pa
                temps = data[:, 1]  # in K
                densities = data[:, 2]  # in kg/m^3
                unique_pressures = np.unique(pressures)
                unique_temps = np.unique(temps)

                is_regular = len(data) == len(unique_pressures) * len(unique_temps)

                if is_regular:
                    # Check if pressures and temps are sorted as expected
                    if not (
                        np.all(np.diff(unique_pressures) > 0)
                        and np.all(np.diff(unique_temps) > 0)
                    ):  # pragma: no cover - shipped Seager tables are sorted by construction; defensive
                        raise ValueError(
                            'Pressures or temperatures are not sorted as expected in EOS file.'
                        )

                    # Reshape densities to a 2D grid for interpolation
                    density_grid, t_lo, t_hi = _fill_invalid_cells(
                        densities.reshape(len(unique_pressures), len(unique_temps)),
                        unique_temps,
                    )

                    # Create a RegularGridInterpolator for rho(P,T)
                    interpolator = RegularGridInterpolator(
                        (unique_pressures, unique_temps),
                        density_grid,
                        bounds_error=False,
                        fill_value=None,
                    )
                    interpolation_functions[eos_file] = {
                        'type': 'regular',
                        'interp': interpolator,
                        'p_min': unique_pressures[0],
                        'p_max': unique_pressures[-1],
                        't_min': unique_temps[0],
                        't_max': unique_temps[-1],
                        'unique_pressures': unique_pressures,
                        't_lo': t_lo,
                        't_hi': t_hi,
                    }
                else:
                    # Irregular grid (e.g. RTPress100TPa melt table where the
                    # valid T range varies with P). Use scattered-data
                    # interpolation via Delaunay triangulation.
                    logger.info(
                        f'EOS file {eos_file} has irregular grid '
                        f'({len(data)} rows vs {len(unique_pressures)}x{len(unique_temps)} '
                        f'= {len(unique_pressures) * len(unique_temps)} expected). '
                        f'Using LinearNDInterpolator.'
                    )
                    # Work in log-P space for better triangulation of the
                    # logarithmically spaced pressure axis
                    log_pressures = np.log10(pressures)
                    interpolator = LinearNDInterpolator(
                        np.column_stack([log_pressures, temps]),
                        densities,
                    )
                    interpolation_functions[eos_file] = {
                        'type': 'irregular',
                        'interp': interpolator,
                        'p_min': unique_pressures[0],
                        'p_max': unique_pressures[-1],
                        't_min': unique_temps[0],
                        't_max': unique_temps[-1],
                        # Per-pressure T bounds for out-of-domain detection
                        'p_tmax': {p: temps[pressures == p].max() for p in unique_pressures},
                        'unique_pressures': unique_pressures,
                    }
            else:
                # Load rho-P file
                data = np.loadtxt(eos_file, delimiter=',', skiprows=1)
                pressure_data = data[:, 1] * 1e9  # Convert from GPa to Pa
                density_data = data[:, 0] * 1e3  # Convert from g/cm^3 to kg/m^3
                interpolation_functions[eos_file] = {
                    'type': '1d',
                    'interp': interp1d(
                        pressure_data,
                        density_data,
                        bounds_error=False,
                        fill_value='extrapolate',
                    ),
                }

        cached = interpolation_functions[eos_file]  # Retrieve from cache

        # Perform interpolation
        if cached['type'] == 'paleos':
            # PALEOS: interpolate in log10(P)-log10(T) space
            if temperature is None:
                raise ValueError('Temperature must be provided.')
            p_min, p_max = cached['p_min'], cached['p_max']

            # Clamp pressure to global bounds
            if pressure < p_min or pressure > p_max:
                logger.debug(
                    f'PALEOS: Pressure {pressure:.2e} Pa out of bounds '
                    f'[{p_min:.2e}, {p_max:.2e}]. Clamping.'
                )
                pressure = np.clip(pressure, p_min, p_max)

            log_p = np.log10(pressure)
            log_t = np.log10(max(temperature, 1.0))  # guard log10(0)

            # Per-cell clamping: restrict T to the valid data range at this P
            log_t_clamped, was_clamped = _paleos_clamp_temperature(log_p, log_t, cached)
            if was_clamped and eos_file not in _paleos_clamp_warned:
                _paleos_clamp_warned.add(eos_file)
                t_orig = temperature
                t_new = 10.0**log_t_clamped
                logger.warning(
                    f'PALEOS per-cell clamping active for {os.path.basename(eos_file)}: '
                    f'T={t_orig:.0f} K clamped to {t_new:.0f} K at P={pressure:.2e} Pa. '
                    f'The table has no valid data at this (P,T). '
                    f'Density values near the table boundary may be inaccurate.'
                )

            density = _fast_bilinear(log_p, log_t_clamped, cached['density_grid'], cached)

            # Nearest-neighbor fallback: bilinear interpolation returns NaN
            # when a valid cell neighbors a NaN cell near the ragged domain
            # boundary. Fall back to the nearest valid grid cell.
            if not np.isfinite(density):
                density = float(cached['density_nn']((log_p, log_t_clamped)))
        elif material == 'melted_mantle' or material == 'solid_mantle':
            if temperature is None:
                raise ValueError('Temperature must be provided.')

            grid_type = cached['type']
            p_min, p_max = cached['p_min'], cached['p_max']
            t_min, t_max = cached['t_min'], cached['t_max']

            # Global temperature bounds check
            if temperature < t_min or temperature > t_max:
                raise ValueError(
                    f'Temperature {temperature:.2f} K is out of bounds '
                    f'for EOS data [{t_min:.1f}, {t_max:.1f}].'
                )

            # Pressure clamping (both grid types)
            if pressure < p_min or pressure > p_max:
                logger.debug(
                    f'Pressure {pressure:.2e} Pa out of bounds for EOS table '
                    f'[{p_min:.2e}, {p_max:.2e}]. Clamping to boundary.'
                )
                pressure = np.clip(pressure, p_min, p_max)

            if grid_type == 'regular':
                _warn_beyond_row_edge(cached, eos_file, pressure, temperature)
                density = cached['interp']((pressure, temperature))
            else:
                up = cached['unique_pressures']
                idx = np.searchsorted(up, pressure, side='right')
                idx = min(idx, len(up) - 1)
                idx_lo = max(0, idx - 1)
                local_tmax_lo = cached['p_tmax'][up[idx_lo]]
                local_tmax_hi = cached['p_tmax'][up[idx]]
                if up[idx] != up[idx_lo]:
                    frac = (pressure - up[idx_lo]) / (up[idx] - up[idx_lo])
                    local_tmax = local_tmax_lo + frac * (local_tmax_hi - local_tmax_lo)
                else:
                    local_tmax = local_tmax_lo

                if temperature > local_tmax:
                    logger.debug(
                        f'Temperature {temperature:.1f} K exceeds local T_max '
                        f'{local_tmax:.1f} K at P={pressure:.2e} Pa. '
                        f'Clamping to boundary.'
                    )
                    temperature = local_tmax

                density = float(cached['interp']([[np.log10(pressure), temperature]])[0])
        else:
            density = cached['interp'](pressure)

        if density is None or not np.isfinite(density):
            raise ValueError(
                f'Density calculation failed for {material} ({os.path.basename(eos_file)}) '
                f'at P={pressure:.2e} Pa, T={temperature}.'
            )

        return density

    except (ValueError, OSError) as e:
        logger.error(
            f'Error with tabulated EOS for {material} at P={pressure:.2e} Pa, T={temperature}: {e}'
        )
        return None
    except Exception as e:
        logger.error(
            f'Unexpected error with tabulated EOS for {material} at P={pressure:.2e} Pa, T={temperature}: {e}'
        )
        return None
