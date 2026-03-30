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
)

logger = logging.getLogger(__name__)


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
                    ):
                        raise ValueError(
                            'Pressures or temperatures are not sorted as expected in EOS file.'
                        )

                    # Reshape densities to a 2D grid for interpolation
                    density_grid = densities.reshape(len(unique_pressures), len(unique_temps))

                    # Create a RegularGridInterpolator for rho(P,T)
                    interpolator = RegularGridInterpolator(
                        (unique_pressures, unique_temps),
                        density_grid,
                        method='cubic',
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
                density = cached['interp']((pressure, temperature))
            else:
                # Irregular grid: check per-pressure T upper bound.
                # Find the nearest pressure in the table to get the local T_max.
                up = cached['unique_pressures']
                idx = np.searchsorted(up, pressure, side='right')
                idx = min(idx, len(up) - 1)
                # Interpolate between the two nearest pressures' T_max
                idx_lo = max(0, idx - 1)
                local_tmax_lo = cached['p_tmax'][up[idx_lo]]
                local_tmax_hi = cached['p_tmax'][up[idx]]
                if up[idx] != up[idx_lo]:
                    frac = (pressure - up[idx_lo]) / (up[idx] - up[idx_lo])
                    local_tmax = local_tmax_lo + frac * (local_tmax_hi - local_tmax_lo)
                else:
                    local_tmax = local_tmax_lo

                if temperature > local_tmax:
                    # Clamp temperature to the local domain boundary
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
                f'Density calculation failed for {material} at P={pressure:.2e} Pa, T={temperature}.'
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
