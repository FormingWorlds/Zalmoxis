"""
EOS-related functions for ZALMOXIS, including loading tabulated EOS data,
performing interpolation, and calculating density based on pressure and
temperature.

Imports
-----
`eos_analytic`: provides `get_analytic_density`
"""

from __future__ import annotations

import logging
import os

import numpy as np
from scipy.interpolate import (
    LinearNDInterpolator,
    NearestNDInterpolator,
    RegularGridInterpolator,
    interp1d,
)

from .eos_analytic import get_analytic_density

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')

logger = logging.getLogger(__name__)


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
        'logt_valid_min': logt_valid_min,
        'logt_valid_max': logt_valid_max,
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
    # Read numeric columns (0-8)
    data_numeric = np.genfromtxt(eos_file, usecols=range(9), comments='#')

    # Read phase column (9) as strings
    phase_strings = np.genfromtxt(eos_file, usecols=(9,), dtype=str, comments='#')

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
        'logt_valid_min': logt_valid_min,
        'logt_valid_max': logt_valid_max,
        'liquidus_log_p': liquidus_log_p,
        'liquidus_log_t': liquidus_log_t,
        'phase_grid': phase_grid,
    }


# Track whether the per-cell clamping warning has been issued for each file,
# to avoid flooding the log with repeated messages.
_paleos_clamp_warned = set()


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


def get_tabulated_eos(
    pressure, material_dictionary, material, temperature=None, interpolation_functions=None
):
    """
    Retrieve density from tabulated EOS data for a given material.

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
                # Load P-T–ρ file
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

                    # Create a RegularGridInterpolator for ρ(P,T)
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
                    }
                else:
                    # Irregular grid (e.g. RTPress100TPa melt table where the
                    # valid T range varies with P). Use scattered-data
                    # interpolation via Delaunay triangulation.
                    logger.info(
                        f'EOS file {eos_file} has irregular grid '
                        f'({len(data)} rows vs {len(unique_pressures)}×{len(unique_temps)} '
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
                # Load ρ-P file
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

            density = float(cached['density_interp']((log_p, log_t_clamped)))

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


def load_melting_curve(melt_file):
    """
    Load a melting curve for MgSiO3 from a text file.

    Parameters
    ----------
    melt_file : str or path-like
        Path to the melting curve data file. The file is expected to contain
        two columns: pressure in Pa and temperature in K.

    Returns
    -------
    scipy.interpolate.interp1d or None
        One-dimensional interpolation function returning temperature as a
        function of pressure. Returns ``None`` if the file cannot be loaded.
    """
    try:
        data = np.loadtxt(melt_file, comments='#')
        pressures = data[:, 0]  # in Pa
        temperatures = data[:, 1]  # in K
        interp_func = interp1d(
            pressures, temperatures, kind='linear', bounds_error=False, fill_value=np.nan
        )
        return interp_func
    except Exception as e:
        print(f'Error loading melting curve data: {e}')
        return None


def get_solidus_liquidus_functions(
    solidus_id='Stixrude14-solidus', liquidus_id='Stixrude14-liquidus'
):
    """Load solidus and liquidus melting curves by config identifier.

    Delegates to :func:`zalmoxis.melting_curves.get_solidus_liquidus_functions`.

    Parameters
    ----------
    solidus_id : str
        Solidus curve identifier.
    liquidus_id : str
        Liquidus curve identifier.

    Returns
    -------
    tuple of callable
        ``(solidus_func, liquidus_func)``
    """
    from .melting_curves import get_solidus_liquidus_functions as _get

    return _get(solidus_id, liquidus_id)


def get_Tdep_density(
    pressure,
    temperature,
    material_properties_iron_Tdep_silicate_planets,
    solidus_func,
    liquidus_func,
    interpolation_functions=None,
):
    """
    Compute mantle density for a temperature-dependent EOS with phase changes.

    Parameters
    ----------
    pressure : float
        Pressure at which to evaluate the EOS, in Pa.
    temperature : float
        Temperature at which to evaluate the EOS, in K.
    material_properties_iron_Tdep_silicate_planets : dict
        Dictionary containing temperature-dependent material properties for
        the MgSiO3 EOS.
    solidus_func : callable
        Interpolation function for the solidus melting curve.
    liquidus_func : callable
        Interpolation function for the liquidus melting curve.
    interpolation_functions : dict, optional
        Cache of interpolation functions used to avoid redundant loading of
        EOS tables.

    Returns
    -------
    float or None
        Density in kg/m^3. Returns ``None`` if the density cannot be evaluated.

    Raises
    ------
    ValueError
        If ``solidus_func`` or ``liquidus_func`` is not provided.

    Notes
    -----
    The mantle phase is determined by comparing the input temperature to the
    solidus and liquidus temperatures at the given pressure.

    In the mixed-phase region, the density is computed using linear melt
    fraction and volume additivity.
    """

    if interpolation_functions is None:
        interpolation_functions = {}

    if solidus_func is None or liquidus_func is None:
        raise ValueError(
            'solidus_func and liquidus_func must be provided for WolfBower2018:MgSiO3 EOS.'
        )

    T_sol = solidus_func(pressure)
    T_liq = liquidus_func(pressure)

    # Pressure outside melting curve range — default to solid phase
    if np.isnan(T_sol) or np.isnan(T_liq):
        logger.debug(
            f'Melting curve undefined at P={pressure:.2e} Pa. Defaulting to solid phase.'
        )
        return get_tabulated_eos(
            pressure,
            material_properties_iron_Tdep_silicate_planets,
            'solid_mantle',
            temperature,
            interpolation_functions,
        )

    if temperature <= T_sol:
        # Solid phase
        rho = get_tabulated_eos(
            pressure,
            material_properties_iron_Tdep_silicate_planets,
            'solid_mantle',
            temperature,
            interpolation_functions,
        )
        return rho

    elif temperature >= T_liq:
        # Liquid phase
        rho = get_tabulated_eos(
            pressure,
            material_properties_iron_Tdep_silicate_planets,
            'melted_mantle',
            temperature,
            interpolation_functions,
        )
        return rho

    else:
        # Mixed phase: linear melt fraction between solidus and liquidus.
        # Guard against degenerate melting curves where T_liq == T_sol.
        if T_liq <= T_sol:
            return get_tabulated_eos(
                pressure,
                material_properties_iron_Tdep_silicate_planets,
                'melted_mantle',
                temperature,
                interpolation_functions,
            )
        frac_melt = (temperature - T_sol) / (T_liq - T_sol)
        rho_solid = get_tabulated_eos(
            pressure,
            material_properties_iron_Tdep_silicate_planets,
            'solid_mantle',
            temperature,
            interpolation_functions,
        )
        rho_liquid = get_tabulated_eos(
            pressure,
            material_properties_iron_Tdep_silicate_planets,
            'melted_mantle',
            temperature,
            interpolation_functions,
        )
        # Guard against out-of-bounds pressure returning None
        if rho_solid is None or rho_liquid is None:
            return None
        # Calculate mixed density by volume additivity
        specific_volume_mixed = frac_melt * (1 / rho_liquid) + (1 - frac_melt) * (1 / rho_solid)
        rho_mixed = 1 / specific_volume_mixed
        return rho_mixed


def get_Tdep_material(pressure, temperature, solidus_func, liquidus_func):
    """
    Determine the mantle phase for a temperature-dependent EOS.

    Parameters
    ----------
    pressure : float or array_like
        Pressure in Pa. May be a scalar or an array.
    temperature : float or array_like
        Temperature in K. May be a scalar or an array.
    solidus_func : callable
        Interpolation function for the solidus melting curve.
    liquidus_func : callable
        Interpolation function for the liquidus melting curve.

    Returns
    -------
    str or numpy.ndarray
        Material phase label(s). Possible values are ``"solid_mantle"``,
        ``"mixed_mantle"``, and ``"melted_mantle"``. Returns a string for
        scalar inputs and a NumPy array of strings for array inputs.
    """

    # Define per-point evaluation
    def evaluate_phase(P, T):
        T_sol = solidus_func(P)
        T_liq = liquidus_func(P)
        # Guard against degenerate melting curves where T_liq == T_sol
        if T_liq <= T_sol:
            return 'melted_mantle' if T >= T_sol else 'solid_mantle'
        frac_melt = (T - T_sol) / (T_liq - T_sol)
        if frac_melt < 0:
            return 'solid_mantle'
        elif frac_melt <= 1.0:
            return 'mixed_mantle'
        else:
            return 'melted_mantle'

    # Vectorize function for array support
    vectorized_eval = np.vectorize(evaluate_phase, otypes=[str])

    # Apply depending on input type
    if np.isscalar(pressure) and np.isscalar(temperature):
        return evaluate_phase(pressure, temperature)
    else:
        return vectorized_eval(pressure, temperature)


def _ensure_unified_cache(eos_file, interpolation_functions):
    """Ensure a unified PALEOS table is loaded into the interpolation cache.

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
        interpolation_functions[eos_file] = load_paleos_unified_table(eos_file)
    return interpolation_functions[eos_file]


def get_paleos_unified_density(
    pressure, temperature, material_dict, mushy_zone_factor, interpolation_functions
):
    """Look up density from a unified PALEOS table.

    When ``mushy_zone_factor == 1.0`` (no mushy zone), the density is read
    directly from the table (the stable phase at each (P, T) is already
    encoded). When ``mushy_zone_factor < 1.0``, a synthetic solidus is
    derived as ``T_sol = T_liq * mushy_zone_factor`` and the density in the
    mushy zone is volume-averaged between the solid-side and liquid-side
    table values.

    Parameters
    ----------
    pressure : float
        Pressure in Pa.
    temperature : float
        Temperature in K.
    material_dict : dict
        Material properties dict with 'eos_file' and 'format' keys.
    mushy_zone_factor : float
        Cryoscopic depression factor. 1.0 = no mushy zone (sharp boundary).
        < 1.0 = solidus at this fraction of the extracted liquidus.
    interpolation_functions : dict
        Shared interpolation cache.

    Returns
    -------
    float or None
        Density in kg/m^3, or None on failure.
    """
    eos_file = material_dict['eos_file']
    try:
        cached = _ensure_unified_cache(eos_file, interpolation_functions)

        p_min, p_max = cached['p_min'], cached['p_max']
        if pressure < p_min or pressure > p_max:
            logger.debug(
                f'PALEOS unified: P={pressure:.2e} Pa out of bounds '
                f'[{p_min:.2e}, {p_max:.2e}]. Clamping.'
            )
            pressure = np.clip(pressure, p_min, p_max)

        log_p = np.log10(pressure)
        log_t = np.log10(max(temperature, 1.0))

        # Per-cell clamping
        log_t_clamped, was_clamped = _paleos_clamp_temperature(log_p, log_t, cached)
        if was_clamped and eos_file not in _paleos_clamp_warned:
            _paleos_clamp_warned.add(eos_file)
            logger.warning(
                f'PALEOS unified per-cell clamping active for '
                f'{os.path.basename(eos_file)}: '
                f'T={temperature:.0f} K clamped to {10.0**log_t_clamped:.0f} K '
                f'at P={pressure:.2e} Pa.'
            )

        if mushy_zone_factor >= 1.0 or len(cached['liquidus_log_p']) == 0:
            # Direct lookup: no mushy zone
            density = float(cached['density_interp']((log_p, log_t_clamped)))
            if not np.isfinite(density):
                density = float(cached['density_nn']((log_p, log_t_clamped)))
            return density if np.isfinite(density) else None

        # Mushy zone: interpolate liquidus T at this P.
        # If query pressure is outside the liquidus coverage (e.g. at
        # pressures where no liquid phase exists), fall back to direct lookup.
        liq_lp = cached['liquidus_log_p']
        if log_p < liq_lp[0] or log_p > liq_lp[-1]:
            density = float(cached['density_interp']((log_p, log_t_clamped)))
            if not np.isfinite(density):
                density = float(cached['density_nn']((log_p, log_t_clamped)))
            return density if np.isfinite(density) else None

        log_t_liq = float(np.interp(log_p, liq_lp, cached['liquidus_log_t']))
        T_liq = 10.0**log_t_liq
        T_sol = T_liq * mushy_zone_factor
        log_t_sol = np.log10(max(T_sol, 1.0))

        if temperature >= T_liq:
            # Above liquidus: direct lookup
            density = float(cached['density_interp']((log_p, log_t_clamped)))
            if not np.isfinite(density):
                density = float(cached['density_nn']((log_p, log_t_clamped)))
            return density if np.isfinite(density) else None

        if temperature <= T_sol:
            # Below solidus: direct lookup
            density = float(cached['density_interp']((log_p, log_t_clamped)))
            if not np.isfinite(density):
                density = float(cached['density_nn']((log_p, log_t_clamped)))
            return density if np.isfinite(density) else None

        # In mushy zone: volume-average between solid-side and liquid-side
        phi = (temperature - T_sol) / (T_liq - T_sol)

        # Solid-side: density at T_sol
        log_t_sol_c, _ = _paleos_clamp_temperature(log_p, log_t_sol, cached)
        rho_sol = float(cached['density_interp']((log_p, log_t_sol_c)))
        if not np.isfinite(rho_sol):
            rho_sol = float(cached['density_nn']((log_p, log_t_sol_c)))

        # Liquid-side: density at T_liq
        log_t_liq_c, _ = _paleos_clamp_temperature(log_p, log_t_liq, cached)
        rho_liq = float(cached['density_interp']((log_p, log_t_liq_c)))
        if not np.isfinite(rho_liq):
            rho_liq = float(cached['density_nn']((log_p, log_t_liq_c)))

        if not (np.isfinite(rho_sol) and np.isfinite(rho_liq)):
            return None

        # Volume additivity
        specific_volume = phi * (1.0 / rho_liq) + (1.0 - phi) * (1.0 / rho_sol)
        return 1.0 / specific_volume

    except Exception as e:
        logger.error(
            f'Error in PALEOS unified density at P={pressure:.2e} Pa, '
            f'T={temperature:.1f} K: {e}'
        )
        return None


def _get_paleos_unified_nabla_ad(pressure, temperature, material_dict, interpolation_functions):
    """Look up nabla_ad from a unified PALEOS cache entry.

    Parameters
    ----------
    pressure : float
        Pressure in Pa.
    temperature : float
        Temperature in K.
    material_dict : dict
        Material properties dict with 'eos_file' key.
    interpolation_functions : dict
        Shared interpolation cache.

    Returns
    -------
    float or None
        Dimensionless adiabatic gradient, or None if lookup fails.
    """
    eos_file = material_dict['eos_file']
    cached = _ensure_unified_cache(eos_file, interpolation_functions)

    p_clamped = np.clip(pressure, cached['p_min'], cached['p_max'])
    log_p = np.log10(p_clamped)
    log_t = np.log10(max(temperature, 1.0))

    log_t_clamped, was_clamped = _paleos_clamp_temperature(log_p, log_t, cached)
    if was_clamped and eos_file not in _paleos_clamp_warned:
        _paleos_clamp_warned.add(eos_file)
        logger.warning(
            f'PALEOS unified per-cell clamping active for nabla_ad in '
            f'{os.path.basename(eos_file)}: '
            f'T={temperature:.0f} K clamped to {10.0**log_t_clamped:.0f} K '
            f'at P={pressure:.2e} Pa.'
        )

    val = float(cached['nabla_ad_interp']((log_p, log_t_clamped)))
    if not np.isfinite(val):
        val = float(cached['nabla_ad_nn']((log_p, log_t_clamped)))

    return val if np.isfinite(val) else None


def calculate_density(
    pressure,
    material_dictionaries,
    layer_eos,
    temperature,
    solidus_func,
    liquidus_func,
    interpolation_functions=None,
    mushy_zone_factor=1.0,
):
    """
    Calculate density for a single layer given its EOS identifier.

    Parameters
    ----------
    pressure : float
        Pressure at which to evaluate the EOS, in Pa.
    material_dictionaries : dict
        EOS registry dict keyed by EOS identifier string
        (from ``eos_properties.EOS_REGISTRY``).
    layer_eos : str
        Per-layer EOS identifier, for example ``"Seager2007:iron"``,
        ``"WolfBower2018:MgSiO3"``, ``"PALEOS:iron"``, or ``"Analytic:iron"``.
    temperature : float
        Temperature at which to evaluate the EOS, in K.
    solidus_func : callable or None
        Interpolation function for the solidus melting curve.
    liquidus_func : callable or None
        Interpolation function for the liquidus melting curve.
    interpolation_functions : dict, optional
        Cache of interpolation functions used to avoid redundant loading.
    mushy_zone_factor : float, optional
        Cryoscopic depression factor for unified PALEOS tables. 1.0 = no
        mushy zone (sharp boundary from table). Default 1.0.

    Returns
    -------
    float or None
        Density in kg/m^3, or ``None`` on failure.

    Raises
    ------
    ValueError
        If ``layer_eos`` is not recognized.
    """
    if interpolation_functions is None:
        interpolation_functions = {}

    # Analytic EOS: no material dict needed
    if layer_eos.startswith('Analytic:'):
        material_key = layer_eos.split(':', 1)[1]
        return get_analytic_density(pressure, material_key)

    # Look up material properties from the registry
    mat = material_dictionaries.get(layer_eos)
    if mat is None:
        raise ValueError(f"Unknown layer EOS '{layer_eos}'.")

    # Unified PALEOS tables (single file per material, all phases included)
    if mat.get('format') == 'paleos_unified':
        return get_paleos_unified_density(
            pressure, temperature, mat, mushy_zone_factor, interpolation_functions
        )

    # T-dependent EOS with separate solid/liquid tables (WB2018, RTPress, PALEOS-2phase)
    if 'melted_mantle' in mat:
        return get_Tdep_density(
            pressure, temperature, mat, solidus_func, liquidus_func, interpolation_functions
        )

    # Seager2007 static EOS (1D P-rho tables)
    # Determine the layer key from the material dict
    for layer_key in ('core', 'mantle', 'ice_layer'):
        if layer_key in mat:
            return get_tabulated_eos(
                pressure, mat, layer_key, interpolation_functions=interpolation_functions
            )

    raise ValueError(f"Cannot determine layer key for EOS '{layer_eos}'.")


def _get_paleos_nabla_ad(pressure, temperature, material_dict, phase, interpolation_functions):
    """Look up nabla_ad from a PALEOS cache entry.

    Parameters
    ----------
    pressure : float
        Pressure in Pa.
    temperature : float
        Temperature in K.
    material_dict : dict
        Material properties dict (e.g. ``material_properties_iron_PALEOS_silicate_planets``).
    phase : str
        ``'solid_mantle'`` or ``'melted_mantle'``.
    interpolation_functions : dict
        Shared interpolation cache.

    Returns
    -------
    float or None
        Dimensionless adiabatic gradient nabla_ad, or None if lookup fails.
    """
    props = material_dict[phase]
    eos_file = props['eos_file']

    # Ensure the PALEOS table is loaded into the cache
    if eos_file not in interpolation_functions:
        interpolation_functions[eos_file] = load_paleos_table(eos_file)

    cached = interpolation_functions[eos_file]
    p_min, p_max = cached['p_min'], cached['p_max']

    # Clamp pressure to global bounds
    p_clamped = np.clip(pressure, p_min, p_max)
    log_p = np.log10(p_clamped)
    log_t = np.log10(max(temperature, 1.0))

    # Per-cell clamping: restrict T to the valid data range at this P
    log_t_clamped, was_clamped = _paleos_clamp_temperature(log_p, log_t, cached)
    if was_clamped and eos_file not in _paleos_clamp_warned:
        _paleos_clamp_warned.add(eos_file)
        logger.warning(
            f'PALEOS per-cell clamping active for nabla_ad in '
            f'{os.path.basename(eos_file)}: '
            f'T={temperature:.0f} K clamped to {10.0**log_t_clamped:.0f} K '
            f'at P={pressure:.2e} Pa.'
        )

    val = float(cached['nabla_ad_interp']((log_p, log_t_clamped)))

    # Nearest-neighbor fallback for ragged boundary
    if not np.isfinite(val):
        val = float(cached['nabla_ad_nn']((log_p, log_t_clamped)))

    if np.isfinite(val):
        return val
    return None


def compute_adiabatic_temperature(
    radii,
    pressure,
    mass_enclosed,
    surface_temperature,
    cmb_mass,
    core_mantle_mass,
    layer_eos_config,
    material_dictionaries,
    interpolation_functions=None,
    solidus_func=None,
    liquidus_func=None,
    mushy_zone_factor=1.0,
):
    """
    Compute an adiabatic temperature profile using native EOS gradient tables.

    For WolfBower2018 and RTPress100TPa, uses ``(dT/dP)_S`` tables (liquid
    phase only). For PALEOS-2phase, uses ``nabla_ad`` from both solid and
    liquid tables with melt-fraction weighting. For unified PALEOS tables
    (PALEOS:iron, PALEOS:MgSiO3, PALEOS:H2O), uses ``nabla_ad`` directly
    from the single table.

    Parameters
    ----------
    radii : numpy.ndarray
        Radial grid in ascending order from center to surface, in m.
    pressure : numpy.ndarray
        Pressure at each radius, in Pa.
    mass_enclosed : numpy.ndarray
        Enclosed mass at each radius, in kg.
    surface_temperature : float
        Temperature at the surface, in K.
    cmb_mass : float
        Core-mantle boundary mass, in kg.
    core_mantle_mass : float
        Total core-plus-mantle mass, in kg.
    layer_eos_config : dict
        Per-layer EOS configuration, for example
        ``{"core": "PALEOS:iron", "mantle": "PALEOS:MgSiO3"}``.
    material_dictionaries : dict
        EOS registry dict keyed by EOS identifier string.
    interpolation_functions : dict, optional
        Cache of interpolation functions used to avoid redundant loading.
    solidus_func : callable, optional
        Interpolation function for the solidus melting curve. Required for
        PALEOS-2phase phase-aware adiabat.
    liquidus_func : callable, optional
        Interpolation function for the liquidus melting curve. Required for
        PALEOS-2phase phase-aware adiabat.
    mushy_zone_factor : float, optional
        Cryoscopic depression factor for unified PALEOS tables. Default 1.0.

    Returns
    -------
    numpy.ndarray
        Temperature at each radial point, in K.

    Raises
    ------
    ValueError
        If adiabatic mode is requested but no layer uses a T-dependent EOS.
    ValueError
        If the selected mantle EOS requires an ``adiabat_grad_file`` but none
        is provided (WolfBower2018/RTPress only).

    Notes
    -----
    This is a standalone ZALMOXIS feature for structure calculations where no
    external interior solver provides ``T(r)``.

    In the PROTEUS-SPIDER coupling, this function is not called because SPIDER
    computes its own temperature profile through entropy evolution.

    The integration proceeds inward from the surface according to

    ``T[i] = T[i+1] + (dT/dP)_S * (P[i] - P[i+1])``.

    For PALEOS, ``(dT/dP)_S = nabla_ad * T / P``, where nabla_ad is the
    dimensionless adiabatic gradient read from the PALEOS tables.

    For temperature-independent EOS layers, such as a Seager (2007) iron core,
    the temperature is held constant.
    """
    from .constants import TDEP_EOS_NAMES
    from .structure_model import get_layer_eos

    if interpolation_functions is None:
        interpolation_functions = {}

    # Check which layer EOS values are T-dependent
    tdep_eos_used = {v for v in layer_eos_config.values() if v in TDEP_EOS_NAMES}

    if not tdep_eos_used:
        raise ValueError(
            f'Adiabatic temperature mode requires at least one T-dependent EOS '
            f'in the layer config, but none found in {layer_eos_config}. '
            f"Use 'linear' or 'isothermal' instead."
        )

    mantle_eos = layer_eos_config.get('mantle')

    # Determine which EOS types are in use for adiabat computation
    use_paleos_2phase = 'PALEOS-2phase:MgSiO3' in tdep_eos_used

    # Preload PALEOS-2phase material dict if needed (for any layer, not just mantle)
    mat_PALEOS_2ph = (
        material_dictionaries.get('PALEOS-2phase:MgSiO3') if use_paleos_2phase else None
    )

    # Build WolfBower/RTPress dT/dP wrapper dicts
    dtdp_dicts = {}
    for eos_name in ('WolfBower2018:MgSiO3', 'RTPress100TPa:MgSiO3'):
        mat = material_dictionaries.get(eos_name, {})
        grad_file = mat.get('melted_mantle', {}).get('adiabat_grad_file')
        if grad_file is not None:
            dtdp_dicts[eos_name] = {'melted_mantle': {'eos_file': grad_file}}

    # Validate that non-unified T-dep EOS have their required data
    if mantle_eos in ('WolfBower2018:MgSiO3', 'RTPress100TPa:MgSiO3'):
        if mantle_eos not in dtdp_dicts:
            raise ValueError(
                f"Adiabatic mode requires an 'adiabat_grad_file' in the "
                f"melted_mantle material dictionary for EOS '{mantle_eos}'. "
                f'Run get_zalmoxis.sh to download the required tables.'
            )

    n = len(radii)
    T = np.zeros(n)
    T[n - 1] = surface_temperature

    # Integrate from surface (index n-1) inward (decreasing index).
    # NOTE: No thermal boundary layer is modeled at the CMB. The adiabat
    # transitions directly from mantle to core, producing an isothermal
    # core at the CMB temperature (unless the core EOS is also T-dependent).
    for i in range(n - 2, -1, -1):
        layer_eos = get_layer_eos(
            mass_enclosed[i],
            cmb_mass,
            core_mantle_mass,
            layer_eos_config,
        )

        if layer_eos not in TDEP_EOS_NAMES:
            # T-independent EOS: isothermal
            T[i] = T[i + 1]
            continue

        P_eval = pressure[i + 1]
        T_eval = T[i + 1]
        dP = pressure[i] - pressure[i + 1]

        # Determine dT/dP based on the EOS type
        layer_mat = material_dictionaries.get(layer_eos, {})
        dtdp = None

        if layer_mat.get('format') == 'paleos_unified':
            # Unified PALEOS: nabla_ad directly from the table.
            # NOTE: when mushy_zone_factor < 1.0, the density is volume-averaged
            # in the mushy zone but nabla_ad is not weighted. This is a known
            # simplification; implementing mushy-zone weighting for nabla_ad
            # would require the same liquidus extraction logic as the density
            # lookup. For the default mushy_zone_factor=1.0 (sharp boundary),
            # this is exact.
            nabla = _get_paleos_unified_nabla_ad(
                P_eval, T_eval, layer_mat, interpolation_functions
            )
            if nabla is not None and nabla > 0 and P_eval > 0 and T_eval > 0:
                dtdp = nabla * T_eval / P_eval

        elif layer_eos == 'PALEOS-2phase:MgSiO3':
            dtdp = _compute_paleos_dtdp(
                P_eval,
                T_eval,
                mat_PALEOS_2ph,
                solidus_func,
                liquidus_func,
                interpolation_functions,
            )

        elif layer_eos in dtdp_dicts:
            dtdp = get_tabulated_eos(
                P_eval,
                dtdp_dicts[layer_eos],
                'melted_mantle',
                T_eval,
                interpolation_functions,
            )

        if dtdp is not None and dtdp > 0:
            T[i] = T_eval + dtdp * dP
        else:
            T[i] = T_eval

    return T


def _compute_paleos_dtdp(
    pressure, temperature, mat_PALEOS, solidus_func, liquidus_func, interpolation_functions
):
    """Compute dT/dP from PALEOS nabla_ad with phase-aware weighting.

    Uses solid nabla_ad below solidus, liquid nabla_ad above liquidus,
    and melt-fraction-weighted average in the mushy zone.

    Parameters
    ----------
    pressure : float
        Pressure in Pa.
    temperature : float
        Temperature in K.
    mat_PALEOS : dict
        PALEOS material properties dict.
    solidus_func : callable or None
        Solidus melting curve interpolation function.
    liquidus_func : callable or None
        Liquidus melting curve interpolation function.
    interpolation_functions : dict
        Shared interpolation cache.

    Returns
    -------
    float or None
        dT/dP in K/Pa, or None if lookup fails.
    """
    if pressure <= 0 or temperature <= 0:
        return None

    # Determine phase
    T_sol = solidus_func(pressure) if solidus_func is not None else np.nan
    T_liq = liquidus_func(pressure) if liquidus_func is not None else np.nan

    if np.isnan(T_sol) or np.isnan(T_liq) or T_liq <= T_sol:
        # Outside melting curve range or degenerate: use solid table
        nabla = _get_paleos_nabla_ad(
            pressure, temperature, mat_PALEOS, 'solid_mantle', interpolation_functions
        )
    elif temperature <= T_sol:
        # Solid phase
        nabla = _get_paleos_nabla_ad(
            pressure, temperature, mat_PALEOS, 'solid_mantle', interpolation_functions
        )
    elif temperature >= T_liq:
        # Liquid phase
        nabla = _get_paleos_nabla_ad(
            pressure, temperature, mat_PALEOS, 'melted_mantle', interpolation_functions
        )
    else:
        # Mixed phase: melt-fraction-weighted nabla_ad
        phi = (temperature - T_sol) / (T_liq - T_sol)
        nabla_solid = _get_paleos_nabla_ad(
            pressure, temperature, mat_PALEOS, 'solid_mantle', interpolation_functions
        )
        nabla_liquid = _get_paleos_nabla_ad(
            pressure, temperature, mat_PALEOS, 'melted_mantle', interpolation_functions
        )
        if nabla_solid is None and nabla_liquid is None:
            return None
        # Fall back to whichever is available if one is None
        if nabla_solid is None:
            nabla = nabla_liquid
        elif nabla_liquid is None:
            nabla = nabla_solid
        else:
            nabla = (1.0 - phi) * nabla_solid + phi * nabla_liquid

    if nabla is None or nabla <= 0:
        return None

    # Convert: dT/dP = nabla_ad * T / P
    return nabla * temperature / pressure


def calculate_temperature_profile(
    radii,
    temperature_mode,
    surface_temperature,
    center_temperature,
    input_dir,
    temp_profile_file,
):
    """
    Return a callable temperature profile for a planetary interior model.

    Parameters
    ----------
    radii : array_like
        Radial grid of the planet, in m.
    temperature_mode : {"isothermal", "linear", "prescribed", "adiabatic"}
        Temperature profile mode.

        - ``"isothermal"``: constant temperature equal to
          ``surface_temperature``.
        - ``"linear"``: linear profile from ``center_temperature`` at
          ``r = 0`` to ``surface_temperature`` at the surface.
        - ``"prescribed"``: read the temperature profile from a text file.
        - ``"adiabatic"``: return a linear profile as an initial guess; the
          actual adiabat is computed elsewhere in the main iteration loop.
    surface_temperature : float
        Temperature at the surface, in K.
    center_temperature : float
        Temperature at the center, in K. Used for ``"linear"`` and
        ``"adiabatic"`` modes.
    input_dir : str or path-like
        Directory containing the prescribed temperature profile file.
    temp_profile_file : str
        Name of the file containing the prescribed temperature profile from
        center to surface. Required when ``temperature_mode="prescribed"``.

    Returns
    -------
    callable
        Function of radius returning temperature in K. The callable accepts a
        scalar or array-like radius and returns a float or NumPy array.

    Raises
    ------
    ValueError
        If ``temperature_mode="prescribed"`` and the file does not exist.
    ValueError
        If the prescribed temperature profile length does not match
        ``radii``.
    ValueError
        If ``temperature_mode`` is not recognized.
    """
    radii = np.array(radii)

    if temperature_mode == 'isothermal':
        return lambda r: np.full_like(r, surface_temperature, dtype=float)

    elif temperature_mode == 'linear':
        return lambda r: (
            surface_temperature
            + (center_temperature - surface_temperature) * (1 - np.array(r) / radii[-1])
        )

    elif temperature_mode == 'prescribed':
        temp_profile_path = os.path.join(input_dir, temp_profile_file)
        if not os.path.exists(temp_profile_path):
            raise ValueError(
                "Temperature profile file must be provided and exist for 'prescribed' temperature mode."
            )
        temp_profile = np.loadtxt(temp_profile_path)
        if len(temp_profile) != len(radii):
            raise ValueError('Temperature profile length does not match radii length.')
        # Vectorized interpolation for arbitrary radius points
        return lambda r: np.interp(np.array(r), radii, temp_profile)

    elif temperature_mode == 'adiabatic':
        # Return linear profile as initial guess for the first outer iteration.
        # The actual adiabat is computed in main() using P(r), g(r) from the solver.
        return lambda r: (
            surface_temperature
            + (center_temperature - surface_temperature) * (1 - np.array(r) / radii[-1])
        )

    else:
        raise ValueError(
            f"Unknown temperature mode '{temperature_mode}'. "
            f"Valid options: 'isothermal', 'linear', 'prescribed', 'adiabatic'."
        )


def create_pressure_density_files(
    outer_iter, inner_iter, pressure_iter, radii, pressure, density
):
    """
    Append pressure and density profiles to output files for a given iteration.

    Parameters
    ----------
    outer_iter : int
        Current outer iteration index.
    inner_iter : int
        Current inner iteration index.
    pressure_iter : int
        Current pressure iteration index.
    radii : numpy.ndarray
        Radial positions, in m.
    pressure : numpy.ndarray
        Pressure values corresponding to ``radii``, in Pa.
    density : numpy.ndarray
        Density values corresponding to ``radii``, in kg/m^3.

    Returns
    -------
    None
    """

    pressure_file = os.path.join(ZALMOXIS_ROOT, 'output_files', 'pressure_profiles.txt')
    density_file = os.path.join(ZALMOXIS_ROOT, 'output_files', 'density_profiles.txt')

    # Only delete the files once at the beginning of the run
    if outer_iter == 0 and inner_iter == 0 and pressure_iter == 0:
        for file_path in [pressure_file, density_file]:
            if os.path.exists(file_path):
                os.remove(file_path)

    # Append current iteration's pressure profile to file
    with open(pressure_file, 'a') as f:
        f.write(f'# Pressure iteration {pressure_iter}\n')
        np.savetxt(f, np.column_stack((radii, pressure)), header='radius pressure', comments='')
        f.write('\n')

    # Append current iteration's density profile to file
    with open(density_file, 'a') as f:
        f.write(f'# Pressure iteration {pressure_iter}\n')
        np.savetxt(f, np.column_stack((radii, density)), header='radius density', comments='')
        f.write('\n')
