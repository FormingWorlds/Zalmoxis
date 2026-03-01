"""
EOS Data and Functions
"""

from __future__ import annotations

import logging
import os

import numpy as np
from scipy.interpolate import LinearNDInterpolator, RegularGridInterpolator, interp1d

from .eos_analytic import get_analytic_density

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')

logger = logging.getLogger(__name__)


def get_tabulated_eos(
    pressure, material_dictionary, material, temperature=None, interpolation_functions=None
):
    """
    Retrieves density from tabulated EOS data for a given material and choice of EOS.
    Parameters:
        pressure: Pressure at which to evaluate the EOS (in Pa)
        material_dictionary: Dictionary containing material properties and EOS file paths
        material: Material type (e.g., "core", "mantle", "ice_layer", "melted_mantle", "solid_mantle")
        temperature: Temperature at which to evaluate the EOS (in K), if applicable
        interpolation_functions: Cache for interpolation functions to avoid redundant loading
    Returns:
        density: Density corresponding to the given pressure (and temperature if applicable) in kg/m^3
    """
    if interpolation_functions is None:
        interpolation_functions = {}
    props = material_dictionary[material]
    eos_file = props['eos_file']
    try:
        if eos_file not in interpolation_functions:
            if material == 'melted_mantle' or material == 'solid_mantle':
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
        if material == 'melted_mantle' or material == 'solid_mantle':
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
    Loads melting curve data for MgSiO3 from a text file.
    Parameters:
        melt_file: Path to the melting curve data file
    Returns:
        interp_func: Interpolation function for T(P)
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


def get_solidus_liquidus_functions():
    """
    Loads and returns the solidus and liquidus melting curves for temperature-dependent silicate mantle EOS.
    Returns: A tuple containing the solidus and liquidus functions.
    """
    solidus_func = load_melting_curve(
        os.path.join(ZALMOXIS_ROOT, 'data', 'melting_curves_Monteux-600', 'solidus.dat')
    )
    liquidus_func = load_melting_curve(
        os.path.join(ZALMOXIS_ROOT, 'data', 'melting_curves_Monteux-600', 'liquidus.dat')
    )

    return solidus_func, liquidus_func


def get_Tdep_density(
    pressure,
    temperature,
    material_properties_iron_Tdep_silicate_planets,
    solidus_func,
    liquidus_func,
    interpolation_functions=None,
):
    """
    Returns density for mantle material, considering temperature-dependent phase changes.
    Parameters:
        pressure: Pressure at which to evaluate the EOS (in Pa)
        temperature: Temperature at which to evaluate the EOS (in K)
        material_properties_iron_Tdep_silicate_planets: Dictionary containing temperature-dependent material properties for temperature-dependent MgSiO3 EOS
        solidus_func: Interpolation function for the solidus melting curve
        liquidus_func: Interpolation function for the liquidus melting curve
        interpolation_functions: Cache for interpolation functions to avoid redundant loading
    Returns:
        density: Density corresponding to the given pressure and temperature in kg/m^3
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
    Returns type for mantle material, considering temperature-dependent phase changes. Supports scalar and array inputs.
    Parameters:
        pressure: Pressure at which to evaluate the EOS (in Pa), can be scalar or array
        temperature: Temperature at which to evaluate the EOS (in K), can be scalar or array
    Returns:
        material: Material type ("solid_mantle", "melted_mantle", or "mixed_mantle")
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


def calculate_density(
    pressure,
    material_dictionaries,
    layer_eos,
    temperature,
    solidus_func,
    liquidus_func,
    interpolation_functions=None,
):
    """Calculate density for a single layer given its per-layer EOS string.

    Parameters
    ----------
    pressure : float
        Pressure at which to evaluate the EOS (in Pa).
    material_dictionaries : tuple
        Tuple of material property dictionaries
        (iron_silicate, iron_Tdep_silicate, water, iron_RTPress100TPa_silicate).
    layer_eos : str
        Per-layer EOS identifier, e.g. "Seager2007:iron",
        "WolfBower2018:MgSiO3", "Analytic:iron".
    temperature : float
        Temperature at which to evaluate the EOS (in K).
    solidus_func : callable or None
        Interpolation function for the solidus melting curve.
    liquidus_func : callable or None
        Interpolation function for the liquidus melting curve.
    interpolation_functions : dict
        Cache for interpolation functions to avoid redundant loading.

    Returns
    -------
    float or None
        Density in kg/m^3, or None on failure.
    """
    if interpolation_functions is None:
        interpolation_functions = {}

    (
        mat_iron_sil,
        mat_Tdep,
        mat_water,
        mat_RTPress100TPa,
    ) = material_dictionaries

    if layer_eos == 'Seager2007:iron':
        return get_tabulated_eos(
            pressure, mat_iron_sil, 'core', interpolation_functions=interpolation_functions
        )
    elif layer_eos == 'Seager2007:MgSiO3':
        return get_tabulated_eos(
            pressure, mat_iron_sil, 'mantle', interpolation_functions=interpolation_functions
        )
    elif layer_eos == 'WolfBower2018:MgSiO3':
        return get_Tdep_density(
            pressure,
            temperature,
            mat_Tdep,
            solidus_func,
            liquidus_func,
            interpolation_functions,
        )
    elif layer_eos == 'RTPress100TPa:MgSiO3':
        return get_Tdep_density(
            pressure,
            temperature,
            mat_RTPress100TPa,
            solidus_func,
            liquidus_func,
            interpolation_functions,
        )
    elif layer_eos == 'Seager2007:H2O':
        return get_tabulated_eos(
            pressure, mat_water, 'ice_layer', interpolation_functions=interpolation_functions
        )
    elif layer_eos.startswith('Analytic:'):
        material_key = layer_eos.split(':', 1)[1]
        return get_analytic_density(pressure, material_key)
    else:
        raise ValueError(f"Unknown layer EOS '{layer_eos}'.")


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
):
    """Compute adiabatic T(r) using native (dT/dP)_S tables from the EOS.

    Integrates T[i] = T[i+1] + (dT/dP)_S · (P[i] - P[i+1]) from surface
    inward.  The adiabatic gradient (dT/dP)_S = αT/(ρCp) is read directly
    from the ``adiabat_grad_file`` in the melt-phase material dictionary,
    avoiding any intermediate α or Cp computation.

    For T-independent EOS layers (e.g. Seager2007 iron core), no adiabat
    gradient table exists, so the temperature is held constant (isothermal).

    Parameters
    ----------
    radii : np.ndarray
        Radial grid, ascending (center to surface) [m].
    pressure : np.ndarray
        Pressure at each radius [Pa].
    mass_enclosed : np.ndarray
        Enclosed mass at each radius [kg].
    surface_temperature : float
        Temperature at the surface [K].
    cmb_mass : float
        Core-mantle boundary mass [kg].
    core_mantle_mass : float
        Core + mantle mass [kg].
    layer_eos_config : dict
        Per-layer EOS config, e.g.
        ``{"core": "Seager2007:iron", "mantle": "WolfBower2018:MgSiO3"}``.
    material_dictionaries : tuple
        Material property dictionaries.
    interpolation_functions : dict or None
        Cache for interpolation functions.

    Returns
    -------
    np.ndarray
        Temperature at each radial point [K].

    Raises
    ------
    ValueError
        If a T-dependent mantle EOS is used but no ``adiabat_grad_file``
        is present in the material dictionary.
    """
    from .constants import TDEP_EOS_NAMES
    from .structure_model import get_layer_eos

    if interpolation_functions is None:
        interpolation_functions = {}

    # Build a wrapper dict for the dT/dP table so we can reuse get_tabulated_eos()
    # (which handles caching, irregular grids, and bounds checking).
    dtdp_dicts = {}
    for eos_name, mat_idx in [('WolfBower2018:MgSiO3', 1), ('RTPress100TPa:MgSiO3', 3)]:
        mat = material_dictionaries[mat_idx]
        grad_file = mat.get('melted_mantle', {}).get('adiabat_grad_file')
        if grad_file is not None:
            dtdp_dicts[eos_name] = {'melted_mantle': {'eos_file': grad_file}}

    # Verify that the mantle EOS supports adiabatic mode
    mantle_eos = layer_eos_config.get('mantle')
    if mantle_eos not in TDEP_EOS_NAMES:
        raise ValueError(
            f'Adiabatic temperature mode requires a T-dependent mantle EOS '
            f'(WolfBower2018:MgSiO3 or RTPress100TPa:MgSiO3), '
            f"but got '{mantle_eos}'. Use 'linear' or 'isothermal' instead."
        )
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
    # core at the CMB temperature.
    for i in range(n - 2, -1, -1):
        layer_eos = get_layer_eos(
            mass_enclosed[i],
            cmb_mass,
            core_mantle_mass,
            layer_eos_config,
        )

        if layer_eos in TDEP_EOS_NAMES:
            # Evaluate (dT/dP)_S at the current endpoint (i+1).  A midpoint
            # rule would be more accurate, but the endpoint is sufficient
            # for this initial-condition-quality profile.
            dtdp = get_tabulated_eos(
                pressure[i + 1],
                dtdp_dicts[layer_eos],
                'melted_mantle',
                T[i + 1],
                interpolation_functions,
            )
            if dtdp is not None and dtdp > 0:
                dP = pressure[i] - pressure[i + 1]  # positive going inward
                T[i] = T[i + 1] + dtdp * dP
            else:
                T[i] = T[i + 1]
        else:
            # T-independent EOS (e.g. iron core): isothermal
            T[i] = T[i + 1]

    return T


def calculate_temperature_profile(
    radii,
    temperature_mode,
    surface_temperature,
    center_temperature,
    input_dir,
    temp_profile_file,
):
    """
    Returns a callable temperature function for a planetary interior model.

    Parameters:
    radii: Radial grid of the planet [m].
    temperature_mode: Temperature profile mode. Options:
        - "isothermal": constant temperature equal to surface_temperature
        - "linear": linear profile from center_temperature (r=0) to surface_temperature (r=R)
        - "prescribed": read temperature profile from a text file
        - "adiabatic": returns linear profile as initial guess; the actual
          adiabat is computed in the main() outer loop using P(r) and g(r)
    surface_temperature: Temperature at the surface [K] (used for "linear" and "isothermal")
    center_temperature: Temperature at the center [K] (used for "linear")
    input_dir: Directory where the temperature profile file is located.
    temp_profile_file: Name of the file containing the prescribed temperature profile from center to surface. Must have same length as `radii` if temperature_mode="prescribed".

    Returns:
    temperature_func: Function of radius or array of radii for temperature [K]: temperature_func(r) -> float or np.ndarray
    """
    radii = np.array(radii)

    if temperature_mode == 'isothermal':
        return lambda r: np.full_like(r, surface_temperature, dtype=float)

    elif temperature_mode == 'linear':
        return lambda r: surface_temperature + (center_temperature - surface_temperature) * (
            1 - np.array(r) / radii[-1]
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
        return lambda r: surface_temperature + (center_temperature - surface_temperature) * (
            1 - np.array(r) / radii[-1]
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
    Create and append pressure and density profiles to output files for each pressure iteration.
    Parameters:
        outer_iter (int): Current outer iteration index.
        inner_iter (int): Current inner iteration index.
        pressure_iter (int): Current pressure iteration index.
        radii (np.ndarray): Array of radial positions.
        pressure (np.ndarray): Array of pressure values corresponding to the radii.
        density (np.ndarray): Array of density values corresponding to the radii.
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
