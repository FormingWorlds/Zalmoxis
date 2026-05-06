"""Temperature-dependent EOS routines.

Includes melting curve loading, solidus/liquidus dispatch, T-dependent
density evaluation with phase mixing, and PALEOS nabla_ad lookups for
2-phase tables.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .interpolation import (
    _paleos_clamp_temperature,
    _paleos_clamp_warned,
    load_paleos_table,
)
from .seager import get_tabulated_eos

logger = logging.getLogger(__name__)


def load_melting_curve(melt_file):
    """Load a melting curve for MgSiO3 from a text file.

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
    from scipy.interpolate import interp1d

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
    from ..melting_curves import get_solidus_liquidus_functions as _get

    return _get(solidus_id, liquidus_id)


def get_Tdep_density(
    pressure,
    temperature,
    material_properties_iron_Tdep_silicate_planets,
    solidus_func,
    liquidus_func,
    interpolation_functions=None,
):
    """Compute mantle density for a temperature-dependent EOS with phase changes.

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

    # Pressure outside melting curve range -- default to solid phase
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
        # Smoothstep ramp s = x*x*(3 - 2*x) replaces the linear melt
        # fraction so d(1/rho)/dT vanishes at T=T_sol and T=T_liq,
        # eliminating the lever-rule kink that drives inner Picard
        # plateau on hot mushy profiles. Midpoint s(0.5)=0.5 = linear,
        # so the in-mushy density profile is preserved away from the
        # boundaries. Mirror change in jax_eos/tdep.py for parity.
        frac_melt_raw = (temperature - T_sol) / (T_liq - T_sol)
        x = max(0.0, min(1.0, frac_melt_raw))
        frac_melt = x * x * (3.0 - 2.0 * x)
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
    """Determine the mantle phase for a temperature-dependent EOS.

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
