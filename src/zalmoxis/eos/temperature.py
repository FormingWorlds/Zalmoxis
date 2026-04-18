"""Adiabatic temperature profile computation and temperature mode dispatch.

Provides ``compute_adiabatic_temperature`` (using native EOS gradient tables)
and ``calculate_temperature_profile`` (mode-based dispatch for isothermal,
linear, prescribed, and adiabatic profiles).
"""

from __future__ import annotations

import logging
import os

import numpy as np

from ..constants import CONDENSED_RHO_MIN_DEFAULT, CONDENSED_RHO_SCALE_DEFAULT
from .tdep import _get_paleos_nabla_ad

logger = logging.getLogger(__name__)


def compute_adiabatic_temperature(
    radii,
    pressure,
    mass_enclosed,
    surface_temperature,
    cmb_mass,
    core_mantle_mass,
    layer_mixtures,
    material_dictionaries,
    interpolation_functions=None,
    solidus_func=None,
    liquidus_func=None,
    mushy_zone_factors=None,
    condensed_rho_min=CONDENSED_RHO_MIN_DEFAULT,
    condensed_rho_scale=CONDENSED_RHO_SCALE_DEFAULT,
    binodal_T_scale=50.0,
    anchor='surface',
    cmb_temperature=None,
):
    """Compute an adiabatic temperature profile using native EOS gradient tables.

    Supports single-material and multi-material (volume-additive) layers.
    For multi-material layers, nabla_ad is mass-fraction-weighted across
    components.

    Parameters
    ----------
    radii : numpy.ndarray
        Radial grid in ascending order from center to surface, in m.
    pressure : numpy.ndarray
        Pressure at each radius, in Pa.
    mass_enclosed : numpy.ndarray
        Enclosed mass at each radius, in kg.
    surface_temperature : float
        Temperature at the surface, in K. Used as the anchor when
        ``anchor='surface'``; carried through as-is for the core when
        ``anchor='cmb'`` (no integration is done below the CMB).
    cmb_mass : float
        Core-mantle boundary mass, in kg.
    core_mantle_mass : float
        Total core-plus-mantle mass, in kg.
    layer_mixtures : dict
        Per-layer LayerMixture objects.
    material_dictionaries : dict
        EOS registry dict keyed by EOS identifier string.
    interpolation_functions : dict, optional
        Cache of interpolation functions used to avoid redundant loading.
    solidus_func : callable, optional
        Interpolation function for the solidus melting curve.
    liquidus_func : callable, optional
        Interpolation function for the liquidus melting curve.
    mushy_zone_factors : dict or float or None, optional
        Per-EOS mushy zone factors. Dict keyed by EOS name, a single
        float (applied to all), or None (default 1.0 for all).
    condensed_rho_min : float, optional
        Sigmoid center for phase-aware suppression (kg/m^3). Default 322.
    condensed_rho_scale : float, optional
        Sigmoid width for phase-aware suppression (kg/m^3). Default 50.
    binodal_T_scale : float, optional
        Binodal sigmoid width in K for H2 miscibility suppression.
        Default 50.
    anchor : {'surface', 'cmb'}, optional
        Where the adiabat is anchored. ``'surface'`` (default) anchors
        ``T[n-1] = surface_temperature`` and integrates inward to the
        center, the original behaviour. ``'cmb'`` anchors
        ``T[i_cmb] = cmb_temperature`` at the first mantle shell and
        integrates outward to the surface (mantle only); shells below
        the CMB carry the surface anchor through, since the core
        adiabat is decoupled and the energetics solver computes T_core
        independently.
    cmb_temperature : float, optional
        Temperature at the core-mantle boundary, in K. Required when
        ``anchor='cmb'``.

    Returns
    -------
    numpy.ndarray
        Temperature at each radial point, in K.
    """
    from ..mixing import get_mixed_nabla_ad
    from ..structure_model import get_layer_mixture

    if interpolation_functions is None:
        interpolation_functions = {}

    from ..mixing import any_component_is_tdep

    if not any_component_is_tdep(layer_mixtures):
        raise ValueError(
            'Adiabatic temperature mode requires at least one T-dependent EOS '
            'layer, but none found. '
            "Use 'linear' or 'isothermal' instead."
        )

    n = len(radii)
    T = np.zeros(n)

    if anchor == 'cmb':
        if cmb_temperature is None or cmb_temperature <= 0:
            raise ValueError(
                "anchor='cmb' requires a positive cmb_temperature, got "
                f'{cmb_temperature}.'
            )

        # Find the first mantle shell (index where mass_enclosed >= cmb_mass).
        cmb_index = int(np.searchsorted(mass_enclosed, cmb_mass))
        cmb_index = max(1, min(cmb_index, n - 1))

        # Carry the surface anchor through the core (no integration);
        # the energetics solver handles T_core via core_heatcap and the
        # Bower+2018 adiabatic ratio. Anchor the mantle at CMB.
        T[:cmb_index] = surface_temperature
        T[cmb_index] = cmb_temperature

        # Integrate upward from CMB to surface (i = cmb_index .. n-1).
        for i in range(cmb_index + 1, n):
            mixture = get_layer_mixture(
                mass_enclosed[i],
                cmb_mass,
                core_mantle_mass,
                layer_mixtures,
            )

            if not mixture.has_tdep():
                T[i] = T[i - 1]
                continue

            P_eval = pressure[i - 1]
            T_eval = T[i - 1]
            dP = pressure[i] - pressure[i - 1]  # negative going outward

            if P_eval < 1e5 or pressure[i] < 1e5:
                T[i] = T[i - 1]
                continue

            nabla = get_mixed_nabla_ad(
                P_eval,
                T_eval,
                mixture,
                material_dictionaries,
                interpolation_functions,
                solidus_func,
                liquidus_func,
                mushy_zone_factors,
                condensed_rho_min,
                condensed_rho_scale,
                binodal_T_scale,
            )

            if nabla is not None and nabla > 0 and P_eval > 0 and T_eval > 0 and dP < 0:
                dtdp = nabla * T_eval / P_eval
                T_new = T_eval + dtdp * dP  # dP<0 => T cools outward
                T[i] = max(min(T_new, 100000.0), 100.0)
            else:
                T[i] = T_eval

        return T

    # anchor == 'surface' (original behaviour): integrate inward from
    # T[n-1] = surface_temperature to the center.
    T[n - 1] = surface_temperature

    for i in range(n - 2, -1, -1):
        mixture = get_layer_mixture(
            mass_enclosed[i],
            cmb_mass,
            core_mantle_mass,
            layer_mixtures,
        )

        if not mixture.has_tdep():
            T[i] = T[i + 1]
            continue

        P_eval = pressure[i + 1]
        T_eval = T[i + 1]
        dP = pressure[i] - pressure[i + 1]

        # Skip at very low P to prevent T/P divergence
        if P_eval < 1e5 or pressure[i] < 1e5:
            T[i] = T[i + 1]
            continue

        # Get nabla_ad (handles single and multi-component mixtures)
        nabla = get_mixed_nabla_ad(
            P_eval,
            T_eval,
            mixture,
            material_dictionaries,
            interpolation_functions,
            solidus_func,
            liquidus_func,
            mushy_zone_factors,
            condensed_rho_min,
            condensed_rho_scale,
            binodal_T_scale,
        )

        if nabla is not None and nabla > 0 and P_eval > 0 and T_eval > 0 and dP > 0:
            dtdp = nabla * T_eval / P_eval
            T_new = T_eval + dtdp * dP
            # Cap temperature to the PALEOS table maximum (100,000 K).
            # Without this cap, numerical noise at low P or in mixed
            # compositions can cause runaway T during the adiabat
            # integration, which then pollutes the T(P) interpolator
            # and prevents the Brent solver from bracketing.
            T[i] = min(T_new, 100000.0)
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
    cmb_temperature=None,
):
    """Return a callable temperature profile for a planetary interior model.

    Parameters
    ----------
    radii : array_like
        Radial grid of the planet, in m.
    temperature_mode : {"isothermal", "linear", "prescribed", "adiabatic", "adiabatic_from_cmb"}
        Temperature profile mode.

        - ``"isothermal"``: constant temperature equal to
          ``surface_temperature``.
        - ``"linear"``: linear profile from ``center_temperature`` at
          ``r = 0`` to ``surface_temperature`` at the surface.
        - ``"prescribed"``: read the temperature profile from a text file.
        - ``"adiabatic"``: return a linear profile as an initial guess; the
          actual adiabat is computed elsewhere in the main iteration loop
          and is anchored at ``surface_temperature``.
        - ``"adiabatic_from_cmb"``: same as ``"adiabatic"`` but the actual
          adiabat (computed elsewhere) is anchored at ``cmb_temperature``
          at the core-mantle boundary and integrated outward to the
          surface. The initial guess returned here is still linear, with
          ``cmb_temperature`` used as the deep anchor in place of
          ``center_temperature`` so the first density iteration sees a
          reasonable mantle T(r).
    surface_temperature : float
        Temperature at the surface, in K.
    center_temperature : float
        Temperature at the center, in K. Used for ``"linear"`` and
        ``"adiabatic"`` modes.
    cmb_temperature : float, optional
        Temperature at the core-mantle boundary, in K. Required for
        ``"adiabatic_from_cmb"`` mode; ignored otherwise.
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

    elif temperature_mode == 'adiabatic_from_cmb':
        # Initial guess: linear from surface to a deep anchor that uses
        # cmb_temperature when provided (otherwise falls back to
        # center_temperature). The CMB index is unknown at this point in
        # the solve, so anchor at r=0 with the CMB value as a safe
        # over-estimate; the actual upward adiabat from CMB is computed in
        # main() once the converged structure exposes the CMB index.
        deep_anchor = (
            cmb_temperature if cmb_temperature is not None and cmb_temperature > 0
            else center_temperature
        )
        return lambda r: (
            surface_temperature
            + (deep_anchor - surface_temperature) * (1 - np.array(r) / radii[-1])
        )

    else:
        raise ValueError(
            f"Unknown temperature mode '{temperature_mode}'. "
            f"Valid options: 'isothermal', 'linear', 'prescribed', 'adiabatic', "
            f"'adiabatic_from_cmb'."
        )
