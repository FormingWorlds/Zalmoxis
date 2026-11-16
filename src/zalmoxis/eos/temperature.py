"""Adiabatic temperature profile computation and temperature mode dispatch.

Provides ``compute_adiabatic_temperature`` (using native EOS gradient tables)
and ``calculate_temperature_profile`` (mode-based dispatch for isothermal,
linear, prescribed, and adiabatic profiles).
"""

from __future__ import annotations

import logging
import math
import numbers
import os

import numpy as np

from ..constants import CONDENSED_RHO_MIN_DEFAULT, CONDENSED_RHO_SCALE_DEFAULT
from ..invariants import StructureInvariantError
from .tdep import _get_paleos_nabla_ad

logger = logging.getLogger('fwl.' + __name__)


ADIABAT_MAX_DLNP = 0.0025  # largest ln P change of one midpoint substep
ADIABAT_P_FLOOR = 1e5  # Pa: lower edge of the PALEOS rock and iron tables; T is held below it
_T_MAX, _T_MIN_CMB = 100000.0, 100.0  # node caps [K]: PALEOS table maximum, CMB-anchor floor


def _is_real(x):
    """True for a real number that is not a bool (a TOML ``true`` is not a number here)."""
    return isinstance(x, numbers.Real) and not isinstance(x, bool)


def _check_max_dlnp(h):
    """Return ``h`` as a float; raise ValueError unless it is a real number from 1e-4 to 1."""
    if not _is_real(h) or not 1e-4 <= h <= 1.0:
        raise ValueError(f'max_dlnp must be a real number from 1e-4 to 1, got {h!r}.')
    return float(h)


def _segment_temperature(T0, P_from, P_to, nabla_of, h, inward):
    """Temperature at ``P_to`` on the adiabat d ln T / d ln P = nabla through (``P_from``, ``T0``).

    Integrates with the midpoint rule in ln P over ceil(|d ln P| / ``h``) equal substeps, after
    raising both pressures to ``ADIABAT_P_FLOOR``, so T is held below that pressure; a padded
    end (P = 0) counts as the floor. Returns ``T0`` at a NaN or infinite end, for a
    non-positive ``T0``, and when the segment is empty or does not run in the anchor direction (``inward``: towards higher P); T
    is held from the first substep where ``nabla_of(P, T)`` is None or not positive. Inward T
    stops at the node cap ``_T_MAX`` and outward T at the CMB-anchor floor ``_T_MIN_CMB``, in
    every substep and every read of ``nabla_of``; a segment that reaches it returns it exactly.
    """
    if not (P_from < math.inf and P_to < math.inf and 0 < T0 < math.inf):
        return T0
    p, t = max(P_from, ADIABAT_P_FLOOR), T0  # the first read is at the start node itself
    x0, x1 = math.log(p), math.log(max(P_to, ADIABAT_P_FLOOR))
    if x1 == x0 or (x1 > x0) != inward:
        return T0
    n = math.ceil(abs(x1 - x0) / h)
    dx, dy = (x1 - x0) / n, 0.0  # dy: ln T change over the segment, summed from 0
    T_lim, clamp = (_T_MAX, min) if inward else (_T_MIN_CMB, max)
    dy_lim = math.log(T_lim) - math.log(T0)  # ln T change to the cap or floor, no overflow
    for k in range(n):
        k1 = nabla_of(p, t)
        if k1 is None or not k1 > 0:
            break
        t_mid = clamp(t * math.exp(clamp(dx / 2 * k1, dy_lim - dy)), T_lim)
        k2 = nabla_of(math.exp(x0 + (k + 0.5) * dx), t_mid)
        if k2 is None or not k2 > 0:
            break
        dy = clamp(dy + dx * k2, dy_lim)
        if dy == dy_lim:  # T0 exp(dy_lim) can miss T_lim by 1 ulp
            return T_lim
        p, t = math.exp(x0 + (k + 1) * dx), clamp(T0 * math.exp(dy), T_lim)
    return t


def _check_profile(T, first):
    """Raise StructureInvariantError unless ``T`` is finite and positive at every node and does
    not rise outward from node ``first`` (the anchor side of the integrated nodes)."""
    if not np.all(np.isfinite(T) & (T > 0)):
        raise StructureInvariantError(
            'adiabat temperature is not finite and positive at every node.'
        )
    if np.any(np.diff(T[first:]) > 0):
        raise StructureInvariantError('adiabat temperature is not non-increasing outward.')


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
    max_dlnp=ADIABAT_MAX_DLNP,
):
    """Compute an adiabatic temperature profile using native EOS gradient tables.

    Integrates d ln T / d ln P = nabla_ad(P, T) over each node segment with the midpoint rule
    in ln P and ceil(|d ln P| / ``max_dlnp``) substeps. nabla_ad comes from
    ``get_mixed_nabla_ad`` with the layer mixture of the node the segment fills; for a
    multi-material layer it is mass-fraction-weighted across the components. T is held below
    ``ADIABAT_P_FLOOR`` (1e5 Pa, where the PALEOS rock and iron tables start), and a padded
    node (P = 0) counts as that floor. T is also held over a layer without a T-dependent
    component and from the first substep where nabla_ad is missing or not positive. Each node is capped at
    1e5 K, and with ``anchor='cmb'`` floored at 100 K.

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
        center. ``'cmb'`` anchors ``T[i_cmb] = cmb_temperature`` at the first
        mantle shell and integrates outward to the surface (mantle only); shells below
        the CMB carry the surface anchor through, since the core
        adiabat is decoupled and the energetics solver computes T_core
        independently.
    cmb_temperature : float, optional
        Temperature at the core-mantle boundary, in K. Required when
        ``anchor='cmb'``.
    max_dlnp : float, optional
        Largest change of ln P in one substep, from 1e-4 to 1. Default
        ``ADIABAT_MAX_DLNP`` (0.0025).

    Returns
    -------
    numpy.ndarray
        Temperature at each radial point, in K.

    Raises
    ------
    ValueError
        No T-dependent layer, ``max_dlnp`` outside its range, an ``anchor`` other than 'surface'
        or 'cmb', a ``surface_temperature`` that is
        not above 0 K and at most 1e5 K, or ``anchor='cmb'`` without a ``cmb_temperature`` from
        100 K to 1e5 K (the node caps).
    StructureInvariantError
        The profile is not finite and positive at every node, or rises outward from its anchor.
    """
    from ..mixing import any_component_is_tdep, get_mixed_nabla_ad
    from ..structure_model import get_layer_mixture

    if interpolation_functions is None:
        interpolation_functions = {}
    h = _check_max_dlnp(max_dlnp)

    if not any_component_is_tdep(layer_mixtures):
        raise ValueError(
            'Adiabatic temperature mode requires at least one T-dependent EOS '
            'layer, but none found. '
            "Use 'linear' or 'isothermal' instead."
        )

    def nabla_of(mixture):
        def nabla(P, T):
            return get_mixed_nabla_ad(
                P,
                T,
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

        return nabla

    if anchor not in ('surface', 'cmb'):
        raise ValueError(f"anchor must be 'surface' or 'cmb', got {anchor!r}.")
    if not (_is_real(surface_temperature) and 0 < surface_temperature <= _T_MAX):
        raise ValueError(
            f'surface_temperature must be a temperature above 0 and at most {_T_MAX:g} K, '
            f'got {surface_temperature!r}.'
        )
    n = len(radii)
    T = np.zeros(n)
    if anchor == 'cmb':
        if not (_is_real(cmb_temperature) and _T_MIN_CMB <= cmb_temperature <= _T_MAX):
            raise ValueError(
                f"anchor='cmb' requires a cmb_temperature from {_T_MIN_CMB:g} to {_T_MAX:g} K, "
                f'got {cmb_temperature!r}.'
            )
        # first mantle shell: mass_enclosed >= cmb_mass; the core carries the surface anchor
        first = max(1, min(int(np.searchsorted(mass_enclosed, cmb_mass)), n - 1))
        T[:first] = surface_temperature
        T[first] = cmb_temperature
        steps, floor = [(i, i - 1) for i in range(first + 1, n)], _T_MIN_CMB
    else:
        T[n - 1] = surface_temperature
        first, steps, floor = 0, [(i, i + 1) for i in range(n - 2, -1, -1)], -math.inf

    for i, j in steps:  # node i takes its T from node j, with the mixture of node i
        mixture = get_layer_mixture(
            mass_enclosed[i], cmb_mass, core_mantle_mass, layer_mixtures
        )
        if not mixture.has_tdep():
            T[i] = T[j]
            continue
        T_new = _segment_temperature(
            T[j], pressure[j], pressure[i], nabla_of(mixture), h, anchor != 'cmb'
        )
        T[i] = min(max(T_new, floor), _T_MAX)
    _check_profile(T, first)
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

    if np.isnan(T_sol) or np.isnan(T_liq) or T_liq < T_sol:
        # Outside melting curve range or inverted (unphysical) curve: use
        # solid table. T_liq == T_sol (zero-width mushy zone, e.g.
        # mushy_zone_factor = 1.0) falls through to the solid/liquid
        # branches below instead, which bifurcate correctly on temperature.
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
        if (
            nabla_solid is None and nabla_liquid is None
        ):  # pragma: no cover - both phases undefined; defensive
            return None
        if (
            nabla_solid is None
        ):  # pragma: no cover - solid-side phase undefined at table edge; defensive
            nabla = nabla_liquid
        elif (
            nabla_liquid is None
        ):  # pragma: no cover - liquid-side phase undefined at table edge; defensive
            nabla = nabla_solid
        else:
            nabla = (1.0 - phi) * nabla_solid + phi * nabla_liquid

    if nabla is None or nabla <= 0:  # pragma: no cover - non-positive nabla_ad; defensive
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
            cmb_temperature
            if cmb_temperature is not None and cmb_temperature > 0
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
