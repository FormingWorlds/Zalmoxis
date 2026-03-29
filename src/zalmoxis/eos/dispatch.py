"""Main entry points for density calculation, dispatching to the right EOS.

``calculate_density`` and ``calculate_density_batch`` are the primary
public API used by the solver and mixing modules.
"""

from __future__ import annotations

import logging

import numpy as np

from ..eos_analytic import get_analytic_density
from ..eos_vinet import get_vinet_density
from .paleos import get_paleos_unified_density, get_paleos_unified_density_batch
from .seager import get_tabulated_eos
from .tdep import get_Tdep_density

logger = logging.getLogger(__name__)


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
    """Calculate density for a single layer given its EOS identifier.

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

    # Vinet (Rose-Vinet) EOS: no material dict needed
    if layer_eos.startswith('Vinet:'):
        material_key = layer_eos.split(':', 1)[1]
        return get_vinet_density(pressure, material_key)

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


def calculate_density_batch(
    pressures,
    temperatures,
    material_dictionaries,
    layer_eos,
    solidus_func,
    liquidus_func,
    interpolation_functions,
    mushy_zone_factor=1.0,
):
    """Vectorized density lookup for a batch of (P, T) points sharing one EOS.

    For unified PALEOS tables, uses the vectorized interpolator path.
    For other EOS types, falls back to scalar calculate_density per point.

    Parameters
    ----------
    pressures : numpy.ndarray
        1D array of pressures in Pa.
    temperatures : numpy.ndarray
        1D array of temperatures in K.
    material_dictionaries : dict
        EOS registry.
    layer_eos : str
        EOS identifier string.
    solidus_func : callable or None
        Solidus melting curve function.
    liquidus_func : callable or None
        Liquidus melting curve function.
    interpolation_functions : dict
        Interpolation cache.
    mushy_zone_factor : float
        Cryoscopic depression factor.

    Returns
    -------
    numpy.ndarray
        1D array of densities in kg/m^3. NaN where lookup fails.
    """
    if interpolation_functions is None:
        interpolation_functions = {}

    mat = material_dictionaries.get(layer_eos)
    if mat is not None and mat.get('format') == 'paleos_unified':
        return get_paleos_unified_density_batch(
            pressures, temperatures, mat, mushy_zone_factor, interpolation_functions
        )

    # Fallback: scalar loop for non-unified EOS types
    n = len(pressures)
    result = np.full(n, np.nan)
    for i in range(n):
        val = calculate_density(
            pressures[i],
            material_dictionaries,
            layer_eos,
            temperatures[i],
            solidus_func,
            liquidus_func,
            interpolation_functions,
            mushy_zone_factor,
        )
        result[i] = val if val is not None else np.nan
    return result
