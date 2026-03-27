"""EOS package for Zalmoxis.

Re-exports all public and private functions from the submodules so that
``from zalmoxis.eos import calculate_density`` works identically to the
old monolithic ``eos_functions.py``.
"""

from __future__ import annotations

from .dispatch import calculate_density, calculate_density_batch
from .interpolation import (
    _ensure_unified_cache,
    _fast_bilinear,
    _paleos_clamp_temperature,
    load_paleos_table,
    load_paleos_unified_table,
)
from .output import create_pressure_density_files
from .paleos import (
    _get_paleos_unified_nabla_ad,
    get_paleos_unified_density,
    get_paleos_unified_density_batch,
)
from .seager import get_tabulated_eos
from .tdep import (
    _get_paleos_nabla_ad,
    get_solidus_liquidus_functions,
    get_Tdep_density,
    get_Tdep_material,
    load_melting_curve,
)
from .temperature import (
    _compute_paleos_dtdp,
    calculate_temperature_profile,
    compute_adiabatic_temperature,
)

__all__ = [
    # interpolation
    'load_paleos_table',
    'load_paleos_unified_table',
    '_fast_bilinear',
    '_paleos_clamp_temperature',
    '_ensure_unified_cache',
    # seager
    'get_tabulated_eos',
    # paleos
    'get_paleos_unified_density',
    'get_paleos_unified_density_batch',
    '_get_paleos_unified_nabla_ad',
    # tdep
    'load_melting_curve',
    'get_solidus_liquidus_functions',
    'get_Tdep_density',
    'get_Tdep_material',
    '_get_paleos_nabla_ad',
    # dispatch
    'calculate_density',
    'calculate_density_batch',
    # temperature
    'compute_adiabatic_temperature',
    '_compute_paleos_dtdp',
    'calculate_temperature_profile',
    # output
    'create_pressure_density_files',
]
