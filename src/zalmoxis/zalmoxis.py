"""Backward-compatible entry point for the Zalmoxis structure solver.

This module re-exports all public symbols from the split modules
(config, solver, output) so that existing imports continue to work:

    from zalmoxis.zalmoxis import main
    from zalmoxis.zalmoxis import load_zalmoxis_config
    from zalmoxis.zalmoxis import post_processing
"""

from __future__ import annotations

# Re-export config functions
from .config import (  # noqa: F401 — re-exports for backward compatibility
    LEGACY_EOS_MAP,
    PALEOS_MAX_MASS_EARTH,
    PALEOS_UNIFIED_MAX_MASS_EARTH,
    RTPRESS100TPA_MAX_MASS_EARTH,
    VALID_TABULATED_EOS,
    WOLFBOWER2018_MAX_MASS_EARTH,
    _NEEDS_MELTING_CURVES,
    choose_config_file,
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
    parse_eos_config,
    validate_config,
    validate_layer_eos,
)

# Re-export output
from .output import post_processing

# Re-export solver
from .solver import main

__all__ = [
    # Config
    'parse_eos_config',
    'validate_layer_eos',
    'validate_config',
    'choose_config_file',
    'load_zalmoxis_config',
    'load_material_dictionaries',
    'load_solidus_liquidus_functions',
    'LEGACY_EOS_MAP',
    'VALID_TABULATED_EOS',
    'WOLFBOWER2018_MAX_MASS_EARTH',
    'RTPRESS100TPA_MAX_MASS_EARTH',
    'PALEOS_MAX_MASS_EARTH',
    'PALEOS_UNIFIED_MAX_MASS_EARTH',
    # Solver
    'main',
    # Output
    'post_processing',
]
