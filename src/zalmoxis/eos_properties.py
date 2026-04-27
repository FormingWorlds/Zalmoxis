"""Material property dictionaries and EOS registry for Zalmoxis.

Paths to EOS data files are resolved lazily via ``get_zalmoxis_root()``
so that importing this module does not require ZALMOXIS_ROOT to be set.
The registry is built on first access of ``EOS_REGISTRY``.
"""

from __future__ import annotations

import os


def _build_registry() -> dict:
    """Construct the EOS_REGISTRY dict with resolved file paths.

    Called once on first access of EOS_REGISTRY.
    """
    from . import get_zalmoxis_root

    root = get_zalmoxis_root()

    # ── Seager et al. (2007), 300 K static EOS ──────────────────────────
    _seager_iron = {
        'eos_file': os.path.join(root, 'data', 'EOS_Seager2007', 'eos_seager07_iron.txt')
    }
    _seager_silicate = {
        'eos_file': os.path.join(root, 'data', 'EOS_Seager2007', 'eos_seager07_silicate.txt')
    }
    _seager_water = {
        'eos_file': os.path.join(root, 'data', 'EOS_Seager2007', 'eos_seager07_water.txt')
    }

    # ── Wolf & Bower (2018), T-dependent MgSiO3 (up to 1 TPa) ──────────
    _wb2018_melted = {
        'eos_file': os.path.join(root, 'data', 'EOS_WolfBower2018_1TPa', 'density_melt.dat'),
        'adiabat_grad_file': os.path.join(
            root, 'data', 'EOS_WolfBower2018_1TPa', 'adiabat_temp_grad_melt.dat'
        ),
    }
    _wb2018_solid = {
        'eos_file': os.path.join(root, 'data', 'EOS_WolfBower2018_1TPa', 'density_solid.dat'),
    }

    # ── RTPress 100 TPa extended melt + WolfBower2018 solid ─────────────
    _rtpress_melted = {
        'eos_file': os.path.join(root, 'data', 'EOS_RTPress_melt_100TPa', 'density_melt.dat'),
        'adiabat_grad_file': os.path.join(
            root, 'data', 'EOS_RTPress_melt_100TPa', 'adiabat_temp_grad_melt.dat'
        ),
    }

    # ── PALEOS-2phase MgSiO3 (solid + liquid, Zenodo 19680050) ──────────
    # Zenodo 19680050 (2026-04-27) is the new ecosystem-wide PALEOS
    # MgSiO3 reference. It ships two resolutions: 150 pts/decade
    # (default, ~80 MB) and 600 pts/decade (highres, ~1.3 GB).
    # PROTEUS default is the 150-res variant; the highres variant is
    # opt-in via 'PALEOS-2phase:MgSiO3-highres' for sensitivity tests.
    _paleos2ph_melted = {
        'eos_file': os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_liquid.dat'
        ),
        'format': 'paleos',
    }
    _paleos2ph_solid = {
        'eos_file': os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
        ),
        'format': 'paleos',
    }
    _paleos2ph_melted_highres = {
        'eos_file': os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3',
            'paleos_mgsio3_tables_pt_proteus_liquid_highres.dat',
        ),
        'format': 'paleos',
    }
    _paleos2ph_solid_highres = {
        'eos_file': os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3',
            'paleos_mgsio3_tables_pt_proteus_solid_highres.dat',
        ),
        'format': 'paleos',
    }

    # ── PALEOS unified tables (Zenodo 19000316) ─────────────────────────
    _paleos_iron = {
        'eos_file': os.path.join(
            root, 'data', 'EOS_PALEOS_iron', 'paleos_iron_eos_table_pt.dat'
        ),
        'format': 'paleos_unified',
    }
    _paleos_mgsio3 = {
        'eos_file': os.path.join(
            root, 'data', 'EOS_PALEOS_MgSiO3_unified', 'paleos_mgsio3_eos_table_pt.dat'
        ),
        'format': 'paleos_unified',
    }
    _paleos_h2o = {
        'eos_file': os.path.join(
            root, 'data', 'EOS_PALEOS_H2O', 'paleos_water_eos_table_pt.dat'
        ),
        'format': 'paleos_unified',
    }

    # ── PALEOS-API live-tabulated (runtime producer, SHA-keyed cache) ───
    # Registry entries carry the grid spec only; the dispatch layer calls
    # paleos_api_cache.resolve_paleos_api_* to materialise ``eos_file`` on
    # demand (cache hit is cheap; cold cache triggers a generator run
    # parallelised across os.cpu_count()). GridSpec defaults use
    # DEFAULT_PTS_PER_DECADE = 600 (4x Zenodo-shipped resolution).
    from .eos.paleos_api import (
        make_default_grid_h2o,
        make_default_grid_iron,
        make_default_grid_mgsio3,
    )
    _paleos_api_iron = {
        'format': 'paleos_api',
        'material': 'iron',
        'grid_spec': make_default_grid_iron(),
    }
    _paleos_api_mgsio3 = {
        'format': 'paleos_api',
        'material': 'mgsio3',
        'grid_spec': make_default_grid_mgsio3(),
    }
    _paleos_api_h2o = {
        'format': 'paleos_api',
        'material': 'h2o',
        'grid_spec': make_default_grid_h2o(),
        # h2o_table_path resolved by the dispatch helper if None (uses
        # PALEOS's packaged AQUA table).
        'h2o_table_path': None,
    }
    # 2-phase MgSiO3 live-tabulated: used by Aragog via eos_export so that
    # SPIDER P-S tables come from phase-specific endpoints rather than
    # from interpolation across the melting curve.
    _paleos_api_2ph_mgsio3_melted = {
        'format': 'paleos_api_2phase',
        'material': 'mgsio3',
        'side': 'liquid',
        'grid_spec': make_default_grid_mgsio3(),
    }
    _paleos_api_2ph_mgsio3_solid = {
        'format': 'paleos_api_2phase',
        'material': 'mgsio3',
        'side': 'solid',
        'grid_spec': make_default_grid_mgsio3(),
    }

    # ═════════════════════════════════════════════════════════════════════
    # EOS_REGISTRY: flat dict keyed by EOS identifier string.
    # Each value is a dict describing the layer properties needed by
    # calculate_density() and compute_adiabatic_temperature().
    #
    # For T-independent EOS (Seager2007), each key maps to a single-layer
    # dict with 'eos_file'.
    #
    # For T-dependent EOS that use separate solid/liquid tables
    # (WolfBower2018, RTPress100TPa, PALEOS-2phase), keys contain
    # 'melted_mantle' and 'solid_mantle' sub-dicts plus a 'core' sub-dict.
    #
    # For unified PALEOS tables, keys contain a single 'eos_file' and
    # 'format': 'paleos_unified'. Density is looked up directly from the
    # unified table (no separate solid/liquid routing needed).
    # ═════════════════════════════════════════════════════════════════════
    registry = {
        # Seager2007 static EOS (300 K)
        'Seager2007:iron': {'core': _seager_iron},
        'Seager2007:MgSiO3': {'mantle': _seager_silicate},
        'Seager2007:H2O': {'ice_layer': _seager_water},
        # Wolf & Bower 2018 T-dependent MgSiO3
        'WolfBower2018:MgSiO3': {
            'core': _seager_iron,
            'melted_mantle': _wb2018_melted,
            'solid_mantle': _wb2018_solid,
        },
        # RTPress 100 TPa extended melt
        'RTPress100TPa:MgSiO3': {
            'core': _seager_iron,
            'melted_mantle': _rtpress_melted,
            'solid_mantle': _wb2018_solid,
        },
        # PALEOS-2phase MgSiO3 (separate solid/liquid tables, Zenodo 19680050).
        # Default = 150 pts/decade. The -highres variant uses the same
        # registry shape, swapped to 600-pts/decade table files.
        'PALEOS-2phase:MgSiO3': {
            'core': _seager_iron,
            'melted_mantle': _paleos2ph_melted,
            'solid_mantle': _paleos2ph_solid,
        },
        'PALEOS-2phase:MgSiO3-highres': {
            'core': _seager_iron,
            'melted_mantle': _paleos2ph_melted_highres,
            'solid_mantle': _paleos2ph_solid_highres,
        },
        # PALEOS unified tables (single file per material)
        'PALEOS:iron': _paleos_iron,
        'PALEOS:MgSiO3': _paleos_mgsio3,
        'PALEOS:H2O': _paleos_h2o,
        # PALEOS-API live-tabulated (dispatch materialises cached .dat on first use)
        'PALEOS-API:iron': _paleos_api_iron,
        'PALEOS-API:MgSiO3': _paleos_api_mgsio3,
        'PALEOS-API:H2O': _paleos_api_h2o,
        # PALEOS-API 2-phase MgSiO3 (same solid/liquid split as PALEOS-2phase)
        'PALEOS-API-2phase:MgSiO3': {
            'core': _seager_iron,
            'melted_mantle': _paleos_api_2ph_mgsio3_melted,
            'solid_mantle': _paleos_api_2ph_mgsio3_solid,
        },
        # Chabrier+2019/2021 H/He tables (PALEOS-compatible 10-column format)
        'Chabrier:H': {
            'eos_file': os.path.join(root, 'data', 'EOS_Chabrier2021_HHe', 'chabrier2021_H.dat'),
            'format': 'paleos_unified',
        },
    }

    # ── Legacy tuple-indexed material dictionaries ───────────────────────
    # Kept for backward compatibility with code that uses the old interface.
    legacy = {
        'material_properties_iron_silicate_planets': {
            'core': _seager_iron,
            'mantle': _seager_silicate,
        },
        'material_properties_iron_Tdep_silicate_planets': {
            'core': _seager_iron,
            'melted_mantle': _wb2018_melted,
            'solid_mantle': _wb2018_solid,
        },
        'material_properties_water_planets': {
            'core': _seager_iron,
            'mantle': _seager_silicate,
            'ice_layer': _seager_water,
        },
        'material_properties_iron_RTPress100TPa_silicate_planets': {
            'core': _seager_iron,
            'melted_mantle': _rtpress_melted,
            'solid_mantle': _wb2018_solid,
        },
        'material_properties_iron_PALEOS_silicate_planets': {
            'core': _seager_iron,
            'melted_mantle': _paleos2ph_melted,
            'solid_mantle': _paleos2ph_solid,
        },
    }

    return registry, legacy


# Lazy singleton: built on first access
_registry: dict | None = None
_legacy: dict | None = None


def _ensure_registry():
    """Build the registry if not yet initialized."""
    global _registry, _legacy
    if _registry is None:
        _registry, _legacy = _build_registry()


class _LazyRegistry:
    """Dict-like wrapper that builds the EOS registry on first access."""

    def __getitem__(self, key):
        _ensure_registry()
        return _registry[key]

    def __contains__(self, key):
        _ensure_registry()
        return key in _registry

    def __iter__(self):
        _ensure_registry()
        return iter(_registry)

    def __len__(self):
        _ensure_registry()
        return len(_registry)

    def get(self, key, default=None):
        _ensure_registry()
        return _registry.get(key, default)

    def keys(self):
        _ensure_registry()
        return _registry.keys()

    def values(self):
        _ensure_registry()
        return _registry.values()

    def items(self):
        _ensure_registry()
        return _registry.items()

    def __repr__(self):
        _ensure_registry()
        return repr(_registry)


EOS_REGISTRY = _LazyRegistry()


# Legacy accessors (also lazy)
def __getattr__(name):
    """Module-level __getattr__ for lazy access to legacy material dicts."""
    _legacy_names = {
        'material_properties_iron_silicate_planets',
        'material_properties_iron_Tdep_silicate_planets',
        'material_properties_water_planets',
        'material_properties_iron_RTPress100TPa_silicate_planets',
        'material_properties_iron_PALEOS_silicate_planets',
    }
    if name in _legacy_names:
        _ensure_registry()
        return _legacy[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
