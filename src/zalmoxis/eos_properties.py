from __future__ import annotations

import os

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')

# ── Seager et al. (2007), 300 K static EOS ──────────────────────────
_seager_iron = {
    'eos_file': os.path.join(ZALMOXIS_ROOT, 'data', 'EOS_Seager2007', 'eos_seager07_iron.txt')
}
_seager_silicate = {
    'eos_file': os.path.join(
        ZALMOXIS_ROOT, 'data', 'EOS_Seager2007', 'eos_seager07_silicate.txt'
    )
}
_seager_water = {
    'eos_file': os.path.join(ZALMOXIS_ROOT, 'data', 'EOS_Seager2007', 'eos_seager07_water.txt')
}

# ── Wolf & Bower (2018), T-dependent MgSiO3 (up to 1 TPa) ──────────
_wb2018_melted = {
    'eos_file': os.path.join(
        ZALMOXIS_ROOT, 'data', 'EOS_WolfBower2018_1TPa', 'density_melt.dat'
    ),
    'adiabat_grad_file': os.path.join(
        ZALMOXIS_ROOT, 'data', 'EOS_WolfBower2018_1TPa', 'adiabat_temp_grad_melt.dat'
    ),
}
_wb2018_solid = {
    'eos_file': os.path.join(
        ZALMOXIS_ROOT, 'data', 'EOS_WolfBower2018_1TPa', 'density_solid.dat'
    ),
}

# ── RTPress 100 TPa extended melt + WolfBower2018 solid ─────────────
_rtpress_melted = {
    'eos_file': os.path.join(
        ZALMOXIS_ROOT, 'data', 'EOS_RTPress_melt_100TPa', 'density_melt.dat'
    ),
    'adiabat_grad_file': os.path.join(
        ZALMOXIS_ROOT, 'data', 'EOS_RTPress_melt_100TPa', 'adiabat_temp_grad_melt.dat'
    ),
}

# ── PALEOS-2phase MgSiO3 (solid + liquid, Zenodo 18924171) ──────────
_paleos2ph_melted = {
    'eos_file': os.path.join(
        ZALMOXIS_ROOT,
        'data',
        'EOS_PALEOS_MgSiO3',
        'paleos_mgsio3_tables_pt_proteus_liquid.dat',
    ),
    'format': 'paleos',
}
_paleos2ph_solid = {
    'eos_file': os.path.join(
        ZALMOXIS_ROOT,
        'data',
        'EOS_PALEOS_MgSiO3',
        'paleos_mgsio3_tables_pt_proteus_solid.dat',
    ),
    'format': 'paleos',
}

# ── PALEOS unified tables (Zenodo 19000316) ─────────────────────────
_paleos_iron = {
    'eos_file': os.path.join(
        ZALMOXIS_ROOT, 'data', 'EOS_PALEOS_iron', 'paleos_iron_eos_table_pt.dat'
    ),
    'format': 'paleos_unified',
}
_paleos_mgsio3 = {
    'eos_file': os.path.join(
        ZALMOXIS_ROOT, 'data', 'EOS_PALEOS_MgSiO3_unified', 'paleos_mgsio3_eos_table_pt.dat'
    ),
    'format': 'paleos_unified',
}
_paleos_h2o = {
    'eos_file': os.path.join(
        ZALMOXIS_ROOT, 'data', 'EOS_PALEOS_H2O', 'paleos_water_eos_table_pt.dat'
    ),
    'format': 'paleos_unified',
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
EOS_REGISTRY = {
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
    # PALEOS-2phase MgSiO3 (separate solid/liquid tables)
    'PALEOS-2phase:MgSiO3': {
        'core': _seager_iron,
        'melted_mantle': _paleos2ph_melted,
        'solid_mantle': _paleos2ph_solid,
    },
    # PALEOS unified tables (single file per material)
    'PALEOS:iron': _paleos_iron,
    'PALEOS:MgSiO3': _paleos_mgsio3,
    'PALEOS:H2O': _paleos_h2o,
}

# ── Legacy tuple-indexed material dictionaries ───────────────────────
# These are kept for backward compatibility with code that still uses
# the old 5-tuple interface. They are simple views into the registry.

material_properties_iron_silicate_planets = {
    'core': _seager_iron,
    'mantle': _seager_silicate,
}

material_properties_iron_Tdep_silicate_planets = {
    'core': _seager_iron,
    'melted_mantle': _wb2018_melted,
    'solid_mantle': _wb2018_solid,
}

material_properties_water_planets = {
    'core': _seager_iron,
    'mantle': _seager_silicate,
    'ice_layer': _seager_water,
}

material_properties_iron_RTPress100TPa_silicate_planets = {
    'core': _seager_iron,
    'melted_mantle': _rtpress_melted,
    'solid_mantle': _wb2018_solid,
}

material_properties_iron_PALEOS_silicate_planets = {
    'core': _seager_iron,
    'melted_mantle': _paleos2ph_melted,
    'solid_mantle': _paleos2ph_solid,
}
