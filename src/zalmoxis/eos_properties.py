from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')

# Material Properties for iron/silicate planets according to Seager et al. (2007) at 300 K
material_properties_iron_silicate_planets = {
    'core': {
        # Iron, modeled in Seager et al. (2007) using the Vinet EOS fit to the epsilon phase of Fe and DFT calculations
        'eos_file': os.path.join(
            ZALMOXIS_ROOT, 'data', 'EOS_Seager2007', 'eos_seager07_iron.txt'
        )
    },
    'mantle': {
        # Silicate, modeled in Seager et al. (2007) using the fourth-order Birch-Murnaghan EOS fit to MgSiO3 perovskite and DFT calculations
        'eos_file': os.path.join(
            ZALMOXIS_ROOT, 'data', 'EOS_Seager2007', 'eos_seager07_silicate.txt'
        )
    },
}

# Material Properties for iron/silicate planets with iron EOS according to Seager et al. (2007) and silicate melt EOS according to Wolf & Bower (2018)
_wb_cp_melt = os.path.join(
    ZALMOXIS_ROOT, 'data', 'EOS_WolfBower2018_1TPa', 'heat_capacity_melt.dat'
)
_wb_cp_solid = os.path.join(
    ZALMOXIS_ROOT, 'data', 'EOS_WolfBower2018_1TPa', 'heat_capacity_solid.dat'
)
material_properties_iron_Tdep_silicate_planets = {
    'core': {
        # Iron, modeled in Seager et al. (2007) using the Vinet EOS fit to the epsilon phase of Fe and DFT calculations
        'eos_file': os.path.join(
            ZALMOXIS_ROOT, 'data', 'EOS_Seager2007', 'eos_seager07_iron.txt'
        )
    },
    'melted_mantle': {
        # MgSiO3 in melt state, modeled in Wolf & Bower (2018) using their developed high P–T RTpress EOS
        'eos_file': os.path.join(
            ZALMOXIS_ROOT, 'data', 'EOS_WolfBower2018_1TPa', 'density_melt.dat'
        ),
        **({'cp_file': _wb_cp_melt} if os.path.isfile(_wb_cp_melt) else {}),
    },
    'solid_mantle': {
        # MgSiO3 in solid state, modeled in Wolf & Bower (2018) using their developed high P–T RTpress EOS
        'eos_file': os.path.join(
            ZALMOXIS_ROOT, 'data', 'EOS_WolfBower2018_1TPa', 'density_solid.dat'
        ),
        **({'cp_file': _wb_cp_solid} if os.path.isfile(_wb_cp_solid) else {}),
    },
}

# Material Properties for iron/silicate planets with iron EOS according to Seager et al. (2007)
# and silicate melt EOS from the extended RTpress table (100 TPa) with solid from Wolf & Bower (2018)
_rt_cp_melt = os.path.join(
    ZALMOXIS_ROOT, 'data', 'EOS_RTPress_melt_100TPa', 'heat_capacity_melt.dat'
)
if not os.path.isfile(_rt_cp_melt):
    logger.warning(
        'RTPress100TPa melt Cp table not found at %s. '
        'Adiabatic mode will fall back to constant Cp for this EOS.',
        _rt_cp_melt,
    )
material_properties_iron_RTPress100TPa_silicate_planets = {
    'core': {
        # Iron, modeled in Seager et al. (2007) using the Vinet EOS fit to the epsilon phase of Fe and DFT calculations
        'eos_file': os.path.join(
            ZALMOXIS_ROOT, 'data', 'EOS_Seager2007', 'eos_seager07_iron.txt'
        )
    },
    'melted_mantle': {
        # MgSiO3 in melt state, extended RTpress EOS table (P: 1e3–1e14 Pa, T: 400–50000 K)
        'eos_file': os.path.join(
            ZALMOXIS_ROOT, 'data', 'EOS_RTPress_melt_100TPa', 'density_melt.dat'
        ),
        **({'cp_file': _rt_cp_melt} if os.path.isfile(_rt_cp_melt) else {}),
    },
    'solid_mantle': {
        # MgSiO3 in solid state, from Wolf & Bower (2018) / Mosenfelder et al. (2009) (clamped at 1 TPa boundary)
        'eos_file': os.path.join(
            ZALMOXIS_ROOT, 'data', 'EOS_WolfBower2018_1TPa', 'density_solid.dat'
        ),
        # Solid Cp from WolfBower2018 (same tables used for solid density)
        **({'cp_file': _wb_cp_solid} if os.path.isfile(_wb_cp_solid) else {}),
    },
}

# Material Properties for water planets according to Seager et al. (2007) at 300 K
material_properties_water_planets = {
    'core': {
        # Iron, modeled in Seager et al. (2007) using the Vinet EOS fit to the epsilon phase of Fe and DFT calculations
        'eos_file': os.path.join(
            ZALMOXIS_ROOT, 'data', 'EOS_Seager2007', 'eos_seager07_iron.txt'
        )
    },
    'mantle': {
        # Silicate, modeled in Seager et al. (2007) using the fourth-order Birch-Murnaghan EOS fit to MgSiO3 perovskite and DFT calculations
        'eos_file': os.path.join(
            ZALMOXIS_ROOT, 'data', 'EOS_Seager2007', 'eos_seager07_silicate.txt'
        )
    },
    'ice_layer': {
        # Water ice, modeled in Seager et al. (2007) using experimental data, DFT predictions for water ice in phases VIII and X, and DFT calculations.
        'eos_file': os.path.join(
            ZALMOXIS_ROOT, 'data', 'EOS_Seager2007', 'eos_seager07_water.txt'
        )
    },
}
