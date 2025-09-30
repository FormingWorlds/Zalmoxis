from __future__ import annotations

import os

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

# Material Properties for iron/silicate planets according to Seager et al. (2007) at 300 K
material_properties_iron_silicate_planets = {
    "core": {
        # Iron, modeled in Seager et al. (2007) using the Vinet EOS fit to the epsilon phase of Fe and TFD theory
        "eos_file": os.path.join(ZALMOXIS_ROOT, "data", "EOS_Seager2007", "eos_seager07_iron.txt")
    },
    "mantle": {
        # Silicate, modeled in Seager et al. (2007) using the fourth-order Birch-Murnaghan EOS fit to MgSiO3 perovskite and TFD theory
        "eos_file": os.path.join(ZALMOXIS_ROOT, "data", "EOS_Seager2007", "eos_seager07_silicate.txt")
    }
}

# Material Properties for iron/silicate planets with iron EOS according to Seager et al. (2007) and silicate melt EOS according to Wolf & Bower (2018)
material_properties_iron_Tdep_silicate_planets = {
    "core": {
        # Iron, modeled in Seager et al. (2007) using the Vinet EOS fit to the epsilon phase of Fe and TFD theory
        "eos_file": os.path.join(ZALMOXIS_ROOT, "data", "EOS_Seager2007", "eos_seager07_iron.txt")
    },
    "melted_mantle": {
        # MgSiO3 in melt state, modeled in Wolf & Bower (2018) using their developed high P–T RTpress EOS
        "eos_file": os.path.join(ZALMOXIS_ROOT, "data", "EOS_WolfBower2018", "density_melt.dat")
    },
    "solid_mantle": {
        # MgSiO3 in solid state, modeled in Wolf & Bower (2018) using their developed high P–T RTpress EOS
        "eos_file": os.path.join(ZALMOXIS_ROOT, "data", "EOS_WolfBower2018", "density_solid.dat")
    }
}

# Material Properties for water planets according to Seager et al. (2007) at 300 K
material_properties_water_planets = {
    "core": {
        # Iron, modeled in Seager et al. (2007) using the Vinet EOS fit to the epsilon phase of Fe and TFD theory
        "eos_file": os.path.join(ZALMOXIS_ROOT, "data", "EOS_Seager2007","eos_seager07_iron.txt")
    },
    "mantle": {
        # Silicate, modeled in Seager et al. (2007) using the fourth-order Birch-Murnaghan EOS fit to MgSiO3 perovskite and TFD theory
        "eos_file": os.path.join(ZALMOXIS_ROOT, "data", "EOS_Seager2007", "eos_seager07_silicate.txt")
    },
    "water_ice_layer": {
        # Water ice, modeled in Seager et al. (2007) using experimental data, DFT predictions for water ice in phases VIII and X, and TFD theory.
        "eos_file": os.path.join(ZALMOXIS_ROOT, "data", "EOS_Seager2007", "eos_seager07_water.txt")
    }
}
