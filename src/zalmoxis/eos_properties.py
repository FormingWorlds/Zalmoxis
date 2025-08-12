from __future__ import annotations

import os

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

# --- Material Properties for iron/silicate planets according to Seager et al. (2007) ---
## only rho0 and eos_file are used in the code and fact checked
material_properties_iron_silicate_planets = {
    "core": {
        # For liquid iron alloy outer core
        "rho0": 8300,  # From Table 1 of Seager et al. (2007) for the epsilon phase of iron of Fe in kg/m^3
        "K0": 140e9,  # Bulk modulus (Pa)
        "K0prime": 5.5,  # Pressure derivative of the bulk modulus
        "gamma0": 1.5,  # Gruneisen parameter
        "theta0": 1200,  # Debye temperature (K)
        "V0": 1 / 9900,  # Specific volume at reference state
        "P0": 135e9,  # Reference pressure (Pa)
        "eos_file": os.path.join(ZALMOXIS_ROOT, "data", "EOS_Seager2007", "eos_seager07_iron.txt") # Name of the file with tabulated EOS data
    },
    "mantle": {
        # Lower mantle properties based on bridgmanite and ferropericlase
        "rho0": 4100, # From Table 1 of Seager et al. (2007) for bridgmanite in kg/m^3
        "K0": 245e9,  # Bulk modulus (Pa)
        "K0prime": 3.9,  # Pressure derivative of the bulk modulus
        "gamma0": 1.5,  # Gruneisen parameter
        "theta0": 1100,  # Debye temperature (K)
        "V0": 1 / 4110,  # Specific volume at reference state
        "P0": 24e9,  # Reference pressure (Pa)
        "eos_file": os.path.join(ZALMOXIS_ROOT, "data", "EOS_Seager2007", "eos_seager07_silicate.txt") # Name of the file with tabulated EOS data
    }
}

# --- Material Properties for water planets according to Seager et al. (2007) ---
material_properties_water_planets = {
    "core": {
        # For liquid iron alloy outer core
        "rho0": 8300,  # From Table 1 of Seager et al. (2007) for the epsilon phase of iron of Fe in kg/m^3
        "eos_file": os.path.join(ZALMOXIS_ROOT, "data", "EOS_Seager2007","eos_seager07_iron.txt")  # Name of the file with tabulated EOS data
    },
    "bridgmanite_shell": {
            # Inner mantle properties based on bridgmanite
            "rho0": 4100,  # From Table 1 of Seager et al. (2007) for bridgmanite in kg/m^3
            "eos_file": os.path.join(ZALMOXIS_ROOT, "data", "EOS_Seager2007", "eos_seager07_silicate.txt")  # Name of the file with tabulated EOS data
    },
    "water_ice_layer": {
        # Outer water ice layer in ice VII phase
            "rho0": 1460,  # From Table 1 of Seager et al. (2007) for H2O in ice VII phase in kg/m^3
            "eos_file": os.path.join(ZALMOXIS_ROOT, "data", "EOS_Seager2007", "eos_seager07_water.txt")  # Name of the file with tabulated EOS data
    }
}


