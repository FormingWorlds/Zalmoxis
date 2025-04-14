# --- Material Properties ---
material_properties = {
    "mantle": {
        # Lower mantle properties based on bridgmanite and ferropericlase
        "rho0": 4110,  # Reference density (kg/m^3) at 24 GPa (top of lower mantle)
        "K0": 245e9,  # Bulk modulus (Pa)
        "K0prime": 3.9,  # Pressure derivative of the bulk modulus
        "gamma0": 1.5,  # Gruneisen parameter
        "theta0": 1100,  # Debye temperature (K)
        "V0": 1 / 4110,  # Specific volume at reference state
        "P0": 24e9,  # Reference pressure (Pa)
        "eos_file": "../../data/eos_seager07_silicate.txt" # Name of the file with tabulated EOS data
    },
    "core": {
        # For liquid iron alloy outer core
        "rho0": 9900,  # Reference density (kg/m^3) at 135 GPa (core-mantle boundary)
        "K0": 140e9,  # Bulk modulus (Pa)
        "K0prime": 5.5,  # Pressure derivative of the bulk modulus
        "gamma0": 1.5,  # Gruneisen parameter
        "theta0": 1200,  # Debye temperature (K)
        "V0": 1 / 9900,  # Specific volume at reference state
        "P0": 135e9,  # Reference pressure (Pa)
        "eos_file": "../../data/eos_seager07_iron.txt" # Name of the file with tabulated EOS data
    }
}