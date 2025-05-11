# --- Material Properties ---
material_properties = {
    "mantle": {
        # Lower mantle properties based on bridgmanite and ferropericlase
        "rho0": 4100, # From Table 1 of Seager et al. (2007) for bridgmanite
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
        "rho0": 8300,  # From Table 1 of Seager et al. (2007) for the epsilon phase of iron of Fe
        "K0": 140e9,  # Bulk modulus (Pa)
        "K0prime": 5.5,  # Pressure derivative of the bulk modulus
        "gamma0": 1.5,  # Gruneisen parameter
        "theta0": 1200,  # Debye temperature (K)
        "V0": 1 / 9900,  # Specific volume at reference state
        "P0": 135e9,  # Reference pressure (Pa)
        "eos_file": "../../data/eos_seager07_iron.txt" # Name of the file with tabulated EOS data
    }
}

analytical_eos_Seager07_parameters = {
    "Fe": {
        # For Fe(alpha) phase of iron
        "rho0": 8300.00,  # kg/m^3
        "c": 0.00349,  # kg/m^3/Pa^n
        "n": 0.528,  # dimensionless
    },
    "MgSiO3": {
        # For MgSiO3 (perovskite) phase
        "rho0": 4100.00,  # kg/m^3
        "c": 0.00161,  # kg/m^3/Pa^n
        "n": 0.541,  # dimensionless
    },
    "H2O": {
        # For H2O (ice) phase
        "rho0": 1460.00,  # kg/m^3
        "c": 0.00311,  # kg/m^3/Pa^n
        "n": 0.513,  # dimensionless
    }
}