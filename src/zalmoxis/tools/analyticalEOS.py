import numpy as np
import os
from ..eos_properties import analytical_eos_Seager07_parameters

# Set the working directory to the current file
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Generate pressure values sampled logarithmically from 1e-8 GPa to 1e12 GPa and include 0
P_GPa = np.concatenate(([0], np.logspace(-8, 12, num=2001)))  # Include 0 at the start

# Convert pressure from GPa to Pa
P_Pa = P_GPa * 1e9

# Iterate over each material in the analytical_eos_Seager07 dictionary
for material in analytical_eos_Seager07_parameters.keys():
    # Compute density using the modified polytropic EOS for the current material
    rho = (
        analytical_eos_Seager07_parameters[material]["rho0"]
        + analytical_eos_Seager07_parameters[material]["c"] * P_Pa**analytical_eos_Seager07_parameters[material]["n"]
    )

    # Convert density back to g/cm^3 (1 kg/m^3 = 0.001 g/cm^3)
    rho_g_cm3 = rho * 0.001

    # Save results to a text file
    output_file = f"../../../data/analytical_eos_seager07_{material}.txt"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as file:
        file.write("Density (g/cm^3), Pressure (GPa)\n")
        for density, pressure in zip(rho_g_cm3, P_GPa):
            file.write(f"{density:.9e}, {pressure:.6e}\n") 
    print(f"Data for {material} saved to {output_file}")
