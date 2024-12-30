"""
EOS Data and Functions
"""

from .eos_properties import material_properties
from scipy.interpolate import interp1d
import numpy as np

def mie_gruneisen_debye(P, P0, rho0, K0, K0prime, gamma0, theta0, V0, T):
    """
    Calculates density from the Mie-Grüneisen-Debye EOS.
    """
    V = V0 * (P0 / (P + P0))**(1/K0prime)
    rho = rho0 * (V0 / V)

    # Thermal pressure correction (simplified)
    gamma = gamma0 * (V / V0)**1  # Assuming q = 1 for simplicity
    theta = theta0 * (V / V0)**(-gamma)
    P_thermal = (gamma * rho * 8.314 * (T - 300))  # Simplified thermal pressure

    return rho

def birch_murnaghan(P, P0, rho0, K0, K0prime, V0):
    """
    Calculates density from the 3rd order Birch-Murnaghan EOS.
    """
    eta = (3/2) * (K0prime - 4)
    V = V0 * (1 + (3/4) * (K0prime - 4) * ((P - P0)/K0))**(-2/(3 * (K0prime - 4)))
    
    density = rho0 * (V0 / V)

    return density


# --- Temperature Profile (Adiabatic) ---
def calculate_temperature(radius, core_radius, surface_temp, cmb_temp, core_temp, radius_guess):
    """
    Calculates an adiabatic temperature profile.
    """
    if radius > core_radius:
        # Mantle adiabat
        temperature = surface_temp + (cmb_temp - surface_temp) * (
            (radius_guess - radius) / (radius_guess - core_radius)
        )**0.45
    else:
        # Core adiabat
        temperature = core_temp * (1 - (radius / core_radius)**2)**0.3 + core_temp * 0.8
    return temperature

# --- EOS Calculation ---
def calculate_density(pressure, radius, core_radius, material, radius_guess, cmb_temp, core_temp, eos_choice, interpolation_functions={}):
    """Calculates density with caching for tabulated EOS."""

    T = calculate_temperature(radius, core_radius, 300, cmb_temp, core_temp, radius_guess)
    props = material_properties[material]

    props = material_properties[material]  # Shorthand

    if eos_choice == "Mie-Gruneisen-Debye":
        density = mie_gruneisen_debye(pressure, props["P0"], props["rho0"], props["K0"], props["K0prime"], props["gamma0"], props["theta0"], props["V0"], T)
        return density
    elif eos_choice == "Birch-Murnaghan":
        density = birch_murnaghan(pressure, props["P0"], props["rho0"], props["K0"], props["K0prime"], props["V0"])
        return density
    elif eos_choice == "Tabulated":
        try:
            eos_file = props["eos_file"]
            # Caching: Store interpolation functions for reuse
            if eos_file not in interpolation_functions:
                data = np.loadtxt(eos_file, delimiter=',', skiprows=1)
                pressure_data = data[:, 1] * 1e9
                density_data = data[:, 0] * 1e3
                interpolation_functions[eos_file] = interp1d(pressure_data, density_data, bounds_error=False, fill_value="extrapolate")

            interpolation_function = interpolation_functions[eos_file]  # Retrieve from cache
            density = interpolation_function(pressure)  # Call to cached function

            if density is None or np.isnan(density):
                raise ValueError(f"Density calculation failed for {material} at {pressure:.2e} Pa.")

            return density

        except (ValueError, OSError) as e: # Catch file errors
            print(f"Error with tabulated EOS for {material} at {pressure:.2e} Pa: {e}")
            return None
        except Exception as e: # Other errors
            print(f"Unexpected error with tabulated EOS for {material} at {pressure:.2e} Pa: {e}")
            return None

    else:
        raise ValueError("Invalid EOS choice.")


# Define the ODEs for mass, gravity and pressure
def coupled_odes(radius, y):
    mass, gravity, pressure = y

    if radius < cmb_radius:
        material = "core"
    else:
        material = "mantle" # Assign material only once per call
    
    # Calculate density at the current radius, using pressure from y
    current_density = calculate_density(pressure, radius, cmb_radius, material, radius_guess, cmb_temp_guess, core_temp_guess, EOS_CHOICE, interpolation_cache)

    # Handle potential errors in density calculation
    if current_density is None:
        print(f"Warning: Density calculation failed at radius {radius}. Using previous density.") # Print warning only
        current_density = old_density[np.argmin(np.abs(radii - radius))]

    # Calculate temperature
    temperature = calculate_temperature(radius, cmb_radius, 300, cmb_temp_guess, core_temp_guess, radius_guess)

    dMdr = 4 * np.pi * radius**2 * current_density
    dgdr = 4 * np.pi * G * current_density - 2 * gravity / (radius + 1e-20) if radius > 0 else 0
    dPdr = -current_density * gravity

    return [dMdr, dgdr, dPdr]