import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)

# --- Input Parameter ---
planet_mass = 5.972e24  # kg (Earth mass)

# --- Assumptions and Initial Guesses ---
core_radius_fraction = 0.545  # Core radius as a fraction of the total radius (Earth-like)
avg_density_guess = 5515  # Initial guess for average density (kg/m^3)

# --- EOS Choice ---
EOS_CHOICE = "Birch-Murnaghan"  # "Birch-Murnaghan", "Mie-Gruneisen-Debye", or "Tabulated"

# --- EOS Data and Functions ---

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
        "eos_file": "mantle.txt" # Name of the file with tabulated EOS data
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
        "eos_file": "core.txt" # Name of the file with tabulated EOS data
    }
}

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

            interpolation_function = interpolation_functions[eos_file] # Retrieve from cache


            # Hybrid approach (using cached interpolation function)
            P_transition = 75e9
            smoothing_range = 20e9

            if pressure <= P_transition - smoothing_range / 2:
                density = birch_murnaghan(pressure, props["P0"], props["rho0"], props["K0"], props["K0prime"], props["V0"])
            elif pressure >= P_transition + smoothing_range / 2:
                density = interpolation_function(pressure)  # Call to cached function
            else:
                density_tabulated = interpolation_function(pressure)   # Call to cached function
                density_birch = birch_murnaghan(pressure, props["P0"], props["rho0"], props["K0"], props["K0prime"], props["V0"])
                weight_tabulated = (pressure - (P_transition - smoothing_range / 2)) / smoothing_range
                weight_birch = 1 - weight_tabulated
                density = weight_birch * density_birch + weight_tabulated * density_tabulated


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

# --- Calculations ---
num_layers = 500 # Reduced number of layers for faster computation

# Iterative Process to find Radius and Density Profile
max_iterations_outer = 100
tolerance_outer = 1e-4
max_iterations_inner = 100
tolerance_inner = 1e-4

radius_guess = (3 * planet_mass / (4 * math.pi * avg_density_guess))**(1/3)

# Reference values for Earth (PREM and other sources)
earth_radius = 6371e3  # m
earth_cmb_radius = 3480e3  # m
earth_surface_pressure = 101325  # Pa (1 atm)
earth_cmb_pressure = 135e9  # Pa
earth_center_pressure = 364e9  # Pa
earth_surface_temperature = 288  # K (approximate average)
earth_cmb_temperature = 4100 # K
earth_center_temperature = 5300 # K
earth_center_density = 13000 # kg/m^3, from PREM

for outer_iter in range(max_iterations_outer):
    start_time = time.time()
    # Define radial layers:
    radii = np.linspace(0, radius_guess, num_layers)
    dr = radii[1] - radii[0]

    # Initialize arrays:
    density = np.zeros(num_layers)
    mass_enclosed = np.zeros(num_layers)
    gravity = np.zeros(num_layers)
    pressure = np.zeros(num_layers)
    temperature = np.zeros(num_layers)

    # Initial density guess:
    cmb_radius = core_radius_fraction * radius_guess

    # Initial temperature at the core-mantle boundary (CMB) and the center
    cmb_temp_guess = 4100  # Initial guess for CMB temperature (K)
    core_temp_guess = 5300

    # Estimate initial pressure at the center (needed for solve_ivp)
    pressure[0] = earth_center_pressure

    for i in range(num_layers):
        if radii[i] < cmb_radius:
            # Core (simplified initial guess)
            density[i] = material_properties["core"]["rho0"]
        else:
            # Mantle (simplified initial guess)
            density[i] = material_properties["mantle"]["rho0"]

    for inner_iter in range(max_iterations_inner):
        old_density = density.copy() # Avoid unnecessary copying in each iteration

        # Caching for density interpolation in coupled_odes
        interpolation_cache = {}  # Initialize empty cache

        # a. Calculate enclosed mass, gravity and pressure using 4th order Runge-Kutta:
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

        # Initial conditions for solve_ivp
        y0 = [0, 0, pressure[0]]  # Initial mass, gravity, pressure at r=0

        # Solve the ODEs using solve_ivp
        sol = solve_ivp(coupled_odes, (radii[0], radii[-1]), y0, t_eval=radii, method='RK45')

        # Extract mass, gravity, and pressure profiles
        mass_enclosed = sol.y[0]
        gravity = sol.y[1]
        pressure = sol.y[2]

        # d. Update density based on pressure using EOS:
        for i in range(num_layers):
            if radii[i] < cmb_radius:
                # Core
                material = "core"
            else:
                # Mantle
                material = "mantle"

            new_density = calculate_density(pressure[i], radii[i], cmb_radius, material, radius_guess, cmb_temp_guess, core_temp_guess, EOS_CHOICE)

            # Handle potential errors in density calculation
            if new_density is None:
                print(f"Warning: Density calculation failed at radius {radii[i]}. Using previous density.")
                new_density = old_density[i]

            # Calculate temperature
            temperature[i] = calculate_temperature(radii[i], cmb_radius, 300, cmb_temp_guess, core_temp_guess, radius_guess)

            # Relaxation
            density[i] = 0.5 * (new_density + old_density[i]) # Use simple averaging for relaxation

        # Check for convergence (inner loop):
        relative_diff_inner = np.max(np.abs((density - old_density) / (old_density + 1e-20)))
        if relative_diff_inner < tolerance_inner:
            break

    # Calculate total mass:
    calculated_mass = mass_enclosed[-1]

    # Update radius guess and core-mantle boundary:
    radius_guess = radius_guess * (planet_mass / calculated_mass)**(1/3)
    cmb_radius = core_radius_fraction * radius_guess

    # Check for convergence (outer loop):
    relative_diff_outer = abs((calculated_mass - planet_mass) / planet_mass)
    if relative_diff_outer < tolerance_outer:
        print(f"Outer loop (radius and cmb) converged after {outer_iter + 1} iterations.")
        break
    end_time = time.time()
    print(f"Outer iteration {outer_iter+1} took {end_time - start_time:.2f} seconds")

    if outer_iter == max_iterations_outer - 1:
        print(f"Warning: Maximum outer iterations ({max_iterations_outer}) reached. Radius and cmb may not be fully converged.")

planet_radius = radius_guess

# --- Output ---
cmb_index = np.argmin(np.abs(radii - cmb_radius))
average_density = planet_mass / (4/3 * math.pi * planet_radius**3)

print("Exoplanet Internal Structure Model (Mass Only Input, with Improved EOS)")
print("----------------------------------------------------------------------")
print(f"Planet Mass: {planet_mass:.2e} kg")
print(f"Planet Radius (Self-Consistently Calculated): {planet_radius:.2e} m")
print(f"Core Radius: {cmb_radius:.2e} m")
print(f"Mantle Density (at CMB): {density[cmb_index]:.2f} kg/m^3")
print(f"Core Density (at CMB): {density[cmb_index - 1]:.2f} kg/m^3")
print(f"Pressure at Core-Mantle Boundary (CMB): {pressure[cmb_index]:.2e} Pa")
print(f"Pressure at Center: {pressure[0]:.2e} Pa")
print(f"Average Density: {average_density:.2f} kg/m^3")

# --- Plotting ---

fig, ax = plt.subplots(1, 4, figsize=(16, 6))

# Density vs. Radius
ax[0].plot(density, radii / 1e3, label="Exoplanet Model")
ax[0].set_xlabel("Density (kg/m^3)")
ax[0].set_ylabel("Radius (km)")
ax[0].set_title("Exoplanet Internal Structure")
ax[0].axhline(y=cmb_radius / 1e3, color='r', linestyle='--', label="CMB")
ax[0].grid()

# Add average density as a vertical line
ax[0].axvline(x=average_density, color='b', linestyle='-.', label=f"Average Density = {average_density:.0f} kg/m^3")

# Gravity vs. Radius
ax[1].plot(gravity, radii / 1e3, label="Exoplanet Model")
ax[1].set_xlabel("Gravity (m/s^2)")
ax[1].set_ylabel("Radius (km)")
ax[1].axhline(y=cmb_radius / 1e3, color='r', linestyle='--')
ax[1].grid()

# Pressure vs. Radius
ax[2].plot(pressure / 1e9, radii / 1e3, label="Exoplanet Model")
ax[2].set_xlabel("Pressure (GPa)")
ax[2].set_ylabel("Radius (km)")
ax[2].axhline(y=cmb_radius / 1e3, color='r', linestyle='--')
ax[2].grid()

# Temperature vs. Radius
ax[3].plot(temperature, radii / 1e3, label="Exoplanet Model")
ax[3].set_xlabel("Temperature (K)")
ax[3].set_ylabel("Radius (km)")
ax[3].axhline(y=cmb_radius / 1e3, color='r', linestyle='--')
ax[3].grid()

# Add reference Earth values to the plots
ax[0].plot(0, earth_radius / 1e3, 'ro', label="Earth's Surface")
ax[0].axhline(y=earth_cmb_radius / 1e3, color='r', linestyle=':', label="Earth's CMB")
ax[0].plot(5515, earth_radius / 1e3, 'go', label='Earth Average Density')
ax[0].plot(earth_center_density, 0, 'yo', label="Earth's Center Density")

ax[1].plot(0, earth_radius / 1e3, 'ro')
ax[1].axhline(y=earth_cmb_radius / 1e3, color='r', linestyle=':')
ax[1].plot(9.81, earth_radius / 1e3, 'bo', label="Earth's Surface Gravity") # Surface gravity

ax[2].plot(earth_surface_pressure / 1e9, earth_radius / 1e3, 'ro')
ax[2].plot(earth_cmb_pressure / 1e9, earth_cmb_radius / 1e3, 'ro', label="Earth's CMB")
ax[2].plot(earth_center_pressure / 1e9, 0, 'ro', label="Earth's Center")
ax[2].axhline(y=earth_cmb_radius / 1e3, color='r', linestyle=':')

ax[3].plot(earth_surface_temperature, earth_radius / 1e3, 'ro')
ax[3].plot(earth_cmb_temperature, earth_cmb_radius / 1e3, 'ro', label="Earth's CMB")
ax[3].plot(earth_center_temperature, 0, 'ro', label="Earth's Center")
ax[3].axhline(y=earth_cmb_radius / 1e3, color='r', linestyle=':')

# Save plotted data to output files
np.savetxt("density_vs_radius.txt", np.column_stack((radii, density)), header="Radius (m)\tDensity (kg/m^3)")
np.savetxt("gravity_vs_radius.txt", np.column_stack((radii, gravity)), header="Radius (m)\tGravity (m/s^2)")
np.savetxt("pressure_vs_radius.txt", np.column_stack((radii, pressure)), header="Radius (m)\tPressure (Pa)")
np.savetxt("temperature_vs_radius.txt", np.column_stack((radii, temperature)), header="Radius (m)\tTemperature (K)")

# Add legends
for a in ax:
    a.legend()

plt.tight_layout()
plt.show()
