import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import os

# Constants
G = 6.67430e-11  # Gravitational constant, m^3 kg^-1 s^-2

# User-defined input
planet_mass = 5.972e24  # Mass of the planet, kg (example: Earth mass)

# Read the EOS data for silicates
data_folder = '../data/'
eos_file = os.path.join(data_folder, 'eos_seager07_silicate.txt')
eos_data = np.loadtxt(eos_file, delimiter=',', skiprows=1)
pressures_eos = eos_data[:, 1] * 1e9  # Convert GPa to Pa
densities_eos = eos_data[:, 0] * 1000  # Convert g/cm^3 to kg/m^3

# Create an interpolation function for density as a function of pressure
density_interp = interp1d(pressures_eos, densities_eos, bounds_error=False, fill_value="extrapolate")

# Define the differential equations
def equations(r, y):
    m, P, g = y
    if r == 0:
        return [0, 0, 0]  # Avoid division by zero at the center
    rho = density_interp(P)
    dmdr = 4 * np.pi * r**2 * rho
    dPdr = -G * m * rho / r**2
    dgdr = 4 * np.pi * G * rho * r - 2 * g / r
    return [dmdr, dPdr, dgdr]

# Initial conditions
r0 = 1e-6  # Small initial radius to avoid division by zero
m0 = 0  # Mass at the center
g0 = 0  # Initial gravity at the center

# Define the bisection method to find the correct central pressure
def bisection_method(target_mass, P_min, P_max, tol=1e-4, max_iter=100):
    for i in range(max_iter):
        P0 = (P_min + P_max) / 2
        r_max = (3 * target_mass / (4 * np.pi * density_interp(P0)))**(1/3)
        r_span = (r0, r_max)
        r_eval = np.linspace(r0, r_max, 1000)
        sol = solve_ivp(equations, r_span, [m0, P0, g0], t_eval=r_eval, method='RK45')
        m_final = sol.y[0][-1]
        print(f"Iteration {i}: P0={P0:.2e}, m_final={m_final:.2e}, target_mass={target_mass:.2e}")
        if abs(m_final - target_mass) < tol:
            return P0, sol
        elif m_final > target_mass:
            P_max = P0
        else:
            P_min = P0
    raise ValueError("Bisection method did not converge")

# Use the bisection method to find the central pressure
P_min = 1e5  # Lower bound for central pressure, Pa
P_max = 1e35  # Upper bound for central pressure, Pa
P0, sol = bisection_method(planet_mass, P_min, P_max)

# Extract the results
radii = sol.t
masses = sol.y[0]
pressures = sol.y[1]
gravities = sol.y[2]

# Combine the results into one array
results = np.column_stack((radii, masses, pressures, gravities))

# Save the results to a single .txt file
header = 'Radius (m)\tEnclosed Mass (kg)\tPressure (Pa)\tGravity (m/s^2)'
np.savetxt('planet_structure.txt', results, header=header)

# Create the plot
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6), sharey=True)

# Subplot A - Plot the mass distribution
ax1.plot(masses / 1e24, radii / 1e3, label='Enclosed Mass')
ax1.axvline(x=planet_mass / 1e24, color='r', linestyle='--', label='Total Mass of Earth')
ax1.set_xlabel('Enclosed Mass (10^24 kg)')
ax1.set_ylabel('Radius (km)')
ax1.set_title('Mass Distribution')
ax1.set_xscale('log')
ax1.legend()

# Subplot B - Plot the pressure distribution
ax2.plot(pressures / 1e9, radii / 1e3)
ax2.set_xlabel('Pressure (GPa)')
ax2.set_title('Pressure Distribution')
ax2.set_xscale('log')

# Subplot C - Plot the gravity distribution
ax3.plot(gravities, radii / 1e3)
ax3.set_xlabel('Gravity (m/s^2)')
ax3.set_title('Gravity Distribution')

# Invert the y-axis to make it downward-increasing
ax1.invert_yaxis()
ax2.invert_yaxis()
ax3.invert_yaxis()

plt.tight_layout()
plt.show()