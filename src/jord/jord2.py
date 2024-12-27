import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant, m^3 kg^-1 s^-2
rho = 5500  # Density of silicate, kg/m^3

# User-defined input
planet_mass = 5.972e24  # Mass of the planet, kg (example: Earth mass)

# Define the differential equations
def equations(r, y):
    m, P, g = y
    if r == 0:
        return [0, 0, 0]  # Avoid division by zero at the center
    dmdr = 4 * np.pi * r**2 * rho
    dPdr = -G * m * rho / r**2
    dgdr = 4 * np.pi * G * rho * r - 2 * g / r
    return [dmdr, dPdr, dgdr]

# Initial conditions
r0 = 1e-6  # Small initial radius to avoid division by zero
m0 = 0  # Mass at the center
P0 = 1e16  # Initial guess for central pressure, Pa
g0 = 0  # Initial gravity at the center

# Define the radius range for integration
r_max = (3 * planet_mass / (4 * np.pi * rho))**(1/3)  # Estimate the radius of the planet
r_span = (r0, r_max)
r_eval = np.linspace(r0, r_max, 1000)

# Solve the differential equations
sol = solve_ivp(equations, r_span, [m0, P0, g0], t_eval=r_eval, method='RK45')

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
ax1.plot(masses, radii / 1e3, label='Enclosed Mass')
ax1.axvline(x=planet_mass, color='r', linestyle='--', label='Total Mass of Earth')
ax1.set_xlabel('Enclosed Mass (kg)')
ax1.set_ylabel('Radius (km)')
ax1.set_title('Mass Distribution')
ax1.set_xscale('log')
ax1.legend()

# Subplot B - Plot the pressure distribution
ax2.plot(pressures / 1e9, radii / 1e3)
ax2.set_xlabel('Pressure (GPa)')
ax2.set_title('Pressure Distribution')
ax2.set_xscale('log')
ax2.axvline(x=136, color='r', linestyle=':', label='Earth\'s core-mantle boundary')
ax2.axvline(x=364, color='r', linestyle='--', label='Earth\'s center')
ax2.legend()

# Subplot C - Plot the gravity distribution
ax3.plot(gravities, radii / 1e3)
ax3.set_xlabel('Gravity (m/s^2)')
ax3.set_title('Gravity Distribution')
ax3.axvline(x=9.81, color='r', linestyle='--', label='Earth\'s Surface Gravity')
ax3.legend()

plt.tight_layout()
plt.show()