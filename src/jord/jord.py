import os, sys, time, math, toml
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from .constants import *
from .eos_functions import calculate_density, calculate_temperature, birch_murnaghan, mie_gruneisen_debye
from .eos_properties import material_properties
from .structure_model import coupled_odes
from .plots.plot_profiles import plot_planet_profile_single
from .plots.plot_eos import plot_eos_material

# Run file via command line: python -m src.jord.jord -c ../../input/default.toml

def main():
    
    """
    Main function to run the exoplanet internal structure model.

    This function reads the configuration file, initializes parameters, and performs
    an iterative solution to calculate the internal structure of an exoplanet based on
    the given mass and other parameters. It outputs the calculated planet radius, core
    radius, densities, pressures, and temperatures at various layers, and optionally
    saves the data to a file and plots the results.
    """

    # Set the working directory to the current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load the configuration file either from terminal (-c flag) or default path
    if "-c" in sys.argv:
        index = sys.argv.index("-c")
        try:
            config_file_path = sys.argv[index + 1]
            config = toml.load(config_file_path)
            print(f"Reading config file from: {config_file_path}")
        except IndexError:
            print("Error: -c flag provided but no config file path specified.")
            sys.exit(1) # Exit with error code
        except FileNotFoundError:
            print(f"Error: Config file not found at {config_file_path}")
            sys.exit(1)
    else:
        config_default_path = "../../input/default.toml"
        config = toml.load(config_default_path)
        print(f"Reading default config file from {config_default_path}")

    # Access parameters from the configuration file
    planet_mass = config['InputParameter']['planet_mass']  # Mass of the planet (kg)
    core_radius_fraction = config['AssumptionsAndInitialGuesses']['core_radius_fraction']  # Initial guess for the core radius as a fraction of the total radius
    core_mass_fraction = config['AssumptionsAndInitialGuesses']['core_mass_fraction']  # Initial guess for the core mass as a fraction of the total mass
    EOS_CHOICE = config['EOS']['choice']  # Choice of equation of state (e.g., "Birch-Murnaghan", "Mie-Gruneisen-Debye", "Tabulated")
    num_layers = config['Calculations']['num_layers']  # Number of radial layers for calculations

    # Parameters for the iterative solution process
    max_iterations_outer = config['IterativeProcess']['max_iterations_outer']  # Maximum iterations for the outer loop (radius and CMB adjustment)
    tolerance_outer = config['IterativeProcess']['tolerance_outer']  # Convergence tolerance for the outer loop
    max_iterations_inner = config['IterativeProcess']['max_iterations_inner']  # Maximum iterations for the inner loop (density profile)
    tolerance_inner = config['IterativeProcess']['tolerance_inner']  # Convergence tolerance for the inner loop

    # Parameters for adjusting the surface pressure to the target value
    target_surface_pressure = config['PressureAdjustment']['target_surface_pressure']  # Target surface pressure (Pa)
    pressure_tolerance = config['PressureAdjustment']['pressure_tolerance']  # Tolerance for surface pressure convergence
    max_iterations_pressure = config['PressureAdjustment']['max_iterations_pressure']  # Maximum iterations for pressure adjustment
    pressure_relaxation = config['PressureAdjustment']['pressure_relaxation'] # Relaxation factor for pressure adjustment (currently unused, but kept for potential future use)
    pressure_adjustment_factor = config['PressureAdjustment']['pressure_adjustment_factor']  # Adjustment factor for updating the central pressure guess

    # Output control parameters
    data_output_enabled = config['Output']['data_enabled']  # Flag to enable saving data to a file (True/False)
    plotting_enabled = config['Output']['plots_enabled']  # Flag to enable plotting the results (True/False)


    # Initial radius guess based on mass and average
    avg_density_guess = 5515  # kg/m^3

    # Initial radius guess based on mass and average
    radius_guess = (3 * planet_mass / (4 * math.pi * avg_density_guess))**(1/3)

    # --- Iterative Solution ---
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

        # Initial cmb radius and mass guesses:
        cmb_radius = core_radius_fraction * radius_guess
        cmb_mass = core_mass_fraction * planet_mass

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

            # Initial conditions for solve_ivp - initial pressure guess
            pressure_guess = earth_center_pressure # or some other initial guess
            adjustment_factor = pressure_adjustment_factor  # Initial adjustment factor for pressure

            for _ in range(max_iterations_pressure): # Innermost loop for pressure adjustment

                # Initial conditions for solve_ivp
                y0 = [0, 0, pressure_guess]  # Initial mass, gravity, pressure at r=0

                # Solve the ODEs using solve_ivp
                # sol = solve_ivp(coupled_odes, (radii[0], radii[-1]), y0, t_eval=radii, method='RK45', dense_output=True)
                sol = solve_ivp(lambda r, y: coupled_odes(r, y, cmb_radius, radius_guess, cmb_temp_guess, core_temp_guess, EOS_CHOICE, interpolation_cache), 
                    (radii[0], radii[-1]), y0, t_eval=radii, method='RK45', dense_output=True)


                # Extract mass, gravity, and pressure profiles
                mass_enclosed = sol.y[0]
                gravity = sol.y[1]
                pressure = sol.y[2]

                surface_pressure = pressure[-1]
                pressure_diff = surface_pressure - target_surface_pressure

                if abs(pressure_diff) < pressure_tolerance:
                    print("Surface pressure converged!")
                    break  # Exit the pressure adjustment loop

                pressure_guess_previous = pressure_guess
                pressure_guess -= pressure_diff * adjustment_factor
                pressure_guess = 0.5 * (pressure_guess + pressure_guess_previous) # Relaxation
                adjustment_factor *= 0.95  # Reduce adjustment factor

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

    # --- Save output data to a file ---
    if data_output_enabled:
        # Combine and save plotted data to a single output file
        output_data = np.column_stack((radii, density, gravity, pressure, temperature))
        header = "Radius (m)\tDensity (kg/m^3)\tGravity (m/s^2)\tPressure (Pa)\tTemperature (K)"
        np.savetxt("planet_profile.txt", output_data, header=header)

    # --- Plotting ---
    if plotting_enabled:
        plot_planet_profile_single(radii, density, gravity, pressure, temperature, cmb_radius, average_density) # Plot planet profile 
        eos_data_files = ['eos_seager07_iron.txt', 'eos_seager07_silicate.txt', 'eos_seager07_water.txt']  # Example files (adjust the filenames accordingly)
        eos_data_folder = "../../data/"  # Path to the folder where EOS data is stored
        plot_eos_material(eos_data_files, eos_data_folder)  # Call the EOS plotting function
        #plt.show()  # Show the plots

if __name__ == "__main__":
    main()
