import os, sys, time, math, toml
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from .constants import *
from .eos_functions import calculate_density, calculate_temperature, birch_murnaghan, mie_gruneisen_debye
from .eos_properties import material_properties
from .structure_model import coupled_odes
from .plots.plot_profiles import plot_planet_profile_single
from .plots.plot_eos import plot_eos_material
from .setup import download_data

# Run file via command line with default configuration file: python -m src.zalmoxis.zalmoxis -c ../../input/default.toml

# Function to choose the configuration file to run the main function
def choose_config_file(temp_config_path=None):
    """
    Function to choose the configuration file to run the main function.
    The function will first check if a temporary configuration file is provided.
    If not, it will check if the -c flag is provided in the command line arguments.
    If the -c flag is provided, the function will read the configuration file path from the next argument.
    If no temporary configuration file or -c flag is provided, the function will read the default configuration file.
    """
    # Set the working directory to the current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Load the configuration file either from terminal (-c flag) or default path
    if temp_config_path:
        try:
            config = toml.load(temp_config_path)
            print(f"Reading temporary config file from: {temp_config_path}")
        except FileNotFoundError:
            print(f"Error: Temporary config file not found at {temp_config_path}")
            sys.exit(1)
    elif "-c" in sys.argv:
        index = sys.argv.index("-c")
        try:
            config_file_path = sys.argv[index + 1]
            config = toml.load(config_file_path)
            print(f"Reading config file from: {config_file_path}")
        except IndexError:
            print("Error: -c flag provided but no config file path specified.")
            sys.exit(1)  # Exit with error code
        except FileNotFoundError:
            print(f"Error: Config file not found at {config_file_path}")
            sys.exit(1)
    else:
        config_default_path = "../../input/default.toml"
        try:
            config = toml.load(config_default_path)
            print(f"Reading default config file from {config_default_path}")
        except FileNotFoundError:
            print(f"Error: Default config file not found at {config_default_path}")
            sys.exit(1)

    return config

def main(temp_config_path=None, id_mass=None):
    
    """
    Main function to run the exoplanet internal structure model.

    This function reads the configuration file, initializes parameters, and performs
    an iterative solution to calculate the internal structure of an exoplanet based on
    the given mass and other parameters. It outputs the calculated planet radius, core
    radius, densities, pressures, and temperatures at various layers, and optionally
    saves the data to a file and plots the results.
    """
    # Download data if not already present
    download_data()

    config = choose_config_file(temp_config_path)  # Choose the configuration file
    
    # Access parameters from the configuration file
    planet_mass = config['InputParameter']['planet_mass']  # Mass of the planet (kg)
    core_mass_fraction = config['AssumptionsAndInitialGuesses']['core_mass_fraction']  # Initial guess for the core mass as a fraction of the total mass
    weight_iron_fraction = config['AssumptionsAndInitialGuesses']['weight_iron_fraction']  # Initial guess for the weight fraction of iron in the core
    EOS_CHOICE = config['EOS']['choice']  # Choice of equation of state (e.g., "Birch-Murnaghan", "Mie-Gruneisen-Debye", "Tabulated")
    num_layers = config['Calculations']['num_layers']  # Number of radial layers for calculations

    # Parameters for the iterative solution process
    max_iterations_outer = config['IterativeProcess']['max_iterations_outer']  # Maximum iterations for the outer loop (radius and CMB adjustment)
    tolerance_outer = config['IterativeProcess']['tolerance_outer']  # Convergence tolerance for the outer loop
    tolerance_radius = config['IterativeProcess']['tolerance_radius']  # Convergence tolerance for the core radius
    max_iterations_inner = config['IterativeProcess']['max_iterations_inner']  # Maximum iterations for the inner loop (density profile)
    tolerance_inner = config['IterativeProcess']['tolerance_inner']  # Convergence tolerance for the inner loop
    relative_tolerance = config['IterativeProcess']['relative_tolerance']  # Relative tolerance for integration in solve_ivp
    absolute_tolerance = config['IterativeProcess']['absolute_tolerance']  # Absolute tolerance for integration in solve_ivp

    # Parameters for adjusting the surface pressure to the target value
    target_surface_pressure = config['PressureAdjustment']['target_surface_pressure']  # Target surface pressure (Pa)
    pressure_tolerance = config['PressureAdjustment']['pressure_tolerance']  # Tolerance for surface pressure convergence
    max_iterations_pressure = config['PressureAdjustment']['max_iterations_pressure']  # Maximum iterations for pressure adjustment
    pressure_adjustment_factor = config['PressureAdjustment']['pressure_adjustment_factor'] # Adjustment factor for updating the central pressure guess

    # Output control parameters
    data_output_enabled = config['Output']['data_enabled']  # Flag to enable saving data to a file (True/False)
    plotting_enabled = config['Output']['plots_enabled']  # Flag to enable plotting the results (True/False)

    # Setup initial guesses for the planet radius and core-mantle boundary radius
    radius_guess = 1000*(7030-1840*weight_iron_fraction)*(planet_mass/earth_mass)**0.282 # Initial guess for the interior planet radius [m] based on the scaling law in Noack et al. 2020
    cmb_radius = 0 # Initial guess for the core-mantle boundary radius [m]
    cmb_radius_previous = cmb_radius # Initial guess for the previous core-mantle boundary radius [m]
    
    # Initialize temperature profile
    temperature = np.zeros(num_layers)

    # Solve the interior structure
    for outer_iter in range(max_iterations_outer): # Outer loop for radius and mass convergence
        # Start timing the outer loop
        start_time = time.time()
        # Setup initial guess for the radial grid based on the radius guess
        radii = np.linspace(0, radius_guess, num_layers)

        # Initialize arrays for mass, gravity, density, and pressure grids
        density = np.zeros(num_layers)
        mass_enclosed = np.zeros(num_layers)
        gravity = np.zeros(num_layers)
        pressure = np.zeros(num_layers)

        # Setup initial guess for the core-mantle boundary mass
        cmb_mass = core_mass_fraction * planet_mass

        # Setup initial guess for the pressure at the center of the planet (needed for solving the ODEs)
        pressure[0] = earth_center_pressure
        
        # Setup initial guess for the density grid
        for i in range(num_layers):
            if radii[i] < cmb_radius:
                # Core (simplified initial guess)
                density[i] = material_properties["core"]["rho0"]
            else:
                # Mantle (simplified initial guess)
                density[i] = material_properties["mantle"]["rho0"]

        for inner_iter in range(max_iterations_inner): # Inner loop for density adjustment
            old_density = density.copy() # Store the old density for convergence check

            # Initialize empty cache for interpolation functions for density calculations
            interpolation_cache = {}  

            # Setup initial pressure guess at the center of the planet based on empirical scaling law derived from the hydrostatic equilibrium equation
            pressure_guess = earth_center_pressure * (planet_mass/earth_mass)**2 * (radius_guess/earth_radius)**(-4) 

            for pressure_iter in range(max_iterations_pressure): # Innermost loop for pressure adjustment

                # Setup initial conditions for the mass, gravity, and pressure at the center of the planet
                y0 = [0, 0, pressure_guess] 

                # Solve the ODEs using solve_ivp
                sol = solve_ivp(lambda r, y: coupled_odes(r, y, cmb_mass, EOS_CHOICE, interpolation_cache), 
                    (radii[0], radii[-1]), y0, t_eval=radii, rtol=relative_tolerance, atol=absolute_tolerance, method='RK45', dense_output=True)

                # Extract mass, gravity, and pressure grids from the solution
                mass_enclosed = sol.y[0]
                gravity = sol.y[1]
                pressure = sol.y[2]

                # Extract the calculated surface pressure from the last element of the pressure array
                surface_pressure = pressure[-1]

                # Calculate the pressure difference between the calculated surface pressure and the target surface pressure
                pressure_diff = surface_pressure - target_surface_pressure

                # Check for convergence of the surface pressure and overall pressure positivity
                if abs(pressure_diff) < pressure_tolerance and np.all(pressure > 0):
                    print(f"Surface pressure converged after {pressure_iter + 1} iterations and all pressures are positive.")
                    break  # Exit the pressure adjustment loop
        
                # Update the pressure guess at the center of the planet based on the pressure difference at the surface using an adjustment factor
                pressure_guess -= pressure_diff * pressure_adjustment_factor

            # Update density grid based on the calculated pressure and mass enclosed
            for i in range(num_layers):
                # Define the material type based on the calculated enclosed mass up to the core-mantle boundary
                if mass_enclosed[i] < cmb_mass:
                    # Core
                    material = "core"
                else:
                    # Mantle
                    material = "mantle"

                # Calculate the new density using the equation of state
                new_density = calculate_density(pressure[i], material, EOS_CHOICE)

                # Handle potential errors in density calculation
                if new_density is None:
                    print(f"Warning: Density calculation failed at radius {radii[i]}. Using previous density.")
                    new_density = old_density[i]

                # Update the density grid with a weighted average of the new and old density
                density[i] = 0.5 * (new_density + old_density[i]) 

            # Check for convergence of density using relative difference between old and new density
            relative_diff_inner = np.max(np.abs((density - old_density) / (old_density + 1e-20)))
            if relative_diff_inner < tolerance_inner:
                print(f"Inner loop converged after {inner_iter + 1} iterations.")
                break # Exit the inner loop 

        # Extract the calculated total interior mass of the planet from the last element of the mass array
        calculated_mass = mass_enclosed[-1]

        # Update the total interior radius by scaling the initial guess based on the calculated mass
        radius_guess = radius_guess * (planet_mass / calculated_mass)**(1/3)

        # Update the core-mantle boundary radius based on the calculated mass grid
        cmb_radius = radii[np.argmax(mass_enclosed >= cmb_mass)]
        
        # Update the core-mantle boundary mass based on the core mass fraction and calculated total interior mass of the planet
        cmb_mass = core_mass_fraction * calculated_mass

        # Calculate relative differences of the calculated total interior mass and core-mantle boundary radius
        relative_diff_outer = abs((calculated_mass - planet_mass) / planet_mass)
        relative_diff_radius = abs((cmb_radius - cmb_radius_previous) / cmb_radius)

        # Check for convergence of the calculated total interior mass and core-mantle boundary radius of the planet
        if relative_diff_outer < tolerance_outer and relative_diff_radius < tolerance_radius:
            print(f"Outer loop (cmb radius and total mass) converged after {outer_iter + 1} iterations.")
            break  # Exit the outer loop
        
        # Update previous core-mantle boundary radius for the next iteration
        cmb_radius_previous = cmb_radius

        # End timing the outer loop
        end_time = time.time()
        print(f"Outer iteration {outer_iter+1} took {end_time - start_time:.2f} seconds")

        # Check if maximum iterations for outer loop are reached
        if outer_iter == max_iterations_outer - 1:
            print(f"Warning: Maximum outer iterations ({max_iterations_outer}) reached. Radius and cmb may not be fully converged.")

    # Extract the final calculated total interior radius of the planet 
    planet_radius = radius_guess

    # Extract the index of the core-mantle boundary mass in the mass array
    cmb_index = np.argmax(mass_enclosed >= cmb_mass)

    # Calculate the average density of the planet using the calculated mass and radius
    average_density = calculated_mass / (4/3 * math.pi * planet_radius**3)

    # Calculate the temperature profile 
    temperature = calculate_temperature(radii, cmb_radius, 300, material_properties, gravity, density, material_properties["mantle"]["K0"], dr=planet_radius/num_layers)

    print("Exoplanet Internal Structure Model (Mass Only Input)")
    print("----------------------------------------------------------------------")
    print(f"Calculated Planet Mass: {calculated_mass:.2e} kg")
    print(f"Calculated Planet Radius: {planet_radius:.2e} m")
    print(f"Core Radius: {cmb_radius:.2e} m")
    print(f"Mantle Density (at CMB): {density[cmb_index]:.2f} kg/m^3")
    print(f"Core Density (at CMB): {density[cmb_index - 1]:.2f} kg/m^3")
    print(f"Pressure at Core-Mantle Boundary (CMB): {pressure[cmb_index]:.2e} Pa")
    print(f"Pressure at Center: {pressure[0]:.2e} Pa")
    print(f"Average Density: {average_density:.2f} kg/m^3")
    print(f"CMB Mass Fraction: {cmb_mass / calculated_mass:.2f}")
    print(f"Calculated Core Radius Fraction: {cmb_radius / planet_radius:.2f}")

    # --- Save output data to a file ---
    if data_output_enabled:
        # Create output directory if it does not exist
        if not os.path.exists("output_files"):
            os.makedirs("output_files")
        # Combine and save plotted data to a single output file
        output_data = np.column_stack((radii, density, gravity, pressure, temperature, mass_enclosed))
        header = "Radius (m)\tDensity (kg/m^3)\tGravity (m/s^2)\tPressure (Pa)\tTemperature (K)\tMass Enclosed (kg)"
        np.savetxt(f"output_files/planet_profile{id_mass}.txt", output_data, header=header)
        # Append calculated mass and radius of the planet to a file in dedicated columns
        output_file = "output_files/calculated_planet_mass_radius.txt"
        if not os.path.exists(output_file):
            header = "Calculated Mass (kg)\tCalculated Radius (m)"
            with open(output_file, "w") as file:
                file.write(header + "\n")
        with open(output_file, "a") as file:
            file.write(f"{calculated_mass}\t{planet_radius}\n")
         

    # --- Plotting ---
    if plotting_enabled:
        #plot_planet_profile_single(radii, density, gravity, pressure, temperature, cmb_radius, cmb_mass, average_density, mass_enclosed, id_mass) # Plot planet profile for a single planet
        eos_data_files = ['eos_seager07_iron.txt', 'eos_seager07_silicate.txt', 'eos_seager07_water.txt']  # Example files (adjust the filenames accordingly)
        eos_data_folder = "../../data/"  # Path to the folder where EOS data is stored
        plot_eos_material(eos_data_files, eos_data_folder)  # Call the EOS plotting function
        #plt.show()  # Show the plots
    
