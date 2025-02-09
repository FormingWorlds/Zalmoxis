import os, sys
import tempfile
import toml
import subprocess
from src.jord import jord  
from src.jord.plots.plot_MR import plot_mass_radius_relationship
from src.jord.plots.plot_profiles_all_in_one import plot_profiles_all_in_one    
from concurrent.futures import ProcessPoolExecutor
import time

# Run file via command line: python -m src.tests.MRtest Wagner or python -m src.tests.MRtest Boujibar


# Function to run the main function with a temporary configuration file
def run_jord(id_mass=None):
    '''
        This function sets the working directory to the current file's directory,
        loads a default configuration file, modifies specific configuration parameters,
        creates a temporary configuration file, runs the main function with the temporary
        configuration file, and then cleans up the temporary configuration file.

        Parameters:
        id_mass (float, optional): The mass of the planet in Earth masses. Defaults to None.

        Raises:
        FileNotFoundError: If the default configuration file is not found.
        toml.TomlDecodeError: If there is an error decoding the TOML configuration file.
        Exception: If there is an error during the execution of the main function.

        Returns:
        None
    '''
    # Set the working directory to the current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Path to the default configuration file
    default_config_path = '../../input/default.toml'

    # Load the default configuration
    with open(default_config_path, 'r') as file:
        config = toml.load(file)

    # Modify the configuration parameters as needed
    config['InputParameter']['planet_mass'] = id_mass*5.972e24
    config['Calculations']['num_layers'] = 150
    config['IterativeProcess']['tolerance_outer'] = 1e-3
    config['IterativeProcess']['tolerance_inner'] = 1e-4
    config['IterativeProcess']['tolerance_radius'] = 1e-3
    config['IterativeProcess']['max_iterations_outer'] = 20
    config['IterativeProcess']['relative_tolerance'] = 1e-5
    config['IterativeProcess']['absolute_tolerance'] = 1e-6 

    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.toml') as temp_config_file:
        toml.dump(config, temp_config_file)  # Dump the updated config into the file
        temp_config_path = temp_config_file.name  # Get the path to the temporary file

    # Run the main function with the temporary configuration file
    jord.main(temp_config_path, id_mass)

    # Clean up the temporary configuration file after running
    os.remove(temp_config_path)

def MRtest(choice):
    '''
        This function sets the working directory to the current file's directory,
        deletes the contents of the calculated_planet_mass_radius.txt file if it exists,
        runs jord in parallel for a range of planet masses, plots the mass-radius relationship,
        and calls the function to plot the profiles of all planets in one plot.

        Parameters:
        choice (str): Choice of comparison data. Options are 'Wagner', 'Boujibar', or 'default'.

        Raises:
        ValueError: If an invalid choice is provided for the comparison data.

        Returns: 
        None
    '''

    # Set the working directory to the current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Delete the contents of the calculated_planet_mass_radius.txt file if it exists
    calculated_file_path = '../jord/calculated_planet_mass_radius.txt'
    if os.path.exists(calculated_file_path):
        with open(calculated_file_path, 'w') as file:
            file.truncate(0)
            header = "Calculated Mass (kg)\tCalculated Radius (m)"
            file.write(header + "\n")    

    if choice == "Wagner":
        target_mass_array = [1, 2.5, 5, 7.5, 10, 12.5, 15]
    elif choice == "Boujibar":
        target_mass_array = range(1, 11)
    elif choice == "default":
        print("No choice selected for the comparison, defaulting to 1 to 10 Earth masses simulation.")
        target_mass_array = range(1, 11)
    else:
        raise ValueError("Invalid choice. Please select 'Wagner', 'Boujibar', or 'default'.")

    # Run jord in parallel for a range of planet masses
    with ProcessPoolExecutor() as executor:
        executor.map(run_jord, target_mass_array)

    # Plot the mass-radius relationship 
    plot_mass_radius_relationship(target_mass_array)

    # Call the function to plot the profiles of all planets in one plot 
    plot_profiles_all_in_one(target_mass_array, choice)

if __name__ == "__main__":
    start_time = time.time()
    
    if len(sys.argv) != 2:
        print("Usage: python -m src.tests.MRtest <choice>")
        sys.exit(1)
    
    choice = sys.argv[1]
    MRtest(choice)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total running time: {total_time:.2f} seconds")
