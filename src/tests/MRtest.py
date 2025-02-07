import os, sys
import tempfile
import toml
import subprocess
from src.jord import jord  
from src.jord.plots.plot_MR import plot_mass_radius_relationship
from src.jord.plots.plot_profiles_all_in_one import plot_profiles_all_in_one    

# Run file via command line: python -m src.tests.MRtest

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
        temp_config_path = temp_config_file.name

    # Run the main function with the temporary configuration file
    jord.main(temp_config_path, id_mass)

    # Clean up the temporary configuration file after running
    os.remove(temp_config_path)

"""# Run jord for a range of planet masses (1 to 10 Earth masses in steps of 1)
for id_mass in range(1, 11):
    run_jord(id_mass)

# Plot the mass-radius relationship for 1 to 10 Earth masses in steps of 1
plot_mass_radius_relationship()

# Call the function to plot the profiles of all planets in one plot for 1 to 10 Earth masses in steps of 1
plot_profiles_all_in_one()"""

# Run jord for a range of planet masses (1 to 15 Earth masses in steps of 2.5)
for id_mass in [1, 2.5, 5, 7.5, 10, 12.5, 15]:
    run_jord(id_mass)

# Plot the mass-radius relationship for 1 to 15 Earth masses in steps of 2.5
plot_mass_radius_relationship()

# Call the function to plot the profiles of all planets in one plot for 1 to 15 Earth masses in steps of 2.5
plot_profiles_all_in_one()
