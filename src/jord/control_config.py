import os, sys
import tempfile
import toml
import subprocess
from src.jord import jord  

# Run file via command line: python -m src.jord.control_config

# Function to run the main function with a temporary configuration file
def run_main_with_temp_config(id_mass=None):
    # Set the working directory to the current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Path to the default configuration file
    default_config_path = '../../input/default.toml'

    # Load the default configuration
    with open(default_config_path, 'r') as file:
        config = toml.load(file)

    # Modify the configuration parameters as needed
    config['InputParameter']['planet_mass'] = id_mass*5.972e24
    config['Calculations']['num_layers'] = 10000
    config['IterativeProcess']['tolerance_outer'] = 1e-3
    config['IterativeProcess']['tolerance_inner'] = 1e-4
    config['IterativeProcess']['tolerance_radius'] = 1e-3
    config['IterativeProcess']['max_iterations_outer'] = 20

    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.toml') as temp_config_file:
        toml.dump(config, temp_config_file)  # Dump the updated config into the file
        temp_config_path = temp_config_file.name

    # Run the main function with the temporary configuration file
    jord.main(temp_config_path, id_mass)

    # Clean up the temporary configuration file after running
    os.remove(temp_config_path)

# Run the function for a range of planet masses (1 to 10 Earth masses)
for id_mass in range(1, 11):
    run_main_with_temp_config(id_mass)
