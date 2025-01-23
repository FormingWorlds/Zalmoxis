import os
import tempfile
import toml
import subprocess
from src.jord import jord  

# Run file via command line: python -m src.jord.control_config

def run_with_temp_config():
    # Set the working directory to the current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Path to the default configuration file
    default_config_path = '../../input/default.toml'

    # Load the default configuration
    with open(default_config_path, 'r') as file:
        config = toml.load(file)

    # Modify the configuration parameters as needed
    config['InputParameter']['planet_mass'] = 5.972e25

    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.toml') as temp_config_file:
        toml.dump(config, temp_config_file)  # Dump the updated config into the file
        temp_config_path = temp_config_file.name

    # Run the main function with the temporary configuration file
    jord.main(temp_config_path)

    # Clean up the temporary configuration file after running
    os.remove(temp_config_path)

# Run the function
run_with_temp_config()
