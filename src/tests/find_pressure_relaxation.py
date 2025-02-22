# script to find the ideal pressure relaxation value for a target planet mass. 
# The function should return a dictionary which stores the planet mass as the key and the corresponding pressure relaxation value as the value.
# The function should also plot a graph of the pressure relaxation values against the planet mass (why not).
# The ideal pressure relaxation for a target mass is found by running the zalmoxis.py script with a beginning guess value knowing that the paramter space to be tested for this relaxation is between 0.500000 and 0.510000.
# After running MRtest.py, this script should check for the convergence of the calculated pressure values of the planet profiles (saved in output_files folder), specifically by checking whether all pressure values are positive.
# If the pressure values are positive, the script should return the pressure relaxation value as the ideal value for the target mass.
# If the pressure values are negative, the script should adjust the pressure relaxation value and rerun MRtest.py until the pressure values are positive. This adjustment should be done using the bisection method and to a precision of 6 decimals.
# The script should return the pressure relaxation value as the ideal value for the target mass.
# The script should also plot a graph of the pressure relaxation values against the planet mass.
# The script should be able to run in parallel for multiple target masses stored in an array (use the ProcessPoolExecutor class from concurrent.futures module).
# The precision of this relaxation values is 6 decimal places (make this adjustable in the future).

import os
import sys
import time
import shutil
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from src.zalmoxis import zalmoxis  
import toml
import tempfile

# Modified run_zalmoxis by adding pressure_relaxation as parameter


# Function to run the main function with a temporary configuration file
def run_zalmoxis2(id_mass=None, pressure_relaxation=None):
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
    config['IterativeProcess']['max_iterations_outer'] = 20
    config['IterativeProcess']['tolerance_outer'] = 1e-3
    config['IterativeProcess']['tolerance_radius'] = 1e-3
    config['IterativeProcess']['tolerance_inner'] = 1e-4
    config['IterativeProcess']['relative_tolerance'] = 1e-5
    config['IterativeProcess']['absolute_tolerance'] = 1e-6 
    config['PressureAdjustment']['target_surface_pressure'] = 101325 
    config['PressureAdjustment']['pressure_tolerance'] = 1000
    config['PressureAdjustment']['max_iterations_pressure'] = 100 
    config['PressureAdjustment']['pressure_relaxation'] = pressure_relaxation 
    config['PressureAdjustment']['pressure_adjustment_factor'] = 0.95 

    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.toml') as temp_config_file:
        toml.dump(config, temp_config_file)  # Dump the updated config into the file
        temp_config_path = temp_config_file.name  # Get the path to the temporary file

    # Run the main function with the temporary configuration file
    zalmoxis.main(temp_config_path, id_mass)

    # Clean up the temporary configuration file after running
    os.remove(temp_config_path)


def find_ideal_pressure_relaxation(target_masses):
    # Dictionary mapping id_mass to pressure
    pressure_relaxation_dict = {}
    # Loop through each target mass
    for id_mass in target_masses:
        # Set the initial bounds for the bisection method
        lower_bound = 0.500000
        upper_bound = 0.510000
        # Set the initial guess for the pressure relaxation
        pressure_relaxation = (lower_bound + upper_bound) / 2
        # Run the main function with the initial guess
        run_zalmoxis2(id_mass, pressure_relaxation)
        # Load the pressure values from the output file
        output_path = f'../zalmoxis/output_files/planet_profile{id_mass}.txt'
        pressure_values = np.loadtxt(output_path, usecols=(3,))
        # Check if the pressure values are positive
        if np.all(pressure_values > 0):
            # If all pressure values are positive, set the pressure relaxation as the ideal value
            pressure_relaxation_dict[id_mass] = pressure_relaxation
        else:
            
    
    
    


