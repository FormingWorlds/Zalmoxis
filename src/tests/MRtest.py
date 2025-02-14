import os, sys
import tempfile
import toml
import subprocess
from src.zalmoxis import zalmoxis  
from src.zalmoxis.plots.plot_MR import plot_mass_radius_relationship
from src.zalmoxis.plots.plot_profiles_all_in_one import plot_profiles_all_in_one    
from concurrent.futures import ProcessPoolExecutor
import time
import subprocess
import shutil

# Run file via command line: python -m src.tests.MRtest Wagner/Boujibar/default/Seager/custom

# Function to run the main function with a temporary configuration file
def run_zalmoxis(id_mass=None):
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
    config['PressureAdjustment']['target_surface_pressure'] = 101325 # experiment with this, default is 101325
    config['PressureAdjustment']['pressure_tolerance'] = 1000 # experiment with this, default is 1000
    config['PressureAdjustment']['max_iterations_pressure'] = 100 # don't change for now, default is 100
    config['PressureAdjustment']['pressure_relaxation'] = 0.501290 # experiment with this, default is 0.5
    config['PressureAdjustment']['pressure_adjustment_factor'] = 0.95 # experiment with this, default is 0.95

    # Create a temporary configuration file
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.toml') as temp_config_file:
        toml.dump(config, temp_config_file)  # Dump the updated config into the file
        temp_config_path = temp_config_file.name  # Get the path to the temporary file

    # Run the main function with the temporary configuration file
    zalmoxis.main(temp_config_path, id_mass)

    # Clean up the temporary configuration file after running
    os.remove(temp_config_path)

def MRtest(choice):
    '''
        This function sets the working directory to the current file's directory,
        checks if the data folder exists and downloads/extracts it if not,
        deletes the contents of the calculated_planet_mass_radius.txt file if it exists,
        runs zalmoxis in parallel for a range of planet masses based on the provided choice,
        plots the mass-radius relationship, and calls the function to plot the profiles
        of all planets in one plot.

        Parameters:
        choice (str): Choice of comparison data. Options are 'Wagner', 'Boujibar', 'default', 'Seager', or 'custom'.

        Raises:
        ValueError: If an invalid choice is provided for the comparison data.

        Returns: 
        None
    '''

    # Set the working directory to the current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Define URL, token, and paths
    download_url = "https://files.de-1.osf.io/v1/resources/dpkjb/providers/osfstorage/67aa034e023777550617fbad"
    access_token = "wARvp1OBy1fOvmHruw63XP4ALtBecGDrnbskSqdhODRlZlQ8YBrAzmr9GGD9uxk4DzzmoV"
    download_path = '../../data.zip'
    extract_folder = '../../data'

    # Check if the folder already exists
    if not os.path.exists(extract_folder):
        # Download and extract in one go
        subprocess.run(f"curl -L -H 'Authorization: Bearer {access_token}' -o {download_path} {download_url}", shell=True, check=True)
        os.makedirs(extract_folder, exist_ok=True)
        subprocess.run(f"unzip {download_path} -d {extract_folder}", shell=True, check=True)

        print(f"Download and extraction complete! Files are in '{extract_folder}'.")

        # Remove the __MACOSX folder if it exists
        macosx_folder = os.path.join(extract_folder, '__MACOSX')
        if os.path.exists(macosx_folder):
            shutil.rmtree(macosx_folder)

        # Move the contents of the inner 'data' folder to the outer 'data' folder
        inner_data_folder = os.path.join(extract_folder, 'data')  # Path to inner 'data' folder
        outer_data_folder = os.path.join(extract_folder)  # Path to outer 'data' folder

        if os.path.exists(inner_data_folder):
            for item in os.listdir(inner_data_folder):
                # Move each item from inner data to outer data
                s = os.path.join(inner_data_folder, item)
                d = os.path.join(outer_data_folder, item)
                if os.path.isdir(s):
                    shutil.move(s, d)
                else:
                    shutil.move(s, d)

        # After moving the contents, remove the inner 'data' folder
        shutil.rmtree(inner_data_folder)

        # Remove the leftover 'data.zip' and 'data_folder' after extraction
        os.remove(download_path)
    else:
        print(f"Folder '{extract_folder}' already exists. Skipping download and extraction.")
    
    # Delete the contents of the calculated_planet_mass_radius.txt file if it exists
    calculated_file_path = '../zalmoxis/output_files/calculated_planet_mass_radius.txt'
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
    elif choice == "Seager":
        target_mass_array = [1, 5, 10, 50]
    elif choice == "custom":
        target_mass_array = range(50, 51)
    else:
        raise ValueError("Invalid choice. Please select 'Wagner', 'Boujibar', or 'default'.")

    # Run zalmoxis in parallel for a range of planet masses
    with ProcessPoolExecutor() as executor:
        executor.map(run_zalmoxis, target_mass_array)

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
