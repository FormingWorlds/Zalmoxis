# Setup script for downloading and extracting data

import os, shutil, subprocess

def download_data():
    """
    Download and extract data from osf.io if the folder does not already exist.
    The data is downloaded as a zip file and extracted to a specified folder.
    The script also removes any __MACOSX folders and moves the contents of the inner 'data' folder to the outer 'data' folder.
    If the folder already exists, it skips the download and extraction process.
    It also deletes the contents of the calculated_planet_mass_radius.txt file if it exists.
    """
    # Set the working directory to the current file
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Define URL, token, and paths
    download_url = "https://osf.io/download/tf9u5/"
    download_path = '../../data.zip'
    extract_folder = '../../data'

    # Check if the folder already exists
    if not os.path.exists(extract_folder):
        # Download and extract in one go
        subprocess.run(f"curl -L -o {download_path} {download_url}", shell=True, check=True)
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
