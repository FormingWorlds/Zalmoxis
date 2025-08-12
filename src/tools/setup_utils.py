# Setup script for downloading and extracting data
from __future__ import annotations

import logging
import os
from pathlib import Path

from osfclient.api import OSF

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

# Set up logging
logger = logging.getLogger(__name__)

def get_osf(id: str):
    """
    Get the OSF storage for a given project ID.
    Inputs:
        - id: OSF project ID
    Returns:
        - OSF storage object for the project
    """
    osf = OSF()
    project = osf.project(id)
    return project.storage('osfstorage')

def download_OSF_folder(*, storage, folders: list[str], data_dir: Path):
    """
    Download a specific folder in the OSF repository

    Inputs :
        - storage : OSF storage name
        - folders : folder names to download
        - data_dir : local repository where data are saved
    """
    for file in storage.files:
        for folder in folders:
            if not file.path[1:].startswith(folder):
                continue
            parts = file.path.split('/')[1:]
            target = Path(data_dir, *parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f'Downloading {file.path}...')
            with open(target, 'wb') as f:
                file.write_to(f)
            break

def download(folder: str, osf_id: str, data_dir: Path):
    """    Download a folder from OSF and extract it to the specified data directory.
    Inputs:
        - folder: Name of the folder to download
        - osf_id: OSF project ID
        - data_dir: Directory where the data will be saved
    """
    # Get the target path for the folder
    target_path = data_dir / folder

    if target_path.exists():
        logger.info(f"Folder '{folder}' already exists in '{data_dir}'. Skipping download.")
        return

    logger.info(f"Folder '{folder}' does not exist in '{data_dir}'. Proceeding with download.")
    logger.info(f"Downloading folder '{folder}' from OSF project '{osf_id}' to '{data_dir}'...")

    storage = get_osf(osf_id)
    download_OSF_folder(storage=storage, folders=[folder], data_dir=data_dir)

    logger.info(f"Download of '{folder}' complete.")

def create_output_files():
    """
    Create output files directory if it does not exist.
    This directory will store the results of the calculations.
    """
    output_dir = os.path.join(ZALMOXIS_ROOT, "output_files")  # Path to output files directory

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Output files directory created at '{output_dir}'.")
    else:
        logger.info(f"Output files directory already exists at '{output_dir}'.")

def download_data():
    """
    Download and extract data for Zalmoxis.
    This includes downloading the EOS data, radial profiles, and mass-radius curves.
    """
    # Download the necessary data from OSF
    download(folder='EOS_Seager2007', osf_id='dpkjb', data_dir=Path(ZALMOXIS_ROOT, "data"))
    download(folder='radial_profiles', osf_id='dpkjb', data_dir=Path(ZALMOXIS_ROOT, "data"))
    download(folder='mass_radius_curves', osf_id='dpkjb', data_dir=Path(ZALMOXIS_ROOT, "data"))

if __name__ == "__main__":
    logger.info("Starting data download...")
    download_data() # Download and extract data for Zalmoxis
    create_output_files() # Create output files directory
    logger.info("Setup completed successfully!")
