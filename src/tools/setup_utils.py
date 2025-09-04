# Setup script for downloading and extracting data
from __future__ import annotations

import logging
import os
import shutil
import subprocess as sp
import tempfile
from pathlib import Path

from osfclient.api import OSF
from tqdm import tqdm

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

def download_zenodo_folder(zenodo_id: str, folder_dir: Path, keep_files: list[str] = None):
    """
    Download a specific Zenodo record into a specified folder and optionally keep only selected files.

    Parameters
    ----------
    zenodo_id : str
        Zenodo record ID to download.
    folder_dir : Path
        Local directory where the Zenodo record (or selected files) will be saved.
    keep_files : list[str] or None, optional
        - None (default): Keep all files from the record.
        - List of filenames: Only keep those files from the downloaded record.
    """
    # Clear target folder
    shutil.rmtree(folder_dir, ignore_errors=True)
    folder_dir.mkdir(parents=True, exist_ok=True)

    # Temporary folder for downloading
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        # Download record into temporary folder
        cmd = ["zenodo_get", str(zenodo_id), "-o", str(tmp_path)]
        process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)

        # Use tqdm to show progress per file downloaded
        with tqdm(desc=f"Downloading Zenodo {zenodo_id}", unit="file") as pbar:
            for line in process.stdout:
                # Update progress bar for each file
                if "Downloading" in line:
                    pbar.update(1)
            process.wait()

        if process.returncode != 0:
            raise RuntimeError(f"zenodo_get failed with exit code {process.returncode}")

        # Copy files based on keep_files filter
        if keep_files is None:
            logger.info("No file filter specified, keeping all files.")
            for item in tmp_path.iterdir():
                shutil.move(str(item), folder_dir)
        else:
            logger.info(f"Keeping only specified files: {keep_files}")
            for fname in keep_files:
                src_file = tmp_path / fname
                if src_file.exists():
                    shutil.move(str(src_file), folder_dir)
                else:
                    logger.warning(f"Requested file '{fname}' not found in Zenodo record {zenodo_id}.")

def download(folder: str, zenodo_id: str, osf_id: str, data_dir: Path, keep_files: list[str] = None):
    """    Download a folder from Zenodo or OSF and save it to the specified data directory.
    Parameters:
        folder (str): Name of the folder to download.
        zenodo_id (str): Zenodo record ID to download from.
        osf_id (str): OSF project ID to download from.
        data_dir (Path): Local directory where the folder will be saved.
    Raises:
        RuntimeError: If the folder cannot be downloaded from both Zenodo and OSF.
    """
    # Get the target path for the folder
    folder_dir = data_dir / folder

    if folder_dir.exists():
        logger.info(f"Folder '{folder}' already exists in '{data_dir}'. Skipping download.")
        return

    logger.info(f"Folder '{folder}' does not exist in '{data_dir}'. Proceeding with download.")
    logger.info(f"Downloading folder '{folder}' from OSF project '{osf_id}' to '{data_dir}'...")

    # Try with Zenodo first
    try:
        logger.info(f"Downloading from Zenodo record '{zenodo_id}'...")
        download_zenodo_folder(zenodo_id=zenodo_id, folder_dir=folder_dir, keep_files=keep_files)
    except Exception as e:
        logger.error(f"Failed to download from Zenodo: {e}")
        logger.info("Trying to download from OSF...")

        # If Zenodo fails, try with OSF
        try:
            logger.info(f"Downloading from OSF project '{osf_id}'...")
            download_OSF_folder(storage=get_osf(osf_id), folders=[folder], data_dir=data_dir)
            logger.info(f"Download of '{folder}' complete.")
        except Exception as e:
            logger.error(f"Failed to download from OSF: {e}")
            raise RuntimeError(f"Failed to download folder '{folder}' from both Zenodo and OSF.")

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
    # Download the necessary data folders
    download(folder='EOS_Seager2007', zenodo_id=15727998, osf_id='dpkjb', data_dir=Path(ZALMOXIS_ROOT, "data"))
    download(folder='radial_profiles', zenodo_id=16837954, osf_id='dpkjb', data_dir=Path(ZALMOXIS_ROOT, "data"))
    download(folder='mass_radius_curves', zenodo_id=15727899, osf_id='dpkjb', data_dir=Path(ZALMOXIS_ROOT, "data"), keep_files=['massradiusEarthlikeRocky.txt', 'massradius_50percentH2O_300K_1mbar.txt'])

if __name__ == "__main__":
    logger.info("Starting data download...")
    download_data() # Download and extract data for Zalmoxis
    create_output_files() # Create output files directory
    logger.info("Setup completed successfully!")
