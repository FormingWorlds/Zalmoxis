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

# Read the environment variable for get_zalmoxis_root()
from zalmoxis import get_zalmoxis_root

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
        cmd = ['zenodo_get', str(zenodo_id), '-o', str(tmp_path)]
        process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)

        # Use tqdm to show progress per file downloaded
        with tqdm(desc=f'Downloading Zenodo {zenodo_id}', unit='file') as pbar:
            for line in process.stdout:
                # Update progress bar for each file
                if 'Downloading' in line:
                    pbar.update(1)
            process.wait()

        if process.returncode != 0:
            raise RuntimeError(f'zenodo_get failed with exit code {process.returncode}')

        # Copy files based on keep_files filter
        if keep_files is None:
            logger.info('No file filter specified, keeping all files.')
            for item in tmp_path.iterdir():
                shutil.move(str(item), folder_dir)
        else:
            logger.info(f'Keeping only specified files: {keep_files}')
            for fname in keep_files:
                src_file = tmp_path / fname
                if src_file.exists():
                    shutil.move(str(src_file), folder_dir)
                else:
                    logger.warning(
                        f"Requested file '{fname}' not found in Zenodo record {zenodo_id}."
                    )


def download(
    folder: str,
    data_dir: Path,
    zenodo_id: str = None,
    osf_id: str = None,
    keep_files: list[str] = None,
):
    """Download a folder from Zenodo or OSF and save it to the specified data directory.
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
        download_zenodo_folder(
            zenodo_id=zenodo_id, folder_dir=folder_dir, keep_files=keep_files
        )
    except Exception as e:
        logger.error(f'Failed to download from Zenodo: {e}')
        logger.info('Trying to download from OSF...')

        # If Zenodo fails, try with OSF (if an OSF ID was provided)
        if osf_id is None:
            raise RuntimeError(
                f"Failed to download folder '{folder}' from Zenodo and no OSF fallback configured."
            )
        try:
            logger.info(f"Downloading from OSF project '{osf_id}'...")
            download_OSF_folder(storage=get_osf(osf_id), folders=[folder], data_dir=data_dir)
            logger.info(f"Download of '{folder}' complete.")
        except Exception as e:
            logger.error(f'Failed to download from OSF: {e}')
            raise RuntimeError(
                f"Failed to download folder '{folder}' from both Zenodo and OSF."
            )


def download_zenodo_tarball(zenodo_id: str, folder_dir: Path):
    """Download and extract a tarball from a Zenodo record.

    Some Zenodo records contain a single .tar.gz file instead of individual
    files. This function downloads the tarball and extracts it.

    Parameters
    ----------
    zenodo_id : str
        Zenodo record ID.
    folder_dir : Path
        Target directory. The tarball is expected to contain a single
        top-level directory whose contents will be placed here.
    """
    folder_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        cmd = ['zenodo_get', str(zenodo_id), '-o', str(tmp_path)]
        process = sp.Popen(cmd, stdout=sp.PIPE, stderr=sp.STDOUT, text=True)

        with tqdm(desc=f'Downloading Zenodo {zenodo_id}', unit='file') as pbar:
            for line in process.stdout:
                if 'Downloading' in line:
                    pbar.update(1)
            process.wait()

        if process.returncode != 0:
            raise RuntimeError(f'zenodo_get failed with exit code {process.returncode}')

        # Find and extract tarball
        import tarfile

        tarballs = list(tmp_path.glob('*.tar.gz'))
        if not tarballs:
            raise RuntimeError(f'No .tar.gz found in Zenodo record {zenodo_id}')

        with tarfile.open(tarballs[0], 'r:gz') as tar:
            tar.extractall(path=tmp_path, filter='data')

        # Move extracted directory contents to target, skipping macOS
        # resource forks (._* files) and __MACOSX directories
        extracted_dirs = [d for d in tmp_path.iterdir() if d.is_dir() and d.name != '__MACOSX']
        if extracted_dirs:
            src_dir = extracted_dirs[0]
            for item in src_dir.iterdir():
                if item.name.startswith('._') or item.name == '.DS_Store':
                    continue
                dest = folder_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), folder_dir)


def create_output():
    """
    Create output files directory if it does not exist.
    This directory will store the results of the calculations.
    """
    output_dir = os.path.join(get_zalmoxis_root(), 'output')  # Path to output files directory

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
    download(
        folder='EOS_Seager2007',
        data_dir=Path(get_zalmoxis_root(), 'data'),
        zenodo_id=15727998,
        osf_id='dpkjb',
    )
    download(
        folder='EOS_WolfBower2018_1TPa',
        data_dir=Path(get_zalmoxis_root(), 'data'),
        zenodo_id=17417017,
        keep_files=['density_melt.dat', 'density_solid.dat', 'adiabat_temp_grad_melt.dat'],
    )
    download(
        folder='radial_profiles',
        data_dir=Path(get_zalmoxis_root(), 'data'),
        zenodo_id=16837954,
        osf_id='dpkjb',
    )
    download(
        folder='mass_radius_curves',
        data_dir=Path(get_zalmoxis_root(), 'data'),
        zenodo_id=15727899,
        osf_id='dpkjb',
        keep_files=[
            'massradiusEarthlikeRocky.txt',
            'massradiusFe.txt',
            'massradiusmgsio3.txt',
            'massradius_50percentH2O_300K_1mbar.txt',
            'massradius_100percentH2O_300K_1mbar.txt',
        ],
    )
    download(
        folder='EOS_RTPress_melt_100TPa',
        data_dir=Path(get_zalmoxis_root(), 'data'),
        zenodo_id=18819027,
        keep_files=['density_melt.dat', 'adiabat_temp_grad_melt.dat'],
    )
    download(
        folder='melting_curves_Monteux-600',
        data_dir=Path(get_zalmoxis_root(), 'data'),
        zenodo_id=15728138,
        keep_files=['liquidus.dat', 'solidus.dat'],
    )
    # Zenodo 19680050: ecosystem-wide PALEOS MgSiO3 2-phase reference,
    # supersedes the prior 18924171 record. Ships two resolutions:
    # 150 pts/decade (default, ~80 MB) and 600 pts/decade (highres,
    # ~1.3 GB) for both solid and liquid tables. PROTEUS default is
    # 150 (matches PALEOS-2phase:MgSiO3 registry); 600 is selectable
    # via PALEOS-2phase:MgSiO3-highres for sensitivity tests.
    download(
        folder='EOS_PALEOS_MgSiO3',
        data_dir=Path(get_zalmoxis_root(), 'data'),
        zenodo_id=19680050,
        keep_files=[
            'paleos_mgsio3_tables_pt_proteus_solid.dat',
            'paleos_mgsio3_tables_pt_proteus_liquid.dat',
            'paleos_mgsio3_tables_pt_proteus_solid_highres.dat',
            'paleos_mgsio3_tables_pt_proteus_liquid_highres.dat',
        ],
    )
    # Unified PALEOS tables (Zenodo 19000316): iron, MgSiO3, H2O
    download(
        folder='EOS_PALEOS_iron',
        data_dir=Path(get_zalmoxis_root(), 'data'),
        zenodo_id=19000316,
        keep_files=['paleos_iron_eos_table_pt.dat'],
    )
    download(
        folder='EOS_PALEOS_MgSiO3_unified',
        data_dir=Path(get_zalmoxis_root(), 'data'),
        zenodo_id=19000316,
        keep_files=['paleos_mgsio3_eos_table_pt.dat'],
    )
    download(
        folder='EOS_PALEOS_H2O',
        data_dir=Path(get_zalmoxis_root(), 'data'),
        zenodo_id=19000316,
        keep_files=['paleos_water_eos_table_pt.dat'],
    )
    # Chabrier+2019/2021 H/He EOS (Zenodo 19135021): tarball with 5 tables
    chabrier_dir = Path(get_zalmoxis_root(), 'data', 'EOS_Chabrier2021_HHe')
    if not chabrier_dir.exists():
        logger.info('Downloading Chabrier H/He EOS from Zenodo 19135021...')
        download_zenodo_tarball(zenodo_id='19135021', folder_dir=chabrier_dir)
    else:
        logger.info("Folder 'EOS_Chabrier2021_HHe' already exists. Skipping download.")


if __name__ == '__main__':
    logger.info('Starting data download...')
    download_data()  # Download and extract data for Zalmoxis
    create_output()  # Create output files directory
    logger.info('Setup completed successfully!')
