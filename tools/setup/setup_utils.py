"""Fetch the Zalmoxis input data through fwl-io and link it into ``data/``.

fwl-io downloads each dataset from its Zenodo record, falls back to the DataverseNL mirror
the manifest pins, checks every file against the committed registry, and places the files in
``<FWL_DATA>/<key-as-path>/r<record-id>``. Zalmoxis reads ``<ZALMOXIS_ROOT>/data/<Folder>``, so
each folder is a symbolic link to its dataset there, shared with PROTEUS.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from zalmoxis import get_zalmoxis_root

logger = logging.getLogger(__name__)

# Folder under data/ -> (fwl-io dataset key, path of the folder inside the version directory)
FOLDERS = {
    'EOS_Seager2007': ('interior.eos.seager_2007', ''),
    'EOS_WolfBower2018_1TPa': ('interior.eos.wolf_bower_2018_1tpa', ''),
    'radial_profiles': ('interior.radial_profiles', ''),
    'mass_radius_curves': ('interior.mass_radius.zeng_2019', ''),
    'EOS_RTPress_melt_100TPa': ('interior.eos.rtpress_melt_100tpa', ''),
    'melting_curves_Monteux-600': ('interior.melting_curves.monteux_minus_600', ''),
    'EOS_PALEOS_MgSiO3': ('interior.eos.paleos_mgsio3', ''),
    'EOS_PALEOS_iron': ('interior.eos.paleos_iron', ''),
    'EOS_PALEOS_MgSiO3_unified': ('interior.eos.paleos_mgsio3_unified', ''),
    'EOS_PALEOS_H2O': ('interior.eos.paleos_h2o', ''),
    'EOS_Chabrier2021_HHe': ('interior.eos.chabrier_2021_hhe', 'EOS_Chabrier2021_HHe'),
}


def _datasets() -> dict:
    """Return the datasets of the fwl-io shared manifest and the Zalmoxis manifest by key."""
    from fwl_io import load_manifest
    from fwl_io.manifest import shared_manifest_path

    from zalmoxis.datasets import manifest_path

    return {
        ds.key: ds
        for path in (shared_manifest_path(), manifest_path())
        for ds in load_manifest(path)
    }


def fetch_dataset(key: str, datasets: dict) -> Path:
    """Fetch one dataset into FWL_DATA and return its version directory.

    Raises
    ------
    KeyError
        If no manifest declares ``key``; the installed fwl-io is then older than this
        Zalmoxis needs.
    """
    from fwl_io import create_fetcher

    try:
        ds = datasets[key]
    except KeyError:
        raise KeyError(f'{key} is in no installed fwl-io manifest; upgrade fwl-io') from None
    fetcher = create_fetcher(
        subdir=ds.subdir,
        zenodo=ds.zenodo,
        dataverse=ds.dataverse,
        registry=ds.registry(),
        extract=ds.extract,
        progress=True,
    )
    fetcher.fetch_all()
    return fetcher.target_dir


def link_folder(link: Path, target: Path, data_root: Path) -> bool:
    """Point ``link`` at ``target``; return False when the user's own folder or link is kept.

    A link that is dangling or points into ``data_root`` is replaced, and an empty folder is
    removed first. Anything else can hold data the user wants to keep, so it stays as it is and
    Zalmoxis reads it.
    """
    if link.is_symlink():
        if link.exists() and not link.resolve().is_relative_to(data_root.resolve()):
            return False
        link.unlink()
    elif link.is_dir() and not any(link.iterdir()):
        link.rmdir()
    elif link.exists():
        return False
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)
    return True


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
    """Fetch every dataset Zalmoxis reads and link it into ``<ZALMOXIS_ROOT>/data/``.

    Raises
    ------
    fwl_io.MissingDataRootError
        If FWL_DATA is not set.
    """
    from fwl_io import resolve_data_root

    datasets = _datasets()
    data_dir = Path(get_zalmoxis_root(), 'data')
    kept = []
    for folder, (key, inner) in FOLDERS.items():
        logger.info("Fetching '%s' (%s)...", folder, key)
        target = fetch_dataset(key, datasets) / inner
        if not target.is_dir():
            raise FileNotFoundError(f"Dataset {key} has no folder '{inner}' after the fetch")
        if not link_folder(data_dir / folder, target, resolve_data_root()):
            kept.append(data_dir / folder)
    if kept:
        logger.warning(
            'Zalmoxis keeps reading these folders from an earlier setup, not the fetched data. '
            'To use the fetched data, remove them and run get_zalmoxis.sh again:\n%s',
            '\n'.join(f"  rm -r '{path}'" for path in kept),
        )


if __name__ == '__main__':
    logger.info('Starting data download...')
    download_data()  # Download and extract data for Zalmoxis
    create_output()  # Create output files directory
    logger.info('Setup completed successfully!')
