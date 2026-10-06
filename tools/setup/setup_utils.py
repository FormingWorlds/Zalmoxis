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
DATASETS = {
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

    ds = datasets[key]
    fetcher = create_fetcher(
        subdir=ds.subdir,
        zenodo=ds.zenodo,
        dataverse=ds.dataverse,
        registry=ds.registry(),
        extract=ds.extract,
    )
    fetcher.fetch_all()
    return fetcher.target_dir


def link_folder(link: Path, target: Path) -> None:
    """Point ``link`` at ``target``, keeping a real folder from an earlier setup.

    A link to another place is replaced. A real folder is left as it is, with a warning, since
    it can hold data the user wants to keep; Zalmoxis then reads that folder.
    """
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        logger.warning(
            "'%s' is a folder from an earlier setup, so Zalmoxis keeps reading it. "
            "To use the fwl-io copy, remove it (rm -r '%s') and run get_zalmoxis.sh again.",
            link,
            link,
        )
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target, target_is_directory=True)


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
    datasets = _datasets()
    data_dir = Path(get_zalmoxis_root(), 'data')
    for folder, (key, inner) in DATASETS.items():
        logger.info("Fetching '%s' (%s)...", folder, key)
        target = fetch_dataset(key, datasets) / inner
        if not target.is_dir():
            raise FileNotFoundError(f"Dataset {key} has no folder '{inner}' after the fetch")
        link_folder(data_dir / folder, target)


if __name__ == '__main__':
    logger.info('Starting data download...')
    download_data()  # Download and extract data for Zalmoxis
    create_output()  # Create output files directory
    logger.info('Setup completed successfully!')
