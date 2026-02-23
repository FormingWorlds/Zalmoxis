# setup_zalmoxis.py (at root)
from __future__ import annotations

from src.tools.setup_utils import create_output_files, download_data

if __name__ == '__main__':
    # Download and extract data
    download_data()
    # Create output files directory
    create_output_files()
