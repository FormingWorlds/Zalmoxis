# setup_zalmoxis.py (at root)
from __future__ import annotations

from tools.setup.setup_utils import create_output, download_data

if __name__ == '__main__':
    # Download and extract data
    download_data()
    # Create output files directory
    create_output()
