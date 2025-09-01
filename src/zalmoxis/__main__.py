from __future__ import annotations

import logging
import os

from .zalmoxis import load_zalmoxis_config, post_processing

if __name__ == "__main__":
    # Read the environment variable for ZALMOXIS_ROOT
    ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
    if not ZALMOXIS_ROOT:
        raise RuntimeError("ZALMOXIS_ROOT environment variable not set")

    # Set up logging
    logging.basicConfig(filename=os.path.join(ZALMOXIS_ROOT, "output_files", "zalmoxis.log"), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
    config_params = load_zalmoxis_config()
    post_processing(config_params)
