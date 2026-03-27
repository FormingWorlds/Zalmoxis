from __future__ import annotations

import logging
import os

from . import get_zalmoxis_root
from .config import load_zalmoxis_config
from .output import post_processing

if __name__ == '__main__':
    root = get_zalmoxis_root()

    logging.basicConfig(
        filename=os.path.join(root, 'output', 'zalmoxis.log'),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w',
    )
    config_params = load_zalmoxis_config()
    post_processing(config_params)
