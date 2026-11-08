from __future__ import annotations

import logging
import logging.handlers
import os

from . import get_zalmoxis_root
from .config import load_zalmoxis_config
from .output import post_processing

if __name__ == '__main__':
    root = get_zalmoxis_root()

    # The log goes to the output folder of the config, so the lines of the config load wait in memory.
    log = logging.getLogger()
    early = logging.handlers.MemoryHandler(capacity=10_000, flushLevel=logging.CRITICAL + 1)
    log.addHandler(early)
    log.setLevel(logging.INFO)
    config_params = {}
    try:
        config_params = load_zalmoxis_config()
    except Exception:
        logging.getLogger('fwl.zalmoxis').exception('The config was refused.')
        raise
    finally:  # a refused config logs into <root>/output, an unwritable folder to stderr
        output_dir = config_params.setdefault('output_dir', os.path.join(root, 'output'))
        try:
            os.makedirs(output_dir, exist_ok=True)
            log_file = logging.FileHandler(os.path.join(output_dir, 'zalmoxis.log'), mode='w')
        except OSError:
            log_file = logging.StreamHandler()
        log_file.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        early.setTarget(log_file)
        log.addHandler(log_file)
        log.removeHandler(early)
        early.close()
    post_processing(config_params)
