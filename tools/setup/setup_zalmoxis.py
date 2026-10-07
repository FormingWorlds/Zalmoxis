# setup_zalmoxis.py (at root)
from __future__ import annotations

import logging

from tools.setup.setup_utils import create_output, download_data, report_kept

logger = logging.getLogger(__name__)


def main():
    """Fetch and link the data and create ``output/``; any kept paths are reported last."""
    kept = download_data()
    create_output()
    logger.info('Data setup complete!')
    report_kept(kept)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
