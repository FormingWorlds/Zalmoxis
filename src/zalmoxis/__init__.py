from __future__ import annotations

__version__ = '26.01.06'
import os
import pathlib
import sys

# Determine ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')

# Set ZALMOXIS_ROOT automatically if not set
if not ZALMOXIS_ROOT:
    # Infer root as parent of the package directory
    ZALMOXIS_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
    os.environ['ZALMOXIS_ROOT'] = ZALMOXIS_ROOT

# Final check for ZALMOXIS_ROOT validity
if not ZALMOXIS_ROOT or not pathlib.Path(ZALMOXIS_ROOT).exists():
    sys.stderr.write(
        'Error: ZALMOXIS_ROOT environment variable is not set. Set it explicitly to the root of the repo with: export ZALMOXIS_ROOT=$(pwd)\n'
    )
    sys.exit(1)
