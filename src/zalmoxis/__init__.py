from __future__ import annotations

__version__ = "25.09.07"
import os
import pathlib
import sys

# Check for ZALMOXIS_ROOT environment variable
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    sys.stderr.write(
        "Error: ZALMOXIS_ROOT environment variable is not set. Set it explicitly to the root of the repo with: export ZALMOXIS_ROOT=$(pwd)"
    )
    sys.exit(1)
