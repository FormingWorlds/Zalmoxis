from __future__ import annotations

__version__ = "25.09.07"
import os
import pathlib

# Set the environment variable for ZALMOXIS_ROOT to point to the parent directory of the package folder
ZALMOXIS_ROOT = os.getenv("ZALMOXIS_ROOT")
if not ZALMOXIS_ROOT:
    ZALMOXIS_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
    os.environ["ZALMOXIS_ROOT"] = ZALMOXIS_ROOT
