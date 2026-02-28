from __future__ import annotations

__version__ = '26.01.06'
import os
import pathlib
import sys

# Determine ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')

# Set ZALMOXIS_ROOT automatically if not set
if not ZALMOXIS_ROOT:
    # Infer repo root: __file__ is src/zalmoxis/__init__.py, so repo root
    # is three levels up (.parent = zalmoxis/, .parent.parent = src/,
    # .parent.parent.parent = repo root where data/, input/, pyproject.toml live).
    _candidate = pathlib.Path(__file__).parent.parent.parent.resolve()
    if (_candidate / 'pyproject.toml').exists():
        ZALMOXIS_ROOT = str(_candidate)
    else:
        # Fallback for editable installs where src/ is the repo root
        ZALMOXIS_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())
    os.environ['ZALMOXIS_ROOT'] = ZALMOXIS_ROOT

# Final check for ZALMOXIS_ROOT validity
if not ZALMOXIS_ROOT or not pathlib.Path(ZALMOXIS_ROOT).exists():
    sys.stderr.write(
        'Error: ZALMOXIS_ROOT environment variable is not set. Set it explicitly to the root of the repo with: export ZALMOXIS_ROOT=$(pwd)\n'
    )
    sys.exit(1)
