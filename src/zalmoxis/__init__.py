from __future__ import annotations

__version__ = '26.01.06'

import os
import pathlib

_zalmoxis_root: str | None = None


def get_zalmoxis_root() -> str:
    """Return the Zalmoxis repository root directory.

    Resolves lazily on first call: checks ZALMOXIS_ROOT env var, then
    auto-detects from the package file location. Caches the result.

    Returns
    -------
    str
        Absolute path to the Zalmoxis repository root.

    Raises
    ------
    RuntimeError
        If the root cannot be determined.
    """
    global _zalmoxis_root
    if _zalmoxis_root is not None:
        return _zalmoxis_root

    root = os.getenv('ZALMOXIS_ROOT')
    if not root:
        # Infer repo root: __file__ is src/zalmoxis/__init__.py, so repo root
        # is three levels up (.parent = zalmoxis/, .parent.parent = src/,
        # .parent.parent.parent = repo root where data/, input/, pyproject.toml live).
        candidate = pathlib.Path(__file__).parent.parent.parent.resolve()
        if (candidate / 'pyproject.toml').exists():
            root = str(candidate)
        else:
            # Fallback for editable installs where src/ is the repo root
            root = str(pathlib.Path(__file__).parent.parent.resolve())
        os.environ['ZALMOXIS_ROOT'] = root

    if not root or not pathlib.Path(root).exists():
        raise RuntimeError(
            'ZALMOXIS_ROOT environment variable is not set and could not be '
            'auto-detected. Set it explicitly: export ZALMOXIS_ROOT=/path/to/Zalmoxis'
        )

    _zalmoxis_root = root
    return _zalmoxis_root
