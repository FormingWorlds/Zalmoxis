from __future__ import annotations

import os
import pathlib

try:
    from ._version import __version__, __version_tuple__
except ImportError:
    # Fallback for when the package is not installed (e.g., running from source
    # without setuptools-scm having generated _version.py).
    __version__ = '0.0.0.dev0'
    __version_tuple__ = (0, 0, 0, 'dev0')

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
