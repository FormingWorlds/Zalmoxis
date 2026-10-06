"""The fwl-io manifest of the datasets only Zalmoxis reads."""

from __future__ import annotations

from pathlib import Path


def manifest_path() -> Path:
    """Return the path of the Zalmoxis fwl-io manifest."""
    return Path(__file__).with_name('zalmoxis_manifest.toml')
