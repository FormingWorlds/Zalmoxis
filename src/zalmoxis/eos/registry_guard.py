"""Check that a PALEOS table on disk is the release the fwl-io registry pins.

A PALEOS table whose file name an installed fwl-io manifest registry pins is read
only when its md5 is a pinned one. A table that differs is fetched again when it
sits in its version directory under the current FWL_DATA and is refused anywhere
else, so an older release is never read without notice. A file whose name no
registry lists (a table of your own, a PALEOS-API table) is not checked, so a
table of your own must not carry the name of a registry file.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# md5 per (real path, device, inode, size, mtime, ctime), so each table is hashed once.
_HASHES: dict[tuple, str] = {}


class StaleTableError(Exception):
    """A PALEOS table differs from the release its registry pins.

    Not a RuntimeError or ValueError, so a caller that turns those into a failed
    structure solve does not hide it.
    """


@functools.cache
def _manifest() -> tuple:
    """Return the PALEOS datasets of the fwl-io shared manifest and the Zalmoxis manifest."""
    from fwl_io import load_manifest
    from fwl_io.manifest import shared_manifest_path

    from zalmoxis.datasets import manifest_path

    return tuple(
        ds
        for path in (shared_manifest_path(), manifest_path())
        for ds in load_manifest(path)
        if 'paleos' in ds.key
    )


@functools.cache
def _pins() -> dict[str, dict[str, str]]:
    """Return the pinned md5 of each PALEOS file name, per dataset key."""
    pins: dict[str, dict[str, str]] = {}
    for ds in _manifest():
        for name, known in ds.registry().items():
            algorithm, _, digest = known.partition(':')
            if algorithm == 'md5':
                pins.setdefault(name, {})[ds.key] = digest
    return pins


def _pinned(eos_file: str) -> dict[str, str]:
    """Return the pinned md5 per dataset key of the name of ``eos_file`` and of the file it
    resolves to, so a link of another name is checked as the file it reads."""
    pins = _pins()
    names = {os.path.basename(eos_file), os.path.basename(os.path.realpath(eos_file))}
    return {k: v for n in names for k, v in pins.get(n, {}).items()}


def _md5(path: str) -> str:
    """Return the md5 hex digest of a file."""
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'md5').hexdigest()


def file_md5(path: str) -> str:
    """Return the md5 of ``path``, hashed again when the file changes on disk."""
    st = os.stat(path)
    key = (
        os.path.realpath(path),
        st.st_dev,
        st.st_ino,
        st.st_size,
        st.st_mtime_ns,
        st.st_ctime_ns,
    )
    if key not in _HASHES:
        _HASHES[key] = _md5(path)
    return _HASHES[key]


def _refetch(real: Path, key: str, digest: str) -> bool:
    """Fetch ``real`` again through fwl-io when it sits in the version directory of ``key``
    under the current FWL_DATA; fwl-io places the file only once its md5 is ``digest``.

    Returns False when ``real`` is not in that version directory.

    Raises
    ------
    StaleTableError
        If fwl-io cannot fetch the pinned file (offline, or no mirror serves it).
    """
    import fwl_io
    from fwl_io.doi import zenodo_record_id

    ds = next(d for d in _manifest() if d.key == key)
    try:
        root = os.path.realpath(fwl_io.resolve_data_root(create=False))
    except Exception:  # no FWL_DATA: nothing to fetch into
        return False
    if not ds.zenodo:
        return False
    rel = os.path.join(root, ds.subdir, f'r{zenodo_record_id(ds.zenodo)}')
    if str(real.parent) != rel:
        return False
    try:
        fwl_io.create_fetcher(
            subdir=ds.subdir,
            zenodo=ds.zenodo,
            dataverse=ds.dataverse,
            registry={real.name: f'md5:{digest}'},
            data_root=root,
            extract=ds.extract,
        ).fetch(real.name)
    except Exception as exc:
        raise StaleTableError(
            f'{real} differs from the pinned release of {key} and could not be fetched '
            f'again: {exc}'
        ) from exc
    return True


def verify_digest(eos_file: str, digest: str) -> None:
    """Raise unless ``digest`` is a pinned md5 of the name of ``eos_file``.

    Used for a binary cache whose text table is absent: the cache carries the
    md5 of the table it was built from. A name no PALEOS registry lists passes.

    Raises
    ------
    StaleTableError
        If the name is pinned and ``digest`` is none of its pinned md5 values.
    """
    pinned = _pinned(eos_file)
    if pinned and digest not in pinned.values():
        raise _stale(eos_file, pinned, digest)


def _stale(eos_file: str, pinned: dict[str, str], digest: str) -> StaleTableError:
    """Return the error for a table whose md5 is not a pinned one."""
    return StaleTableError(
        f'{eos_file} is not the release the registry pins for '
        f'{", ".join(sorted(pinned))}: md5 {digest}, '
        f'pinned {", ".join(sorted(set(pinned.values())))}. Move it aside and fetch the '
        'pinned table: under data/, run bash tools/setup/get_zalmoxis.sh; elsewhere, '
        'fwl-io fetch --key <key>. A table of your own must not use a registry file name.'
    )


def verify(eos_file: str) -> None:
    """Raise unless ``eos_file`` is the release its registry pins.

    A file in its version directory under the current FWL_DATA is fetched again
    first. A file whose name, and the name of the file it resolves to, no PALEOS
    registry lists is not checked.

    Raises
    ------
    StaleTableError
        If the md5 of the file differs from every pinned md5 of its name and the
        pinned file cannot be put in its place.
    """
    pinned = _pinned(eos_file)
    if not pinned:
        return
    digest = file_md5(eos_file)
    if digest in pinned.values():
        return
    real = Path(os.path.realpath(eos_file))
    for key, want in pinned.items():
        if _refetch(real, key, want):
            logger.warning(
                '%s differed from the pinned release of %s; fetched it again', real, key
            )
            return
    raise _stale(eos_file, pinned, digest)
