"""Check that a PALEOS table on disk is the release the fwl-io registry pins.

Every PALEOS table that Zalmoxis reads has an md5 pinned in an installed fwl-io
manifest registry. A table whose md5 differs is fetched again when it sits in
its fwl-io version directory and is refused anywhere else, so an older release
is never read without notice. A file whose name no registry lists (a table of
your own, a PALEOS-API table) is not checked, so a table of your own must not
carry the name of a registry file.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# md5 per (real path, size, mtime in ns), so each table is hashed once per process.
_HASHES: dict[tuple[str, int, int], str] = {}


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


@functools.cache
def _datasets() -> dict[str, tuple]:
    """Return ``(subdir, zenodo, dataverse, extract)`` of each PALEOS dataset key."""
    return {ds.key: (ds.subdir, ds.zenodo, ds.dataverse, ds.extract) for ds in _manifest()}


def _md5(path: str) -> str:
    """Return the md5 hex digest of a file."""
    digest = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def file_md5(path: str) -> str:
    """Return the md5 of ``path``, hashed again only when its size or mtime changes."""
    st = os.stat(path)
    key = (os.path.realpath(path), st.st_size, st.st_mtime_ns)
    if key not in _HASHES:
        _HASHES[key] = _md5(path)
    return _HASHES[key]


def _refetch(real: Path, key: str, digest: str) -> bool:
    """Fetch ``real`` again through fwl-io when it sits in the version directory of ``key``.

    Returns False when ``real`` is not in that version directory.

    Raises
    ------
    StaleTableError
        If fwl-io cannot fetch the pinned file (offline, or no mirror serves it).
    """
    from fwl_io import create_fetcher
    from fwl_io.doi import zenodo_record_id

    subdir, zenodo, dataverse, extract = _datasets()[key]
    if not zenodo:
        return False
    rel = Path(subdir, f'r{zenodo_record_id(zenodo)}').parts
    parent = real.parent.parts
    if parent[-len(rel) :] != rel:
        return False
    fetcher = create_fetcher(
        subdir=subdir,
        zenodo=zenodo,
        dataverse=dataverse,
        registry={real.name: f'md5:{digest}'},
        data_root=Path(*parent[: -len(rel)]),
        extract=extract,
    )
    try:
        fetcher.fetch(real.name)
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
    pinned = _pins().get(os.path.basename(eos_file))
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

    A file in its fwl-io version directory is fetched again first. A file whose
    name no PALEOS registry lists is not checked.

    Raises
    ------
    StaleTableError
        If the md5 of the file differs from every pinned md5 of its name and the
        pinned file cannot be put in its place.
    """
    pinned = _pins().get(os.path.basename(eos_file))
    if not pinned:
        return
    digest = file_md5(eos_file)
    if digest in pinned.values():
        return
    real = Path(os.path.realpath(eos_file))
    for key, want in pinned.items():
        if _refetch(real, key, want) and file_md5(eos_file) == want:
            logger.warning(
                '%s differed from the pinned release of %s; fetched it again', real, key
            )
            return
    raise _stale(eos_file, pinned, digest)
