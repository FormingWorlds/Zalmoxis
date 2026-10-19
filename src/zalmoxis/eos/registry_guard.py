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

# md5 per (device, inode, size, mtime, ctime), so each table is hashed once.
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


def _pinned(name: str) -> dict[str, str]:
    """Return the pinned md5 per dataset key of a file name, compared case-insensitively."""
    return next((v for n, v in _pins().items() if n.casefold() == name.casefold()), {})


def _accepted(eos_file: str) -> tuple[set[str], list[str]] | None:
    """Return the md5 values pinned for every pinned name among the name of ``eos_file`` and
    the name of the file it resolves to, and those dataset keys; None when neither is pinned.
    A link of one pinned name to another pinned table therefore accepts no md5."""
    names = {os.path.basename(eos_file), os.path.basename(os.path.realpath(eos_file))}
    pinned = [p for p in map(_pinned, names) if p]
    if not pinned:
        return None
    return set.intersection(*(set(p.values()) for p in pinned)), sorted(
        {k for p in pinned for k in p}
    )


def _md5(path: str) -> str:
    """Return the md5 hex digest of a file."""
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'md5').hexdigest()


def file_md5(path: str) -> str:
    """Return the md5 of ``path``, hashed again when the file changes on disk."""
    st = os.stat(path)
    key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
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

    ds = next((d for d in _manifest() if d.key == key), None)
    try:
        root = os.path.realpath(fwl_io.resolve_data_root(create=False))
    except fwl_io.MissingDataRootError:  # no FWL_DATA: nothing to fetch into
        return False
    if ds is None or not ds.zenodo:
        return False
    try:
        version_dir = Path(root, ds.subdir, f'r{zenodo_record_id(ds.zenodo)}').resolve()
        if real.parent != version_dir:
            return False
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


def verify_digest(eos_file: str, digest: str | None) -> None:
    """Raise unless ``digest`` is a pinned md5 of the name of ``eos_file``.

    Used for a binary cache whose text table is absent: the cache carries the
    md5 of the table it was built from. A name no PALEOS registry lists passes.

    Raises
    ------
    StaleTableError
        If the name is pinned and ``digest`` is none of its pinned md5 values.
    """
    pinned = _pinned(os.path.basename(eos_file))
    if pinned and digest not in pinned.values():
        raise _stale(eos_file, sorted(pinned), set(pinned.values()), digest)


def _stale(eos_file: str, keys: list[str], want: set[str], digest) -> StaleTableError:
    """Return the error for a table whose md5 is not a pinned one."""
    return StaleTableError(
        f'{eos_file} is not the release the registry pins for {", ".join(keys)}: '
        f'md5 {digest}, pinned {", ".join(sorted(want)) or "no md5 common to its names"}. '
        'Move it aside and fetch the pinned table: under data/, run bash '
        'tools/setup/get_zalmoxis.sh; elsewhere, fwl-io fetch --key <key>. A table of your '
        'own must not use a registry file name.'
    )


def verify(eos_file: str) -> None:
    """Raise unless ``eos_file`` is the release its registry pins.

    The md5 must be pinned for the file name and, for a link, also for the name of the
    file it resolves to. A file in its version directory under the current FWL_DATA is
    fetched again first and checked once more. A file whose names no PALEOS registry
    lists is not checked; names are compared case-insensitively.

    Raises
    ------
    StaleTableError
        If the md5 of the file is not pinned for its names and the pinned file cannot be
        put in its place.
    """
    found = _accepted(eos_file)
    if found is None:
        return
    accepted, keys = found
    digest = file_md5(eos_file)
    if digest in accepted:
        return
    real = Path(os.path.realpath(eos_file))
    for key, want in _pinned(real.name).items():
        if want in accepted and _refetch(real, key, want) and file_md5(eos_file) in accepted:
            logger.warning(
                '%s differed from the pinned release of %s; fetched it again', real, key
            )
            return
    raise _stale(eos_file, keys, accepted, digest)
