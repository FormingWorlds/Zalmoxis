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
    """Return the pin of each PALEOS file name per dataset key: a bare md5, or
    ``<algorithm>:<digest>`` for another algorithm (a bare registry hash is sha256)."""
    pins: dict[str, dict[str, str]] = {}
    for ds in _manifest():
        for name, known in ds.registry().items():
            algorithm, sep, digest = known.lower().partition(':')
            if not sep:
                algorithm, digest = 'sha256', algorithm
            pins.setdefault(name, {})[ds.key] = (
                digest if algorithm == 'md5' else f'{algorithm}:{digest}'
            )
    return pins


def _pinned(name: str) -> dict[str, str]:
    """Return the pins per dataset key of a file name, compared case-insensitively.

    Raises
    ------
    StaleTableError
        If two datasets pin the name to different releases.
    """
    pinned = next((v for n, v in _pins().items() if n.casefold() == name.casefold()), {})
    if len(set(pinned.values())) > 1:
        raise StaleTableError(f'the registries disagree for {name}: {pinned}')
    return pinned


def _accepted(eos_file: str) -> tuple[set[str], list[str]] | None:
    """Return the pins common to every pinned name among the name of ``eos_file`` and the
    name of the file it resolves to, and those dataset keys; None when neither is pinned.
    A link of one pinned name to another pinned table therefore accepts no release."""
    names = {os.path.basename(eos_file), os.path.basename(os.path.realpath(eos_file))}
    pinned = [p for p in map(_pinned, names) if p]
    if not pinned:
        return None
    return set.intersection(*(set(p.values()) for p in pinned)), sorted(
        {k for p in pinned for k in p}
    )


def _hash(path: str, algorithm: str = 'md5') -> str:
    """Return the hex digest of a file."""
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, algorithm).hexdigest()


_md5 = _hash


def file_hash(path: str, algorithm: str = 'md5') -> str:
    """Return the digest of ``path``, hashed again when the file changes on disk."""
    st = os.stat(path)
    key = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns, algorithm)
    if key not in _HASHES:
        _HASHES[key] = _md5(path) if algorithm == 'md5' else _hash(path, algorithm)
    return _HASHES[key]


def file_md5(path: str) -> str:
    """Return the md5 of ``path``, hashed again when the file changes on disk."""
    return file_hash(path)


def _matches(path: str, pins: set[str]) -> bool:
    """Whether the file has the digest of one of ``pins`` (a bare md5 or ``alg:digest``)."""
    for pin in pins:
        algorithm, sep, digest = pin.partition(':')
        algorithm, digest = (algorithm, digest) if sep else ('md5', pin)
        try:
            if file_hash(path, algorithm) == digest:
                return True
        except ValueError as exc:  # an algorithm hashlib does not know
            raise StaleTableError(f'the pin {pin} of {path} cannot be checked: {exc}') from exc
    return False


def _refetch(real: Path, key: str, digest: str) -> bool:
    """Fetch ``real`` again through fwl-io when it sits in the version directory of ``key``
    under the current FWL_DATA; fwl-io places the file only once it matches the pin ``digest``.

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
            registry={real.name: digest if ':' in digest else f'md5:{digest}'},
            data_root=root,
            extract=ds.extract,
        ).fetch(real.name)
    except Exception as exc:
        raise StaleTableError(
            f'{real} differs from the pinned release of {key} and could not be fetched '
            f'again: {exc}'
        ) from exc
    return True


def verify_digest(eos_file: str, digest: str | None, label: str | None = None) -> None:
    """Raise unless ``digest`` is a pinned md5 of the name of ``eos_file``.

    Used for a binary cache whose text table is absent or unreadable: the cache carries
    the md5 of the table it was built from, and ``label`` (the cache file) names it in the
    error. A name no PALEOS registry lists passes.

    Raises
    ------
    StaleTableError
        If the name is pinned and ``digest`` is none of its pinned md5 values.
    """
    pinned = _pinned(os.path.basename(eos_file))
    if pinned and digest not in pinned.values():
        raise _stale(label or eos_file, sorted(pinned), set(pinned.values()), digest)


def _stale(eos_file: str, keys: list[str], want: set[str], digest) -> StaleTableError:
    """Return the error for a table whose digest is not a pinned one."""
    return StaleTableError(
        f'{eos_file} is not the release the registry pins for {", ".join(keys)}: '
        f'md5 {digest}, pinned {", ".join(sorted(want)) or "no release common to its names"}. '
        'Move it aside and fetch the pinned table: under data/, run bash '
        'tools/setup/get_zalmoxis.sh; elsewhere, fwl-io fetch --key <key>. A table of your '
        'own must not use a registry file name.'
    )


def verify(eos_file: str) -> None:
    """Raise unless ``eos_file`` is the release its registry pins.

    The digest must be pinned for the file name and, for a link, also for the name of the
    file it resolves to; names are compared case-insensitively. A file in its version
    directory under the current FWL_DATA is fetched again first and checked once more. A
    file whose names no PALEOS registry lists is not checked.

    Raises
    ------
    StaleTableError
        If the file is not a release pinned for its names and the pinned file cannot be
        put in its place, or if the check itself fails (an unreadable file, a broken
        registry, a pin that cannot be checked).
    """
    try:
        _verify(eos_file)
    except StaleTableError:
        raise
    except Exception as exc:
        raise StaleTableError(f'cannot check {eos_file} against its registry: {exc}') from exc


def _verify(eos_file: str) -> None:
    """``verify`` without the conversion of other errors."""
    found = _accepted(eos_file)
    if found is None:
        return
    accepted, keys = found
    if _matches(eos_file, accepted):
        return
    real = Path(os.path.realpath(eos_file))
    for key, want in _pinned(real.name).items():
        if want in accepted and _refetch(real, key, want) and _matches(eos_file, accepted):
            logger.warning(
                '%s differed from the pinned release of %s; fetched it again', real, key
            )
            return
    raise _stale(eos_file, keys, accepted, file_md5(eos_file))
