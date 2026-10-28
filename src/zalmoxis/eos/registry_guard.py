"""Check that a PALEOS table on disk is the release the fwl-io registry pins.

A PALEOS table whose file name an installed fwl-io manifest registry pins is read
only when its md5 is a pinned one. A table that differs is fetched again when it
sits in its version directory under the current FWL_DATA and is refused anywhere
else, so an older release is never read without notice. A file whose name no
registry lists (a table of your own, a PALEOS-API table) is not checked, so a
table of your own must not carry the name of a registry file. Without an
importable fwl-io no file can be checked, since the manifest says which names are
pinned, so every checked read raises MissingFwlIoError.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import os
from pathlib import Path

logger = logging.getLogger('fwl.' + __name__)

# md5 per (device, inode, size, mtime, ctime), so each table is hashed once.
_HASHES: dict[tuple, str] = {}


class StaleTableError(Exception):
    """A PALEOS table differs from the release its registry pins.

    Not a RuntimeError or ValueError, so a caller that turns those into a failed
    structure solve does not hide it.
    """


class MissingFwlIoError(StaleTableError):
    """fwl-io cannot be imported, so no PALEOS table can be checked against its registry.

    A StaleTableError, so the PALEOS, Seager and melting-curve readers, which pass
    StaleTableError on and turn other errors into None, stop; not an ImportError, so the
    binary-cache reader, which rebuilds a cache it cannot unpickle, does not take it for
    a broken cache.
    """


def _fwl_io():
    """Return the fwl_io module once every part of it this module uses imports.

    Raises
    ------
    MissingFwlIoError
        If fwl-io, or a part of it this module uses, cannot be imported.
    """
    try:
        import fwl_io
        from fwl_io import (  # noqa: F401
            MissingDataRootError,
            create_fetcher,
            load_manifest,
            resolve_data_root,
        )
        from fwl_io.doi import zenodo_record_id  # noqa: F401
        from fwl_io.manifest import shared_manifest_path  # noqa: F401
    except ImportError as exc:
        raise MissingFwlIoError(
            f'fwl-io cannot be imported ({exc}), so no EOS table can be checked against the '
            'fwl-io registry. Install it with pip install "fwl-io>=26.10.7", or '
            'pip install -e <your fwl-io checkout>.'
        ) from exc
    return fwl_io


@functools.cache
def _manifest() -> tuple:
    """Return the PALEOS datasets of the fwl-io shared manifest and the Zalmoxis manifest.

    Raises
    ------
    MissingFwlIoError
        If fwl-io, or a part of it this module uses, cannot be imported.
    """
    fwl_io = _fwl_io()
    from zalmoxis.datasets import manifest_path

    return tuple(
        ds
        for path in (fwl_io.manifest.shared_manifest_path(), manifest_path())
        for ds in fwl_io.load_manifest(path)
        if 'paleos' in ds.key
    )


@functools.cache
def _pins() -> dict[str, dict[str, str]]:
    """Return the pin of each PALEOS file name per dataset key: a bare md5, or
    ``<algorithm>:<digest>`` for another form (a bare registry hash is sha256), which
    :func:`_pinned` refuses."""
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
        If a pin of the name is not an md5, or two datasets pin it to different releases.
    """
    pinned = next((v for n, v in _pins().items() if n.casefold() == name.casefold()), {})
    for key, pin in pinned.items():
        if ':' in pin:
            raise StaleTableError(
                f'{name} is pinned as {pin} by {key}; only md5 pins are checked, so the table '
                'is not read'
            )
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


def _matches(path: str, pins: set[str]) -> bool:
    """Whether the md5 of the file is one of ``pins``."""
    return file_md5(path) in pins


def _refetch(real: Path, key: str, digest: str) -> bool:
    """Fetch ``real`` again through fwl-io when it sits in the version directory of ``key``
    under the current FWL_DATA; fwl-io places the file only once it matches the pin ``digest``.

    Returns False when ``real`` is not in that version directory, when no FWL_DATA is set,
    or when the manifest has no dataset ``key`` or no Zenodo record for it.

    Raises
    ------
    StaleTableError
        If fwl-io cannot fetch the pinned file (offline, or no mirror serves it).
    """
    fwl_io = _fwl_io()
    ds = next((d for d in _manifest() if d.key == key), None)
    try:
        root = os.path.realpath(fwl_io.resolve_data_root(create=False))
    except fwl_io.MissingDataRootError:  # no FWL_DATA: nothing to fetch into
        return False
    if ds is None or not ds.zenodo:
        return False
    try:
        version_dir = Path(
            root, ds.subdir, f'r{fwl_io.doi.zenodo_record_id(ds.zenodo)}'
        ).resolve()
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


def verify_digest(eos_file: str, digest: str | None, label: str | None = None) -> None:
    """Raise unless ``digest`` is a pin common to the pinned names of ``eos_file``: its own
    name and, for a link, the name of the file it resolves to, also when that file is gone.

    Used for a binary cache whose text table is absent or unreadable: the cache carries
    the md5 of the table it was built from, and ``label`` (the cache file) names it in the
    error. A file whose names no PALEOS registry lists passes.

    Raises
    ------
    MissingFwlIoError
        If fwl-io cannot be imported.
    StaleTableError
        If a name is pinned and ``digest`` is not a pin common to the pinned names, or if
        the check itself fails (a broken registry, a pin that cannot be checked).
    """
    with _checking(label or eos_file):
        found = _accepted(eos_file)
        if found is not None and digest not in found[0]:
            raise _stale(label or eos_file, found[1], found[0], digest)


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
    MissingFwlIoError
        If fwl-io cannot be imported; the message names ``eos_file``.
    StaleTableError
        If the file is not a release pinned for its names and the pinned file cannot be
        put in its place, or if the check itself fails (an unreadable file, a broken
        registry, a pin that cannot be checked).
    """
    with _checking(eos_file):
        _verify(eos_file)


@contextlib.contextmanager
def _checking(name: str):
    """Name ``name`` in a MissingFwlIoError and turn any other failure of the check that is
    not a StaleTableError into the cannot-check StaleTableError."""
    try:
        yield
    except MissingFwlIoError as exc:
        raise MissingFwlIoError(f'{name} is not read: {exc}') from exc.__cause__
    except StaleTableError:
        raise
    except Exception as exc:
        raise StaleTableError(f'cannot check {name} against its registry: {exc}') from exc


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
