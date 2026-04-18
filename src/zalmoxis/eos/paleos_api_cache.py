"""Cache resolver for ``PALEOS-API:*`` live-tabulated tables.

Converts a ``(material, GridSpec)`` or ``(GridSpec,)`` cache key to an
on-disk ``.dat`` path, regenerating via ``paleos_api.py`` on cache miss.

Cache layout under ``$ZALMOXIS_ROOT/data/EOS_PALEOS_API/``::

    unified_<material>_<sha10>_<gridhash>.dat
    2phase_solid_<sha10>_<gridhash>.dat
    2phase_liquid_<sha10>_<gridhash>.dat

Cache key is ``(PALEOS_SHA, GridSpec.hash_short())``. Both are stamped in
the generated file's header, so:

1. A missing file → regenerate.
2. An existing file whose header SHA does not match the currently installed
   PALEOS → regenerate (caught by `_header_sha_matches`).

The SHA guard is the backstop for Stage B (upstream PALEOS changes):
renaming `_phase_eos_map` or a boundary function on upstream will produce
a different SHA even if the on-disk file still exists.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from .paleos_api import (
    GridSpec,
    generate_paleos_api_2phase_mgsio3_tables,
    generate_paleos_api_unified_table,
    paleos_installed_sha,
)

logger = logging.getLogger(__name__)

_CACHE_DIR_NAME = 'EOS_PALEOS_API'


def _cache_root() -> Path:
    """Resolve the on-disk cache root under ``$ZALMOXIS_ROOT/data/``."""
    from .. import get_zalmoxis_root
    return Path(get_zalmoxis_root()) / 'data' / _CACHE_DIR_NAME


def _sha_short(sha: str) -> str:
    """First 10 chars of the SHA, or the full string if shorter."""
    return sha[:10] if len(sha) >= 10 else sha


def _header_sha_matches(path: Path, expected_sha: str) -> bool:
    """Return True iff the file exists and its header carries ``expected_sha``.

    Reads only the comment prefix; stops at the first non-comment line.
    Tolerates files without the PALEOS_SHA tag (legacy or hand-placed) by
    returning False so they get regenerated.
    """
    if not path.exists():
        return False
    try:
        with open(path) as f:
            for line in f:
                if not line.startswith('#'):
                    break
                if 'PALEOS_SHA:' in line:
                    shipped = line.split('PALEOS_SHA:', 1)[1].strip()
                    return shipped == expected_sha
    except OSError as e:
        logger.warning('paleos_api_cache: could not read %s (%s); regenerating', path, e)
        return False
    return False


def resolve_paleos_api_unified(
    material: str,
    grid: GridSpec,
    *,
    h2o_table_path: str | None = None,
    force: bool = False,
    n_workers: int = -1,
) -> Path:
    """Return the path to a cached unified PALEOS-API table, regenerating if stale.

    Parameters
    ----------
    material : {'iron', 'mgsio3', 'h2o'}
    grid : GridSpec
    h2o_table_path : str or None
        AQUA table path, passed through to the generator when ``material=='h2o'``.
    force : bool
        If True, regenerate even if a valid cached file exists. Intended for
        debugging / explicit cache-invalidation commands.
    n_workers : int
        Parallelism for the generator on cache-miss. Default ``-1`` =
        ``os.cpu_count()`` (cold-cache builds should use all cores; cache hits
        don't run the generator at all).
    """
    sha = paleos_installed_sha()
    sha_s = _sha_short(sha)
    grid_s = grid.hash_short()

    fname = f'unified_{material}_{sha_s}_{grid_s}.dat'
    path = _cache_root() / fname

    if not force and _header_sha_matches(path, sha):
        logger.debug('paleos_api_cache hit: %s', path)
        return path

    logger.info('paleos_api_cache miss: generating %s', path)
    path.parent.mkdir(parents=True, exist_ok=True)
    generate_paleos_api_unified_table(
        material=material,
        out_path=path,
        grid=grid,
        h2o_table_path=h2o_table_path,
        n_workers=n_workers,
    )
    return path


def resolve_paleos_api_2phase_mgsio3(
    grid: GridSpec,
    *,
    force: bool = False,
    n_workers: int = -1,
) -> tuple[Path, Path]:
    """Return (solid_path, liquid_path) for cached 2-phase MgSiO3 tables.

    Regenerates both files as a pair (same SHA + grid_hash) to ensure they
    are always from the same PALEOS commit. If only one side is stale, we
    still regenerate both — the generation cost is O(few min) and pairing
    them guarantees consistency for downstream mushy-zone mixing.

    Parameters
    ----------
    grid : GridSpec
    force : bool
        Regenerate even when cached files look valid.
    n_workers : int
        Parallelism for the generator on cache-miss. Default ``-1`` =
        ``os.cpu_count()``.
    """
    sha = paleos_installed_sha()
    sha_s = _sha_short(sha)
    grid_s = grid.hash_short()

    solid = _cache_root() / f'2phase_solid_{sha_s}_{grid_s}.dat'
    liquid = _cache_root() / f'2phase_liquid_{sha_s}_{grid_s}.dat'

    if (not force
            and _header_sha_matches(solid, sha)
            and _header_sha_matches(liquid, sha)):
        logger.debug('paleos_api_cache 2-phase hit: %s, %s', solid, liquid)
        return solid, liquid

    logger.info('paleos_api_cache 2-phase miss: generating %s + %s', solid, liquid)
    solid.parent.mkdir(parents=True, exist_ok=True)
    generate_paleos_api_2phase_mgsio3_tables(
        out_solid=solid,
        out_liquid=liquid,
        grid=grid,
        n_workers=n_workers,
    )
    return solid, liquid


def resolve_registry_entry(material_dict: dict) -> dict:
    """Materialise ``PALEOS-API:*`` / ``PALEOS-API-2phase:*`` registry entries in place.

    Registry entries for the live-tabulated EoS carry grid + material metadata
    but no file path. This helper detects ``format=='paleos_api'`` or
    ``format=='paleos_api_2phase'`` in either the top-level dict (unified
    materials) or in the ``solid_mantle`` / ``melted_mantle`` sub-dicts
    (2-phase materials), calls the appropriate cache resolver, writes the
    resolved on-disk path into ``eos_file``, and switches ``format`` to the
    matching downstream format (``paleos_unified`` or ``paleos``). Subsequent
    dispatch goes through the normal code path unchanged.

    Rewriting in place (rather than returning a copy) is intentional: the
    registry entry is a mutable dict that downstream consumers hold a
    reference to. Once resolved, further lookups on the same entry are O(1)
    (plain dict access).

    Parameters
    ----------
    material_dict : dict
        A registry entry. For unified materials the dict itself has
        ``format='paleos_api'``. For 2-phase materials the outer dict has
        no ``format`` key but contains ``solid_mantle`` and
        ``melted_mantle`` sub-dicts that each have ``format='paleos_api_2phase'``.

    Returns
    -------
    dict
        The same ``material_dict`` (mutated in place). Returned for chaining.
    """
    if material_dict is None:
        return material_dict

    if material_dict.get('format') == 'paleos_api':
        _resolve_unified_in_place(material_dict)
        return material_dict

    # 2-phase / Tdep layout: sub-dicts per phase.
    for sub_key in ('solid_mantle', 'melted_mantle'):
        sub = material_dict.get(sub_key)
        if isinstance(sub, dict) and sub.get('format') == 'paleos_api_2phase':
            _resolve_2phase_in_place(material_dict)
            break

    return material_dict


def _resolve_unified_in_place(mat: dict) -> None:
    """Resolve a single ``paleos_api`` unified entry in place."""
    path = resolve_paleos_api_unified(
        mat['material'],
        mat['grid_spec'],
        h2o_table_path=mat.get('h2o_table_path'),
    )
    mat['eos_file'] = str(path)
    mat['format'] = 'paleos_unified'


def _resolve_2phase_in_place(outer_mat: dict) -> None:
    """Resolve ``paleos_api_2phase`` sub-dicts (paired solid+liquid) in place.

    Generates both files as a pair (single cache lookup) and populates the
    sub-dicts. Matches the ``format='paleos'`` layout consumed by
    ``zalmoxis.eos.seager.get_tabulated_eos``.
    """
    solid_dict = outer_mat.get('solid_mantle')
    liquid_dict = outer_mat.get('melted_mantle')
    if solid_dict is None or liquid_dict is None:
        raise ValueError(
            'PALEOS-API 2-phase entry must have both solid_mantle and melted_mantle'
        )
    grid = solid_dict['grid_spec']
    if liquid_dict.get('grid_spec') is not grid:
        # Not strictly required, but guards against an entry that silently
        # mixes grids between solid and liquid sides.
        raise ValueError(
            'PALEOS-API 2-phase solid_mantle and melted_mantle must share a GridSpec'
        )
    solid_path, liquid_path = resolve_paleos_api_2phase_mgsio3(grid)
    solid_dict['eos_file'] = str(solid_path)
    solid_dict['format'] = 'paleos'
    liquid_dict['eos_file'] = str(liquid_path)
    liquid_dict['format'] = 'paleos'


def invalidate_cache(material: str | None = None) -> int:
    """Remove cache files for ``material`` (or all if ``None``).

    Returns the number of files removed. Intended for CLI / debugging use.
    """
    root = _cache_root()
    if not root.exists():
        return 0
    removed = 0
    for p in root.iterdir():
        if not p.is_file() or p.suffix != '.dat':
            continue
        if material is not None:
            # Match on the ``_<material>_`` segment (unified) or ``2phase_*`` (all).
            tag_unified = f'unified_{material}_'
            is_match = p.name.startswith(tag_unified)
            if material == 'mgsio3':
                is_match = is_match or p.name.startswith('2phase_')
            if not is_match:
                continue
        try:
            os.remove(p)
            removed += 1
        except OSError as e:
            logger.warning('paleos_api_cache: could not remove %s (%s)', p, e)
    return removed
