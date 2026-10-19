# PALEOS-API cache resolver

On-disk cache for `PALEOS-API:*` tables under `$ZALMOXIS_ROOT/data/EOS_PALEOS_API/`. Converts a `(material, GridSpec)` cache key into a concrete `.dat` path, regenerating via `paleos_api.py` on cache miss. Cache key is `(PALEOS_SHA, BUILD_KEY, GridSpec.hash_short())`, all in the file name and the first two stamped into the file header, so a missing file, a SHA mismatch (upstream PALEOS upgraded) or a build key mismatch (RTpress, the AQUA table, numpy, scipy or sympy changed) triggers regeneration. The SHA guard is the backstop for upstream PALEOS changes: renaming `_phase_eos_map` or a boundary function on upstream produces a different SHA and invalidates the disk file even if the path still exists.

::: zalmoxis.eos.paleos_api_cache
    options:
      inherited_members: false
      show_source: true
      members:
        - resolve_paleos_api_unified
        - resolve_paleos_api_2phase_mgsio3
        - resolve_registry_entry
        - invalidate_cache
