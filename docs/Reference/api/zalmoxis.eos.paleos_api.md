# PALEOS-API live tabulation (producer)

Live producer side of the `PALEOS-API:*` registry family. Generates 10-column P-T `.dat` files bit-format-identical to the shipped `EOS_PALEOS_*` Zenodo tables, but sourced at runtime from the installed `paleos` Python package rather than from a flat-file checkout. This lets users opt into a different upstream PALEOS version (or a custom phase map) without re-publishing tables. The output is consumed unchanged by `eos.interpolation.load_paleos_table` (density + nabla_ad) and by `eos_export.load_paleos_all_properties` (SPIDER P-S export). Cache logic and on-disk layout live in `paleos_api_cache`; this module contains pure producers only.

The PALEOS-API path is tested with the PALEOS commit `PALEOS_TESTED_SHA`. At its first use in a process for each PALEOS SHA, `check_paleos_commit` reads the commit of the installed PALEOS checkout: that commit runs, a newer one that contains it runs with a warning, and a checkout with changes to tracked files runs with a warning that says so. An older or unrelated commit, a clone that lacks the tested commit (a shallow clone is told to `git fetch --unshallow`), a `paleos` package that is not a tracked file of its checkout (a copy in a virtual environment, for example), or a PALEOS installed without git stops with `PaleosVersionError`. A checkout with changes to tracked files gets `-dirty-` and a hash of its diff after its SHA, so each set of changes has its own cache. The scheduled workflow `paleos-main.yml` runs `tools/check_paleos_main.py` each day and fails when PALEOS main is at another commit or is not found; a Zalmoxis change then moves `PALEOS_TESTED_SHA` once the PALEOS-API tests pass on it.

::: zalmoxis.eos.paleos_api
    options:
      inherited_members: false
      show_source: true
      members:
        - GridSpec
        - make_grid_at_resolution
        - make_default_grid_iron
        - make_default_grid_mgsio3
        - make_default_grid_h2o
        - PALEOS_TESTED_SHA
        - PaleosVersionError
        - check_paleos_commit
        - paleos_installed_sha
        - generate_paleos_api_unified_table
        - generate_paleos_api_2phase_mgsio3_tables
