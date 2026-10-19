# PALEOS-API live tabulation (producer)

Live producer side of the `PALEOS-API:*` registry family. Generates 10-column P-T `.dat` files bit-format-identical to the shipped `EOS_PALEOS_*` Zenodo tables, but sourced at runtime from the installed `paleos` Python package rather than from a flat-file checkout. This lets users opt into a different upstream PALEOS version (or a custom phase map) without re-publishing tables. The output is consumed unchanged by `eos.interpolation.load_paleos_table` (density + nabla_ad) and by `eos_export.load_paleos_all_properties` (SPIDER P-S export). Cache logic and on-disk layout live in `paleos_api_cache`; this module contains pure producers only.

The PALEOS-API path is tested with the PALEOS commit `PALEOS_TESTED_SHA`. At its first use in a process for each PALEOS SHA, in the cache resolvers and in the two generators, `check_paleos_commit` reads the commit of the installed PALEOS checkout: that commit runs, and a newer one that contains it runs with a warning. An older or unrelated commit, a clone that lacks the tested commit (a shallow clone is told to `git fetch --unshallow`), a `paleos` package that is not a tracked file of its checkout (a copy in a virtual environment, for example), a PALEOS installed without git, or package files that cannot be read stop with `PaleosVersionError`. So does package code that differs from its commit: an edit, a deleted module, older package files checked out at a newer commit, or an untracked file Python imports (`.py`, `.so`, `.pyd`, or a `.pyc` outside `__pycache__`, ignored or not). Setting `ZALMOXIS_PALEOS_ALLOW_DIRTY=1` lets such code run with a warning, for a custom phase map; its key gets `-dirty-` and a hash of that code after the SHA, so each such state has its own cache. Tracked files are compared by content as git hashes them, with its line-end filters, and a symlink by its target, so a fresh checkout is clean; a mode change, `__pycache__` and files Python does not import do not count. git runs without the repository variables of the caller (`GIT_DIR` and the like). In one process the code in memory is the code of the first import, so a serial build stops when the key has moved since then. The cache key also holds `build_key`: the RTpress install (version and commit), the md5 of the AQUA table for H2O, and the numpy, scipy and sympy versions. The scheduled workflow `paleos-main.yml` runs `tools/check_paleos_main.py` each day and fails when PALEOS main is at another commit or is not found, or when the pin cannot be read or is no 40-character lowercase commit SHA; a Zalmoxis change then moves `PALEOS_TESTED_SHA` once the PALEOS-API tests pass on it.

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
        - build_inputs
        - build_key
        - ALLOW_DIRTY_ENV
        - generate_paleos_api_unified_table
        - generate_paleos_api_2phase_mgsio3_tables
