# PALEOS-API live tabulation (producer)

Live producer side of the `PALEOS-API:*` registry family. Generates 10-column P-T `.dat` files bit-format-identical to the shipped `EOS_PALEOS_*` Zenodo tables, but sourced at runtime from the installed `paleos` Python package rather than from a flat-file checkout. This lets users opt into a different upstream PALEOS version (or a custom phase map) without re-publishing tables. The output is consumed unchanged by `eos.interpolation.load_paleos_table` (density + nabla_ad) and by `eos_export.load_paleos_all_properties` (SPIDER P-S export). Cache logic and on-disk layout live in `paleos_api_cache`; this module contains pure producers only.

The PALEOS-API path is tested with the PALEOS commit `PALEOS_TESTED_SHA`. At its first use in a process for each PALEOS SHA, in the cache resolvers and in the two generators, `check_paleos_commit` reads the commit of the installed PALEOS checkout: that commit runs, and a newer one that contains it runs with a warning. An older or unrelated commit, a clone that lacks the tested commit (a shallow clone is told to `git fetch --unshallow`), a `paleos` package that is not a tracked file of its checkout (a copy in a virtual environment, for example), a PALEOS installed without git, or package files that cannot be read stop with `PaleosVersionError`. So does package code that differs from its commit: an edit, a deleted module, older package files checked out at a newer commit, an untracked file Python imports (`.py`, `.so`, `.pyd`, or a `.pyc` outside `__pycache__`, ignored or not), or such a file added to the index but not committed. A link to a directory, a nested repository or a submodule under `paleos/`, committed or not, stops also with the opt-in below, because no key can describe the code in it; a plain untracked directory counts only through the files in it that Python imports. Setting `ZALMOXIS_PALEOS_ALLOW_DIRTY=1` lets such code run with a warning, for a custom phase map; its key gets `-dirty-` and a hash of that code after the SHA, so each such state has its own cache. Tracked files are compared by content as git hashes them without filters, so a checkout with line-end conversion or a clean filter counts as changed, and a symlink by its target; a `paleos` package below the top of its repository (under `src/`, for example) is compared by its paths in the repository; a mode change, `__pycache__` and files Python does not import do not count. git runs without the repository variables of the caller (`GIT_DIR` and the like). In one process the code in memory is the code of the first import, so a serial build stops when the key has moved since the first key; when `paleos` was imported before that first key, a warning says that a change on disk before it is not covered. The cache key also holds `build_key`: the RTpress install (version and commit), the md5 of the AQUA table for H2O, and the numpy, scipy and sympy versions. For an editable RTpress the key holds its HEAD only, so uncommitted edits to RTpress are not keyed; for RTpress installed from a directory without git it holds the version only. The scheduled workflow `paleos-main.yml` runs `tools/check_paleos_main.py` each day and fails when PALEOS main does not contain the tested commit (diverged or behind), when the repository, main or the tested commit is not found, when GitHub refuses the request (401, or 403 with requests left and no Retry-After), or when the pin cannot be read or is no 40-character lowercase commit SHA; main ahead of the tested commit warns and lists the new commits in the run summary; a used-up or secondary rate limit, a 5xx, a network fault or a malformed reply only warns; a Zalmoxis change then moves `PALEOS_TESTED_SHA` once the PALEOS-API tests pass on it.

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
