# PALEOS-API live tabulation (producer)

Live producer side of the `PALEOS-API:*` registry family. Generates 10-column P-T `.dat` files bit-format-identical to the shipped `EOS_PALEOS_*` Zenodo tables, but sourced at runtime from the installed `paleos` Python package rather than from a flat-file checkout. This lets users opt into a different upstream PALEOS version (or a custom phase map) without re-publishing tables. The output is consumed unchanged by `eos.interpolation.load_paleos_table` (density + nabla_ad) and by `eos_export.load_paleos_all_properties` (SPIDER P-S export). Cache logic and on-disk layout live in `paleos_api_cache`; this module contains pure producers only.

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
        - paleos_installed_sha
        - generate_paleos_api_unified_table
        - generate_paleos_api_2phase_mgsio3_tables
