# EOS interpolation

Shared infrastructure for PALEOS-format tables: file readers (`load_paleos_table` for the 2-phase format, `load_paleos_unified_table` for the single-file-per-material format), the log-uniform grid builder used by both, and the inner kernels `_fast_bilinear` (O(1) bilinear lookup) and `_paleos_clamp_temperature` (per-cell temperature clamp to the table's valid range). These kernels are called millions of times per Zalmoxis solve, so they avoid `np.interp` binary searches by exploiting the log-uniform grid. The cache populated by `_ensure_unified_cache` is reused across calls within the same process; its binary copy (`x.cache.pkl` beside the table `x.dat`) holds the md5 of the table it was built from and the identity of the code that built it (`cache_builder`: the loader source taken at import, or its compiled code when no source is installed, and the numpy and scipy versions), carries a sha256 of its bytes, is written to a temporary file renamed into place, is not written when the table changes while it is read, and is used only for that table and that code.

::: zalmoxis.eos.interpolation
    options:
      inherited_members: false
      show_source: true
      members:
        - load_paleos_table
        - load_paleos_unified_table
        - _fast_bilinear
        - fast_bilinear_batch
        - _paleos_clamp_temperature
        - _ensure_unified_cache
