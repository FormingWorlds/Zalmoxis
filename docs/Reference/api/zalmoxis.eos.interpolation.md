# EOS interpolation

Shared infrastructure for PALEOS-format tables: file readers (`load_paleos_table` for the 2-phase format, `load_paleos_unified_table` for the single-file-per-material format), the log-uniform grid builder used by both, and the inner kernels `_fast_bilinear` (O(1) bilinear lookup) and `_paleos_clamp_temperature` (per-cell temperature clamp to the table's valid range). These kernels are called millions of times per Zalmoxis solve, so they avoid `np.interp` binary searches by exploiting the log-uniform grid. The cache populated by `_ensure_unified_cache` is reused across calls within the same process.

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
