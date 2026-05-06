# Seager2007 tabulated EOS

Loaders and interpolators for tabulated equations of state in the Seager et al. (2007) format, covering the 1D pressure-density tables shipped under `data/EOS_Seager2007/` (Fe via Vinet fit, MgSiO$_3$ perovskite via 4th-order Birch-Murnaghan, H$_2$O ice via experimental + DFT). Also handles the 2D variants used by RTPress100TPa and PALEOS-2phase as fallback density tables. All lookups go through SciPy interpolators with bounds extrapolation guarded by per-cell PALEOS clamping when the table carries a phase-valid range.

::: zalmoxis.eos.seager
    options:
      inherited_members: false
      show_source: true
      members:
        - get_tabulated_eos
