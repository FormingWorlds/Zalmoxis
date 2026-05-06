# EOS dispatch

The main entry points for density evaluation. `calculate_density` and `calculate_density_batch` route a single (P, T) query (or a vectorised pair of arrays) to the right EOS family based on the `EOS_REGISTRY` entry: tabulated Seager2007, T-dependent (WolfBower2018, RTPress100TPa, PALEOS-2phase), unified PALEOS, the analytic polytrope, or the Vinet EOS. The dispatcher is what `mixing.calculate_mixed_density` calls per component before the harmonic mean is taken. `PALEOS-API:*` registry entries are detected here and lazily resolved through `paleos_api_cache` on first use.

::: zalmoxis.eos.dispatch
    options:
      inherited_members: false
      show_source: true
      members:
        - calculate_density
        - calculate_density_batch
