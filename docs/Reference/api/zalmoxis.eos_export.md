# EOS export (Zalmoxis to SPIDER / Aragog)

Converts PALEOS pressure-temperature tables into the entropy-pressure (P-S) coordinates that SPIDER expects, and into the rectangular P-T grids that Aragog expects. Generates phase boundaries via direct PALEOS S(P, T) lookup (no inversion), and full 2D tables for density, temperature, heat capacity, thermal expansion, and adiabatic gradient. Solid and melt are written to separate files so SPIDER's mushy-zone mixing has phase-pure endpoints. Also produces the surface-entropy anchor consumed by SPIDER's interior solver.

::: zalmoxis.eos_export
    options:
      inherited_members: false
      show_source: true
      members:
        - load_paleos_all_properties
        - generate_spider_phase_boundaries
        - generate_spider_eos_tables
        - generate_aragog_pt_tables
        - generate_aragog_pt_tables_2phase
        - compute_surface_entropy
        - compute_entropy_adiabat
