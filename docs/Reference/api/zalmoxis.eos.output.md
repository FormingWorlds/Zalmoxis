# EOS output

Append-mode writer for per-iteration pressure and density profiles. Used by the inner solver loop to record every Picard iterate so post-run plotting can show the convergence trajectory of the radial structure.

::: zalmoxis.eos.output
    options:
      inherited_members: false
      show_source: true
      members:
        - create_pressure_density_files
