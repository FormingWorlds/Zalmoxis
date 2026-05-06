# Temperature-dependent EOS

Density and phase routing for the temperature-dependent EOS families: WolfBower2018 RTPress on MgSiO$_3$ to 1 TPa, RTPress100TPa for the extended-pressure melt, and PALEOS-2phase (separate solid and liquid MgSiO$_3$ tables with metastable extensions). `get_Tdep_density` evaluates per-phase tables and blends them across the mushy zone using external solidus / liquidus curves; `get_Tdep_material` returns the active phase label; `_get_paleos_nabla_ad` returns $(\partial \ln T/\partial \ln P)_S$ for the adiabatic integration. Melting curve loaders are in `load_melting_curve` and `get_solidus_liquidus_functions`.

::: zalmoxis.eos.tdep
    options:
      inherited_members: false
      show_source: true
      members:
        - load_melting_curve
        - get_solidus_liquidus_functions
        - get_Tdep_density
        - get_Tdep_material
        - _get_paleos_nabla_ad
