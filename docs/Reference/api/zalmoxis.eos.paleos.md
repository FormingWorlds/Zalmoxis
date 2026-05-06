# PALEOS unified density

Density and adiabatic gradient lookup for the unified PALEOS table format, in which every stable phase of a material (Fe with five phases, MgSiO$_3$ with six phases, H$_2$O with seven phases) lives in a single P-T table tagged by a phase column. The phase boundary used for mushy-zone blending is extracted at load time from that column, removing the need for an external melting curve. `get_paleos_unified_density` and its batch variant handle the five branches of the mushy-zone logic (no mushy zone, P out of liquidus coverage, above liquidus, below solidus, mushy interior). `_get_paleos_unified_nabla_ad` returns $(\partial \ln T/\partial \ln P)_S$ used by the adiabatic temperature integration.

::: zalmoxis.eos.paleos
    options:
      inherited_members: false
      show_source: true
      members:
        - get_paleos_unified_density
        - get_paleos_unified_density_batch
        - _get_paleos_unified_nabla_ad
