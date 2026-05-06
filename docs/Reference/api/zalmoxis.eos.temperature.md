# Temperature profile

Computes the temperature column of the structure profile. `calculate_temperature_profile` dispatches on the configured mode (`isothermal`, `linear`, `prescribed`, `adiabatic`); `compute_adiabatic_temperature` integrates an adiabat using native EOS gradient tables, with two anchor choices: `anchor='surface'` (default, historic) sets $T(R) = T_{\mathrm{surf}}$ and integrates inward, while `anchor='cmb'` sets $T(r_{\mathrm{cmb}}) = T_{\mathrm{cmb}}$ and integrates outward through the mantle only (used when an interior energetics solver downstream of Aragog or SPIDER already constrains the CMB temperature). For multi-material layers `nabla_ad` is mass-fraction-weighted across components.

::: zalmoxis.eos.temperature
    options:
      inherited_members: false
      show_source: true
      members:
        - compute_adiabatic_temperature
        - calculate_temperature_profile
        - _compute_paleos_dtdp
