# Energetics

Gravitational binding energies and the initial post-accretion thermal state of a rocky planet, following White & Li (2025, JGRP). Computes $U_u = 3GM^2/5R$ for an undifferentiated reference, the differentiated binding energy from the actual $\rho(r)$, the differentiation energy $U_d - U_u$, and assembles the initial CMB temperature $T_{\mathrm{CMB}} = T_{\mathrm{eq}} + \Delta T_G + \Delta T_D + \Delta T_{\mathrm{ad}}$. Used by PROTEUS to seed a self-consistent post-accretion state before the atmosphere module equilibrates the surface.

::: zalmoxis.energetics
    options:
      inherited_members: false
      show_source: true
      members:
        - gravitational_binding_energy
        - gravitational_binding_energy_uniform
        - differentiation_energy
        - initial_thermal_state
