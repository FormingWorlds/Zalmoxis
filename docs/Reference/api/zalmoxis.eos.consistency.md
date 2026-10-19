# EOS table consistency

Thermodynamic identities of a regular (log10 P, log10 T) table, written as relative residuals that vanish for an exact equation of state: `cp = T (ds/dT)_P`, the Maxwell relation `(ds/dP)_T = -alpha / rho`, `alpha = -(1/rho) (drho/dT)_P`, `(du/dT)_P = cp - P alpha / rho`, `nabla_ad = P alpha / (rho cp)` and `cp - cv = T alpha^2 / (rho kappa_T)`. The derivatives are central differences in ln P and ln T with a step of `k` grid cells; a stencil counts only when all its cells hold numbers of one phase, so a phase boundary or a missing cell never enters a derivative. The input is a table in the layout of `zalmoxis.eos_export.load_paleos_all_properties`.

`check` tests every cell: a derivative identity fails where the Richardson residual `R_x = (4 R_h - R_2h) / 3` exceeds the round-off floor of the stored values plus twice the h^4 term measured at that cell, `(R_x(2h) - R_x(h)) / 15`; `nabla_ad` is held to its floor alone. `floors` gives that floor for a stored relative precision `eps` (`5e-9` for 9 significant digits). `signs` flags cells with a non-positive density or heat capacity, `cv` above `cp`, or a density that falls with pressure.

Resolution at 9 stored digits, measured at one cell of an exact Murnaghan table at 150 points per decade (P = 46 MPa, T = 1,581 K): a single stored value off by `2e-8` (relative) or more fails at the cells one step from it, and by `3e-7` (entropy) or `1e-6` (density) or more also at the cells two steps from it. The exact ideal gas and Murnaghan tables give no failing cell at this precision.

::: zalmoxis.eos.consistency
    options:
      inherited_members: false
      show_source: true
      members:
        - check
        - floors
        - signs
        - residuals
        - derivative
        - segments
