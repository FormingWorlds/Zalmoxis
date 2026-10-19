# EOS table consistency

Thermodynamic identities of a regular (log10 P, log10 T) table, written as relative residuals that vanish for an exact equation of state: `cp = T (ds/dT)_P`, the Maxwell relation `(ds/dP)_T = -alpha / rho`, `alpha = -(1/rho) (drho/dT)_P`, `(du/dT)_P = cp - P alpha / rho`, `nabla_ad = P alpha / (rho cp)` and `cp - cv = T alpha^2 / (rho kappa_T)`. The derivatives are central differences in ln P and ln T with a step of `k` grid cells; a stencil counts only when all its cells hold numbers of one phase, so a phase boundary or a missing cell never enters a derivative. The input is a table in the layout of `zalmoxis.eos_export.load_paleos_all_properties`.

`check` takes a table on a grid uniform in log10 P and log10 T and tests each cell that lies 4 or more steps from an edge, a phase boundary or a missing cell along the axis of the derivative: a derivative identity fails where the Richardson residual `R_x = (4 R_h - R_2h) / 3` exceeds the round-off floor of the stored values plus twice the h^4 term measured at that cell, `(R_x(2h) - R_x(h)) / 15`; `nabla_ad` is held to its floor alone. `floors` gives that floor for a stored relative precision `eps` (`5e-9` for 9 significant digits). `signs` flags cells with a non-positive density or heat capacity, `cv` above `cp`, or a density that falls with pressure, at the same floor; each condition tests only the cells that hold its values.

Resolution at 9 stored digits, measured at one cell of an exact Murnaghan table at 150 points per decade (P = 46 MPa, T = 1,581 K): a single stored value off by `2e-8` (relative) or more fails at the cells one step from it, and by `3e-7` (entropy) or `1e-6` (density) or more also at the cells two steps from it. The exact ideal gas and Murnaghan tables give no failing cell at this precision. `check` raises a smaller `eps` to `EPS_FLOOR = 5e-9`, so every table, also one stored in full double precision, is checked at the 9-digit floor, and faults below about `2e-8` (relative) are out of reach on any table.

Detection of a `1e-6` fault depends on the grid: on the Aragog P-T layout (`check_pt`), a planted `1e-6` entropy or density was found one and two steps away on the 200 x 200 grid that PROTEUS writes, but two such faults were missed on a 60 x 60 grid, where the measured h^4 term is larger. Tables that eos_export writes by linear interpolation of the PALEOS tables hold the identities only to the interpolation error: with the PROTEUS settings (2-phase MgSiO3, 200 x 200, P up to 350 GPa) 86 to 97 % of the tested cells fail, with residuals of about `1e-4`. `tools/validation/eos_table_consistency_report.py` writes every failing cell of a PALEOS-format table to a CSV file.

::: zalmoxis.eos.consistency
    options:
      inherited_members: false
      show_source: true
      members:
        - check
        - check_pt
        - floors
        - signs
        - residuals
        - derivative
        - segments
