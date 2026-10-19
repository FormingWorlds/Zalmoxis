# EOS table consistency

Thermodynamic identities of a regular (log10 P, log10 T) table, written as relative residuals that vanish for an exact equation of state: `cp = T (ds/dT)_P`, the Maxwell relation `(ds/dP)_T = -alpha / rho`, `alpha = -(1/rho) (drho/dT)_P`, `(du/dT)_P = cp - P alpha / rho`, `nabla_ad = P alpha / (rho cp)` and `cp - cv = T alpha^2 / (rho kappa_T)`. The derivatives are central differences in ln P and ln T with a step of `k` grid cells; a stencil counts only when all its cells hold numbers of one phase, so a phase boundary or a missing cell never enters a derivative. The input is a table in the layout of `zalmoxis.eos_export.load_paleos_all_properties`.

::: zalmoxis.eos.consistency
    options:
      inherited_members: false
      show_source: true
      members:
        - residuals
        - derivative
        - segments
