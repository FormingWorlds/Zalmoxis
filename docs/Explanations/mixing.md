# Multi-Material Mixing

This page documents how Zalmoxis handles layers containing multiple materials, including the phase-aware density suppression that prevents non-condensed volatiles from dominating the mixture.
For configuration syntax, see the [configuration guide](../How-to/configuration.md#multi-material-mixing).
For the governing equations and layer architecture, see the [model overview](model.md).

---

## Volume-additive mixing

When a layer contains multiple materials (e.g., `"PALEOS:MgSiO3:0.85+PALEOS:H2O:0.15"`), the density at each radial shell is computed from the individual component densities at the local $(P, T)$.
Single-component layers use the component's density directly with no mixing overhead.

For multi-component layers, the standard volume-additive (ideal mixing) harmonic mean is:

$$
\rho_{\mathrm{mix}} = \left( \sum_i \frac{w_i}{\rho_i} \right)^{-1}
$$

where $w_i$ are mass fractions and $\rho_i(P, T)$ is each component's density evaluated independently from its own EOS table.
This assumes that partial specific volumes add linearly (no excess volume of mixing).

---

## Harmonic mean sensitivity to low-density components

The harmonic mean is dominated by the lightest component: even a small mass fraction of a low-density material can reduce $\rho_{\mathrm{mix}}$ dramatically.
At high temperatures and low pressures (near the planetary surface), H$_2$O transitions from condensed phases (liquid, ice) to vapor or low-density supercritical fluid with $\rho \sim 10$ to $100$ kg/m$^3$.
In a rock-water mixture with 85% MgSiO$_3$ ($\rho \sim 4000$ kg/m$^3$) and 15% H$_2$O ($\rho \sim 50$ kg/m$^3$), the standard harmonic mean gives $\rho_{\mathrm{mix}} \approx 311$ kg/m$^3$, far below the rock density that should dominate structurally.
The low-density vapor effectively inflates the planet's radius, producing non-physical results (e.g., $R \sim 24\,R_\oplus$ for a 10 $M_\oplus$ planet that should be $R \sim 2\,R_\oplus$).

Iron ($\rho > 7000$ kg/m$^3$) and MgSiO$_3$ ($\rho > 2500$ kg/m$^3$) exist only in condensed phases within the PALEOS tables and never trigger this problem.
Only H$_2$O (and, in future, other volatiles like CO$_2$, NH$_3$, H$_2$, He) has gas-phase states in the EOS tables.

---

## Sigmoid-weighted suppression of non-condensed components

![Sigmoid suppression function](../img/sigmoid_suppression.png)
*Condensed weight $\sigma_i$ as a function of component density. Low-density phases (vapor, supercritical gas) are smoothly suppressed; condensed phases (liquid, solid, rock, iron) are fully included. The sigmoid center is set near the H$_2$O critical density (322 kg/m$^3$).*

To prevent non-condensed volatiles from dominating the harmonic mean, each component's contribution is weighted by a smooth sigmoid function of its density:

$$
\sigma_i = \frac{1}{1 + \exp\!\left( -\frac{\rho_i - \rho_{\mathrm{min}}}{\rho_{\mathrm{scale}}} \right)}
$$

where $\rho_{\mathrm{min}}$ is the sigmoid center and $\rho_{\mathrm{scale}}$ controls the transition width.
The suppressed harmonic mean becomes:

$$
\rho_{\mathrm{mix}} = \frac{\sum_i w_i \, \sigma_i}{\sum_i w_i \, \sigma_i / \rho_i}
$$

This formulation has two key properties:

1. **When all $\sigma_i \approx 1$** (all components condensed), the expression reduces exactly to the standard harmonic mean.
   For iron ($\rho > 7000$) and MgSiO$_3$ ($\rho > 2500$), the sigmoid returns $\sigma \approx 1.0$ to machine precision, so existing single-material and all-condensed configurations produce numerically identical results.

2. **When $\sigma_i \to 0$** (component is vapor-like), both the numerator and denominator lose the $i$-th term proportionally, and the component drops out smoothly.
   There is no discontinuity, which is important for the adaptive ODE solver (RK45) that requires smooth right-hand-side functions.

The same sigmoid weighting is applied to the adiabatic gradient $\nabla_{\mathrm{ad}}$: components that are suppressed in the density calculation are also suppressed in the temperature profile calculation, maintaining internal consistency.

---

## Sigmoid parameters

The default parameters are calibrated for H$_2$O:

| Parameter | Default | Physical meaning |
|---|---|---|
| `condensed_rho_min` | 322 kg/m$^3$ | Sigmoid center = H$_2$O critical density (647 K, 22.1 MPa) |
| `condensed_rho_scale` | 50 kg/m$^3$ | Transition width; $\sigma$ goes from 0.02 to 0.98 over a range of about 390 kg/m$^3$ (~8$\times$ the scale) |

Behavior at representative densities:

| Density (kg/m$^3$) | Phase example | $\sigma$ |
|---|---|---|
| 10 | H$_2$O vapor at 1 bar, 3000 K | 0.002 |
| 100 | H$_2$O low-density supercritical | 0.012 |
| 322 | H$_2$O critical density (sigmoid center) | 0.500 |
| 500 | Dense supercritical H$_2$O | 0.972 |
| 1000 | Liquid H$_2$O at high $P$ | 1.000 |
| 4000 | MgSiO$_3$ | 1.000 |
| 13000 | Fe | 1.000 |

The defaults are appropriate for H$_2$O, the only volatile with gas-phase data in the current EOS tables.
Once other volatiles are added, $\rho_{\mathrm{min}}$ must be adjusted per material (CO$_2$: ~470, NH$_3$: ~225, He: ~70, H$_2$: ~30 kg/m$^3$).
Both parameters are user-configurable in the `[EOS]` section of the TOML file (see [configuration](../How-to/configuration.md#phase-aware-mixing-parameters-multi-material-only)).

---

## Physical interpretation and known limitations

The suppressed harmonic mean is a pragmatic approximation, not a thermodynamically rigorous mixing model.
Several limitations should be understood when interpreting results:

### Mass non-conservation

When a component is suppressed ($\sigma \to 0$), its mass is excluded from the structural density calculation.
This is equivalent to treating vapor-phase volatiles as having negligible structural contribution, effectively discarding their mass from the hydrostatic equilibrium.
The approximation is valid when the suppressed fraction is small and the vapor does not contribute meaningfully to hydrostatic support.
For a mantle with 15% H$_2$O by mass, suppression primarily affects the outermost shells where pressure is low enough for H$_2$O to be vapor-like; at depth, the H$_2$O is condensed and fully included.

### Temperature profile consistency

The same suppression is applied to $\nabla_{\mathrm{ad}}$: when a component is suppressed in the density, its adiabatic gradient is also suppressed.
This means the temperature profile ignores the thermodynamic contribution of vapor-phase volatiles.
In reality, vapor affects heat transport and the temperature gradient.
This approximation is self-consistent within the model (the density and temperature calculations "see" the same effective mixture) but differs from the physical situation where vapor is present and thermodynamically active.

### Density-based criterion avoids phase-label ambiguity

Unlike approaches that classify each EOS grid cell as "condensed" or "non-condensed" using phase labels, the sigmoid operates on density alone.
This avoids ambiguity in the supercritical regime, where phase labels like "supercritical" span a wide range of densities: supercritical H$_2$O at 10 GPa and 3000 K has $\rho \sim 1200$ to $1400$ kg/m$^3$ (dense, liquid-like, correctly included), while supercritical H$_2$O at 1 bar and 3000 K has $\rho \sim 1$ kg/m$^3$ (gas-like, correctly suppressed).
A density-based criterion handles this naturally without consulting phase tables.

---

## Extension to volatile-rich and sub-Neptune interiors

The sigmoid framework is forward-compatible with miscible sub-Neptune interiors (e.g., [Young et al. 2024](https://ui.adsabs.harvard.edu/abs/2024PSJ.....5..268Y/abstract), [2025](https://ui.adsabs.harvard.edu/abs/2025PSJ.....6..251Y/abstract)), where rock-volatile composition varies continuously with depth.
Future extensions can:

- Replace the fixed sigmoid parameters with per-component values or physics-based miscibility weights from DFT-MD calculations.
- Add depth-dependent fractions via the `LayerMixture.update_fractions()` interface (already supported).
- Introduce non-ideal mixing corrections (excess volume) in `calculate_mixed_density()` without changing the upstream solver code.
