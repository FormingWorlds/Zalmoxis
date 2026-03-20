# Binodal (Miscibility) Suppression

This page documents the binodal suppression models used in Zalmoxis for multi-component mixtures involving molecular hydrogen.
When H$_2$ is mixed with silicate (MgSiO$_3$) or water (H$_2$O) in a mantle layer, the binodal determines whether H$_2$ is thermodynamically miscible with its partner at each local $(P, T)$ condition.
Below the binodal temperature, H$_2$ is immiscible and should not contribute to the structural density.
Above the binodal, H$_2$ is miscible and participates in the harmonic mean.

For configuration syntax, see the [configuration guide](../How-to/configuration.md#binodal-miscibility-suppression-parameters-h2-containing-mixtures-only).
For the density-based sigmoid suppression (the other half of the total suppression), see the [mixing documentation](mixing.md).

---

## Motivation

In sub-Neptune interiors, H$_2$ can be dissolved into the silicate magma at high temperatures and pressures.
This is a fundamentally different physical regime from the traditional core-envelope structure: instead of a sharp boundary between a rocky interior and a gaseous envelope, the composition varies continuously with depth.
The binodal (miscibility boundary) defines the $(P, T)$ conditions where the single-phase miscible state becomes thermodynamically unstable and separates into two coexisting phases (H$_2$-rich gas and silicate-rich melt).

Without binodal suppression, the density-based sigmoid alone would include H$_2$ in the harmonic mean whenever its density exceeds the critical density (~31 kg/m$^3$), even at temperatures where H$_2$ and silicate are immiscible and should exist as separate phases.
The binodal sigmoid adds a physically motivated temperature criterion.

---

## Total suppression weight

For an H$_2$ component in a mixture, the total suppression weight is:

$$
\sigma_{\mathrm{total}} = \sigma_{\mathrm{density}}(\rho_{\mathrm{H_2}}) \times \sigma_{\mathrm{binodal}}(P, T)
$$

where:

- $\sigma_{\mathrm{density}}$ is the density-based sigmoid with per-component parameters (`condensed_rho_min` = 30 kg/m$^3$, `condensed_rho_scale` = 10 kg/m$^3$ for H$_2$).
- $\sigma_{\mathrm{binodal}}$ is the temperature-based sigmoid centered on the binodal temperature $T_b(P)$:

$$
\sigma_{\mathrm{binodal}} = \frac{1}{1 + \exp\!\left( -\frac{T - T_b}{T_{\mathrm{scale}}} \right)}
$$

The `binodal_T_scale` parameter (default 50 K) controls the transition width.

When multiple binodal models apply (H$_2$ mixed with both silicate and water), the most restrictive suppression weight (the minimum) is used.
For non-H$_2$ components, $\sigma_{\mathrm{binodal}} = 1$ always.

---

## Rogers+2025: H$_2$-MgSiO$_3$ binodal

### Source

[Rogers, Young & Schlichting (2025)](https://doi.org/10.1093/mnras/stae2268), MNRAS 544, 3496. Analytic fit to the binodal temperature $T_b(x_{\mathrm{H_2}}, P)$ from Eqs. A1-A11. Based on DFT-MD calculations of H$_2$-MgSiO$_3$ miscibility from [Gilmore & Stixrude (2025)](https://doi.org/10.1038/s41586-024-08509-7) and asymmetric Margules parameters from [Stixrude & Gilmore (2025)](https://doi.org/10.1016/j.icarus.2025.116401).

### Physics

The binodal defines the phase boundary in $(T, P, x_{\mathrm{H_2}})$ space between:

- **Above the binodal**: a single miscible supercritical phase where H$_2$ and MgSiO$_3$ mix at the molecular level.
- **Below the binodal**: two immiscible phases (H$_2$-rich gas and silicate-rich melt) that separate gravitationally.

The binodal temperature is computed from two generalized logistic branches: an ascending branch (small $x_{\mathrm{H_2}}$, rising toward $T_c$) and a descending branch (large $x_{\mathrm{H_2}}$, falling from $T_c$). Their natural crossing occurs near the critical mole fraction $x_c \approx 0.739$. The implementation evaluates both branches at every composition and takes the minimum, $T_b = \min(T_{\mathrm{asc}}, T_{\mathrm{desc}})$, which produces a smooth peak without an artificial kink at $x_c$.

### Pressure dependence

The critical temperature $T_c$ at the binodal peak decreases linearly with pressure:

$$
T_c(P) = 4223 \times \left(1 - \frac{P}{35\,\mathrm{GPa}}\right) \; \mathrm{K}
$$

| Pressure | Peak $T_b$ | Interpretation |
|---|---|---|
| 1 GPa | ~4100 K | Shallow mantle: H$_2$ miscible only at very high $T$ |
| 10 GPa | ~3020 K | Mid-mantle: miscibility requires moderate $T$ |
| 35 GPa | 0 K | Always miscible at any temperature |

Above 35 GPa, the binodal vanishes entirely, and H$_2$ and MgSiO$_3$ are always miscible regardless of temperature. This makes the binodal primarily relevant in the upper mantle of sub-Neptunes.

### Composition dependence

The binodal temperature depends on the H$_2$ mole fraction in the H$_2$-MgSiO$_3$ binary system.
Mass fractions from the config (e.g., 0.03 for 3% H$_2$ by mass) are converted to mole fractions internally using the molar masses ($\mu_{\mathrm{H_2}} = 2.016$ g/mol, $\mu_{\mathrm{MgSiO_3}} = 100.39$ g/mol).
A 3% mass fraction of H$_2$ corresponds to a mole fraction of ~0.60.

### Implementation

The function `rogers2025_binodal_temperature(x_H2, P_GPa)` returns the binodal temperature for a given composition and pressure. The suppression weight function `rogers2025_suppression_weight(P_Pa, T_K, w_H2, w_sil, T_scale)` converts mass fractions to mole fractions, evaluates the binodal, and returns the sigmoid weight.

---

## Gupta+2025: H$_2$-H$_2$O critical curve

### Source

[Gupta, Stixrude & Schlichting (2025)](https://doi.org/10.3847/2041-8213/adb8f5), ApJL 982, L35. Gibbs free energy model with asymmetric Margules mixing parameters (Eqs. A2-A9, Table 1).

### Physics

The model provides the Gibbs free energy of mixing for the H$_2$-H$_2$O binary system:

$$
G_{\mathrm{mix}} = RT \left[ y \ln y + (1-y) \ln(1-y) \right] + W \cdot y \cdot (1-y)
$$

where $y$ is a transformed composition variable accounting for asymmetric mixing (Eq. A3), and $W(T, P)$ is the Margules interaction parameter:

$$
W = W_H - T \cdot W_S + P \cdot W_V(T)
$$

The critical pressure $P_c(T)$ above which the system is always single-phase is given by Eq. A8. The critical temperature at a given pressure is obtained by inverting this relationship.

### The 647 K floor

The Gupta+2025 Margules model is fitted to supercritical DFT-MD calculations and is valid only when both H$_2$ and H$_2$O are in supercritical or fluid states. Below 647 K (the critical temperature of pure H$_2$O), water condenses to liquid or ice. In this regime:

- The Margules parameters ($W_V$ and $\lambda$ have $1/T^2$ and $1/T$ dependencies) extrapolate into physically meaningless territory.
- H$_2$ and condensed H$_2$O are always immiscible, regardless of what the Margules model predicts.

A floor of 647 K is imposed on the critical temperature returned by the model. If the Margules-derived critical temperature falls below 647 K, it is replaced by 647 K. This guarantees that H$_2$ is always suppressed below the H$_2$O condensation temperature.

### Performance

The critical temperature $T_c(P)$ is needed at every ODE step (~150,000 evaluations per structure solve). Rather than calling the Brent root finder each time, a lookup table of 2000 log-spaced pressure points from 1 MPa to ~3 TPa is precomputed at module import time. Each subsequent evaluation is a single `np.interp` call, falling back to Brent only for out-of-range pressures.

### Implementation

The function `gupta2025_critical_temperature(P_GPa)` returns the critical temperature for a given pressure (with the 647 K floor applied). The suppression weight function `gupta2025_suppression_weight(P_Pa, T_K, w_H2, w_H2O, T_scale)` evaluates the critical temperature, applies the floor, and returns the sigmoid weight.

The function `gupta2025_coexistence_compositions(T, P_GPa)` computes the coexisting H$_2$ mole fractions from a common-tangent construction on the Gibbs energy surface. This is available for diagnostic purposes but is not used in the mixing hot path.

---

## Configuration

All binodal suppression is configured in the `[EOS]` section of the TOML file.
The only user-facing parameter is `binodal_T_scale`:

```toml
[EOS]
mantle          = "PALEOS:MgSiO3:0.97+Chabrier:H:0.03"
binodal_T_scale = 50.0    # K, sigmoid width for binodal transition
```

The binodal models are activated automatically when `Chabrier:H` coexists with MgSiO$_3$ or H$_2$O in the same layer. No additional configuration is needed.

For a sub-Neptune configuration example, see [configuration example 4](../How-to/configuration.md#configuration-examples).

---

## Validation

The binodal suppression has been validated against:

- Reproduction of the Rogers+2025 Figure A1 binodal curve.
- Convergence of the structure solver for 5 $M_\oplus$ planets with 3% H$_2$ by mass in the mantle.
- Correct behavior at limiting cases: pure endmembers ($x_{\mathrm{H_2}} = 0$ or $1$), high pressure (always miscible above 35 GPa for H$_2$-MgSiO$_3$), and low temperature (always immiscible below 647 K for H$_2$-H$_2$O).
