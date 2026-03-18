# Interior Structure Model

Zalmoxis solves the coupled ordinary differential equations (ODEs) of hydrostatic equilibrium for a differentiated planet with 2--3 compositionally distinct layers (iron core, silicate mantle, and optional water/ice envelope).
Given a total planet mass, layer mass fractions, and a per-layer equation of state (EOS) specification, the code iteratively determines self-consistent radial profiles of pressure, density, gravity, and enclosed mass from the center to the surface.

Several EOS families are supported (see [Equations of State](eos_physics.md) for detailed physics):

- **[PALEOS](eos_physics.md#paleos-iron-mgsio3-h2o)** (recommended): all stable phases per material in a single file, $T$-dependent with $\nabla_{\mathrm{ad}}$, up to 50 $M_\oplus$
- [PALEOS-2phase](eos_physics.md#paleos-2phase-mgsio3-only): earlier variant with separate solid/liquid tables (MgSiO$_3$ only)
- [Seager et al. (2007)](eos_physics.md#seager-et-al-2007-tabulated-eos): merged tabulated EOS at 300 K
- [Wolf & Bower (2018)](eos_physics.md#wolf-bower-2018-temperature-dependent-eos): temperature-dependent RTpress EOS ($\leq 7\,M_\oplus$)
- [RTPress100TPa](eos_physics.md#rtpress100tpa-extended-melt-eos): extended melt table to 100 TPa
- [Analytic polytrope](eos_physics.md#analytic-modified-polytrope-seager-et-al-2007): fast closed-form approximation, no data files

Layers can contain [multiple materials mixed by volume additivity](mixing.md), with phase-aware suppression of non-condensed volatiles.

---

## Governing Equations

The internal structure model assumes hydrostatic and thermodynamic equilibrium within each layer.
Starting from the center and integrating outward, the code solves the following coupled ODEs using `scipy.integrate.solve_ivp` (explicit Runge--Kutta, RK45):

$$
\frac{dM}{dr} = 4 \pi r^2 \rho
$$

$$
\frac{dg}{dr} = 4 \pi G \rho - \frac{2g}{r}
$$

$$
\frac{dP}{dr} = -\rho \, g
$$

where $M(r)$ is the enclosed mass within radius $r$, $\rho(r)$ is the local density, $g(r)$ is the gravitational acceleration, $P(r)$ is the pressure, and $G$ is the gravitational constant.

The density $\rho(P)$ --- or $\rho(P, T)$ when a temperature-dependent EOS is used --- closes the system at each radial shell.

---

## Per-Layer EOS Architecture

### Configuration format

Each structural layer is assigned its own EOS via a string of the form `"<source>:<composition>"` in the `[EOS]` section of the TOML configuration file:

```toml
[EOS]
core      = "PALEOS:iron"
mantle    = "PALEOS:MgSiO3"
ice_layer = ""   # empty = 2-layer model
```

Layer boundaries are determined by cumulative mass fractions: a radial shell belongs to the core when $M(r) < M_{\mathrm{core}}$, to the mantle when $M_{\mathrm{core}} \le M(r) < M_{\mathrm{core}} + M_{\mathrm{mantle}}$, and to the outer ice layer (if present) otherwise.
The function `get_layer_eos()` in `structure_model.py` maps the enclosed mass at each integration step to the appropriate per-layer EOS string, which is then dispatched to `calculate_density()`.

This architecture replaced an earlier design that used a single global `EOS_CHOICE` string (e.g., `"Tabulated:iron/silicate"`) to select a fixed combination of materials for the entire planet.
The per-layer system allows arbitrary mixing of tabulated and analytic EOS across layers --- for instance, an analytic iron core with a temperature-dependent silicate mantle --- without modifying the solver.

Each layer can also contain multiple materials mixed by volume additivity.
The config format uses `+` to combine materials with mass fractions: `"PALEOS:MgSiO3:0.85+PALEOS:H2O:0.15"`.
See the [multi-material mixing](mixing.md) page for the full mixing model, including the phase-aware density suppression that prevents non-condensed volatiles from dominating the mixture.
In the PROTEUS ecosystem, mixing fractions are set by CALLIOPE (solubility model) or PROTEUS (volatile trapping in the mantle) at runtime via `LayerMixture.update_fractions()`.

Legacy global strings are still accepted via a backward-compatible mapping in `parse_eos_config()`.

### Valid per-layer EOS identifiers

| Identifier | Source | Material | Temperature |
|---|---|---|---|
| **`PALEOS:iron`** | **PALEOS** | **Fe (5 phases)** | **$T$-dependent ($\leq 50\,M_\oplus$), includes $\nabla_{\mathrm{ad}}$** |
| **`PALEOS:MgSiO3`** | **PALEOS** | **MgSiO$_3$ (6 phases)** | **$T$-dependent ($\leq 50\,M_\oplus$), includes $\nabla_{\mathrm{ad}}$** |
| **`PALEOS:H2O`** | **PALEOS** | **H$_2$O (7 EOS)** | **$T$-dependent ($\leq 50\,M_\oplus$), includes $\nabla_{\mathrm{ad}}$** |
| `PALEOS-2phase:MgSiO3` | PALEOS (2-phase variant) | MgSiO$_3$ (solid + liquid) | $T$-dependent ($\leq 50\,M_\oplus$), includes $\nabla_{\mathrm{ad}}$ |
| `Seager2007:iron` | Tabulated | Fe ($\epsilon$) | 300 K |
| `Seager2007:MgSiO3` | Tabulated | MgSiO$_3$ perovskite | 300 K |
| `Seager2007:H2O` | Tabulated | Water ice (VII/VIII/X) | 300 K |
| `WolfBower2018:MgSiO3` | Tabulated | MgSiO$_3$ (solid + melt) | $T$-dependent ($\leq 7\,M_\oplus$) |
| `RTPress100TPa:MgSiO3` | Tabulated | MgSiO$_3$ (solid + melt) | $T$-dependent ($\leq 50\,M_\oplus$) |
| `Analytic:<material>` | Analytic fit | Any of 6 materials | 300 K |

See [Equations of State](eos_physics.md) for detailed physics of each EOS family.

---

## Validity Ranges

### By EOS type

| EOS | Pressure range | Temperature | Max planet mass | Notes |
|-----|----------------|-------------|-----------------|-------|
| `Seager2007:iron` | 0--$10^{16}$ Pa | 300 K (fixed) | ~50 $M_\oplus$ | Vinet + DFT + TFD |
| `Seager2007:MgSiO3` | 0--$10^{16}$ Pa | 300 K (fixed) | ~50 $M_\oplus$ | 4th-order BME + DFT + TFD |
| `Seager2007:H2O` | 0--$10^{16}$ Pa | 300 K (fixed) | ~50 $M_\oplus$ | Experimental + DFT + TFD |
| `WolfBower2018:MgSiO3` | 0--$10^{12}$ Pa (1 TPa) | 0--16500 K | 7 $M_\oplus$ | RTpress; $P$ clamped at table edge, $T$ out-of-bounds raises error |
| `RTPress100TPa:MgSiO3` | $10^3$--$10^{14}$ Pa (100 TPa) | 400--50000 K | 50 $M_\oplus$ | Extended melt table; solid from WB2018 (clamped at 1 TPa) |
| `PALEOS-2phase:MgSiO3` | 1 bar--100 TPa | 300--100000 K | 50 $M_\oplus$ | Solid + liquid with $\nabla_{\mathrm{ad}}$; 75%/53% grid fill |
| `PALEOS:iron` | 1 bar--100 TPa | 300--100000 K | 50 $M_\oplus$ | Unified 5-phase Fe with $\nabla_{\mathrm{ad}}$ |
| `PALEOS:MgSiO3` | 1 bar--100 TPa | 300--100000 K | 50 $M_\oplus$ | Unified 6-phase MgSiO$_3$ with $\nabla_{\mathrm{ad}}$ |
| `PALEOS:H2O` | 1 bar--100 TPa | 100--100000 K | 50 $M_\oplus$ | Unified 7-EOS H$_2$O with $\nabla_{\mathrm{ad}}$ |
| `Analytic:*` | 0--$10^{16}$ Pa | 300 K (fixed) | ~50 $M_\oplus$ | 2--12% accuracy vs. tabulated |

### General limits

- **Mass range:** The model is designed for rocky and water-rich planets in the range $\sim 0.1$--$50 \, M_{\oplus}$.
  Below $\sim 0.1 \, M_{\oplus}$, the assumption of hydrostatic equilibrium and the EOS parameterizations become unreliable.
  Above $\sim 50 \, M_{\oplus}$, the planet enters the gas-giant regime where the absence of an H/He envelope EOS limits applicability.
- **Pressure range:** The Seager et al. (2007) tabulated and analytic EOS are valid up to $P \sim 10^{16}$ Pa ($10^{10}$ GPa), which exceeds central pressures for all planets within the supported mass range.
  The WolfBower2018 tables are limited to ~1 TPa; out-of-bounds pressures are clamped to the table edge (see [EOS Physics > Wolf & Bower 2018](eos_physics.md#wolf-bower-2018-temperature-dependent-eos)).
- **Temperature range (Wolf & Bower 2018 only):** The $P$--$T$ tables cover 0--16500 K; the code raises a `ValueError` if the requested temperature falls outside this grid.
  Out-of-bounds *pressures* are clamped to the table edge (see above), but out-of-bounds *temperatures* are not.
  The Seager et al. (2007) EOS (both tabulated and analytic) is evaluated at a fixed 300 K and carries no temperature dependence.
- **Composition:** Multi-material volume-additive mixing is supported within layers (see [Multi-material mixing](mixing.md)). Layer boundaries (core/mantle/ice) remain sharp. Non-condensed volatile components are smoothly suppressed to prevent unphysical density deflation.
