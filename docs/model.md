# Interior Structure Model

Zalmoxis solves the coupled ordinary differential equations (ODEs) of hydrostatic equilibrium for a differentiated planet with 2--3 compositionally distinct layers (iron core, silicate mantle, and optional water/ice envelope).
Given a total planet mass, layer mass fractions, and a per-layer equation of state (EOS) specification, the code iteratively determines self-consistent radial profiles of pressure, density, gravity, and enclosed mass from the center to the surface.
Three families of EOS are supported and can be mixed arbitrarily across layers: the [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) merged tabulated EOS at 300 K, the temperature-dependent [Wolf & Bower (2018)](https://www.sciencedirect.com/science/article/pii/S0031920117301449) RTpress EOS with phase-aware melt fractions, and a fast analytic modified-polytrope approximation from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346).

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
core      = "Seager2007:iron"
mantle    = "WolfBower2018:MgSiO3"
ice_layer = ""   # empty = 2-layer model
```

Layer boundaries are determined by cumulative mass fractions: a radial shell belongs to the core when $M(r) < M_{\mathrm{core}}$, to the mantle when $M_{\mathrm{core}} \le M(r) < M_{\mathrm{core}} + M_{\mathrm{mantle}}$, and to the outer ice layer (if present) otherwise.
The function `get_layer_eos()` in `structure_model.py` maps the enclosed mass at each integration step to the appropriate per-layer EOS string, which is then dispatched to `calculate_density()`.

This architecture replaced an earlier design that used a single global `EOS_CHOICE` string (e.g., `"Tabulated:iron/silicate"`) to select a fixed combination of materials for the entire planet.
The per-layer system allows arbitrary mixing of tabulated and analytic EOS across layers --- for instance, an analytic iron core with a temperature-dependent silicate mantle --- without modifying the solver.

Legacy global strings are still accepted via a backward-compatible mapping in `parse_eos_config()`.

### Valid per-layer EOS identifiers

| Identifier | Source | Material | Temperature |
|---|---|---|---|
| `Seager2007:iron` | Tabulated | Fe ($\epsilon$) | 300 K |
| `Seager2007:MgSiO3` | Tabulated | MgSiO$_3$ perovskite | 300 K |
| `Seager2007:H2O` | Tabulated | Water ice (VII/VIII/X) | 300 K |
| `WolfBower2018:MgSiO3` | Tabulated | MgSiO$_3$ (solid + melt) | $T$-dependent ($\leq 2\,M_\oplus$) |
| `Analytic:<material>` | Analytic fit | Any of 6 materials | 300 K |

---

## EOS Physics

### Seager et al. (2007) Tabulated EOS

The tabulated EOS from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) provides $\rho(P)$ at 300 K for iron (Fe $\epsilon$-phase), MgSiO$_3$ perovskite, and water ice.
The tables are constructed by merging two regimes:

- **Low pressure** ($P \lesssim 200$ GPa): Vinet EOS for iron, fourth-order Birch--Murnaghan EOS for MgSiO$_3$ and water ice, fitted to experimental data and DFT calculations.
- **High pressure** ($P \gtrsim 10^4$ GPa): Thomas--Fermi--Dirac (TFD) theory, which becomes exact in the limit of fully ionized, degenerate electron gas.

The two regimes are smoothly joined in the intermediate range.

**Limitations.** The tabulated EOS is evaluated at a fixed temperature of 300 K and does not account for thermal pressure, phase transitions (e.g., post-perovskite), compositional gradients, or partial melting.
It is appropriate for cold, fully solidified structure models.

### Wolf & Bower (2018) Temperature-Dependent EOS

The `WolfBower2018:MgSiO3` EOS implements the RTpress (Reciprocal Transform pressure) framework from [Wolf & Bower (2018)](https://www.sciencedirect.com/science/article/pii/S0031920117301449), providing $\rho(P, T)$ for MgSiO$_3$ in both the solid and liquid phases.
This EOS captures the thermal expansivity and compressibility of silicate mantle material over a wide $P$--$T$ range.

Phase boundaries are defined using solidus and liquidus melting curves derived from [Monteux et al. (2016)](https://www.sciencedirect.com/science/article/pii/S0012821X16302199), with the solidus offset by $-600$ K from the liquidus.
At each radial shell, the code evaluates the local melt fraction:

$$
f_{\mathrm{melt}} = \frac{T - T_{\mathrm{sol}}}{T_{\mathrm{liq}} - T_{\mathrm{sol}}}
$$

where $T$ is the local temperature, $T_{\mathrm{sol}}$ is the solidus temperature, and $T_{\mathrm{liq}}$ is the liquidus temperature at the local pressure.
Three phase regimes are distinguished:

- **Solid** ($T \le T_{\mathrm{sol}}$, $f_{\mathrm{melt}} \le 0$): density from the solid-state EOS table.
- **Melt** ($T \ge T_{\mathrm{liq}}$, $f_{\mathrm{melt}} \ge 1$): density from the liquid-state EOS table.
- **Mixed / mush** ($T_{\mathrm{sol}} < T < T_{\mathrm{liq}}$): density from volume-additive interpolation between solid and liquid densities:

$$
\rho_{\mathrm{mixed}} = \left[ (1 - f_{\mathrm{melt}}) \, \rho_{\mathrm{solid}}^{-1} + f_{\mathrm{melt}} \, \rho_{\mathrm{liquid}}^{-1} \right]^{-1}
$$

This EOS is appropriate when modeling hot rocky planets whose mantles may be partially or fully molten, which is critical for accurately coupling to thermal evolution codes.
A temperature profile (isothermal, linear, or prescribed from file) must be supplied.
When this EOS is selected for any layer, the radial integration is split into two segments (controlled by `adaptive_radial_fraction`) to handle the steep density gradients near the surface.

**Mass limit.** The WolfBower2018 tables cover pressures up to ~1 TPa.
For planets above $2\,M_\oplus$, deep-mantle pressures near the core-mantle boundary exceed this limit.
The code raises a `ValueError` if `WolfBower2018:MgSiO3` is used with a planet mass exceeding this threshold.
For higher-mass planets, use `Seager2007:MgSiO3` or `Analytic:MgSiO3` instead.

### Analytic Modified Polytrope (Seager et al. 2007)

The analytic EOS implements the modified polytropic fit from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346), Table 3, Eq. 11:

$$
\rho(P) = \rho_0 + c \cdot P^n
$$

where $\rho$ is density in kg/m$^3$, $P$ is pressure in Pa, and $\rho_0$, $c$, $n$ are material-specific parameters.
This closed-form expression approximates the full merged Vinet/BME + TFD EOS without requiring tabulated data files.

Accuracy is 2--5% at low ($P < 5$ GPa) and high ($P > 30$ TPa) pressures, and degrades to up to 12% at intermediate pressures where the transition between the low-pressure and TFD regimes occurs.
The fit is valid for $P < 10^{16}$ Pa.

Six materials are available:

| Material key | Compound | $\rho_0$ (kg/m$^3$) | $c$ (kg m$^{-3}$ Pa$^{-n}$) | $n$ |
|---|---|---|---|---|
| `iron` | Fe ($\epsilon$) | 8300 | 0.00349 | 0.528 |
| `MgSiO3` | MgSiO$_3$ perovskite | 4100 | 0.00161 | 0.541 |
| `MgFeSiO3` | (Mg,Fe)SiO$_3$ | 4260 | 0.00127 | 0.549 |
| `H2O` | Water ice (VII/VIII/X) | 1460 | 0.00311 | 0.513 |
| `graphite` | C (graphite) | 2250 | 0.00350 | 0.514 |
| `SiC` | Silicon carbide | 3220 | 0.00172 | 0.537 |

Like the Seager et al. (2007) tabulated EOS, this assumes 300 K and no phase transitions.
Because any of the six materials can be assigned to any structural layer, the analytic EOS enables modeling of exotic compositions (e.g., carbon planets with iron/SiC or iron/graphite layers) that are not available in the tabulated data.

---

## Validity Ranges

- **Mass range:** The model is designed for rocky and water-rich planets in the range $\sim 0.1$--$50 \, M_{\oplus}$.
  Below $\sim 0.1 \, M_{\oplus}$, the assumption of hydrostatic equilibrium and the EOS parameterizations become unreliable.
  Above $\sim 50 \, M_{\oplus}$, the planet enters the gas-giant regime where the absence of an H/He envelope EOS limits applicability.
- **Pressure range:** The tabulated and analytic EOS are valid up to $P \sim 10^{16}$ Pa ($10^{10}$ GPa), which exceeds central pressures for all planets within the supported mass range.
  The code emits a warning if pressures exceed this limit.
- **Temperature range (Wolf & Bower 2018 only):** The $P$--$T$ tables have finite extent; the code raises an error if the requested temperature falls outside the tabulated grid.
  The Seager et al. (2007) EOS (both tabulated and analytic) is evaluated at a fixed 300 K and carries no temperature dependence.
- **Composition:** All EOS assume single-component layers with sharp compositional boundaries (no mixing gradients across interfaces).

---

## Process Flow

### Configuration Loading

The function `load_zalmoxis_config()` reads a TOML configuration file that specifies planet mass, layer mass fractions, per-layer EOS identifiers, temperature profile settings, solver tolerances, and output options.
The `[EOS]` section is parsed by `parse_eos_config()`, which accepts both the new per-layer format and the legacy global-string format.

### Initial Parameter Setup

The initial guess for the planet radius follows the scaling law from [Noack et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020A%26A...638A.129N/abstract):

$$
R_p \; [\mathrm{m}] = 1000 \times (7030 - 1840 \times X_{\mathrm{Fe}}) \times \left( \frac{M_p}{M_{\oplus}} \right)^{0.282}
$$

where $X_{\mathrm{Fe}}$ is the weight iron fraction, $M_p$ is the planet mass, and $M_{\oplus}$ is Earth's mass.

The initial guess for the core mass is:

$$
M_{\mathrm{core}} = X_{\mathrm{CMF}} \times M_p
$$

where $X_{\mathrm{CMF}}$ is the core mass fraction.

The initial guess for the central pressure is based on an empirical scaling:

$$
P_c = P_{c,\oplus} \times \left( \frac{M_p}{M_{\oplus}} \right)^{2} \times \left( \frac{R_p}{R_{\oplus}} \right)^{-4}
$$

where $P_{c,\oplus}$ is Earth's central pressure and $R_{\oplus}$ is Earth's radius.

### Iterative Solution

The model uses three nested convergence loops:

1. **Outer loop (mass convergence):** Updates the total planet radius estimate by comparing the calculated total mass against the target mass and rescaling: $R_p \leftarrow R_p \times (M_{\mathrm{target}} / M_{\mathrm{calculated}})^{1/3}$.

2. **Inner loop (density profile convergence):** For each radial shell, recalculates the density from the local pressure (and temperature, if applicable) using the per-layer EOS returned by `get_layer_eos()`.
   Density updates are damped by averaging with the previous iteration.

3. **Innermost loop (pressure adjustment):** Integrates the coupled ODEs via `solve_structure()` and adjusts the central pressure guess to match the target surface pressure.

### Convergence Checks

- The outer loop converges when the relative mass difference falls below `tolerance_outer`.
- The inner loop converges when the maximum relative density change across all shells falls below `tolerance_inner`.
- The pressure adjustment loop converges when the surface pressure is within `pressure_tolerance` of the target and all pressures remain positive.

The model reports overall convergence only when all three loops have converged.

### Output

On convergence, the model returns radial profiles of gravity, pressure, density, temperature, and enclosed mass.
Derived structural parameters include: core radius, CMB pressure, central pressure, average density, core mass fraction, core radius fraction, and (if applicable) mantle phase fractions.

---

## Key Functions

- **`get_layer_eos()`** (`structure_model.py`): Maps the enclosed mass at a given radial shell to the per-layer EOS string by comparing against core and core+mantle mass thresholds.

- **`coupled_odes()`** (`structure_model.py`): Defines the derivatives $dM/dr$, $dg/dr$, $dP/dr$ for the ODE solver.
  Calls `calculate_density()` at each evaluation to close the system.

- **`solve_structure()`** (`structure_model.py`): Integrates the coupled ODEs across the planetary radius.
  When any layer uses `WolfBower2018:MgSiO3`, the radial grid is split into two segments for numerical stability near the surface; otherwise, a single integration pass is performed.

- **`calculate_density()`** (`eos_functions.py`): Dispatches the per-layer EOS string to the appropriate density calculation: tabulated lookup for `Seager2007:*` and `Seager2007:H2O`, temperature-dependent phase-aware lookup for `WolfBower2018:MgSiO3`, or direct evaluation for `Analytic:*` strings.

- **`get_Tdep_density()`** (`eos_functions.py`): Computes mantle density accounting for temperature-dependent phase transitions using the solidus/liquidus melting curves and volume-additive mixing in the mush regime.

- **`calculate_temperature_profile()`** (`eos_functions.py`): Returns a callable $T(r)$ for three modes: `"isothermal"` (uniform), `"linear"` (center-to-surface gradient), or `"prescribed"` (loaded from file).

- **`parse_eos_config()`** (`zalmoxis.py`): Parses the `[EOS]` TOML section into a per-layer dictionary.
  Handles both the new per-layer format and legacy global-string format via `LEGACY_EOS_MAP`.

- **`validate_layer_eos()`** (`zalmoxis.py`): Validates that all per-layer EOS strings correspond to known tabulated or analytic options.
