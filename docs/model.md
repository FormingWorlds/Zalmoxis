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
| `WolfBower2018:MgSiO3` | Tabulated | MgSiO$_3$ (solid + melt) | $T$-dependent ($\leq 7\,M_\oplus$) |
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
For planets above ~2 $M_\oplus$, deep-mantle pressures near the core-mantle boundary begin to exceed this table ceiling.
The Brent pressure solver (see [Pressure Solver](#pressure-solver-brents-method)) with out-of-bounds pressure clamping handles this gracefully up to $7\,M_\oplus$: pressures beyond the table boundary are clamped to the table edge, returning the boundary density.
This approximation is acceptable for planets up to ~7 $M_\oplus$, where the clamped region is a small fraction of the mantle.
Beyond $7\,M_\oplus$, the clamped densities diverge too far from reality and the code raises a `ValueError`.
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

### By EOS type

| EOS | Pressure range | Temperature | Max planet mass | Notes |
|-----|----------------|-------------|-----------------|-------|
| `Seager2007:iron` | 0--$10^{16}$ Pa | 300 K (fixed) | ~50 $M_\oplus$ | Vinet + DFT + TFD |
| `Seager2007:MgSiO3` | 0--$10^{16}$ Pa | 300 K (fixed) | ~50 $M_\oplus$ | 4th-order BME + DFT + TFD |
| `Seager2007:H2O` | 0--$10^{16}$ Pa | 300 K (fixed) | ~50 $M_\oplus$ | Experimental + DFT + TFD |
| `WolfBower2018:MgSiO3` | 0--$10^{12}$ Pa (1 TPa) | 0--16500 K | 7 $M_\oplus$ | RTpress; $P$ clamped at table edge, $T$ out-of-bounds raises error |
| `Analytic:*` | 0--$10^{16}$ Pa | 300 K (fixed) | ~50 $M_\oplus$ | 2--12% accuracy vs. tabulated |

### General limits

- **Mass range:** The model is designed for rocky and water-rich planets in the range $\sim 0.1$--$50 \, M_{\oplus}$.
  Below $\sim 0.1 \, M_{\oplus}$, the assumption of hydrostatic equilibrium and the EOS parameterizations become unreliable.
  Above $\sim 50 \, M_{\oplus}$, the planet enters the gas-giant regime where the absence of an H/He envelope EOS limits applicability.
- **Pressure range:** The Seager et al. (2007) tabulated and analytic EOS are valid up to $P \sim 10^{16}$ Pa ($10^{10}$ GPa), which exceeds central pressures for all planets within the supported mass range.
  The WolfBower2018 tables are limited to ~1 TPa; out-of-bounds pressures are clamped to the table edge (see [EOS Physics > Wolf & Bower 2018](#wolf-bower-2018-temperature-dependent-eos)).
- **Temperature range (Wolf & Bower 2018 only):** The $P$--$T$ tables cover 0--16500 K; the code raises a `ValueError` if the requested temperature falls outside this grid.
  Out-of-bounds *pressures* are clamped to the table edge (see above), but out-of-bounds *temperatures* are not.
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
R_p \; [\mathrm{m}] = 1000 \times (7030 - 1840 \times X_{\mathrm{CMF}}) \times \left( \frac{M_p}{M_{\oplus}} \right)^{0.282}
$$

where $X_{\mathrm{CMF}}$ is the core mass fraction (`core_mass_fraction`), $M_p$ is the planet mass, and $M_{\oplus}$ is Earth's mass. The original scaling law uses the planetary iron weight fraction; for a differentiated planet with a pure-iron core, this equals the core mass fraction.

The initial guess for the core mass is:

$$
M_{\mathrm{core}} = X_{\mathrm{CMF}} \times M_p
$$

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

3. **Pressure solver (Brent's method):** Finds the central pressure $P_c$ such that the surface pressure matches the target boundary condition. See [Pressure Solver](#pressure-solver-brents-method) below.

### Pressure Solver (Brent's Method)

The central pressure $P_c$ is determined by solving a 1D root-finding problem: find $P_c$ such that the residual

$$
f(P_c) = P_{\mathrm{surface}}(P_c) - P_{\mathrm{target}}
$$

is zero, where $P_{\mathrm{surface}}(P_c)$ is the surface pressure obtained by integrating the coupled ODEs from the center outward with initial condition $P(r=0) = P_c$.

Zalmoxis uses [Brent's method](https://en.wikipedia.org/wiki/Brent%27s_method) (`scipy.optimize.brentq`) for this root-finding step.
Brent's method combines bisection, secant, and inverse quadratic interpolation: it maintains a valid bracket (like bisection, guaranteeing convergence) while accelerating with superlinear methods when possible.
This provides both robustness and speed, typically converging in 20--36 function evaluations.

**Bracket construction.**
The initial bracket $[P_{\mathrm{low}}, P_{\mathrm{high}}]$ is constructed around an empirical scaling-law estimate $\hat{P}_c$:

$$
\hat{P}_c = P_{c,\oplus} \times \left( \frac{M_p}{M_{\oplus}} \right)^{2} \times \left( \frac{R_p}{R_{\oplus}} \right)^{-4}
$$

with $P_{\mathrm{low}} = \max(10^6 \, \mathrm{Pa}, \; 0.1 \, \hat{P}_c)$ and $P_{\mathrm{high}} = 10 \, \hat{P}_c$.
When using `WolfBower2018:MgSiO3`, $P_{\mathrm{high}}$ is further capped at `max_center_pressure_guess` to prevent excessively high central pressures that would push deep-mantle pressures far beyond the 1 TPa WolfBower2018 table ceiling.

**Terminal event.**
Each ODE integration includes a terminal event that stops `solve_ivp` when the pressure crosses zero ($P \to 0^-$).
Without this event, trial central pressures far below the true value cause the integrator to grind with vanishingly small step sizes in the zero-derivative region (where `coupled_odes()` returns $[0, 0, 0]$ for non-physical pressure).
The terminal event reduces evaluation time for bad guesses from minutes to milliseconds.

**Early termination handling.**
When the terminal event fires, the ODE integration stops short of the planet surface and the solution arrays are padded with zeros.
The residual function detects this ($P_{\mathrm{surface}} \leq 0$) and returns $-P_{\mathrm{target}}$ --- a negative value that signals to Brent's method that $P_c$ is too low, maintaining a valid bracket.

**Closure state capture.**
Since `brentq` only returns the root value (not intermediate ODE solutions), the residual function uses a mutable closure dict to capture the mass, gravity, and pressure arrays from the last evaluation.
After convergence, these arrays are extracted for the density update step.

### Convergence Checks

- The outer loop converges when the relative mass difference falls below `tolerance_outer`.
- The inner loop converges when the maximum relative density change across all shells falls below `tolerance_inner`.
- The pressure solver converges when (a) `brentq` reports convergence, (b) the surface pressure residual is within `pressure_tolerance` of the target, and (c) all pressures are non-negative ($P \geq 0$, allowing for zero-padded surface points from the terminal event).

The model reports overall convergence only when all three loops have converged.

### Output

On convergence, the model returns radial profiles of gravity, pressure, density, temperature, and enclosed mass.
Derived structural parameters include: core radius, CMB pressure, central pressure, average density, core mass fraction, core radius fraction, and (if applicable) mantle phase fractions.

---

## Code Architecture

### Module overview

```
src/zalmoxis/
├── zalmoxis.py          # Orchestration: config loading, iteration loops, Brent solver, output
├── structure_model.py   # ODE system: coupled_odes(), solve_structure(), terminal events
├── eos_functions.py     # EOS dispatch: calculate_density(), Tdep phase logic, temperature profiles
├── eos_analytic.py      # Analytic modified polytrope: get_analytic_density()
├── eos_properties.py    # Material property dictionaries (file paths, unit conversions)
├── constants.py         # Physical constants (G, earth_mass, earth_radius, etc.)
└── plots/               # Visualization (profile plots, P-T phase diagrams)
```

### Data flow

```
TOML config
    │
    ▼
parse_eos_config() ──► layer_eos_config dict
    │                   {"core": "Seager2007:iron", "mantle": "WolfBower2018:MgSiO3"}
    ▼
main() ─── outer loop (radius) ─── inner loop (density) ─── brentq(_pressure_residual)
    │                                                             │
    │                                                             ▼
    │                                                     solve_structure()
    │                                                             │
    │                                                             ▼
    │                                                     solve_ivp(coupled_odes)
    │                                                             │
    │                                                             ▼
    │                                                     get_layer_eos() → calculate_density()
    │                                                             │
    │                                                      ┌──────┼──────┐
    │                                                      ▼      ▼      ▼
    │                                               Seager2007  WB2018  Analytic
    │                                              (tabulated) (Tdep)  (polytrope)
    ▼
post_processing() ──► output files + plots
```

### Key functions

- **`main()`** (`zalmoxis.py`): Orchestrates the three nested convergence loops.
  The innermost loop uses `scipy.optimize.brentq` to find the central pressure root (see [Pressure Solver](#pressure-solver-brents-method)).

- **`_pressure_residual()`** (`zalmoxis.py`, closure inside `main()`): Residual function $f(P_c) = P_{\mathrm{surface}}(P_c) - P_{\mathrm{target}}$ passed to `brentq`.
  Calls `solve_structure()` for each trial $P_c$ and captures the ODE solution via a mutable closure dict.

- **`get_layer_eos()`** (`structure_model.py`): Maps the enclosed mass at a given radial shell to the per-layer EOS string by comparing against core and core+mantle mass thresholds.
  Purely geometric --- no EOS-type branching.

- **`coupled_odes()`** (`structure_model.py`): Defines the derivatives $dM/dr$, $dg/dr$, $dP/dr$ for the ODE solver.
  Calls `calculate_density()` at each evaluation to close the system.
  Returns $[0, 0, 0]$ for non-physical states (negative pressure, invalid density) to signal the adaptive solver to reject the step and retry smaller.

- **`solve_structure()`** (`structure_model.py`): Integrates the coupled ODEs across the planetary radius using `scipy.integrate.solve_ivp` (RK45).
  Includes a terminal event that stops integration when pressure crosses zero.
  When any layer uses `WolfBower2018:MgSiO3`, the radial grid is split into two segments for numerical stability near the surface; otherwise, a single integration pass is performed.
  Pads output arrays to `len(radii)` if the terminal event truncates the integration.

- **`calculate_density()`** (`eos_functions.py`): Dispatches the per-layer EOS string to the appropriate density calculation: tabulated lookup for `Seager2007:*`, temperature-dependent phase-aware lookup for `WolfBower2018:MgSiO3`, or direct evaluation for `Analytic:*` strings.

- **`get_Tdep_density()`** (`eos_functions.py`): Computes mantle density accounting for temperature-dependent phase transitions using the solidus/liquidus melting curves and volume-additive mixing in the mush regime.
  Guards against `None` returns from out-of-bounds table lookups.

- **`calculate_temperature_profile()`** (`eos_functions.py`): Returns a callable $T(r)$ for three modes: `"isothermal"` (uniform), `"linear"` (center-to-surface gradient), or `"prescribed"` (loaded from file).

- **`parse_eos_config()`** (`zalmoxis.py`): Parses the `[EOS]` TOML section into a per-layer dictionary.
  Handles both the new per-layer format and legacy global-string format via `LEGACY_EOS_MAP`.

- **`validate_layer_eos()`** (`zalmoxis.py`): Validates that all per-layer EOS strings correspond to known tabulated or analytic options.

### Error handling in the ODE system

The adaptive ODE solver (RK45) evaluates the right-hand side at trial points that may be non-physical (e.g., negative pressure).
Two mechanisms prevent crashes:

1. **Zero-derivative guard in `coupled_odes()`:** If the pressure or density is non-physical, the function returns $[0, 0, 0]$ instead of raising an exception.
   This makes the local error estimate large, causing the solver to reject the trial step and retry with a smaller step size.

2. **Terminal event in `solve_structure()`:** A terminal event `_pressure_zero` stops the integration when $P$ crosses zero from above.
   This prevents the solver from wasting time in the zero-derivative region and enables the Brent residual function to detect that $P_c$ was too low.

These two mechanisms work together: the zero-derivative guard handles transient non-physical trial points within valid integrations, while the terminal event handles structurally too-low central pressure guesses.
