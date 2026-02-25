# Configuration file

Zalmoxis uses [TOML](https://toml.io/en/) to structure its configuration file. The default is `default.toml` in the `input/` directory.

The configuration file defines all parameters needed to run the planetary interior structure model: planet properties, equation of state (EOS) selection for each structural layer, numerical solver settings, and output options. The sections below document each parameter.

## Configuration sections

### `InputParameter`

Defines the basic input for the planetary model.

| Parameter | Type | Unit | Description |
|---|---|---|---|
| `planet_mass` | float | Earth masses | Total planet mass ($M_\oplus = 5.972 \times 10^{24}$ kg). |

### `AssumptionsAndInitialGuesses`

Initial guesses and assumptions for the planetary structure.

| Parameter | Type | Unit | Description |
|---|---|---|---|
| `core_mass_fraction` | float | -- | Core mass as a fraction of total mass. Also used in the Seager et al. (2007) scaling relation for the initial radius guess. Earth: ~0.325. |
| `mantle_mass_fraction` | float | -- | Mantle mass as a fraction of total mass. Set to 0 for a 2-layer model where the mantle fills the remainder (1 - `core_mass_fraction`). Must be > 0 when using a 3-layer model with an ice layer. |
| `temperature_mode` | string | -- | Temperature profile type. One of: `"isothermal"`, `"linear"`, `"prescribed"`. See [Temperature profiles](#temperature-profiles). |
| `surface_temperature` | float | K | Surface temperature. Used when `temperature_mode` is `"isothermal"` or `"linear"`. |
| `center_temperature` | float | K | Central temperature. Used when `temperature_mode` is `"linear"`. |
| `temperature_profile_file` | string | -- | Filename (relative to `input/`) containing a prescribed radial temperature profile. Used when `temperature_mode` is `"prescribed"`. |

#### Temperature profiles

Temperature profiles are only relevant when using a temperature-dependent EOS (i.e., `WolfBower2018:MgSiO3`). For all other EOS choices, a fixed 300 K is assumed internally and these parameters are ignored.

- **`"isothermal"`**: Constant temperature equal to `surface_temperature` at all radii.
- **`"linear"`**: Linear interpolation from `center_temperature` at $r = 0$ to `surface_temperature` at $r = R$.
- **`"prescribed"`**: Reads a temperature profile from `temperature_profile_file`. The file must contain one temperature value per line (center to surface), with the same number of entries as `num_layers`.

---

### `EOS`

Specifies the equation of state for each structural layer. Each layer is configured independently using a `"<source>:<composition>"` string.

```toml
[EOS]
core      = "Seager2007:iron"
mantle    = "WolfBower2018:MgSiO3"
ice_layer = ""  # Empty string = 2-layer model
```

**Fields:**

| Field | Required | Description |
|---|---|---|
| `core` | Yes | EOS for the innermost layer (iron core). |
| `mantle` | Yes | EOS for the mantle layer. |
| `ice_layer` | No | EOS for an outer volatile layer. If empty or omitted, the model uses a 2-layer structure (core + mantle). If set, the model uses a 3-layer structure and `mantle_mass_fraction` must be > 0 in `[AssumptionsAndInitialGuesses]`. |

#### Naming convention

All EOS identifiers follow the format `<source>:<composition>`, where:

- **`<source>`** identifies the data source or method: `Seager2007` (tabulated), `WolfBower2018` (tabulated, temperature-dependent), or `Analytic` (closed-form fit, no data files needed).
- **`<composition>`** identifies the material: `iron`, `MgSiO3`, `MgFeSiO3`, `H2O`, `graphite`, `SiC`.

#### Available EOS options

| EOS string | Source | Composition | Temperature | Data files required | Notes |
|---|---|---|---|---|---|
| `Seager2007:iron` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) | Fe (epsilon phase) | Fixed 300 K | Yes | Vinet EOS + DFT, tabulated $\rho(P)$. |
| `Seager2007:MgSiO3` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) | MgSiO3 perovskite | Fixed 300 K | Yes | 4th-order Birch-Murnaghan + DFT, tabulated $\rho(P)$. |
| `Seager2007:H2O` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) | Water ice (phases VII, VIII, X) | Fixed 300 K | Yes | Experimental + DFT, tabulated $\rho(P)$. |
| `WolfBower2018:MgSiO3` | [Wolf & Bower (2018)](https://www.sciencedirect.com/science/article/pii/S0031920117301449) | MgSiO3 (melt + solid) | T-dependent | Yes | RTpress EOS with phase-aware melting. **Limited to $\leq 7\,M_\oplus$** (table max ~1 TPa; out-of-bounds pressures clamped). Requires `temperature_mode` configuration. Uses [Monteux et al.](https://doi.org/10.1016/j.epsl.2016.05.010) solidus/liquidus curves. |
| `Analytic:iron` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) Table 3 | Fe (epsilon) | Fixed 300 K | No | Modified polytrope: $\rho(P) = \rho_0 + c \cdot P^n$. |
| `Analytic:MgSiO3` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) Table 3 | MgSiO3 perovskite | Fixed 300 K | No | Modified polytrope. |
| `Analytic:MgFeSiO3` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) Table 3 | (Mg,Fe)SiO3 | Fixed 300 K | No | Modified polytrope. |
| `Analytic:H2O` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) Table 3 | Water ice | Fixed 300 K | No | Modified polytrope. |
| `Analytic:graphite` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) Table 3 | C (graphite) | Fixed 300 K | No | Modified polytrope. |
| `Analytic:SiC` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) Table 3 | Silicon carbide | Fixed 300 K | No | Modified polytrope. |

**Analytic EOS details.** The analytic options use the modified polytropic fit from Seager et al. (2007, Eq. 11, Table 3):

$$\rho(P) = \rho_0 + c \cdot P^n$$

where $\rho_0$ is the zero-pressure density, $c$ and $n$ are fitted constants. This approximation is valid for $P < 10^{16}$ Pa and reproduces the full tabulated EOS to 2--12% accuracy across all planetary pressures. The analytic EOS requires no external data files, making it useful for quick exploration and testing.

**Temperature-dependent EOS.** The `WolfBower2018:MgSiO3` EOS is the only temperature-dependent option. It uses separate tabulated $\rho(P, T)$ grids for solid and melt phases, with a linear melt-fraction interpolation between the solidus and liquidus. When this EOS is assigned to any layer, the `temperature_mode`, `surface_temperature`, and `center_temperature` parameters in `[AssumptionsAndInitialGuesses]` become active.

!!! warning "Mass limit for WolfBower2018"
    The WolfBower2018 tables cover pressures up to ~1 TPa. For planets above ~2 $M_\oplus$, deep-mantle pressures begin to exceed this limit. The Brent pressure solver with out-of-bounds clamping handles this gracefully up to $7\,M_\oplus$. Beyond $7\,M_\oplus$, clamped densities become unreliable and the code raises a `ValueError`. Use `Seager2007:MgSiO3` or `Analytic:MgSiO3` for higher-mass planets.

#### EOS decision guide

**Use tabulated EOS** (`Seager2007:*`, `WolfBower2018:*`) when:

- Accuracy matters (e.g., comparison with published mass-radius relations).
- Data files are available (downloaded via `get_data.sh` or equivalent).
- Temperature-dependent phase behavior is needed (mantle melting).

**Use analytic EOS** (`Analytic:*`) when:

- Running quick parameter sweeps or exploratory calculations.
- Data files are not yet downloaded or not available.
- Exotic compositions are needed (graphite, SiC, MgFeSiO3) that have no tabulated counterpart.
- Testing or debugging the structure solver.

**Mixing tabulated and analytic EOS** across layers is fully supported. For example, a tabulated iron core with an analytic silicate mantle is a valid configuration.

#### Configuration examples

**Earth-like planet** (iron core + silicate mantle, 2-layer):
```toml
[EOS]
core      = "Seager2007:iron"
mantle    = "Seager2007:MgSiO3"
ice_layer = ""
```

**Earth-like with temperature-dependent mantle** (iron core + T-dependent silicate, 2-layer):
```toml
[EOS]
core      = "Seager2007:iron"
mantle    = "WolfBower2018:MgSiO3"
ice_layer = ""
```
Requires `temperature_mode`, `surface_temperature`, and `center_temperature` to be set in `[AssumptionsAndInitialGuesses]`.

**Mercury-like planet** (large iron core + (Mg,Fe)SiO3 mantle, analytic):
```toml
[AssumptionsAndInitialGuesses]
core_mass_fraction = 0.68
# ...

[EOS]
core      = "Analytic:iron"
mantle    = "Analytic:MgFeSiO3"
ice_layer = ""
```

**Water world** (iron core + silicate mantle + water ice layer, 3-layer):
```toml
[AssumptionsAndInitialGuesses]
core_mass_fraction   = 0.25
mantle_mass_fraction = 0.50
# ...

[EOS]
core      = "Seager2007:iron"
mantle    = "Seager2007:MgSiO3"
ice_layer = "Seager2007:H2O"
```
Note: `mantle_mass_fraction` must be > 0 for a 3-layer model.

**Carbon planet** (iron core + SiC mantle):
```toml
[EOS]
core      = "Analytic:iron"
mantle    = "Analytic:SiC"
ice_layer = ""
```

**Carbon planet with graphite mantle**:
```toml
[EOS]
core      = "Analytic:iron"
mantle    = "Analytic:graphite"
ice_layer = ""
```

**Mixed-EOS model** (tabulated core + analytic mantle):
```toml
[EOS]
core      = "Seager2007:iron"
mantle    = "Analytic:MgSiO3"
ice_layer = ""
```

**Pure iron sphere** (single-composition, core fills entire planet):
```toml
[AssumptionsAndInitialGuesses]
core_mass_fraction   = 1.0
mantle_mass_fraction = 0
# ...

[EOS]
core      = "Analytic:iron"
mantle    = "Analytic:iron"
ice_layer = ""
```
Both `core` and `mantle` must be set even for a single-composition model. The mantle EOS is still evaluated but occupies zero mass.

#### Legacy configuration (backward compatibility)

The previous single-field `choice` syntax is still recognized and auto-expanded to the per-layer format:

| Legacy `choice` value | Expands to |
|---|---|
| `"Tabulated:iron/silicate"` | `core = "Seager2007:iron"`, `mantle = "Seager2007:MgSiO3"` |
| `"Tabulated:iron/Tdep_silicate"` | `core = "Seager2007:iron"`, `mantle = "WolfBower2018:MgSiO3"` |
| `"Tabulated:water"` | `core = "Seager2007:iron"`, `mantle = "Seager2007:MgSiO3"`, `ice_layer = "Seager2007:H2O"` |
| `"Analytic:Seager2007"` | Per-layer analytic EOS from `core_material`, `mantle_material`, `water_layer_material` fields |

Example of the old format (still works but deprecated):
```toml
[EOS]
choice = "Tabulated:iron/silicate"
```

Example of the old analytic format (still works but deprecated):
```toml
[EOS]
choice               = "Analytic:Seager2007"
core_material        = "iron"
mantle_material      = "MgSiO3"
water_layer_material = ""
```

New configurations should use the per-layer format. The legacy format may be removed in a future version.

---

### `Calculations`

Numerical grid settings.

| Parameter | Type | Unit | Description |
|---|---|---|---|
| `num_layers` | int | -- | Number of radial grid points. Higher values increase resolution and accuracy at the cost of computation time. Default: 150. |

---

### `IterativeProcess`

Controls the three nested iteration loops (outer mass convergence, inner density convergence, Brent pressure solver) and the ODE solver tolerances.

| Parameter | Type | Unit | Default | Description |
|---|---|---|---|---|
| `max_iterations_outer` | int | -- | 100 | Maximum iterations for the outer loop (total mass convergence). |
| `tolerance_outer` | float | -- | 3e-3 | Relative tolerance for outer loop convergence ($\lvert M_\mathrm{calc} - M_\mathrm{target} \rvert / M_\mathrm{target}$). |
| `max_iterations_inner` | int | -- | 100 | Maximum iterations for the inner loop (density profile convergence). |
| `tolerance_inner` | float | -- | 1e-4 | Relative tolerance for inner loop density convergence. |
| `relative_tolerance` | float | -- | 1e-5 | Relative tolerance for `solve_ivp` (ODE integrator). |
| `absolute_tolerance` | float | -- | 1e-6 | Absolute tolerance for `solve_ivp`. |
| `maximum_step` | float | m | 250000 | Maximum radial step size for `solve_ivp`. |
| `adaptive_radial_fraction` | float | -- | 0.98 | Fraction (0--1) of the radial domain where `solve_ivp` uses adaptive stepping before switching to fixed steps. Primarily relevant for the `WolfBower2018:MgSiO3` EOS near the surface. |
| `max_center_pressure_guess` | float | Pa | 10e12 | Upper bound on the Brent solver's pressure bracket ($P_{\mathrm{high}}$). Prevents excessively high central pressures that would push deep-mantle pressures far beyond the WolfBower2018 table ceiling (~1 TPa). Only active when `WolfBower2018:MgSiO3` is used; the Seager2007 EOS tables extend to much higher pressures, so this cap is not needed for pure-Seager runs. |

#### Guidance on reasonable parameter ranges

The default values work well for Earth-mass planets with the default EOS. For other configurations:

- **Super-Earths (1--10 $M_\oplus$):** The defaults generally suffice. For masses above ~5 $M_\oplus$, consider tightening `tolerance_outer` to 1e-4 and increasing `num_layers` to 200--300. The Brent pressure solver converges in 20--36 evaluations regardless of planet mass.
- **Sub-Earths (< 1 $M_\oplus$):** May converge faster. Defaults are conservative.
- **Temperature-dependent EOS (`WolfBower2018:MgSiO3`):** Limited to $\leq 7\,M_\oplus$. If convergence is slow, try reducing `maximum_step` (e.g., to 100000 m) and ensuring `adaptive_radial_fraction` is close to 1.0 (e.g., 0.98--0.99). The `max_center_pressure_guess` caps the Brent solver's upper bracket to prevent deep-mantle pressures from exceeding the WolfBower2018 table by too much. The default (10 TPa) is sufficient for planets up to 7 $M_\oplus$.
- **Analytic EOS:** Convergence is typically fast and robust. Looser tolerances and fewer layers are usually sufficient for exploration.
- **3-layer models:** Adding an ice layer increases the number of density discontinuities. Consider increasing `num_layers` to 200+ for smooth profiles.

---

### `PressureAdjustment`

Controls the Brent root-finding solver that determines the central pressure.
The solver finds $P_c$ such that the surface pressure after ODE integration matches the target.
See the [model documentation](model.md#pressure-solver-brents-method) for details on the algorithm.

| Parameter | Type | Unit | Default | Description |
|---|---|---|---|---|
| `target_surface_pressure` | float | Pa | 101325 | Target pressure at the planetary surface. Default is 1 atm. |
| `pressure_tolerance` | float | Pa | 1e9 | Convergence criterion: the solver iterates until $\lvert P_\mathrm{surface} - P_\mathrm{target} \rvert$ falls below this value. |
| `max_iterations_pressure` | int | -- | 200 | Maximum number of function evaluations for the Brent solver. Typical convergence requires 20--36 evaluations. |
| `pressure_adjustment_factor` | float | -- | 1.1 | **Deprecated.** Retained for backward compatibility. The Brent solver does not use this parameter. |

---

### `Output`

Controls what the model writes after a run.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_enabled` | bool | true | Write the computed radial profiles (radius, density, gravity, pressure, temperature, enclosed mass) to a text file in `output_files/`. |
| `plots_enabled` | bool | false | Generate profile plots after the run. |
| `verbose` | bool | false | Log detailed convergence diagnostics and warnings. When false, only essential messages (final results, errors) are shown. |
| `iteration_profiles_enabled` | bool | false | Write pressure and density profiles for every iteration to files. Useful for debugging convergence behavior; produces large output. |
