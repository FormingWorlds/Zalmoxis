# Configuration file

Zalmoxis uses [TOML](https://toml.io/en/) to structure its configuration file. The default is `default.toml`.

The configuration file contains various parameters required to run the planetary structure model. Below are the sections, their parameters and a brief explanation of each.

## Configuration Sections

### `InputParameter`
Defines the basic input parameters for the planetary model.

- `planet_mass` - Planet mass [Earth masses].

### `AssumptionsAndInitialGuesses`
Contains assumptions and initial guesses for the model's structure.

- `core_mass_fraction` - Core mass fraction of total mass.
- `mantle_mass_fraction` - Mantle mass fraction of total mass.
- `weight_iron_fraction` - Weight iron fraction in the planet.
- `temperature_mode` - Choice of input temperature profile: "isothermal", "linear", "prescribed".
- `surface_temperature` - Surface temperature (K), required for temperature_mode="isothermal" or "linear".
- `center_temperature` - Central temperature (K), required for temperature_mode="linear".
- `temperature_profile_file` - Filename containing a prescribed temperature profile, required for temperature_mode="prescribed".

The last four parameters (`temperature_mode`, `surface_temperature`, `center_temperature`, and `temperature_profile_file`) are only needed when using the `"Tabulated:iron/Tdep_silicate"` EOS.

### `EOS`
Specifies the equation of state (EOS) choice for planetary material properties.

- `choice` - The EOS model to use. Available choices:

    - **`"Tabulated:iron/silicate"`** — [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) tabulated data at 300 K for 2-layer planets (iron core + MgSiO3 mantle). Requires data download.

    - **`"Tabulated:iron/Tdep_silicate"`** — [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) iron core + [Wolf and Bower (2018)](https://www.sciencedirect.com/science/article/pii/S0031920117301449) temperature-dependent silicate mantle with phase-aware melting. Requires data download and temperature profile configuration.

    - **`"Tabulated:water"`** — [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) tabulated data at 300 K for 3-layer planets (iron core + MgSiO3 mantle + H2O ice). Requires data download.

    - **`"Analytic:Seager2007"`** — Analytic modified polytropic EOS from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) Table 3, Eq. 11. No data download needed. Supports 6 materials as configurable per-layer choices. Fixed at 300 K. Accurate to 2–12% vs the full tabulated EOS.

#### Per-layer material configuration (Analytic:Seager2007 only)

When using `choice = "Analytic:Seager2007"`, the following fields configure which material is used for each structural layer:

- `core_material` — Material for the core layer. Required. One of: `iron`, `MgSiO3`, `MgFeSiO3`, `H2O`, `graphite`, `SiC`.
- `mantle_material` — Material for the mantle layer. Required. Same options as above.
- `water_layer_material` — Material for the outer layer. Optional. If non-empty, enables a 3-layer model (requires `mantle_mass_fraction > 0`). Same options as above. Leave empty (`""`) for a 2-layer model.

**Example configurations:**

| Planet type | `core_material` | `mantle_material` | `water_layer_material` |
|---|---|---|---|
| Earth-like | `iron` | `MgSiO3` | |
| Mercury-like | `iron` | `MgFeSiO3` | |
| Water world | `iron` | `MgSiO3` | `H2O` |
| Carbon planet (SiC) | `iron` | `SiC` | |
| Carbon planet (graphite) | `iron` | `graphite` | |
| Pure iron | `iron` | `iron` | |

### `Calculations`
Defines calculation settings for the planetary model.

- `num_layers` - Number of planet layers.

### `IterativeProcess`
Configures the iterative process settings for solving the planetary structure.

- `max_iterations_outer` - Maximum number of iterations for the outer loop.
- `tolerance_outer` - Convergence tolerance for the outer loop.
- `max_iterations_inner` - Maximum number of iterations for the inner loop.
- `tolerance_inner` - Convergence tolerance for the inner loop.
- `relative_tolerance` - Relative tolerance for `solve_ivp`.
- `absolute_tolerance` - Absolute tolerance for `solve_ivp`.
- `maximum_step` - Maximum integration step size for solve_ivp [m].
- `adaptive_radial_fraction` - Fraction (0–1) of the radial domain defining where `solve_ivp` transitions from adaptive integration to fixed-step integration when using the temperature-dependent `"Tabulated:iron/Tdep_silicate"` EOS.

### `PressureAdjustment`
Adjusts the pressure at the surface based on the desired target pressure.

- `target_surface_pressure` - Target surface pressure [Pa].
- `pressure_tolerance` - Convergence tolerance for surface pressure convergence [Pa].
- `max_iterations_pressure` - Maximum iterations for pressure adjustment loop.
- `pressure_adjustment_factor` - Adjustment factor for updating the central pressure guess.

### `Output`
Configures output settings.

- `data_enabled` - Flag to enable saving data to a file (true/false).
- `plots_enabled` - Flag to enable plotting the results in post_processing() (true/false).
- `verbose` - If true, logs detailed convergence info and warnings; if false, only essential messages are shown (true/false).
- `iteration_profiles_enabled` - If true, writes pressure and density profiles for each iteration to files (true/false).
