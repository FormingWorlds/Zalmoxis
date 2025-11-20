# Configuration file

Zalmoxis uses [TOML](https://toml.io/en/) to structure its configuration file. The default is `default.toml`.

The configuration file contains various parameters required to run the planetary structure model. Below are the sections, their parameters and a brief explanation of each. 

## Configuration Sections

### `InputParameter`  
Defines the basic input parameters for the planetary model.  

- `planet_mass` - Planet mass [Earth masses].

### `AssumptionsAndInitialGuesses`
Contains assumptions and initial guesses for the model’s structure.  

- `core_mass_fraction` - Core mass fraction of total mass.
- 'mantle_mass_fraction' - Mantle mass fraction of total mass.
- `weight_iron_fraction` - Weight iron fraction in the planet.
- `temperature_mode` - Choice of input temperature profile: "isothermal", "linear", "prescribed".
- `surface_temperature` - Surface temperature (K), required for temperature_mode="isothermal" or "linear".
- `center_temperature` - Central temperature (K), required for temperature_mode="linear".
- `temperature_profile_file` - Filename containing a prescribed temperature profile, required for temperature_mode="prescribed".

The last four parameters (`temperature_mode`, `surface_temperature`, `center_temperature`, and `temperature_profile_file`) are only needed when using the `"Tabulated:iron/Tdep_silicate"` EOS.

### `EOS`  
Specifies the equation of state (EOS) choice for planetary material properties. The `"Tabulated:iron/silicate"` and `"Tabulated:water"` choices use the [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) EOS data for modeling two types of planets: `"Tabulated:iron/silicate"` for super-Earths (with iron cores and MgSiO3 mantles) and `"Tabulated:water"` for water planets (with iron cores, silicate mantles and an outer water ice layer). The `"Tabulated:iron/Tdep_silicate"` choice uses the [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) iron EOS for the core and the [Wolf and Bower (2018)](https://www.sciencedirect.com/science/article/pii/S0031920117301449) high P-T RTpress MgSiO3 EOS for the mantle. Unlike the [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) EOS, which provides pressure–density data at 300 K, the [Wolf and Bower (2018)](https://www.sciencedirect.com/science/article/pii/S0031920117301449) EOS provides pressure–density data that explicitly depends on temperature. This allows the mantle to be modeled as solid, partially molten, or fully molten, given an input temperature profile.

- `choice` - Choices: `"Tabulated:iron/silicate"`, `"Tabulated:iron/Tdep_silicate"`, `"Tabulated:water"`.

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