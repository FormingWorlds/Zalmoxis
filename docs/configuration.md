# Configuration file

Zalmoxis uses [TOML](https://toml.io/en/) to structure its configuration file. The default is `default.toml`.

The configuration file contains various parameters required to run the planetary structure model. Below are the sections, their parameters and a brief explanation of each. 

## Configuration Sections

### `InputParameter`  
Defines the basic input parameters for the planetary model.  

- `planet_mass` - Planet mass [kg].

### `AssumptionsAndInitialGuesses`
Contains assumptions and initial guesses for the model’s structure.  

- `core_mass_fraction` - Core mass fraction of total mass.
- `weight_iron_fraction` - Weight iron fraction.

### `EOS`  
Specifies the equation of state (EOS) choice for planetary material properties. The `"Tabulated"` choice uses the Seager EOS data. The `"Birch-Murnaghan"` and `"Mie-Gruneisen-Debye"` choices are under development.

- `choice` - Choices: `"Birch-Murnaghan"`, `"Mie-Gruneisen-Debye"`, `"Tabulated"`.

### `Calculations`
Defines calculation settings for the planetary model.  

- `num_layers` - Number of planet layers.

### `IterativeProcess`
Configures the iterative process settings for solving the planetary structure.  

- `max_iterations_outer` - Maximum number of iterations for the outer loop.
- `tolerance_outer` - Convergence tolerance for the outer loop.
- `tolerance_radius` - Convergence tolerance for the core-mantle boundary radius calculation in the outer loop.
- `max_iterations_inner` - Maximum number of iterations for the inner loop.
- `tolerance_inner` - Convergence tolerance for the inner loop.
- `relative_tolerance` - Relative tolerance for `solve_ivp`.
- `absolute_tolerance` - Absolute tolerance for `solve_ivp`.

### `PressureAdjustment` 
Adjusts the pressure at the surface based on the desired target pressure.  

- `target_surface_pressure` - Target surface pressure [Pa].
- `pressure_tolerance` - Tolerance for surface pressure matching [Pa].
- `max_iterations_pressure` - Maximum iterations for pressure adjustment.
- `pressure_adjustment_factor` - Reduction factor for pressure adjustment.

### `Output` 
Configures output settings.  

- `data_enabled` - Enables output of the data. 
- `plots_enabled` - Enables output of plots.