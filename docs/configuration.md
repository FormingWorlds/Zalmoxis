# Configuration file

Zalmoxis uses [TOML](https://toml.io/en/) to structure its configuration file. The default is `default.toml`.

The configuration file contains various parameters required to run the planetary structure model. Below are the sections, their parameters and a brief explanation of each. 

## Configuration Sections

### [InputParameter]  
Defines the basic input parameters for the planetary model.  

- `planet_mass` - Planet mass [kg]

### [AssumptionsAndInitialGuesses]  
Contains assumptions and initial guesses for the model’s structure.  

- **core_radius_fraction** = `0.545`  # Core radius fraction of total radius
- **core_mass_fraction** = `0.32`  # Core mass fraction of total mass
- **weight_iron_fraction** = `0.35`  # Iron fraction by weight in the planet's composition

### [EOS]  
Specifies the equation of state (EOS) choice for planetary material properties. The `"Tabulated"` choice uses the Seager EOS data. The `"Birch-Murnaghan"` and `"Mie-Gruneisen-Debye"` choices are under development.

- **choice** = `"Tabulated"`  # Options: `"Birch-Murnaghan"`, `"Mie-Gruneisen-Debye"`, `"Tabulated"`

### [Calculations]  
Defines calculation settings for the planetary model.  

- **num_layers** = `150`  # Number of planet layers

### [IterativeProcess]  
Configures the iterative process settings for solving the planetary structure.  

- **max_iterations_outer** = `100`  # Maximum number of iterations for the outer loop
- **tolerance_outer** = `1e-4`  # Convergence tolerance for the outer loop
- **tolerance_radius** = `1e-4`  # Convergence tolerance for the core-mantle boundary (CMB) radius calculation in the outer loop
- **max_iterations_inner** = `100`  # Maximum number of iterations for the inner loop
- **tolerance_inner** = `1e-4`  # Convergence tolerance for the inner loop
- **relative_tolerance** = `1e-3`  # Relative tolerance for numerical solvers (e.g., `solve_ivp`)
- **absolute_tolerance** = `1e-6`  # Absolute tolerance for numerical solvers

### [PressureAdjustment]  
Adjusts the pressure at the surface based on the desired target pressure.  

- **target_surface_pressure** = `101325`  # Pa (standard atmospheric pressure, 1 atm)
- **pressure_tolerance** = `1000`  # Pa, tolerance for surface pressure matching
- **max_iterations_pressure** = `100`  # Maximum iterations for pressure adjustment
- **pressure_adjustment_factor** = `0.95`  # Reduction factor for pressure adjustment

### [Output]  
Configures output settings.  

- **data_enabled** = `true`  # Enables output of the data (e.g., planet structure)
- **plots_enabled** = `true`  # Enables output of plots