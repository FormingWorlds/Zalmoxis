# Interior Structure Model

## Overview
The model calculates the internal structure of an exoplanet based on its mass and various other parameters, including temperature, pressure and density profiles. The solution is derived iteratively and the model currently uses a simplified equation of state (EOS) for the core and mantle from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346). The model currently supports simulations of Earth-like exoplanets of up to 50 Earth masses.

## Main function
The `main` function runs the exoplanet interior structure model. It reads the configuration file, initializes parameters and performs an iterative solution to calculate the planet's internal structure. The model outputs various parameters such as the calculated planet mass, planet radius, core radius, mantle density at the Core Mantle Boundary (CMB), core density at the CMB, pressure at the CMB, pressure at center, average density, CMB mass fraction and core radius fraction.

### Parameters
- `temp_config_path` (optional): Path to the configuration file. If not provided, the default configuration file is chosen.
- `id_mass` (optional): Identifier for the planet mass in Earth masses.

### Process Overview
1. **Configuration File Loading**:
   - The function reads a configuration file that defines the initial guesses, assumptions and iterative process parameters.

2. **Initial Parameter Setup**:
   - The initial guess for the planet's radius is set based on the scaling law in [Noack et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...638A.129N/abstract) as:

   $$
   R_p[\text{m}] = 1000 \times (7030 - 1840 \times X_{\text{Fe}}) \times \left( \frac{M_p}{M_{\text{Earth}}} \right)^{0.282}
   $$
    where \( X_{\text{Fe}} \) is the weight iron fraction, \( M_p \) is the planet mass, \( M_{\text{Earth}} \) is Earth's mass

   - The initial guess for the planet's core radius is set as:
   $$
   R_{\text{core}} = X_{\text{CMF}} \times R_p[\text{m}]
   $$

   where \( X_{\text{CMF}} \) is the core mass fraction and \( R_p[\text{m}] \) is the guessed planet radius from above.

3. **Iterative Solution**:
   - The model iteratively adjusts the planet's radius and core-mantle boundary (CMB) using an outer loop.
   - In the inner loop, the model iteratively adjusts the planet's density profile, using `solve_ivp` to solve the coupled ordinary differential equations (ODEs) for mass, gravity, and pressure as a function of radius.

4. **Pressure Adjustment**:
   - The pressure profile is adjusted iteratively to match the target surface pressure, with each adjustment made using a scaling factor.

5. **Convergence Checks**:
   - Both the outer and inner loops check for convergence based on the relative differences in mass and radius for the planet, as well as the density profile.

6. **Output Generation**:
   - Once the solution has converged, the final radius, core radius, and other calculated parameters are printed.
   - The model also saves the data to a file and optionally generates plots.

### Key Variables and Parameters
- **Planet Parameters**:
  - `planet_mass`: Mass of the exoplanet in kilograms.
  - `core_radius_fraction`: Initial guess for the core radius as a fraction of the total planet radius.
  - `core_mass_fraction`: Initial guess for the core mass as a fraction of the total planet mass.
  - `weight_iron_fraction`: Initial guess for the iron weight fraction in the core.

- **Equation of State (EOS)**:
  - `EOS_CHOICE`: Specifies the choice of the equation of state (e.g., "Birch-Murnaghan", "Mie-Gruneisen-Debye", "Tabulated").

- **Iterative Process Parameters**:
  - `max_iterations_outer`, `tolerance_outer`, `max_iterations_inner`, etc., define the iterative limits and tolerances for the outer and inner loops.

- **Pressure Adjustment Parameters**:
  - `target_surface_pressure`: Target surface pressure for the planet.
  - `pressure_adjustment_factor`: Adjustment factor for pressure recalculations.

- **Output Control**:
  - `data_output_enabled`: Flag to enable saving output data to a file.
  - `plotting_enabled`: Flag to enable plotting the results.

### Function Flow

1. **Initial Setup**:
   - The configuration file is read, and initial guesses for planet parameters are set.
   - The `radius_guess` is calculated using a scaling law from Noack et al. (2020), which depends on the planet's mass and the weight fraction of iron in the core.

2. **Outer Iteration Loop**:
   - In each outer iteration, the function sets up a radial grid (`radii`) and initializes arrays for density, mass enclosed, gravity, and pressure.
   - The `cmb_mass` (core mass) is estimated, and an initial guess for the pressure at the center is set.
   - A simplified initial density profile is assigned based on the core and mantle assumptions.

3. **Inner Iteration Loop**:
   - The inner loop iterates to solve for the density profile. For each layer, the density is recalculated based on the pressure profile and the equation of state.

4. **Pressure Adjustment**:
   - The `solve_ivp` function is used to solve the coupled ODEs for mass, gravity, and pressure.
   - The pressure at the surface is adjusted to match the target surface pressure using an iterative approach.

5. **Convergence Checks**:
   - The outer loop checks for convergence based on the relative differences in the calculated mass and core radius.
   - The inner loop checks for convergence based on the relative differences in the density profile.

6. **Final Output**:
   - Once convergence is reached, the final calculated mass, radius, core radius, density at the core-mantle boundary (CMB), and pressure profiles are printed.
   - Data is optionally saved to a file and plotted.

### Physical Model Description
The internal structure model of the exoplanet is based on a simplified approach using the following assumptions:
- The core and mantle are modeled as two distinct layers with different densities.
- The density profile is derived from the equation of state (EOS), which defines the relationship between pressure, density, and temperature.
- The pressure profile is solved using the `solve_ivp` function, which integrates the coupled ODEs for mass, gravity, and pressure.
- A simple scaling law is used to estimate the initial radius of the planet based on its mass.
- The model iterates to adjust the core-mantle boundary and the density profile until the solution converges within the specified tolerance limits.

### Key Functions and Equations

- **Coupled ODEs** (`coupled_odes`):
  - The function `coupled_odes` defines the derivatives of mass, gravity, and pressure with respect to radius. These equations are used to solve for the planet's internal structure.
  
- **Density Calculation** (`calculate_density`):
  - This function calculates the density at each layer based on the pressure and material properties of the core and mantle using the equation of state.

- **Temperature Calculation** (`calculate_temperature`):
  - The temperature profile is calculated for the planet based on the density and material properties, with the temperature increasing towards the center of the planet.

### Optional: Data Output and Plotting
- The model can output the results to a text file in the "output_files" directory. The file will contain the planet's radial profile, including density, gravity, pressure, temperature and mass enclosed.
- The model can also generate plots of the planet's density, gravity, pressure and mass enclosed profiles.

