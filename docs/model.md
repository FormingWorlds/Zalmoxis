# Interior Structure Model

## Overview
The model calculates the internal structure of an exoplanet based on its mass and various other parameters, including temperature, pressure and density profiles. The solution is derived iteratively and the model currently uses a simplified equation of state (EOS) for the core and mantle from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346). The model currently supports simulations of Earth-like exoplanets of up to 50 Earth masses.

## Main function
The `main` function runs the exoplanet interior structure model. It reads the configuration file, initializes parameters and performs an iterative solution to calculate the planet's internal structure.

### Parameters
- `temp_config_path` (optional): Path to the configuration file. If not provided, the default configuration file is chosen.
- `id_mass` (optional): Identifier for the planet mass in Earth masses.

### Process Flow

1. **Configuration File Loading**:  
   The function reads a configuration file that defines the initial guesses, assumptions, and iterative process parameters.

2. **Initial Parameter Setup**:  

   The initial guess for the planet's radius is set based on the scaling law in [Noack et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...638A.129N/abstract) as:  

   $$
   R_p[\text{m}] = 1000 \times (7030 - 1840 \times X_{\text{Fe}}) \times \left( \frac{M_p}{M_{\text{Earth}}} \right)^{0.282}
   $$  

   where \( X_{\text{Fe}} \) is the weight iron fraction, \( M_p \) is the planet mass, and \( M_{\text{Earth}} \) is Earth's mass.  

   The initial guess for the planet's core radius is set as:  

   $$
   R_{\text{core}} = X_{\text{CMF}} \times R_p[\text{m}]
   $$  

   where \( X_{\text{CMF}} \) is the core mass fraction, and \( R_p[\text{m}] \) is the guessed planet radius from above.  

3. **Iterative Solution**:  

   The model iteratively adjusts the planet's radius and core-mantle boundary (CMB) radius using an outer loop. In each outer iteration:
   
   - The function sets up a radial grid and initializes arrays for **density, mass enclosed, gravity, and pressure**.
   - The **core mass is estimated**, and an initial guess for the pressure at the center is set.
   - A simplified initial density profile is assigned based on the core and mantle assumptions.

   In the **inner loop**, the model iteratively adjusts the planet's density profile, using `solve_ivp` to solve the coupled **ordinary differential equations (ODEs)** for mass, gravity, and pressure as a function of radius.  
   
   - For each layer, the density is recalculated based on the pressure profile and the equation of state.

   In the **innermost loop (pressure adjustment loop)**, the `solve_ivp` function is used again to solve the coupled ODEs for mass, gravity, and pressure.  
   
   - The pressure profile is adjusted iteratively to match the target surface pressure, with each adjustment made using a scaling factor.

4. **Convergence Checks**:  

   - The **outer loop** checks for convergence based on the relative differences in the calculated **mass and core radius**.  
   - The **inner loop** checks for convergence based on the relative differences in the **density profile**.  
   - The **pressure adjustment loop** also checks for convergence, ensuring that the **pressure difference is within the defined tolerance** and that all pressure values remain **physically valid (all positive)**.

5. **Output Generation**:  

   Once the solution has converged, the final values for **mass, radius, core radius, and other calculated parameters** are printed. Other output parameters include:  

   - Mantle density at the **Core-Mantle Boundary (CMB)**
   - Core density at the **CMB**
   - Pressure at the **CMB**
   - Pressure at the **center**
   - Average density
   - **CMB mass fraction**  
   - **Core radius fraction**

### Physical Model Description
The internal structure model is based on a simplified approach using the following assumptions:

- The core and mantle are modeled as two distinct layers with different densities.
- The density profile is derived from the equation of state (EOS), which defines the relationship between pressure, density and temperature.
- The pressure profile is solved using the `solve_ivp` function, which integrates the coupled ODEs for mass, gravity and pressure.
- A simple scaling law is used to estimate the initial radius of the planet based on its mass.
- The model iterates to adjust the core-mantle boundary and the density profile until the solution converges within the specified tolerance limits.

## Other Key Functions and Equations

- **Coupled ODEs** (`coupled_odes`): Defines the derivatives of mass, gravity and pressure with respect to radius. These equations are used to solve for the planet's internal structure.
  
- **Density Calculation** (`calculate_density`): Calculates the density at each layer based on the pressure and material properties of the core and mantle using the equation of state.