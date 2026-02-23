# Interior Structure Model

## Overview
The model calculates the internal structure of two- or three-layered fully differentiated exoplanets based primarily on its total mass and compositional mass fractions. The internal pressure, density, gravity, and radius profiles are computed iteratively, using physically motivated equations of state (EOS). Earth-like rocky planets with an iron core and silicate mantle and water-rich planets with an Earth-like rocky interior and an outer water ice layer are modeled using an EOS adapted from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) at 300 K. With this EOS, The model supports planets of up to 50 Earth masses. Alternatively, planets where the mantle may experience partial or complete melting under high temperatures are modeled using the EOS from [Wolf and Bower (2018)](https://www.sciencedirect.com/science/article/pii/S0031920117301449), which provides a temperature-dependent, phase-aware description of the mantle. This EOS allows the model to capture the transition between solid, partially molten, and fully molten states, which is critical for accurately modeling the internal structure and subsequent thermal evolution of hot rocky planets.

## Main function
The `main` function runs the exoplanet interior structure model. It initializes parameters and iteratively adjusts the planet's internal structure until convergence is reached.

### Physical Model Description
The internal structure model is based on a simplified approach using the following assumptions:

- The core, mantle, and water layer are modeled as distinct layers with different densities.
- The density profile is derived from the equation of state (EOS), which defines the relationship between pressure and density.
- Assuming hydrostatic and thermodynamic equilibrium, the pressure profile is solved using the `solve_ivp` function, which integrates the coupled ODEs for mass, gravity and pressure, starting from the center and progressing outward:

  $$
  \frac{dM}{dr} = 4 \pi r^2 \rho
  $$ 

  $$
  \frac{dg}{dr} = 4 \pi G \rho - \frac{2g}{r}
  $$

  $$
  \frac{dP}{dr} = -\rho g
  $$
  
  where \( M(r) \) is the enclosed mass within radius \( r \), \( \rho(r) \) is the local density at radius \( r \), \( g(r) \) is the gravitational acceleration at radius \( r \), \( P(r) \) is the pressure at radius \( r \), \( G \) is the gravitational constant and \( r \) is the radial coordinate from the planet's center.  

### Process Flow
- **Configuration File Loading**
   
    The function reads a configuration file that defines the initial guesses, assumptions and iterative process parameters.

- **Initial Parameter Setup**

    The initial guess for the planet's radius is set based on the scaling law in [Noack et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...638A.129N/abstract) as:

    $$
    R_p[\text{m}] = 1000 \times (7030 - 1840 \times X_{\text{Fe}}) \times \left( \frac{M_p}{M_E} \right)^{0.282}
    $$

    where \( X_{\text{Fe}} \) is the weight iron fraction, \( M_p \) is the planet mass, \( M_E \) is Earth's mass.

    The initial guess for the planet's core mass is set as:
    $$
    M_{\text{core}} = X_{\text{CMF}} \times M_p
    $$

    where \( X_{\text{CMF}} \) is the core mass fraction and \( M_p \) is the planet mass.

    The initial guess for the planet's mass up to the inner mantle boundary and including the core is set as:
    $$
    M_{\text{core+IM}} = (X_{\text{CMF}}+X_{\text{IMF}}) \times M_p
    $$

    where \( X_{\text{CMF}} \) is the core mass fraction, \( X_{\text{IMF}} \) is the inner mantle mass fraction and \( M_p \) is the planet mass.

    The initial guess for the planet's pressure at the center is based on an empirical scaling law derived from the hydrostatic equilibrium equation and set as:

    $$
    P_p = P_E \times \left( \frac{M_p}{M_E} \right)^{2} \times \left( \frac{R_p}{R_E} \right)^{-4}
    $$

    where \( P_E \) is Earth's pressure at the center, \( M_p \) is the planet mass, \( M_E \) is Earth's mass, \( R_p \) is the guessed planet radius and \( R_E \) is Earth's radius.

- **Iterative Solution**

    The model iterates through nested loops to refine mass, radius, pressure, and density profiles until convergence.

    Outer Loop (Mass Convergence):
    Updates the total interior mass estimate based on current mass fraction guesses and recalculates the overall planetary radius and profile.

    Inner Loop (Density Profile Convergence):
    For each layer and radius, recalculates the density based on the pressure profile and the suitable EOS.

    Innermost Loop (Pressure Adjustment):
    Using numerical ODE solvers (`solve_ivp`), integrates coupled hydrostatic equilibrium equations to update the pressure profile. It also adjusts central pressure to match target surface pressure, ensuring physical consistency.

- **Convergence Checks**

    * The outer loop checks for convergence based on the relative differences in the calculated mass.

    * The inner loop checks for convergence based on the relative differences in the density profile.
    
    * The pressure adjustment loop also checks for convergence, ensuring that the pressure difference is within the defined tolerance and that all pressure values remain physically valid (all positive).

- **Output Generation**

    Once the solution has converged, the model returns the final radial profiles of gravity, pressure, density, and mass throughout the planet. In addition to these profiles, several key structural parameters are extracted, including: the core radius, mantle density at the core mantle boundary (CMB), core density at the CMB, pressure at the CMB, pressure at center, average density, CMB mass fraction, core radius fraction, inner mantle mass fraction and inner mantle radius fraction. The model also records the total computation time and a convergence flag to indicate whether the solution successfully met the stopping criteria.

## Other Key Functions

- **Coupled ODEs** (`coupled_odes`): Defines the derivatives of mass, gravity and pressure with respect to radius. These equations are used to solve for the planet's internal structure.

- **Structure Solver** (`solve_structure`): Integrates the coupled ODEs for mass, gravity, and pressure across the planetary radius using `solve_ivp`. For temperature-dependent EOS (`"Tabulated:iron/Tdep_silicate"`), the radial grid is split to handle large step sizes near the surface, while for fixed-temperature EOS a single integration is performed.

- **Density Calculation** (`calculate_density`): Determines the density at a given pressure (and temperature if applicable) for a specified material/layer and EOS choice.

- **Temperature-Dependent Density** (`get_Tdep_density`): Computes the mantle density by accounting for temperature-dependent phase transitions, using melting curves derived from [Monteux et al. (2016)](https://www.sciencedirect.com/science/article/pii/S0012821X16302199?via%3Dihub). In this implementation, the liquidus corresponds to Equations (11) and (13) from [Monteux et al. (2016)](https://www.sciencedirect.com/science/article/pii/S0012821X16302199?via%3Dihub), while the solidus is defined as the same liquidus curve shifted -600 K.

If the local temperature $T$ is below the solidus temperature $T_{\mathrm{sol}}$, the mantle material is considered fully solid. If $T$ exceeds the liquidus temperature $T_{\mathrm{liq}}$, the mantle is treated as completely molten. For temperatures between $T_{\mathrm{sol}}$ and $T_{\mathrm{liq}}$, corresponding to the mixed or mush phase, the density is obtained by linearly interpolating the specific volume (inverse of density) between the solid and liquid phases.

The melt fraction of the mantle material is defined as:

$$
f_{\text{melt}} = \frac{T - T_{\text{sol}}}{T_{\text{liq}} - T_{\text{sol}}}
$$

where $T$ is the local temperature, $T_{\text{sol}}$ is the solidus temperature, and $T_{\text{liq}}$ is the liquidus temperature.

Assuming volume additivity, the mixed-phase specific volume is:

$$
\frac{1}{\rho_{\text{mixed}}} = (1 - f_{\text{melt}}) \frac{1}{\rho_{\text{solid}}} + f_{\text{melt}} \frac{1}{\rho_{\text{liquid}}}
$$

where $\rho_{\text{solid}}$ is the density of the solid mantle and $\rho_{\text{liquid}}$ is the density of the molten mantle.

Thus, the temperature-dependent mixed-phase density is given by:

$$
\rho_{\text{mixed}} = \frac{1}{(1 - f_{\text{melt}}) \frac{1}{\rho_{\text{solid}}} + f_{\text{melt}} \frac{1}{\rho_{\text{liquid}}}}
$$

- **Temperature Profile** (`calculate_temperature_profile`): Returns a callable function that provides the temperature at any radius within the planet. Supports three modes: "isothermal" for a uniform temperature, "linear" for a linear gradient between the center and surface, and "prescribed" for a user-provided temperature profile loaded from a file.

## Analytic EOS: Modified Polytrope (Seager et al. 2007)

The `"Analytic:Seager2007"` EOS implements the modified polytropic fit from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346), Table 3, Eq. 11:

$$
\rho(P) = \rho_0 + c \cdot P^n
$$

where $\rho$ is density in kg/m$^3$, $P$ is pressure in Pa, and $\rho_0$, $c$, $n$ are material-specific parameters. This analytic form approximates the full EOS, which combines low-pressure Vinet or fourth-order Birch-Murnaghan fits with high-pressure Thomas-Fermi-Dirac (TFD) theory.

The fit is valid for $P < 10^{16}$ Pa, covering all planetary pressures encountered in Zalmoxis models. Accuracy is 2–5% at low ($P < 5$ GPa) and high ($P > 30$ TPa) pressures, and up to 12% at intermediate pressures where the transition between the low-pressure and TFD regimes occurs.

The following 6 materials are available:

| Material key | Compound | $\rho_0$ (kg/m$^3$) | $c$ (kg m$^{-3}$ Pa$^{-n}$) | $n$ |
|---|---|---|---|---|
| `iron` | Fe ($\epsilon$) | 8300 | 0.00349 | 0.528 |
| `MgSiO3` | MgSiO$_3$ perovskite | 4100 | 0.00161 | 0.541 |
| `MgFeSiO3` | (Mg,Fe)SiO$_3$ | 4260 | 0.00127 | 0.549 |
| `H2O` | Water ice (VII/VIII/X) | 1460 | 0.00311 | 0.513 |
| `graphite` | C (graphite) | 2250 | 0.00350 | 0.514 |
| `SiC` | Silicon carbide | 3220 | 0.00172 | 0.537 |

This EOS assumes zero/low temperature (300 K) and is appropriate for cold solid exoplanet structure models. Any of the 6 materials can be assigned to any structural layer (core, mantle, outer layer), enabling modeling of exotic compositions such as carbon planets (iron/SiC or iron/graphite) that are not available with the tabulated EOS options.