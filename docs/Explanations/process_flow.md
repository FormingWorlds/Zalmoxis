# Process Flow

## Configuration Loading

The function `load_zalmoxis_config()` reads a TOML configuration file that specifies planet mass, layer mass fractions, per-layer EOS identifiers, temperature profile settings, solver tolerances, and output options.
The `[EOS]` section is parsed by `parse_eos_config()`, which accepts both the new per-layer format and the legacy global-string format.

## Initial Parameter Setup

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

## Iterative Solution

The model uses three nested convergence loops:

1. **Outer loop (mass convergence):** Updates the total planet radius estimate by comparing the calculated total mass against the target mass and rescaling: $R_p \leftarrow R_p \times (M_{\mathrm{target}} / M_{\mathrm{calculated}})^{1/3}$.

2. **Inner loop (density profile convergence):** For each radial shell, recalculates the density from the local pressure (and temperature, if applicable) using the per-layer EOS returned by `get_layer_eos()`.
   For multi-material layers, the density is computed via the phase-aware suppressed harmonic mean (see [model documentation](model.md#multi-material-mixing-with-phase-aware-suppression)): each component's contribution is weighted by a sigmoid function of its density, preventing non-condensed volatiles from inflating the mixture density.
   Density updates are damped by averaging with the previous iteration.

3. **Pressure solver (Brent's method):** Finds the central pressure $P_c$ such that the surface pressure matches the target boundary condition. See [Pressure Solver](#pressure-solver-brents-method) below.

## Pressure Solver (Brent's Method)

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

## Convergence Checks

- The outer loop converges when the relative mass difference falls below `tolerance_outer`.
- The inner loop converges when the maximum relative density change across all shells falls below `tolerance_inner`.
- The pressure solver converges when (a) `brentq` reports convergence, (b) the surface pressure residual is within `pressure_tolerance` of the target, and (c) all pressures are non-negative ($P \geq 0$, allowing for zero-padded surface points from the terminal event).

The model reports overall convergence only when all three loops have converged.

## Output

On convergence, the model returns radial profiles of gravity, pressure, density, temperature, and enclosed mass.
Derived structural parameters include: core radius, CMB pressure, central pressure, average density, core mass fraction, core radius fraction, and (if applicable) mantle phase fractions.


