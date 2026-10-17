# Process Flow

## Configuration Loading

The function `load_zalmoxis_config()` (in `config.py`) reads a TOML configuration file that specifies planet mass, layer mass fractions, per-layer EOS identifiers, temperature profile settings, solver tolerances, and output options.
The `[EOS]` section is parsed by `parse_eos_config()` (also in `config.py`), which accepts both the new per-layer format and the legacy global-string format.

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

2. **Inner loop (density profile convergence):** For each radial shell, recalculates the density from the local pressure (and temperature, if applicable) using the per-layer EOS returned by `get_layer_mixture()`.
   For multi-material layers, the density is computed via the phase-aware suppressed harmonic mean (see [multi-material mixing](mixing.md)): each component's contribution is weighted by a sigmoid function of its density, preventing non-condensed volatiles from inflating the mixture density.
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
This provides both robustness and speed, typically converging in 20 to 36 function evaluations.

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
When the terminal event fires, or the step size collapses, the ODE integration stops short of the outermost grid point.
A stop at a pressure of at most $10^{-6} P_c$, or below the target surface pressure, is treated as the surface: the remaining points take the mass and gravity at the stop and zero pressure, so the enclosed mass is continuous as the stop moves through the last shell.
After a step-size collapse, the stop state comes from a re-integration of the one shell in which the integration stopped, which applies the same rule; a re-integration that passes the last shell completes the profile.
A stop at a higher pressure, or with a non-finite mass, gravity or pressure, is a failed solve: the remaining points are NaN and a warning names the stop.
A failed solve is an error: at the first structure solve of the pressure search, or the re-solve at its root, that is not finite, `main` raises `StructureSolveError` with the radius, the outer and inner iteration, the central pressure and the stop, for both the Picard and the Newton outer solver.
This includes a trial central pressure at a bracket end: one failed solve stops the run, also where a wider bracket would avoid it.
The JAX path applies the same rule to the state where `diffeqsolve` stopped, at the event or at its last accepted step, without a re-integration, so near the $10^{-6} P_c$ limit a stop can pad on one path and fail on the other.
The residual function detects a surface stop ($P_{\mathrm{surface}} \leq 0$) and returns $-P_{\mathrm{target}}$, which is negative because the configuration requires $P_{\mathrm{target}} > 0$; this signals to Brent's method that $P_c$ is too low and keeps the bracket valid.

**Closure state capture.**
Since `brentq` only returns the root value (not intermediate ODE solutions), the residual function uses a mutable closure dict to capture the mass, gravity, and pressure arrays from the last evaluation.
After convergence, these arrays are extracted for the density update step.

## Convergence Checks

- The outer loop converges when the relative mass difference falls below `tolerance_outer`.
- The inner loop converges when the maximum relative density change across all shells falls below `tolerance_inner`.
- The pressure solver converges when (a) `brentq` reports convergence, (b) the surface pressure residual is within `pressure_tolerance` of the target, and (c) all pressures are non-negative ($P \geq 0$, allowing for zero-padded surface points from the terminal event).

The numerical solve is accepted when one of the tests below passes: the result then has `converged = True`, and `accepted_on` names the test, or `timeout` when the wall clock stopped the solve of the returned profile; it is None when no test passes.
`stopped_on_wall_clock` is True when the wall clock stopped that solve, also when no test passed.
The table band check (see [EOS physics](eos_physics.md)) runs after these tests: a band failure sets `converged = False` and keeps `accepted_on`, so read `converged` for the final verdict.

| `accepted_on` | Outer solver | Test |
|---|---|---|
| `strict` | Picard | The mass, density and pressure flags are True. Besides a change below `tolerance_inner`, the inner loop sets the density flag when it stops stuck or at `max_iterations_inner` with a relative density change below `max(100 * tolerance_inner, 0.1)`, or after 15 oscillations below `max(10 * tolerance_inner, 0.1)`. |
| `mass_only` | Picard | Mass converged and the smallest mass error is below `tolerance_outer`; density or pressure did not converge. |
| `max_iter` | Picard | The outer loop stopped at `max_iterations_outer` with the smallest mass error below 3 %. |
| `timeout` | Picard, Newton | The solve of the returned profile reached `wall_timeout` (in the outer or the inner loop) and one of the other tests passed; `timeout` replaces its label. |
| `oscillation` | Picard | The mass error changed sign 10 times since its last improvement; the best profile is returned, with its mass error below `3 * tolerance_outer`. This test does not fire in the current code. |
| `retry_skip` | Picard | The first solve was not accepted, density and pressure converged and its smallest mass error is below 7 %; `main` skips the retry. |
| `newton_tol` | Newton | The smallest mass error of the Newton steps is below `newton_tol`, the inner solve of the returned profile reports converged pressure and its density is the EOS density (`density_from_eos`), or the mass error at the brentq root is below `newton_tol`. |
| `brentq_best` | Newton | The bracket search or brentq stopped early with the smallest mass error seen below `newton_tol`. The returned profile is that of the last Newton step, and its mass error can be much larger. |

`mass_error_at_accept` is the relative error $|M - M_p| / M_p$ of the last value of `mass_enclosed` in the returned profile, also when no test passed, and None when that value is not finite.
It can differ from `best_mass_error`, the smallest error over the outer iterations or Newton evaluations (the error at the root on the brentq root exit), because the returned profile is not always the one with that error.
The returned `density` is the EOS density $\rho_{\mathrm{EOS}}(P, T)$ at the returned pressure and temperature of each node, and not the damped Picard iterate.
`density_from_eos` is False only when the wall clock stopped the solve before its first pressure solve; `density` is then the initial density, and only then are the outermost shells smoothed.
`converged_density` reports the inner Picard loop and does not bound the difference between the iterate and the EOS density.
A trapezoid integral of the returned `density` differs from `mass_enclosed[-1]` by the quadrature error of the grid, which is largest at density jumps such as the core-mantle boundary.

## Output

On convergence, the model returns radial profiles of gravity, pressure, density, temperature, and enclosed mass.
Derived structural parameters include: core radius, CMB pressure, central pressure, average density, core mass fraction, core radius fraction, and (if applicable) mantle phase fractions.


