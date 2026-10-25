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

3. **Pressure solver (Brent's method):** Searches with `brentq` for the central pressure $P_c$ at which the surface pressure equals the target boundary condition, and accepts the profile at the root it returns by the tests in [Convergence Checks](#convergence-checks); a [padded profile](#padded-accepts) can pass them.
   See [Pressure Solver](#pressure-solver-brents-method) below.

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
A stop at a pressure of at most $10^{-6} P_c$, or below the target surface pressure, is treated as the surface: the remaining points take the mass at the stop and zero pressure, so the enclosed mass is continuous as the stop moves through the last shell, and the gravity of that mass at their own radius, since no mass lies above the stop.
The returned density at these points is the EOS density at the target surface pressure and the temperature of the point; with the adiabat integrated from the core-mantle boundary, that temperature is the adiabat continued from the last point with positive pressure to the target surface pressure, once the adiabat blend has reached 1. A result accepted earlier in the blend keeps the linear profile at the outer radius, the value of the temperature function at zero pressure, so its padded temperature can lie above the point below it: by 112, 244 and 373 K at blend 0.25, 0.5 and 0.75 for 1 Earth mass on 50 points with the CMB at 4000 K. A trapezoid sum of this density over the points, as in Aragog, counts the shell between the stop and the outer radius at that density, while the enclosed mass of the structure does not include it; the midpoint mantle integrals of `solve_strong_partition` and `solve_miscible_interior` take the shell that crosses the stop at the mean of its two point densities and temperatures.
After a step-size collapse, the stop state comes from a re-integration of the one shell in which the integration stopped, which applies the same rule; a re-integration that passes the last shell completes the profile.
A stop at a higher pressure, or with a non-finite mass, gravity or pressure, is a failed solve: the remaining points are NaN and a warning names the stop.
A failed solve is an error: at the first structure solve of the pressure search, or the re-solve at its root, that is not finite, `main` raises `StructureSolveError` with the radius, the outer and inner iteration, the central pressure and the stop, for both the Picard and the Newton outer solver.
An accepted result that breaks a physical invariant is an error too: `main` checks every accepted result with `zalmoxis.invariants.check_accepted` and raises `StructureInvariantError`, which names the first broken invariant, its deviation and tolerance, the values that enter it, the accept label and, for an invariant of the profile, the node; the mass closure uses the `accept_mass_limit` that the accept path wrote. The core-mantle boundary must be at a node with `mass_enclosed >= cmb_mass`, compared exactly as PROTEUS reads the profile, so a run with `core_mass_fraction` 1, or with an ice layer and `core_mass_fraction + mantle_mass_fraction` 1, passes only when the solve ends with a mass surplus. It is not a `RuntimeError`, so a caller that keeps its previous structure after a failed solve does not catch it.
This includes a trial central pressure at a bracket end: one failed solve stops the run, also where a wider bracket would avoid it.
The JAX path applies the same rule to the state where `diffeqsolve` stopped, at the event or at its last accepted step, without a re-integration, so near the $10^{-6} P_c$ limit a stop can pad on one path and fail on the other.
The residual function detects a surface stop ($P_{\mathrm{surface}} \leq 0$) and returns $-P_{\mathrm{target}}$, which is negative because the configuration requires $P_{\mathrm{target}} > 0$; this signals to Brent's method that $P_c$ is too low and keeps the bracket valid.

**Closure state capture.**
Since `brentq` only returns the root value (not intermediate ODE solutions), the residual function stores the mass, gravity, and pressure arrays of each evaluation in a mutable closure dict.
After Brent's method returns, `solve_structure` runs once more at the root, and that profile goes to the accept test and the density update.
The stored arrays of the last evaluation are used only when the bracket is invalid.

### Padded accepts

A padded profile is the result of a surface stop (see the Early termination handling paragraph above): the integration stopped before the outer radius $R$, and the remaining points have $P = 0$.
Its pressure at the outer radius is $P(R) = 0$ instead of $P_{\mathrm{target}}$, so $\lvert P(R) - P_{\mathrm{target}} \rvert = P_{\mathrm{target}}$.
The pressure test (b) in [Convergence Checks](#convergence-checks) passes such a profile for any target below `pressure_tolerance`, $10^9$ Pa by default.
The retry that the Picard outer solver runs after an unconverged solve uses 0.1 times this tolerance, so there a padded profile passes for a target below that value.
`result['padded']` is True when the returned pressure is 0 at the outer radius.

A trial central pressure whose integration ends in a surface stop gives $f = -P_{\mathrm{target}}$; one that stops at a higher pressure or reaches a non-finite state raises `StructureSolveError`.
Brent's method stops when its bracket is narrower than `xtol` $+$ `rtol` $\lvert P_c \rvert$, with `xtol` $= 10^6$ Pa and `rtol` $= 10^{-10}$.
It then returns the bracket end with the smaller $\lvert f \rvert$.
It therefore returns the end with a surface stop when the other end reaches the outer radius with $P(R) > 2 P_{\mathrm{target}}$.

If $P(R)$ is continuous at the stop and rises with a slope $dP(R)/dP_c$ of at most $S$, the reached end of the final bracket has $P(R)$ of at most about $S \times 10^6$ Pa.
A padded first root then needs $S > 2 P_{\mathrm{target}} / 10^6$ Pa, that is $S > 0.2$ for a target of 1 atm.
The condition is necessary, not sufficient.
It does not apply when $P(R)$ jumps at the stop or scatters.

The solve then refines the root in 2 steps (`_refine_pressure_root`).
When the root misses $P_{\mathrm{target}}$ by more than $0.1 P_{\mathrm{target}}$, a second Brent solve runs on the nearest pair of evaluated central pressures whose residuals bracket the target, with `xtol` $= 0.1 P_{\mathrm{target}}$ divided by the secant slope of $f$ over that pair, when that `xtol` is below $10^6$ Pa and either the slope is at most 1 or the reached end of the pair misses $P_{\mathrm{target}}$ by at most $P_{\mathrm{target}}$.
A smooth profile whose density does not fall with pressure at a fixed radius, along its temperature profile, has $dP(R)/dP_c \le 1$ while the enclosed mass at every radius rises with $P_c$, also across the layer boundaries at fixed enclosed mass.
A steeper pair whose reached end misses by more than $P_{\mathrm{target}}$ holds a jump at the stop or scatter of $P(R)$ over the pair; there a second solve does not improve on the reached end and costs structure solves.
Where the density falls with pressure (a mushy zone, the condensed-density or binodal weights of a mixture), a smooth pair can be steeper than 1; the second solve is then skipped, and the result is the first root or, for a padded root, the reached end within $P_{\mathrm{target}}$, never further from the target than without the refinement.
Its root replaces the first one only when it misses $P_{\mathrm{target}}$ by no more; a second solve that raises keeps the first root.
When the root still ends in a surface stop, the reached end of that pair replaces it if its $\lvert P(R) - P_{\mathrm{target}} \rvert$ is at most $P_{\mathrm{target}}$ (`_PADDED_REPLACE_MAX_MISS_PT` $= 1$).
Otherwise the padded root stays. The replacement, the kept padded root and a failed second solve log at the DEBUG level, since a Picard run can keep a padded root in hundreds of pressure solves; `main` logs one warning when it accepts a padded profile.

The figure shows $\lvert P(R) - P_{\mathrm{target}} \rvert / P_{\mathrm{target}}$ of the accepted profile for 26 setups built from `input/default.toml`.
5 of the 6 setups with the Newton outer solver at `relative_tolerance` $10^{-9}$ reach the outer radius; setup 10 (PALEOS 1 on the JAX path) ends padded, since $P(R)$ near the stop scatters on that path.
15 of the 19 setups with the Picard outer solver at the tolerances of `input/default.toml` that return a profile end padded, `input/default.toml` itself (setup 8) among them: in each, the reached end of the nearest sign pair misses $P_{\mathrm{target}}$ by more than $P_{\mathrm{target}}$.
Every padded profile has one padded point, the outermost.
Setup 24 (PALEOS 3-layer ice 1) returns no profile: a trial central pressure gives a non-finite profile, and the solve raises `StructureSolveError`.
This setup fails in 2 of 6 starts whose planet mass differs by $k \times 10^{-13}$ of its value ($k$ = 0 to 5), with and without the root refinement.
The 9 reached profiles miss $P_{\mathrm{target}}$ by at most $0.38 P_{\mathrm{target}}$ (median $0.077 P_{\mathrm{target}}$).

![Surface pressure miss of the accepted profile for 26 solver setups](../assets/validation/pressure_root_sweep.svg#only-light){ width="80%" }
![Surface pressure miss of the accepted profile for 26 solver setups](../assets/validation/pressure_root_sweep_dark.svg#only-dark){ width="80%" }

$\lvert P(R) - P_{\mathrm{target}} \rvert / P_{\mathrm{target}}$ of the accepted profile per setup: reached (circles), padded (squares, at 1 since $P(R) = 0$) and without a profile (cross above the cap); the dashed line is the cap of the reached end that may replace a padded root.
Setups: 0 Analytic 0.1, 1 Analytic 1, 2 Analytic 10, 3 Analytic 1, 2000 layers, 4 Analytic 1, Newton, 5 WB isothermal 3000 K, 6 WB isothermal 4000 K, 7 WB linear, 8 PALEOS 1 (default.toml), 9 PALEOS 1, Newton, 10 PALEOS 1, JAX Newton, 11 PALEOS 0.1, 12 PALEOS 3, 13 PALEOS 3, JAX Newton, 14 PALEOS 10, Newton, 15 PALEOS linear 1, 16 PALEOS isothermal 1, 17 PALEOS 1, 2000 layers, 18 PALEOS 1, P_t 1e7 Pa, 19 PALEOS 1, P_t 1e3 Pa, 20 PALEOS-2phase linear 1, 21 PALEOS-2phase 1, 22 PALEOS-2phase 3, 23 PALEOS + 3 % H 1, 24 PALEOS 3-layer ice 1, 25 PALEOS 5, Newton.
Made with `python -m tools.validation.pressure_root_sweep run <dir> -j 2`, then `python -m tools.validation.pressure_root_sweep plot <dir>`.

## Convergence Checks

- The outer loop converges when the relative mass difference falls below `tolerance_outer`.
- The inner loop converges when the maximum relative density change across all shells falls below `tolerance_inner`.
- The pressure solver converges when (a) `brentq` reports convergence, (b) the surface pressure residual is within `pressure_tolerance` of the target, and (c) all pressures are non-negative ($P \geq 0$, allowing for zero-padded surface points from the terminal event).
  A padded profile passes test (b); see [Padded accepts](#padded-accepts).

The numerical solve is accepted when one of the tests below passes: the result then has `converged = True`, and `accepted_on` names the test, or `timeout` when the wall clock stopped the solve of the returned profile; it is None when no test passes.
`stopped_on_wall_clock` is True when the wall clock stopped that solve, also when no test passed.
The table band check (see [EOS physics](eos_physics.md)) runs after these tests: a band failure sets `converged = False` and keeps `accepted_on`, so read `converged` for the final verdict.

| `accepted_on` | Outer solver | Test |
|---|---|---|
| `strict` | Picard | The mass, density and pressure flags are True. Besides a change below `tolerance_inner`, the inner loop sets the density flag when it stops stuck or at `max_iterations_inner` with a relative density change below `max(100 * tolerance_inner, 0.1)`, or after 15 oscillations below `max(10 * tolerance_inner, 0.1)`. |
| `mass_only` | Picard | Mass converged and the smallest mass error is below `tolerance_outer`; density or pressure did not converge. |
| `max_iter` | Picard | The outer loop stopped at `max_iterations_outer` with the mass error of the best pass below 3 %; the best pass is returned. |
| `timeout` | Picard, Newton | The solve reached `wall_timeout` (in the outer or the inner loop) and one of the other tests passed; `timeout` replaces its label. On Picard the returned profile is the best pass, which can be an earlier, complete pass. |
| `oscillation` | Picard | The mass error changed sign 10 times since the best pass last changed, each time with a magnitude at least 99 % of that of the pass before, and the mass error of the best pass is below `3 * tolerance_outer` and below 3 %. With an adiabatic temperature mode, a T-dependent EOS and no external temperature profile, the stop waits until the adiabat blend reaches 1. A pass the wall clock cut ends the solve as `timeout` instead. The best profile is returned and accepted on `3 * tolerance_outer`, or on the 3 % ceiling when its density or pressure did not converge. |
| `retry_skip` | Picard | The first solve was not accepted, density and pressure converged and the mass error of the returned profile is below 7 %; `main` skips the retry. |
| `newton_tol` | Newton | The smallest mass error of the Newton steps, or the mass error at the brentq root, is below `newton_tol`, and the inner solve of the returned profile reports converged pressure and the EOS density (`density_from_eos`). |
| `brentq_best` | Newton | The bracket search or brentq stopped early with the smallest mass error of the evaluations below `newton_tol`; the returned profile is that of the evaluation with that error, and its inner solve reports converged pressure and the EOS density. |

`mass_error_at_accept` is the relative error $|M - M_p| / M_p$ of the last value of `mass_enclosed` in the returned profile, also when no test passed, and None when that value is not finite.
It can differ from `best_mass_error`, because the returned profile is not always the one with that error. In the Picard solve `best_mass_error` is the mass error of the best pass (below); in the Newton solve it is the smallest error over the Newton evaluations, or the error at the root on the brentq root exit.
The returned `density` is the EOS density $\rho_{\mathrm{EOS}}(P, T)$ at the returned pressure and temperature of each node, and not the damped Picard iterate.
After a wall-clock stop in the outer loop after at least one pass, accepted or not, after a `max_iter` accept, after an `oscillation` stop, and after an accepted solve whose last pass the wall clock stopped before its first pressure solve, the best pass is returned, with its own gravity, `converged_density`, `converged_pressure` and `adiabat_blend`. The best pass is the pass with the smallest mass error among the passes with a pressure solve on the newest adiabat blend step (0, 0.25, 0.5, 0.75, 1); until a pass on a new blend step has a pressure solve, the best pass of the step before stays. When no accept test of the stop would take the newest best pass (the strict test, the mass test of `tolerance_outer` or the 3 % ceiling in the table above), the best pass of the earlier steps is returned instead if its mass error is smaller and the stop returns it: a restore of the best pass or a wall-clock stop does, an iteration cap only when the 3 % ceiling takes the earlier pass. It comes with its `adiabat_blend`, a warning that names both blends, and its mass error as `best_mass_error`; that pass ended below `tolerance_outer`, as the blend ramp starts only then, and a wall-clock stop takes its mass verdict again from it. An accept of the newest pass by the mass test of the loop keeps that pass. `main` warns when it accepts a result of an adiabatic temperature mode with a T-dependent EOS whose `adiabat_blend` is below 1.
`cmb_mass` and `core_mantle_mass` are the enclosed masses at the core-mantle boundary and at the top of the mantle: `core_mass_fraction` and `core_mass_fraction + mantle_mass_fraction` times `planet_mass`. Every pass integrates the structure and looks up the density with these boundaries, so a node of the returned profile has the EOS density of the layer that `cmb_mass` and `core_mantle_mass` give it, and the core mass fraction of the returned profile is `cmb_mass` over the last value of `mass_enclosed`, `core_mass_fraction` times `planet_mass` over that value.
`density_from_eos` is False only when no test passed and the returned pass has no pressure solve: the wall clock stopped the solve before its first pass, or before the first pressure solve of the first pass or of the pass at `max_iterations_outer`; a pass without a pressure solve is never the best pass. The result then has the radius of that pass (the initial guess before the first pass), a pressure and a `density` that are zero except at most at the centre node, and a `mass_enclosed` that is zero. A cut before the first pressure solve of any other pass leaves the outer loop running, and its next check returns the best pass with its EOS density.
`converged_density` reports the inner Picard loop and does not bound the difference between the iterate and the EOS density.
When the first solve is not accepted, `main` retries with tighter parameters, seeded with the returned outer radius when it lies between 0.2 and 5 times $R_\oplus (\max(M_p, 0.01\,M_\oplus)/M_\oplus)^{0.282}$, and hands the retry the returned `density` when a node of it is above zero; `_solve` uses that seed only when more than one node is above zero. After a pass with a pressure solve that seed is the EOS density; after an unaccepted result without a pressure solve only the radius seeds the retry. After an unaccepted wall-clock stop in the outer loop the seed radius is that of the best pass, not the newest radius guess of the outer loop. PROTEUS also seeds its next structure solve with the returned `density`.
On the Newton path, a step whose mass test passes while its inner solve does not report converged pressure and the EOS density ends the search with `converged = False`, and `main` returns that result without a retry. PROTEUS then retries once, unless `best_mass_error` is above 5 %, with three times the `tolerance_outer`, twice the `max_iterations_outer`, a 600 s wall clock, and without the density seed and the central pressure hint of its first call. Each Newton step runs one outer pass, so the first two settings do not change the Newton test; the retry can pass only through its different start or the wall clock. PROTEUS stops the run when the retry also fails.
A trapezoid integral of the returned `density` differs from `mass_enclosed[-1]` by the quadrature error of the grid, which is largest at density jumps such as the core-mantle boundary.

## Output

On convergence, the model returns radial profiles of gravity, pressure, density, temperature, and enclosed mass.
Derived structural parameters include: core radius, CMB pressure, central pressure, average density, core mass fraction, core radius fraction, and (if applicable) mantle phase fractions.


