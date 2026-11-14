# Hydrostatic structure

## What this page checks

This page checks how Zalmoxis computes the inside of a planet in hydrostatic equilibrium: the mass, the gravity and the pressure at each radius.
Validation here means that we run the code on a case where another source gives the right answer.
The test then checks that the code gives that answer within a stated tolerance.
The right answer comes from closed-form solutions, from conservation laws, from a second implementation of the same equations, or from the pressure that the planet must have at its surface.
If this part of the code is wrong, every structure that Zalmoxis returns is wrong. This holds also for the structures it gives to PROTEUS: the radius of the planet and the pressure and gravity at each depth.
Each case below says why the check exists, what it tests, how we know the right answer and why it matters.
The box "Test details" at the end of each case gives the reference, the tolerance, the test id and the command.
The [overview](index.md) lists every case of the tab.

## Terms used on this page

- $r$ and $R$: the radius, from 0 at the centre, and the outer radius of the planet.
- $R_\oplus$: the Earth radius of `zalmoxis.constants`, 6335.4 km (the mean Earth radius is 6371 km); $M_\oplus$: the Earth mass, $5.972 \times 10^{24}$ kg.
- $M(r)$, $g(r)$, $P(r)$, $\rho(r)$: the mass inside radius $r$ (the enclosed mass, kg), the gravitational acceleration (m/s$^2$), the pressure (Pa) and the density (kg/m$^3$) at radius $r$; $G$ is the gravitational constant.
- EOS (equation of state): the relation $\rho(P, T)$ between the density, the pressure and the temperature of a material.
- CMB: the core-mantle boundary. $\rho_c$: the density at the centre.
- ODE integrator: the code that steps the three structure equations (ordinary differential equations) outward in $r$.
- $N$ and `rtol`: the number of radial grid points, and the relative tolerance of the ODE integrator (RK45, an adaptive Runge-Kutta method of order 4 and 5); the configuration key is `relative_tolerance`.
- Relative error: $|x_\mathrm{num} - x_\mathrm{exact}| / |x_\mathrm{exact}|$, the difference between the computed and the exact value, divided by the exact value.
- Round-off level: a relative error of about $10^{-16}$ to $10^{-13}$, the size of the rounding error of floating-point arithmetic. A result at this level is exact within the precision of the computer.
- Central pressure $P_c$: the pressure at the centre of the planet.
- Pressure solve: the loop that adjusts $P_c$ until the pressure at the outer radius equals the target surface pressure $P_t$ (written $P_\mathrm{target}$ on the process-flow page).
- Brent's method (`scipy.optimize.brentq`): the root finder of the pressure solve. It narrows a bracket around a sign change of the residual and stops when the bracket is narrower than `xtol`, $10^6$ Pa in the pressure solve. Newton's method, used by the Newton outer solver, instead steps towards the zero of a function along its slope.
- Residual $f(P_c) = P(R) - P_t$: the pressure $P(R)$ at the outer radius $R$ minus the target surface pressure $P_t$.
- Target surface pressure $P_t$: 1 atm, 101325 Pa, unless a case says otherwise.
- Reached: the integration from a trial $P_c$ reaches the outer radius.
- Surface stop: the integration ends where the pressure reaches 0 before the outer radius.
- Miss: $\lvert P(R) - P_t \rvert$, often given in units of $P_t$.
- Converged: the outer solver met one of its accept tests; `accepted_on` names the test.
- Relative planet-mass change: a change of the planet mass divided by the planet mass, for example $10^{-9}$.
- Padded: the integration stops before the outer radius, and the remaining nodes, the padded nodes, get zero pressure. The residual is then $-P_t$.
- Sign pair: 2 neighbouring evaluated central pressures, with a finite negative residual at the lower one and a finite positive residual at the higher one, so $f$ changes sign between them. The higher one is the reached end.
- PALEOS: the tabulated equations of state for iron, MgSiO$_3$ and water that Zalmoxis uses as its standard ([EOS physics](../Explanations/eos_physics.md)).
- Unified and 2-phase PALEOS mantle: one table for all phases of MgSiO$_3$ (unified), or separate tables for the solid and the melt (2-phase). The mushy zone factor sets the solidus as a fraction of the liquidus (0.8 in the JAX cases).
- Wet mantle: a mantle with a dissolved volatile, here 1 % PALEOS H$_2$O.
- State (JAX cases): one set of $r$, $M$, $g$ and $P$ at which both codes evaluate the right-hand side.
- JAX path: an optional version of the structure integration in JAX, a Python library for compiled numerical code; it is off by default.
- Picard outer solver: the default outer loop. It solves the structure for a trial radius, compares the computed mass with the planet mass, and rescales the radius.
- Newton outer solver: the outer loop with Newton steps and a Brent fallback (`outer_solver = 'newton'`); PROTEUS uses it by default. It needs `relative_tolerance` at or below $10^{-7}$.
- Outer pass: one step of the outer loop, which integrates the structure at a trial outer radius $R_n$ and returns the mass $M_n$.
- Blend ramp: the outer passes that move the temperature from the linear first guess (blend 0) to the adiabat (blend 1). A full-adiabat pass is a pass at blend 1 on a temperature table built on a pass at blend 1.
- Newton tolerance (`newton_tol`, default $10^{-4}$): the Newton outer solver accepts a structure when its relative mass error is below this value and its pressure solve converged (`accepted_on = 'newton_tol'`). In an adiabatic temperature mode where Zalmoxis computes the temperature itself, the Newton root holds the linear first guess, so it only seeds the blend ramp to the adiabat, which runs at the mass tolerance `min(tolerance_outer, newton_tol)` and gives a Picard label.

## Governing equations

Every case on this page checks one part of the solution of these equations.
Zalmoxis solves three coupled ordinary differential equations (ODEs) for the radial structure of a self-gravitating, spherically symmetric body in hydrostatic equilibrium:

$$
\frac{dM}{dr} = 4\pi r^2 \rho(r)
$$

$$
\frac{dg}{dr} = 4\pi G \rho(r) - \frac{2\,g(r)}{r}
$$

$$
\frac{dP}{dr} = -\rho(r)\, g(r)
$$

The first equation adds the mass of each thin shell, the second is the derivative of $g = GM/r^2$, and the third says that the pressure falls outward by the weight of the material above.
The integration runs outward from the centre ($r = 0$) with the initial conditions $M(0) = 0$, $g(0) = 0$ and $P(0) = P_c$.

The gravity equation is singular at $r = 0$ because of the $2g/r$ term.
L'Hopital's rule gives the limit $dg/dr|_{r=0} = \frac{4}{3}\pi G \rho_c$, and `coupled_odes` in `zalmoxis.structure_model` uses this limit at $r = 0$.

An equation of state $\rho = \rho(P, T)$ closes the system.
The uniform-sphere, two-layer, conservation and binding-energy cases, and the uniform-sphere part of the grid and tolerance checks, fix $\rho$ to a constant.
The n = 1 polytrope case uses $P = K\rho^2$.
The full-solver part of the grid and tolerance checks, and the Earth benchmark and the mass-radius scaling on the [mass-radius page](mass_radius.md), use the Analytic EOS, the modified polytrope of Seager et al. (2007)[^cite-seager2007], $\rho(P) = \rho_0 + c P^n$, with fitted constants $\rho_0$, $c$ and $n$ for each material (this $n$ is not the index of the $n = 1$ polytrope).
None of these needs table data; the JAX table cases, the PALEOS Newton setups and the PALEOS setups of the sweep figure use the PALEOS tables.

## Uniform-density sphere

**Why this check exists.** The integrator of the three equations must use the right coefficients and signs. A wrong factor in $dM/dr$ or $dg/dr$ (for example $2\pi$ in place of $4\pi$, or $g/r$ in place of $2g/r$), or the wrong sign of $dP/dr$, makes the computed mass, gravity or pressure differ from the exact solution, and this test fails.

**What it tests.** The density is fixed: the test replaces `calculate_mixed_density()` by a function that returns a constant $\rho$, so no EOS table is used and only the RK45 integration of the three equations is tested. The setup is $\rho = 5000$ kg/m$^3$ (an average rocky planet), $R = 6.4 \times 10^6$ m (about $1\, R_\oplus$), $P_c = 360$ GPa, $N = 300$ radial grid points and `rtol` $= 10^{-10}$. The test makes six checks:

1. **Mass profile:** $|M_\mathrm{num} - M_\mathrm{exact}| / M_\mathrm{exact} < 10^{-6}$ at all $r > 0$.
2. **Gravity profile:** $|g_\mathrm{num} - g_\mathrm{exact}| / g_\mathrm{exact} < 10^{-6}$ at all $r > 0$.
3. **Pressure profile:** $|P_\mathrm{num} - P_\mathrm{exact}| / P_\mathrm{exact} < 10^{-6}$ where $P > 0$.
4. **Central boundary conditions:** $M(0) = 0$, $g(0) = 0$ and $P(0) = P_c$ to machine precision.
5. **Gravity slope at the centre:** the slope of the computed gravity over the first two grid steps matches $\frac{4}{3}\pi G \rho$ to $< 10^{-4}$. This is the slope that the integration produces, not the value that `coupled_odes` uses at $r = 0$.
6. **Pressure monotonicity:** $dP/dr < 0$ everywhere $P > 0$, as hydrostatic equilibrium requires with positive density and gravity.

**How we know the right answer.** For a sphere of constant density the equations have an exact closed-form solution, a textbook result with no free parameters:

$$
M(r) = \frac{4}{3}\pi \rho\, r^3, \qquad
g(r) = \frac{4}{3}\pi G \rho\, r, \qquad
P(r) = P_c - \frac{2}{3}\pi G \rho^2 r^2
$$

The mass grows as $r^3$ (the volume of a sphere), gravity is linear in $r$ (Gauss's law for a uniform sphere), and the pressure falls as a parabola (hydrostatic equilibrium with a gravity that grows linearly).

**Why it matters.** Every structure that Zalmoxis returns comes from this integrator. An error here enters every profile, for every equation of state.

![Uniform-density sphere: mass, gravity and pressure against radius, code and analytic](../assets/validation/zalmoxis_uniform_sphere.svg#only-light){ width="80%" }
![Uniform-density sphere: mass, gravity and pressure against radius, code and analytic](../assets/validation/zalmoxis_uniform_sphere_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The 3 panels show the enclosed mass, the gravity and the pressure against $r/R$. The lines are the analytic solution and the marks are the code values, every 15th node. The tests of the class assert all 3 profiles against the analytic solution to a relative tolerance of $10^{-6}$, too small to see. The pressure test draws the plot and asserts its pressure panel. The mass and gravity panels come from a second solve in the plot code; the mass and gravity tests assert them. A pass means that every mark lies on its line. The tests of the centre values and of the falling pressure are not drawn.

**Known limit.** This test does not check the value of the centre limit of $dg/dr$: when `coupled_odes` uses $\pi G \rho_c$ in place of $\frac{4}{3}\pi G \rho_c$ at $r = 0$, every test of the uniform-sphere, two-layer, conservation and binding-energy cases, the grid and tolerance checks and the $n = 1$ polytrope case still passes.

??? info "Test details"
    - Reference: closed-form solution of the uniform-density sphere
    - Tolerance: `rtol=1e-6` on the mass, gravity and pressure profiles; `rel=1e-4` on the central gravity slope
    - Test id: `tests/test_first_principles.py::TestUniformDensitySphere`
    - Command: `pytest -o "addopts=" tests/test_first_principles.py::TestUniformDensitySphere`

## Two-layer sphere

**Why this check exists.** Zalmoxis assigns each radial shell to a layer (core or mantle) by the enclosed mass. If this assignment is wrong, for example if the core ends at 90 % of its mass, the mass profile differs from the piecewise exact solution and this test fails. Wrong coefficients in $dM/dr$ or $dg/dr$ also make it fail.

**What it tests.** A sphere of two layers with constant densities: $\rho_\mathrm{core} = 13{,}000$ kg/m$^3$ (an iron-like core) and $\rho_\mathrm{mantle} = 4000$ kg/m$^3$ (a silicate-like mantle), with planet mass $M_p = 1\, M_\oplus$ and core mass fraction $f_c = 0.325$. The density jumps at the core-mantle boundary (CMB), the sharpest feature that the integrator must handle. The CMB radius is

$$
R_\mathrm{cmb} = \left(\frac{3\, f_c\, M_p}{4\pi \rho_\mathrm{core}}\right)^{1/3} = 3291 \text{ km}
$$

This is close to the Earth's CMB radius of 3480 km, and the planet radius is 6513 km (1.03 $R_\oplus$, or 1.02 times the mean Earth radius of 6371 km). The central pressure of this profile, 380 GPa, comes from integrating the hydrostatic equation over the known density profile. The integration starts 5 % above it, at 399 GPa, so that the pressure stays positive out to the surface. The test compares the enclosed mass with the exact solution and checks Gauss's law ($g = GM/r^2$) in both layers.

**How we know the right answer.** With two constant densities, the enclosed mass has an exact piecewise solution:

$$
M(r) = \begin{cases}
\frac{4}{3}\pi \rho_\mathrm{core}\, r^3 & r \le R_\mathrm{cmb} \\
f_c M_p + \frac{4}{3}\pi \rho_\mathrm{mantle} (r^3 - R_\mathrm{cmb}^3) & r > R_\mathrm{cmb}
\end{cases}
$$

**Why it matters.** Every Zalmoxis planet has at least a core and a mantle. A wrong layer boundary puts the wrong material at a radius, so the density, the radius and the pressure at the CMB are wrong.

![Two-layer sphere: enclosed mass against the piecewise solution, and gravity against Gauss's law](../assets/validation/zalmoxis_two_layer_sphere.svg#only-light){ width="80%" }
![Two-layer sphere: enclosed mass against the piecewise solution, and gravity against Gauss's law](../assets/validation/zalmoxis_two_layer_sphere_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The top panel shows the piecewise analytic mass (line) and the code mass (marks, every 20th node). The bottom panel shows $GM/r^2$ from the code mass (line) and the code gravity (marks). The dotted line is the core-mantle boundary. This test asserts the mass to a relative tolerance of $10^{-5}$, too small to see. The bottom panel comes from a second solve in the plot code. The Gauss's law test asserts it to $10^{-4}$ on the same setup. A pass means that every mark lies on its line.

??? info "Test details"
    - Reference: piecewise closed-form solution of the two-layer sphere
    - Tolerance: `rtol=1e-5` on the mass profile, `rtol=1e-4` on Gauss's law
    - Test id: `tests/test_first_principles.py::TestTwoLayerSphere`
    - Command: `pytest -o "addopts=" tests/test_first_principles.py::TestTwoLayerSphere`

## Conservation diagnostics

**Why this check exists.** Zalmoxis integrates $M(r)$ and $g(r)$ as two separate ODE components; it does not compute $g$ from $M$. A wrong factor in $dM/dr$ or $dg/dr$ makes them disagree, and the Gauss's law check fails. A wrong sign of $dP/dr$ breaks hydrostatic balance, and the balance check fails.

**What it tests.** Two conservation laws must hold at every point of any physically consistent solution, for any EOS. The test checks them on the uniform-density sphere above.

- **Gauss's law:** $g(r) = G\,M(r) / r^2$ for all $r > 0$. The relative residual $|g - GM/r^2| / (GM/r^2)$ measures how far the two integrated components drift apart.
- **Hydrostatic balance:** $dP/dr + \rho\, g = 0$ at every interior point. This is the defining equation of the solver. The test computes $dP/dr$ by central finite differences and divides the residual by $P_\mathrm{scale} = \frac{2}{3}\pi G \rho^2 R$, a pressure gradient scale in Pa/m. For the uniform sphere $P(r)$ is a parabola, so the central difference is exact and the residual shows the integration error alone.

**How we know the right answer.** Gauss's law and hydrostatic balance follow from the equations themselves, so the exact residual is zero.

**Why it matters.** These laws must hold for every planet that Zalmoxis computes, also where no closed-form solution exists.

![Pressure gradient of the uniform sphere against radius, code dP/dr and minus rho g](../assets/validation/zalmoxis_conservation_diagnostics.svg#only-light){ width="80%" }
![Pressure gradient of the uniform sphere against radius, code dP/dr and minus rho g](../assets/validation/zalmoxis_conservation_diagnostics_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The line is $-\rho g$ and the marks are the code $dP/dr$ by central differences, every 20th interior node, against $r/R$. The balance test asserts that their difference stays below $10^{-4}$ of the gradient scale $\frac{2}{3}\pi G \rho^2 R$. The difference is round-off, at most $1.5 \times 10^{-13}$ of that scale, so the marks lie on the line. A pass means that every mark lies on the line.

![Gravity of the uniform sphere against radius, code gravity and GM/r^2](../assets/validation/zalmoxis_gauss_law.svg#only-light){ width="80%" }
![Gravity of the uniform sphere against radius, code gravity and GM/r^2](../assets/validation/zalmoxis_gauss_law_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The line is $GM/r^2$ from the code mass and the marks are the code gravity, every 25th node, on the 300-point grid. The Gauss's law test asserts that they agree to a relative tolerance of $10^{-5}$, too small to see. A pass means that every mark lies on the line.

??? info "Test details"
    - Reference: Gauss's law $g = GM/r^2$ and hydrostatic balance $dP/dr + \rho g = 0$
    - Tolerance: `rtol=1e-5` on Gauss's law, largest relative hydrostatic residual below `1e-4`
    - Test id: `tests/test_first_principles.py::TestGaussLaw`, `tests/test_first_principles.py::TestHydrostaticBalance`
    - Command: `pytest -o "addopts=" tests/test_first_principles.py::TestGaussLaw tests/test_first_principles.py::TestHydrostaticBalance`

## Gravitational binding energy

**Why this check exists.** The cases above check the profiles point by point. An integral over the whole planet checks that the mass and density profiles agree as a whole. A wrong factor in $dM/dr$ makes this test fail.

**What it tests.** The gravitational binding energy $E_\mathrm{grav}$ \[J\], the energy needed to disperse the planet to infinity, is

$$
E_\mathrm{grav} = -\int_0^R \frac{G\,M(r)}{r}\, 4\pi r^2 \rho\, dr
$$

The test integrates it with the trapezoidal rule over the $M(r)$ and $\rho(r)$ profiles of the uniform-density sphere ($N = 500$).

**How we know the right answer.** For uniform density the integral has the closed form $E_\mathrm{grav} = -\frac{3}{5}\, G\, M_p^2 / R$. The numerical value agrees with it to $7 \times 10^{-6}$; the test asserts $< 1\%$.

**Why it matters.** The binding energy integrates the whole profile, so an error that adds up over the planet shows here even when each point looks right.

![Running binding-energy integral of the uniform sphere against radius, with the closed form](../assets/validation/zalmoxis_binding_energy.svg#only-light){ width="80%" }
![Running binding-energy integral of the uniform sphere against radius, with the closed form](../assets/validation/zalmoxis_binding_energy_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The line shows how the binding-energy integral builds up from the centre outward, from the code mass profile; it is not asserted. The test asserts the full integral (circle) against $-\frac{3}{5} G M_p^2 / R$ (square, with its error bar of 1 %). A pass means that the circle lies inside the error bar.

??? info "Test details"
    - Reference: uniform-sphere binding energy $E_\mathrm{grav} = -\frac{3}{5} G M_p^2 / R$
    - Tolerance: `rel=0.01`
    - Test id: `tests/test_first_principles.py::TestGravitationalBindingEnergy`
    - Command: `pytest -o "addopts=" tests/test_first_principles.py::TestGravitationalBindingEnergy`

The Earth benchmark and the mass-radius scaling are on the [mass-radius page](mass_radius.md).

## n = 1 polytrope

**Why this check exists.** The first 4 cases on this page fix the density. This case lets the density follow the pressure through an equation of state, as in a real planet. A wrong factor in $dM/dr$ or $dg/dr$, or the wrong sign of $dP/dr$, moves the surface radius or the integrated mass, and the test fails.

**What it tests.** `solve_structure` with the `Analytic:polytrope_n1` verification material, $P = K \rho^2$ (a polytrope of index $n = 1$, with $K$ a constant in Pa m$^6$ kg$^{-2}$). The test runs 600 radial points from the centre to $1.03\,R_\oplus$; the surface radius comes from extrapolating the density to zero. For $n = 1$ the radius $R = \pi \sqrt{K / 2\pi G}$ depends on $K$ only, not on the central density; $K$ is set so that $R$ is one Earth radius. The test also checks the mass: the central density is chosen so that the exact mass is $2\,M_\oplus$, and the integrated mass must match it. It runs the case at 1 and $2\,M_\oplus$ and checks that both give the same radius.

**How we know the right answer.** The Lane-Emden equation is the dimensionless hydrostatic equation of a polytrope. For $n = 1$ it has the closed-form solution $\rho(r) = \rho_c \sin\xi / \xi$ with the dimensionless radius $\xi = \pi r / R$, and the mass $M = 4\rho_c R^3/\pi$.

**Why it matters.** Here the density follows the pressure and the right answer is still known in closed form, so this case checks the link between the integrator and an equation of state.

![Density of the n = 1 polytrope against radius, code and Lane-Emden, with the surface radius](../assets/validation/zalmoxis_polytrope_n1.svg#only-light){ width="80%" }
![Density of the n = 1 polytrope against radius, code and Lane-Emden, with the surface radius](../assets/validation/zalmoxis_polytrope_n1_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The line is the Lane-Emden density $\sin\xi/\xi$ against $r/R_\oplus$ and gives context. The marks are the density that the test derives from the code pressure, over the central density, every 25th node, for 1 Earth mass. The test asserts the code surface radius (circle) against the exact radius $R_\oplus$ (square) to a relative tolerance of $10^{-4}$, too small to see. The cross is $\sqrt{2}\,R_\oplus$, the radius with $K$ off by a factor 2. A pass means that the circle sits in the square, far from the cross. The mass test and the mass-independence test are not drawn.

??? info "Test details"
    - Reference: the analytic $n = 1$ Lane-Emden solution of Chandrasekhar (1939)[^cite-chandrasekhar1939], named in the test docstring
    - Tolerance: surface radius within `rel=1e-4` of $1\,R_\oplus$; integrated mass within `rel=1e-4` of the seeded $2\,M_\oplus$; radius at $1\,M_\oplus$ and $2\,M_\oplus$ equal within `rel=1e-4` and within `rel=1e-3` of $1\,R_\oplus$
    - Test id: `tests/test_polytrope_verification.py`
    - Command: `pytest -o "addopts=" tests/test_polytrope_verification.py`

## Grid and tolerance checks

**Why this check exists.** A finer grid or a tighter tolerance must not make the solution worse. The test fails in these cases:

- the mass error at $N = 400$ reaches $10^{-8}$;
- the pressure error at $N = 400$ reaches $10^{-6}$;
- the mass error at `rtol` $= 10^{-12}$ reaches $10^{-10}$;
- the mass error grows by more than a factor 1.5 from one grid or tolerance step to the next, but only above $10^{-13}$, so at the measured errors of about $2 \times 10^{-15}$ this rule does not act;
- a full-solver run does not converge, or the radii at $N = 200$ and $N = 400$ differ by 1 % or more.

A wrong factor in $dM/dr$ or $dg/dr$, or the wrong sign of $dP/dr$, also makes the grid convergence test fail.

**What it tests.** The uniform-density sphere above on several grids and with several tolerances, and the full solver (the Picard outer solver and Brent's method) with the Analytic EOS at $1\,M_\oplus$ on 3 grids.

- **Grid convergence:** the uniform sphere at $N$ from 50 to 400 with `rtol` $= 10^{-10}$. The mass error is $2$ to $3 \times 10^{-15}$ and the pressure error about $4 \times 10^{-16}$ at every $N$. The uniform-sphere solution is a cubic and a parabola in $r$, which the integrator reproduces to round-off on every grid, so this case shows no convergence rate.
- **Full solver:** the planet radius is the same, $0.96512\,R_\oplus$ (6114.5 km), at $N = 100$, 200 and 400.
- **Tolerance convergence:** for `rtol` from $10^{-6}$ to $10^{-12}$ at $N = 200$ the mass error stays between $2$ and $4 \times 10^{-15}$, with no trend: the error is at round-off level for every tolerance tested.

**How we know the right answer.** The closed-form solution of the uniform-density sphere gives the error. For the full solver, the radius on the finer grid ($N = 400$) is the reference for the radius at $N = 200$ (self-consistency).

**Why it matters.** The default grid and tolerances of Zalmoxis must give a result that does not change when they are refined.

![Largest relative mass and pressure error of the uniform sphere against the number of grid points, with the asserted bounds](../assets/validation/zalmoxis_grid_convergence.svg#only-light){ width="80%" }
![Largest relative mass and pressure error of the uniform sphere against the number of grid points, with the asserted bounds](../assets/validation/zalmoxis_grid_convergence_dark.svg#only-dark){ width="80%" }

**How to read the plot.** Both axes are logarithmic. The circles are the asserted largest relative mass errors of the uniform sphere at $N$ = 50 to 400. The crosses are its pressure errors from a second solve in the plot code; the pressure test asserts them. All are round-off, below $3 \times 10^{-15}$, and are drawn at $10^{-14}$ so that the plot is the same on every platform. The dashed and dash-dotted lines are the bounds at $N = 400$: $10^{-8}$ for the mass and $10^{-6}$ for the pressure. The dotted line at $10^{-13}$ is the floor above which an error may grow by at most a factor 1.5 per step. A pass means that the marks at $N = 400$ lie below their bounds; the full-solver and tolerance tests are not drawn.

**Known limit.** The uniform sphere is solved to round-off at every $N$, so the test shows no order of convergence. All errors are below $10^{-13}$, so the factor 1.5 check is never applied.

??? info "Test details"
    - Reference: closed-form solution of the uniform-density sphere; for the full solver, the radius at $N = 200$ against $N = 400$ (self-consistency)
    - Tolerance: the error may rise by at most a factor 1.5 from one grid or tolerance step to the next above a `1e-13` floor; mass error below `1e-8` and pressure error below `1e-6` at $N = 400$; full-solver radius at $N = 200$ and $N = 400$ within 1%; mass error below `1e-10` at `rtol=1e-12`
    - Test id: `tests/test_first_principles.py::TestGridConvergence`, `tests/test_first_principles.py::TestToleranceConvergence`
    - Command: `pytest -o "addopts=" tests/test_first_principles.py::TestGridConvergence tests/test_first_principles.py::TestToleranceConvergence`

## JAX and numpy right-hand sides

**Why this check exists.** The optional JAX path (`use_jax = True`) evaluates the right-hand side of the structure equations in its own code, `zalmoxis.jax_eos.rhs.coupled_odes_jax`. If it differs from the numpy `zalmoxis.structure_model.coupled_odes` by more than the tolerance, a JAX run gives another structure than a numpy run, and these tests fail.

**What it tests.** The cases compare $dM/dr$, $dg/dr$ and $dP/dr$ of the two codes on the same states.

Each table case draws random states with a fixed seed: $r$ from $10^5$ to $6.4 \times 10^6$ m, $M$ from 0 to $1\, M_\oplus$, $g$ from 0.5 to 25 m/s$^2$ and $P$ from $10^6$ to $3 \times 10^{11}$ Pa, with a PALEOS iron core and a PALEOS MgSiO$_3$ mantle. A state counts when both codes return a nonzero right-hand side, and more than 100 states must count. JAX must never return a nonzero value where numpy returns zero.

The Tier column names the test group: unit tests need no table data, and integration tests need the PALEOS tables. The synthetic cases build small tables in memory, so they need no EOS files and run in the unit tier.

| Test id | Mantle | Tolerance | Tier |
|---|---|---|---|
| `tests/test_jax_unified_mantle_parity.py::test_unified_mantle_dry_parity` | unified MgSiO$_3$, mushy zone factor 0.8 | largest relative difference `1e-6` | integration |
| `tests/test_jax_unified_mantle_parity.py::test_unified_mantle_wet_parity` | unified MgSiO$_3$ with 1% PALEOS H$_2$O | `1e-5` | integration |
| `tests/test_jax_unified_mantle_parity.py::test_unified_wet_parity_without_melting_curves` | as above, without external melting curves | `1e-5` | integration |
| `tests/test_jax_wet_mantle_parity.py::test_coupled_odes_jax_wet_parity` | 2-phase MgSiO$_3$ (solid and melted sub-tables) with 1% PALEOS H$_2$O | `1e-5` | integration |
| `tests/test_jax_parity_synthetic.py::test_unified_kernel_parity_synthetic` | synthetic unified table, density kernel | `rel=1e-12` | unit |
| `tests/test_jax_parity_synthetic.py::test_rhs_parity_synthetic_unified_wet_and_dry` | synthetic unified mantle, dry and wet | `rtol=1e-10` | unit |

**How we know the right answer.** Two codes of the same equations must give the same $dM/dr$, $dg/dr$ and $dP/dr$, so the numpy `coupled_odes` is the reference (a cross-code check).

**Why it matters.** Every run with `use_jax = True` integrates the structure with the JAX right-hand side, also when PROTEUS calls Zalmoxis on the JAX path.

![Relative difference between the JAX and numpy right-hand sides](../assets/validation/jax_unified_mantle_dry_parity.svg#only-light){ width="80%" }
![Relative difference between the JAX and numpy right-hand sides](../assets/validation/jax_unified_mantle_dry_parity_dark.svg#only-dark){ width="80%" }

**How to read the plot.** Each marker is one compared state of the dry unified mantle case, with a PALEOS iron core and a unified PALEOS MgSiO$_3$ mantle. It shows the largest relative difference of the 3 right-hand-side terms between `coupled_odes_jax` and `coupled_odes`, against the pressure of that state. The dashed line is the tolerance, 1e-6. A difference below half the tolerance is drawn at that floor, because its exact value changes between platforms. On the machine that made the plot, the largest difference is 1.4e-7. A pass means that every marker lies below the dashed line.

**Known limit.** In at most 4 of 200 states numpy returns a nonzero value where JAX returns zero: the numpy tables fall back to the nearest table cell outside their range, and the JAX path has no such fallback.

??? info "Test details"
    - Reference: numpy `coupled_odes` (cross-code check).
    - Tolerance: largest relative difference `1e-6` over all compared states.
    - Test id: `tests/test_jax_unified_mantle_parity.py::test_unified_mantle_dry_parity`
    - Command: `pytest -o "addopts=" -m reference_pinned tests/test_jax_unified_mantle_parity.py --validation-plots`

## Surface boundary condition of the pressure solve

**Why this check exists.** Brent's method can return a central pressure for which the integration stops below the surface, and the solve must then refine that root. If the refinement does not run, or keeps the padded root, the profile ends with a pressure of 0 at the outer radius instead of the surface pressure.

**What it tests.** The central pressure $P_c$ is the root of the residual $f(P_c) = P(R) - P_t$. Here $P(R)$ is the pressure at the outer radius $R$, and $P_t$ is the target surface pressure, 1 atm here.

A trial $P_c$ whose integration stops before the outer radius gives $f = -P_t$, and the remaining points get zero pressure: this is a padded profile. Brent's method stops when its bracket is narrower than `xtol` $= 10^6$ Pa, and it can then return such a padded root. The solve then refines the root ([padded accepts](../Explanations/process_flow.md#padded-accepts)).

The test uses an analytic residual that needs no structure solve. Below a central pressure $x_0$ the integration stops; above it, $P(R)$ rises from $0.95\,P_t$ at a slope $dP(R)/dP_c$ of 0.47. The Brent root of the first solve is then padded, and the refined root must reach the outer radius and miss $P_t$ by at most $0.1 P_t$.

A second test checks the same rule with real structure solves. It uses the Analytic 1 Earth-mass planet with the Newton outer solver at `relative_tolerance` $10^{-9}$, and shifts the planet mass by a relative amount of 0 and $\pm 10^{-9}$ until a pressure solve has a padded Brent root. The refined root must then miss $P_t$ by at most $0.1 P_t$. The accepted profile must not be padded, and it must miss $P_t$ by at most $P_t$.

**How we know the right answer.** The boundary condition $P(R) = P_t$ gives the right answer. A profile that reaches the outer radius with $P(R)$ close to $P_t$ meets it. A padded profile has $P(R) = 0$, so it misses by the full $P_t$.

**Why it matters.** The accepted profile is the structure that Zalmoxis returns, for example to PROTEUS. Its outer point must have the surface pressure that the run asks for, not a padded pressure of 0.

![Surface pressure of the evaluated central pressures of one pressure solve](../assets/validation/pressure_root_boundary_condition.svg#only-light){ width="80%" }
![Surface pressure of the evaluated central pressures of one pressure solve](../assets/validation/pressure_root_boundary_condition_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the central pressure relative to the Brent root of the first solve, in MPa, and the y-axis is the surface pressure $P(R)$ in bar. Each circle is one evaluated central pressure. The square is the padded Brent root of the first solve ($P(R) = 0$), and the star is the refined root, which reaches the outer radius. Evaluations above 3 bar are drawn as triangles at the top of the axis, and the dashed line is $P_t$ = 1 atm. A pass means that the star lies within $0.1 P_t$ of the dashed line.

**Known limit.** Whether a real solve pads depends on round-off, so the test with real structure solves has no plot.

??? info "Test details"
    - Reference: the boundary condition $P(R) = P_t$.
    - Tolerance: $\lvert P(R) - P_t \rvert \le 0.1 P_t$ at the refined root, which is reached.
    - Test id: `tests/test_pressure_root.py::test_the_refined_root_meets_the_surface_boundary_condition`
    - Command: `pytest -o "addopts=" -m reference_pinned tests/test_pressure_root.py --validation-plots`
    - Test id of the real-solve test: `tests/test_pressure_root_solves.py::test_the_accepted_root_meets_the_surface_boundary_condition`
    - Tolerance of the real-solve test: $\lvert P(R) - P_t \rvert \le 0.1 P_t$ at the refined root of a padded Brent root; the accepted profile is not padded and misses $P_t$ by at most $P_t$.
    - Command of the real-solve test: `pytest -o "addopts=" tests/test_pressure_root_solves.py::test_the_accepted_root_meets_the_surface_boundary_condition`

## Surface boundary condition in every Newton setup

**Why this check exists.** The refinement must also work on the real equations of state of a planet, not only on an analytic residual. If it fails there, the accepted profile of a Newton run ends padded, or it misses the surface pressure by more than $P_t$.

**What it tests.** The test runs the 6 Newton setups: the Analytic EOS (equation of state) at 1 Earth mass, PALEOS at 1, 5 and 10 Earth masses, and PALEOS on the JAX path with $T(r)$ from a numpy Picard solve at 1 and 3 Earth masses.

The Analytic setup and the 2 JAX setups, which take $T(r)$ from the caller, must accept on the Newton tolerance. The 3 numpy PALEOS setups are adiabatic, so their Newton root seeds the blend ramp, and they must accept on the full adiabat with a strict or mass-only accept and a mass error below $10^{-4}$. Every reached profile must miss $P_t$ by at most $P_t$, and no profile may have a padded node, except on the JAX path at 1 Earth mass and in the 3 numpy PALEOS setups.

On these paths $P(R)$ varies irregularly from one trial central pressure to the next near the surface stop, where the integration ends before the outer radius, and the accepted profile can be padded; which of the 2 a solve gives depends on round-off: over planet-mass changes of $k \times 10^{-12}$ ($k$ = 0, 1, 2), the numpy PALEOS 1 Earth-mass setup is padded once and reaches $R$ twice. The case then requires one of 2 things: the reached end of the first sign pair of that pressure solve misses $P_t$ by more than $P_t$, or the solve has no sign pair.

**How we know the right answer.** As in the case above, the boundary condition $P(R) = P_t$ gives the right answer.

**Why it matters.** The Newton outer solver at `relative_tolerance` $10^{-9}$ is the outer solver that PROTEUS uses by default. These setups run it on the PALEOS tables from 1 to 10 Earth masses, on the numpy and the JAX path.

![Surface pressure miss of the accepted profile for 26 solver setups](../assets/validation/pressure_root_sweep.svg#only-light){ width="80%" }
![Surface pressure miss of the accepted profile for 26 solver setups](../assets/validation/pressure_root_sweep_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the number of the setup, for 26 setups built from `input/default.toml` and numbered from 0 to 25. The y-axis is $\lvert P(R) - P_t \rvert / P_t$ of the accepted profile, on a log scale. Circles are reached profiles, squares are padded profiles (at 1, since $P(R) = 0$), and a cross above the cap marks a setup without an accepted profile. The dashed line is the cap of $P_t$. Setups 4, 9, 10, 13, 14 and 25 use the Newton outer solver at `relative_tolerance` $10^{-9}$, and the test asserts only these 6; the other 20 setups are shown for comparison; [padded accepts](../Explanations/process_flow.md#padded-accepts) lists every setup. A pass means that every Newton setup is a circle at or below the dashed line. Only setups 9, 10, 14 and 25 may be squares, and only under the rule above.

**Known limit.** Whether a profile ends padded depends on round-off, as the 3 starts above show. The miss of a reached profile can also change, so another platform can give other values. In the plotted run the 4 reached Newton setups 4, 13, 14 and 25 miss by 0.046, 0.18, 0.050 and 0.25 $P_t$. The test skips when the PALEOS tables are not under `data/`.

??? info "Test details"
    - Reference: the boundary condition $P(R) = P_t$.
    - Tolerance: the Analytic and JAX setups accept on the Newton tolerance and the numpy PALEOS setups on the full adiabat; $\lvert P(R) - P_t \rvert \le P_t$ for every reached profile; no padded node except on the JAX path at 1 Earth mass and in the numpy PALEOS setups; the padded flag matches the root of the last pressure solve.
    - Test id: `tests/test_pressure_root_solves.py::test_every_newton_setup_meets_the_surface_boundary_condition`
    - Command: `pytest -o "addopts=" -m accept_solves_newton_setups -k every_newton_setup tests/test_pressure_root_solves.py`; the figure: `python -m tools.validation.pressure_root_sweep run <dir> -j 2`, then `python -m tools.validation.pressure_root_sweep plot <dir> --assets <dir>`.

## Newton and Picard on the PROTEUS initial structure

**Why this check exists.** PROTEUS starts a coupled run from a Zalmoxis structure solved with the Newton outer solver. The run starts from a wrong state if the Newton result keeps its linear first guess of the temperature, or if a solve returns a mantle temperature that is not the adiabat of its own pressure. Each of the 2 tests below catches one of these faults.

**What it tests.** Each setup has 1 Earth mass, the PALEOS iron core and PALEOS MgSiO$_3$ mantle, and the temperature mode `adiabatic_from_cmb`: the mantle follows the adiabat upward from a fixed CMB temperature $T_\mathrm{cmb}$. The setup `all_options` has a core mass fraction of 0.325 and anchors of $T_\mathrm{surf}$ = 3936 K and $T_\mathrm{cmb}$ = 6121 K; the setup `dummy` has 0.55, 4000 K and 6000 K. Both run on the JAX path with 150 layers, `relative_tolerance` $10^{-9}$ and `absolute_tolerance` $10^{-10}$, the settings in `tools/validation/proteus_ic_solvers.py`.

- The first test solves each setup twice, with the Picard and with the Newton outer solver. The mantle temperatures at 39 mass points between the CMB and the surface may differ by at most 5 K, and the CMB pressures by at most 0.1 GPa.
- The second test solves `dummy` and `all_options` with Newton, and `all_options` with Picard at the Newton mass tolerance of $10^{-4}$. It computes the adiabat again on the returned pressure. At each mantle node the returned temperature may differ from it by at most 1 K. Two nodes have their own bounds (see the known limit below): 6 K from $T_\mathrm{cmb}$ at the first mantle node, the CMB anchor, and 25 K at the top node.

**How we know the right answer.** The 2 outer solvers solve the same equations for the same adiabat, so 2 correct results differ only by the error that the mass tolerance of each solve allows. This is a cross-solver check with no reference solution. The second test is an internal consistency check, not an independent reference: the solve takes its temperature from the adiabat of its pressure, so the adiabat that the same `compute_adiabatic_temperature` gives on the returned structure is the temperature it must return.

**Why it matters.** PROTEUS uses the Newton outer solver by default, and the temperature and pressure of this structure set the initial state of the interior evolution.

![Mantle temperature difference between the Newton and Picard results of 2 PROTEUS setups](../assets/validation/proteus_ic_solvers.svg#only-light){ width="70%" }
![Mantle temperature difference between the Newton and Picard results of 2 PROTEUS setups](../assets/validation/proteus_ic_solvers_dark.svg#only-dark){ width="70%" }

**How to read the plot.** The x-axis is the mantle mass coordinate, 0 at the CMB and 1 at the surface. The y-axis is the absolute difference between the Newton and the Picard mantle temperature in K, on a log scale; differences below $10^{-3}$ K are drawn at $10^{-3}$ K. Each setup has a solid line, and the broken line is the 5 K limit. A pass means that both lines stay below the limit. The CMB pressures differ by 0.061 GPa in `all_options` and by 0.065 GPa in `dummy`, inside the limit of 0.1 GPa.

**Known limit.** Two nodes of the returned mantle do not follow the adiabat of their own pressure. The Picard solve of the first test also carries the error of its mass tolerance.

- The CMB anchor node reads one T(P) table that holds $T_\mathrm{surf}$ on the core nodes and $T_\mathrm{cmb}$ at the CMB, so its temperature takes a small part of that jump. It lies 5.41 K below $T_\mathrm{cmb}$ in `all_options` with Newton, 5.42 K with Picard at $10^{-4}$, and 0.23 K in `dummy` with Newton. The next node is at most 0.1 K off.
- The adiabat takes no step from or to a node with a pressure below $10^5$ Pa, so the top node of a setup with a surface pressure near 1 atm (`dummy`) can lose its step, 17 to 22 K in the measured passes of this setup. In the solves of the second test the top node is 0.06 K (`dummy`) and 0.08 K (`all_options`) off the adiabat.
- With the surface anchor (`temperature_mode = "adiabatic"`) the same skip moves the whole profile, since the adiabat is integrated inward from the surface. In `dummy` with 60 layers and the Newton outer solver, 2 of 26 starts whose planet mass differs by $k \times 10^{-12}$ skip the top step: their central temperature is 111 K lower, the mantle up to 51 K cooler and the radius 10 km smaller. Over 32 such starts (26 with the Newton and 6 with the Picard outer solver) the returned top node lies 56 to 57 K off the adiabat in 14 and at most 0.6 K off in the other 18. Neither test uses this mode.
- These bounds hold for these 2 setups. The start-up solves of the PROTEUS grid go beyond them in the places that [Secant tail and seeded re-solve on the PROTEUS grid start-up](#secant-tail-and-seeded-re-solve-on-the-proteus-grid-start-up) lists.
- The first test runs Picard at the mass tolerance of the PROTEUS setups, $3 \times 10^{-3}$. There the `all_options` mantle lies on average 1.6 K below the adiabat of its own pressure. At $10^{-4}$ the difference is 0.1 K or less.

??? info "Test details"
    - Reference: the Picard result of the same setup (cross-solver check); the adiabat that the same `compute_adiabatic_temperature` gives on the returned pressure (internal consistency check).
    - Tolerance: 5 K at 39 mantle mass points and 0.1 GPa at the CMB; 1 K at the mantle nodes, 6 K from $T_\mathrm{cmb}$ at the CMB anchor node and 25 K at the top node.
    - Test id: `tests/test_accepted_on.py::test_newton_gives_the_mantle_adiabat_of_picard_on_the_proteus_initial_structure`, `tests/test_accepted_on.py::test_the_returned_mantle_temperature_is_the_adiabat_of_the_returned_pressure`
    - Markers: `smoke` and `accept_solves_long` (first test), `smoke` and `accept_solves` (second test); both run in the nightly workflow. They carry no `reference_pinned` marker, since their reference is another solve, not a published value.
    - Command: `pytest -o "addopts=" -m accept_solves_long tests/test_accepted_on.py -k proteus_initial_structure` and `pytest -o "addopts=" -m accept_solves tests/test_accepted_on.py -k returned_mantle_temperature`; the figure: `python -m tools.validation.proteus_ic_solvers run <dir>`, then `python -m tools.validation.proteus_ic_solvers plot <dir> --assets <dir>`.

## Secant tail and seeded re-solve on the PROTEUS grid start-up

**Why this check exists.** Two rules cut the cost of the start-up structure solves of a PROTEUS run: secant steps on the radius at the end of the blend ramp, and a re-solve that starts from the seed, the `radii`, `pressure` and `mass_enclosed` of the solve before (`adiabat_seed`).
A wrong secant step can leave the accepted structure farther from the adiabat of its own planet than the plain ramp does, and a seeded solve could accept a pass whose temperature table no pass of the solve built.
The test fails when a returned structure is outside the tolerances below, when its accept label is not `strict`, or when its pass is not on a table of blend 1.

**What it tests.** The setup is the first structure solve of the PROTEUS grid test case with a core mass fraction of 0.95: 0.7 Earth masses, the PALEOS iron core and MgSiO$_3$ mantle, `temperature_mode = "adiabatic_from_cmb"`, 150 layers, the Newton outer solver on the numpy path.
The first start-up re-solve follows, with a surface pressure of 51.3 MPa, a planet mass 0.07 % lower and a core mass fraction of 0.9505316; it runs once from the seed of the first solve and once without it.
The first solve has a surface pressure of 1 atm (101325 Pa).
A repeat of the first solve at a planet mass $2 \times 10^{-12}$ higher runs in the nightly job of the stored reference; on the development machine its top node loses the adiabat step (see the known limit below).
Each solve is compared with REF, the same solve at a mass tolerance of $10^{-6}$.
The lag of a node is the difference between the returned temperature at that node and the adiabat computed again on the returned pressure.
Relative differences are $|x - x_\mathrm{REF}| / |x_\mathrm{REF}|$.

- The mantle temperature at 39 mass points, from the lowest to the second-highest of the 12 mantle nodes, differs from REF by at most 1 K.
- The temperature of the top mantle node differs from REF by at most 1 K in the re-solves; in the first solves it is not compared (see the known limit below).
- The outer radius, the central pressure and the core radius differ from REF by at most $2 \times 10^{-4}$ (relative).
- The CMB pressure differs from REF by at most $4 \times 10^{-3}$ (relative). This is $2\,\mathrm{tol} / (1 - f_\mathrm{core})$, with the mass tolerance $\mathrm{tol} = 10^{-4}$ and the core mass fraction $f_\mathrm{core} = 0.95$: the core mass is fixed, so the mass error sits in the mantle and moves the CMB pressure by that error divided by the mantle mass fraction $1 - f_\mathrm{core}$. The core radius and the CMB pressure are read at the core mass by interpolation in mass.
- The lag is at most 6 K at the CMB node and 1 K at the other mantle nodes in every solve, and at most 25 K at the top node in the 2 re-solves.
- The seeded and the unseeded re-solve differ by at most 2.5 K at the 39 mass points and at the top node, and $5 \times 10^{-4}$ in radius.
- The mass-weighted mean temperature of the core nodes differs from REF by at most 10 K in every solve.
  The last core node, just below the CMB node, differs by at most 700 K: it reads the edge of the mantle temperature table (see the known limit below).
  Every other core node differs by at most 1 K. Each core node is compared with REF interpolated in mass over the core nodes of REF, not across the temperature step at the CMB node of REF.
- The accept label of every solve is `strict`, and the accepted pass is on a table of blend 1.
- The first solve needs at most 3 passes on a full-adiabat table. The seeded re-solve needs at most 4 passes and at most half the structure integrations of the unseeded one.

**How we know the right answer.** REF runs until its mass error is 100 times smaller than the mass tolerance of the ramp, $10^{-4}$.
So REF stands for the converged structure; this is a convergence check, not an external reference.
REF is solved once and stored in `tests/data/grid_start_up_reference.json`, with the md5 of both EOS tables and a hash of the Zalmoxis source that solved it (every Zalmoxis module the solve loads, without `_version.py`, which the install writes with the commit in it).
The test stops when a table differs from the stored one.
A separate nightly job solves REF again and fails when it moves by more than 0.1 K in the mantle, $10^{-5}$ in radius, central pressure or core radius, or $4 \times 10^{-5}$ in CMB pressure, or when the source hash differs.
Then `python -m tools.validation.grid_start_up_reference` writes the file again.

**Why it matters.** These solves set the radius, the core size and the initial mantle temperature of every PROTEUS run with Zalmoxis.

![Relative mass error per outer pass of the first solve and of the re-solve with and without the seed](../assets/validation/grid_start_up_ramp.svg#only-light){ width="70%" }
![Relative mass error per outer pass of the first solve and of the re-solve with and without the seed](../assets/validation/grid_start_up_ramp_dark.svg#only-dark){ width="70%" }

**How to read the plot.** The x-axis is the outer pass of the solve that returns.
In the first solve and in the re-solve without the seed, pass 1 is the pass at the radius of the Newton root.
Passes 2 to 5 blend in the adiabat, and the passes on a full-adiabat table start at pass 6.
The y-axis is the relative mass error $|M_n - M_p| / M_p$ on a log scale, with errors below $10^{-7}$ drawn at $10^{-7}$.
Circles are the first solve, open squares the re-solve without the seed and triangles the re-solve with it.
The broken line is the mass tolerance $10^{-4}$; the dotted line is twice that, below which the secant steps stop.
The test asserts the pass counts; the error values are context.
A pass means that the first-solve curve ends below the broken line after at most 3 passes from pass 6, and the seeded curve ends below it within 4 passes.

**Known limit.** The tests bound the lag for the 2 setups of [Newton and Picard on the PROTEUS initial structure](#newton-and-picard-on-the-proteus-initial-structure) and for the setup above, 1 of the 4 PROTEUS grid cases.
Replays of the saved start-up inputs of all 4 grid cases (core mass fraction 0.35 and 0.95; 1000 and 2000 ppmw H, parts per million by weight of hydrogen), taken from a PROTEUS grid run and outside the tests of this repository, give these values; the row of the re-solves at core mass fraction 0.95 comes from the grid start-up test of this repository.
The first 5 rows exceed a bound; the other 7 are not bounded or lie inside one.

| Case | Quantity | Value | Bound |
|---|---|---|---|
| Core mass fraction 0.35, first solve | lag at the CMB node | 93.2 K | 6 K |
| Core mass fraction 0.35, first solve | lag at the other mantle nodes | 2.09 K | 1 K |
| Core mass fraction 0.35, first solve | mantle temperature from REF | 2.17 K | 1 K |
| Core mass fraction 0.35, re-solves | lag at the mantle nodes | up to 1.36 K | 1 K |
| Core mass fraction 0.35, re-solves | mantle temperature from REF, without the seed / with it | up to 1.47 K / 1.25 K | 1 K |
| Core mass fraction 0.95, first solve at 1 atm | lag at the top node | 28.8 K | none for this solve |
| Core mass fraction 0.95, first solve at 1 atm | top node from REF | 28.65 to 28.81 K | not compared |
| Core mass fraction 0.95, first solve | last core node from REF, secant tail / cube-root rule | 587 K / 17.6 K | 700 K, inside |
| Core mass fraction 0.95, re-solves | last core node from REF, unseeded / seeded | 526 K / 0 K | 700 K, inside |
| Core mass fraction 0.95, first solve | mass-weighted mean core temperature from REF, secant tail / cube-root rule | 4.5 K / 0.14 K | 10 K, inside |
| All 4 cases, every solve | CMB pressure from REF, secant tail / cube-root rule | up to $9.4 \times 10^{-4}$ / $1.9 \times 10^{-3}$ | $4 \times 10^{-3}$ at core mass fraction 0.95, inside |
| Core mass fraction 0.35, 2000 ppmw H, seeded re-solve | lag at the CMB node | up to 5.4 K | 6 K, inside |

- At core mass fraction 0.35 the cube-root rule converges in 2 full-adiabat passes, so the secant tail does not step and the first solve is the same as without it.
- At 1 atm the adiabat step of the top node flips with round-off (the skip below $10^5$ Pa described under [Newton and Picard on the PROTEUS initial structure](#newton-and-picard-on-the-proteus-initial-structure)). In 4 of 6 starts whose planet mass differs by $k \times 10^{-12}$ ($k$ = 0 to 5) the top node lies 28.65 K from REF; in 2 of them, and in REF, it also lags its own adiabat by 28.8 K. The node below it lies within 0.15 K of REF in all 6 starts. On a Linux runner (AMD EPYC 7763) the reference solved again lies 28.81 K from the stored one at this node. The top node of a first solve thus takes one of 2 states, about -0.15 K or 28.65 to 28.81 K from REF, and a bound there only tests whether the step flipped. The tests leave it out of the comparison with REF in the first solves and bound it in the re-solves: 1 K from REF and a lag of at most 25 K.
- The core nodes and the CMB node read one T(P) table with the mantle, and the solve builds that table on the pressure of the pass before.
  The last core node (index 137, 0-based, of 150 nodes at core mass fraction 0.95) therefore reads the edge of that table.
  With the secant tail, which moves the CMB pressure more from pass to pass, this node lies 587 K above REF in the first solve, 526 K in the unseeded re-solve and 0 K in the seeded re-solve; with the cube-root rule it lies 17.6 K above REF in the first solve.
  REF has the core temperature at this node.
  The node is in the cell below the CMB, which holds 0.7 % of the planet mass.
  So the mass-weighted mean temperature of the core nodes lies only 4.5 K above REF, and the core radius stays within $3 \times 10^{-5}$ of REF.
  The tests bound the mean at 10 K, this node at 700 K and the other core nodes at 1 K; the other core nodes lie 0 K from REF in the 3 solves of the test.
  One more pass on the accepted structure gives this node the core temperature again (4000 K, measured); the solve does not do it.
- In the `adiabatic_from_cmb` mode every core node has the surface temperature (4000 K here) and the energetics module computes the core temperature. A core at the CMB temperature (6000 K) gives a larger core radius and a lower mean core density inside it; [Zalmoxis issue 131](https://github.com/FormingWorlds/Zalmoxis/issues/131) gives their size in the first solve of this case.

??? info "Test details"
    - Reference: REF, the same solve at a mass tolerance of $10^{-6}$ (convergence check), stored in `tests/data/grid_start_up_reference.json`.
    - Tolerance: 1 K at 39 mantle mass points up to the second-highest mantle node; top mantle node 1 K from REF in the re-solves, not compared in the first solves; $2 \times 10^{-4}$ in outer radius, central pressure and core radius; $4 \times 10^{-3}$ in CMB pressure; lags of 6 K (CMB node) and 1 K (other mantle nodes) in every solve and 25 K (top node) in the re-solves; seeded against unseeded 2.5 K and $5 \times 10^{-4}$; mass-weighted mean core temperature 10 K, last core node 700 K and the other core nodes 1 K from REF; at most 3 full-adiabat passes in the first solve, at most 4 passes and half the integrations in the seeded re-solve.
    - Test id: `tests/test_accepted_on.py::test_the_secant_tail_and_the_seed_keep_the_converged_adiabat_of_the_grid_start_up`
    - Repeat of the first solve: `tests/test_accepted_on.py::test_a_first_solve_at_a_planet_mass_2e12_higher_keeps_the_stored_reference` (markers `slow`, `accept_solves_reference` and `physics_invariant`; nightly job `accept_solves_reference`, 3600 s per test).
    - Tests of the rules alone, on analytic densities:
        - `tests/test_ramp_tail.py`
        - `tests/test_accepted_on.py::test_the_secant_tail_reaches_the_mass_in_3_full_adiabat_passes`
        - `tests/test_accepted_on.py::test_the_secant_steps_above_2_tolerances_until_a_secant_step_does_not_shrink_the_error`
        - `tests/test_accepted_on.py::test_a_ramp_without_the_secant_key_takes_no_secant_step`
        - `tests/test_accepted_on.py::test_a_seeded_solve_accepts_its_second_pass`
        - `tests/test_accepted_on.py::test_the_first_seeded_pass_carries_the_seed_label_and_is_not_accepted`
        - `tests/test_accepted_on.py::test_the_seed_table_pass_starts_no_secant`
    - Markers: `smoke`, `accept_solves_long`, `physics_invariant` and `reference_pinned`; it runs in the nightly workflow. It writes its plot only with `--validation-plots` and does not compare it with the committed pair, because the number of passes can change with round-off.
    - Plotted run (can change with round-off): in the first solve the error rises to about $3 \times 10^{-3}$ in passes 2 to 5 and is $2.4 \times 10^{-3}$ at pass 6; the cube-root rule reaches $1.7 \times 10^{-3}$ at pass 7 and the secant step $1.4 \times 10^{-5}$ at pass 8. A replay of the same first solve with the cube-root rule only (the solver before the secant tail) takes 10 passes at blend 1, from $2.4 \times 10^{-3}$ to $8.5 \times 10^{-5}$. The re-solve takes 4 passes from the seed against 8 without it (1 at the Newton root, 4 blend passes, 3 full-adiabat passes).
    - Command: `pytest -o "addopts=" -m accept_solves_long tests/test_accepted_on.py -k grid_start_up --validation-plots`.
    - Stored REF: `tests/test_accepted_on.py::test_the_stored_grid_start_up_reference_is_the_converged_solve_of_the_source` solves it again (markers `slow`, `accept_solves_reference` and `physics_invariant`; nightly job `accept_solves_reference`, 7800 s per test, 150 min per job). Tolerance: 0.1 K in the mantle up to the second-highest node, top node 1 K in the re-solve (not compared in the first solve), $10^{-5}$ in radius, central pressure and core radius, $2\,\mathrm{tol}_\mathrm{REF} / (1 - f_\mathrm{core}) = 4 \times 10^{-5}$ in CMB pressure, and the stored source hash. Command to write it again: `python -m tools.validation.grid_start_up_reference`; the hash leaves out the files in `GENERATED` of that script (`_version.py`).

## Reproducing the results

The tests draw the plots of these cases. This command writes them again to `docs/assets/validation/`:

```console
pytest -o "addopts=" tests/test_first_principles.py tests/test_polytrope_verification.py --validation-plots
```

Without `--validation-plots` the tests write nothing: each compares the data of its plot with the committed pair, and a missing or stale file fails the test.

[^cite-seager2007]: S. Seager, M. Kuchner, C. A. Hier-Majumder, B. Militzer, *[Mass-Radius Relationships for Solid Exoplanets](https://doi.org/10.1086/521346)*, The Astrophysical Journal, 669(2), 1279-1297, 2007. [SciX](https://scixplorer.org/abs/2007ApJ...669.1279S/abstract).
[^cite-chandrasekhar1939]: S. Chandrasekhar, *An Introduction to the Study of Stellar Structure*, book (no DOI), Chicago, 1939. [SciX](https://scixplorer.org/abs/1939isss.book.....C/abstract).
