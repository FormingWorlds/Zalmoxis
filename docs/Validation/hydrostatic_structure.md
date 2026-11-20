# Hydrostatic structure

## What this page checks

This page checks how Zalmoxis computes the inside of a planet in hydrostatic equilibrium: the mass, the gravity and the pressure at each radius.
Validation here means that we run the code on a case where another source gives the right answer, and check that the code gives that answer within a stated tolerance.
If this part of the code is wrong, every structure that Zalmoxis returns is wrong, also the radius and the pressure and gravity at each depth that it gives to PROTEUS.
The right answer comes from one of these sources:

- a closed-form solution or a conservation law;
- a second implementation of the same equations (the JAX and the numpy code, or the Newton and the Picard outer solver);
- the same solve at a tighter tolerance;
- a condition that the result must meet: the surface pressure, the iron melting curve at the inner-core boundary, or the core temperature table that the caller gives.

Each case below says why the check exists, what it tests, how we know the right answer and why it matters, and ends with a box "Test details" with the reference, the tolerance, the test id and the command.
The [overview](index.md) lists every case of the Validation tab.

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
- ICB: the inner-core boundary of a two-layer core, between the solid inner core and the liquid outer core. $M_\mathrm{ic}$ and $M_\mathrm{core}$: the inner-core mass and the core mass; $w_\mathrm{oc}$: the mass fraction of a light element in the outer core.
- Liquidus rule: the ICB lies where the liquid of the outer core meets an iron melting curve $T_\mathrm{liq}(P)$ \[K\] (`core_liquidus`), here the pure-iron curve of Anzellini et al. (2013)[^cite-anzellini2013]. On an adiabat the rule uses the liquid branch; on a given profile (isothermal, linear, prescribed) it uses the profile. The mass rule sets the ICB by an inner-core mass fraction (`inner_core_mass_fraction`).
- Liquid branch $T_\mathrm{l}(P)$: the outer-core adiabat continued below the ICB to the centre with the liquid side.
- Superheat $S = T_\mathrm{l}(P_\mathrm{ICB}) - T_\mathrm{liq}(P_\mathrm{ICB})$ \[K\]: how far the liquid branch lies above the melting curve at the ICB pressure.
- `tolerance_outer`: the convergence tolerance of the outer loop, default $10^{-3}$. It limits the relative mass error $\lvert M_\mathrm{calc} - M_\mathrm{target} \rvert / M_\mathrm{target}$. With `core_liquidus` it also limits the distance of the ICB from the ICB of its adiabat and, on a given profile, from the liquidus root, both as a fraction of the core mass.

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

The Analytic setup and the 2 JAX setups, which take $T(r)$ from the caller, must accept on the Newton tolerance. The 3 numpy PALEOS setups are adiabatic, so their Newton root seeds the blend ramp, and they must accept on the full adiabat with a strict or mass-only accept and a mass error below $10^{-4}$. Every reached profile must miss $P_t$ by at most $P_t$, and no profile may have a padded node, except on the JAX path and in the 3 numpy PALEOS setups.

On the JAX path and in the 3 numpy PALEOS setups, $P(R)$ varies irregularly from one trial central pressure to the next near the surface stop, where the integration ends before the outer radius, and the accepted profile can be padded. Round-off decides whether a solve gives a padded or a reached profile:

- numpy PALEOS at 1 Earth mass, planet-mass changes of $k \times 10^{-12}$ ($k$ = 0, 1, 2): padded once, reaches $R$ twice;
- JAX at 3 Earth masses, $k$ = 0 to 4: reaches $R$ on macOS arm64, padded at $k$ = 0 on the Linux runner.

A padded profile passes the case only if the reached end of the first sign pair of that pressure solve misses $P_t$ by more than $P_t$, or if the solve has no sign pair.

**How we know the right answer.** As in the case above, the boundary condition $P(R) = P_t$ gives the right answer.

**Why it matters.** The Newton outer solver at `relative_tolerance` $10^{-9}$ is the outer solver that PROTEUS uses by default. These setups run it on the PALEOS tables from 1 to 10 Earth masses, on the numpy and the JAX path.

![Surface pressure miss of the accepted profile for 26 solver setups](../assets/validation/pressure_root_sweep.svg#only-light){ width="80%" }
![Surface pressure miss of the accepted profile for 26 solver setups](../assets/validation/pressure_root_sweep_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the number of the setup, for 26 setups built from `input/default.toml` and numbered from 0 to 25. The y-axis is $\lvert P(R) - P_t \rvert / P_t$ of the accepted profile, on a log scale. Circles are reached profiles, squares are padded profiles (at 1, since $P(R) = 0$), and a cross above the cap marks a setup without an accepted profile. The dashed line is the cap of $P_t$. Setups 4, 9, 10, 13, 14 and 25 use the Newton outer solver at `relative_tolerance` $10^{-9}$, and the test asserts only these 6; the other 20 setups are shown for comparison; [padded accepts](../Explanations/process_flow.md#padded-accepts) lists every setup. A pass means that every Newton setup is a circle at or below the dashed line. Only setups 9, 10, 13, 14 and 25 may be squares, and only under the rule above.

**Known limit.** Whether a profile ends padded depends on round-off, as the starts above show. The miss of a reached profile can also change, so another platform can give other values. In the plotted run the 4 reached Newton setups 4, 13, 14 and 25 miss by 0.046, 0.18, 0.050 and 0.25 $P_t$. The test skips when the PALEOS tables are not under `data/`.

??? info "Test details"
    - Reference: the boundary condition $P(R) = P_t$.
    - Tolerance: the Analytic and JAX setups accept on the Newton tolerance and the numpy PALEOS setups on the full adiabat; $\lvert P(R) - P_t \rvert \le P_t$ for every reached profile; no padded node except on the JAX path and in the numpy PALEOS setups; the padded flag matches the root of the last pressure solve.
    - Test id: `tests/test_pressure_root_solves.py::test_every_newton_setup_meets_the_surface_boundary_condition`
    - Command: `pytest -o "addopts=" -m accept_solves_newton_setups -k every_newton_setup tests/test_pressure_root_solves.py`; the figure: `python -m tools.validation.pressure_root_sweep run <dir> -j 2`, then `python -m tools.validation.pressure_root_sweep plot <dir> --assets <dir>`.

## Newton and Picard on the PROTEUS initial structure

**Why this check exists.** PROTEUS starts a coupled run from a Zalmoxis structure solved with the Newton outer solver. The run starts from a wrong state if the Newton result keeps its linear first guess of the temperature, or if a solve returns a mantle temperature that is not the adiabat of its own pressure. Each of the 2 tests below catches one of these faults.

**What it tests.** Each setup has 1 Earth mass, the PALEOS iron core and PALEOS MgSiO$_3$ mantle, and the temperature mode `adiabatic_from_cmb`: the mantle follows the adiabat upward from a fixed CMB temperature $T_\mathrm{cmb}$. The setup `all_options` has a core mass fraction of 0.325 and anchors of $T_\mathrm{surf}$ = 3936 K and $T_\mathrm{cmb}$ = 6121 K; the setup `dummy` has 0.55, 4000 K and 6000 K. Both run on the JAX path with 150 layers, `relative_tolerance` $10^{-9}$ and `absolute_tolerance` $10^{-10}$, the settings in `tools/validation/proteus_ic_solvers.py`.

- The first test solves each setup twice, with the Picard and with the Newton outer solver. The mantle temperatures at 39 mass points between the CMB and the surface may differ by at most 5 K, and the CMB pressures by at most 0.1 GPa.
- The second test solves `dummy` and `all_options` with Newton, and `all_options` with Picard at the Newton mass tolerance of $10^{-4}$. It computes the adiabat again on the returned pressure. At each mantle node, the top node included, the returned temperature may differ from it by at most 1 K. One node has its own bound (see the known limit below): 6 K from $T_\mathrm{cmb}$ at the first mantle node, the CMB anchor.

In the solves of the second test, the mantle nodes other than the CMB anchor node are at most 0.26 K off the adiabat; the top node is 0.21 K (`dummy`) and 0.08 K (`all_options`) off. Below $10^5$ Pa the adiabat holds the temperature, so a top node near 1 atm changes continuously with the surface pressure ([Temperature across 1 bar](thermal_state.md#temperature-across-1-bar)).

**How we know the right answer.** The 2 outer solvers solve the same equations for the same adiabat, so 2 correct results differ only by the error that the mass tolerance of each solve allows. This is a cross-solver check with no reference solution. The second test is an internal consistency check, not an independent reference: the solve takes its temperature from the adiabat of its pressure, so the adiabat that the same `compute_adiabatic_temperature` gives on the returned structure is the temperature it must return.

**Why it matters.** PROTEUS uses the Newton outer solver by default, and the temperature and pressure of this structure set the initial state of the interior evolution.

![Mantle temperature difference between the Newton and Picard results of 2 PROTEUS setups](../assets/validation/proteus_ic_solvers.svg#only-light){ width="70%" }
![Mantle temperature difference between the Newton and Picard results of 2 PROTEUS setups](../assets/validation/proteus_ic_solvers_dark.svg#only-dark){ width="70%" }

**How to read the plot.** The x-axis is the mantle mass coordinate, 0 at the CMB and 1 at the surface. The y-axis is the absolute difference between the Newton and the Picard mantle temperature in K, on a log scale; differences below $10^{-3}$ K are drawn at $10^{-3}$ K. Each setup has a solid line, and the broken line is the 5 K limit. A pass means that both lines stay below the limit. The CMB pressures differ by 0.060 GPa in `all_options` and by 0.068 GPa in `dummy`, inside the limit of 0.1 GPa.

**Known limit.** These bounds have 3 limits:

- The CMB anchor node does not follow the adiabat of its own pressure. It reads one T(P) table that holds $T_\mathrm{surf}$ on the core nodes and $T_\mathrm{cmb}$ at the CMB, so its temperature takes a small part of that jump. It lies 5.41 K below $T_\mathrm{cmb}$ in `all_options` with Newton, 5.38 K with Picard at $10^{-4}$, and 0.25 K in `dummy` with Newton.
- The Picard solve of the first test runs at the mass tolerance of the PROTEUS setups, $3 \times 10^{-3}$, and carries the error of that tolerance. There the `all_options` mantle lies on average 1.7 K below the adiabat of its own pressure. At $10^{-4}$ the difference is 0.1 K or less.
- These bounds hold for these 2 setups. The start-up solves of the PROTEUS grid go beyond them in the places that [Secant tail and seeded re-solve on the PROTEUS grid start-up](#secant-tail-and-seeded-re-solve-on-the-proteus-grid-start-up) lists.

??? info "Test details"
    - Reference: the Picard result of the same setup (cross-solver check); the adiabat that the same `compute_adiabatic_temperature` gives on the returned pressure (internal consistency check).
    - Tolerance: 5 K at 39 mantle mass points and 0.1 GPa at the CMB; 1 K at the mantle nodes, the top node included, and 6 K from $T_\mathrm{cmb}$ at the CMB anchor node.
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
A repeat of the first solve at a planet mass $2 \times 10^{-12}$ higher runs in the nightly job of the stored reference; on the development machine its top node lies 0.20 K from REF (see the known limit below).
Each solve is compared with REF, the same solve at a mass tolerance of $10^{-6}$.
The lag of a node is the difference between the returned temperature at that node and the adiabat computed again on the returned pressure.
Relative differences are $|x - x_\mathrm{REF}| / |x_\mathrm{REF}|$.

- The mantle temperature at 39 mass points, from the lowest to the second-highest of the 12 mantle nodes, differs from REF by at most 1 K.
- The temperature of the top mantle node differs from REF by at most 1 K in the re-solves and 3 K in the first solves (see the known limit below). The test of a first solve at a planet mass $2 \times 10^{-12}$ higher and the test of the stored REF use a bound of 2 K for the first solves.
- The outer radius, the central pressure and the core radius differ from REF by at most $2 \times 10^{-4}$ (relative).
- The CMB pressure differs from REF by at most $4 \times 10^{-3}$ (relative). This is $2\,\mathrm{tol} / (1 - f_\mathrm{core})$, with the mass tolerance $\mathrm{tol} = 10^{-4}$ and the core mass fraction $f_\mathrm{core} = 0.95$: the core mass is fixed, so the mass error sits in the mantle and moves the CMB pressure by that error divided by the mantle mass fraction $1 - f_\mathrm{core}$. The core radius and the CMB pressure are read at the core mass by interpolation in mass.
- The lag is at most 6 K at the CMB node and 1 K at the other mantle nodes in every solve, and at the top node at most 3 K in the first solves and 1 K in the 2 re-solves.
- The seeded and the unseeded re-solve differ by at most 2.5 K at the 39 mass points and at the top node, and $5 \times 10^{-4}$ in radius.
- The mass-weighted mean temperature of the core nodes differs from REF by at most 10 K in every solve.
  The last core node, just below the CMB node, differs by at most 700 K: it reads the edge of the mantle temperature table (see the known limit below).
  Every other core node differs by at most 1 K. Each core node is compared with REF interpolated in mass over the core nodes of REF, not across the temperature step at the CMB node of REF.
  The REF core is at 4000 K at every core node, and its last core node has a weight of 0.78 % in the mass-weighted mean; so for a solve with the core node masses of REF, these 2 node bounds keep the mean within 6.5 K in the first solve and in the re-solve, and the 10 K bound of the mean follows from them. The measured mean lies 4.5 K from REF in the first solve and 4.1 K in the unseeded re-solve.
- The accept label of every solve is `strict`, and the accepted pass is on a table of blend 1.
- The first solve takes at most 4 passes on a full-adiabat table, after pass 1 and the 4 blend passes: at most 9 in total. The seeded re-solve takes at most 5 passes in total, and at most half the structure integrations of the unseeded re-solve.
- Each solve ends with 1 pass of waiting: the strict test accepts a pass only when its temperature table comes from a pass that met the mass tolerance ([How the density solve reads the core temperature](../Explanations/adiabat_integration.md#how-the-density-solve-reads-the-core-temperature)). On the numpy path the first solve, the seeded re-solve and the unseeded re-solve take 9, 5 and 9 passes, and 8, 4 and 8 without that wait.
- On the JAX path (`use_jax = true`, the PROTEUS default) a second test runs the same 3 solves against REF solved on that path, with the same REF bounds for the first solve and the seeded re-solve and the same pass bounds for the seeded re-solve; it does not bound the passes of the first solve. The unseeded re-solve is compared only with the seeded one (2.5 K, $5 \times 10^{-4}$ in radius): accepted at the mass tolerance $10^{-4}$, it lies 1.006 K from that REF in the mantle on macOS (0.261 K on the numpy path; both from this repository's test helpers), and 0.757 K from the seeded re-solve. At the reference tolerance the JAX REF lies within 0.07 K in the mantle and $1.4 \times 10^{-5}$ in radius of the numpy REF.

**How we know the right answer.** REF runs until its mass error is 100 times smaller than the mass tolerance of the ramp, $10^{-4}$.
So REF stands for the converged structure; this is a convergence check, not an external reference.
REF is solved once on each path and stored in `tests/data/grid_start_up_reference.json`, with the md5 of both EOS tables and a hash of the Zalmoxis source that solved it (every Zalmoxis module the solve loads, without `_version.py`, which the install writes with the commit in it).
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
A pass means that the first-solve curve ends below the broken line after at most 4 passes from pass 6, and the seeded curve ends below it within 5 passes.

**Known limit.** The tests bound the lag for the 2 setups of [Newton and Picard on the PROTEUS initial structure](#newton-and-picard-on-the-proteus-initial-structure) and for the setup above, 1 of the 4 PROTEUS grid cases.
Replays of the saved start-up inputs of all 4 grid cases (core mass fraction 0.35 and 0.95; 1000 and 2000 ppmw H, parts per million by weight of hydrogen), taken from a PROTEUS grid run and outside the tests of this repository, give these values; the rows of the first solve at 1 atm and of the re-solves at core mass fraction 0.95 come from the grid start-up test of this repository.
The first 5 rows exceed a bound; the other 7 lie inside one.

| Case | Quantity | Value | Bound |
|---|---|---|---|
| Core mass fraction 0.35, first solve | lag at the CMB node | 92.4 K | 6 K |
| Core mass fraction 0.35, first solve | lag at the other mantle nodes | 2.09 K | 1 K |
| Core mass fraction 0.35, first solve | mantle temperature from REF | 2.17 K | 1 K |
| Core mass fraction 0.35, re-solves | lag at the mantle nodes, top node included | up to 1.26 K | 1 K |
| Core mass fraction 0.35, re-solves | mantle temperature from REF, without the seed / with it | up to 1.29 K / 1.25 K | 1 K |
| Core mass fraction 0.95, first solve at 1 atm | lag at the top node | 0.07 K | 3 K, inside |
| Core mass fraction 0.95, first solve at 1 atm | top node from REF | 0.20 K | 3 K, inside |
| Core mass fraction 0.95, first solve | last core node from REF | 586 K | 700 K, inside |
| Core mass fraction 0.95, re-solves | last core node from REF, unseeded / seeded | 526 K / 0 K | 700 K, inside |
| Core mass fraction 0.95, first solve | mass-weighted mean core temperature from REF | 4.5 K | 10 K, inside |
| All 4 cases, every solve | CMB pressure from REF | up to $7.1 \times 10^{-4}$ on the numpy path, $1.7 \times 10^{-3}$ in the JAX first solve at core mass fraction 0.95 | $4 \times 10^{-3}$ at core mass fraction 0.95, inside |
| Core mass fraction 0.35, 2000 ppmw H, seeded re-solve | lag at the CMB node | up to 5.4 K | 6 K, inside |

- At core mass fraction 0.35 the cube-root rule converges in 2 full-adiabat passes, so the secant tail does not step and the first solve is the same as without it.
- At 1 atm the first solve builds its temperature table on the pressure of the pass before; with the adiabat over the top cell of that table, its top node lags its own adiabat by 0.07 K on the numpy path and 0.003 K on the JAX path, against at most 0.1 K at the other mantle nodes. It lies 0.20 K (numpy), 0.88 K (JAX) and 0.20 K (planet mass $2 \times 10^{-12}$ higher) from REF on macOS. The tests bound the top node of the first solves at 3 K from REF and a lag of 3 K; the repeat at a planet mass $2 \times 10^{-12}$ higher and the test of the stored REF use a bound of 2 K from REF. The re-solves, at 51.3 MPa, are bound at 1 K from REF and a lag of 1 K. These bounds are wider than the measured values, so they do not test the temperature lookup over the top cell. That lookup has its own tests: [Temperature lookup over the top cell](thermal_state.md#temperature-lookup-over-the-top-cell).
- The core nodes and the CMB node read one T(P) table with the mantle, and the solve builds that table on the pressure of the pass before.
  The last core node (index 137, 0-based, of 150 nodes at core mass fraction 0.95) therefore reads the edge of that table.
  With the secant tail, which moves the CMB pressure more from pass to pass, this node lies 586 K above REF in the first solve, 526 K in the unseeded re-solve and 0 K in the seeded re-solve.
  REF has the core temperature at this node.
  The node is in the cell below the CMB, which holds 0.7 % of the planet mass.
  So the mass-weighted mean temperature of the core nodes lies only 4.5 K above REF, and the core radius stays within $3 \times 10^{-5}$ of REF.
  The tests bound the mean at 10 K, this node at 700 K and the other core nodes at 1 K; the other core nodes lie 0 K from REF in the 3 solves of the test.
  One more pass on the accepted structure gives this node the core temperature again (4000 K, measured); the solve does not do it.
- In the `adiabatic_from_cmb` mode every core node has the surface temperature (4000 K here) and the energetics module computes the core temperature. A core at the CMB temperature (6000 K) gives a larger core radius and a lower mean core density inside it; [Zalmoxis issue 131](https://github.com/FormingWorlds/Zalmoxis/issues/131) gives their size in the first solve of this case.

??? info "Test details"
    - Reference: REF, the same solve at a mass tolerance of $10^{-6}$ (convergence check), stored in `tests/data/grid_start_up_reference.json`.
    - Tolerance: 1 K at 39 mantle mass points up to the second-highest mantle node; top mantle node 1 K from REF in the re-solves and 3 K in the first solves (2 K in the test at a planet mass $2 \times 10^{-12}$ higher); $2 \times 10^{-4}$ in outer radius, central pressure and core radius; $4 \times 10^{-3}$ in CMB pressure; lags of 6 K (CMB node) and 1 K (other mantle nodes) in every solve, and at the top node 3 K in the first solves and 1 K in the re-solves; seeded against unseeded 2.5 K and $5 \times 10^{-4}$; mass-weighted mean core temperature 10 K, last core node 700 K and the other core nodes 1 K from REF; at most 4 full-adiabat passes in the first solve, at most 5 passes and half the integrations in the seeded re-solve.
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
    - Plotted run (can change with round-off): in the first solve the error lies from $1.3 \times 10^{-3}$ to $3.4 \times 10^{-3}$ in passes 2 to 5 and is $2.2 \times 10^{-3}$ at pass 6; the cube-root rule reaches $1.6 \times 10^{-3}$ at pass 7 and the secant step $5.2 \times 10^{-6}$ at pass 8, and the strict test accepts pass 9 ($3.5 \times 10^{-6}$), the first on a table of a pass that met the tolerance. A replay of this first solve with the cube-root rule alone, without the secant tail, takes 11 full-adiabat passes, from $2.2 \times 10^{-3}$ at pass 6 to $9.4 \times 10^{-5}$ at pass 16, and the strict test accepts pass 17. The re-solve takes 5 passes from the seed against 9 without it (1 at the Newton root, 4 blend passes, 4 full-adiabat passes).
    - Command: `pytest -o "addopts=" -m accept_solves_long tests/test_accepted_on.py -k grid_start_up --validation-plots`.
    - Stored REF: `tests/test_accepted_on.py::test_the_stored_grid_start_up_reference_is_the_converged_solve_of_the_source` solves it again (markers `slow`, `accept_solves_reference` and `physics_invariant`; nightly job `accept_solves_reference`, 7800 s per test, 150 min per job). Tolerance: 0.1 K in the mantle up to the second-highest node, top node 1 K in the re-solves and 2 K in the first solves, $10^{-5}$ in radius, central pressure and core radius, $2\,\mathrm{tol}_\mathrm{REF} / (1 - f_\mathrm{core}) = 4 \times 10^{-5}$ in CMB pressure, and the stored source hash. Command to write it again: `python -m tools.validation.grid_start_up_reference`; the hash leaves out the files in `GENERATED` of that script (`_version.py`).

## Inner-core boundary of a two-layer core

**Why this check exists.** With `core_liquidus`, the inner-core boundary (ICB) of a [two-layer core](../Explanations/mixing.md#two-layer-core) lies where the liquid of the outer core reaches the iron melting curve. On an adiabat the temperature profile depends on the ICB, since each layer takes the adiabatic gradient of its own side. A rule that reads the combined profile, liquid above the ICB and solid below it, accepts an ICB with the liquid tens of K above the melting curve, and its ICB then depends on the start of the solve.

**What it tests.** The tests solve the default 1 $M_\oplus$ planet with a two-layer core and the iron melting curve of Anzellini et al. (2013)[^cite-anzellini2013] as `core_liquidus`. The 2 iron side tables have 40 points per decade over $10^9$ to $3 \times 10^{12}$ Pa and $10^3$ to $2 \times 10^4$ K, not the default grid of 150 points per decade. On an adiabat, the rule continues the outer-core adiabat below the ICB with the liquid side, the liquid branch $T_\mathrm{l}(P)$. The superheat $S = T_\mathrm{l}(P_\mathrm{ICB}) - T_\mathrm{liq}(P_\mathrm{ICB})$ is the distance of that liquid from the melting curve $T_\mathrm{liq}$ at the ICB pressure $P_\mathrm{ICB}$. The outer loop stops only when the relative mass error is below `tolerance_outer` and 2 more tests hold:

1. $|S| \le 0.5$ K at the ICB of the layers; with no inner core the liquid branch must not lie below the melting curve at the centre, and with a core solid to the core-mantle boundary (CMB) it must not lie above it there;
2. the adiabat of the pass, which takes the inner-core mass of the pass before, has the inner-core mass of the layers to `tolerance_outer` of the core mass.

On a given profile (isothermal, linear, prescribed) the profile does not depend on the phase, and test 1 is the distance of the ICB from the first root of $T(P) - T_\mathrm{liq}(P)$ of the profile, at most `tolerance_outer` of the core mass.

On an adiabat the result reports $S$ as `icb_superheat` (at the centre with no inner core, at the core mass with a full one) and the inner-core mass of the adiabat of the returned pass as `icb_adiabat` ([core arrays](../How-to/usage.md#core-arrays)). The ICB update takes the mass of the liquidus root with $m^{2/3}$ linear in $P$, the inverse of the pressure that test 1 and the core arrays take at that mass, so the update, test 1 and the core arrays take one pressure for one ICB mass, except where the core arrays hold `P_icb` at their CMB node ([the pressure at the inner-core boundary](../Explanations/mixing.md#the-pressure-at-the-inner-core-boundary)). Test 1 checks that the ICB of the layers lies on the liquid branch of the pass; the update takes that ICB from the same branch on the structure of the pass before, so $S$ shrinks as the structure stops moving, and test 2 checks that the adiabat and the layers share the ICB. Near 2500 K, $S$ changes by about 102 K per unit of core mass, so 0.5 K corresponds to about 0.005 of the core mass (mass-rule solves at 0.3 and 0.5 of the core mass, default side-table grid).

The cases follow in 4 groups below.

**How we know the right answer.** The zero-superheat condition is the ICB of a growing inner core in core thermal models. Labrosse (2015)[^cite-labrosse2015], Sect. 2, takes the ICB temperature as the liquidus temperature of the outer core, and Nimmo (2015)[^cite-nimmo2015], Fig. 2, defines the inner core by the intersection of the adiabat and the melting curve. For the pure-iron adiabat the test integrates the liquid branch with its own integrator (DOP853, relative tolerance $10^{-13}$), not the adiabat of the solve. The closed-form root follows from the 2 power laws alone. For the isothermal core the test finds the root of the Anzellini curve at 5000 K itself, by Brent's method on the curve. For the Fe-S adiabat the test records the inner-core mass that the adiabat of each pass takes and compares the last one with `icb_adiabat` and with the inner-core mass of the returned layers. The bulk S is summed over the core arrays and checked against $(1 - M_\mathrm{ic}/M_\mathrm{core})\,w_\mathrm{oc}$; the inner-core fractions are regression values of this code, not an outside reference, pinned to 0.01, 3 times their spread over 5 perturbed starts at 5000 K. For the scan, the 2 walks check each other.

**Why it matters.** The ICB sets the mass of the solid inner core and the density step between the 2 sides. An ICB with superheated liquid above it is not a state of a freezing core, and an ICB that depends on the start gives different planets for the same input.

### Zero superheat at the ICB on an adiabat

- **Zero superheat on an adiabat.** The pure-iron core at 2500 K at the surface holds a partial inner core, with 0.368 of the core mass. The test integrates the liquid side $\nabla_\mathrm{ad}$ itself, from the second outer-core node to the centre: the returned T at the first outer-core node comes from the temperature table of the pass before and can hold part of the solid side: at most 1.03 K in 11 starts at the default `tolerance_outer` (1 $M_\oplus$ at 2500 and 2510 K, 3 $M_\oplus$ at 3000 K). It finds $|S| \le 0.5$ K at the returned ICB, and the root of its own liquid branch within 0.005 of the core mass of the returned inner core.
- **A closed-form root.** A unit test writes a liquid branch with a constant $\nabla_\mathrm{ad} = n$, $T_\mathrm{l} = T_0 (P/P_0)^n$, and a power-law melting curve, Eq. 3 of Anzellini et al. (2013)[^cite-anzellini2013] with its scale pressure set to the triple-point pressure $P_\mathrm{TP}$: $T_\mathrm{liq} = T_\mathrm{TP} (P/P_\mathrm{TP})^q$ with $q = 1/1.72$. The 2 curves meet at $P_B = (T_\mathrm{TP} P_0^n / (T_0 P_\mathrm{TP}^q))^{1/(n - q)}$. With a node at $P_B$ the superheat is 0 to $10^{-8}$ of $T_\mathrm{liq}(P_B)$. Off the root, a property test over $n$ from 0.1 to 0.5 and any ICB in the core finds the closed-form $|S|$ within the error of linear interpolation in $\ln P$, $(\Delta \ln P)^2 n^2 T / 8$, at most 0.22 K here. A property test takes the first liquidus root of a power-law liquid branch by the rule itself, over $n$ from 0.1 to 0.5 and roots from 0.02 to 0.98 of the core mass, and finds the superheat at that inner-core mass 0 to $10^{-9}$ of $T_0$.
- **One adiabat above the ICB.** Above the ICB of a pass, its liquid branch and its temperature table take the same adiabat, so they agree to round-off. The solve checks this at each pass and raises `RuntimeError` above a relative difference of $10^{-12}$. A test shifts the liquid branch by 1 K and expects the error.
- **Adiabat ICB.** With the 13.85 wt% S outer core and 2480 K at the surface, the core holds a partial inner core. The solve stops on its strict test (`accepted_on = 'strict'`), and the adiabat of the returned pass takes the inner-core mass of that pass to $10^{-3}$ of the core mass.

### The adiabat across the ICB

- **The adiabat across the ICB.** 5 starts of the pure-iron core at 2500 K, with the planet mass and the surface temperature changed by a relative $k \times 10^{-12}$ ($k$ = 0 to 4). An independent DOP853 integration of $\nabla_\mathrm{ad}$ (relative tolerance $10^{-12}$) runs on the liquid side from the first outer-core node to the ICB pressure of the core arrays, then on the solid side to the first inner-core node, and from that node on the solid side to the centre. The returned temperature must lie within 3 K of it at the first inner-core node and within 1.5 K at the centre; a segment that took one side over its whole length misses by up to 21 K. The largest misses of the 5 starts on macOS arm64 are 0.64 K and 0.49 K. A unit test pins the pressure of the cut: on a uniform sphere with a constant $\nabla_\mathrm{ad}$ per layer, with the ICB between 2 core nodes or in the node segment of the CMB, the adiabat meets the piecewise power law to round-off, with its break at the one ICB pressure that the superheat and the ICB rule give; at this position of the CMB the core arrays give the same pressure. The solve test catches a segment that takes one side over its whole length; its bounds do not resolve the pressure rule of the cut, which the unit test checks. This case has no plot: its 5 starts give 2 misses each, which the text states as numbers.

### Freezing band and start independence

- **The freezing band.** 5 solves of the pure-iron core from 2460 to 2530 K at the surface: the inner-core mass fraction does not rise with the surface temperature, the core is solid to the CMB at 2460 K and liquid at 2530 K, and each solve passes test 1.
- **Start independence.** Walks down from a liquid core and up from a solid core, each step started from the state of the step before, give the same ICB at every step. `tools/core_icb_scan.py` made the scan, which is stored, with the source commit in its header, in `tests/validation/data/core_icb_scan.csv`; the plot test reads that file and checks the 2 walks against each other, and does not solve the scan again. The walks agree to $10^{-3}$ of the core mass at every step except 2475 K (see the known limit below).

![Inner-core mass fraction of the pure-iron core against surface temperature, 2460 to 2530 K](../assets/validation/core_two_phase_pure_iron_freezing_band.svg#only-light){ width="80%" }
![Inner-core mass fraction of the pure-iron core against surface temperature, 2460 to 2530 K](../assets/validation/core_two_phase_pure_iron_freezing_band_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the surface temperature of the adiabat, and the y-axis the inner-core mass fraction $M_\mathrm{ic}/M_\mathrm{core}$ of the returned solve. The 5 marks are the solves of the freezing band case; the line only joins them. A pass means that the marks do not rise to the right, start at 1 (solid to the CMB) and end at 0 (liquid).

![Inner-core mass fraction against surface temperature for the walks down and up of the start-dependence scan](../assets/validation/core_two_phase_icb_start_dependence.svg#only-light){ width="80%" }
![Inner-core mass fraction against surface temperature for the walks down and up of the start-dependence scan](../assets/validation/core_two_phase_icb_start_dependence_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the surface temperature of the adiabat, 2300 to 2700 K in 25 K steps, and the y-axis the inner-core mass fraction of the returned solve of the Fe-S core. The solid line walks down from a liquid core and the dashed line up from a solid core; each step starts from the state of the step before. A pass means that both walks converge at every step and agree to $10^{-3}$, and to 0.0164 at 2475 K, as the test asserts. The only step inside the band is 2475 K (row $k = 0$ of the table below); the core is solid to the CMB up to 2450 K and liquid from 2500 K.

### The ICB on a given profile and when an accept skips the ICB tests

- **Isothermal core.** At 5000 K the melting curve gives the root at 206.4 GPa: the ICB pressure at most a relative $5 \times 10^{-3}$ from the root, and the inner-core mass at most `tolerance_outer` of the core mass from the mass at the root.
- **An accept with the ICB off.** Some accept paths skip the 2 tests: an iteration cap, a clock stop, an oscillation stop. Such a path returns `converged = False` and `icb_converged = False` when the result fails test 1 or 2, and `main` then retries instead of accepting it; such a retry can take minutes. An isothermal 5000 K core with `max_iterations_outer = 2` stops at its iteration cap while the ICB still moves, and the test checks the 2 flags and the retry. A second test sets the distance to the ICB of the adiabat to the core mass while the root is met: the cap accept is again not converged. A third test sets $S$ to $-10$ K on every pass while the adiabat ICB is met: the cap accept of the 2480 K Fe-S adiabat is again not converged, and the warning gives the miss in K. A fourth test stops the 2480 K Fe-S adiabat at 10 passes: the cap accept restores the best pass, and the final ICB test uses the ICB and the liquid branch of that pass.
- **Bounds of the rule.** A centre above the liquidus (isothermal 8000 K) gives no inner core ($M_\mathrm{ic} = 0$), and a core wholly below it (3000 K) is all inner core. Nodes above the ICB that lie below the liquidus, as in a profile that crosses the liquidus more than once (a linear profile from 7000 K at the centre), stay in the outer core. Their mass, as a fraction of the core mass, is `solid_above_icb_mass_fraction`, and `solid_above_icb` is set.
- **The rule between grid nodes.** Unit tests apply the rule to profiles that the test writes by hand. A node segment is the stretch between 2 neighbouring grid nodes. When the profile reaches the liquidus between the last core node and the CMB, the ICB lies on that segment: the rule adds a point at the core mass, with $\ln P$ linear in the enclosed mass and $T$ linear in $\ln P$ on that segment. A core below the liquidus up to the CMB is all inner core, at the CMB pressure. On a uniform sphere, an inner core that ends in the first or the second node segment gets $R_\mathrm{ICB}$ from $r^3$ linear in the enclosed mass and $P_\mathrm{ICB}$ from $P$ linear in the enclosed mass to the power 2/3; both are exact for a uniform density.
- **Bulk composition.** The light-element fractions of each layer are inputs, so the light-element mass of the core changes when the ICB moves. For the 13.85 wt% S outer core on an isothermal core, the inner core holds 0.480 of the core mass at 5000 K and 0.159 at 5500 K, so the bulk S of the core is 7.20 and 11.65 wt%. A caller that keeps the bulk composition, as an evolution model with element partitioning does, sets the layer fractions for each solve.

### Known limits and test details

**Known limits.**

- **Round-off at 2475 K.** At 2475 K the start-dependence scan scatters with round-off. 5 pairs of walks, with the surface temperature of every step changed by a relative $k \times 10^{-12}$ ($k$ = 0 to 4), give these inner-core mass fractions:

    | $k$ | Walk down | Walk up |
    |---|---|---|
    | 0 | 0.9214 | 0.9269 |
    | 1 | 0.9268 | 0.9265 |
    | 2 | 0.9236 | 0.9218 |
    | 3 | 0.9271 | 0.9237 |
    | 4 | 0.9208 | 0.9237 |

    All 10 converge with both ICB tests met. They lie from 0.9208 to 0.9271, and a pair differs by up to 0.0055. The test bounds this step at 0.0164, 3 times that difference. The stored scan is the pair with $k = 0$.
- **The ICB at the default tolerance.** At the default `tolerance_outer` ($10^{-3}$), 5 starts of the pure-iron core at 2500 K, with the planet mass and the surface temperature changed by a relative $k \times 10^{-12}$ ($k = 0$ to 4), give inner-core mass fractions from 0.3590 to 0.3624 on the default side-table grid of 150 points per decade, all with $|S| \le 0.11$ K; the test, on 40 points per decade, gives 0.368. 1 K of surface temperature moves the ICB by about 0.02 of the core mass near 2500 K (0.361 at 2500 K, 0.152 at 2510 K, start 0, default grid).
- **A tight outer tolerance on an adiabat.** The ICB of the adiabat: the second ICB test, at most `tolerance_outer` of the core mass between the ICB of the adiabat and that of the layers, needs a pass where the ICB no longer moves.
    - At the default tolerance, every start tested converges on the strict test in its first solve.
    - At a `tolerance_outer` of $10^{-4}$, every start tested returns `converged = False`: the 5 starts of the pure-iron core at 2500 K and the 5 starts of the 13.85 wt% S outer core at 2480 K, where the retry of `main` stops on the wall clock or on oscillation.
    - From the tenth pass on, the distance between the 2 ICBs does not fall. In the retry of the Fe-S starts its median lies 35 to 55 times above the retry bound of $10^{-5}$ of the core mass, so a start converges only when a single pass falls below the bound.
    - Without the wall clock, the Fe-S start $k = 0$ at 2480 K runs 47 passes in the retry, none below the bound, and stops on oscillation.
    - A floor set by the radial resolution of the ICB is one possible cause; no test has checked it.
    - Measured runs of the second ICB test (macOS arm64, next to other jobs, so where the wall clock stops depends on the load): at the default tolerance, 10 starts (1 $M_\oplus$ at 2500 K, 3 $M_\oplus$ at 3000 K, $k$ = 0 to 4) converge in 158 to 955 s; at $10^{-4}$ the 5 starts at 2500 K stop after 1021 to 1099 s; for the Fe-S core at 2480 K (core mass fraction 0.325, 150 nodes) no start converges: the wall clock stops the retry of 4 after 900 to 901 s and 1 stops on oscillation after 865 s, all with inner-core mass fractions of 0.7166 to 0.7186 and $|S| \le 0.009$ K; the median distance between the 2 ICBs from the tenth pass on is $3.1 \times 10^{-3}$ to $6.0 \times 10^{-3}$ of the core mass in the first solve and $3.5 \times 10^{-4}$ to $5.5 \times 10^{-4}$ in the retry (bound $10^{-5}$), with 0 to 1 passes per start below the bound; at 2490 K the retry also stops on the wall clock.
- **A steep Fe-S case.** The inner core of this Fe-S case depends strongly on the temperature. At its ICB the liquid adiabat and the melting curve are near tangent: their slopes are $1.056$ and $1.115 \times 10^{-8}$ K/Pa, a difference of $-5.96 \times 10^{-10}$ K/Pa. 1 K more at the CMB makes the inner core smaller by about 0.010 of the core mass, and from 2480 to 2490 K at the surface the inner-core mass fraction falls from 0.717 to 0.228, 0.049 per K; at 2470 K the core is solid to the CMB. This is a property of the case, not of the solver: an independent DOP853 integration of the adiabat, from 1 bar through the mantle to the CMB pressure and on through the liquid core, with its root on the Anzellini curve and no node grid, gives 0.7186 on the returned structure at 2480 K (the solve: 0.7173) and 0.2275 at 2490 K (the solve: 0.2275). Both solves, at a `tolerance_outer` of $10^{-4}$, end on the wall clock with `converged = False`, at most 0.0013 of the core mass from the reference. An inner-core fraction of this case is only as accurate as the surface temperature that sets it.
- **A tight outer tolerance on a given profile.** With `core_liquidus`, the outer loop stops only when the total mass and the ICB mass both meet `tolerance_outer`, and near convergence one pass moves the ICB by about 0.4 to 2.1 times the change of the relative mass. Well below the default $10^{-3}$, both tests reach the jitter of the Picard iteration and can alternate, and the solve then returns `converged = False`. For the isothermal 5000 K core of the default 1 $M_\oplus$ planet with `max_iterations_outer = 2`, the first solve stops at its iteration cap and the tightened retry of `main` ($10^{-4}$, 4 passes, since the retry doubles the cap) ends with the ICB $8.8 \times 10^{-5}$ of the core mass off the root on macOS arm64, a 12 % margin, and $1.38 \times 10^{-4}$ off on an AMD EPYC 9V45. With `max_iterations_outer = 3` the retry runs 6 passes and does not converge on macOS arm64 either. On a given profile, a result with `converged = True` has its ICB within `tolerance_outer` of the root and of the ICB of its adiabat (`icb_converged`).
- **The entropy of the solid side table.** The solid iron side table of PALEOS breaks the Maxwell relation of its entropy from about 205 to 400 GPa. At the 5230 K node of the default side grid, $(\partial s/\partial \ln P)_T$ agrees with $-\alpha P/\rho$ of the same table to 4 digits from 105 to 200 GPa and to $2.5 \times 10^{-3}$ from 417 to 600 GPa, but has the wrong sign at 30 of 117 nodes from 229 to 358 GPa ($+156.6$ against $-151.7$ J/(kg K) at 293 GPa). In the same window the $\nabla_\mathrm{ad}$ column lies 1.5 to 6.3 % below $\alpha P/(\rho c_P)$ of the table; outside it they agree to $10^{-8}$. The adiabat of the solve reads the $\nabla_\mathrm{ad}$ column, so its inner core follows no isentrope of the entropy column there: from an ICB at 282 GPa and 5779 K to 407 GPa, the $\nabla_\mathrm{ad}$ column gives 6474 K at the centre, $\alpha P/(\rho c_P)$ 6503 K and an isentrope of the $s$ column 5533 K. A code that reads the entropy of the solid side, as for P-S tables, gets that isentrope.

??? info "Test details"
    - Reference: the zero-superheat condition of Labrosse (2015) and Nimmo (2015) with an independent DOP853 integration of the liquid branch; the closed-form root of 2 power laws; the root of $T - T_\mathrm{liq}$ on the returned profile and the inner-core mass of the last adiabat (self-consistency); regression pins for the bulk S; for the scan, the 2 walks (self-consistency).
    - Tolerance: $|S| \le 0.5$ K and the liquid-branch root to 0.005 of the core mass; the superheat at the rule's own root to $10^{-9}$ of $T_0$; the closed-form superheat to $10^{-8}$ at the root and within the interpolation error off it; the liquid branch to a relative $10^{-12}$ of the temperature table above the ICB; ICB pressure to a relative $5 \times 10^{-3}$ of the root and inner-core mass to `tolerance_outer` ($10^{-3}$ of the core mass) of the mass at the root; the adiabat ICB to $10^{-3}$ of the core mass; inner-core fractions to 0.01 and bulk S to $2 \times 10^{-3}$; the freezing band monotone to 0.05; the adiabat across the ICB to 3 K at the first inner-core node and 1.5 K at the centre, and the unit cut to a relative $10^{-12}$; on the hand-written profiles the ICB to a relative $10^{-12}$ and $R_\mathrm{ICB}$ and $P_\mathrm{ICB}$ to $10^{-9}$.
    - Test ids: `tests/test_core_two_phase_solve.py::test_a_pure_iron_core_has_its_icb_where_its_liquid_adiabat_meets_iron_melting`, `::test_the_inner_core_of_a_pure_iron_core_grows_as_the_surface_cools`, `::test_the_adiabat_crosses_the_icb_at_its_pressure`, `::test_a_liquid_branch_off_the_t_table_above_the_icb_stops_the_solve`, `::test_an_isothermal_core_has_its_icb_where_iron_melts`, `::test_the_adiabat_of_a_converged_liquidus_solve_takes_its_icb`, `::test_an_accept_with_an_icb_off_the_liquidus_is_not_converged`, `::test_an_accept_with_the_adiabat_icb_off_is_not_converged`, `::test_a_cap_accept_reads_the_liquid_branch_of_the_restored_pass`, `::test_an_accept_with_the_liquid_off_the_liquidus_is_not_converged`, `::test_a_core_wholly_above_or_below_the_liquidus`, `::test_solid_above_a_liquid_centre_is_reported`, `::test_the_bulk_s_of_the_core_follows_the_icb` (integration tier, nightly job `core_two_phase`, `::test_the_adiabat_crosses_the_icb_at_its_pressure` in `core_two_phase_icb_cross`; needs PALEOS and the PALEOS tables under `data/`); `tests/test_core_two_phase.py::test_the_icb_superheat_is_0_at_the_closed_form_root`, `::test_the_adiabat_cuts_its_segments_at_the_pressures_of_the_icb_readers`, `::test_the_icb_superheat_is_0_at_the_rules_own_root`, `::test_a_one_layer_adiabat_reports_no_icb_keys`, `::test_the_icb_superheat_off_the_root_meets_the_closed_form`, `::test_the_icb_superheat_of_no_or_a_full_inner_core_counts_the_wrong_side_only`, `::test_a_non_finite_liquid_branch_at_the_icb_misses_by_inf`, `::test_the_icb_misses_are_the_distances_to_the_root_and_to_the_adiabat_icb`, `::test_the_icb_can_lie_between_the_last_core_node_and_the_cmb`, `::test_a_cold_top_counts_its_segment_to_the_cmb`, `::test_r_icb_and_p_icb_of_a_small_inner_core_meet_the_uniform_sphere` (unit tier); the scan plot: `tests/test_core_two_phase.py::test_the_start_dependence_scan_of_the_liquidus_rule` (unit tier), which reads the scan stored in `tests/validation/data/core_icb_scan.csv`; the solid side table: `tests/test_core_two_phase_solve.py::test_the_solid_iron_table_meets_the_maxwell_relation_outside_200_to_440_gpa` and `::test_the_solid_iron_table_meets_the_maxwell_relation_from_200_to_440_gpa` (strict xfail, the known limit of the solid side table)
    - Command: `pytest -o "addopts=" tests/test_core_two_phase_solve.py -k "pure_iron or crosses_the_icb or liquid_branch or isothermal_core_has or adiabat_of or accept_with or wholly or solid_above or bulk_s or cap_accept or liquid_off"`; the unit tests: `pytest -o "addopts=" tests/test_core_two_phase.py -k "superheat or cuts_its_segments or icb_misses or last_core_node or segment_to_the_cmb or small_inner_core or one_layer_adiabat"`; the plots: `pytest -o "addopts=" tests/test_core_two_phase_solve.py -k grows_as --validation-plots` and `pytest -o "addopts=" tests/test_core_two_phase.py -k start_dependence --validation-plots`; the scan: `python tools/core_icb_scan.py <out> down` (or `up`)

## Core temperature from a table given by the caller

**Why this check exists.** A coupled model such as PROTEUS can give Zalmoxis the core temperature as a table, `core_temperature_arrays = (P_arr, T_arr)`: the temperature $T$ against the pressure $P$. Every core node and every evaluation of the structure equations in the core must then read the table, and every mantle node must read the mantle temperature. If one part of the solve reads the wrong source (the numpy or the JAX integration, the node temperatures, the ICB rule or the core arrays), the core density or the ICB is wrong, and these tests fail.

**What it tests.** Most cases solve a planet of 1 $M_\oplus$ with a PALEOS iron core and a PALEOS-2phase MgSiO$_3$ mantle on 150 radial nodes. Its mantle temperature is a profile in radius that falls from 5500 K at 3460 km to 3830 K at 7060 km. The first table point, at 100 GPa, lies 150 K above the mantle temperature at the CMB, and the table rises with pressure, so a core node that reads the mantle temperature is off by more than 150 K. The ICB case solves the default 1 $M_\oplus$ planet of `input/default.toml` with a two-layer pure-iron core and a mantle at 2500 K. The cases:

- **Node rule.** On the JAX and on the numpy path, every returned core node has $T$ = table($P$) and every mantle node has the mantle temperature, both to a relative $10^{-9}$. A core node is a node whose enclosed mass is below the core mass, the rule that also picks the core material. The core mass also closes on the returned density: the integral of $4\pi r^2 \rho$ over the core nodes meets the enclosed mass at the top core node to $2 \times 10^{-3}$. A core integration that read the mantle temperature misses by about $8 \times 10^{-3}$.
- **JAX and numpy paths.** The 2 paths with the same table give the radius $R$, the central pressure $P_c$, the CMB pressure $P_\mathrm{cmb}$ and the central density $\rho_c$ to a relative $10^{-4}$, at an ODE tolerance of $10^{-8}$.
- **A table of the run's own core.** A run whose radius profile also covers the core gives a table of its own core nodes. A second run with that table gives the same $R$, $P_c$, $P_\mathrm{cmb}$ and $\rho_c$ to a relative $10^{-4}$, at an ODE tolerance of $10^{-8}$.
- **Pressures outside the table.** A table that leaves the lowest 5 % of the core pressures out. There the temperature continues with the slope of the end interval. The result counts these nodes, `core_T_nodes_outside`, and gives their largest distance beyond the table as a fraction of its range, `core_T_max_excess`.
- **A table far from the core.** A table with $P$ in GPa instead of Pa puts every core node far beyond it. The solve converges on the extension, but with `core_T_max_excess` above 0.1 it is not accepted (`converged` is False, with a warning). A steep table whose extension reaches $T \le 0$ K at a core node stops with a `StructureInvariantError` that names the table and the node pressure.
- **A realistic table at 10 Earth masses.** The core of the 10 $M_\oplus$ planet on its own adiabat, from 9646 K at 1.49 TPa to 15977 K at 4.62 TPa, given back as a table, with the mantle temperature of that solve: the result is accepted and the core nodes carry the table. The table ends at the first of the centre nodes of that solve, which hold one temperature, so the centre takes the rising end slope.
- **The Newton outer solver.** The default outer solver of PROTEUS, on the JAX path, at the ODE tolerance that PROTEUS sets for it (`relative_tolerance` $10^{-9}$): the node rule and the mass closure hold.
- **The ICB from the table.** A two-layer pure-iron core with the iron melting curve of Anzellini et al. (2013)[^cite-anzellini2013] as `core_liquidus` and `temperature_mode = 'adiabatic_from_cmb'`, the mode of a PROTEUS run; with the table, this mode does not set the core temperature. The table is linear in $P$, with half the slope of the melting curve at 330 GPa, and meets the curve there. All core nodes carry table($P$), also the nodes of the core arrays with their CMB node, and the core mass closes as in the node rule.
- **The numpy fallback.** When every JAX structure solve fails and the solve runs on numpy, the numpy solve keeps the table: the node rule and the mass closure hold.
- **A core that reads no temperature.** An Analytic iron core has no temperature-dependent equation of state. The table is then unused: the result equals the solve without it, bit for bit, `core_T_used` is False and no node is counted outside the table.
- **No table.** `core_temperature_arrays = None` gives the solve without the keyword, bit for bit, and no `core_T_*` keys.
- **Refusals and the checks of an accepted result (unit tier).** A table must be 2 one-dimensional arrays of equal length of at least 2, all finite, with positive, strictly increasing pressures and positive temperatures that do not fall with pressure; any other form or value raises `ValueError`, also in a property test over random bad values. `main` raises `ValueError` before any solve for a bad table and for a table without a mantle temperature. The liquidus rule with `adiabatic_from_cmb` is refused without a table and accepted with one. On hand-written results, an accepted result raises `StructureInvariantError` when a core node is at $T \le 0$ K or a node is more than a relative $10^{-9}$ off its temperature source; with `core_T_max_excess` at 0.09 it stays accepted, at 0.11 it is no longer accepted (`converged` is False); a result that was not accepted is not checked. `is_core`, the rule that places the table, gives the core where the material rule gives the core material, also within round-off of the core mass. The interpolation and its end slopes give the same numbers on numpy and JAX to $10^{-14}$.

**How we know the right answer.** The node rule follows from the input itself: the table gives $T(P)$ in the core. The 2 paths are 2 integrators with their own temperature code, so this is a cross-code check. A table of the run's own core gives the same temperature at each node, so the structure must stay the same (self-consistency). In the ICB case the root $P_B$ of table($P$) = $T_\mathrm{liq}(P)$ is known in closed form, 330 GPa, and the inner-core mass of the solve must equal the mass at $P_B$ to `tolerance_outer` of the core mass.

**Why it matters.** In a coupled run the core temperature sets the core density and the inner-core mass, and both enter the heat budget of the core.

![Core temperature table, iron melting curve and core nodes against pressure, with the ICB of the solve](../assets/validation/core_temperature_table_icb.svg#only-light){ width="80%" }
![Core temperature table, iron melting curve and core nodes against pressure, with the ICB of the solve](../assets/validation/core_temperature_table_icb_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the pressure in GPa, with the centre on the left. The steep line is the melting curve and the straight line the core table. The circles are every 5th core node of the solve, with the CMB node of the core arrays. The dashed line is the ICB of the solve. A pass means that the circles lie on the table line and that the dashed line passes through the crossing of the 2 lines. The other cases compare numbers at the nodes or 4 numbers per run, so they have no plot.

**Known limit.** The 2 paths integrate the equations with their own step control. At the ODE tolerance of the test config, `relative_tolerance` $10^{-5}$, they differ by up to $1.4 \times 10^{-3}$ in $P_\mathrm{cmb}$ and $7.4 \times 10^{-4}$ in $P_c$, also without a table. With the table at $10^{-8}$ they agree to $2.1 \times 10^{-7}$. The 2 cases that compare whole runs therefore run at $10^{-8}$. This command prints the differences: `python -m tools.validation.core_table_path_gap`.

??? info "Test details"
    - Reference: the table itself (node rule); the numpy path (cross-code); the run's own core table (self-consistency); the closed-form root of table($P$) = $T_\mathrm{liq}(P)$ with the melting curve of Anzellini et al. (2013); the solve without the table (unused table).
    - Tolerance: node temperatures to a relative $10^{-9}$; $R$, $P_c$, $P_\mathrm{cmb}$ and $\rho_c$ to a relative $10^{-4}$ at an ODE tolerance of $10^{-8}$; the inner-core mass to `tolerance_outer` ($10^{-3}$) of the core mass from the mass at $P_B$; the core mass closure to $2 \times 10^{-3}$; the node count exactly and the largest distance to a relative $10^{-12}$; not accepted above 0.1 of the table range; the unused table bit for bit.
    - Test ids: `tests/test_core_temperature_arrays.py` (unit tier, every test), `tests/test_core_temperature_arrays_solve.py` (integration tier, every test, nightly job `main`), `tests/test_core_two_phase_solve.py::test_a_two_layer_core_takes_its_icb_from_the_core_temperature_table` (nightly job `core_two_phase`; needs PALEOS and the PALEOS tables under `data/`)
    - Command: `pytest -o "addopts=" tests/test_core_temperature_arrays.py tests/test_core_temperature_arrays_solve.py`; `pytest -o "addopts=" tests/test_core_two_phase_solve.py -k core_temperature_table`

## Reproducing the results

The tests draw the plots of these cases. This command writes them again to `docs/assets/validation/`:

```console
pytest -o "addopts=" tests/test_first_principles.py tests/test_polytrope_verification.py --validation-plots
```

Without `--validation-plots` the tests write nothing: each compares the data of its plot with the committed pair, and a missing or stale file fails the test.

[^cite-seager2007]: S. Seager, M. Kuchner, C. A. Hier-Majumder, B. Militzer, *[Mass-Radius Relationships for Solid Exoplanets](https://doi.org/10.1086/521346)*, The Astrophysical Journal, 669(2), 1279-1297, 2007. [SciX](https://scixplorer.org/abs/2007ApJ...669.1279S/abstract).
[^cite-chandrasekhar1939]: S. Chandrasekhar, *An Introduction to the Study of Stellar Structure*, book (no DOI), Chicago, 1939. [SciX](https://scixplorer.org/abs/1939isss.book.....C/abstract).
[^cite-anzellini2013]: S. Anzellini, A. Dewaele, M. Mezouar, P. Loubeyre, G. Morard, *[Melting of Iron at Earth's Inner Core Boundary Based on Fast X-ray Diffraction](https://doi.org/10.1126/science.1233514)*, Science, 340, 464-466, 2013. [SciX](https://scixplorer.org/abs/2013Sci...340..464A/abstract).
[^cite-labrosse2015]: S. Labrosse, *[Thermal evolution of the core with a high thermal conductivity](https://doi.org/10.1016/j.pepi.2015.02.002)*, Physics of the Earth and Planetary Interiors, 247, 36-55, 2015. [SciX](https://scixplorer.org/abs/2015PEPI..247...36L/abstract).
[^cite-nimmo2015]: F. Nimmo, *[Energetics of the Core](https://doi.org/10.1016/B978-0-444-53802-4.00139-1)*, in Treatise on Geophysics, 2nd edition, vol. 8, 27-55, Elsevier, 2015.
