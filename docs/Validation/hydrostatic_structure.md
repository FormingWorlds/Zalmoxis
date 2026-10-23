# Hydrostatic structure

These cases check the integration of the structure equations against closed-form solutions, conservation laws and a second implementation of the same equations. The [overview](index.md) lists every case of the tab.

## Governing equations

Zalmoxis solves three coupled ordinary differential equations for the radial structure of a self-gravitating, spherically symmetric body in hydrostatic equilibrium:

$$
\frac{dM}{dr} = 4\pi r^2 \rho(r)
$$

$$
\frac{dg}{dr} = 4\pi G \rho(r) - \frac{2\,g(r)}{r}
$$

$$
\frac{dP}{dr} = -\rho(r)\, g(r)
$$

where $M(r)$ is the enclosed mass, $g(r)$ is gravitational acceleration, $P(r)$ is pressure, and $\rho(r)$ is density. These are integrated outward from the center ($r = 0$) with initial conditions $M(0) = 0$, $g(0) = 0$, $P(0) = P_c$.

Note that the gravity equation is singular at $r = 0$ due to the $2g/r$ term. L'Hopital's rule gives the analytic limit $dg/dr|_{r=0} = \frac{4}{3}\pi G \rho_c$, which `coupled_odes` in `zalmoxis.structure_model` uses at $r = 0$.

The system is closed by an equation of state $\rho = \rho(P, T)$, but for verification purposes we either fix $\rho$ to a constant (Tests 1-4) or use the Seager et al. (2007) analytic polytrope $\rho(P) = \rho_0 + c P^n$ (Tests 5-7), both of which eliminate any dependence on tabulated EOS data.

## Test 1: Uniform-density sphere

**What it tests:** The ODE integrator in complete isolation from any EOS complexity. By patching `calculate_mixed_density()` to return a constant, we test only the RK45 integration of the three coupled ODEs.

**Why it matters:** If the ODE coefficients (the $4\pi$ factors, the $2g/r$ singularity handling at $r = 0$, the signs) contain any error, this test will catch it with zero ambiguity. The analytic solution is a textbook result with no free parameters.

For a sphere of constant density $\rho$, the governing equations have exact closed-form solutions:

$$
M(r) = \frac{4}{3}\pi \rho\, r^3, \qquad
g(r) = \frac{4}{3}\pi G \rho\, r, \qquad
P(r) = P_c - \frac{2}{3}\pi G \rho^2 r^2
$$

The mass grows as $r^3$ (volume of a sphere), gravity is linear in $r$ (Gauss's law for a uniform sphere), and pressure is a downward parabola (hydrostatic equilibrium with linearly increasing gravity).

**Setup:** $\rho = 5000$ kg/m$^3$ (representative rocky planet average), $R = 6.4 \times 10^6$ m ($\approx 1\, R_\oplus$), $P_c = 360$ GPa, $N = 300$ radial grid points, `rtol` $= 10^{-10}$.

**Results:** At $N = 300$ the largest relative errors are $3 \times 10^{-15}$ for the mass, $1 \times 10^{-15}$ for gravity and $4 \times 10^{-16}$ for pressure, the round-off level; the test asserts $< 10^{-6}$. The curves are visually indistinguishable:

![Uniform-density sphere profiles](../img/validation/uniform_sphere_profiles.png)

Six individual assertions verify specific aspects:

1. **Mass profile accuracy:** $|M_\mathrm{num} - M_\mathrm{exact}| / M_\mathrm{exact} < 10^{-6}$ at all $r > 0$
2. **Gravity profile accuracy:** $|g_\mathrm{num} - g_\mathrm{exact}| / g_\mathrm{exact} < 10^{-6}$ at all $r > 0$
3. **Pressure profile accuracy:** $|P_\mathrm{num} - P_\mathrm{exact}| / P_\mathrm{exact} < 10^{-6}$ where $P > 0$
4. **Central boundary conditions:** $M(0) = 0$, $g(0) = 0$, $P(0) = P_c$ to machine precision
5. **Gravity slope at center:** $dg/dr|_{r=0}$ matches the analytic limit $\frac{4}{3}\pi G \rho$ to $< 10^{-4}$, verifying the singularity handling
6. **Pressure monotonicity:** $dP/dr < 0$ everywhere $P > 0$, as required by hydrostatic equilibrium with positive density and gravity

- Reference: closed-form solution of the uniform-density sphere
- Tolerance: `rtol=1e-6` on the mass, gravity and pressure profiles; `rel=1e-4` on the central gravity slope
- Test id: `tests/test_first_principles.py::TestUniformDensitySphere`
- Command: `pytest -o "addopts=" tests/test_first_principles.py::TestUniformDensitySphere`

## Test 2: Two-layer sphere

**What it tests:** The layer transition logic at the core-mantle boundary (CMB). Zalmoxis assigns each radial shell to a layer (core or mantle) based on the enclosed mass. A density discontinuity at the CMB is the sharpest feature the integrator must handle, and errors in the boundary comparison (e.g., off-by-one in the mass threshold, wrong tolerance for floating-point comparison) would produce visible deviations from the piecewise analytical solution.

**Setup:** A two-layer sphere with constant densities $\rho_c = 13{,}000$ kg/m$^3$ (iron-like core) and $\rho_m = 4000$ kg/m$^3$ (silicate-like mantle), $M_p = 1\, M_\oplus$, core mass fraction $f_c = 0.325$. The CMB radius is:

$$
R_\mathrm{cmb} = \left(\frac{3\, f_c\, M_p}{4\pi \rho_c}\right)^{1/3} = 3291 \text{ km}
$$

This is close to Earth's actual CMB radius of 3480 km, and the total planet radius is 6513 km (1.02 times 6371 km). The central pressure of this profile, 380 GPa, comes from integrating the hydrostatic equation over the known density profile; the integration starts at 5% above it, 399 GPa, so that the pressure stays positive out to the surface.

The enclosed mass profile is piecewise:

$$
M(r) = \begin{cases}
\frac{4}{3}\pi \rho_c\, r^3 & r \le R_\mathrm{cmb} \\
f_c M_p + \frac{4}{3}\pi \rho_m (r^3 - R_\mathrm{cmb}^3) & r > R_\mathrm{cmb}
\end{cases}
$$

**Results:** The largest relative error of the mass profile against the piecewise formula is $3 \times 10^{-9}$, and Gauss's law ($g = GM/r^2$) holds to $1 \times 10^{-10}$ across both layers including the CMB transition; the test asserts $< 10^{-5}$ and $< 10^{-4}$. The left panel shows the mass profile with the CMB location marked; the right panel overlays the numerical gravity with the Gauss's law prediction:

![Two-layer sphere](../img/validation/two_layer_sphere.png)

The smooth transition across the CMB confirms that `get_layer_mixture()` correctly assigns shells to layers and that the ODE integrator handles the density discontinuity without introducing spurious oscillations.

- Reference: piecewise closed-form solution of the two-layer sphere
- Tolerance: `rtol=1e-5` on the mass profile, `rtol=1e-4` on Gauss's law
- Test id: `tests/test_first_principles.py::TestTwoLayerSphere`
- Command: `pytest -o "addopts=" tests/test_first_principles.py::TestTwoLayerSphere`

## Test 3: Conservation diagnostics

**What it tests:** Pointwise satisfaction of two fundamental conservation laws that must hold for any physically consistent solution, independent of the EOS.

**Gauss's law:** $g(r) = G\,M(r) / r^2$ for all $r > 0$. This is an algebraic identity relating the gravitational acceleration to the enclosed mass. Since Zalmoxis integrates both $M(r)$ and $g(r)$ as separate ODE components (rather than computing $g$ algebraically from $M$), any drift in the integration would cause the two to disagree. The relative residual $|g - GM/r^2| / (GM/r^2)$ quantifies this drift.

**Hydrostatic balance:** $dP/dr + \rho\, g = 0$ at every interior point. This is the defining equation of the solver; the residual is evaluated with central finite differences. For the uniform sphere $P(r)$ is a parabola, so the central difference is exact and the residual shows the integration error alone.

**Results:** Both residuals are at round-off level at all radii, far below the asserted $10^{-5}$ and $10^{-4}$:

![Conservation diagnostics](../img/validation/conservation_diagnostics.png)

On the 500-point grid of the figure, the Gauss's law residual (left) is at most $2.7 \times 10^{-15}$ and the hydrostatic residual (right) at most $1.4 \times 10^{-13}$.

- Reference: Gauss's law $g = GM/r^2$ and hydrostatic balance $dP/dr + \rho g = 0$
- Tolerance: `rtol=1e-5` on Gauss's law, largest relative hydrostatic residual below `1e-4`
- Test id: `tests/test_first_principles.py::TestGaussLaw`, `tests/test_first_principles.py::TestHydrostaticBalance`
- Command: `pytest -o "addopts=" tests/test_first_principles.py::TestGaussLaw tests/test_first_principles.py::TestHydrostaticBalance`

## Test 4: Gravitational binding energy

**What it tests:** Global consistency of the mass and density profiles via an integral quantity.

The gravitational binding energy, the energy required to disperse the planet to infinity, is:

$$
E_\mathrm{grav} = -\int_0^R \frac{G\,M(r)}{r}\, 4\pi r^2 \rho\, dr
$$

For uniform density, this integral evaluates to the well-known result $E_\mathrm{grav} = -\frac{3}{5}\, G\, M_p^2 / R$. Numerical trapezoidal integration over the $M(r)$ and $\rho(r)$ profiles ($N = 500$) agrees with the analytical value to $7 \times 10^{-6}$; the test asserts $< 1\%$.

This test is complementary to Tests 1-3: it verifies that errors do not cancel when integrated over the full planetary volume.

- Reference: uniform-sphere binding energy $E_\mathrm{grav} = -\frac{3}{5} G M_p^2 / R$
- Tolerance: `rel=0.01`
- Test id: `tests/test_first_principles.py::TestGravitationalBindingEnergy`
- Command: `pytest -o "addopts=" tests/test_first_principles.py::TestGravitationalBindingEnergy`

## n = 1 polytrope

**What it tests:** `solve_structure` with the `Analytic:polytrope_n1` verification material, $P = K \rho^2$, against the closed-form Lane-Emden solution $\rho(r) = \rho_c \sin\xi / \xi$ with $\xi = \pi r / R$. For $n = 1$ the radius $R = \pi \sqrt{K / 2\pi G}$ depends on $K$ only, not on the central density; $K$ is set so that $R$ is one Earth radius.

**Setup:** 600 radial points from the centre to $1.03\,R_\oplus$; the surface radius comes from extrapolating the density to zero.

- Reference: the analytic $n = 1$ Lane-Emden solution (Chandrasekhar 1939, reference named in the test docstring)
- Tolerance: surface radius within `rel=1e-4` of $1\,R_\oplus$; integrated mass within `rel=1e-4` of the seeded $2\,M_\oplus$; radius at $1\,M_\oplus$ and $2\,M_\oplus$ equal within `rel=1e-4` and within `rel=1e-3` of $1\,R_\oplus$
- Test id: `tests/test_polytrope_verification.py`
- Command: `pytest -o "addopts=" tests/test_polytrope_verification.py`

## Test 7: Numerical convergence

**What it tests:** That the error does not grow as the grid is refined and the tolerances are tightened, and that the full solver gives the same radius on two grids.

### Grid convergence

Running the uniform-density sphere at $N = 25, 50, 100, 200, 400, 800$ with `rtol` $= 10^{-10}$:

![Grid convergence](../img/validation/grid_convergence.png)

The mass error is $2$ to $3 \times 10^{-15}$ and the pressure error about $4 \times 10^{-16}$ at every $N$ from 25 to 800. The uniform-sphere solution is a cubic and a parabola in $r$, which the integrator reproduces to round-off on every grid, so this case shows no convergence rate.

For the full solver (including Picard iteration and Brent root-finding) at $1\,M_\oplus$, the planet radius is the same, $0.96544\,R_\oplus$, at $N = 100$, 200 and 400; the test asserts that $N = 200$ and $N = 400$ agree to $< 1\%$.

### Tolerance convergence

For `rtol` from $10^{-6}$ to $10^{-12}$ at $N = 200$ the mass error stays between $1.5$ and $3.5 \times 10^{-15}$, with no trend: the error is at round-off level for every tolerance tested.

- Reference: closed-form solution of the uniform-density sphere; for the full solver, the radius at $N = 200$ against $N = 400$ (self-consistency)
- Tolerance: the error may rise by at most a factor 1.5 from one grid or tolerance step to the next above a `1e-13` floor; mass error below `1e-8` and pressure error below `1e-6` at $N = 400$; full-solver radius at $N = 200$ and $N = 400$ within 1%; mass error below `1e-10` at `rtol=1e-12`
- Test id: `tests/test_first_principles.py::TestGridConvergence`, `tests/test_first_principles.py::TestToleranceConvergence`
- Command: `pytest -o "addopts=" tests/test_first_principles.py::TestGridConvergence tests/test_first_principles.py::TestToleranceConvergence`

## JAX and numpy right-hand sides

**What it tests:** The optional JAX path (`use_jax = True`) evaluates the right-hand side of the structure equations in `zalmoxis.jax_eos.rhs.coupled_odes_jax`. These cases compare it with the numpy `zalmoxis.structure_model.coupled_odes` on the same states. Two implementations of the same equations must give the same $dM/dr$, $dg/dr$ and $dP/dr$, so this is a cross-code check.

**Setup:** The table cases draw random states with a fixed seed: $r$ from $10^5$ to $6.4 \times 10^6$ m, $M$ from 0 to $1\, M_\oplus$, $g$ from 0.5 to 25 m/s$^2$ and $P$ from $10^6$ to $3 \times 10^{11}$ Pa, with a PALEOS iron core and a PALEOS MgSiO$_3$ mantle. A state counts when both implementations return a nonzero right-hand side. JAX must never return a nonzero value where numpy returns zero; numpy may return a nonzero value where JAX returns zero in at most 4 of 200 states (the nearest-neighbour fallback of the numpy tables, which the JAX path does not have); more than 100 states must count. The synthetic cases build small tables in memory, so they need no EOS files and run in the unit tier.

| Test id | Mantle | Tolerance | Tier |
|---|---|---|---|
| `tests/test_jax_unified_mantle_parity.py::test_unified_mantle_dry_parity` | unified MgSiO$_3$, mushy zone factor 0.8 | largest relative difference `1e-6` | integration |
| `tests/test_jax_unified_mantle_parity.py::test_unified_mantle_wet_parity` | unified MgSiO$_3$ with 1% PALEOS H$_2$O | `1e-5` | integration |
| `tests/test_jax_unified_mantle_parity.py::test_unified_wet_parity_without_melting_curves` | as above, without external melting curves | `1e-5` | integration |
| `tests/test_jax_wet_mantle_parity.py::test_coupled_odes_jax_wet_parity` | 2-phase MgSiO$_3$ (solid and melted sub-tables) with 1% PALEOS H$_2$O | `1e-5` | integration |
| `tests/test_jax_parity_synthetic.py::test_unified_kernel_parity_synthetic` | synthetic unified table, density kernel | `rel=1e-12` | unit |
| `tests/test_jax_parity_synthetic.py::test_rhs_parity_synthetic_unified_wet_and_dry` | synthetic unified mantle, dry and wet | `rtol=1e-10` | unit |

### Plot: dry unified mantle

![Relative difference between the JAX and numpy right-hand sides](../assets/validation/jax_unified_mantle_dry_parity.svg#only-light){ width="80%" }
![Relative difference between the JAX and numpy right-hand sides](../assets/validation/jax_unified_mantle_dry_parity_dark.svg#only-dark){ width="80%" }

Largest relative difference of the three right-hand-side terms between `coupled_odes_jax` and `coupled_odes` at each compared state (markers), against the pressure of that state, for the PALEOS iron core and the unified PALEOS MgSiO$_3$ mantle. The dashed line is the tolerance.

- Reference: numpy `coupled_odes` (cross-code check).
- Tolerance: largest relative difference `1e-6` over all compared states.
- Test id: `tests/test_jax_unified_mantle_parity.py::test_unified_mantle_dry_parity`
- Command: `pytest -m reference_pinned tests/test_jax_unified_mantle_parity.py --validation-plots`

## Surface residual of the central pressure solve

The central pressure solve finds the root of $f(P_c) = P(R) - P_\mathrm{target}$. A solve whose pressure ends at $r_\mathrm{stop} < R$, at the pressure $P_\mathrm{stop}$ (zero within round-off at the zero-pressure event, or up to $\max(10^{-6} P_c, P_\mathrm{target})$ where the right-hand side fails near the surface), takes
$f = P_\mathrm{stop} - P_\mathrm{target} - k (R - r_\mathrm{stop})$ with $k = \rho_\mathrm{ref} G M_\mathrm{target} / R^2$, where $\rho_\mathrm{ref}$ is the outermost density of the density iterate; $k$ is fixed for one pressure solve. Where the stop reaches $R$, $P(R)$ falls to $P_\mathrm{stop}$, so the two branches meet, and their slopes differ by the factor $\rho_\mathrm{ref} g_\mathrm{ref} / (\rho_s g_s)$, close to 1 once the mass loop has converged. The steps of the integrator still move $f$ by up to its noise (about 4e3 Pa on the analytic EOS, 2e5 Pa on PALEOS at `rtol = 1e-9`), so $f$ is continuous at the stop but not smooth everywhere.

### Plot: continuity at a zero-pressure stop

![Surface residual across the central pressure where the stop reaches R](../assets/validation/surface_residual_continuity.svg#only-light){ width="80%" }
![Surface residual across the central pressure where the stop reaches R](../assets/validation/surface_residual_continuity_dark.svg#only-dark){ width="80%" }

Surface residual over the target pressure at 201 central pressures (every second one drawn) around the root of a 1 Earth-mass planet on the Seager et al. (2007) analytic iron and MgSiO$_3$ EOS, 150 nodes, target 101325 Pa. Circles: solves that stop below $R$; squares: solves that reach $R$; the dashed line marks $-1$, where the stop reaches $R$. Each step of 2.9e5 Pa in $P_c$ moves $f$ by 9.7e4 to 1.2e5 Pa, so a jump at the stop would show far above the integrator noise.

- Reference: the closed form of the stopped branch above, equal to $-P_\mathrm{target}$ at $r_\mathrm{stop} = R$ for a zero-pressure stop.
- Tolerance: every step rises; the median step exceeds 10 times the noise of 4e3 Pa; no step exceeds 3 times the median step.
- Test id: `tests/test_surface_residual.py::test_the_residual_rises_through_the_surface_stop`
- Command: `pytest -m reference_pinned tests/test_surface_residual.py --validation-plots`

### Plot: continuity at a stop above zero pressure

![Surface residual across a stop at 5 MPa](../assets/validation/surface_residual_stop_pressure.svg#only-light){ width="80%" }
![Surface residual across a stop at 5 MPa](../assets/validation/surface_residual_stop_pressure_dark.svg#only-dark){ width="80%" }

Surface residual at 41 central pressures, 1e5 Pa apart, across the central pressure where the stop reaches $R$, for the PALEOS profile of the JAX parity tests with the temperature at 0.6 of its values, a target surface pressure of 1e7 Pa, and a temperature undefined below 5e6 Pa, which stands for an EOS or temperature failure near the surface. Circles: solves that stop below $R$, all at $P_\mathrm{stop} = 5.0$ MPa; squares: solves that reach $R$. The dashed line is the stopped branch without $P_\mathrm{stop}$, at $-P_\mathrm{target}$: it would put a step of 5.0e6 Pa at the junction, where the measured step is 2.9e4 Pa.

- Reference: the closed form of the stopped branch above, equal to $P(R) - P_\mathrm{target}$ at $r_\mathrm{stop} = R$.
- Tolerance: no step, the junction step included, exceeds $0.1 P_\mathrm{stop}$ = 5e5 Pa.
- Test id: `tests/test_surface_residual_stop_pressure.py::test_the_residual_has_no_jump_at_a_stop_above_zero_pressure`
- Command: `pytest -m reference_pinned tests/test_surface_residual_stop_pressure.py --validation-plots`

## Reproducing the results

The first-principles figures can be made again, as PDF files, with:

```console
python -m tools.validation.run_first_principles_validation
```

Output is saved to `output/first_principles_validation/` (PDF format). The full test suite can be run with:

```console
pytest -o "addopts=" tests/test_first_principles.py -v
```
