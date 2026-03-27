# First-Principles Verification

This page documents the first-principles verification of the Zalmoxis ODE integrator, conservation laws, and full solver chain. All tests use either constant-density models or the [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) analytic equation of state, isolating the numerics from tabulated EOS data.

The test suite is implemented in [`tests/test_first_principles.py`](https://github.com/FormingWorlds/Zalmoxis) (25 pytest tests across 3 tiers). A standalone plotting script at `tools/validation/run_first_principles_validation.py` generates all figures shown below.

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

## Test 1: Uniform-density sphere

**What it tests:** The ODE integrator in isolation, with no EOS complexity.

For a sphere of constant density $\rho$, the governing equations have exact closed-form solutions:

$$
M(r) = \frac{4}{3}\pi \rho\, r^3, \qquad
g(r) = \frac{4}{3}\pi G \rho\, r, \qquad
P(r) = P_c - \frac{2}{3}\pi G \rho^2 r^2
$$

The test patches `calculate_mixed_density()` to return a fixed $\rho = 5000$ kg/m$^3$ and compares the numerical profiles against these formulas.

**Setup:** $R = 6.4 \times 10^6$ m, $P_c = 360$ GPa, $N = 300$ radial grid points, `rtol` $= 10^{-10}$.

**Results:** All profiles match the analytical solution to relative error $< 10^{-6}$.

![Uniform-density sphere profiles](../img/verification/uniform_sphere_profiles.png)

Six individual assertions verify:

1. Mass profile: $|M_\mathrm{num} - M_\mathrm{exact}| / M_\mathrm{exact} < 10^{-6}$
2. Gravity profile: $|g_\mathrm{num} - g_\mathrm{exact}| / g_\mathrm{exact} < 10^{-6}$
3. Pressure profile: $|P_\mathrm{num} - P_\mathrm{exact}| / P_\mathrm{exact} < 10^{-6}$
4. Central boundary conditions: $M(0) = 0$, $g(0) = 0$, $P(0) = P_c$ exactly
5. Gravity slope at center: $dg/dr|_{r=0} = \frac{4}{3}\pi G \rho$ to $< 10^{-4}$
6. Pressure monotonicity: $dP/dr < 0$ everywhere $P > 0$

## Test 2: Two-layer sphere

**What it tests:** Layer transition logic at the core-mantle boundary.

A two-layer sphere with constant densities $\rho_c = 13{,}000$ kg/m$^3$ (core) and $\rho_m = 4000$ kg/m$^3$ (mantle), $M_p = 1\, M_\oplus$, core mass fraction $f_c = 0.325$. The CMB radius is:

$$
R_\mathrm{cmb} = \left(\frac{3\, f_c\, M_p}{4\pi \rho_c}\right)^{1/3} = 3291 \text{ km}
$$

The enclosed mass profile is piecewise:

$$
M(r) = \begin{cases}
\frac{4}{3}\pi \rho_c\, r^3 & r \le R_\mathrm{cmb} \\
f_c M_p + \frac{4}{3}\pi \rho_m (r^3 - R_\mathrm{cmb}^3) & r > R_\mathrm{cmb}
\end{cases}
$$

**Results:** Mass profile matches the piecewise formula to $< 10^{-5}$, and Gauss's law ($g = GM/r^2$) is satisfied to $< 10^{-4}$ across both layers including the CMB transition.

![Two-layer sphere](../img/verification/two_layer_sphere.png)

## Test 3: Conservation diagnostics

**What it tests:** Pointwise satisfaction of Gauss's law and hydrostatic equilibrium.

Two algebraic identities must hold at every mesh point for any converged structure:

1. **Gauss's law:** $g(r) = G\,M(r) / r^2$ for all $r > 0$
2. **Hydrostatic balance:** $dP/dr + \rho\, g = 0$ (evaluated via central finite differences)

**Results:** Both residuals are well below $10^{-4}$ at all radii.

![Conservation diagnostics](../img/verification/conservation_diagnostics.png)

## Test 4: Gravitational binding energy

**What it tests:** Consistency of the mass and density profiles via an integral quantity.

The gravitational binding energy is:

$$
E_\mathrm{grav} = -\int_0^R \frac{G\,M(r)}{r}\, 4\pi r^2 \rho\, dr
$$

For uniform density, this evaluates to $E_\mathrm{grav} = -\frac{3}{5}\, G\, M_p^2 / R$. Numerical trapezoidal integration over the $M(r)$ and $\rho(r)$ profiles agrees with the analytical value to $< 1\%$.

## Test 5: Earth benchmark

**What it tests:** The complete solver chain (all three nested loops) against known Earth properties.

**Setup:** $M_p = 1\, M_\oplus$, $f_c = 0.325$, Seager+2007 analytic EOS (iron core, MgSiO$_3$ mantle). No tabulated data needed.

**Results:**

| Quantity | Zalmoxis | Earth | Deviation |
|----------|----------|-------|-----------|
| Radius | 6119 km (0.966 $R_\oplus$) | 6371 km | 4% |
| Central pressure | 391 GPa | 365 GPa | 7% |
| Surface gravity | 10.3 m/s$^2$ | 9.81 m/s$^2$ | 5% |

The deviations are consistent with the Seager modified polytrope being a simplified, temperature-independent EOS.

![Earth benchmark](../img/verification/earth_benchmark.png)

## Test 6: Mass-radius scaling and CMF sensitivity

**What it tests:** Physical scaling laws and monotonicity constraints.

### Mass-radius relation

The mass-radius relation for rocky planets follows an approximate power law $R \propto M^\alpha$ where $\alpha \approx 0.27$ ([Seager et al. 2007](https://iopscience.iop.org/article/10.1086/521346)). Running Zalmoxis from 0.3 to 10 $M_\oplus$ with $f_c = 0.325$ and fitting log $R$ vs log $M$ gives $\alpha = 0.294$.

![Mass-radius scaling](../img/verification/mr_scaling.png)

### CMF monotonicity

At fixed mass ($1\, M_\oplus$), increasing the core mass fraction (replacing less dense silicate with denser iron) must monotonically decrease the radius. Zalmoxis produces a strictly decreasing curve from $R \approx 1.04\, R_\oplus$ at $f_c = 0.05$ to $R \approx 0.77\, R_\oplus$ at $f_c = 0.95$.

The pure iron endpoint ($f_c = 1.0$) gives $R = 0.77\, R_\oplus$, matching the Seager+2007 published value.

![CMF sweep](../img/verification/cmf_sweep.png)

### Additional conservation checks

The integration tests also verify:

- **Mass conservation:** $M(R_\mathrm{surface}) = M_p$ to within the solver tolerance, confirmed independently by trapezoidal integration of $4\pi r^2 \rho(r)$.
- **Surface pressure:** $P(R_\mathrm{surface}) = P_\mathrm{target}$ (101,325 Pa) to within the Brent solver's pressure tolerance.
- **Gauss's law on full solver output:** $g = GM/r^2$ satisfied to $< 10^{-3}$ at all radii in the converged solution.

## Test 7: Numerical convergence

**What it tests:** That the ODE integrator converges with grid refinement and tolerance tightening.

### Grid convergence

Running the uniform-density sphere at $N = 25, 50, 100, 200, 400, 800$ with `rtol` $= 10^{-10}$. The mass and pressure errors reach machine precision ($\sim 10^{-15}$) at $N \ge 100$, confirming that RK45 achieves its theoretical accuracy for this smooth problem.

For the full solver (including Picard iteration and Brent root-finding), the planet radius at $N = 200$ and $N = 400$ agrees to $< 1\%$.

![Grid convergence](../img/verification/grid_convergence.png)

### Tolerance convergence

Reducing `rtol` from $10^{-6}$ to $10^{-12}$ drives the mass error from $\sim 10^{-12}$ to $\sim 10^{-15}$ (machine precision), confirming that the solver tolerance propagates correctly through the integration.

## Summary

| Test | Target | Key metric | Result |
|------|--------|------------|--------|
| 1. Uniform sphere | ODE integration | $\|M_\mathrm{num} - M_\mathrm{exact}\|/M_\mathrm{exact}$ | $< 10^{-6}$ |
| 2. Two-layer sphere | Layer transitions | $\|M_\mathrm{num} - M_\mathrm{exact}\|/M_\mathrm{exact}$ | $< 10^{-5}$ |
| 3. Conservation | Gauss + hydrostatic | Pointwise residuals | $< 10^{-4}$ |
| 4. Binding energy | Energy integral | $\|E_\mathrm{num} - E_\mathrm{exact}\|/\|E_\mathrm{exact}\|$ | $< 1\%$ |
| 5. Earth benchmark | Full solver | $R_p$, $P_c$, $g_\mathrm{surf}$ | 4%, 7%, 5% |
| 6. M-R scaling | Physics | $\alpha$ in $R \propto M^\alpha$ | 0.294 (expected $\sim 0.27$) |
| 7. Convergence | Numerics | Error at $N = 400$ | $\sim 10^{-15}$ (machine prec.) |

These tests are independent of any tabulated EOS data, melting curves, or coupled atmosphere models. They verify the numerical machinery of Zalmoxis from first principles.

## Reproducing the plots

All figures can be regenerated with:

```console
python -m tools.validation.run_first_principles_validation
```

Output is saved to `output/first_principles_validation/` (PDF format). The test suite can be run with:

```console
pytest -o "addopts=" tests/test_first_principles.py -v
```
