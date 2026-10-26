# Mass-radius relation

## What this page checks

This page checks the full structure solver of Zalmoxis: the radius, the central pressure and the gravity of a planet of given mass and core mass fraction (CMF, the mass fraction of the iron core). The cases run the full solver with the Seager et al. (2007) analytic equation of state (EOS) and compare the planet with the Earth, published scaling laws and physical trends. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If the solver is wrong, every planet radius and interior profile of Zalmoxis is wrong. Each case below says why the check exists, what it tests, how we know the right answer and why it matters, and ends with a box "Test details". The [overview](index.md) lists every case of the tab.

## Test 5: Earth benchmark

**Why this check exists.** The solver has 3 nested loops: the mass-radius convergence, the Picard iteration of the density and the Brent root search of the central pressure. An error in any loop can still give a converged but wrong planet, without an error message.

**What it tests.** The complete solver chain against the best-known planet in the solar system. Setup: planet mass $M_p = 1\, M_\oplus$, core mass fraction $f_c = 0.325$, Seager et al. (2007) analytic EOS (iron core, MgSiO$_3$ mantle). No tabulated data files needed. The analytic EOS uses the modified polytrope $\rho(P) = \rho_0 + c P^n$ with published coefficients.

The results:

| Quantity | Zalmoxis | Earth | Deviation |
|----------|----------|-------|-----------|
| Radius | 6116 km | 6371 km | 4% below |
| Central pressure | 391 GPa | 365 GPa | 7% above |
| Surface gravity | 10.67 m/s$^2$ | 9.81 m/s$^2$ | 9% above |

The test compares the radius with the Earth radius of `zalmoxis.constants`, 6335.4 km: 6116 km is 0.965 of that value.

![Earth benchmark: enclosed mass, gravity and pressure against radius, with the reference values and tolerances](../assets/validation/zalmoxis_earth_benchmark.svg#only-light){ width="80%" }
![Earth benchmark: enclosed mass, gravity and pressure against radius, with the reference values and tolerances](../assets/validation/zalmoxis_earth_benchmark_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The 3 panels show the enclosed mass, the gravity and the pressure of the 1 Earth-mass solve against the radius in Earth radii (lines). The squares with error bars are the reference values with their tolerances: $1\,R_\oplus \pm 0.05$ for the radius (top, horizontal bar), 9.8 m/s$^2 \pm 10$ % for the surface gravity (middle) and 360 GPa $\pm 15$ % for the central pressure (bottom). The circles are the code values. A pass means that each circle lies inside its error bar.

**How we know the right answer.** The radius and surface gravity of the Earth, and its central pressure from models (360 GPa in the test; `zalmoxis.constants` cites 365 GPa from Nimmo 2015). The 4-9% deviations are expected and physically understood. The Seager modified polytrope is a temperature-independent, zero-Kelvin EOS fit to the merged Vinet/Birch-Murnaghan (low pressure) and Thomas-Fermi-Dirac (high pressure) equations of state. It does not account for thermal expansion, which makes real Earth's mantle less dense (and therefore larger) than the zero-temperature prediction. The density profile (panel a) shows the core-mantle discontinuity at about 3380 km radius, with iron densities of 11,100-12,900 kg/m$^3$ in the core and silicate densities of 4,100-5,900 kg/m$^3$ in the mantle. The gravity profile (panel c) peaks at the CMB, at about 11.2 m/s$^2$, falls to about 10.4 m/s$^2$ near 4700 km and rises again to 10.67 m/s$^2$ at the surface.

**Why it matters.** This case checks all 3 loops of the solver at once.

**Known limit.** The tolerances are wide (radius 5 %, gravity 10 %, central pressure 15 %), because the zero-temperature EOS is 4-9 % from the Earth. A loop error that moves the radius by about 1 % passes this case.

??? info "Test details"
    - Reference: Earth values as the test pins them: radius $1\,R_\oplus$, surface gravity 9.8 m/s$^2$, central pressure 360 GPa (the test names no source; the table above compares with 6371 km, 9.81 m/s$^2$ and 365 GPa)
    - Tolerance: radius within `abs=0.05` of $1\,R_\oplus$, surface gravity within `rel=0.10` of 9.8 m/s$^2$, central pressure within `rel=0.15` of 360 GPa
    - Test id: `tests/test_first_principles.py::TestEarthBenchmark`
    - Command: `pytest -o "addopts=" tests/test_first_principles.py::TestEarthBenchmark`

## Test 6: Mass-radius scaling and CMF sensitivity

**Why this check exists.** A solver can match one planet and still give wrong trends with mass or composition. Gross errors in a trend fail these checks; the exponent test accepts 0.23 to 0.31, so a small slope error passes.

**What it tests.** Whether the solver gives physically correct trends across parameter space, not just a single point: the mass-radius relation, the radius against the CMF, and conservation checks.

### Mass-radius relation

For rocky planets with Earth-like composition, the mass-radius relation follows an approximate power law $R \propto M^\alpha$ where $\alpha \approx 0.27$ ([Seager et al. 2007](https://iopscience.iop.org/article/10.1086/521346)). The exponent is less than $1/3$ (the value for an incompressible sphere of constant density) because larger planets have higher internal pressures that compress the material, partially offsetting the volume increase from added mass.

Running Zalmoxis from 0.3 to 10 $M_\oplus$ with $f_c = 0.325$ and fitting $\log R$ vs $\log M$ gives $\alpha = 0.293$ (the test fits 0.5 to 10 $M_\oplus$ and gets 0.289):

![Radius against mass of 5 solves on log axes, with the fitted slope and the slopes 0.23 and 0.31](../assets/validation/zalmoxis_mass_radius_scaling.svg#only-light){ width="80%" }
![Radius against mass of 5 solves on log axes, with the fitted slope and the slopes 0.23 and 0.31](../assets/validation/zalmoxis_mass_radius_scaling_dark.svg#only-dark){ width="80%" }

**How to read the plot.** Both axes are logarithmic. The marks are the code radii of the 5 solves at 0.5, 1, 2, 5 and 10 Earth masses. The solid line is their least-squares fit, of slope 0.289. The dotted lines have the slopes 0.23 and 0.31, the edges of the asserted range $0.27 \pm 0.04$, drawn through the centre of the fit. A pass means that the slope of the solid line lies between the slopes of the dotted lines.

The fitted exponent is slightly higher than the Seager et al. value of 0.27, which reflects the sensitivity of the power-law fit to the mass range used. Seager et al. fit over 1-10 $M_\oplus$, where Zalmoxis gives 0.284; including sub-Earth masses (0.3 $M_\oplus$) steepens the slope because low-mass planets are less compressed, making the $R(M)$ curve slightly steeper at the low-mass end.

### CMF monotonicity

At fixed mass ($1\, M_\oplus$), increasing the core mass fraction (replacing less dense silicate with denser iron) must monotonically decrease the radius. This is a basic physical constraint: iron ($\rho_0 = 8300$ kg/m$^3$) is roughly twice as dense as MgSiO$_3$ ($\rho_0 = 4100$ kg/m$^3$), so more iron means a more compact planet.

Zalmoxis produces a strictly decreasing curve from $R \approx 1.02\, R_\oplus$ at $f_c = 0.05$ (nearly pure silicate) to $R \approx 0.81\, R_\oplus$ at $f_c = 0.95$ (nearly pure iron), in units of the 6335.4 km Earth radius of `zalmoxis.constants`:

![Radius of a 1 Earth-mass planet against core mass fraction, with the pure-iron radius](../assets/validation/zalmoxis_cmf_sweep.svg#only-light){ width="80%" }
![Radius of a 1 Earth-mass planet against core mass fraction, with the pure-iron radius](../assets/validation/zalmoxis_cmf_sweep_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The circles joined by the line are the code radii at the core mass fractions 0.1, 0.2, 0.3, 0.5 and 0.7. The monotonicity test asserts that each radius is smaller than the one before. The triangle is the code radius of a pure iron planet, which the pure-iron test asserts against $0.77\,R_\oplus$ (square, with its error bar of 0.05). A pass means that the line falls at every step and the triangle lies inside the error bar.

A pure iron planet ($f_c = 1$) has $0.80\, R_\oplus$, against the Seager et al. (2007) value of $0.77\, R_\oplus$ for a 1 $M_\oplus$ iron planet; the test accepts `abs=0.05`. The radius falls faster with CMF at high CMF: by 0.043 $R_\oplus$ from $f_c = 0.3$ to 0.5 and by 0.047 $R_\oplus$ from 0.5 to 0.7.

### Conservation checks on full solver output

The smoke tests of `TestConservationLaws` also verify three conservation properties of the full solver output:

- **Mass conservation:** $M(R_\mathrm{surface}) = M_p$ to within 1 %, and to within 2 % by trapezoidal integration of $4\pi r^2 \rho(r)$ over the solver's own density and radii.
- **Surface pressure:** $P(R_\mathrm{surface}) = P_\mathrm{target}$ (101,325 Pa) to within 2 GPa, twice the default surface-pressure tolerance of the solver (1 GPa).
- **Gauss's law:** the integrated $g$ equals $GM/r^2$ to $< 10^{-3}$ relative error at every radius except the first 6 grid points and the points with $P \leq 0$.

![Mass of the solve over the input mass by 2 integrations, and pressure against radius with the 2 GPa bound](../assets/validation/zalmoxis_conservation_full_solver.svg#only-light){ width="80%" }
![Mass of the solve over the input mass by 2 integrations, and pressure against radius with the 2 GPa bound](../assets/validation/zalmoxis_conservation_full_solver_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The left panel shows the mass of the solve over the input mass: the outermost ODE mass and the trapezoidal integral (circles), against 1 with error bars of 1 % and 2 % (squares). The right panel shows the code pressure against the radius on a log scale, with the surface value (circle) and the 2 GPa bound (dotted). A pass means that each circle lies inside its error bar and the surface circle lies below the dotted line. The Gauss's law check is not drawn.

**Known limit.** The surface-pressure check allows 2 GPa against a target of 101,325 Pa, so it cannot tell a correct surface pressure from one 10000 times too large. The solver treats a residual below 1 GPa as pressure-converged, but it can also accept a solution on mass convergence alone, so this 2 GPa check is then the only guard. The mass checks allow 1 % and 2 %.

**How we know the right answer.** The scaling law and the pure-iron radius of Seager et al. (2007), and physical constraints (monotonicity, mass, surface pressure, Gauss's law) with no published reference.

**Why it matters.** These trends are what a grid of Zalmoxis planets depends on.

??? info "Test details"
    - Reference: `TestMassRadiusScaling`, $\alpha \approx 0.27$ for masses 0.5, 1, 2, 5 and 10 $M_\oplus$, and `TestPureIronPlanet`, $0.77\,R_\oplus$ for CMF = 1.0, both from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346); `TestCMFMonotonicity` (CMF 0.1 to 0.7) and `TestConservationLaws` check physical constraints with no published reference
    - Tolerance: exponent within `abs=0.04` of 0.27; strictly decreasing radius with CMF; pure-iron radius within `abs=0.05` of $0.77\,R_\oplus$; mass within 1% (2% by trapezoidal integration); surface pressure within 2e9 Pa of the target; Gauss's law `rtol=1e-3`
    - Test id: `tests/test_first_principles.py::TestMassRadiusScaling`, `tests/test_first_principles.py::TestCMFMonotonicity`, `tests/test_first_principles.py::TestPureIronPlanet`, `tests/test_first_principles.py::TestConservationLaws`
    - Command: `pytest -o "addopts=" tests/test_first_principles.py::TestMassRadiusScaling tests/test_first_principles.py::TestCMFMonotonicity tests/test_first_principles.py::TestPureIronPlanet tests/test_first_principles.py::TestConservationLaws`

## Reproducing the results

The tests draw the plots of these cases. This command writes them again to `docs/assets/validation/`:

```console
pytest -o "addopts=" tests/test_first_principles.py tests/test_polytrope_verification.py --validation-plots
```

Without `--validation-plots` the tests write nothing: each compares the data of its plot with the committed pair, and a missing or stale file fails the test.
