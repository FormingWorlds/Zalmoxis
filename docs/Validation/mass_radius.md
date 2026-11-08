# Mass-radius relation

## What this page checks

This page checks the full structure solver of Zalmoxis: the radius, the central pressure and the gravity of a planet of given mass and core mass fraction (CMF, the mass fraction of the iron core). The cases run the full solver with the Seager et al. (2007) [^cite-seager2007] analytic equation of state (EOS), the PREM case with the PALEOS tables, and compare the planet with the Earth, published scaling laws and physical trends. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If the solver is wrong, every planet radius and interior profile of Zalmoxis is wrong. Each case below says why the check exists, what it tests, how we know the right answer and why it matters, and ends with a box "Test details". The [overview](index.md) lists every case of the tab.

## Earth benchmark

**Why this check exists.** The solver has 3 nested loops: the mass-radius convergence, the Picard iteration of the density and the Brent root search of the central pressure. A gross error in any loop can still give a converged but wrong planet, without an error message; the case catches it only when it takes the radius, the surface gravity or the central pressure outside its wide tolerance (5, 10 and 15 %).

**What it tests.** The complete solver chain against the best-known planet in the solar system. Setup: planet mass $M_p = 1\, M_\oplus$, core mass fraction $f_c = 0.325$, Seager et al. (2007) analytic EOS (iron core, MgSiO$_3$ mantle). No tabulated data files needed. The analytic EOS uses the modified polytrope $\rho(P) = \rho_0 + c P^n$ with published coefficients.

The results:

| Quantity | Zalmoxis | Earth | Deviation |
|----------|----------|-------|-----------|
| Radius | 6114.5 km | 6371 km | 4% below |
| Central pressure | 391 GPa | 365 GPa | 7% above |
| Surface gravity | 10.67 m/s$^2$ | 9.81 m/s$^2$ | 9% above |

The test compares the radius with the Earth radius of `zalmoxis.constants`, 6335.4 km: 6114.5 km is 0.965 of that value.

![Enclosed mass of the 1 Earth-mass solve against radius, with the planet radius and its tolerance](../assets/validation/zalmoxis_earth_benchmark.svg#only-light){ width="80%" }
![Enclosed mass of the 1 Earth-mass solve against radius, with the planet radius and its tolerance](../assets/validation/zalmoxis_earth_benchmark_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The line is the enclosed mass of the 1 Earth-mass solve against radius and gives context. The radius test asserts the planet radius (circle) against $1\,R_\oplus$ (square, with the horizontal error bar 0.05). A pass means that the circle lies inside the error bar.

![Gravity of the 1 Earth-mass solve against radius, with the surface gravity and its tolerance](../assets/validation/zalmoxis_earth_gravity.svg#only-light){ width="80%" }
![Gravity of the 1 Earth-mass solve against radius, with the surface gravity and its tolerance](../assets/validation/zalmoxis_earth_gravity_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The line is the gravity of the same solve. The gravity test asserts the surface gravity (circle) against 9.8 m/s$^2$ (square, with its error bar of 10 %). The circle, 10.67 m/s$^2$, lies close to the upper end of the bar, 10.78 m/s$^2$. A pass means that the circle lies inside the error bar.

![Pressure of the 1 Earth-mass solve against radius, with the central pressure and its tolerance](../assets/validation/zalmoxis_earth_central_pressure.svg#only-light){ width="80%" }
![Pressure of the 1 Earth-mass solve against radius, with the central pressure and its tolerance](../assets/validation/zalmoxis_earth_central_pressure_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The line is the pressure of the same solve. The pressure test asserts the central pressure (circle) against 360 GPa (square, with its error bar of 15 %). A pass means that the circle lies inside the error bar.

**How we know the right answer.** The radius and surface gravity of the Earth, and its central pressure from models (360 GPa in the test; `zalmoxis.constants` cites 365 GPa from Nimmo 2015). The 4-9% deviations are expected and physically understood. The Seager modified polytrope is a temperature-independent, zero-Kelvin EOS fit to the merged Vinet/Birch-Murnaghan (low pressure) and Thomas-Fermi-Dirac (high pressure) equations of state. It does not account for thermal expansion, which makes real Earth's mantle less dense (and therefore larger) than the zero-temperature prediction. In this solve the density jumps at the core-mantle boundary at about 3380 km radius, with iron densities of 11,100-12,900 kg/m$^3$ in the core and silicate densities of 4,100-5,900 kg/m$^3$ in the mantle. The gravity peaks at the CMB, at about 11.2 m/s$^2$, falls to about 10.4 m/s$^2$ near 4700 km and rises again to 10.67 m/s$^2$ at the surface. No assert checks these profile values; only the surface gravity is asserted.

**Why it matters.** This case checks all 3 loops of the solver at once.

**Known limit.** The tolerances are wide (radius 5 %, gravity 10 %, central pressure 15 %), because the zero-temperature EOS is 4-9 % from the Earth. A loop error that moves the radius by about 1 % passes this case.

??? info "Test details"
    - Reference: Earth values as the test pins them: radius $1\,R_\oplus$, surface gravity 9.8 m/s$^2$, central pressure 360 GPa (the test names no source; the table above compares with 6371 km, 9.81 m/s$^2$ and 365 GPa)
    - Tolerance: radius to `abs=0.05` of $1\,R_\oplus$, surface gravity to `rel=0.10` of 9.8 m/s$^2$, central pressure to `rel=0.15` of 360 GPa
    - Test id: `tests/test_first_principles.py::TestEarthBenchmark`
    - Command: `pytest -o "addopts=" tests/test_first_principles.py::TestEarthBenchmark`

## Mass-radius scaling and CMF sensitivity

**Why this check exists.** A solver can match one planet and still give wrong trends with mass or composition. Gross errors in a trend fail these checks; the exponent test accepts 0.23 to 0.31, so a small slope error passes.

**What it tests.** Whether the solver gives physically correct trends across parameter space, not just a single point: the mass-radius relation, the radius against the CMF, and conservation checks.

### Mass-radius relation

For rocky planets with Earth-like composition, the mass-radius relation follows an approximate power law $R \propto M^\alpha$ where $\alpha \approx 0.27$, after Seager et al. (2007) [^cite-seager2007]. The exponent is less than $1/3$ (the value for an incompressible sphere of constant density) because larger planets have higher internal pressures that compress the material, partially offsetting the volume increase from added mass.

Running Zalmoxis from 0.3 to 10 $M_\oplus$ with $f_c = 0.325$ and fitting $\log R$ vs $\log M$ gives $\alpha = 0.293$ (the test fits 0.5 to 10 $M_\oplus$ and gets 0.289). The plot shows the solves of the test.

![Radius against mass of 5 solves on log axes, with the fitted slope and the slopes 0.23 and 0.31](../assets/validation/zalmoxis_mass_radius_scaling.svg#only-light){ width="80%" }
![Radius against mass of 5 solves on log axes, with the fitted slope and the slopes 0.23 and 0.31](../assets/validation/zalmoxis_mass_radius_scaling_dark.svg#only-dark){ width="80%" }

**How to read the plot.** Both axes are logarithmic. The circles are the code radii of the 5 solves at 0.5, 1, 2, 5 and 10 Earth masses. The solid line is their least-squares fit, with exponent $\alpha$ = 0.289. The dotted lines have the exponents 0.23 and 0.31, the edges of the asserted range $0.27 \pm 0.04$, drawn through the centre of the fit. A pass means that the exponent of the solid line lies between those of the dotted lines.

The fitted exponent is slightly higher than the Seager et al. value of 0.27, which reflects the sensitivity of the power-law fit to the mass range used. Seager et al. fit over 1-10 $M_\oplus$, where Zalmoxis gives 0.284; including sub-Earth masses (0.3 $M_\oplus$) steepens the slope because low-mass planets are less compressed, making the $R(M)$ curve slightly steeper at the low-mass end.

### CMF monotonicity

At fixed mass ($1\, M_\oplus$), increasing the core mass fraction (replacing less dense silicate with denser iron) must monotonically decrease the radius. This is a basic physical constraint: iron ($\rho_0 = 8300$ kg/m$^3$) is roughly twice as dense as MgSiO$_3$ ($\rho_0 = 4100$ kg/m$^3$), so more iron means a more compact planet.

Zalmoxis produces a strictly decreasing curve from $R \approx 1.02\, R_\oplus$ at $f_c = 0.05$ (nearly pure silicate) to $R \approx 0.81\, R_\oplus$ at $f_c = 0.95$ (nearly pure iron), in units of the 6335.4 km Earth radius of `zalmoxis.constants`. The plot shows the solves of the test.

![Radius of a 1 Earth-mass planet against core mass fraction](../assets/validation/zalmoxis_cmf_sweep.svg#only-light){ width="80%" }
![Radius of a 1 Earth-mass planet against core mass fraction](../assets/validation/zalmoxis_cmf_sweep_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The circles joined by the line are the code radii of 1 Earth-mass planets at the core mass fractions 0.1, 0.2, 0.3, 0.5 and 0.7. The test asserts that each radius is smaller than the one before. A pass means that the line falls at every step.

![Radius of a 1 Earth-mass pure iron planet with the Seager et al. (2007) value](../assets/validation/zalmoxis_pure_iron.svg#only-light){ width="80%" }
![Radius of a 1 Earth-mass pure iron planet with the Seager et al. (2007) value](../assets/validation/zalmoxis_pure_iron_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The pure-iron test asserts the code radius of a pure iron planet (circle) against $0.77\,R_\oplus$ (square). The square is the Seager et al. (2007) value, with its error bar of 0.05. A pass means that the circle lies inside the error bar; it lies at $0.80\,R_\oplus$.

A pure iron planet ($f_c = 1$) has $0.80\, R_\oplus$, against the Seager et al. (2007) value of $0.77\, R_\oplus$ for a 1 $M_\oplus$ iron planet; the test accepts `abs=0.05`. The radius falls faster with CMF at high CMF: by 0.043 $R_\oplus$ from $f_c = 0.3$ to 0.5 and by 0.047 $R_\oplus$ from 0.5 to 0.7.

### Conservation checks on full solver output

The smoke tests of `TestConservationLaws` also verify three conservation properties of the full solver output:

- **Mass conservation:** $M(R_\mathrm{surface}) = M_p$ to 1 %, and to 2 % by trapezoidal integration of $4\pi r^2 \rho(r)$ over the solver's own density and radii.
- **Surface pressure:** $P(R_\mathrm{surface}) = P_\mathrm{target}$ (101,325 Pa) to 2 GPa, twice the default surface-pressure tolerance of the solver (1 GPa).
- **Gauss's law:** the integrated $g$ equals $GM/r^2$ to $< 10^{-3}$ relative error at every radius except the first 6 grid points and the points with $P \leq 0$.

![Enclosed mass of the 1 Earth-mass solve over the input mass against radius](../assets/validation/zalmoxis_conservation_mass.svg#only-light){ width="80%" }
![Enclosed mass of the 1 Earth-mass solve over the input mass against radius](../assets/validation/zalmoxis_conservation_mass_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The line is the enclosed mass of the solve over the input mass, to 4 decimals, against radius. The mass test asserts the outermost mass (circle) against the input mass (square, with its error bar of 1 %). A pass means that the circle lies inside the error bar.

![Mass from the trapezoidal integral of the density over the input mass](../assets/validation/zalmoxis_conservation_trapezoid.svg#only-light){ width="80%" }
![Mass from the trapezoidal integral of the density over the input mass](../assets/validation/zalmoxis_conservation_trapezoid_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The trapezoid test integrates $4\pi r^2 \rho$ over the radius with the trapezoidal rule. It asserts the result (circle, to 4 decimals) against the input mass (square, with its error bar of 2 %). A pass means that the circle lies inside the error bar.

![Pressure of the 1 Earth-mass solve against radius, with the surface value and the 2 GPa bound](../assets/validation/zalmoxis_surface_pressure.svg#only-light){ width="80%" }
![Pressure of the 1 Earth-mass solve against radius, with the surface value and the 2 GPa bound](../assets/validation/zalmoxis_surface_pressure_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The line is the pressure of the solve on a log axis. Pressures below 1 GPa are drawn at 1 GPa, so that the plot is the same on every platform. The surface value itself is about $10^5$ Pa. The surface-pressure test asserts that it differs from 101325 Pa by less than 2 GPa (dotted line). A pass means that the surface circle lies below the dotted line.

**Known limit.** The surface-pressure check allows 2 GPa against a target of 101,325 Pa, so it cannot tell a correct surface pressure from one 10000 times too large. The solver treats a residual below 1 GPa as pressure-converged, but it can also accept a solution on mass convergence alone, so this 2 GPa check is then the only guard. The mass checks allow 1 % and 2 %.

**How we know the right answer.** The scaling law and the pure-iron radius of Seager et al. (2007), and physical constraints (monotonicity, mass, surface pressure, Gauss's law) with no published reference.

**Why it matters.** These trends are what a grid of Zalmoxis planets depends on.

??? info "Test details"
    - Reference: `TestMassRadiusScaling`, $\alpha \approx 0.27$ for masses 0.5, 1, 2, 5 and 10 $M_\oplus$, and `TestPureIronPlanet`, $0.77\,R_\oplus$ for CMF = 1.0, both from Seager et al. (2007) [^cite-seager2007]; `TestCMFMonotonicity` (CMF 0.1 to 0.7) and `TestConservationLaws` check physical constraints with no published reference
    - Tolerance: exponent to `abs=0.04` of 0.27; strictly decreasing radius with CMF; pure-iron radius to `abs=0.05` of $0.77\,R_\oplus$; mass within 1% (2% by trapezoidal integration); surface pressure at most 2e9 Pa from the target; Gauss's law `rtol=1e-3`
    - Test id: `tests/test_first_principles.py::TestMassRadiusScaling`, `tests/test_first_principles.py::TestCMFMonotonicity`, `tests/test_first_principles.py::TestPureIronPlanet`, `tests/test_first_principles.py::TestConservationLaws`
    - Command: `pytest -o "addopts=" tests/test_first_principles.py::TestMassRadiusScaling tests/test_first_principles.py::TestCMFMonotonicity tests/test_first_principles.py::TestPureIronPlanet tests/test_first_principles.py::TestConservationLaws`

## A two-layer Earth core against PREM

**Why this check exists.** A core model can meet its own consistency checks and still give a core much denser than the core of the Earth. This case shows how far the core of a 1 $M_\oplus$ planet lies from the seismic Earth model PREM, with and without a two-layer core and light elements.

**What it tests.** The test solves the default 1 $M_\oplus$ planet (core mass fraction 0.325, `PALEOS:MgSiO3` mantle, adiabatic profile from 3000 K at the surface) with 3 cores:

1. `PALEOS:iron`, a one-layer iron core.
2. A two-layer iron core: a solid inner core on `PALEOS-API-2phase:iron-solid` and a liquid outer core on `PALEOS-API-2phase:iron-liquid`.
3. The same two-layer core with the 13.85 wt% S outer core of Hirose et al. (2021) [^cite-hirose2021].

The two-layer cores take the inner-core mass fraction of PREM, 0.0508. At each core node the test compares the density with the PREM density at the same pressure in the same region, inner or outer core. It pins the median and the largest relative difference per region. It also checks the PREM reference against Table II of the paper.

**How we know the right answer.** The PREM of Dziewonski & Anderson (1981) [^cite-dziewonski1981]: the core density polynomials of Table I. The pressure comes from a hydrostatic integral of that density inward from the CMB pressure of Table II, 135.751 GPa. It meets the centre and ICB pressures of Table II to $4.1 \times 10^{-4}$ and $3.9 \times 10^{-4}$ and the gravity at the CMB to $6.5 \times 10^{-4}$. The comparison is at equal pressure, since the radii of the planet and of its CMB differ from those of the Earth. There is no tolerance against PREM: the test pins the differences of the code as regression values. For comparison, Attia et al. (2026) [^cite-attia2026] find the iron core of their PALEOS Earth denser than PREM by a median of 10.6 % and at most 12.1 %, compared by radius (their Fig. 4, with 10 km around each CMB left out).

| Core | Inner core: median | Inner core: largest | Outer core: median | Outer core: largest |
| --- | --- | --- | --- | --- |
| Fe, 1 layer | $+2.02$ % (13 nodes) | $+2.12$ % | $+8.05$ % | $+9.90$ % |
| Fe, 2 layers | $+3.02$ % (6 nodes) | $+3.10$ % | $+8.05$ % | $+9.90$ % |
| Fe-S, 2 layers | $+3.15$ % | $+3.45$ % | $+0.43$ % | $+2.13$ % |

| Core | Radius | CMB radius | CMB pressure | Centre pressure | By radius: median, largest |
| --- | --- | --- | --- | --- | --- |
| PREM | 6371 km | 3480 km | 135.75 GPa | 363.85 GPa | |
| Fe, 1 layer | 6525 km | 3381 km | 135.5 GPa | 386.2 GPa | $+8.52$ %, $+8.66$ % |
| Fe-S, 2 layers | 6562 km | 3477 km | 128.2 GPa | 358.5 GPa | $+0.39$ %, $+3.00$ % |

The second table comes from the same 3 solves, on the side tables of 40 points per decade of the test (the tutorial prints 6561.4 km for the Fe-S planet on the default grid). By radius, each core node is compared with PREM at its radius, as Attia et al. (2026) [^cite-attia2026] do; nodes within 10 km of either CMB are left out. The node counts in the first table are the core nodes in each PREM region. The iron cores reach 386 GPa at the centre, above the 364 GPa of PREM, so only their nodes up to 364 GPa enter the inner-core numbers. The outer cores of the 2 iron cores are the same: on this adiabat the outer core lies above the melting curve of iron, where `PALEOS:iron` is the PALEOS liquid. The S, Si, O and H midpoint cores of Hirose et al. (2021) [^cite-hirose2021] have the same density by the calibration of the linear law ([case](equation_of_state.md#light-element-outer-cores-on-the-liquid-iron-side)), so the S core stands for all 4.

**Why it matters.** The core density sets the core radius and the gravity of the planet. An iron outer core is 8 % denser than the core of the Earth, and its light elements close most of that gap.

![Core density against pressure for PREM and the 3 cores of the default 1 Earth-mass planet](../assets/validation/core_two_phase_prem_pressure.svg#only-light){ width="80%" }
![Core density against pressure for PREM and the 3 cores of the default 1 Earth-mass planet](../assets/validation/core_two_phase_prem_pressure_dark.svg#only-dark){ width="80%" }

![Core density against radius for PREM and the 3 cores of the default 1 Earth-mass planet](../assets/validation/core_two_phase_prem_radius.svg#only-light){ width="80%" }
![Core density against radius for PREM and the 3 cores of the default 1 Earth-mass planet](../assets/validation/core_two_phase_prem_radius_dark.svg#only-dark){ width="80%" }

**How to read the plots.** The first plot gives the core density against pressure, with the centre on the left; the test numbers come from it. The second plot gives the same densities against radius. The dashed line is PREM, with its density step at the ICB. The 2 iron cores are the same in the outer core: the wide line of the one-layer core shows under the line of the two-layer core there, and the Fe-S line lies close to PREM there. The two-layer cores step down at their ICB, from the solid to the liquid side.

**Known limit.** The two-layer cores take the inner-core mass of PREM, not their own ICB. On this adiabat the centre is at 7710 to 8060 K, above the Anzellini melting curve (6403 K at 349 GPa), so the liquidus rule would give no inner core, and the solid inner core lies above the melting curve of iron. The test runs on side tables of 40 points per decade over $10^9$ to $3 \times 10^{12}$ Pa and $10^3$ to $2 \times 10^4$ K, made in a scratch folder, not on the default grid of 150 points per decade. The pins hold to $10^{-3}$, 28 times the largest change of a pinned value of the Fe-S core ($3.6 \times 10^{-5}$) over 5 starts with round-off changes of the planet mass and surface temperature.

??? info "Test details"
    - Reference: the PREM of Dziewonski & Anderson (1981) [^cite-dziewonski1981], Table I (density) and Table II (pressure and gravity checks), `tests/validation/data/prem_core_polynomials.csv` and `prem_core_table2.csv`.
    - Tolerance: PREM against Table II, $10^{-3}$ in pressure and gravity and $10^{-5}$ in density; the medians and largest differences pinned to $10^{-3}$; at least 5 compared nodes per region.
    - Test id: `tests/test_core_two_phase_solve.py::test_an_earth_core_against_prem` (integration tier; needs PALEOS and the PALEOS tables under `data/`)
    - Command: `pytest -o "addopts=" tests/test_core_two_phase_solve.py -k prem`; the plots: `pytest -o "addopts=" tests/test_core_two_phase_solve.py -k prem --validation-plots`

## Reproducing the results

The tests draw the plots of these cases. This command writes them again to `docs/assets/validation/`:

```console
pytest -o "addopts=" tests/test_first_principles.py tests/test_polytrope_verification.py --validation-plots
```

Without `--validation-plots` the tests write nothing: each compares the data of its plot with the committed pair, and a missing or stale file fails the test.

[^cite-attia2026]: M. Attia, T. Lichtenberg, E. Jungová, M. Sastre (2026). *[PALEOS: Multiphase equations of state and mass-radius relations for exoplanet interiors](https://doi.org/10.1051/0004-6361/202660790)*. Astronomy & Astrophysics, 713, A225.

[^cite-dziewonski1981]: A. M. Dziewonski, D. L. Anderson (1981). *[Preliminary reference Earth model](https://doi.org/10.1016/0031-9201(81)90046-7)*. Physics of the Earth and Planetary Interiors, 25, 297-356.

[^cite-hirose2021]: K. Hirose, B. Wood, L. Vočadlo (2021). *[Light elements in the Earth's core](https://doi.org/10.1038/s43017-021-00203-6)*. Nature Reviews Earth & Environment, 2, 645-658.

[^cite-seager2007]: S. Seager, M. Kuchner, C. A. Hier-Majumder, B. Militzer (2007). *[Mass-Radius Relationships for Solid Exoplanets](https://doi.org/10.1086/521346)*. The Astrophysical Journal, 669, 1279-1297.
