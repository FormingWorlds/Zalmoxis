# Equation of state

## What this page checks

This page checks 2 parts of Zalmoxis. The first is the export reader of the PALEOS equation-of-state (EOS) tables; Zalmoxis uses it to write the tables for Aragog and SPIDER. The second is the density of an iron core that holds light elements. An EOS gives the density and other properties of a material as functions of pressure and temperature. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. If the core density is wrong, the structure solve uses the wrong value at every core node. The radius and the interior profiles of the planet are then wrong too. Each case below says why the check exists, what it tests, how we know the right answer and why it matters, and ends with a box "Test details". The [overview](index.md) lists every case of the tab.

## PALEOS table loader

**Why this check exists.** A table reader can put a value in the wrong grid cell, drop a column or mislabel a phase. The EOS tables that Zalmoxis writes for Aragog and SPIDER then hold wrong properties without an error message.

**What it tests.** `zalmoxis.eos_export.load_paleos_all_properties` reads a PALEOS pressure-temperature table into grids over the unique $\log_{10} P$ and $\log_{10} T$ values. It is the export reader: the functions that write the EOS tables for Aragog and SPIDER call it. The structure solve has its own reader, `zalmoxis.eos.interpolation.load_paleos_unified_table`, which this case does not test. The shipped tables are the unified MgSiO$_3$ table and the solid and liquid MgSiO$_3$ sub-tables; the highres tables are larger solid and liquid sub-tables. A table is taken from `$FWL_DATA/zalmoxis_eos` or from the `data/` directory; a table that is in neither place is not tested.

The test reads the same file with `numpy.genfromtxt` and compares the 2 readings. These must be identical:

- the pressure and temperature axes;
- every grid cell of density, internal energy, entropy, $c_p$, $c_v$, thermal expansivity and $\nabla_\mathrm{ad}$, with $c_p$ and $c_v$ the heat capacities at constant pressure and volume and $\nabla_\mathrm{ad}$ the adiabatic gradient;
- every phase label.

Every grid cell that no table row fills must be NaN, with an empty phase label.

**How we know the right answer.** A second reader of the same file, `numpy.genfromtxt`, gives the reference. This is a cross-code check between 2 readers of one file.

**Why it matters.** The EOS tables that Zalmoxis writes for Aragog and SPIDER take their densities and thermal properties from these grids.

This case has no plot: it checks that 2 readers give identical arrays, an identity with nothing to draw.

??? info "Test details"
    - Reference: `numpy.genfromtxt` reading of the same table (cross-code check).
    - Tolerance: identical arrays (`assert_array_equal`), no tolerance.
    - Test id: `tests/test_eos_export_real_tables.py::test_shipped_tables_are_identical_to_the_genfromtxt_reader` (integration tier), `tests/test_eos_export_highres.py::test_highres_tables_are_identical_to_the_genfromtxt_reader` (slow tier)
    - Command: `pytest -o "addopts=" tests/test_eos_export_real_tables.py tests/test_eos_export_highres.py`

## Liquid iron and iron-water members of Luo et al. (2024)

**Why this check exists.** The density of liquid iron comes from a root solve of an equation with many fit constants. A wrong constant or a root solve that stops early gives a wrong core density without an error message.

**What it tests.** `zalmoxis.eos.core_alloy.luo_density` solves Methods Eq. 14 of Luo et al. (2024)[^cite-luo2024] for the cell volume of liquid Fe (64 Fe) and of Fe with water (64 Fe + 8 H$_2$O). The 2 materials are the members of the fit. The test uses 15 pressures from 50 to 1300 GPa at 8000, 11000 and 14000 K, the fit range. It solves the same equation by bisection, with the constants typed again from the paper. Both members must agree with the bisection. At 8000 K, Fe with water must be less dense than liquid Fe at every pressure, and liquid Fe at 14000 K less dense than at 8000 K. Liquid Fe at 136 GPa and 8000 K must lie between 10.3 and 10.5 g/cm$^3$. A second test checks that both members have zero pressure at their reference volume and temperature.

**How we know the right answer.** Eq. 14 of Luo et al. (2024), solved by bisection, gives the reference; this checks the root solve and the transcription of the constants. `test_luo_dry_member_equals_paleos_liquid_iron` compares the dry member with the PALEOS liquid iron table, an independent implementation. They agree to 4e-6 at the liquid points of 60 to 1300 GPa and 8000 to 14000 K.

**Why it matters.** The light-element core takes the excess volume of its water carrier from the difference of these 2 members (`carrier_excess_volume`).

![Density of liquid Fe and Fe with water from Luo et al. (2024) Eq. 14](../assets/validation/core_alloy_luo2024_members.svg#only-light){ width="80%" }
![Density of liquid Fe and Fe with water from Luo et al. (2024) Eq. 14](../assets/validation/core_alloy_luo2024_members_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the pressure in GPa on a log scale, and the y-axis the density in g/cm$^3$. The legend gives one colour for liquid Fe and one for Fe with water. The solid, dashed and dotted lines are 8000, 11000 and 14000 K. Each line is the asserted code density. The bisection agrees to a relative 1e-6, too small to see, so it is not drawn. A pass means that Fe with water lies below liquid Fe at 8000 K, and liquid Fe at 14000 K below liquid Fe at 8000 K.

**Known limit.** The wet member has no external anchor here: the bisection uses the same equation and constants.

??? info "Test details"
    - Reference: Luo et al. (2024), Methods Eq. 14 (analytical).
    - Tolerance: relative 1e-6.
    - Test id: `tests/test_core_alloy.py::TestLuoMembers::test_eq14_density_matches_an_independent_bisection` (unit tier)
    - Command: `pytest -o "addopts=" tests/test_core_alloy.py -k eq14`

## Light-element core on the PALEOS iron table

**Why this check exists.** A core with light elements must be lighter than pure iron, and with no light element it must be pure iron. A wrong mixing rule, or a difference between the 2 code branches, gives a wrong core density without an error message.

**What it tests.** The [light-element core](../Explanations/mixing.md#light-element-core) in `calculate_mixed_density` and its batch twin, on the PALEOS iron table at 8000 K. The test uses 12 pressures from 140 to 360 GPa. It takes a core with no light element, and cores with 0.5 and 1 wt% H as the Fe-H-O carrier (`CoreAlloy:H-Luo`), with the O of the water carrier (7.94 times the H). With every fraction 0, the core density must be the PALEOS iron density, bit for bit. With H and the carrier O, the density must be the harmonic mean of iron and the carrier density $1/\Delta v_c$, with $\Delta v_c$ the excess volume of the carrier per unit mass. A mass-weighted mean must lie more than 1 % away. The scalar branch, which the structure equations use, and the batch branch, which updates the nodes, must agree at every node.

**How we know the right answer.** The limits of the volume-additive mixing rule: no light element gives pure iron, and added volumes give the harmonic mean of the densities.

**Why it matters.** The structure solve uses this density for a core with light elements, and the core density sets the radius of the planet.

![Core density on the PALEOS iron table with 0, 0.5 and 1 wt% H as the carrier](../assets/validation/core_alloy_paleos_iron_limits.svg#only-light){ width="80%" }
![Core density on the PALEOS iron table with 0, 0.5 and 1 wt% H as the carrier](../assets/validation/core_alloy_paleos_iron_limits_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the pressure in GPa, and the y-axis the core density in g/cm$^3$ at 8000 K. The solid line is pure PALEOS iron, and the dashed and dotted lines are 0.5 and 1 wt% H as the carrier (`CoreAlloy:H-Luo`) with its O. Each line is the asserted code density. The harmonic mean agrees to a relative 1e-12, too small to see, so it is not drawn. A pass means that each line meets its asserted limit; more H then gives a lower density at every pressure.

**Known limit.** The test takes the carrier density from the code's own `carrier_excess_volume`, so it checks the mixing rule, not the carrier volume. The only physical anchor is `tests/test_core_alloy_solve.py`: a 1 $M_\oplus$ planet with 1 wt% H as the carrier keeps its core between 0.83 and 0.90 of the iron density.

??? info "Test details"
    - Reference: the volume-additive limits of the mixing rule (analytical).
    - Tolerance: identical arrays for no light element; relative 1e-12 for the harmonic mean; relative 1e-9 between the scalar and batch branches.
    - Test id: `tests/test_core_alloy.py::TestPaleosCore::test_core_mixture_limits_on_paleos_iron` (unit tier; it needs the PALEOS iron table under `data/` and skips without it)
    - Command: `pytest -o "addopts=" tests/test_core_alloy.py -k limits`

## Light-element cores against the outer core of Hirose et al. (2021)

**Why this check exists.** The light elements set how much less dense than iron the core of a planet is, and with it the core radius. A wrong $\beta$ gives a core that is too light or too heavy for its composition, and no error message.

**What it tests.** The linear law of the [light-element core](../Explanations/mixing.md#light-element-core). This is a calibration check: $\beta$ is set to $7/w_\mathrm{mid}$, so the test shows that the code holds the stated calibration, not that the calibration is right. For each of S, Si, O and H (`CoreAlloy:H`), the core at the midpoint of its range must be 7 % less dense than its iron base, to $10^{-12}$ percentage points, on any iron density and on the PALEOS iron table at the core-mantle boundary (136 GPa, 4137 K) and the inner-core boundary (330 GPa, 5400 K) of the estimate of Hirose et al. (2021); the cores at the ends of the range must be 6 to 8 % less dense. The $\beta$ values of Li et al. (2019)[^cite-li2019] must miss the band for S, O and H. The test also pins the deficit of the Luo H law (`CoreAlloy:H-Luo`) for 0.96 and 0.99 wt% H with no O, to $10^{-3}$ percentage points (regression values, which depend on the version of the PALEOS table), on the PALEOS table and on the Luo liquid iron at 8000 K, and asserts that each is above 8 %.

**How we know the right answer.** Hirose et al. (2021)[^cite-hirose2021], p. 3: an outer core with only S, Si, O or H fits the seismic outer core with 13.3-14.4 wt% S, 9.3-9.8 wt% Si, 8.4-8.9 wt% O or 0.96-0.99 wt% H, and is 6 to 8 % less dense than pure liquid iron.

**Why it matters.** The structure solve takes the core density from this law, so the core radius of a planet with light elements follows it.

![Density deficit of the single-element outer cores of Hirose et al. 2021 against the PALEOS iron table](../assets/validation/core_alloy_hirose2021_deficit.svg#only-light){ width="80%" }
![Density deficit of the single-element outer cores of Hirose et al. 2021 against the PALEOS iron table](../assets/validation/core_alloy_hirose2021_deficit_dark.svg#only-dark){ width="80%" }

**How to read the plot.** Each column is one light element; the y-axis is the density deficit against the PALEOS iron table in percent. Each vertical segment runs from the deficit at the low end of the range of Hirose et al. (2021) to that at the high end; for these 4 the deficit is the same at both boundaries, since the linear law does not depend on pressure and temperature. Filled round markers are the calibrated $\beta$ (the column `H` is the default `CoreAlloy:H` law), open markers the $\beta$ of Li et al. (2019). The grey band is the 6 to 8 % of Hirose et al. (2021). The last 2 columns, `H-Luo CMB` and `H-Luo ICB`, are their own series with square markers: the Luo H law (`CoreAlloy:H-Luo`) of Luo et al. (2024) at the core-mantle and inner-core boundaries, on the PALEOS table, which is solid at those temperatures. A pass means that every filled S, Si, O and H segment lies inside the band, with 7 % at its midpoint.

**Known limits.**

- On a solid PALEOS base the calibrated core is less than 7 % below liquid iron. The PALEOS iron table is solid at both states (the test asserts the phase), and there the calibrated core is 4.9 % (136 GPa) and 5.8 % (330 GPa) less dense than the liquid iron of Luo et al. (2024, Eq. 14), extrapolated below its 8000 K fit range, as the solid base is 2.3 and 1.2 % denser (the test pins these ratios); at 600 to 1000 GPa and 8000 K, where the table is also solid, it is 4.6 to 5.6 % less dense. A two-layer core reads its outer core from the liquid side instead ([next case](#light-element-outer-cores-on-the-liquid-iron-side)).
- The 2 sources of the H laws disagree on the pressure dependence. Hirose et al. (2021) give a density deficit of the outer core that is uniform with depth, over 136 to 330 GPa; the Luo H law falls over the same range from 13.2 to 11.0 % for 0.975 wt% H (on the Luo liquid iron at 8000 K), and to 6.6 % at 1300 GPa. The default H law follows Hirose et al. (2021), and its constant $\beta$ above 330 GPa is an extrapolation.
- The core takes the solid or liquid phase of pure Fe from the PALEOS table, so it changes phase at the melting curve of pure Fe (between 4220 and 4230 K at 136 GPa, and between 6280 and 6290 K at 330 GPa), not at the alloy liquidus. A core above its alloy liquidus but below the melting temperature of pure Fe is computed on the solid branch. A two-layer core takes its phase from the inner-core boundary instead ([two-layer core](../Explanations/mixing.md#two-layer-core)).
- C and N are not calibrated against the outer core. The Luo H law (`CoreAlloy:H-Luo`) gives 11 to 15 % for the H core of Hirose et al. (2021), above their range.

??? info "Test details"
    - Reference: Hirose et al. (2021), p. 3.
    - Tolerance: $10^{-12}$ percentage points at the midpoints; 6 to 8 % at the ends; $10^{-3}$ percentage points for the Luo H law.
    - Test id: `tests/test_core_alloy.py::TestHiroseCalibration`, its plot from `tests/test_core_alloy.py::TestHiroseCalibration::test_each_single_element_outer_core_is_7_percent_lighter_than_iron` (unit tier; the tests on the PALEOS iron table need it under `data/` and skip without it; `::test_the_calibration_gives_7_percent_on_any_iron_base` needs no table)
    - Command: `pytest -o "addopts=" tests/test_core_alloy.py -k Hirose`

## Light-element outer cores on the liquid iron side

**Why this check exists.** A two-layer core reads its outer core from the liquid side table of PALEOS iron (`PALEOS-API-2phase:iron-liquid`). A wrong side table, a wrong read of it or a wrong light-element law on it gives a wrong outer-core density, and no error message.

**What it tests.** The test takes the 2 states of the case above: the core-mantle boundary (136 GPa, 4137 K) and the inner-core boundary (330 GPa, 5400 K) of Hirose et al. (2021)[^cite-hirose2021]. At each state it makes the liquid side table on the 4 by 4 nodes of the default side grid around the state and reads it through the `PALEOS-API-2phase:iron-liquid` entry. The table density must equal the PALEOS liquid of Luo et al. (2024)[^cite-luo2024] at the state, within the bilinear interpolation bound of its nodes. Each single-element outer core at the midpoint of its range must be 7 % less dense than the liquid side, to $10^{-12}$ percentage points, and the cores at the ends of the range 6 to 8 %. On the solid base `PALEOS:iron` the deficit of the midpoint cores against the PALEOS liquid is pinned.

**How we know the right answer.** This is a consistency check, not an independent validation. $\beta$ is set to $7/w_\mathrm{mid}$ ([light-element core](../Explanations/mixing.md#light-element-core)), so 7 % holds on any iron base by construction. The case shows that the liquid side table is the PALEOS liquid and that the light-element law acts on it. The bound of the bilinear interpolation in $\log_{10} P$ and $\log_{10} T$ is $(\max|\Delta^2_P \rho| + \max|\Delta^2_T \rho|)/8$, from the second differences of the node densities, plus $10^{-4}$ kg/m$^3$ for the 9 digits of the table.

| State | Liquid side table | PALEOS liquid | Difference | Bound |
| --- | --- | --- | --- | --- |
| 136 GPa, 4137 K | 11058.173 kg/m$^3$ | 11058.187 kg/m$^3$ | $-0.014$ kg/m$^3$ | 0.050 kg/m$^3$ |
| 330 GPa, 5400 K | 13300.153 kg/m$^3$ | 13300.134 kg/m$^3$ | $+0.019$ kg/m$^3$ | 0.058 kg/m$^3$ |

| Midpoint cores | 136 GPa | 330 GPa |
| --- | --- | --- |
| On the liquid side, below the liquid side | 7 % | 7 % |
| On the solid base, below the PALEOS liquid | 4.890 % | 5.845 % |

**Why it matters.** In a two-layer core the outer core is liquid. On the solid base the outer cores of Hirose et al. (2021)[^cite-hirose2021] would be 4.9 to 5.8 % less dense than liquid iron, not 6 to 8 %, so the outer core would be too dense and the core radius too small.

![Density deficit of the single-element outer cores of Hirose et al. 2021 on the liquid iron side and on the solid PALEOS base](../assets/validation/core_two_phase_hirose_liquid_side.svg#only-light){ width="80%" }
![Density deficit of the single-element outer cores of Hirose et al. 2021 on the liquid iron side and on the solid PALEOS base](../assets/validation/core_two_phase_hirose_liquid_side_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis gives the light element at the core-mantle boundary (CMB, left) and at the inner-core boundary (ICB, right). The y-axis is the density deficit against liquid iron in percent. Each solid segment runs from the low to the high end of the range of Hirose et al. (2021)[^cite-hirose2021] on the liquid side. Each dashed segment is the same range on the solid base, against the PALEOS liquid. The grey band is the 6 to 8 % of Hirose et al. (2021)[^cite-hirose2021]. A pass means that every solid segment lies inside the band, with 7 % at its midpoint.

**Known limit.** Both states lie below the PALEOS melting curve of pure iron (case above), so the liquid side there is the metastable extension of the PALEOS liquid. The test makes the table on 16 nodes around each state, not on the full default grid; the nodes are those of the default grid, so the values are those of the cached table of the same PALEOS version.

??? info "Test details"
    - Reference: Hirose et al. (2021)[^cite-hirose2021], p. 3, and the PALEOS liquid of Luo et al. (2024)[^cite-luo2024] (consistency check).
    - Tolerance: the bilinear bound plus $10^{-4}$ kg/m$^3$, below $10^{-5}$ of the density; $10^{-12}$ percentage points at the midpoints; 6 to 8 % at the ends; $10^{-3}$ percentage points for the pinned solid-base deficits.
    - Test id: `tests/test_core_alloy.py::TestHiroseCalibration::test_the_outer_cores_on_the_liquid_iron_side` (unit tier; needs PALEOS and the PALEOS iron table under `data/`, skips without them)
    - Command: `pytest -o "addopts=" tests/test_core_alloy.py -k liquid_iron_side`; the plot: `pytest -o "addopts=" tests/test_core_alloy.py -k liquid_iron_side --validation-plots`

## Related tests

The other tests in `tests/test_core_alloy.py` pin the linear law at 1 wt% against the $\beta$ values of the [mixing page](../Explanations/mixing.md#light-element-core), the effective $\beta_\mathrm{H}$ of the Fe-H-O carrier, its continuity where the O rest changes sign and across the faces of the Luo fit box, and the composition and density bounds, the warning where S, O and C together need more Fe than the core holds in the units of the alloy liquidus (Fe$_3$S, FeO, Fe$_3$C, Fe$_7$C$_3$), and, on the PALEOS iron table, that the density falls with H for Si-bearing cores down to the linear-factor bound, up to 16000 K at every pressure and up to 59000 K above 20 GPa, and that without the bound it can rise. `tests/test_core_alloy_solve.py` (integration tier) solves a 1 $M_\oplus$ planet with 1 wt% H as the carrier (`CoreAlloy:H-Luo`) in the core, whose first trials sample the outer core below the Luo fit range; it converges, passes the accepted-structure checks and keeps the core 10 to 17 % lighter than iron at every core node. The same planet with 1 wt% of the default `CoreAlloy:H` has a core of $1 - 0.07/0.975$ of the iron density at every core node, which the Luo law misses by up to 8.2 %.

[^cite-hirose2021]: K. Hirose, B. Wood, L. Vočadlo, *[Light elements in the Earth's core](https://doi.org/10.1038/s43017-021-00203-6)*, Nature Reviews Earth & Environment, 2, 645-658, 2021. [SciX](https://scixplorer.org/abs/2021NRvEE...2..645H/abstract).
[^cite-luo2024]: H. Luo, C. Dorn, J. Deng, *[The interior as the dominant water reservoir in super-Earths and sub-Neptunes](https://doi.org/10.1038/s41550-024-02347-z)*, Nature Astronomy, 8, 1399-1407, 2024. [SciX](https://scixplorer.org/abs/2024NatAs...8.1399L/abstract).
[^cite-li2019]: J. Li, B. Chen, M. Mookherjee, G. Morard, *[Carbon versus Other Light Elements in Earth's Core](https://doi.org/10.1017/9781108677950.003)*, in Deep Carbon, eds. B. N. Orcutt, I. Daniel, R. Dasgupta, Cambridge University Press, 40-65, 2019.
