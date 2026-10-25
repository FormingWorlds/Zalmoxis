# Equation of state

## What this page checks

This page checks how Zalmoxis gets the density of each layer of a planet. An equation of state (EOS) gives the density of a material at a given pressure and temperature. By default Zalmoxis reads it from PALEOS tables; for a core with light elements such as H, O or Si, mixing laws lower the iron density. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. The answer comes from a second reader of the same file, a published paper, another implementation, or a limit of the mixing rule. A wrong density here gives a wrong radius and pressure profile for the planet. Each core case below has the same parts and ends with a box "Test details" (reference, tolerance, test id, command). The [overview](index.md) lists every case of the tab.

## PALEOS table loader

**What it tests:** `zalmoxis.eos_export.load_paleos_all_properties` reads a PALEOS pressure-temperature table into grids over the unique $\log_{10} P$ and $\log_{10} T$ values. The test reads the same file with `numpy.genfromtxt` and compares the two readings, so this is a cross-code check between 2 readers of one file.

**Setup:** The shipped tables are the unified MgSiO$_3$ table and the solid and liquid MgSiO$_3$ sub-tables; the highres tables are larger solid and liquid sub-tables. A table is taken from `$FWL_DATA/zalmoxis_eos` or from the `data/` directory; a table that is in neither place is not tested.

**Result asserted:** The pressure and temperature axes, every grid cell of density, internal energy, entropy, $c_p$, $c_v$, thermal expansivity and $\nabla_\mathrm{ad}$, and every phase label are identical to the `genfromtxt` reading. Every grid cell that no table row fills is NaN, with an empty phase label.

- Reference: `numpy.genfromtxt` reading of the same table (cross-code check).
- Tolerance: identical arrays (`assert_array_equal`), no tolerance.
- Test id: `tests/test_eos_export_real_tables.py::test_shipped_tables_are_identical_to_the_genfromtxt_reader` (integration tier), `tests/test_eos_export_highres.py::test_highres_tables_are_identical_to_the_genfromtxt_reader` (slow tier)
- Command: `pytest -o "addopts=" tests/test_eos_export_real_tables.py tests/test_eos_export_highres.py`

## Liquid iron and iron-water members of Luo et al. (2024)

**Why this check exists.** A core with hydrogen uses 2 equations of state from [Luo et al. (2024)](https://doi.org/10.1038/s41550-024-02347-z): one for liquid iron and one for liquid iron with water. The code finds the density from each one by a root solve. A wrong constant or a root solve that stops too early can give a wrong density without an error message.

**What it tests.** `zalmoxis.eos.core_alloy.luo_density` solves Methods Eq. 14 of Luo et al. (2024) for the cell volume of 2 members. A member is one composition of the simulations in the paper: liquid Fe (a cell of 64 Fe atoms) and Fe with water (64 Fe + 8 H$_2$O). The test uses 15 pressures from 50 to 1300 GPa at 8000, 11000 and 14000 K, the range of the fit in the paper. The code must agree with a second solve of the same equation. At 8000 K, water must lower the density at every pressure. Heating liquid Fe from 8000 to 14000 K must lower its density at every pressure. Liquid Fe at 136 GPa and 8000 K must lie between 10.3 and 10.5 g/cm$^3$. A second test checks that both members have zero pressure at their reference volume and temperature.

**How we know the right answer.** The test solves Eq. 14 again by bisection, with the constants typed again from the paper. This checks the root solve and the copy of the constants. A third test, `test_luo_dry_member_equals_paleos_liquid_iron`, compares the dry member with the PALEOS liquid iron table, an independent implementation. At the liquid points of 60 to 1300 GPa and 8000 to 14000 K, the two agree to 4 parts in a million (4e-6); the test asserts better than 1e-5.

**Why it matters.** The difference between the 2 members gives the volume that water adds to liquid iron. That volume sets how much lighter than iron a core with hydrogen is (see the next case and the [mixing page](../Explanations/mixing.md#light-element-core)).

![Density of liquid Fe and Fe with water from Luo et al. (2024) Eq. 14](../assets/validation/core_alloy_luo2024_members.svg#only-light){ width="80%" }
![Density of liquid Fe and Fe with water from Luo et al. (2024) Eq. 14](../assets/validation/core_alloy_luo2024_members_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the pressure on a log scale and the y-axis is the density. The legend gives the colour of each member and the line style of each temperature. At each temperature the Fe + water line lies below the Fe line, and a hotter line lies below a cooler one. The plot shows the densities of the code, not the comparison; the test checks the agreement with the bisection.

**Known limit.** The wet member has no external anchor here: only the dry member is compared with a second code.

??? info "Test details"
    - Reference: Luo et al. (2024), Methods Eq. 14 (analytical).
    - Tolerance: relative 1e-6.
    - Test id: `tests/test_core_alloy.py::TestLuoMembers::test_eq14_density_matches_an_independent_bisection` (unit tier)
    - Command: `pytest -o "addopts=" tests/test_core_alloy.py -k eq14`

## Light-element core on the PALEOS iron table

**Why this check exists.** A core with light elements uses its own branch of the mixing code. If that branch is wrong, a core with no light element can differ from pure iron, or the hydrogen can mix by the wrong rule. Zalmoxis computes the density in 2 places: one point at a time in the structure equations, and all nodes at once in the node update. The two must give the same density.

**What it tests.** The [light-element core](../Explanations/mixing.md#light-element-core) in `calculate_mixed_density`, which gives the density at one point, and `calculate_mixed_density_batch`, which gives it at all nodes at once. The test uses the PALEOS iron table at 8000 K and 12 pressures from 140 to 360 GPa. The cores have no light element, 0.5 wt% H and 1 wt% H. The H enters as water: each kg of H comes with the O of the water, 7.94 kg. This water is the carrier, with mass fraction $w_c$. $\Delta v_c$ is the volume that 1 kg of carrier adds to liquid iron. The test expects:

- With every fraction 0, the core density is identical to the PALEOS iron density.
- With H and the carrier O, the density is the harmonic mean of iron and the carrier density $1/\Delta v_c$. A mass-weighted mean lies more than 1 % away, so the test can tell the 2 rules apart.
- The one-point function, which the structure equations use, and the all-nodes function, which updates the nodes, agree at every node.

**How we know the right answer.** These are limits of the mixing rule. With no light element, nothing changes the iron. With only the carrier, the volumes of iron and carrier add, and added volumes give the harmonic mean of the densities.

**Why it matters.** This branch sets the core density of every run with a light-element core, and with it the core radius and the planet radius.

![Core density on the PALEOS iron table with 0, 0.5 and 1 wt% H](../assets/validation/core_alloy_paleos_iron_limits.svg#only-light){ width="80%" }
![Core density on the PALEOS iron table with 0, 0.5 and 1 wt% H](../assets/validation/core_alloy_paleos_iron_limits_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the pressure and the y-axis is the core density, at 8000 K. The solid line has no H and is the PALEOS iron density; the dashed and dotted lines have 0.5 and 1 wt% H with the O of the carrier. More H gives a lighter core. The plot shows the densities of the code, not the comparison; the test checks the agreement with the harmonic mean.

??? info "Test details"
    - Reference: the volume-additive limits of the mixing rule (analytical).
    - Tolerance: identical arrays for no light element; relative 1e-12 for the harmonic mean; relative 1e-9 between the scalar and batch branches.
    - Test id: `tests/test_core_alloy.py::TestPaleosCore::test_core_mixture_limits_on_paleos_iron` (unit tier; it needs the PALEOS iron table under `data/` and skips without it)
    - Command: `pytest -o "addopts=" tests/test_core_alloy.py -k limits`

Other tests of the light-element core:

- `tests/test_core_alloy.py` pins the linear law at 1 wt% against [Li et al. (2019)](https://doi.org/10.1017/9781108677950.003) Table 3.3.
- The same file pins the effective $\beta_\mathrm{H}$ of the Fe-H-O carrier and the composition and density bounds. It also pins that the density is continuous across the faces of the Luo fit box and where the O rest changes sign. The O rest is the core O left after the carrier takes its share.
- `tests/test_core_alloy_solve.py` (integration tier) solves a 1 $M_\oplus$ planet with 1 wt% H in the core. Its first trials sample the outer core below the Luo fit range. The solve converges, passes the accepted-structure checks and keeps the core 8 to 15 % lighter than iron at every core node.
