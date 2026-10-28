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

**What it tests.** `zalmoxis.eos.core_alloy.luo_density` solves Methods Eq. 14 of [Luo et al. (2024)](https://doi.org/10.1038/s41550-024-02347-z) for the cell volume of liquid Fe (64 Fe) and of Fe with water (64 Fe + 8 H$_2$O). The 2 materials are the members of the fit. The test uses 15 pressures from 50 to 1300 GPa at 8000, 11000 and 14000 K, the fit range. It solves the same equation by bisection, with the constants typed again from the paper. Both members must agree with the bisection. At 8000 K, Fe with water must be less dense than liquid Fe at every pressure, and liquid Fe at 14000 K less dense than at 8000 K. Liquid Fe at 136 GPa and 8000 K must lie between 10.3 and 10.5 g/cm$^3$. A second test checks that both members have zero pressure at their reference volume and temperature.

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

**What it tests.** The [light-element core](../Explanations/mixing.md#light-element-core) in `calculate_mixed_density` and its batch twin, on the PALEOS iron table at 8000 K. The test uses 12 pressures from 140 to 360 GPa. It takes a core with no light element, and cores with 0.5 and 1 wt% H with the O of the water carrier (7.94 times the H). With every fraction 0, the core density must be the PALEOS iron density, bit for bit. With H and the carrier O, the density must be the harmonic mean of iron and the carrier density $1/\Delta v_c$, with $\Delta v_c$ the excess volume of the carrier per unit mass. A mass-weighted mean must lie more than 1 % away. The scalar branch, which the structure equations use, and the batch branch, which updates the nodes, must agree at every node.

**How we know the right answer.** The limits of the volume-additive mixing rule: no light element gives pure iron, and added volumes give the harmonic mean of the densities.

**Why it matters.** The structure solve uses this density for a core with light elements, and the core density sets the radius of the planet.

![Core density on the PALEOS iron table with 0, 0.5 and 1 wt% H](../assets/validation/core_alloy_paleos_iron_limits.svg#only-light){ width="80%" }
![Core density on the PALEOS iron table with 0, 0.5 and 1 wt% H](../assets/validation/core_alloy_paleos_iron_limits_dark.svg#only-dark){ width="80%" }

**How to read the plot.** The x-axis is the pressure in GPa, and the y-axis the core density in g/cm$^3$ at 8000 K. The solid line is pure PALEOS iron, and the dashed and dotted lines are 0.5 and 1 wt% H with the carrier O. Each line is the asserted code density. The harmonic mean agrees to a relative 1e-12, too small to see, so it is not drawn. A pass means that each line meets its asserted limit; more H then gives a lower density at every pressure.

**Known limit.** The test takes the carrier density from the code's own `carrier_excess_volume`, so it checks the mixing rule, not the carrier volume. The only physical anchor is `tests/test_core_alloy_solve.py`: a 1 $M_\oplus$ planet with 1 wt% H keeps its core between 0.85 and 0.92 of the iron density.

??? info "Test details"
    - Reference: the volume-additive limits of the mixing rule (analytical).
    - Tolerance: identical arrays for no light element; relative 1e-12 for the harmonic mean; relative 1e-9 between the scalar and batch branches.
    - Test id: `tests/test_core_alloy.py::TestPaleosCore::test_core_mixture_limits_on_paleos_iron` (unit tier; it needs the PALEOS iron table under `data/` and skips without it)
    - Command: `pytest -o "addopts=" tests/test_core_alloy.py -k limits`

## Related tests

The other tests in `tests/test_core_alloy.py` pin the linear law at 1 wt% against [Li et al. (2019)](https://doi.org/10.1017/9781108677950.003) Table 3.3, the effective $\beta_\mathrm{H}$ of the Fe-H-O carrier, its continuity where the O rest changes sign and across the faces of the Luo fit box, and the composition and density bounds. `tests/test_core_alloy_solve.py` (integration tier) solves a 1 $M_\oplus$ planet with 1 wt% H in the core, whose first trials sample the outer core below the Luo fit range; it converges, passes the accepted-structure checks and keeps the core 8 to 15 % lighter than iron at every core node.
