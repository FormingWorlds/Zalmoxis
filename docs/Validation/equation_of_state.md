# Equation of state

These cases check how Zalmoxis reads the equation-of-state tables. The [overview](index.md) lists every case of the tab.

## PALEOS table loader

**What it tests:** `zalmoxis.eos_export.load_paleos_all_properties` reads a PALEOS pressure-temperature table into grids over the unique $\log_{10} P$ and $\log_{10} T$ values. The test reads the same file with `numpy.genfromtxt` and compares the two readings, so this is a cross-code check between 2 readers of one file.

**Setup:** The shipped tables are the unified MgSiO$_3$ table and the solid and liquid MgSiO$_3$ sub-tables; the highres tables are larger solid and liquid sub-tables. A table is taken from `$FWL_DATA/zalmoxis_eos` or from the `data/` directory; a table that is in neither place is not tested.

**Result asserted:** The pressure and temperature axes, every grid cell of density, internal energy, entropy, $c_p$, $c_v$, thermal expansivity and $\nabla_\mathrm{ad}$, and every phase label are identical to the `genfromtxt` reading. Every grid cell that no table row fills is NaN, with an empty phase label.

- Reference: `numpy.genfromtxt` reading of the same table (cross-code check).
- Tolerance: identical arrays (`assert_array_equal`), no tolerance.
- Test id: `tests/test_eos_export_real_tables.py::test_shipped_tables_are_identical_to_the_genfromtxt_reader` (integration tier), `tests/test_eos_export_highres.py::test_highres_tables_are_identical_to_the_genfromtxt_reader` (slow tier)
- Command: `pytest -o "addopts=" tests/test_eos_export_real_tables.py tests/test_eos_export_highres.py`

## Liquid iron and iron-water members of Luo et al. (2024)

**What it tests:** `zalmoxis.eos.core_alloy.luo_density` solves Methods Eq. 14 of [Luo et al. (2024)](https://doi.org/10.1038/s41550-024-02347-z) for the cell volume of liquid Fe (64 Fe) and of Fe with water (64 Fe + 8 H$_2$O). The test solves the same equation by bisection with constants typed again from the paper, so this is a check of the root solve and of the constants against an independent reading.

**Setup:** 15 pressures from 50 to 1300 GPa and 8000, 11000 and 14000 K, the fit range.

**Result asserted:** Both members agree with the bisection; water lowers the density at every point and heating lowers it at fixed pressure; liquid Fe at 136 GPa and 8000 K lies between 10.3 and 10.5 g/cm$^3$. A second test checks that both members have zero pressure at their reference volume and temperature.

- Reference: Luo et al. (2024), Methods Eq. 14 (analytical).
- Tolerance: relative 1e-6.
- Test id: `tests/test_core_alloy.py::TestLuoMembers::test_eq14_density_matches_an_independent_bisection` (unit tier)
- Command: `pytest -o "addopts=" tests/test_core_alloy.py -k eq14`

![Density of liquid Fe and Fe with water from Luo et al. (2024) Eq. 14](../assets/validation/core_alloy_luo2024_members.svg#only-light){ width="80%" }
![Density of liquid Fe and Fe with water from Luo et al. (2024) Eq. 14](../assets/validation/core_alloy_luo2024_members_dark.svg#only-dark){ width="80%" }

## Light-element core on the PALEOS iron table

**What it tests:** the [light-element core](../Explanations/mixing.md#light-element-core) in `calculate_mixed_density` and its batch twin, on the PALEOS iron table at 8000 K.

**Setup:** 12 pressures from 140 to 360 GPa; a core with no light element, and cores with 0.5 and 1 wt% H with the O of the water carrier (7.94 times the H).

**Result asserted:** With every fraction 0 the core density is the PALEOS iron density, bit for bit. With H and the carrier O the density is the harmonic mean of iron and the carrier density $1/\Delta v_c$, and a mass-weighted mean lies more than 1 % away. The scalar branch, which the structure equations use, and the batch branch, which updates the nodes, agree at every node.

- Reference: the volume-additive limits of the mixing rule (analytical).
- Tolerance: identical arrays for no light element; relative 1e-12 for the harmonic mean; relative 1e-9 between the scalar and batch branches.
- Test id: `tests/test_core_alloy.py::TestPaleosCore::test_core_mixture_limits_on_paleos_iron` (unit tier)
- Command: `pytest -o "addopts=" tests/test_core_alloy.py -k limits`

![Core density on the PALEOS iron table with 0, 0.5 and 1 wt% H](../assets/validation/core_alloy_paleos_iron_limits.svg#only-light){ width="80%" }
![Core density on the PALEOS iron table with 0, 0.5 and 1 wt% H](../assets/validation/core_alloy_paleos_iron_limits_dark.svg#only-dark){ width="80%" }

The other tests in `tests/test_core_alloy.py` pin the linear law at 1 wt% against [Li et al. (2019)](https://doi.org/10.1017/9781108677950.003) Table 3.3, the effective $\beta_\mathrm{H}$ of the Fe-H-O carrier, its continuity where the O rest changes sign and across the faces of the Luo fit box, and the composition and density bounds. `tests/test_core_alloy_solve.py` (integration tier) solves a 1 $M_\oplus$ planet with 1 wt% H in the core, whose first trials sample the outer core below the Luo fit range; it converges, passes the accepted-structure checks and keeps the core 8 to 15 % lighter than iron at every core node.
