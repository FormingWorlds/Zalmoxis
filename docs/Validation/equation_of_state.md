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
