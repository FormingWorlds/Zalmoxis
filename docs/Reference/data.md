# Reference Data

## Download instructions

All tabulated data files are downloaded automatically by running the setup script from the repository root:

```bash
bash tools/setup/get_zalmoxis.sh
```

This invokes `tools/setup/setup_zalmoxis.py`, which fetches the required data through fwl-io into `$FWL_DATA` and links each folder of the `data/` directory to its dataset. Set `FWL_DATA` first.
See Installation Step 4 of the [installation guide](../How-to/installation.md) for details.

Zalmoxis reads a PALEOS table only when its digest is the one the fwl-io registry pins for its file name and, for a link, also for the name of the file the link points to; names are compared case-insensitively, and a pin that is not an md5 stops the run and names the pin (`zalmoxis.eos.registry_guard`). A table that differs in its version directory under the current `FWL_DATA` is fetched again; a table anywhere else that differs, for example an older copy in a real `data/` folder or under an earlier `FWL_DATA`, stops the run with `StaleTableError`, which names the file, both digests and the refresh command; a table that cannot be read, or a registry that is broken or pins one name twice with different digests, stops it the same way. Without an importable fwl-io no table can be checked, since the manifest says which names are pinned, so every checked read stops with `MissingFwlIoError`, a `StaleTableError` that names the file and the `pip install` of fwl-io. Every Zalmoxis reader of the tables, the export of the SPIDER and Aragog tables included, passes the error on. A table of your own must therefore not carry the file name of a registry table: give it another name and point the EOS entry at it. The binary cache of a unified table `x.dat` is `x.cache.pkl` beside it; it holds the md5 of the table it was built from and the identity of the code that built it (the loader source taken at import, or its compiled code when no source is installed, and the numpy and scipy versions), is written in one step and not at all when the table changes while it is read, and is used only when its bytes match their digest and both identities match; any other `.pkl` file beside a table is not read. Beside a table that cannot be read, a cache whose md5 is the pinned one is used.

## Data inventory

The table below lists every data file used by Zalmoxis, organised by subdirectory.

### Equation of state tables

| File | Location | Format | Source | Description |
|------|----------|--------|--------|-------------|
| `eos_seager07_iron.txt` | `data/EOS_Seager2007/` | CSV (rho in g/cm^3, P in GPa) | Seager et al. (2007)[^cite-seager2007] | Fe epsilon pressure-density EOS |
| `eos_seager07_silicate.txt` | `data/EOS_Seager2007/` | CSV (rho in g/cm^3, P in GPa) | Seager et al. (2007)[^cite-seager2007] | MgSiO3 perovskite pressure-density EOS |
| `eos_seager07_water.txt` | `data/EOS_Seager2007/` | CSV (rho in g/cm^3, P in GPa) | Seager et al. (2007)[^cite-seager2007] | Water ice (phases VII, VIII, X) pressure-density EOS |
| `density_melt.dat` | `data/EOS_WolfBower2018_1TPa/` | TSV (P in Pa, T in K, rho in kg/m^3) | Wolf & Bower (2018)[^cite-wolf2018] | MgSiO3 melt density EOS (P: 0--1 TPa, T: 0--16500 K). Out-of-bounds pressures are clamped to the table edge, and a temperature outside 0--16500 K raises an error; the fill cells outside each pressure's valid T band take the band-edge value. |
| `density_solid.dat` | `data/EOS_WolfBower2018_1TPa/` | TSV (P in Pa, T in K, rho in kg/m^3) | Wolf & Bower (2018)[^cite-wolf2018] | MgSiO3 solid (bridgmanite) density EOS, derived from Mosenfelder et al. (2009)[^cite-mosenfelder2009] (P: 0--1 TPa, T: 0--16500 K). Out-of-bounds pressures are clamped to the table edge, and a temperature outside 0--16500 K raises an error; the fill cells outside each pressure's valid T band take the band-edge value. |
| `adiabat_temp_grad_melt.dat` | `data/EOS_WolfBower2018_1TPa/` | TSV (P in Pa, T in K, dT/dP in K/Pa) | Wolf & Bower (2018)[^cite-wolf2018] | MgSiO3 melt adiabatic temperature gradient $(dT/dP)_S$ (P: 0--1 TPa, T: 0--16500 K). Used for `"adiabatic"` temperature mode; its fill cells take the band-edge value, as in the density tables. |
| `density_melt.dat` | `data/EOS_RTPress_melt_100TPa/` | TSV (P in Pa, T in K, rho in kg/m^3) | Extended RTpress melt table | MgSiO3 melt density EOS extended to 100 TPa (P: 1e3--1e14 Pa, T: 400--50000 K). Used by `RTPress100TPa:MgSiO3`. Solid phase uses the WolfBower2018 table above, with its band-edge fill. |
| `adiabat_temp_grad_melt.dat` | `data/EOS_RTPress_melt_100TPa/` | TSV (P in Pa, T in K, dT/dP in K/Pa) | Extended RTpress melt table | MgSiO3 melt adiabatic temperature gradient $(dT/dP)_S$ extended to 100 TPa (P: 1e3--1e14 Pa, T: 400--50000 K). Used for `"adiabatic"` temperature mode with `RTPress100TPa:MgSiO3`. |
| `paleos_mgsio3_tables_pt_proteus_solid.dat` | `data/EOS_PALEOS_MgSiO3/` | Space-separated, 10 columns (P, T, rho, u, s, cp, cv, alpha, nabla_ad, phase_id) | Attia et al. (2026)[^cite-attia2026]; PALEOS (Zenodo 19680050, v1.1.0) | MgSiO3 solid-phase EOS with adiabatic gradient (P: 1 bar--100 TPa, T: 300 K--max(T_liquidus), 150 ppd log-uniform grid, 75% filled). |
| `paleos_mgsio3_tables_pt_proteus_liquid.dat` | `data/EOS_PALEOS_MgSiO3/` | Space-separated, 10 columns (same as solid) | Attia et al. (2026)[^cite-attia2026]; PALEOS (Zenodo 19680050, v1.1.0) | MgSiO3 liquid-phase EOS with adiabatic gradient (P: 1 bar--100 TPa, T: min(T_solidus)--100000 K, 150 ppd log-uniform grid, 53% filled). |
| `paleos_iron_eos_table_pt.dat` | `data/EOS_PALEOS_iron/` | Space-separated, 10 columns (P, T, rho, u, s, cp, cv, alpha, nabla_ad, phase) | Attia et al. (2026)[^cite-attia2026]; PALEOS (Zenodo 22776069, v1.3.0) | Unified Fe EOS, 5 stable phases (alpha-bcc, delta-bcc, gamma-fcc, epsilon-hcp, liquid). P: 1 bar--100 TPa, T: 300--100000 K. 75.3 MB. |
| `paleos_mgsio3_eos_table_pt.dat` | `data/EOS_PALEOS_MgSiO3_unified/` | Space-separated, 10 columns (same as iron) | Attia et al. (2026)[^cite-attia2026]; PALEOS (Zenodo 22776069, v1.3.0) | Unified MgSiO3 EOS, 6 stable phases (3 pyroxene, bridgmanite, postperovskite, liquid). P: 1 bar--100 TPa, T: 300--100000 K. 48.6 MB. |
| `paleos_water_eos_table_pt.dat` | `data/EOS_PALEOS_H2O/` | Space-separated, 10 columns (same as iron) | Attia et al. (2026)[^cite-attia2026]; PALEOS (Zenodo 22776069, v1.3.0) | Unified H2O EOS, 7 EOS (ice Ih--X, liquid, vapor, superionic). P: 1 bar--100 TPa, T: 100--100000 K. 192.7 MB. |
| `chabrier2021_H.dat` | `data/EOS_Chabrier2021_HHe/` | Space-separated, 10 columns (P, T, rho, u, s, cp, cv, alpha, nabla_ad, phase_id) | Chabrier et al. (2019)[^cite-chabrier2019], Chabrier & Debras (2021)[^cite-chabrier2021] | Pure H$_2$ EOS (molecular, atomic, ionized). Grid: 121 x 441 (logT, logP), T: 100--$10^8$ K, P: 1 Pa--$10^{22}$ Pa. Registered as `Chabrier:H`. |
| `chabrier2021_HE.dat` | `data/EOS_Chabrier2021_HHe/` | Same as above | Same | Pure He EOS. Not registered (available for future use). |
| `chabrier2021_HHe_Y0275.dat` | `data/EOS_Chabrier2021_HHe/` | Same as above | Same | H/He mixture at $Y = 0.275$. Not registered (available for future use). |
| `chabrier2021_HHe_Y0292.dat` | `data/EOS_Chabrier2021_HHe/` | Same as above | Same | H/He mixture at $Y = 0.292$. Not registered (available for future use). |
| `chabrier2021_HHe_Y0297.dat` | `data/EOS_Chabrier2021_HHe/` | Same as above | Same | H/He mixture at $Y = 0.297$. Not registered (available for future use). |

### Melting curves

**Tabulated** (downloaded data files):

| File | Location | Format | Source | Description |
|------|----------|--------|--------|-------------|
| `solidus.dat` | `data/melting_curves_Monteux-600/` | Space-separated (P in Pa, T in K) | Monteux et al. (2016)[^cite-monteux2016] | MgSiO3 solidus curve (offset: solidus = liquidus - 600 K). Legacy; selected via `rock_solidus = "Monteux600-solidus-tabulated"`. |
| `liquidus.dat` | `data/melting_curves_Monteux-600/` | Space-separated (P in Pa, T in K) | Monteux et al. (2016)[^cite-monteux2016] | MgSiO3 liquidus curve. Legacy; selected via `rock_liquidus = "Monteux600-liquidus-tabulated"`. |

**Analytic** (hardcoded, no data files needed):

| Identifier | Source | Description |
|------------|--------|-------------|
| `Monteux16-solidus` | Monteux et al. (2016)[^cite-monteux2016] Eqs. 10/12 | Piecewise Simon-Glatzel solidus, valid to ~500 GPa. |
| `Monteux16-liquidus-A-chondritic` | Monteux et al. (2016)[^cite-monteux2016] Eqs. 11/13 | A-chondritic liquidus, valid to ~500 GPa. |
| `Monteux16-liquidus-F-peridotitic` | Monteux et al. (2016)[^cite-monteux2016] Eqs. 11/13 | F-peridotitic liquidus, valid to ~660 GPa. |
| `Stixrude14-solidus` | Stixrude (2014)[^cite-stixrude2014] Eqs. 1.9+1.10 | Simon-like power law with cryoscopic depression ($x_0 = 0.79$). Valid at all $P$. |
| `Stixrude14-liquidus` | Stixrude (2014)[^cite-stixrude2014] Eq. 1.9 | Pure MgSiO$_3$ Simon-like power law. Valid at all $P$. |
| `PALEOS-liquidus` | Belonoshko et al. (2005)[^cite-belonoshko2005] / Fei et al. (2021)[^cite-fei2021] | Piecewise Simon-Glatzel liquidus matching the unified PALEOS MgSiO$_3$ table. Valid at all $P$. |

See [configuration guide](../How-to/configuration.md#melting-curve-selection) for usage.

### Mass-radius curves

| File | Location | Format | Source | Description |
|------|----------|--------|--------|-------------|
| `massradiusEarthlikeRocky.txt` | `data/mass_radius_curves/` | Space-separated (M in M_Earth, R in R_Earth) | Zeng et al. (2019)[^cite-zeng2019] ([model curves](https://lweb.cfa.harvard.edu/~lzeng/planetmodels.html)) | Earth-like rocky M-R curve (32.5% Fe + 67.5% MgSiO3) |
| `massradius_50percentH2O_300K_1mbar.txt` | `data/mass_radius_curves/` | Space-separated (M in M_Earth, R in R_Earth) | Zeng et al. (2019)[^cite-zeng2019] ([model curves](https://lweb.cfa.harvard.edu/~lzeng/planetmodels.html)) | 50% H2O + 50% rocky M-R curve (300 K, 1 mbar surface) |

### Radial profiles (validation benchmarks)

| File | Location | Format | Source | Description |
|------|----------|--------|--------|-------------|
| `radiusdensitySeagerEarth.txt` | `data/radial_profiles/` | CSV (r, rho) | Seager et al. (2007)[^cite-seager2007] | Earth-like radial density profile |
| `radiusdensitySeagerEarthbymass.txt` | `data/radial_profiles/` | CSV (M, r, rho) | Seager et al. (2007)[^cite-seager2007] | Earth-like radial density profile (indexed by enclosed mass) |
| `radiusdensitySeagerwater.txt` | `data/radial_profiles/` | CSV (r, rho) | Seager et al. (2007)[^cite-seager2007] | Water-world radial density profile |
| `radiusdensitySeagerwaterbymass.txt` | `data/radial_profiles/` | CSV (M, r, rho) | Seager et al. (2007)[^cite-seager2007] | Water-world radial density profile (indexed by enclosed mass) |
| `radiusdensityEarthBoujibar.txt` | `data/radial_profiles/` | CSV (r, rho) | Boujibar et al. (2020)[^cite-boujibar2020] | Earth radial density profile |
| `radiuspressureEarthBoujibar.txt` | `data/radial_profiles/` | CSV (r, P) | Boujibar et al. (2020)[^cite-boujibar2020] | Earth radial pressure profile |
| `radiusdensityWagner.txt` | `data/radial_profiles/` | CSV (r, rho) | Wagner et al. (2011)[^cite-wagner2011] | Earth radial density profile |
| `radiuspressureWagner.txt` | `data/radial_profiles/` | CSV (r, P) | Wagner et al. (2011)[^cite-wagner2011] | Earth radial pressure profile |
| `radiusgravityWagner.txt` | `data/radial_profiles/` | CSV (r, g) | Wagner et al. (2011)[^cite-wagner2011] | Earth radial gravity profile |

## Seager EOS data

Zalmoxis uses the tabulated equation of state (EOS) data from Seager et al. (2007)[^cite-seager2007], providing pressure-density relations for iron (Fe epsilon), silicate (MgSiO3 perovskite), and water ice (ice VII).
These tables are automatically downloaded during the setup process described above.

## Analytic EOS

The `Analytic:<material>` EOS options (e.g. `Analytic:iron`, `Analytic:MgSiO3`, `Analytic:H2O`) use the analytic modified polytropic fits from Table 3 of Seager et al. (2007)[^cite-seager2007].
All fit parameters are hardcoded directly from the paper and **do not require any data downloads**.
This makes the analytic EOS suitable for quick setup, CI environments, or situations where the tabulated data files are not available.

Any EOS name beginning with `Analytic:` is self-contained.
The analytic EOS additionally provides three materials (MgFeSiO3, graphite, SiC) that are not available in the tabulated data.

## Mass-radius relations

The following curves serve as reference benchmarks for validating the Zalmoxis internal structure model under both rocky and water-rich configurations:

* Earth-like rocky exoplanets, with a 32.5% Fe + 67.5% MgSiO3 composition, taken from Zeng et al. (2019)[^cite-zeng2019] ([model curves](https://lweb.cfa.harvard.edu/~lzeng/planetmodels.html)).

* Water exoplanets, with a 50% Earth-like rocky core (32.5% Fe + 67.5% MgSiO3) and 50% water by mass at 300 K and surface pressure of 1 mbar.

[^cite-attia2026]: M. Attia, T. Lichtenberg, E. Jungová, M. Sastre, *[PALEOS: Multiphase equations of state and mass-radius relations for exoplanet interiors](https://doi.org/10.1051/0004-6361/202660790)*, Astronomy and Astrophysics, 713, A225, 2026. [SciX](https://scixplorer.org/abs/2026A%26A...713A.225A/abstract).
[^cite-belonoshko2005]: A. B. Belonoshko, N. V. Skorodumova, A. Rosengren, R. Ahuja, B. Johansson, L. Burakovsky, D. L. Preston, *[High-pressure melting of MgSiO$_3$](https://doi.org/10.1103/PhysRevLett.94.195701)*, Physical Review Letters, 94, 195701, 2005. [SciX](https://scixplorer.org/abs/2005PhRvL..94s5701B/abstract).
[^cite-boujibar2020]: A. Boujibar, P. Driscoll, Y. Fei, *[Super-Earth internal structures and initial thermal states](https://doi.org/10.1029/2019JE006124)*, Journal of Geophysical Research: Planets, 125, e2019JE006124, 2020. [SciX](https://scixplorer.org/abs/2020JGRE..12506124B/abstract).
[^cite-chabrier2019]: G. Chabrier, S. Mazevet, F. Soubiran, *[A new equation of state for dense hydrogen-helium mixtures](https://doi.org/10.3847/1538-4357/aaf99f)*, The Astrophysical Journal, 872, 51, 2019. [SciX](https://scixplorer.org/abs/2019ApJ...872...51C/abstract).
[^cite-chabrier2021]: G. Chabrier, F. Debras, *[A new equation of state for dense hydrogen-helium mixtures. II. Taking into account hydrogen-helium interactions](https://doi.org/10.3847/1538-4357/abfc48)*, The Astrophysical Journal, 917, 4, 2021. [SciX](https://scixplorer.org/abs/2021ApJ...917....4C/abstract).
[^cite-fei2021]: Y. Fei, C. T. Seagle, J. P. Townsend, C. A. McCoy, A. Boujibar, P. Driscoll, L. Shulenburger, M. D. Furnish, *[Melting and density of MgSiO$_3$ determined by shock compression of bridgmanite to 1254 GPa](https://doi.org/10.1038/s41467-021-21170-y)*, Nature Communications, 12, 876, 2021. [SciX](https://scixplorer.org/abs/2021NatCo..12..876F/abstract).
[^cite-monteux2016]: J. Monteux, D. Andrault, H. Samuel, *[On the cooling of a deep terrestrial magma ocean](https://doi.org/10.1016/j.epsl.2016.05.010)*, Earth and Planetary Science Letters, 448, 140-149, 2016. [SciX](https://scixplorer.org/abs/2016E%26PSL.448..140M/abstract).
[^cite-mosenfelder2009]: J. L. Mosenfelder, P. D. Asimow, D. J. Frost, D. C. Rubie, T. J. Ahrens, *[The MgSiO$_3$ system at high pressure: Thermodynamic properties of perovskite, postperovskite, and melt from global inversion of shock and static compression data](https://doi.org/10.1029/2008JB005900)*, Journal of Geophysical Research: Solid Earth, 114, B01203, 2009. [SciX](https://scixplorer.org/abs/2009JGRB..114.1203M/abstract).
[^cite-seager2007]: S. Seager, M. Kuchner, C. A. Hier-Majumder, B. Militzer, *[Mass-radius relationships for solid exoplanets](https://doi.org/10.1086/521346)*, The Astrophysical Journal, 669, 1279-1297, 2007. [SciX](https://scixplorer.org/abs/2007ApJ...669.1279S/abstract).
[^cite-stixrude2014]: L. Stixrude, *[Melting in super-earths](https://doi.org/10.1098/rsta.2013.0076)*, Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences, 372, 20130076, 2014. [SciX](https://scixplorer.org/abs/2014RSPTA.37230076S/abstract).
[^cite-wagner2011]: F. W. Wagner, F. Sohl, H. Hussmann, M. Grott, H. Rauer, *[Interior structure models of solid exoplanets using material laws in the infinite pressure limit](https://doi.org/10.1016/j.icarus.2011.05.027)*, Icarus, 214, 366-376, 2011. [SciX](https://scixplorer.org/abs/2011Icar..214..366W/abstract).
[^cite-wolf2018]: A. S. Wolf, D. J. Bower, *[An equation of state for high pressure-temperature liquids (RTpress) with application to MgSiO$_3$ melt](https://doi.org/10.1016/j.pepi.2018.02.004)*, Physics of the Earth and Planetary Interiors, 278, 59-74, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..278...59W/abstract).
[^cite-zeng2019]: L. Zeng, S. B. Jacobsen, D. D. Sasselov, M. I. Petaev, A. Vanderburg, M. Lopez-Morales, J. Perez-Mercader, T. R. Mattsson, G. Li, M. Z. Heising, A. S. Bonomo, M. Damasso, T. A. Berger, H. Cao, A. Levi, R. D. Wordsworth, *[Growth model interpretation of planet size distribution](https://doi.org/10.1073/pnas.1812905116)*, Proceedings of the National Academy of Sciences, 116, 9723-9728, 2019. [SciX](https://scixplorer.org/abs/2019PNAS..116.9723Z/abstract).
