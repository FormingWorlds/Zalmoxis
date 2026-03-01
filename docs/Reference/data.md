# Reference Data

## Download instructions

All tabulated data files are downloaded automatically by running the setup script from the repository root:

```bash
bash src/get_zalmoxis.sh
```

This invokes `src/setup_zalmoxis.py`, which downloads and extracts the required data into the `data/` directory.
See Installation Step 4 of the [installation guide](https://proteus-framework.org/Zalmoxis/installation/) for details.

## Data inventory

The table below lists every data file used by Zalmoxis, organised by subdirectory.

### Equation of state tables

| File | Location | Format | Source | Description |
|------|----------|--------|--------|-------------|
| `eos_seager07_iron.txt` | `data/EOS_Seager2007/` | CSV (rho in g/cm^3, P in GPa) | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) | Fe epsilon pressure-density EOS |
| `eos_seager07_silicate.txt` | `data/EOS_Seager2007/` | CSV (rho in g/cm^3, P in GPa) | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) | MgSiO3 perovskite pressure-density EOS |
| `eos_seager07_water.txt` | `data/EOS_Seager2007/` | CSV (rho in g/cm^3, P in GPa) | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) | Water ice VII pressure-density EOS |
| `density_melt.dat` | `data/EOS_WolfBower2018_1TPa/` | TSV (P in Pa, T in K, rho in kg/m^3) | [Wolf & Bower (2018)](https://doi.org/10.1016/j.pepi.2018.02.004) | MgSiO3 melt density EOS (P: 0--1 TPa, T: 0--16500 K). Out-of-bounds pressures are clamped to the table edge. |
| `density_solid.dat` | `data/EOS_WolfBower2018_1TPa/` | TSV (P in Pa, T in K, rho in kg/m^3) | [Wolf & Bower (2018)](https://doi.org/10.1016/j.pepi.2018.02.004) | MgSiO3 solid (bridgmanite) density EOS, derived from [Mosenfelder et al. (2009)](https://doi.org/10.1029/2008JB005900) (P: 0--1 TPa, T: 0--16500 K). Out-of-bounds pressures are clamped to the table edge. |
| `density_melt.dat` | `data/EOS_RTPress_melt_100TPa/` | TSV (P in Pa, T in K, rho in kg/m^3) | Extended RTpress melt table | MgSiO3 melt density EOS extended to 100 TPa (P: 1e3--1e14 Pa, T: 400--50000 K). Used by `RTPress100TPa:MgSiO3`. Solid phase uses the WolfBower2018 table above. |

### Melting curves

| File | Location | Format | Source | Description |
|------|----------|--------|--------|-------------|
| `solidus.dat` | `data/melting_curves_Monteux-600/` | Space-separated (P in Pa, T in K) | [Monteux et al. (2016)](https://doi.org/10.1016/j.epsl.2016.05.010) | MgSiO3 solidus curve (offset: solidus = liquidus − 600 K) |
| `liquidus.dat` | `data/melting_curves_Monteux-600/` | Space-separated (P in Pa, T in K) | [Monteux et al. (2016)](https://doi.org/10.1016/j.epsl.2016.05.010) | MgSiO3 liquidus curve |

### Mass-radius curves

| File | Location | Format | Source | Description |
|------|----------|--------|--------|-------------|
| `massradiusEarthlikeRocky.txt` | `data/mass_radius_curves/` | Space-separated (M in M_Earth, R in R_Earth) | [Zeng et al. (2019)](https://lweb.cfa.harvard.edu/~lzeng/planetmodels.html) | Earth-like rocky M-R curve (32.5% Fe + 67.5% MgSiO3) |
| `massradius_50percentH2O_300K_1mbar.txt` | `data/mass_radius_curves/` | Space-separated (M in M_Earth, R in R_Earth) | [Zeng et al. (2019)](https://lweb.cfa.harvard.edu/~lzeng/planetmodels.html) | 50% H2O + 50% rocky M-R curve (300 K, 1 mbar surface) |

### Radial profiles (validation benchmarks)

| File | Location | Format | Source | Description |
|------|----------|--------|--------|-------------|
| `radiusdensitySeagerEarth.txt` | `data/radial_profiles/` | CSV (r, rho) | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) | Earth-like radial density profile |
| `radiusdensitySeagerEarthbymass.txt` | `data/radial_profiles/` | CSV (M, r, rho) | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) | Earth-like radial density profile (indexed by enclosed mass) |
| `radiusdensitySeagerwater.txt` | `data/radial_profiles/` | CSV (r, rho) | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) | Water-world radial density profile |
| `radiusdensitySeagerwaterbymass.txt` | `data/radial_profiles/` | CSV (M, r, rho) | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) | Water-world radial density profile (indexed by enclosed mass) |
| `radiusdensityEarthBoujibar.txt` | `data/radial_profiles/` | CSV (r, rho) | [Boujibar et al. (2020)](https://doi.org/10.1029/2019JE006124) | Earth radial density profile |
| `radiuspressureEarthBoujibar.txt` | `data/radial_profiles/` | CSV (r, P) | [Boujibar et al. (2020)](https://doi.org/10.1029/2019JE006124) | Earth radial pressure profile |
| `radiusdensityWagner.txt` | `data/radial_profiles/` | CSV (r, rho) | [Wagner et al. (2011)](https://doi.org/10.1016/j.icarus.2011.05.027) | Earth radial density profile |
| `radiuspressureWagner.txt` | `data/radial_profiles/` | CSV (r, P) | [Wagner et al. (2011)](https://doi.org/10.1016/j.icarus.2011.05.027) | Earth radial pressure profile |
| `radiusgravityWagner.txt` | `data/radial_profiles/` | CSV (r, g) | [Wagner et al. (2011)](https://doi.org/10.1016/j.icarus.2011.05.027) | Earth radial gravity profile |

## Seager EOS data

Zalmoxis uses the tabulated equation of state (EOS) data from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346), providing pressure-density relations for iron (Fe epsilon), silicate (MgSiO3 perovskite), and water ice (ice VII).
These tables are automatically downloaded during the setup process described above.

## Analytic EOS

The `Analytic:<material>` EOS options (e.g. `Analytic:iron`, `Analytic:MgSiO3`, `Analytic:H2O`) use the analytic modified polytropic fits from Table 3 of [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346).
All fit parameters are hardcoded directly from the paper and **do not require any data downloads**.
This makes the analytic EOS suitable for quick setup, CI environments, or situations where the tabulated data files are not available.

Any EOS name beginning with `Analytic:` is self-contained.
The analytic EOS additionally provides three materials (MgFeSiO3, graphite, SiC) that are not available in the tabulated data.

## Mass-radius relations

The following curves serve as reference benchmarks for validating the Zalmoxis internal structure model under both rocky and water-rich configurations:

* Earth-like rocky exoplanets, with a 32.5% Fe + 67.5% MgSiO3 composition, taken from [Zeng et al. (2019)](https://lweb.cfa.harvard.edu/~lzeng/planetmodels.html).

* Water exoplanets, with a 50% Earth-like rocky core (32.5% Fe + 67.5% MgSiO3) and 50% water by mass at 300 K and surface pressure of 1 mbar.
