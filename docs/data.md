# Reference Data

## Mass-radius relations

The following curves serve as reference benchmarks for validating the Zalmoxis internal structure model under both rocky and water-rich configurations:

* Earth-like rocky exoplanets, with a 32.5% Fe + 67.5% MgSiO3 composition, taken from [Zeng et al., 2019](https://lweb.cfa.harvard.edu/~lzeng/planetmodels.html)

* Water exoplanets, with a 50% Earth-like rocky core (32.5% Fe + 67.5% MgSiO3) and 50% water by mass at 300 K and surface pressure of 1 millibar

## Seager EOS data

Zalmoxis makes use of the equation of state (EOS) data from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346), which provide pressure-density relations for iron, silicate, and water ice. This data is automatically downloaded during the setup process (see Installation Step 4 of the [installation guide](https://proteus-framework.org/Zalmoxis/installation/)).

Alternatively, the `"Analytic:Seager2007"` EOS choice uses the analytic modified polytropic fits from Table 3 of [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346). These parameters are hardcoded from the paper and **do not require any data downloads**. This makes the analytic EOS suitable for quick setup, CI environments, or situations where the tabulated data files are not available. The analytic EOS additionally provides 3 materials (MgFeSiO3, graphite, SiC) that are not available in the tabulated data.