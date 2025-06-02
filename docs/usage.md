# Usage

## Running Zalmoxis

You can directly run Zalmoxis using the command:

```console
python -m zalmoxis -c [cfgfile]
```

where `[cfgfile]` is typically the path to the default configuration file (typically `input/default.toml` from the base directory).

### Simulating Earth-like rocky exoplanets

To simulate a rocky exoplanet with Earth-like composition, configure the following parameters in the default configuration file:

```console
[AssumptionsAndInitialGuesses]
core_mass_fraction = 0.325     
inner_mantle_mass_fraction = 0 
weight_iron_fraction = 0.325         

[EOS]
choice = "Tabulated:iron/silicate"
```

This configuration models a fully differentiated two-layer planet consisting of a 32.5% iron core and a silicate mantle by mass.

### Simulating water-rich exoplanets

To simulate a water-rich planet:

```console
[InputParameter]
core_mass_fraction = 0.065
inner_mantle_mass_fraction = 0.485

[EOS]
choice = "Tabulated:water"
```

This configuration models a fully differentiated three-layer planet composed of a 6.5% iron core, a 48.5% silicate mantle, and a 45% outer water layer by mass.

## Running Zalmoxis in parallel for multiple masses

To run Zalmoxis over a range of planetary masses in parallel, you can use the `run_parallel.py` utility from the `src/tools/` directory using the command:

```console
python -m src.tools.run_parallel [choice]
```

where `[choice]` specifies the set of planetary masses to simulate. The available options are:

* `Wagner`: Simulates 7 rocky planets with masses of 1, 2.5, 5, 7.5, 10, 12.5, and 15 Earth masses. This option is designed to reproduce the mass range explored in [Wagner et al., 2012](https://www.aanda.org/articles/aa/full_html/2012/05/aa18441-11/aa18441-11.html). It is physically meaningful when the configuration is set to simulate Earth-like rocky exoplanets as described above.
* `Boujibar`: Simulates rocky planets with integer masses from 1 to 10 Earth masses, corresponding to the mass range used in [Boujibar et al., 2020](https://ui.adsabs.harvard.edu/abs/2020JGRE..12506124B/abstract). This is meaningful when using the Earth-like rocky planet configuration.
* `default`: Runs simulations for planets with masses from 1 to 10 Earth masses (inclusive), in unit steps. This serves as the fallback set if no specific option is provided. 
* `SeagerEarth`: Simulates Earth-like rocky planets with masses of 1, 5, 10, and 50 Earth masses, to replicate the models in [Seager et al., 2007](https://iopscience.iop.org/article/10.1086/521346). Use this with the Earth-like rocky planet configuration.
* `Seagerwater`: Simulates water-rich planets with masses of 1, 5, 10, and 50 Earth masses. Use this with the water-rich planet configuration.
* `custom`: Simulates 50 planets with integer masses from 1 to 50 Earth masses, ideal for generating high-resolution mass-radius curves or broader parametric sweeps.