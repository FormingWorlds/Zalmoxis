# Usage

## Quick Start

Run an Earth-like rocky planet with the default configuration:

```console
python -m zalmoxis -c input/default.toml
```

This uses the default `input/default.toml`, which models a 1 Earth-mass, two-layer planet (iron core + MgSiO3 mantle). Output files are written to the `output_files/` directory.

To change the planet mass, edit the `[InputParameter]` section:

```toml
[InputParameter]
planet_mass = 5  # in Earth masses
```

## Configuration

The full configuration is specified in a TOML file. The key sections are described below. See `input/default.toml` for all available parameters and their defaults.

### EOS Configuration

Each layer's equation of state is set independently in the `[EOS]` section. The format is `"<source>:<composition>"` for each layer. Set `ice_layer` to an empty string (or omit it) for a two-layer model; set it to a valid EOS string for a three-layer model.

Valid EOS values:

| EOS string               | Description                                           |
|---------------------------|-------------------------------------------------------|
| `Seager2007:iron`         | Seager et al. (2007) tabulated Fe epsilon (300 K)     |
| `Seager2007:MgSiO3`      | Seager et al. (2007) tabulated MgSiO3 perovskite (300 K) |
| `Seager2007:H2O`         | Seager et al. (2007) tabulated water ice (300 K)      |
| `WolfBower2018:MgSiO3`   | Wolf & Bower (2018) T-dependent MgSiO3 (melt/solid), **<= 2 M_earth** |
| `Analytic:<material>`    | Seager et al. (2007) analytic polytrope (300 K, no data files needed) |

Valid analytic materials: `iron`, `MgSiO3`, `MgFeSiO3`, `H2O`, `graphite`, `SiC`.

### Temperature Modes

The `temperature_mode` parameter in `[AssumptionsAndInitialGuesses]` controls how the internal temperature profile is computed:

- `"isothermal"`: Constant temperature equal to `surface_temperature` throughout the planet.
- `"linear"`: Linear gradient from `center_temperature` (at the center) to `surface_temperature` (at the surface).
- `"prescribed"`: Read from a file specified by `temperature_profile_file`.

Temperature-dependent EOS (`WolfBower2018:MgSiO3`) requires one of these modes to be set. The 300 K tabulated and analytic EOS options ignore the temperature profile but still require valid mode settings in the config.

### Temperature Profile File Format

When using `temperature_mode = "prescribed"`, the file specified by `temperature_profile_file` must be a plain text file with one column of temperature values (in K), one value per radial grid point, ordered from the planet center to the surface. The number of values must equal `num_layers` in the `[Calculations]` section.

Example for `num_layers = 5`:

```
6000.0
5500.0
4500.0
3200.0
2000.0
```

The file must be placed in the `input/` directory.

## Example Configurations

### Earth-like rocky planet

A fully differentiated two-layer planet with a 32.5% iron core and a cold (300 K) MgSiO3 perovskite mantle, following [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346).

```toml
[InputParameter]
planet_mass = 1

[AssumptionsAndInitialGuesses]
core_mass_fraction = 0.325
mantle_mass_fraction = 0
weight_iron_fraction = 0.325

[EOS]
core = "Seager2007:iron"
mantle = "Seager2007:MgSiO3"
ice_layer = ""
```

### Hot rocky planet (temperature-dependent mantle)

Same iron core, but the mantle uses the Wolf & Bower (2018) temperature-dependent MgSiO3 EOS. The mantle can be solid, partially molten, or fully molten depending on local pressure and temperature. This is in contrast to the 300 K EOS from Seager et al. (2007), which represents a purely solid mantle.

```toml
[InputParameter]
planet_mass = 1

[AssumptionsAndInitialGuesses]
core_mass_fraction = 0.325
mantle_mass_fraction = 0
weight_iron_fraction = 0.325
temperature_mode = "isothermal"
surface_temperature = 2000
center_temperature = 3000
temperature_profile_file = "temp_profile.txt"

[EOS]
core = "Seager2007:iron"
mantle = "WolfBower2018:MgSiO3"
ice_layer = ""
```

For a linear temperature gradient, set `temperature_mode = "linear"` and specify both `surface_temperature` and `center_temperature`. For a custom profile from file, set `temperature_mode = "prescribed"` and provide the file name in `temperature_profile_file`.

### Water-rich planet

A fully differentiated three-layer planet: 6.5% iron core, 48.5% silicate mantle, and 45% outer water-ice layer by mass.

```toml
[InputParameter]
planet_mass = 1

[AssumptionsAndInitialGuesses]
core_mass_fraction = 0.065
mantle_mass_fraction = 0.485
weight_iron_fraction = 0.065

[EOS]
core = "Seager2007:iron"
mantle = "Seager2007:MgSiO3"
ice_layer = "Seager2007:H2O"
```

### Mixed EOS: tabulated core + analytic mantle

Tabulated iron core with an analytic Seager et al. (2007) silicate mantle. The analytic EOS requires no data files and uses a modified polytropic fit accurate to 2--12% for all planetary pressures.

```toml
[EOS]
core = "Seager2007:iron"
mantle = "Analytic:MgSiO3"
ice_layer = ""
```

### Carbon planet

An analytic iron core with an analytic silicon carbide mantle.

```toml
[AssumptionsAndInitialGuesses]
core_mass_fraction = 0.30
mantle_mass_fraction = 0
weight_iron_fraction = 0.30

[EOS]
core = "Analytic:iron"
mantle = "Analytic:SiC"
ice_layer = ""
```

### Water world with tabulated core and mantle

Tabulated iron core and silicate mantle with an analytic H2O outer layer.

```toml
[AssumptionsAndInitialGuesses]
core_mass_fraction = 0.10
mantle_mass_fraction = 0.40
weight_iron_fraction = 0.10

[EOS]
core = "Seager2007:iron"
mantle = "Seager2007:MgSiO3"
ice_layer = "Analytic:H2O"
```

## Output Description

All output files are written to the `output_files/` directory.

### Profile files

- **`planet_profile.txt`**: Full radial profile for a single-mass run. Columns: radius (m), density (kg/m^3), gravity (m/s^2), pressure (Pa), temperature (K), mass enclosed (kg).
- **`planet_profile{mass}.txt`**: Same format, but with the planet mass (in Earth masses) appended to the filename. Produced by parallel batch runs to distinguish between different masses.

### Summary file

- **`calculated_planet_mass_radius.txt`**: Two-column summary (calculated mass in kg, calculated radius in m). Each row corresponds to one completed simulation. Appended across successive runs.

### Iteration profiles (optional)

When `iteration_profiles_enabled = true` in the `[Output]` section:

- **`pressure_profiles.txt`**: Pressure vs. radius for every iteration of the pressure adjustment loop.
- **`density_profiles.txt`**: Density vs. radius for every iteration.

These files are useful for diagnosing convergence behavior. They are overwritten at the start of each new run.

### Plots (optional)

When `plots_enabled = true` in the `[Output]` section, PDF plots of the radial profiles (density, gravity, pressure, temperature) are generated automatically after each run. If the temperature-dependent mantle EOS (`WolfBower2018:MgSiO3`) is used, an additional pressure-temperature phase diagram with mantle phase information is produced.

## Running Zalmoxis in parallel for multiple masses

To run Zalmoxis over a range of planetary masses in parallel, use the `run_parallel.py` utility:

```console
python -m src.tools.run_parallel [choice]
```

where `[choice]` specifies the set of planetary masses to simulate. The available options are:

* `Wagner`: 7 rocky planets with masses of 1, 2.5, 5, 7.5, 10, 12.5, and 15 Earth masses, following [Wagner et al. (2012)](https://www.aanda.org/articles/aa/full_html/2012/05/aa18441-11/aa18441-11.html).
* `Boujibar`: Integer masses from 1 to 10 Earth masses, following [Boujibar et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020JGRE..12506124B/abstract).
* `default`: Masses from 1 to 10 Earth masses in unit steps. Fallback if no option is provided.
* `SeagerEarth`: Masses of 1, 5, 10, and 50 Earth masses, following [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346). Use with the Earth-like rocky planet configuration.
* `Seagerwater`: Same masses as `SeagerEarth`. Use with the water-rich planet configuration.
* `custom`: Integer masses from 1 to 50 Earth masses, for generating high-resolution mass-radius curves.

All parallel runs read from `input/default.toml` and override the planet mass for each run. Each mass produces its own `planet_profile{mass}.txt` file, and all results are collected in `calculated_planet_mass_radius.txt`.
