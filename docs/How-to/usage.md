# Usage

## Quick start

Run the default configuration (1 Earth-mass planet with a PALEOS iron core and MgSiO3 mantle):

```console
python -m zalmoxis -c input/default.toml
```

Output files appear in `output_files/`. The PALEOS data files must be downloaded first (see [installation](installation.md)).

## What happens when you run Zalmoxis

Zalmoxis computes the internal structure of a planet. Given a total mass and composition (which layers the planet has, and what each layer is made of), it finds the radial profiles of pressure, density, gravity, and temperature from the center to the surface. The result is a self-consistent model of the planet's interior: how big it is, where the core-mantle boundary sits, and how conditions change with depth.

For the mathematical details of how the solver works, see the [process flow](../Explanations/process_flow.md) explanation.

## Modifying the configuration

All input parameters live in a single TOML file. To change the planet mass, for example, open `input/default.toml` and edit the `[InputParameter]` section.

Before:

```toml
[InputParameter]
planet_mass = 1.0  # in Earth masses
```

After (5 Earth-mass super-Earth):

```toml
[InputParameter]
planet_mass = 5.0  # in Earth masses
```

Then re-run:

```console
python -m zalmoxis -c input/default.toml
```

For the full list of parameters (EOS selection, temperature modes, solver settings, output options), see the [configuration reference](configuration.md).

## Output files

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

When `plots_enabled = true` in the `[Output]` section, Zalmoxis automatically generates profile plots after each run. To enable:

```toml
[Output]
plots_enabled = true
```

This produces a four-panel figure showing the radial profiles of density, pressure, gravity, and temperature from the center to the surface:

![Example profile plot](../img/example_profile_plot.png)

**Example**: 1 $M_\oplus$ planet with a PALEOS iron core and MgSiO$_3$ mantle at $T_s$ = 3000 K (adiabatic mode). The dashed vertical line marks the core-mantle boundary (CMB) at ~0.5 $R_\oplus$.

- **Density** (top left): drops from ~14,000 kg/m$^3$ at the center (iron core) to ~3,000 kg/m$^3$ at the surface (silicate mantle). The sharp step at the CMB reflects the iron-to-silicate transition.
- **Pressure** (top right): decreases monotonically from ~350 GPa at the center to 1 atm at the surface, on a logarithmic scale.
- **Gravity** (bottom left): rises through the core (as enclosed mass grows faster than $r^2$), peaks near the CMB at ~10 m/s$^2$, and stays roughly constant through the mantle.
- **Temperature** (bottom right): follows the adiabatic gradient from 3000 K at the surface to ~8000 K at the center. The steeper gradient in the iron core reflects iron's different $\nabla_{\mathrm{ad}}$.

If a temperature-dependent mantle EOS is used (any PALEOS, WolfBower2018, or RTPress100TPa EOS), an additional pressure-temperature phase diagram with mantle phase information is produced.

## Running multiple masses in parallel

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
