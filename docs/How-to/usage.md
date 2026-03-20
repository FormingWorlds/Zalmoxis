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

All output files are written to the `output_files/` directory (created automatically on first run). The directory structure looks like:

```
output_files/
├── planet_profile.txt            # radial profiles (single run)
├── planet_profile5.0.txt         # radial profiles (batch run, mass in filename)
├── calculated_planet_mass_radius.txt  # summary: mass and radius per run
├── planet_profile.png            # 6-panel structure plot (if plots_enabled)
├── grid_results/                 # grid runner output (if using run_grid)
│   ├── grid_summary.csv
│   └── *.json
└── pressure_profiles.txt         # iteration diagnostics (if enabled)
```

### Profile files

When `data_enabled = true` (the default), the solver writes radial profiles as tab-separated text:

- **`planet_profile.txt`**: Full radial profile for a single-mass run.
- **`planet_profile{mass}.txt`**: Same format, with the planet mass (in Earth masses) appended. Produced by batch runs to distinguish between different masses.

The columns are:

| Column | Unit | Description |
|--------|------|-------------|
| `radius` | m | Distance from the planet center |
| `density` | kg/m$^3$ | Local density |
| `gravity` | m/s$^2$ | Local gravitational acceleration |
| `pressure` | Pa | Local pressure |
| `temperature` | K | Local temperature |
| `mass_enclosed` | kg | Total mass enclosed within this radius |

The first row is the center (r = 0) and the last row is the surface. Shells beyond the planet surface (where the ODE terminal event fired) are padded with zeros.

### Summary file

- **`calculated_planet_mass_radius.txt`**: Two-column summary (calculated mass in kg, calculated radius in m). Each row corresponds to one completed simulation. Appended across successive runs, so multiple runs accumulate in one file.

### Iteration profiles (optional)

When `iteration_profiles_enabled = true` in the `[Output]` section:

- **`pressure_profiles.txt`**: Pressure vs. radius for every iteration of the pressure adjustment loop.
- **`density_profiles.txt`**: Density vs. radius for every iteration.

These files are useful for diagnosing convergence behavior. They grow large quickly and are overwritten at the start of each new run. Leave disabled for production runs.

### Plots (optional)

When `plots_enabled = true` in the `[Output]` section, Zalmoxis automatically generates profile plots after each run. To enable:

```toml
[Output]
plots_enabled = true
```

This produces a six-panel figure showing the radial profiles of density, pressure, temperature, gravity, phase state, and mass enclosed:

![Example profile plot](../img/example_profile_plot.png)

**Example**: 1 $M_\oplus$ planet with a PALEOS iron core and MgSiO$_3$ mantle at $T_s$ = 3000 K (adiabatic mode). The dashed vertical line marks the core-mantle boundary (CMB).

- **Density** (top left): drops from ~14,000 kg/m$^3$ at the center (iron core) to ~3,000 kg/m$^3$ at the surface (silicate mantle). The sharp step at the CMB reflects the iron-to-silicate transition.
- **Pressure** (top center): decreases monotonically from ~350 GPa at the center to 1 atm at the surface, on a logarithmic scale.
- **Temperature** (top right): follows the adiabatic gradient from 3000 K at the surface to ~8000 K at the center. The steeper gradient in the iron core reflects iron's different $\nabla_{\mathrm{ad}}$.
- **Gravity** (bottom left): rises through the core (as enclosed mass grows faster than $r^2$), peaks near the CMB at ~10 m/s$^2$, and stays roughly constant through the mantle.
- **Phase state** (bottom center): colored bars showing the thermodynamically stable phase at each depth, read from the PALEOS table's phase column. Liquid iron in the core (at 3000 K surface temperature), bridgmanite (solid MgSiO$_3$) in the mantle, with a thin liquid layer near the surface.
- **Mass enclosed** (bottom right): enclosed mass as a function of radius. The CMB is visible as the kink where the slope changes from the dense core to the lighter mantle.

## Running parameter grids

To sweep over any combination of parameters, use the grid runner with a TOML specification file:

```console
python -m src.tools.run_grid <grid.toml> -j <workers>
```

The `-j` flag sets the number of parallel workers (default: 1, serial execution).

### Grid TOML format

A grid TOML file has three sections:

```toml
[base]
config = "input/default.toml"  # base config (relative to ZALMOXIS_ROOT)

[sweep]
# Each key is a parameter name, each value is a list to sweep over.
# The runner generates the Cartesian product of all sweep parameters.
planet_mass = [0.5, 1.0, 3.0, 5.0, 10.0]
surface_temperature = [1000, 2000, 3000]

[output]
dir = "output_files/grid_results"  # output directory (relative to ZALMOXIS_ROOT)
```

The available sweep parameter names and their mapping to TOML config sections are:

| Sweep parameter | Config section | Config key |
|---|---|---|
| `planet_mass` | `[InputParameter]` | `planet_mass` |
| `core_mass_fraction` | `[AssumptionsAndInitialGuesses]` | `core_mass_fraction` |
| `mantle_mass_fraction` | `[AssumptionsAndInitialGuesses]` | `mantle_mass_fraction` |
| `temperature_mode` | `[AssumptionsAndInitialGuesses]` | `temperature_mode` |
| `surface_temperature` | `[AssumptionsAndInitialGuesses]` | `surface_temperature` |
| `center_temperature` | `[AssumptionsAndInitialGuesses]` | `center_temperature` |
| `core` | `[EOS]` | `core` |
| `mantle` | `[EOS]` | `mantle` |
| `ice_layer` | `[EOS]` | `ice_layer` |
| `condensed_rho_min` | `[EOS]` | `condensed_rho_min` |
| `condensed_rho_scale` | `[EOS]` | `condensed_rho_scale` |
| `binodal_T_scale` | `[EOS]` | `binodal_T_scale` |
| `mushy_zone_factor` | `[EOS]` | `mushy_zone_factor` |
| `rock_solidus` | `[EOS]` | `rock_solidus` |
| `rock_liquidus` | `[EOS]` | `rock_liquidus` |
| `num_layers` | `[Calculations]` | `num_layers` |

### Example: mass-radius curve

Sweep planet mass to build a mass-radius relation for rocky planets (file: `input/grids/mass_radius.toml`):

```toml
[base]
config = "input/default.toml"

[sweep]
planet_mass = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

[output]
dir = "output_files/grid_mass_radius"
```

Run with 2 workers:

```console
python -m src.tools.run_grid input/grids/mass_radius.toml -j 2
```

### Example: H$_2$ mixing grid

Sweep planet mass, mantle composition, and surface temperature to explore the effect of dissolved hydrogen on planet radius (file: `input/grids/h2_mixing.toml`):

```toml
[base]
config = "input/default.toml"

[sweep]
planet_mass = [1.0, 3.0, 5.0, 10.0]
mantle = ["PALEOS:MgSiO3", "PALEOS:MgSiO3:0.97+Chabrier:H:0.03", "PALEOS:MgSiO3:0.90+Chabrier:H:0.10"]
surface_temperature = [2000, 3000]

[output]
dir = "output_files/grid_h2_mixing"
```

This produces 4 x 3 x 2 = 24 models.

### Output

Each grid run produces:

- **`grid_summary.csv`**: One row per model with columns for all sweep parameters, calculated radius (R_earth), mass (M_earth), convergence flags, wall time, and any error messages.
- **Per-run JSON files**: Individual `<label>.json` files with the same information, named by the parameter combination.

Plotting and per-profile data output are disabled during grid runs for speed. To generate plots for specific parameter combinations, run them individually with `plots_enabled = true`.

