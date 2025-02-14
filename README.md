# Zalmoxis

# Exoplanet Internal Structure Model

## Overview
This program models the internal structure of exoplanets using an iterative numerical approach. The numerical framework solves a system of coupled ordinary differential equations (ODEs), integrating the equations of hydrostatic equilibrium, mass conservation, and gravity, including realistic material properties. Currently, existing tabulated pressure-density data are used. The model takes as input parameters the planet’s mass and core mass fraction and will integrate the relevant ODEs from the centre to the surface of the planet, checking for convergence at every iteration.

The output includes planetary radius, core radius, density profiles, pressure, and temperature distributions.

## Features
- Reads input parameters from a TOML configuration file
- Supports multiple EOS models (Birch-Murnaghan, Mie-Gruneisen-Debye, etc.) but only the Tabulated choice is tested for now
- Implements an iterative solution for pressure, density, and radius convergence
- Uses `solve_ivp` for numerical integration
- Includes options for outputting data and generating plots

## Installation instructions

TBC

## Documentation

TBC

## Dependencies
Ensure the following Python packages are installed:
```bash
pip install numpy matplotlib scipy toml
```

## Repository Structure
```
.
├── LICENSE
├── README.md
├── input
│   └── default.toml
└── src
    ├── __init__.py
    ├── __pycache__
    │   └── __init__.cpython-310.pyc
    ├── jord
    │   ├── MR_plot.pdf
    │   ├── __init__.py
    │   ├── __pycache__
    │   │   ├── __init__.cpython-310.pyc
    │   │   ├── constants.cpython-310.pyc
    │   │   ├── eos_functions.cpython-310.pyc
    │   │   ├── eos_properties.cpython-310.pyc
    │   │   ├── jord.cpython-310.pyc
    │   │   └── structure_model.cpython-310.pyc
    │   ├── all_profiles_with_colorbar_vs_Boujibar.pdf
    │   ├── all_profiles_with_colorbar_vs_Seager.pdf
    │   ├── all_profiles_with_colorbar_vs_Wagner.pdf
    │   ├── all_profiles_with_colorbar_vs_custom.pdf
    │   ├── all_profiles_with_colorbar_vs_default.pdf
    │   ├── constants.py
    │   ├── eos_functions.py
    │   ├── eos_properties.py
    │   ├── jord.py
    │   ├── output_files
    │   │   ├── calculated_planet_mass_radius.txt
    │   │   ├── planet_profile1.txt
    │   │   ├── planet_profile10.txt
    │   │   ├── planet_profile12.5.txt
    │   │   ├── planet_profile15.txt
    │   │   ├── planet_profile2.5.txt
    │   │   ├── planet_profile5.txt
    │   │   └── planet_profile7.5.txt
    │   ├── planet_eos.pdf
    │   ├── plots
    │   │   ├── __pycache__
    │   │   │   ├── plot_MR.cpython-310.pyc
    │   │   │   ├── plot_eos.cpython-310.pyc
    │   │   │   ├── plot_profiles.cpython-310.pyc
    │   │   │   └── plot_profiles_all_in_one.cpython-310.pyc
    │   │   ├── plot_MR.py
    │   │   ├── plot_eos.py
    │   │   ├── plot_profiles.py
    │   │   └── plot_profiles_all_in_one.py
    │   └── structure_model.py
    └── tests
        ├── MRtest.py
        ├── __pycache__
        │   └── MRtest.cpython-310.pyc
        ├── benchmarks
        │   └── MR-Earth.txt
        ├── test1.py
        └── test2.py
## Data Download
When running `MRtest.py`, the required data files are automatically downloaded when the script is executed. 
The following data files will be downloaded in a data folder in the main directory:

- `eos_seager07_iron.txt`
- `eos_seager07_silicate.txt`
- `eos_seager07_water.txt`
- `massradiusEarthlikeRockyZeng.txt`
- `radiusdensityEarthBoujibar.txt`
- `radiusdensityMarsBoujibar.txt`
- `radiusdensityMercuryBoujibar.txt`
- `radiusdensitySeager.txt`
- `radiusdensityWagner.txt`
- `radiusgravityWagner.txt`
- `radiuspressureEarthBoujibar.txt`
- `radiuspressureMarsBoujibar.txt`
- `radiuspressureMercuryBoujibar.txt`
- `radiuspressureWagner.txt`
- `radiustemperatureWagner.txt`

A brief description of each data file:

- **`eos_seager07_iron.txt`**: Contains equations of state (EOS) data for iron (Seager et al. 2007). Density is in g/cm^-3 (first column) and pressure is in GPa (second column).
- **`eos_seager07_silicate.txt`**: Provides EOS data for silicates (Seager et al. 2007). Density is in g/cm^-3 (first column) and pressure is in GPa (second column).
- **`eos_seager07_water.txt`**: Contains EOS data for water (Seager et al. 2007). Density is in g/cm^-3 (first column) and pressure is in GPa (second column).
- **`massradiusEarthlikeRockyZeng.txt`**: Mass-Radius Curves Data of Earth-like rocky planets (32.5% Fe+67.5% MgSiO3) from Zeng et al. 2019. Masses(first column) and Radii (second column) are in Earth Units.
- **`radiusdensityEarthBoujibar.txt`**: Contains data for the radius and density of super‐Earths of masses ranging 1 to 10 Earth's massess with an Earth-like core mass fraction (CMF) of 0.32 (Boujibar et al. 2020).
- **`radiusdensityMarsBoujibar.txt`**: Similar to the Earth data, but for Mars-like planets, with a CMF of 0.2 (Boujibar et al. 2020).
- **`radiusdensityMercuryBoujibar.txt`**: Similar to the Earth data, but for Mercury-like planets, with a CMF of 0.68 (Boujibar et al. 2020).
- **`radiuspressureEarthBoujibar.txt`**: Contains data for the radius and pressure of super‐Earths of masses ranging 1 to 10 Earth's massess with an Earth-like core mass fraction (CMF) of 0.32 (Boujibar et al. 2020).
- **`radiuspressureMarsBoujibar.txt`**: Similar to the Earth data, but for Mars-like planets, with a CMF of 0.2 (Boujibar et al. 2020).
- **`radiuspressureMercuryBoujibar.txt`**: Similar to the Earth data, but for Mercury-like planets, with a CMF of 0.68 (Boujibar et al. 2020).
- **`radiusdensitySeager.txt`**: Contains data for the radius and density of silicate planets with a 32.5% by mass Fe core and a 67.5% MgSiO3 mantle (Seager et al. 2007).
- **`radiusdensityWagner.txt`**: Contains data for the radius and density for generic Earth-like exoplanets ranging from 1 to 15 Earth's massess (Wagner et al. 2012).
- **`radiusgravityWagner.txt`**: Contains data for the radius and gravity for generic Earth-like exoplanets ranging from 1 to 15 Earth's massess (Wagner et al. 2012).
- **`radiuspressureWagner.txt`**: Contains data for the radius and pressure for generic Earth-like exoplanets ranging from 1 to 15 Earth's massess (Wagner et al. 2012).
- **`radiustemperatureWagner.txt`**: Contains data for the radius and temperature for generic Earth-like exoplanets ranging from 1 to 15 Earth's massess (Wagner et al. 2012).


## Usage
Run the program via the command line:
```bash
python -m src.jord.jord -c input/default.toml
```

Alternatively, specify a different configuration file:
```bash
python -m src.jord.jord -c path/to/config.toml
```

## Tests

TBC

## Configuration
The input parameters are stored in a TOML file. Example:
```toml
[InputParameter]
planet_mass = 5.972e24  # Mass in kg

[AssumptionsAndInitialGuesses]
core_radius_fraction = 0.5
core_mass_fraction = 0.32
weight_iron_fraction = 0.7

[EOS]
choice = "Birch-Murnaghan"

[Calculations]
num_layers = 100

[IterativeProcess]
max_iterations_outer = 50
tolerance_outer = 1e-4

[PressureAdjustment]
target_surface_pressure = 1e5

[Output]
data_enabled = true
plots_enabled = true
```

## Key Functions
### `choose_config_file(temp_config_path=None)`
- Determines which configuration file to use.
- Supports command-line argument `-c` for specifying config files.

### `main(temp_config_path=None, id_mass=None)`
- Reads parameters from the configuration file.
- Initializes planetary structure and EOS parameters.
- Performs an iterative numerical solution using `solve_ivp`.
- Outputs planetary structure properties.

## Plots
- `plot_planet_profile_single`: Visualizes the internal structure profile.
- `plot_eos_material`: Displays EOS relationships for materials.

TBC

