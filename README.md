# Zalmoxis (Exoplanet Interior Structure Model)

## Overview
This program models the internal structure of exoplanets using an iterative numerical approach. The numerical framework solves a system of coupled ordinary differential equations (ODEs), integrating the equations of hydrostatic equilibrium, mass conservation, and gravity, including realistic material properties. Currently, existing tabulated pressure-density data are used. The model takes as input parameters the planet’s mass and core mass fraction and will integrate the relevant ODEs from the centre to the surface of the planet, checking for convergence at every iteration.The output includes planetary radius, core radius, density profiles, pressure, and temperature distributions.

## Features
- Reads input parameters from a TOML configuration file
- Supports multiple EOS models (Birch-Murnaghan, Mie-Gruneisen-Debye, etc.) but only the Tabulated choice works for now
- Implements an iterative solution for pressure, density, and radius convergence
- Uses `solve_ivp` for numerical integration
- Includes options for outputting data and generating plots

## Documentation

TBC

## Dependencies
Ensure the following Python packages are installed:
```bash
pip install numpy matplotlib scipy toml
```

## Repository Structure

```README.md```
```data```
```docs```
```input```
```setup.py```
```src/tests/MRtest```
```src/zalmoxis```

## Data Download
When running `MRtest.py`, the required data files are automatically downloaded when the script is executed. 
The following data files will be downloaded in a data folder in the main directory.

A brief description of each data file:

```
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
```

## Usage
Run the program via the command line:
```bash
python -m src.zalmoxis.zalmoxis -c input/default.toml
```

Alternatively, specify a different configuration file:
```bash
python -m src.zalmoxis.zalmoxis -c path/to/config.toml
```

## Configuration
The input parameters are stored in the file 'default.toml'. 
TBC

## Key Functions
### `choose_config_file(temp_config_path=None)`
- Determines which configuration file to use.

### `main(temp_config_path=None, id_mass=None)`
- Reads parameters from the configuration file.
- Initializes planetary structure and EOS parameters.
- Performs an iterative numerical solution using `solve_ivp`.
- Outputs planetary structure properties.

## Plots
- `plot_planet_profile_single`: Visualizes the internal structure profile.
- `plot_eos_material`: Displays EOS relationships for materials.
TBC

