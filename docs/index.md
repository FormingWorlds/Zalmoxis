![CI](https://github.com/FormingWorlds/Zalmoxis/actions/workflows/CI.yml/badge.svg)
[![codecov](https://codecov.io/gh/FormingWorlds/Zalmoxis/graph/badge.svg)](https://codecov.io/gh/FormingWorlds/Zalmoxis)

# Zalmoxis

**Zalmoxis** is a standalone exoplanet interior structure model that computes self-consistent radial profiles of density, pressure, temperature, gravity, and phase state for differentiated rocky planets and sub-Neptunes from 0.1 to 50 Earth masses.
Given a total mass, layer composition, and temperature mode, it iteratively solves the coupled hydrostatic equilibrium equations to determine the planet's radius and internal structure.
Zalmoxis works as an independent tool for mass-radius modelling and parameter studies, and also serves as the interior structure module within the [PROTEUS](https://proteus-framework.org/PROTEUS) coupled atmosphere-interior evolution framework.

To get started, see the [Getting Started guide](getting_started.md).

## Features

- **Multiple EOS families**: [PALEOS](https://github.com/maraattia/PALEOS) unified tables (iron, MgSiO3, H2O), Chabrier H2 for sub-Neptune interiors, Wolf & Bower 2018, Seager 2007, and analytic polytropes
- **Multi-material mixing**: volume-additive harmonic mean with per-component phase-aware suppression
- **H2 miscibility**: binodal suppression models for H2-silicate and H2-H2O phase boundaries
- **Temperature modes**: adiabatic (self-consistent), isothermal, linear, or prescribed profiles
- **Parameter grids**: sweep any combination of input parameters via [TOML grid files](How-to/usage.md)
- **First-principles verified**: 25 tests against exact analytical solutions (uniform/two-layer spheres, Gauss's law, hydrostatic balance, Earth benchmark, mass-radius scaling)

!!! info "PROTEUS framework"
    When used within PROTEUS, Zalmoxis is called at every coupling timestep to update the planetary radius, gravity profile, and density structure. The documentation for PROTEUS can be found [here](https://proteus-framework.org/PROTEUS).

If you plan to contribute to Zalmoxis, please read our [Code of Conduct](Community/CODE_OF_CONDUCT.md) and [contributing guidelines](Community/CONTRIBUTING.md).
If you are running into problems, please do not hesitate to raise an [Issue](https://github.com/FormingWorlds/Zalmoxis/issues).

## License

See [the included license](https://github.com/FormingWorlds/Zalmoxis/blob/main/LICENSE.txt).
