# Zalmoxis

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Docs](https://img.shields.io/github/actions/workflow/status/FormingWorlds/Zalmoxis/docs.yml?branch=main&label=Docs)](https://proteus-framework.org/Zalmoxis/)
[![Coverage](https://img.shields.io/codecov/c/github/FormingWorlds/Zalmoxis?label=coverage&logo=codecov)](https://app.codecov.io/gh/FormingWorlds/Zalmoxis)
[![Unit Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/Zalmoxis/CI.yml?branch=main&label=Unit%20Tests)](https://github.com/FormingWorlds/Zalmoxis/actions/workflows/CI.yml)
[![Integration Tests](https://img.shields.io/github/actions/workflow/status/FormingWorlds/Zalmoxis/nightly.yml?branch=main&label=Integration%20Tests)](https://github.com/FormingWorlds/Zalmoxis/actions/workflows/nightly.yml)

**Zalmoxis** is the interior structure solver of the [PROTEUS](https://proteus-framework.org/PROTEUS) coupled atmosphere-interior evolution framework, and also works fully self-consistently as an independent tool for mass-radius modelling and parameter studies.
It resolves the planet from its center to the surface, computing self-consistent radial profiles of density, pressure, temperature, gravity, and phase state for differentiated rocky planets and sub-Neptunes from 0.1 to 50 Earth masses.
Given a total mass, layer composition, and temperature mode, Zalmoxis iteratively solves the coupled hydrostatic equilibrium equations to determine the planet's radius and internal structure up to the interior-atmosphere boundary.

!!! tip "New to Zalmoxis?"
    See the **[Getting Started guide](getting_started.md)** for installation, first run, and basic usage.

## Features

- **Multiple EOS families**: PALEOS unified tables for iron, MgSiO$_3$ and H$_2$O from Attia et al. (2026)[^cite-attia2026], Chabrier H$_2$ for sub-Neptune interiors, Wolf & Bower (2018)[^cite-wolf2018], Seager et al. (2007)[^cite-seager2007], and analytic polytropes
- **Multi-material mixing**: volume-additive harmonic mean with per-component phase-aware suppression
- **H$_2$ miscibility**: binodal suppression models for H$_2$-silicate and H$_2$-H$_2$O phase boundaries
- **Temperature modes**: adiabatic (self-consistent), isothermal, linear, or prescribed profiles
- **Parameter grids**: sweep any combination of input parameters via [TOML grid files](How-to/usage.md)
- **First-principles verified**: 25 tests against exact analytical solutions (uniform/two-layer spheres, Gauss's law, hydrostatic balance, Earth benchmark, mass-radius scaling)

!!! info "PROTEUS framework"
    When used within PROTEUS, Zalmoxis is called at intervals set by its update triggers to update the planetary radius, gravity profile, and density structure.
    For the practical TOML recipe (which `[interior_struct.zalmoxis]` flags matter, IC modes, equilibration, update triggers, and the `--deterministic` flag), see [Coupling to PROTEUS (how-to)](How-to/proteus_coupling.md).
    For the theory (per-iteration control flow, IC anchors, mesh handover, Newton vs Picard rationale, JAX path), see [Coupling to PROTEUS (theory)](Explanations/proteus_coupling.md).
    The PROTEUS-side documentation is at [proteus-framework.org/PROTEUS](https://proteus-framework.org/PROTEUS).

If you plan to contribute to Zalmoxis, please read our [Code of Conduct](Community/CODE_OF_CONDUCT.md) and [contributing guidelines](Community/CONTRIBUTING.md).
If you are running into problems, please do not hesitate to raise an [Issue](https://github.com/FormingWorlds/Zalmoxis/issues).

## License

See [the included license](https://github.com/FormingWorlds/Zalmoxis/blob/main/LICENSE.txt).

[^cite-attia2026]: M. Attia, T. Lichtenberg, E. Jungová, M. Sastre, *[PALEOS: Multiphase equations of state and mass-radius relations for exoplanet interiors](https://doi.org/10.1051/0004-6361/202660790)*, Astronomy and Astrophysics, 713, A225, 2026. [SciX](https://scixplorer.org/abs/2026A%26A...713A.225A/abstract).
[^cite-seager2007]: S. Seager, M. Kuchner, C. A. Hier-Majumder, B. Militzer, *[Mass-radius relationships for solid exoplanets](https://doi.org/10.1086/521346)*, The Astrophysical Journal, 669, 1279-1297, 2007. [SciX](https://scixplorer.org/abs/2007ApJ...669.1279S/abstract).
[^cite-wolf2018]: A. S. Wolf, D. J. Bower, *[An equation of state for high pressure-temperature liquids (RTpress) with application to MgSiO$_3$ melt](https://doi.org/10.1016/j.pepi.2018.02.004)*, Physics of the Earth and Planetary Interiors, 278, 59-74, 2018. [SciX](https://scixplorer.org/abs/2018PEPI..278...59W/abstract).
