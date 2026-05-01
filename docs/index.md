![CI](https://github.com/FormingWorlds/Zalmoxis/actions/workflows/CI.yml/badge.svg)
[![codecov](https://codecov.io/gh/FormingWorlds/Zalmoxis/graph/badge.svg)](https://codecov.io/gh/FormingWorlds/Zalmoxis)

# Zalmoxis

**Zalmoxis** is the interior structure solver of the [PROTEUS](https://proteus-framework.org/PROTEUS) coupled atmosphere-interior evolution framework, and also works fully self-consistently as an independent tool for mass-radius modelling and parameter studies.
It resolves the planet from its center to the surface, computing self-consistent radial profiles of density, pressure, temperature, gravity, and phase state for differentiated rocky planets and sub-Neptunes from 0.1 to 50 Earth masses.
Given a total mass, layer composition, and temperature mode, Zalmoxis iteratively solves the coupled hydrostatic equilibrium equations to determine the planet's radius and internal structure up to the interior-atmosphere boundary.

!!! tip "New to Zalmoxis?"
    See the **[Getting Started guide](getting_started.md)** for installation, first run, and basic usage.

## Features

- **Multiple EOS families**: [PALEOS](https://github.com/maraattia/PALEOS) unified tables (iron, MgSiO$_3$, H$_2$O), Chabrier H$_2$ for sub-Neptune interiors, Wolf & Bower 2018, Seager 2007, and analytic polytropes
- **Multi-material mixing**: volume-additive harmonic mean with per-component phase-aware suppression
- **H$_2$ miscibility**: binodal suppression models for H$_2$-silicate and H$_2$-H$_2$O phase boundaries
- **Temperature modes**: adiabatic (self-consistent), isothermal, linear, or prescribed profiles
- **Parameter grids**: sweep any combination of input parameters via [TOML grid files](How-to/usage.md)
- **First-principles verified**: 25 tests against exact analytical solutions (uniform/two-layer spheres, Gauss's law, hydrostatic balance, Earth benchmark, mass-radius scaling)

!!! info "PROTEUS framework"
    When used within PROTEUS, Zalmoxis is called at gated intervals to update the planetary radius, gravity profile, and density structure.
    For the practical TOML recipe (which `[interior_struct.zalmoxis]` flags matter, IC modes, equilibration, update triggers, and the `--deterministic` flag), see [Coupling to PROTEUS (how-to)](How-to/proteus_coupling.md).
    For the theory (per-iteration control flow, IC anchors, mesh handover, Newton vs Picard rationale, JAX path), see [Coupling to PROTEUS (theory)](Explanations/proteus_coupling.md).
    The PROTEUS-side documentation is at [proteus-framework.org/PROTEUS](https://proteus-framework.org/PROTEUS).

If you plan to contribute to Zalmoxis, please read our [Code of Conduct](Community/CODE_OF_CONDUCT.md) and [contributing guidelines](Community/CONTRIBUTING.md).
If you are running into problems, please do not hesitate to raise an [Issue](https://github.com/FormingWorlds/Zalmoxis/issues).

## License

See [the included license](https://github.com/FormingWorlds/Zalmoxis/blob/main/LICENSE.txt).
