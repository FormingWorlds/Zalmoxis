# Code Architecture

## Repository layout

```
Zalmoxis/
├── src/zalmoxis/             # Core package (installed via pip)
│   ├── __init__.py           # get_zalmoxis_root() (lazy), __version__
│   ├── config.py             # Config parsing, validation, EOS/melting setup
│   ├── solver.py             # main() solver loop (3 nested iterations)
│   ├── output.py             # post_processing(), file output
│   ├── structure_model.py    # ODE system (dM/dr, dg/dr, dP/dr), solve_structure()
│   ├── eos/                  # EOS package, organized by family
│   │   ├── __init__.py       # Re-exports all public functions
│   │   ├── interpolation.py  # Shared grid builders, bilinear interp, table loaders
│   │   ├── seager.py         # Seager2007 tabulated 1D P-rho lookups
│   │   ├── paleos.py         # Unified PALEOS density + nabla_ad, mushy zone
│   │   ├── tdep.py           # T-dependent EOS, melting curves, phase routing
│   │   ├── dispatch.py       # calculate_density/batch (main entry points)
│   │   ├── temperature.py    # Adiabat computation, T profiles
│   │   └── output.py         # Pressure/density profile file writing
│   ├── eos_analytic.py       # Seager+2007 analytic polytrope (6 materials)
│   ├── eos_properties.py     # Lazy EOS_REGISTRY (paths built on first access)
│   ├── eos_export.py         # P-S EOS table generation for SPIDER/Aragog
│   ├── mixing.py             # Multi-component mixing, LayerMixture, suppression
│   ├── melting_curves.py     # Solidus/liquidus functions
│   ├── binodal.py            # H2-MgSiO3 and H2-H2O miscibility models
│   └── constants.py          # Physical constants (G, earth_mass, earth_radius)
├── tests/                    # Test suite
├── tools/                    # Standalone scripts
│   ├── setup/                # Test fixtures, data download
│   ├── validation/           # First-principles verification
│   ├── benchmarks/           # EOS benchmarks
│   ├── grids/                # Parameter sweep runners
│   ├── converters/           # EOS format conversion
│   └── plots/                # Visualization scripts
├── input/                    # TOML configs and grid specs
├── data/                     # EOS tables (gitignored, ~600 MB)
├── output/                   # Generated outputs (gitignored)
└── docs/                     # Zensical documentation
```

See the [API reference](../Reference/api/index.md) for detailed function-level documentation.

## Package modules

The core package (`src/zalmoxis/`) is organized by responsibility:

### Configuration (`config.py`)

Handles TOML configuration loading, EOS config parsing (new per-layer format and legacy global-string format), comprehensive parameter validation, material dictionary loading, and melting curve setup.

Key functions:

- **`load_zalmoxis_config()`**: Reads a TOML file and returns a validated config dict.
- **`parse_eos_config()`**: Converts the `[EOS]` section into a per-layer dictionary.
- **`validate_config()`**: Validates all parameters for physical and logical consistency.
- **`load_material_dictionaries()`**: Returns the `EOS_REGISTRY` dict.
- **`load_solidus_liquidus_functions()`**: Returns melting curve callables if needed by the EOS.

### Solver (`solver.py`)

Contains the `main()` function that implements the three nested iterative procedures:

1. **Outermost: Mass-radius convergence.** Adjusts the planet radius until $M(R) = M_p$ using damped cube-root scaling clamped to $[0.5, 2.0]$.
2. **Middle: Density Picard iteration.** Alternates between solving the ODE (for new P) and evaluating the EOS (for new $\rho$), using 0.5 damping.
3. **Innermost: Brent root-finding.** Finds the central pressure $P_c$ such that $P(R) = P_{\mathrm{target}}$ via `scipy.optimize.brentq`.

Each Brent evaluation calls `solve_structure()` from `structure_model.py` to integrate the hydrostatic equilibrium ODEs.

### Output (`output.py`)

The `post_processing()` function runs the solver, writes radial profile data and mass-radius results to files, and optionally generates plots. Plot imports are deferred (inside the `if plotting_enabled:` block) so the core package does not depend on matplotlib at import time.

### Structure model (`structure_model.py`)

Defines the coupled ODEs and integrates them:

- **`coupled_odes()`**: Returns $[dM/dr, dg/dr, dP/dr]$ for a given $(r, M, g, P)$ state. Calls `calculate_mixed_density()` for the density closure. Returns $[0, 0, 0]$ for non-physical states.
- **`solve_structure()`**: Integrates the ODEs via `scipy.integrate.solve_ivp` (RK45) with a terminal event at $P = 0$.
- **`get_layer_mixture()`**: Maps enclosed mass to the per-layer EOS by comparing against core and core+mantle mass thresholds.

### EOS package (`eos/`)

The equation-of-state code is organized by EOS family:

| Module | Purpose |
|--------|---------|
| `eos/interpolation.py` | Shared grid-building, `_fast_bilinear()` O(1) lookup, PALEOS table loaders, temperature clamping |
| `eos/seager.py` | Seager2007 tabulated 1D $\rho(P)$ lookups |
| `eos/paleos.py` | Unified PALEOS density and $\nabla_{\mathrm{ad}}$, mushy zone interpolation |
| `eos/tdep.py` | Temperature-dependent EOS (WolfBower2018, RTPress, PALEOS-2phase), melting curves |
| `eos/dispatch.py` | `calculate_density()` and `calculate_density_batch()` entry points that dispatch to the right EOS |
| `eos/temperature.py` | Adiabatic temperature profiles, $dT/dP$ computation |
| `eos/output.py` | Pressure/density profile file writing |

All public functions are re-exported via `eos/__init__.py`, so `from zalmoxis.eos import calculate_density` works.

### Other modules

| Module | Purpose |
|--------|---------|
| `eos_analytic.py` | Seager+2007 analytic modified polytrope: $\rho(P) = \rho_0 + c P^n$ for 6 materials |
| `eos_properties.py` | Lazy `EOS_REGISTRY` mapping EOS identifiers to data file paths (built on first access) |
| `eos_export.py` | Generates EOS tables in formats required by SPIDER and Aragog |
| `mixing.py` | `LayerMixture` dataclass, multi-material harmonic-mean density with phase-aware suppression |
| `binodal.py` | H2-silicate and H2-H2O miscibility models (Rogers+2025, Gupta+2025) |
| `melting_curves.py` | Solidus/liquidus curve loading and interpolation |
| `constants.py` | Physical constants (`G`, `earth_mass`, `earth_radius`, `TDEP_EOS_NAMES`) |

## Data flow

```
TOML config
    │
    ▼
load_zalmoxis_config() ──► config dict (config.py)
    │
    ▼
main() ─── outer loop (radius) ─── inner loop (density) ─── brentq(_pressure_residual)
    │       (solver.py)                                              │
    │                                                                ▼
    │                                                        solve_structure()
    │                                                        (structure_model.py)
    │                                                                │
    │                                                                ▼
    │                                                        solve_ivp(coupled_odes)
    │                                                                │
    │                                                                ▼
    │                                               get_layer_mixture() → calculate_mixed_density()
    │                                                                │         (mixing.py)
    │                                                     ┌──────────┼──────────┐
    │                                                     ▼          ▼          ▼
    │                                               per-component    sigmoid    harmonic
    │                                               density via      weighting  mean with
    │                                               calculate_       (_condensed effective
    │                                               density()        _weight)   weights
    │                                               (eos/dispatch.py)
    │                                                     │
    │                                          ┌──────────┼──────────┐
    │                                          ▼          ▼          ▼
    │                                    Seager2007   T-dependent  Analytic
    │                                   (eos/        (eos/tdep.py (eos_
    │                                    seager.py)   eos/         analytic.py)
    │                                                 paleos.py)
    ▼
post_processing() ──► output files + plots (output.py)
```

## Lazy initialization

Two key resources are initialized lazily (on first access, not at import time):

1. **`ZALMOXIS_ROOT`**: Resolved by `get_zalmoxis_root()` in `__init__.py`. Auto-detects from the package file path; falls back to the `ZALMOXIS_ROOT` environment variable. This allows the package to be imported without the env var being set (required for unit tests that mock the EOS).

2. **`EOS_REGISTRY`**: Data file paths in `eos_properties.py` are constructed on first dict access via a `_LazyRegistry` wrapper, not at module load time. This prevents import-time crashes when EOS data files are not yet downloaded.

## Error handling in the ODE system

The adaptive ODE solver (RK45) evaluates the right-hand side at trial points that may be non-physical (e.g., negative pressure).
Two mechanisms prevent crashes:

1. **Zero-derivative guard in `coupled_odes()`:** If the pressure or density is non-physical, the function returns $[0, 0, 0]$ instead of raising an exception.
   This makes the local error estimate large, causing the solver to reject the trial step and retry with a smaller step size.

2. **Terminal event in `solve_structure()`:** A terminal event `_pressure_zero` stops the integration when $P$ crosses zero from above.
   This prevents the solver from wasting time in the zero-derivative region and enables the Brent residual function to detect that $P_c$ was too low.

These two mechanisms work together: the zero-derivative guard handles transient non-physical trial points within valid integrations, while the terminal event handles structurally too-low central pressure guesses.
