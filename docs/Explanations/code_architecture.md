# Code Architecture

## Module overview

```
src/zalmoxis/
├── zalmoxis.py          # Orchestration: config loading, iteration loops, Brent solver, output
├── structure_model.py   # ODE system: coupled_odes(), solve_structure(), terminal events
├── eos_functions.py     # EOS dispatch: calculate_density(), Tdep phase logic, temperature profiles
├── eos_analytic.py      # Analytic modified polytrope: get_analytic_density()
├── eos_properties.py    # EOS registry: material property dicts keyed by EOS identifier
├── mixing.py            # Multi-material mixing: LayerMixture, phase-aware density/nabla_ad
├── melting_curves.py    # Solidus/liquidus curves (Monteux+2016, Stixrude 2014, tabulated)
├── constants.py         # Physical constants (G, earth_mass, earth_radius, etc.)
└── plots/               # Visualization (profile plots, P-T phase diagrams)
```

See the [API reference](../Reference/api/index.md) for a detailed API overview.

## Data flow

```
TOML config
    │
    ▼
parse_eos_config() ──► layer_eos_config dict
    │                   {"core": "Seager2007:iron", "mantle": "PALEOS-2phase:MgSiO3"}
    ▼
main() ─── outer loop (radius) ─── inner loop (density) ─── brentq(_pressure_residual)
    │                                                             │
    │                                                             ▼
    │                                                     solve_structure()
    │                                                             │
    │                                                             ▼
    │                                                     solve_ivp(coupled_odes)
    │                                                             │
    │                                                             ▼
    │                                               get_layer_eos() → calculate_mixed_density()
    │                                                             │
    │                                                  ┌──────────┼──────────┐
    │                                                  ▼          ▼          ▼
    │                                            per-component    sigmoid    harmonic
    │                                            density via      weighting  mean with
    │                                            calculate_       (_condensed effective
    │                                            density()        _weight)   weights
    │                                                             │
    │                                                  ┌──────────┼──────────┐
    │                                                  ▼          ▼          ▼
    │                                            Seager2007   T-dependent  Analytic
    │                                           (tabulated)  (WB2018/     (polytrope)
    │                                                        RTPress/
    │                                                        PALEOS-2phase/
    │                                                        PALEOS unified)
    ▼
post_processing() ──► output files + plots
```

## Key functions

- **`main()`** (`zalmoxis.py`): Orchestrates the three nested convergence loops.
  The innermost loop uses `scipy.optimize.brentq` to find the central pressure root (see [Pressure Solver](process_flow.md#pressure-solver-brents-method)).

- **`_pressure_residual()`** (`zalmoxis.py`, closure inside `main()`): Residual function $f(P_c) = P_{\mathrm{surface}}(P_c) - P_{\mathrm{target}}$ passed to `brentq`.
  Calls `solve_structure()` for each trial $P_c$ and captures the ODE solution via a mutable closure dict.

- **`get_layer_eos()`** (`structure_model.py`): Maps the enclosed mass at a given radial shell to the per-layer EOS string by comparing against core and core+mantle mass thresholds.
  Purely geometric; no EOS-type branching.

- **`coupled_odes()`** (`structure_model.py`): Defines the derivatives $dM/dr$, $dg/dr$, $dP/dr$ for the ODE solver.
  Calls `calculate_density()` at each evaluation to close the system.
  Returns $[0, 0, 0]$ for non-physical states (negative pressure, invalid density) to signal the adaptive solver to reject the step and retry smaller.

- **`solve_structure()`** (`structure_model.py`): Integrates the coupled ODEs across the planetary radius using `scipy.integrate.solve_ivp` (RK45).
  Includes a terminal event that stops integration when pressure crosses zero.
  When any layer uses a temperature-dependent EOS (any entry in `TDEP_EOS_NAMES`), the radial grid is split into two segments for numerical stability near the surface; otherwise, a single integration pass is performed.
  Pads output arrays to `len(radii)` if the terminal event truncates the integration.

- **`calculate_density()`** (`eos_functions.py`): Dispatches the per-layer EOS string to the appropriate density calculation via the `EOS_REGISTRY` dict. Three dispatch paths: unified PALEOS tables (single-file, format `paleos_unified`), T-dependent EOS with separate solid/liquid tables (`WolfBower2018`, `RTPress100TPa`, `PALEOS-2phase`), Seager2007 static tables, or direct evaluation for `Analytic:*` strings.

- **`get_Tdep_density()`** (`eos_functions.py`): Computes mantle density accounting for temperature-dependent phase transitions using the solidus/liquidus melting curves and volume-additive mixing in the mush regime.
  Guards against `None` returns from out-of-bounds table lookups.

- **`calculate_temperature_profile()`** (`eos_functions.py`): Returns a callable $T(r)$ for four modes: `"isothermal"` (uniform), `"linear"` (center-to-surface gradient), `"prescribed"` (loaded from file), or `"adiabatic"` (returns a linear initial guess; the self-consistent adiabat is computed later by `compute_adiabatic_temperature()`). In the ODE solver, the temperature function signature is $T(r, P)$: the $P$ argument is used by the adiabatic mode ($T(P)$ parameterization) and ignored by all other modes.

- **`compute_adiabatic_temperature()`** (`eos_functions.py`): Computes the adiabatic temperature profile by integrating EOS gradients from the surface inward: $T_{i} = T_{i+1} + (dT/dP)_S \cdot \Delta P$. For `WolfBower2018:MgSiO3` and `RTPress100TPa:MgSiO3`, uses pre-tabulated $(dT/dP)_S$ from `adiabat_temp_grad_melt.dat`. For `PALEOS-2phase:MgSiO3`, uses the dimensionless $\nabla_{\mathrm{ad}}$ from the table with phase-aware weighting (solid/mixed/liquid via solidus/liquidus curves). For unified PALEOS tables (`PALEOS:iron`, `PALEOS:MgSiO3`, `PALEOS:H2O`), reads $\nabla_{\mathrm{ad}}$ directly from the single table. With `PALEOS:iron`, the iron core follows its own adiabat instead of being isothermal. In the main convergence loop, the adiabat is parameterized as $T(P)$ (not $T(r)$) to ensure thermodynamic consistency during the Brent pressure solver's bracket search. For T-independent layers (e.g., `Seager2007:iron`), the temperature is held constant.

- **`calculate_mixed_density()`** (`mixing.py`): Computes the suppressed harmonic-mean density for a layer mixture.
  Single-component layers use a fast path that bypasses suppression entirely.
  For multi-component layers, each component's density is evaluated via `calculate_density()`, then weighted by `_condensed_weight()` before entering the harmonic mean.

- **`get_mixed_nabla_ad()`** (`mixing.py`): Computes the sigmoid-weighted average adiabatic gradient for a mixture, using the same suppression as density.

- **`_condensed_weight()`** (`mixing.py`): Sigmoid suppression function $\sigma(\rho) = 1/(1 + \exp(-(\rho - \rho_{\min})/\rho_{\mathrm{scale}}))$.
  Returns ~1 for condensed phases, ~0 for vapor.

- **`LayerMixture`** (`mixing.py`): Dataclass holding a layer's EOS components and mass fractions.
  Supports runtime fraction updates from PROTEUS/CALLIOPE via `update_fractions()`.

- **`parse_eos_config()`** (`zalmoxis.py`): Parses the `[EOS]` TOML section into a per-layer dictionary.
  Handles both the new per-layer format and legacy global-string format via `LEGACY_EOS_MAP`.

- **`validate_layer_eos()`** (`zalmoxis.py`): Validates that all per-layer EOS strings correspond to known tabulated or analytic options.

## Error handling in the ODE system

The adaptive ODE solver (RK45) evaluates the right-hand side at trial points that may be non-physical (e.g., negative pressure).
Two mechanisms prevent crashes:

1. **Zero-derivative guard in `coupled_odes()`:** If the pressure or density is non-physical, the function returns $[0, 0, 0]$ instead of raising an exception.
   This makes the local error estimate large, causing the solver to reject the trial step and retry with a smaller step size.

2. **Terminal event in `solve_structure()`:** A terminal event `_pressure_zero` stops the integration when $P$ crosses zero from above.
   This prevents the solver from wasting time in the zero-derivative region and enables the Brent residual function to detect that $P_c$ was too low.

These two mechanisms work together: the zero-derivative guard handles transient non-physical trial points within valid integrations, while the terminal event handles structurally too-low central pressure guesses.
