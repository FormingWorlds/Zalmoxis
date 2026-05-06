# Code Architecture

This page describes the full Zalmoxis codebase: the source tree, the responsibilities of each module, the three-loop solver, the optional JAX path, and the boundary between Zalmoxis and the rest of the PROTEUS ecosystem.

## Package layout

```
Zalmoxis/
  src/zalmoxis/             # Core package (installed via pip)
    __init__.py             # get_zalmoxis_root() (lazy), __version__
    __main__.py             # CLI entry: python -m zalmoxis -c <config.toml>
    config.py               # TOML parsing, schema validation, EOS/melting setup
    constants.py            # G, earth_mass, earth_radius, TDEP_EOS_NAMES, defaults
    solver.py               # main(): 3-loop solver (Picard or Newton outer)
    structure_model.py      # Hydrostatic ODE system + RK45 / diffrax driver
    output.py               # post_processing(), profile + summary file output
    energetics.py           # Gravitational binding energy, initial T_CMB (White & Li 2025)
    mixing.py               # LayerMixture, multi-component density, sigmoid suppression
    melting_curves.py       # Solidus / liquidus loaders (Stixrude+ etc.)
    binodal.py              # H2-MgSiO3 (Rogers+2025) and H2-H2O (Gupta+2025) miscibility
    eos_analytic.py         # Seager+2007 modified polytrope (6 materials)
    eos_properties.py       # Lazy EOS_REGISTRY (data file paths)
    eos_vinet.py            # Rose-Vinet EOS (Smith+2018, Wicks+2018, Fei+2021)
    eos_export.py           # PALEOS P-T -> SPIDER P-S table conversion
    eos/                    # Numpy-side EOS package, organised by family
      __init__.py           # Re-exports public functions
      interpolation.py      # Grid builders, _fast_bilinear, PALEOS loaders, T-clamping
      seager.py             # Seager2007 1D and 2D tabulated lookups
      paleos.py             # Unified PALEOS density + nabla_ad, mushy zone
      tdep.py               # T-dependent EOS (WolfBower2018, RTPress, PALEOS-2phase)
      dispatch.py           # calculate_density / calculate_density_batch
      temperature.py        # Adiabat (surface- or CMB-anchored), mode dispatch
      output.py             # Pressure/density profile file writing
      paleos_api.py         # Live PALEOS-API table generators (PALEOS-API:*)
      paleos_api_cache.py   # SHA-keyed on-disk cache resolver for PALEOS-API:*
    jax_eos/                # JAX ports of the inner kernels (opt-in, default off)
      __init__.py           # Enables jax_enable_x64 at import
      bilinear.py           # fast_bilinear_jax, paleos_clamp_temperature_jax
      paleos.py             # get_paleos_unified_density_jax (mushy-zone branches)
      tdep.py               # get_Tdep_density_jax (PALEOS-2phase mantle)
      rhs.py                # coupled_odes_jax (jax-traceable RHS)
      solver.py             # diffrax Tsit5 integrator + event-based termination
      wrapper.py            # solve_structure_via_jax (numpy-signature entry)
  tests/                    # 480+ tests (unit, integration, slow markers)
  tools/                    # Standalone scripts
    setup/                  # Test fixtures, data download
    validation/             # First-principles verification
    benchmarks/              # EOS and solver benchmarks
    grids/                  # Parameter sweep runners
    converters/             # EOS format conversion
    plots/                  # Visualisation scripts
  input/                    # TOML configs and grid specs
  data/                     # EOS tables (gitignored, ~600 MB; via tools/setup)
  output/                   # Generated outputs (gitignored)
  docs/                     # Zensical documentation
```

See the [API reference](../Reference/api/index.md) for function-level documentation.

## Module responsibilities

### Top-level package

#### `config.py`

Loads TOML, validates schema (per-layer and legacy global EOS strings, mass and temperature bounds, EOS-specific maxima like `WOLFBOWER2018_MAX_MASS_EARTH = 1.5 M_E`), builds the per-layer mixture dict, and resolves melting curve callables. The validators are strict: a pressure scale, density scale, or composition fraction outside its physical envelope raises before the solver starts. See [`zalmoxis.config`](../Reference/api/zalmoxis.config.md).

#### `solver.py`

Houses `main()` (the user-facing entry), `_solve()` (damped-Picard outer loop, default), `_solve_newton_outer()` (Newton + brentq fallback), `_anderson_mix()` (Walker & Ni 2011 Type-II Anderson acceleration for the density Picard loop), and the mass-adaptive `_default_solver_params` / `_tighten_solver_params` helpers used by the auto-retry path. See [`zalmoxis.solver`](../Reference/api/zalmoxis.solver.md).

#### `structure_model.py`

Defines the hydrostatic ODE system $[dM/dr,\; dg/dr,\; dP/dr]$, the layer dispatch helper `get_layer_mixture()`, and `solve_structure()`, which integrates the system from surface to centre via `scipy.integrate.solve_ivp` (RK45) with a terminal pressure-zero event. When `use_jax=True` it forwards to `jax_eos.wrapper.solve_structure_via_jax` instead. See [`zalmoxis.structure_model`](../Reference/api/zalmoxis.structure_model.md).

#### `output.py`

`post_processing()` orchestrates a full run: it calls `solver.main()`, writes radial profiles and mass-radius scalars to the run directory, and (lazily) imports matplotlib only when plotting is enabled. See [`zalmoxis.output`](../Reference/api/zalmoxis.output.md).

#### `energetics.py`

Computes gravitational binding energies (full integrand and the uniform-sphere reference $U = 3GM^2/5R$), the differentiation energy $U_d - U_u$, and the initial CMB temperature following White & Li (2025) Eq. 2: $T_{\mathrm{CMB}} = T_{\mathrm{eq}} + \Delta T_G + \Delta T_D + \Delta T_{\mathrm{ad}}$. Used by PROTEUS to seed a self-consistent post-accretion thermal state. See [`zalmoxis.energetics`](../Reference/api/zalmoxis.energetics.md).

#### `mixing.py`

`LayerMixture` (a frozen dataclass of EOS components and their mass fractions) plus the multi-component closure: per-component densities are computed independently, weighted by mass fraction, and combined via harmonic mean (volume-additive). Two physical suppression mechanisms run on top: a per-component sigmoid that suppresses light volatile components below their critical density (auto-overrides `Chabrier:H` to a centre of 30 kg/m$^3$ and `PALEOS:H2O` to 322 kg/m$^3$), and a binodal weight from `binodal.py` that suppresses miscible H$_2$ above the Rogers+2025 / Gupta+2025 binodal temperature. See [`zalmoxis.mixing`](../Reference/api/zalmoxis.mixing.md).

#### `binodal.py`

H$_2$-silicate (Rogers et al. 2025) and H$_2$-H$_2$O (Gupta et al. 2025) miscibility models. Returns sigmoid suppression weights smoothly switched around the binodal temperature. See [`zalmoxis.binodal`](../Reference/api/zalmoxis.binodal.md).

#### `melting_curves.py`

Solidus and liquidus interpolators for EOS that need external melting curves (WolfBower2018, RTPress100TPa). PALEOS unified tables encode their melting boundary in the phase column and do not load these. See [`zalmoxis.melting_curves`](../Reference/api/zalmoxis.melting_curves.md).

#### `eos_analytic.py`

The Seager et al. (2007) modified polytrope $\rho(P) = \rho_0 + c P^n$ for six materials (Fe, MgSiO$_3$, H$_2$O, two H$_2$O-ice variants, and a low-density envelope). Cheap, P-only, used as an initial guess and as a fallback. See [`zalmoxis.eos_analytic`](../Reference/api/zalmoxis.eos_analytic.md).

#### `eos_properties.py`

The `EOS_REGISTRY` dict. Wrapped in `_LazyRegistry` so that file paths under `$ZALMOXIS_ROOT/data/` are constructed on first access, not at import time. This is what allows `import zalmoxis` to succeed even without EOS data on disk (needed for unit tests with mocked EOS). See [`zalmoxis.eos_properties`](../Reference/api/zalmoxis.eos_properties.md).

#### `eos_vinet.py`

Rose-Vinet EOS, $P(f) = 3 K_0 f^{-2}(1 - f)\exp[\eta(1 - f)]$ with $\eta = \tfrac{3}{2}(K_0' - 1)$, inverted to $\rho(P)$ by Brent root-finding. Material parameters from Smith+2018, Wicks+2018, Fei+2021, Sakai+2016, Stixrude & Lithgow-Bertelloni 2005. See [`zalmoxis.eos_vinet`](../Reference/api/zalmoxis.eos_vinet.md).

#### `eos_export.py`

Converts PALEOS P-T tables into SPIDER P-S format (the entropy-pressure coordinates SPIDER expects) and Aragog P-T format. Produces phase boundaries, full 2D density / temperature / heat capacity / thermal expansion / adiabatic gradient grids, and a surface-entropy anchor. Splits solid and melt into separate files so SPIDER's mushy-zone mixing has phase-pure endpoints. See [`zalmoxis.eos_export`](../Reference/api/zalmoxis.eos_export.md).

### `eos/` subpackage (numpy)

| Module | Purpose | Reference |
|---|---|---|
| `interpolation.py` | Table loaders, `_fast_bilinear` O(1) lookup on log-uniform grids, per-cell PALEOS T-clamping | [API](../Reference/api/zalmoxis.eos.interpolation.md) |
| `seager.py` | Seager2007 1D $\rho(P)$ table reader | [API](../Reference/api/zalmoxis.eos.seager.md) |
| `paleos.py` | `get_paleos_unified_density` and `_get_paleos_unified_nabla_ad`, mushy-zone blending across the phase column | [API](../Reference/api/zalmoxis.eos.paleos.md) |
| `tdep.py` | T-dependent EOS dispatch (WolfBower2018, RTPress100TPa, PALEOS-2phase), melting-curve helpers | [API](../Reference/api/zalmoxis.eos.tdep.md) |
| `dispatch.py` | `calculate_density` / `calculate_density_batch` (the public entry points) | [API](../Reference/api/zalmoxis.eos.dispatch.md) |
| `temperature.py` | `compute_adiabatic_temperature` (surface- or CMB-anchored) and the mode dispatcher `calculate_temperature_profile` | [API](../Reference/api/zalmoxis.eos.temperature.md) |
| `output.py` | Pressure/density profile file writer | [API](../Reference/api/zalmoxis.eos.output.md) |
| `paleos_api.py` | Live PALEOS-API producers (write `.dat` files in PALEOS format) | [API](../Reference/api/zalmoxis.eos.paleos_api.md) |
| `paleos_api_cache.py` | SHA-keyed cache resolver under `$ZALMOXIS_ROOT/data/EOS_PALEOS_API/` | [API](../Reference/api/zalmoxis.eos.paleos_api_cache.md) |

Public functions are re-exported by `eos/__init__.py`, so `from zalmoxis.eos import calculate_density` works.

### `jax_eos/` subpackage (JAX, opt-in)

A line-by-line port of the performance-critical kernels to `jax.numpy` so the inner loop can be JIT-compiled and integrated through `diffrax`. Off by default; enabled per call via `config_params['use_jax'] = True`. See the [JAX path API page](../Reference/api/zalmoxis.jax_eos.md) and the [JAX section below](#jax-path).

## Solver architecture (three nested loops)

The solver implements three nested iterations. Read inside-out: the inner ODE produces $P(r)$ given a density profile; the middle Picard loop drives that profile to self-consistency with the EOS; the outer loop drives the radius to satisfy the mass constraint.

```
main() -- outer: mass-radius (Picard or Newton)
   |       solver.py:main, _solve, _solve_newton_outer
   |
   +-- middle: density Picard with optional Anderson Type-II
   |     solver.py:_solve loop body, _anderson_mix
   |
   +-- innermost: structure ODE
         structure_model.py:solve_structure
         scipy.integrate.solve_ivp(RK45)  -or-  diffrax.diffeqsolve(Tsit5)
         brentq on central pressure for P(R) = P_target
```

### Outermost: mass-radius

Two implementations, both convergent on the same fixed point.

**Picard (default, `outer_solver = 'picard'`)**: at each iteration, solve the structure for the current $R$, measure $M(R)$, and update via damped cube-root scaling clamped to $[0.5, 2.0]$. Robust on Earth-like and rocky cases; cheap per iteration.

**Newton + brentq fallback (`outer_solver = 'newton'`)**: implemented in `_solve_newton_outer` (`solver.py:751`). Recommended on hot fully-molten profiles ($T_{\mathrm{surf}} > 3000$ K) and on super-Earth scales ($M_p > 2 M_\oplus$), where damped Picard has a known basin attractor and the outer R-search oscillates without converging. The advisory at `solver.py:285` logs a one-line INFO suggestion when it detects this regime under the default Picard setting.

!!! note "Default is Picard standalone, Newton when run from PROTEUS"
    The Zalmoxis-side default is `'picard'` (`solver.py:165`) because Newton requires tighter integrator tolerances. For Earth-like configurations Picard converges in a handful of iterations. For hot or massive cases, set `outer_solver = 'newton'` in your TOML. The PROTEUS-side schema (`proteus.config._struct.Zalmoxis`) overrides this default to `'newton'` because PROTEUS-coupled runs routinely include hot fully-molten profiles and super-Earth scales where Picard hits a known basin attractor; see [Coupling to PROTEUS](proteus_coupling.md#outer-solver-and-jax-path).

### Middle: density Picard

Inside a fixed $R$, alternate between integrating the structure ODE (yielding $P(r)$) and re-evaluating $\rho(P, T)$ from the EOS. The trivial update is the damped fixed point $\rho_{k+1} = \alpha\,\rho_{\mathrm{EOS}}(P_k, T_k) + (1-\alpha)\,\rho_k$ with $\alpha = 0.5$. With `use_anderson = True`, `_anderson_mix` (`solver.py:59-118`) accelerates by least-squares-combining the residual history (Walker & Ni 2011, Type-II form, default window $m_{\max} = 5$). On a singular Anderson matrix or non-finite output the helper returns `None` and the caller falls back to damped Picard for that step.

### Innermost: structure ODE

`solve_structure` integrates the coupled ODEs from surface to centre with adaptive RK45 (or Tsit5 on the JAX path). A terminal event at $P = 0$ stops the integration cleanly when the central-pressure guess is too low. The central pressure $P_c$ that satisfies $P(R) = P_{\mathrm{target}}$ is found by `scipy.optimize.brentq` over a guarded bracket. `coupled_odes()` returns $[0, 0, 0]$ at non-physical trial points so the adaptive controller rejects the step rather than crashing.

### Adiabat anchor: surface vs CMB

`compute_adiabatic_temperature` in `eos/temperature.py` supports two anchor points (`anchor='surface'` or `anchor='cmb'`).

- **Surface-anchored** (default, historic). Anchors $T(R) = T_{\mathrm{surf}}$ and integrates inward. This is the appropriate boundary condition when only the surface temperature is known.
- **CMB-anchored** (new, energetics coupling). Anchors $T(r_{\mathrm{cmb}}) = T_{\mathrm{cmb}}$ at the first mantle shell and integrates outward through the mantle only; shells below the CMB carry the surface anchor through (the core adiabat is decoupled, and the energetics solver computes $T_{\mathrm{core}}$ from `core_heatcap` and the Bower+2018 ratio). This is the path used when an interior energy budget already constrains $T_{\mathrm{cmb}}$, for example when Zalmoxis is downstream of Aragog or SPIDER.

## JAX path

The `jax_eos/` subpackage is a parallel implementation of the inner kernels in `jax.numpy`, glued together by `diffrax` (Tsit5 integrator) and JIT-compiled into a single XLA kernel. It is **off by default**; enable it by setting `use_jax = true` in your config (or `config_params['use_jax'] = True` programmatically). When off, `structure_model.solve_structure` runs the numpy path unchanged and the JAX modules are not imported.

What it specialises:

- The bilinear inner kernel (`fast_bilinear_jax`) and PALEOS T-clamp.
- The PALEOS unified density (`get_paleos_unified_density_jax`) including the five mushy-zone branches.
- The PALEOS-2phase mantle density (`get_Tdep_density_jax`, solid + liquid sub-tables, volume-average mushy mix).
- The coupled RHS (`coupled_odes_jax`) and the diffrax-driven integration with an event-based pressure-zero terminator.

Scope today is the Stage-1b two-layer config (PALEOS:iron core + PALEOS-2phase:MgSiO$_3$ mantle). Anything outside that envelope (3-layer ice, multi-component layers, non-Stixrude14 melting) automatically falls back to the numpy path at the caller (`solver._solve`).

**Float64 enforcement.** `jax_eos/__init__.py` calls `jax.config.update('jax_enable_x64', True)` at import. Without this, JAX defaults to float32 and downstream density / pressure / temperature accumulate ~$10^{-7}$ relative error, far above the parity tolerance.

**Parity guarantee.** The JAX kernels match their numpy reference to FP-rounding precision (rtol $\leq 10^{-12}$ on bilinear, rtol $\leq 10^{-6}$ on the full RHS, end-to-end rtol $\leq 10^{-5}$ at solver tolerance). Tests in `tests/test_jax_*_parity.py`.

**SPIDER-coupling specialisation.** `jax_eos/wrapper.py:123-260` accepts both `temperature_function` (callable in $r, P$) and `temperature_arrays = (r_arr, T_arr)` (an explicit r-indexed profile). The latter is preferred when the caller's $T$ is naturally indexed by radius (e.g. SPIDER or Aragog supplying $T(r)$): the P-indexed adiabat tabulation inside the wrapper would collapse to a constant for P-ignoring callables and break convergence.

## Performance hotspots

| Hotspot | Location | Why it matters |
|---|---|---|
| Anderson Type-II acceleration | `solver.py:59-118` | Cuts the density Picard loop from ~20 to ~5 iterations on hot profiles |
| Lazy PALEOS table loader | `eos/interpolation.py:158`, `eos/paleos_api_cache.py` | First-touch only; SHA-validated cache avoids re-parsing 100 MB+ text tables |
| In-table PALEOS T-clamping | `eos/interpolation.py:481-540` | O(1) per-cell index on the log-uniform grid; replaces `np.interp` in the inner loop |
| Bilinear kernel | `eos/interpolation.py:349-417`, `jax_eos/bilinear.py` | Called millions of times; the JAX port specialises this for JIT |
| Module-level interpolation cache | `solver.py:_interpolation_cache` | Persists across `main()` calls in the same process (PROTEUS coupling loop) |

## What is NOT in Zalmoxis

Zalmoxis is a **structure** code, not an evolution code. It returns a self-consistent $\rho(r), g(r), P(r), T(r)$ given $(M_p, T_{\mathrm{surf}}$ or $T_{\mathrm{CMB}},$ composition$)$. It does not advance time. The following live elsewhere in the PROTEUS stack:

- **Mantle-interior energy budget and time evolution**: Aragog (the JAX-native successor) and SPIDER (the legacy C/Python implementation).
- **Volatile partitioning, outgassing, dissolution**: CALLIOPE.
- **Atmospheric radiative transfer and surface flux**: JANUS (clear-sky) and AGNI (RT with clouds).
- **Tidal dissipation, orbital evolution**: handled by PROTEUS' tides module.
- **Atmospheric escape**: ZEPHYRUS.

Zalmoxis only ever produces a snapshot. The coupling layer in PROTEUS (`proteus/interior/zalmoxis.py`) calls `zalmoxis.solver.main()` once per coupled iteration, feeding it the current $T$ profile from Aragog or SPIDER and reading back the updated $\rho(r), g(r), P(r), R_p$.

## See also

- [PROTEUS coupling](proteus_coupling.md): how Zalmoxis is invoked from inside PROTEUS, the in-memory T-profile path, and the structure-update interval.
- [Model](model.md): physical assumptions, EOS hierarchy, mixing and melting choices.
- [EOS physics](eos_physics.md): what each EOS family actually computes and where its tables come from.
- [Process flow](process_flow.md): the exact sequence of calls from `python -m zalmoxis` through to file output.
