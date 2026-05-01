# Coupling to PROTEUS

This page explains *how* and *why* Zalmoxis is integrated into the PROTEUS framework.
For the practical TOML recipe, see the [PROTEUS coupling how-to](../How-to/proteus_coupling.md).

When Zalmoxis runs inside PROTEUS, the coupling is at the **function-call level**, not at the file level:
PROTEUS does not write a Zalmoxis TOML, does not invoke `python -m zalmoxis`, and does not read `output/planet_profile.txt`.
Instead, the wrapper in `proteus/interior_struct/zalmoxis.py` builds a Python dict from the PROTEUS configuration and passes it to `zalmoxis.solver.main()`.
This page documents that dict, where each field comes from, and the loop semantics that make repeated coupled calls give a stable evolution trajectory.

---

## Role of Zalmoxis inside PROTEUS

In a PROTEUS coupled run, three submodules deliver mantle physics:

| Submodule | Role | State variable |
|---|---|---|
| **Zalmoxis** | Static structure: hydrostatic equilibrium, mass-radius, density profile $\rho(r)$, core-mantle boundary $R_\mathrm{cmb}$, $P_\mathrm{cmb}$, gravity $g(r)$ | $\rho(r)$, $g(r)$, $P(r)$ |
| **Aragog** or **SPIDER** | Thermal evolution: entropy ODE, T(r) trajectory, surface heat flux | $S(r)$, $T(r)$ |
| **CALLIOPE** or **atmodeller** | Outgassing: volatile partitioning between magma ocean and atmosphere | $X_i^\mathrm{melt}$, $X_i^\mathrm{atm}$ |

The orchestration in `proteus/proteus.py` calls them in a fixed order per iteration:

```text
                                      ┌──────────────────┐
                                      │   Outgassing     │  CALLIOPE / atmodeller
                                      │   (volatiles)    │  reads phi(r), M_mantle, T_magma
                                      └────────┬─────────┘
                                               │ updates X_i^melt, X_i^atm,
                                               │         dissolved fractions
                                               ▼
                                      ┌──────────────────┐
                                      │    Zalmoxis      │  static structure
                                      │ (gated re-solve) │  reads M_target, T(r), volatile profile
                                      └────────┬─────────┘
                                               │ writes R_int, R_cmb, P_cmb,
                                               │ M_int, M_core, rho(r), g(r),
                                               │ Aragog mesh file
                                               ▼
                                      ┌──────────────────┐
                                      │ Aragog / SPIDER  │  thermal evolution
                                      │  (every step)    │  reads mesh, BCs, advances S/T
                                      └────────┬─────────┘
                                               │ writes T(r), S(r),
                                               │ T_surf, T_cmb, F_atm
                                               ▼
                                      ┌──────────────────┐
                                      │ Atmosphere (AGNI)│  radiative balance
                                      │                  │  reads T_surf, X_i^atm
                                      └──────────────────┘
```

Zalmoxis is the slowest of the four; PROTEUS gates re-solves on physical-state-change triggers rather than calling it every iteration. See [update triggers](#update-triggers) below.

---

## What the wrapper sends to Zalmoxis

The function `load_zalmoxis_configuration(config, hf_row)` in `proteus/interior_struct/zalmoxis.py` builds the dict that `zalmoxis.solver.main()` consumes. The mapping looks like this:

| Zalmoxis dict key | PROTEUS source | Notes |
|---|---|---|
| `planet_mass` | `config.planet.mass_tot * M_earth - sum(volatile masses)` | Subtracts volatile mass so the structure solve sees the dry interior mass. |
| `core_mass_fraction` | `config.interior_struct.core_frac` | Interpretation set by `core_frac_mode`. |
| `core_frac_mode` | `config.interior_struct.core_frac_mode` | `"mass"` or `"radius"`. SPIDER requires `"radius"`. |
| `mantle_mass_fraction` | `config.interior_struct.zalmoxis.mantle_mass_fraction` | 0 for 2-layer + non-T-dep mantle. |
| `temperature_mode` | mapped from `config.planet.temperature_mode` | `"isentropic"` and `"accretion"` collapse to `"adiabatic"`; `"liquidus_super"` becomes `"adiabatic_from_cmb"`. |
| `surface_temperature` | `config.planet.tsurf_init` | Initial guess for the adiabat. |
| `cmb_temperature` | from `config.planet.tcmb_init` *or* derived for `liquidus_super` | See [IC anchors](#initial-condition-anchors). |
| `center_temperature` | `config.planet.tcenter_init` | Initial guess. The solver re-derives self-consistently. |
| `layer_eos_config` | `core_eos`, `mantle_eos`, `ice_layer_eos` | Extended at runtime with volatile EOS (Chabrier:H, PALEOS:H2O) when `dry_mantle = false`. |
| `mushy_zone_factor` | `config.interior_struct.zalmoxis.mushy_zone_factor` | $T_\mathrm{sol} = T_\mathrm{liq} \cdot f$. |
| `num_layers` | `config.interior_struct.zalmoxis.num_levels` | |
| `target_surface_pressure` | `hf_row['P_surf'] * 1e5` *or* fallback from `config.planet.gas_prs` | Atmospheric BC for the structure solve. Floors at 1 atm, caps at 1 GPa on the first call. |
| `outer_solver` | `config.interior_struct.zalmoxis.outer_solver` | Default `"newton"` since T2.3 (2026-04-27). |
| `tolerance_outer`, `tolerance_inner`, `max_iterations_*` | `config.interior_struct.zalmoxis.solver_*` | Newton path auto-overrides `relative_tolerance` to 1e-9. |
| `use_jax`, `use_anderson` | `config.interior_struct.zalmoxis.use_*` | Auto-disabled for calls without a `temperature_function` (init / equilibration). |

This dict is rebuilt every call: PROTEUS does not mutate the shared `Config` object at runtime. Where a temporary override is needed (e.g., SPIDER coupling forcing `adiabatic` mode), the wrapper accepts a `temperature_mode_override` kwarg that takes effect for that single call only.

---

## Initial-condition anchors

A coupled run's first iteration is the hardest. The structure solve and the energetics initial entropy must agree on $T(r)$, $P(r)$, $R_\mathrm{cmb}$ and $M_\mathrm{core}$ before SPIDER or Aragog takes its first time step. PROTEUS supports four IC modes; the most physically grounded for hot magma ocean ICs is `liquidus_super`, added in the `tl/interior-refactor` work.

### `liquidus_super` mode

The CMB temperature is anchored to the **MgSiO$_3$ liquidus at the converged $P_\mathrm{cmb}$**, plus an excess `delta_T_super` representing super-liquidus thermal state:

$$
T_\mathrm{cmb} = T_\mathrm{liq}^\mathrm{Fei2021}(P_\mathrm{cmb}) + \Delta T_\mathrm{super}
$$

where $T_\mathrm{liq}^\mathrm{Fei2021}$ is the Belonoshko+2005 / Fei+2021 piecewise Simon-Glatzel liquidus exposed via `zalmoxis.melting_curves.paleos_liquidus`. Zalmoxis then integrates the adiabat *upward* from $(R_\mathrm{cmb}, T_\mathrm{cmb})$ rather than downward from a surface anchor.

Why this matters: a surface-anchored adiabat gets the deep-mantle $T$ wrong by hundreds of K when the surface $T$ is uncertain (atmospheres at 2000 to 3500 K), but the deep-mantle $T_\mathrm{cmb}$ is well-constrained by the liquidus + a physical excess. Anchoring upward fixes the well-known quantity and lets the surface $T$ float to whatever the radiative balance demands.

### NL20 mass-aware fallback

On the very first call, before Zalmoxis has populated `hf_row['P_cmb']`, the wrapper estimates $P_\mathrm{cmb}$ from a Noack & Lasbleis (2020) mass-aware scaling:

$$
P_\mathrm{cmb}^\mathrm{NL20} = f(M_p, X_\mathrm{CMF}, \mathrm{mode})
$$

This replaces the legacy hardcoded 135 GPa Earth value (PROTEUS commit `6e76832f`). Without the fix, super-Earth ICs were biased by 1500 to 2800 K at iter 0 (the 5 $M_\oplus$ T_cmb anchor went from 6444 K to 9271 K after the fix, a +2827 K correction). After one round-trip through Zalmoxis, the converged $P_\mathrm{cmb}$ replaces the fallback estimate.

### Other modes

| `temperature_mode` | Anchor strategy |
|---|---|
| `"adiabatic"` | Surface-anchored: integrate downward from `tsurf_init`, with `tcenter_init` as initial guess. |
| `"isentropic"` | Same as adiabatic for structure; the energetics IC consumes `ini_entropy` directly. |
| `"accretion"` | White & Li accretion thermal state; structure solved adiabatic, then thermal solver overlays the accretion T(r). |
| `"liquidus_super"` | CMB-anchored as described above. |

---

## Pre-main-loop equilibration

Before the main coupling loop starts, PROTEUS runs a CALLIOPE + Zalmoxis loop in `proteus.interior_energetics.wrapper.equilibrate_initial_state()`:

```text
for i in range(equilibrate_max_iter):
    R_old, P_old = hf_row['R_int'], hf_row['P_surf']
    1. calc_target_elemental_inventories(...)
    2. run_outgassing(dirs, config, hf_row)        # CALLIOPE / atmodeller
    3. zalmoxis_solver(config, outdir, hf_row)     # structure
    4. delta_R = |R_new - R_old| / R_old
       delta_P = |P_new - P_old| / P_old
    if delta_R < tol and delta_P < tol:
        break
```

Why: the first CALLIOPE call needs $M_\mathrm{mantle}$ to solve volatile partitioning, but $M_\mathrm{mantle}$ comes from Zalmoxis. The first Zalmoxis call needs $P_\mathrm{surf}$ and the dissolved-volatile inventory to solve structure, but those come from CALLIOPE. The two converge by iteration; `equilibrate_tol = 0.01` means $R_\mathrm{int}$ and $P_\mathrm{surf}$ each change by less than 1% per iteration.

After equilibration, `generate_spider_tables()` regenerates the SPIDER P-S lookup tables against the final composition, so SPIDER's first time step reads tables consistent with the equilibrated structure.

---

## Update triggers

Re-solving Zalmoxis costs 5 to 60 seconds depending on mass and EOS, which is non-negligible if the coupling loop runs $10^4$ to $10^5$ iterations. PROTEUS gates Zalmoxis re-solves on three conditions, evaluated at every iteration but composed with OR + a time-floor AND:

```text
fire_resolve = (
        (time_since_last >= update_interval)            # ceiling
     OR (|dT_magma| / T_magma >= update_dtmagma_frac)   # composition / thermal change
     OR (|dphi_basic| >= update_dphi_abs)               # phase change
     OR (time_since_successful >= update_stale_ceiling) # T1.5 stale-aware ceiling
) AND (time_since_last >= update_min_interval)          # floor
```

The stale-aware ceiling (`update_stale_ceiling`, default 25 kyr) is the addition most users notice: it forces a re-solve after 25 kyr even if neither of the dynamical triggers fired, because a fall-back path that returned the cached structure can stretch the time-since-call beyond what the physical triggers would have caught. Without staleness, the cached structure could be 100+ kyr stale on a slow run.

The triggers fire even during init-loops, but `equilibrate_initial_state` calls Zalmoxis directly and bypasses the gating logic.

---

## Mesh blending and the `zalmoxis_output.dat` schema

When `interior_energetics.module = "aragog"` and `update_interval > 0`, Zalmoxis writes a fresh Aragog mesh file (`zalmoxis_output.dat`, 5-column TSV: `r`, `g`, `rho`, `P`, `mass_enclosed`) on each re-solve. Aragog reads this file inside `solver.reset()` to rebuild its mass-coordinate mesh.

Two contracts must hold at the file handover boundary:

1. **Top-of-mantle radius equality**: `data[-1, 'r'] == hf_row['R_int']` to relative tolerance 1e-6.
2. **Mantle mass conservation**: $\int 4 \pi r^2 \rho \, dr$ over the mantle = `hf_row['M_int'] - hf_row['M_core']` to relative tolerance 5e-2.

The 5e-2 tolerance is loose because two legitimate noise sources stack:

- The integrator-method gap: Zalmoxis's `mass_enclosed` comes from RK45 with sub-grid substepping, while the schema check re-integrates by trapezoidal shell-sum on the written grid. ~0.8 to 2.0% on stiff CHILI density profiles.
- Mesh blending: when the per-call radius shift exceeds `mesh_max_shift`, `blend_mesh_files` post-modifies `zalmoxis_output.dat` to cap the shift, but does NOT update `hf_row` scalars (T2.5, known latent bug). With $\alpha < 1$ blending, the file's integrated mass drifts up to 5% from the unblended `hf_row['M_int']`.

The tight mass-conservation contract (<0.1%) lives at a different level: in the wrapper's mass-anchor check on `hf_row['M_int'] / hf_row['M_int_target']` (T1.2). The schema check at the file boundary catches column-swap, truncation, and byte-flip corruption, not numerical mass drift.

`validate_zalmoxis_output_schema()` in `proteus/interior_struct/zalmoxis.py` enforces both contracts on every Zalmoxis write.

---

## Volatile profile coupling

When `dry_mantle = false` (default), each Zalmoxis re-solve builds a `VolatileProfile` from the dissolved-volatile masses in `hf_row`:

```python
volatile_profile.w_liquid = {'PALEOS:H2O': X_H2O_liquid, 'Chabrier:H': X_H2_liquid, ...}
volatile_profile.w_solid  = {'PALEOS:H2O': X_H2O_solid,  'Chabrier:H': X_H2_solid,  ...}
```

The wrapper then *extends* the configured mantle EOS string with placeholder fractions for each volatile component:

```text
"PALEOS:MgSiO3"   →   "PALEOS:MgSiO3:0.97+PALEOS:H2O:0.02+Chabrier:H:0.01"
```

Inside Zalmoxis, `LayerMixture` mixes per-component density via the phase-aware suppressed harmonic mean ([mixing](mixing.md)). At each radial shell, the volatile profile re-evaluates the per-phase mass fractions weighted by $\phi(r)$ (melt fraction from the phase routing). This produces a $\phi$-aware structural density that smoothly transitions from a wet liquid mantle to a drier solid mantle as the planet crystallises.

Setting `dry_mantle = true` skips `build_volatile_profile()` entirely. The structure is then determined by the canonical solid + liquid mantle tables alone, regardless of dissolved inventory. Useful for Stage 1 phase-aware coupling diagnostics where you want to isolate the volatile contribution from other physics.

---

## SPIDER P-S table derivation

When `interior_energetics.module = "spider"`, SPIDER reads pressure-entropy lookup tables for density, $\nabla_\mathrm{ad}$, $C_p$, and the phase boundaries. These tables are not shipped: PROTEUS generates them on the fly from the active PALEOS pressure-temperature tables.

The two-stage derivation:

1. `generate_spider_eos_tables` reads the configured PALEOS solid + liquid P-T tables and uses the active liquidus to derive a P-S table by integrating $dS = (C_p / T) dT - \alpha / \rho \, dP$ along isentropes.
2. `generate_spider_phase_boundaries` derives `solidus_P-S.dat` and `liquidus_P-S.dat` from the configured P-T melting file (Monteux+2016, PALEOS-Fei2021, etc.) by inverting $T(P, S) \to S$ at the melting curve.

The single source of truth for melting curves is the PROTEUS-side `[interior_struct].melting_dir`. PROTEUS commit `45ec94f3` (v2.1) closed a previous bookkeeping gap where SPIDER runs were silently using byte-copies of the WB+2018 distribution P-S tables instead of tables derived from the configured P-T file. The resolution `lookup_nP=1350`, `lookup_nS=280` is calibrated against SPIDER's spline tolerance; halving cuts table-gen wall by ~3x but introduces visible interpolation artifacts in the SPIDER adiabat.

---

## Newton outer solver: why it became the default

Zalmoxis's outer mass-radius loop has historically been a damped fixed-point (Picard) iteration: $R_{n+1} = R_n \cdot (M_\mathrm{target} / M_n)^{1/3}$, clamped to $[0.5, 2.0]$ and damped by 0.5. This converges robustly for cool Earth-mass planets but suffers from a basin attractor on hot fully-molten profiles: Newton-Raphson-style oscillations around the true $R$ that damped Picard cannot escape.

The T2.1 work (2026-04-26) added a Newton + brentq outer solver that uses a central-difference $dM/dR$ estimate to step toward the root. T2.3 (2026-04-27) flipped the default from `picard` to `newton` after a 4-mass dry CHILI sweep validated 12/12 G4 starts converging to $\Delta M / M < 1e-4$, including the 3 starts that fail with damped Picard's basin attractor.

Newton requires tighter integrator tolerances than Picard. The default `relative_tolerance = 1e-5` produces ~1e-3 noise in $M(R)$ that swamps Newton's central-difference $dM/dR$ estimate. The wrapper auto-applies `relative_tolerance = 1e-9` and `absolute_tolerance = 1e-10` whenever `outer_solver = "newton"` is selected, only forwarding the Newton block to Zalmoxis when the Newton path is active. Pre-T2.1 Picard runs see a bit-identical config dict.

---

## JAX path and `temperature_arrays`

Zalmoxis has an opt-in JAX + diffrax structure path (`use_jax = true`) plus an Anderson Type-II Picard accelerator (`use_anderson = true`). Both default off for bit-identical reproduction of the numpy path.

The JAX path has a known subtlety in coupled mode. The wrapper detects two argument styles for an external T(r):

| Style | Coupled-mode caller | JAX behaviour |
|---|---|---|
| `temperature_function: f(r, P) -> T` (callable) | `equilibrate_initial_state` and PROTEUS init | JAX RHS path collapses (Zalmoxis `aa3d0b8`): the callable is P-ignoring, which the JAX P-indexed adiabat tabulation cannot interpolate. |
| `temperature_arrays: (r_arr, T_arr)` | Main loop's `update_structure_from_interior` | r-indexed JAX RHS path, no collapse. |

For init / equilibration calls (which only have a callable), the wrapper auto-disables `use_jax` and `use_anderson` and falls back to the numpy path. The one-time cost (~70 s each, 2 to 4 calls) is negligible against a 3 to 4 h full run. For main-loop calls (which have arrays), the JAX path is active.

Additional subtlety: even on the JAX path, do NOT pass a callable when arrays are available. The numpy Picard density update inside `_solve` uses the callable for per-node temperature lookup, which trips the same PALEOS phase-boundary clamps and forces ~75x more inner Picard iterations. Verified 2026-04-24: bench-inside-PROTEUS BENCH-config = 5.9 s, PROTEUS-config-with-function = 156 s, PROTEUS-config-without-function = 2 s.

---

## Determinism and the `--deterministic` flag

The `tl/interior-refactor` Stage 4.4 paper-line work surfaced a class of failures where the same config produced different results across runs: numerical-fragility cases where ~1e-7 floating-point noise in row 6 of the helpfile compounds through Aragog's tight tolerances and lands the solver on a wrong P-S branch by iter 10 to 15.

PROTEUS already pins BLAS thread counts at `cli.py` import time (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`), but BLAS is not the only source of reduction-order non-determinism. JAX/XLA has its own threading model independent of OPENBLAS.

The new `--deterministic` flag (PROTEUS commit `586960a3`) intercepts the argument in raw `sys.argv` *before* any heavy imports, sets `JAX_ENABLE_X64=1` and `XLA_FLAGS=--xla_cpu_enable_fast_math=false`, and self-re-execs once. A sentinel env var prevents infinite re-exec.

In Stage 4.4 falsification tests (2026-04-30), `--deterministic` did not rescue the A2 wet 1 $M_\oplus$ atmodeller anchor, indicating that *that particular* failure has a deeper cause (parallel-sweep-vs-solo launch context). But the flag remains useful infrastructure for any other config showing noise-floor divergence between launches. Solo runs are bit-identical with or without the flag; the flag matters when the JAX layer hits a kernel-dispatch non-determinism that BLAS pinning alone does not catch.

Use sparingly: the flag has a small per-step cost (XLA fast-math disabled). Most coupled runs converge cleanly without it.

---

## Cross-references

### In this site

- [PROTEUS coupling how-to](../How-to/proteus_coupling.md) — TOML recipe and parameter reference.
- [Process flow](process_flow.md) — Zalmoxis's internal three-loop solver.
- [Code architecture](code_architecture.md) — Zalmoxis package layout.
- [Equations of state](eos_physics.md) — PALEOS unified tables and the EOS dispatch model.
- [Multi-material mixing](mixing.md) — phase-aware suppressed harmonic mean used inside the volatile-profile path.

### PROTEUS-side reference (mkdocstrings, single source of truth)

The wrapper itself lives in the PROTEUS repository, not in Zalmoxis. Symbol-level API documentation is rendered directly from the PROTEUS source:

- [`proteus.interior_struct.zalmoxis`](https://proteus-framework.org/PROTEUS/Reference/api/interior_struct_zalmoxis.html) — the wrapper. `zalmoxis_solver`, `load_zalmoxis_configuration`, `validate_zalmoxis_output_schema`, `build_volatile_profile`, and helpers.
- [`proteus.config._struct`](https://proteus-framework.org/PROTEUS/Reference/api/config_struct.html) — the attrs schema for `[interior_struct]` and `[interior_struct.zalmoxis]`. Every TOML key from the [coupling how-to](../How-to/proteus_coupling.md) maps to a field here.
- [`proteus.interior_energetics.wrapper.equilibrate_initial_state`](https://proteus-framework.org/PROTEUS/Reference/api/interior_energetics_wrapper.html) — the pre-main-loop CALLIOPE + Zalmoxis convergence loop.

### External

- [PROTEUS framework documentation](https://proteus-framework.org/PROTEUS).
