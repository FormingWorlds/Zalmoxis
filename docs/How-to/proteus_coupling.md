# Coupling Zalmoxis to PROTEUS

This page is the practical recipe for running Zalmoxis as the interior-structure module inside the PROTEUS framework.
The standalone `python -m zalmoxis -c input/default.toml` workflow described in [Usage](usage.md) does *not* apply when Zalmoxis is driven from PROTEUS.
PROTEUS bypasses Zalmoxis's own TOML sections (`[InputParameter]`, `[AssumptionsAndInitialGuesses]`, `[EOS]`, ...) entirely and builds the call-time configuration from its own `[interior_struct]` block.

For the theory behind this design, see [Coupling to PROTEUS (theory)](../Explanations/proteus_coupling.md).

!!! info "Audience"
    This guide assumes you already have a working PROTEUS install (see the [PROTEUS docs](https://proteus-framework.org/PROTEUS)) and a CHILI-style coupled config for atmosphere + interior + outgassing.
    If you are running Zalmoxis on its own, ignore this page and follow [Usage](usage.md).

---

## Minimal coupling block

In a PROTEUS TOML config, set Zalmoxis as the structure module with:

```toml
[interior_struct]
module          = "zalmoxis"
core_frac       = 0.325        # core mass fraction (0 < x < 1)
core_frac_mode  = "mass"       # "mass" or "radius"
core_density    = "self"       # let Zalmoxis compute self-consistently
core_heatcap    = "self"

[interior_struct.zalmoxis]
core_eos        = "PALEOS:iron"
mantle_eos      = "PALEOS:MgSiO3"
ice_layer_eos   = "none"       # "none" for 2-layer, or e.g. "PALEOS:H2O"
mushy_zone_factor = 0.8        # cryoscopic depression in [0.7, 1.0]
mantle_mass_fraction = 0.0     # 0 for 2-layer, > 0 for 3-layer
num_levels      = 150
```

This is sufficient for an Earth-mass coupled CHILI run. The other `[interior_struct.zalmoxis]` keys default to validated production values; tune them only if you hit a specific issue.

!!! tip "Picking EOS"
    For paper-quality runs, use `PALEOS:iron` + `PALEOS:MgSiO3` (unified tables, $T$-dependent core, valid to 50 $M_\oplus$).
    The legacy `Seager2007:*` and `WolfBower2018:MgSiO3` choices are still supported for sensitivity tests and SPIDER parity comparisons.
    See [Configuration](configuration.md#available-eos-options) for the full table; the same identifiers apply in coupled mode.

---

## Initial-condition mode (`temperature_mode`)

Inside PROTEUS, the IC selection is governed by `[planet].temperature_mode`, **not** the Zalmoxis-side `[AssumptionsAndInitialGuesses]` block. PROTEUS translates the planet-side mode into the Zalmoxis-side structure-solve mode at call time.

| `[planet].temperature_mode` | Zalmoxis structure-solve mode | When to use |
|---|---|---|
| `"adiabatic"` | `adiabatic` (surface-anchored) | Generic. Pairs with `[planet].tsurf_init` and `[planet].tcenter_init`. |
| `"isentropic"` | `adiabatic` (mapped) | CHILI protocol. Energetics consumes `ini_entropy`; Zalmoxis T-profile is used only for the structure solve. |
| `"accretion"` | `adiabatic` (mapped) | White & Li accretion thermal state. Energetics computes its own T(r) post-structure. |
| `"liquidus_super"` | `adiabatic_from_cmb` (CMB-anchored) | Hot magma-ocean ICs. Anchors $T_\mathrm{cmb} = T_\mathrm{liq}(P_\mathrm{cmb}) + \Delta T_\mathrm{super}$ from the Fei+2021 liquidus. Recommended for Stage 4+ super-Earth runs. |
| `"isothermal"`, `"linear"`, `"prescribed"` | unchanged | Diagnostics only. Not for production coupled runs. |

For `liquidus_super`, set `[planet].delta_T_super` (typically 100 to 500 K). On the very first call, before Zalmoxis has populated `hf_row['P_cmb']`, the wrapper falls back to a Noack & Lasbleis (2020) mass-aware $P_\mathrm{cmb}$ estimate so super-Earth runs anchor correctly instead of locking to the legacy 135 GPa Earth value. After one round-trip the fallback is replaced by the converged Zalmoxis $P_\mathrm{cmb}$.

```toml
[planet]
mass_tot            = 5.0
tsurf_init          = 3000
tcenter_init        = 9000
temperature_mode    = "liquidus_super"
delta_T_super       = 200
```

See [the IC anchors theory section](../Explanations/proteus_coupling.md#initial-condition-anchors) for the derivation.

---

## Outer mass-radius solver (`outer_solver`)

Zalmoxis's outer loop adjusts the planet radius until $M(R) = M_\mathrm{target}$. Two solvers are available; the PROTEUS default flipped to **Newton** in T2.3 (2026-04-27) after sweep validation across 1, 3, 5, and 10 $M_\oplus$ dry CHILI configurations.

```toml
[interior_struct.zalmoxis]
outer_solver = "newton"   # "newton" (default) or "picard"
```

| Solver | When it is the right choice | Cost |
|---|---|---|
| `"newton"` | Default for all production runs. Required for hot fully-molten profiles (super-Earths at IW+4) where damped Picard hits a basin attractor and stalls. | Auto-tightens integrator tolerances to `relative_tolerance=1e-9`, `absolute_tolerance=1e-10`. ~1.2 to 2x slower per call than Picard but converges where Picard fails. |
| `"picard"` | Bit-reproducibility comparison runs against pre-T2.1 builds. Earth-mass cool runs converge identically. | Default integrator tolerances (rel=1e-5, abs=1e-6). |

Do not set Newton's tolerances directly unless you know what you are doing. The wrapper only forwards the Newton block (`newton_max_iter`, `newton_tol`, `newton_relative_tolerance`, `newton_absolute_tolerance`) when `outer_solver = "newton"`, so a Picard run sees a bit-identical config dict to pre-T2.1 builds.

---

## Pre-main-loop equilibration

PROTEUS runs a CALLIOPE + Zalmoxis loop *before* the main coupling loop starts, to converge volatile partitioning and structure simultaneously. Each iteration: re-partition volatiles -> re-solve structure -> check $\Delta R / R$ and $\Delta P / P$ until both fall below `equilibrate_tol`.

```toml
[interior_struct.zalmoxis]
equilibrate_init     = true     # default
equilibrate_max_iter = 15       # default
equilibrate_tol      = 0.01     # 1% tolerance on R_int and P_surf
```

Disable only if you are debugging IC behavior. With it disabled, the first SPIDER/Aragog step uses an unequilibrated structure and the first ~10 iterations of the main loop spend wall time on what equilibration would have done in fewer Zalmoxis calls.

See [the equilibration explainer](../Explanations/proteus_coupling.md#pre-main-loop-equilibration) for the rationale.

---

## Update triggers during coupled evolution

Zalmoxis is not called every PROTEUS iteration. Re-solving structure is expensive (~tens of seconds per call) and the planet's bulk structure changes slowly. PROTEUS gates Zalmoxis re-solves on three composable triggers:

```toml
[interior_struct.zalmoxis]
update_interval        = 1e9     # max time between calls [yr]; default 1 Gyr (effectively a ceiling)
update_min_interval    = 0       # min time between calls [yr]; default 0 (no floor)
update_dtmagma_frac    = 0.05    # re-solve if |dT_magma/T_magma| > 5%
update_dphi_abs        = 0.05    # re-solve if |dPhi| > 0.05
update_stale_ceiling   = 2.5e4   # T1.5 stale-aware ceiling [yr]; default 25 kyr
```

A re-solve fires when *any* of these conditions is satisfied AND the time since last call exceeds `update_min_interval`. Set `update_stale_ceiling = 0` for legacy behaviour without staleness.

The defaults are tuned for 1 to 10 $M_\oplus$ CHILI runs. For very long evolutions (Gyr) where structure barely changes, raise `update_dphi_abs` to 0.1 and `update_dtmagma_frac` to 0.1 to cut wall time. For numerical-fragility investigations, set `update_min_interval` to your timestep so Zalmoxis fires every step.

---

## Mesh smoothing

The Aragog mesh that Zalmoxis writes is read by Aragog inside `solver.reset()`. Large radius shifts between successive Zalmoxis calls (e.g., during a magma-ocean to crystallised-mantle transition) can produce non-physical Aragog initial conditions on the new mesh. PROTEUS smooths the transition by capping the per-call radius shift:

```toml
[interior_struct.zalmoxis]
mesh_max_shift             = 0.05    # max fractional R shift per Zalmoxis call (5%)
mesh_convergence_interval  = 10.0    # iterations between full mesh re-converges [yr]
```

A `mesh_max_shift = 0.05` keeps the Aragog initial-condition reload safely within the entropy-table interpolation domain in 99%+ of CHILI runs.

---

## `dry_mantle` toggle

This toggle controls whether dissolved volatiles enter the Zalmoxis density mixing model.

```toml
[interior_struct.zalmoxis]
dry_mantle = false   # default; set true for Stage 1 phase-aware coupling
```

- `dry_mantle = false` (default): the mantle EOS is extended at runtime with Chabrier:H and PALEOS:H2O components, weighted by $\phi(r)$ from the volatile profile.
- `dry_mantle = true`: structure is solved against the canonical solid + liquid mantle tables alone. Volatile partitioning still happens in CALLIOPE, but it does not feed back into structure. Use this for Stage 1 cleanly-decoupled comparisons or when an EOS-mixing instability obscures another diagnostic.

---

## SPIDER P-S table generation

When `interior_energetics.module = "spider"`, Zalmoxis generates the SPIDER lookup tables on the fly from the active PALEOS EOS. Two knobs control resolution:

```toml
[interior_struct.zalmoxis]
lookup_nP = 1350    # number of pressure points
lookup_nS = 280     # number of entropy points
```

The defaults are validated against SPIDER's tolerance for table sparsity. Halving these values cuts table-generation wall time by ~3x but can produce visible spline artifacts in SPIDER's adiabat lookup.

---

## JAX structure path

Zalmoxis has an opt-in JAX + diffrax path for the structure solve, plus an Anderson Type-II Picard accelerator. Both default off for bit-identical behaviour with the numpy path.

```toml
[interior_struct.zalmoxis]
use_jax       = false
use_anderson  = false
```

In coupled mode, the wrapper auto-disables `use_jax` and `use_anderson` for any Zalmoxis call where neither a `temperature_function` nor a `temperature_arrays` argument is supplied (initialisation and equilibration calls), because the JAX RHS path collapses on P-ignoring callables. See the [JAX path explainer](../Explanations/proteus_coupling.md#jax-path-and-temperature-arrays) for why.

---

## `--deterministic` flag for fragile runs

For configurations that fail with `RuntimeError: Aragog retry ladder exhausted` or T_core-jump-guard activations on otherwise-clean restarts, launch PROTEUS with the `--deterministic` flag. It self-re-execs to set `JAX_ENABLE_X64=1` and `XLA_FLAGS=--xla_cpu_enable_fast_math=false` *before* JAX imports.

```bash
nohup proteus start -c <cfg.toml> --offline --deterministic \
    > output/<run>/launch.log 2>&1 & disown
```

Do not enable by default; it carries a small per-step cost. Use only when you observe noise-floor floating-point divergence between launches of the same config (Stage 4.4 atmodeller-vs-CALLIOPE diagnostics, see the [Coupling explainer §determinism](../Explanations/proteus_coupling.md#determinism-and-the-deterministic-flag)).

---

## Worked example: Stage 4 super-Earth coupled CHILI

```toml
[planet]
mass_tot            = 5.0
tsurf_init          = 3500
tcenter_init        = 10000
temperature_mode    = "liquidus_super"
delta_T_super       = 200
ini_entropy         = 3900

[interior_struct]
module          = "zalmoxis"
core_frac       = 0.30
core_frac_mode  = "mass"
core_density    = "self"
core_heatcap    = "self"

[interior_struct.zalmoxis]
core_eos             = "PALEOS:iron"
mantle_eos           = "PALEOS:MgSiO3"
mushy_zone_factor    = 0.8
mantle_mass_fraction = 0.0
num_levels           = 150
outer_solver         = "newton"
equilibrate_init     = true
update_dphi_abs      = 0.05
mesh_max_shift       = 0.05

[interior_energetics]
module = "aragog"
# ...

[interior_energetics.aragog]
dilatation = true   # paper-line default since 2026-04-30
```

Run with:

```bash
nohup proteus start -c <this.toml> --offline > output/<run>/launch.log 2>&1 & disown
```

For numerically fragile anchors (wet 1 $M_\oplus$ at IW+4, reduced 1 $M_\oplus$ at IW-2), add `--deterministic`.

---

## Common pitfalls

- **Setting `[InputParameter]` in a PROTEUS config does nothing.** Those keys are read only by `python -m zalmoxis`. PROTEUS reads its own `[planet]` and `[interior_struct.zalmoxis]` blocks.
- **Using `core_frac_mode = "mass"` with `module = "spider"` fails validation.** Mass-mode core fractions require Zalmoxis. SPIDER only accepts radius-mode.
- **Picard convergence stalls at high mass + hot.** Switch to `outer_solver = "newton"` (already default). If still failing on a Newton run, check the helpfile for $\Delta T_\mathrm{cmb}$ noise patterns; a `--deterministic` rerun may be needed.
- **`mantle_mass_fraction != 0` for a 2-layer model.** For 2-layer models without ice and without a $T$-dependent mantle EOS, set `mantle_mass_fraction = 0` so the solver derives it as `1 - core_frac`.
- **Equilibration warns "did not converge after 15 iterations".** Usually a sign that CALLIOPE is oscillating. Inspect $\Delta P/P$ trace; if it is dropping but slowly, raise `equilibrate_max_iter`. If it is non-monotonic, check the volatile inventory.

For the why behind these, see the [Coupling to PROTEUS theory page](../Explanations/proteus_coupling.md).
