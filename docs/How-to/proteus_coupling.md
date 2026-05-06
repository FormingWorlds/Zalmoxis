# Coupling Zalmoxis to PROTEUS

This page is the practical recipe for running Zalmoxis as the interior-structure module inside the PROTEUS framework.
The standalone `python -m zalmoxis -c input/default.toml` workflow described in [Usage](usage.md) does *not* apply when Zalmoxis is driven from PROTEUS.
PROTEUS bypasses Zalmoxis's own TOML sections (`[InputParameter]`, `[AssumptionsAndInitialGuesses]`, `[EOS]`, ...) entirely and builds the call-time configuration from its own `[interior_struct]` block.

For the theory behind this design, see [Coupling to PROTEUS (theory)](../Explanations/proteus_coupling.md).

!!! info "Audience"
    This guide assumes you already have a working PROTEUS install (see the [PROTEUS docs](https://proteus-framework.org/PROTEUS)) and a coupled atmosphere + interior + outgassing config.
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
mantle_mass_fraction = 0.0     # 0 for PALEOS or Seager 2-layer; > 0 for 3-layer or WolfBower2018/RTPress100TPa mantle
num_levels      = 150
```

This is sufficient for an Earth-mass coupled run. The other `[interior_struct.zalmoxis]` keys default to validated production values; tune them only if you hit a specific issue.

!!! tip "Picking EOS"
    For paper-quality runs, use `PALEOS:iron` + `PALEOS:MgSiO3` (unified tables, $T$-dependent core, valid to 50 $M_\oplus$).
    The `Seager2007:*` and `WolfBower2018:MgSiO3` choices are supported for sensitivity tests and SPIDER parity comparisons.
    See [Configuration](configuration.md#available-eos-options) for the full table; the same identifiers apply in coupled mode.

---

## Initial-condition mode (`temperature_mode`)

Inside PROTEUS, the IC selection is governed by `[planet].temperature_mode`, **not** the Zalmoxis-side `[AssumptionsAndInitialGuesses]` block. PROTEUS translates the planet-side mode into the Zalmoxis-side structure-solve mode at call time.

| `[planet].temperature_mode` | Zalmoxis structure-solve mode | When to use |
|---|---|---|
| `"adiabatic_from_cmb"` | `adiabatic_from_cmb` (CMB-anchored) | Default. Integrates the adiabat upward from `tcmb_init` at the converged $R_\mathrm{cmb}$. |
| `"liquidus_super"` | `adiabatic_from_cmb` (CMB-anchored, $T_\mathrm{cmb}$ from liquidus) | Recommended for hot magma-ocean ICs. Anchors $T_\mathrm{cmb} = T_\mathrm{liq}(P_\mathrm{cmb}) + \Delta T_\mathrm{super}$ from the Fei+2021 liquidus. |
| `"adiabatic"` | `adiabatic` (surface-anchored) | Surface-anchored adiabat. Pairs with `[planet].tsurf_init` and `[planet].tcenter_init`. |
| `"isentropic"` | `adiabatic` (mapped) | Energetics consumes `ini_entropy`; Zalmoxis T-profile is used only for the structure solve. |
| `"accretion"` | `adiabatic` (mapped) | White & Li accretion thermal state. Energetics computes its own T(r) post-structure. |
| `"isothermal"`, `"linear"`, `"prescribed"` | unchanged | Diagnostics only. Not for production coupled runs. |

For `liquidus_super`, set `[planet].delta_T_super` (default 500 K, validated `>= 0`). The `tsurf_init` and `tcenter_init` fields are ignored under `liquidus_super`: the adiabat is set by $T_\mathrm{liq}(P_\mathrm{cmb}) + \Delta T_\mathrm{super}$ at the CMB and integrated upward. Before Zalmoxis has populated `hf_row['P_cmb']` on the first call, the wrapper estimates $P_\mathrm{cmb}$ from a Noack & Lasbleis (2020) mass-aware scaling so super-Earth runs anchor at the correct deep-mantle pressure rather than at an Earth-only value. After one round-trip the converged Zalmoxis $P_\mathrm{cmb}$ replaces the estimate.

```toml
[planet]
mass_tot            = 5.0
temperature_mode    = "liquidus_super"
delta_T_super       = 200       # K above the Fei+2021 liquidus at P_cmb
# tsurf_init / tcenter_init are ignored under liquidus_super
```

See [the IC anchors theory section](../Explanations/proteus_coupling.md#initial-condition-anchors) for the derivation.

---

## Outer mass-radius solver (`outer_solver`)

Zalmoxis's outer loop adjusts the planet radius until $M(R) = M_\mathrm{target}$. Two solvers are available; **Newton** is the default.

```toml
[interior_struct.zalmoxis]
outer_solver = "newton"   # "newton" (default) or "picard"
```

| Solver | When it is the right choice | Cost |
|---|---|---|
| `"newton"` | Default for all production runs. Required for hot fully-molten profiles (super-Earths at IW+4) where damped Picard hits a basin attractor and stalls. | Auto-tightens integrator tolerances to `relative_tolerance=1e-9`, `absolute_tolerance=1e-10`. ~1.2 to 2x slower per call than Picard but converges where Picard fails. |
| `"picard"` | Earth-mass cool runs; comparison runs that must match the historic Picard fixed-point trajectory. | Default integrator tolerances (rel=1e-5, abs=1e-6). |

Do not set Newton's tolerances directly unless you know what you are doing. The wrapper only forwards the Newton block (`newton_max_iter`, `newton_tol`, `newton_relative_tolerance`, `newton_absolute_tolerance`) when `outer_solver = "newton"`, so a Picard run sees an unchanged config dict.

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
update_stale_ceiling   = 2.5e4   # stale-aware ceiling [yr]; default 25 kyr
```

A re-solve fires when *any* of these conditions is satisfied AND the time since last call exceeds `update_min_interval`. Set `update_stale_ceiling = 0` to disable the stale-aware ceiling.

The defaults are tuned for 1 to 10 $M_\oplus$ coupled runs. For very long evolutions (Gyr) where structure barely changes, raise `update_dphi_abs` to 0.1 and `update_dtmagma_frac` to 0.1 to cut wall time. For numerical-fragility investigations, set `update_min_interval` to your timestep so Zalmoxis fires every step.

---

## Mesh smoothing

The Aragog mesh that Zalmoxis writes is read by Aragog inside `solver.reset()`. Large radius shifts between successive Zalmoxis calls (e.g., during a magma-ocean to crystallised-mantle transition) can produce non-physical Aragog initial conditions on the new mesh. PROTEUS smooths the transition by capping the per-call radius shift:

```toml
[interior_struct.zalmoxis]
mesh_max_shift             = 0.05    # max fractional R shift per Zalmoxis call (5%)
mesh_convergence_interval  = 10.0    # iterations between full mesh re-converges [yr]
```

A `mesh_max_shift = 0.05` keeps the Aragog initial-condition reload safely within the entropy-table interpolation domain in 99%+ of coupled runs.

---

## `dry_mantle` toggle

This toggle controls whether dissolved volatiles enter the Zalmoxis density mixing model.

```toml
[interior_struct.zalmoxis]
dry_mantle = true    # default; structure ignores dissolved-volatile mass
```

- `dry_mantle = true` (default): structure is solved against the canonical solid + liquid mantle tables alone. Volatile partitioning still happens in the outgassing module, but it does not feed back into structure. This is the production setting for paper runs and for cleanly-decoupled module comparisons.
- `dry_mantle = false`: the mantle EOS is extended at runtime with Chabrier:H and PALEOS:H2O components, weighted by $\phi(r)$ from the volatile profile. Use this when you specifically want phase-aware volatile mixing in the structure; expect a wall-time cost from the volatile-extended EOS.

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

Zalmoxis runs the structure solve through a JAX + diffrax path by default, with an opt-in Anderson Type-II Picard accelerator. The numpy path is selectable for bit-identical reproduction of the legacy trajectory or on systems without a JAX-compatible backend.

```toml
[interior_struct.zalmoxis]
use_jax       = true     # default
use_anderson  = false    # default
```

In coupled mode, the wrapper auto-disables `use_jax` and `use_anderson` for any Zalmoxis call where neither a `temperature_function` nor a `temperature_arrays` argument is supplied (initialisation and equilibration calls), because the JAX RHS path collapses on P-ignoring callables. See the [JAX path explainer](../Explanations/proteus_coupling.md#jax-path-and-temperature-arrays) for why.

---

## `--deterministic` flag for fragile runs

For configurations that fail with `RuntimeError: Aragog retry ladder exhausted` or T_core-jump-guard activations on otherwise-clean restarts, launch PROTEUS with the `--deterministic` flag. It self-re-execs to set `JAX_ENABLE_X64=1` and `XLA_FLAGS=--xla_cpu_enable_fast_math=false` *before* JAX imports.

```bash
nohup proteus start -c <cfg.toml> --offline --deterministic \
    > output/<run>/launch.log 2>&1 & disown
```

Do not enable by default; it carries a small per-step cost. Use only when you observe noise-floor floating-point divergence between launches of the same config. Background and mechanism in the [Coupling explainer §determinism](../Explanations/proteus_coupling.md#determinism-and-the-deterministic-flag).

---

## Worked example: super-Earth coupled run

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
module     = "aragog"
num_levels = 80
rtol       = 1e-8
atol       = 1e-10

[interior_energetics.aragog]
backend     = "jax"
core_bc     = "energy_balance"
phi_step_cap = 0.05
```

Run with:

```bash
nohup proteus start -c <this.toml> --offline > output/<run>/launch.log 2>&1 & disown
```

For numerically fragile anchors (wet 1 $M_\oplus$ at IW+4, reduced 1 $M_\oplus$ at IW-2), add `--deterministic`.

---

## Prioritised settings

The list below is the short version of "what actually matters" when you stand up a new coupled Zalmoxis run.
Knobs are ordered from highest to lowest impact on results and stability.
Every default is read directly from `proteus.config` (see `src/proteus/config/_struct.py`, `_interior.py`, and `_planet.py`); if a quoted default disagrees with the source, the source wins.

### 1. `interior_struct.module` and `interior_energetics.module`

**Defaults**: `interior_struct.module = "zalmoxis"`, `interior_energetics.module = "aragog"`.
**Choices**: `interior_struct.module` is one of `"zalmoxis"`, `"spider"`, `"dummy"`; `interior_energetics.module` is one of `"aragog"`, `"spider"`, `"dummy"`.
**Recommendation**: keep `"zalmoxis"` + `"aragog"` for new production runs.

The structure module (Zalmoxis) and the energetics module (Aragog or SPIDER) are configured independently.
Zalmoxis + Aragog is the default pairing because Aragog handles mushy-zone stiffness through the SUNDIALS CVODE + JAX-derived Jacobian path, and the Zalmoxis-side P-T tables for Aragog are full rectangular grids (no scipy fallback to unstructured interpolation).
Use `interior_energetics.module = "spider"` only for SPIDER-parity comparisons, and only with `core_frac_mode = "radius"` because SPIDER does not accept mass-mode core fractions.

### 2. `interior_struct.zalmoxis.update_interval` [yr]

**Default**: `1e9` (effectively a ceiling).
**Recommendation**: set to `5e4` (50 kyr) for evolutionary runs that traverse the magma-ocean to crystallised-mantle transition; raise to `>= 1e9` to freeze the mesh for clean atmosphere-interior parity tests.

This is the maximum wall time between two Zalmoxis re-solves.
Real re-solves still gate on the dynamic triggers (`update_dphi_abs`, `update_dtmagma_frac`, `update_stale_ceiling`) and respect the `update_min_interval` floor.
Setting `update_interval >= 1e9` together with `update_dphi_abs = 1.0` and `update_dtmagma_frac = 1.0` reduces Zalmoxis to a one-shot pre-main-loop solve, which is the right setup for "static-Zalmoxis" diagnostic runs.
The 50 kyr value is tight enough to track the rheological transition, loose enough not to dominate wall time.

### 3. `interior_struct.zalmoxis.outer_solver`

**Default**: `"newton"` (`_struct.py:198`).
**Choices**: `"newton"`, `"picard"`.
**Recommendation**: stay on Newton.

Newton is the validated default for the 1 / 3 / 5 / 10 $M_\oplus$ super-Earth sweep.
Damped Picard hits a basin attractor on hot fully-molten profiles at high mass and stalls; Newton + brentq bracketing converges on every G4 starting point in that sweep.
The wrapper auto-tightens the integrator tolerances to `relative_tolerance = 1e-9` / `absolute_tolerance = 1e-10` whenever Newton is selected, so do not set those by hand.
Use `"picard"` only when reproducing a Picard fixed-point trajectory; expect 1.2 to 2x speedups on Earth-mass cool runs at the cost of giving up convergence on hot super-Earths.

### 4. `interior_energetics.aragog.phi_step_cap` [$\Delta\phi$ fraction]

**Default**: `0.0` (cap disabled).
**Recommendation**: `0.05` for standard evolutionary runs; tighten to `0.001`-`0.01` only if mushy-zone oscillations appear in the first few snapshots.

This is the per-call $|\Delta\phi|$ cap implemented as a SUNDIALS root function (Strategy B v3).
CVODE evaluates `g(t, y) = cap - |Phi_global(t, y) - Phi_global(start)|` at every internal step and returns at the exact zero-crossing, so the integrator's adaptive step never overshoots a physically meaningful melt-fraction excursion.
With `phi_step_cap = 0` the rootfn is not registered and Aragog runs with its native step control.
Use the cap to keep the rheological transition trackable when interior structure or atmospheric flux changes quickly.

### 5. `planet.prevent_warming`

**Default**: `false`.
**Recommendation**: leave at `false`. Only enable when you specifically need to enforce monotonic cooling.

When set to `true`, all atmosphere modules and termination checks enforce a `T_magma = min(T_new, T_prev)` clamp.
This is energy-non-conserving in any regime where the integrator transiently warms the magma (heat-pump quasi-equilibrium, dynamic Zalmoxis re-solves, atmosphere-interior coupling overshoot): the clamp pins the surface byte-exactly while the interior continues to deliver flux, producing an apparent T_magma plateau with an energy-leak signature.
All bundled CHILI configs default to `prevent_warming = false`; follow that lead.

### 6. `interior_struct.zalmoxis.mushy_zone_factor` [0.7-1.0]

**Default**: `0.8` (PROTEUS schema; standalone Zalmoxis defaults to 1.0).
**Recommendation**: `0.8` for paper-quality runs with PALEOS MgSiO3.

This sets the solidus relative to the liquidus as $T_\mathrm{sol} = f \cdot T_\mathrm{liq}$, controlling the width of the mushy zone in the PALEOS unified EOS.
The 0.8 default approximates the [Stixrude+2014](https://ui.adsabs.harvard.edu/abs/2014RSPTA.37230076S) cryoscopic depression for MgSiO3 and is applied consistently across Zalmoxis density interpolation, SPIDER phase boundaries, and the VolatileProfile $\phi$-blending.
Setting `mushy_zone_factor = 1.0` collapses the mushy band to a sharp boundary.
This factor only affects PALEOS unified tables; it is silently ignored for `WolfBower2018` and `RTPress100TPa`, which use explicit melting-curve files.

### 7. `interior_struct.zalmoxis.num_levels` and `interior_energetics.num_levels`

**Defaults**: `interior_struct.zalmoxis.num_levels = 150`; `interior_energetics.num_levels = 80`.
**Recommendation**: keep the schema defaults. Zalmoxis 150 layers (structure mesh) and Aragog 80 layers (energetics mesh) is the validated production combination; SPIDER also runs at 80.

The two meshes serve different purposes.
Zalmoxis solves the static structure ODE (mass + hydrostatic + EOS) and benefits from the higher resolution to resolve thin core-mantle boundary regions in super-Earths.
Aragog and SPIDER solve the time-dependent thermal evolution on a coarser mesh; 80 layers matches the SPIDER reference and keeps CVODE matrix factorisations cheap.
Halving either does not save wall time linearly because the dominant cost is per-cell EOS evaluation; raising either above the default rarely pays off.

### 8. `interior_struct.zalmoxis.equilibrate_init`

**Default**: `true`.
**Recommendation**: leave on for production; disable only for IC-debugging.

This drives the CALLIOPE + Zalmoxis pre-main-loop equilibration described above.
With it disabled, the first SPIDER or Aragog step uses an unequilibrated structure and the first ~10 main-loop iterations spend wall time doing what a few cheap Zalmoxis-only iterations would have done up front.
The relative cost of the equilibration loop is small (typically 5-15 Zalmoxis calls before the main loop runs) and it almost always reduces total wall time.

#### Summary table

| Knob | Default | Recommended | Why it matters |
|---|---|---|---|
| `interior_struct.module` | `"zalmoxis"` | `"zalmoxis"` | Mass-mode core fractions, self-consistent EOS, SPIDER P-S tables on demand. |
| `interior_energetics.module` | `"aragog"` | `"aragog"` | CVODE + JAX-derived Jacobian, robust on stiff mushy-zone profiles. |
| `interior_struct.zalmoxis.update_interval` | `1e9` | `5e4` | Tracks the rheological transition without dominating wall time. |
| `interior_struct.zalmoxis.outer_solver` | `"newton"` | `"newton"` | Converges on hot super-Earths where damped Picard stalls. |
| `interior_energetics.aragog.phi_step_cap` | `0.0` | `0.05` | SUNDIALS root function caps per-call $|\Delta\phi|$ excursion. |
| `planet.prevent_warming` | `false` | `false` | The clamp is energy-non-conserving in warming sub-steps. |
| `interior_struct.zalmoxis.mushy_zone_factor` | `0.8` | `0.8` | Stixrude+2014 cryoscopic depression for PALEOS MgSiO3. |
| `interior_struct.zalmoxis.num_levels` | `150` | `150` | Zalmoxis structure mesh resolution. |
| `interior_energetics.num_levels` | `80` | `80` | Aragog / SPIDER energetics mesh resolution. |
| `interior_struct.zalmoxis.equilibrate_init` | `true` | `true` | Saves wall time in the first ~10 main-loop iterations. |

---

## Worked example: minimal copy-paste TOML excerpt

Anchored on `input/chili/chili_paleos_v1_1_0_1me_150res.toml` (the 1 $M_\oplus$ PALEOS reference run).
This excerpt covers only the load-bearing blocks for Zalmoxis + Aragog coupling; merge it into a complete PROTEUS config with your own `[star]`, `[orbit]`, `[atmos_clim]`, `[outgas]`, `[escape]`, and `[params]` sections.

```toml
[planet]
mass_tot           = 1.0
temperature_mode   = "adiabatic_from_cmb"
tcmb_init          = 7199.0
tsurf_init         = 3830.0
tcenter_init       = 6000.0
ini_entropy        = 3900.0
ini_dsdr           = -4.698e-6
volatile_mode      = "elements"
volatile_reservoir = "mantle"
prevent_warming    = false

[interior_struct]
core_frac      = 0.325
core_frac_mode = "mass"
module         = "zalmoxis"
core_density   = "self"
core_heatcap   = "self"

[interior_struct.zalmoxis]
core_eos             = "PALEOS:iron"
mantle_eos           = "PALEOS-2phase:MgSiO3"
ice_layer_eos        = "none"
mushy_zone_factor    = 0.8
mantle_mass_fraction = 0.0
num_levels           = 150
outer_solver         = "newton"
update_interval      = 50000.0
update_min_interval  = 100.0
update_dphi_abs      = 0.05
update_dtmagma_frac  = 0.05
update_stale_ceiling = 25000.0
mesh_max_shift       = 0.05
equilibrate_init     = true
equilibrate_tol      = 0.01
dry_mantle           = true
use_jax              = true

[interior_energetics]
module           = "aragog"
num_levels       = 80
rtol             = 1e-8
atol             = 1e-10
trans_conduction = true
trans_convection = true
trans_grav_sep   = true
trans_mixing     = true
heat_radiogenic  = true

[interior_energetics.aragog]
backend                     = "jax"
core_bc                     = "energy_balance"
solver_method               = "cvode"
atol_temperature_equivalent = 1e-8
phi_step_cap                = 0.05
```

For the full reference (with `[params.dt]`, `[params.stop]`, atmosphere, escape, and outgassing blocks), see `input/chili/chili_paleos_v1_1_0_1me_150res.toml` in the PROTEUS repo.

---

## Common pitfalls

- **Setting `[InputParameter]` in a PROTEUS config does nothing.** Those keys are read only by `python -m zalmoxis`. PROTEUS reads its own `[planet]` and `[interior_struct.zalmoxis]` blocks.
- **Using `core_frac_mode = "mass"` with `module = "spider"` fails validation.** Mass-mode core fractions require Zalmoxis. SPIDER only accepts radius-mode.
- **Picard convergence stalls at high mass + hot.** Switch to `outer_solver = "newton"` (already default). If still failing on a Newton run, check the helpfile for $\Delta T_\mathrm{cmb}$ noise patterns; a `--deterministic` rerun may be needed.
- **`mantle_mass_fraction != 0` rejected by the validator.** Only the `WolfBower2018:MgSiO3` and `RTPress100TPa:MgSiO3` mantle EOS prefixes (legacy 2-phase tables) require a non-zero `mantle_mass_fraction`. For PALEOS, PALEOS-2phase, Seager2007, or analytic mantles in a 2-layer model, set `mantle_mass_fraction = 0` so the solver derives it as `1 - core_frac`. 3-layer models with an ice layer also require `mantle_mass_fraction > 0`.
- **Equilibration warns "did not converge after 15 iterations".** Usually a sign that CALLIOPE is oscillating. Inspect $\Delta P/P$ trace; if it is dropping but slowly, raise `equilibrate_max_iter`. If it is non-monotonic, check the volatile inventory.

For the why behind these, see the [Coupling to PROTEUS theory page](../Explanations/proteus_coupling.md).
