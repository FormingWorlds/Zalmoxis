# Coupling to PROTEUS

This page is the **theory** of how Zalmoxis plugs into a PROTEUS coupled run: where the structure solver sits in the per-iteration sequence, what the wrapper does on init and on every dynamic update, which `hf_row` keys it consumes and produces, and where the known landmines live. For the practical TOML recipe, see the [PROTEUS coupling how-to](../How-to/proteus_coupling.md).

When Zalmoxis runs inside PROTEUS, the coupling is at the **function-call level**, not at the file level. PROTEUS does not write a Zalmoxis TOML, does not invoke `python -m zalmoxis`, and does not read `output/planet_profile.txt`. The wrapper in `proteus/interior_struct/zalmoxis.py` builds a Python dict from the PROTEUS configuration and calls `zalmoxis.solver.main()` (or `solve_miscible_interior()` when the global-miscibility path is active). State exchange happens through the shared `hf_row` dictionary, plus one file (`zalmoxis_output.dat`) that Aragog reads back as its mantle mesh.

## Where Zalmoxis sits in the iteration

PROTEUS's main loop in `src/proteus/proteus.py` (the `start()` method, lines ~540 onwards) walks each module in a fixed order. Zalmoxis is invoked twice per loop family: once on init (via `solve_structure` and the equilibration loop), and during the main loop as an optional structure refresh **after the interior thermal solve and after `Time` has been advanced**. The relevant slice is:

```python
# Per main-loop iteration (init_stage = False, post-equilibration)

# 1. Interior thermal evolution: SPIDER or Aragog or dummy
run_interior(self.directories, self.config, self.hf_all,
             self.hf_row, self.interior_o, write_data=is_snapshot)

# 2. Advance simulation time using the interior step
self.hf_row['Time']     += self.interior_o.dt
self.hf_row['age_star'] += self.interior_o.dt

# 3. Optional structure refresh (Zalmoxis, gated)
if (not self.init_stage
    and self.config.interior_struct.module == 'zalmoxis'
    and self.config.interior_struct.zalmoxis.update_interval > 0):
    update_structure_from_interior(self.directories, self.config,
                                   self.hf_row, self.interior_o, ...)

# 4. Orbit and tides
run_orbit(self.hf_row, self.config, self.directories, self.interior_o)

# 5. Stellar flux (instellation, spectrum)
update_stellar_quantities(...)
get_new_spectrum(...)

# 6. Atmospheric escape (ZEPHYRUS), only after init_loops + 2 iters
run_escape(...)

# 7. Outgassing: CALLIOPE / atmodeller / crystallized / desiccated branch
run_outgassing(self.directories, self.config, self.hf_row)

# 8. Atmosphere radiative transfer (AGNI / JANUS)
run_atmosphere(...)
```

Two ordering details matter for how the wrapper reads `hf_row`. First, the interior thermal solver updates `T_magma`, `Phi_global`, `T_surf`, `T_core`, and `F_int` **before** the Zalmoxis trigger fires, so the structure update sees the most recent thermal state. Second, `Time` is advanced **before** the trigger, so any "elapsed since last update" comparison inside `update_structure_from_interior` is referenced against the new time. The `interior_o.last_successful_struct_time` clock is the partner anchor and is only advanced when a Zalmoxis re-solve actually converges.

Zalmoxis is the most expensive of the per-iteration submodules (5 to 60 s depending on mass and EOS; CALLIOPE is tens of milliseconds, AGNI seconds-to-minutes), so the wrapper gates it on physical-state-change triggers rather than calling it every iteration. See [dynamic update triggers](#dynamic-update-triggers) below.

## What the PROTEUS wrapper does

The wrapper is split between two files. `src/proteus/interior_struct/zalmoxis.py` builds the Zalmoxis input dict, resolves the EOS material dictionary, drives the structure solve (either `solve_miscible_interior` or the standard solver path), and writes the Aragog mesh file `zalmoxis_output.dat`. `src/proteus/interior_energetics/wrapper.py` contains the orchestration: `equilibrate_initial_state` (init-time CALLIOPE+Zalmoxis loop), `solve_structure` (initial structure solve dispatcher), and `update_structure_from_interior` (main-loop trigger evaluation, fall-back logic, mass-anchor check, mesh blending). The split is functional, not modular: the interior-struct file is the lower-level Zalmoxis adapter, and the interior-energetics wrapper is the orchestration glue that ties Zalmoxis to whichever energetics module is active. SPIDER and Aragog both call into the same `update_structure_from_interior`.

The wrapper builds the Zalmoxis input dict fresh on every call from `config` and `hf_row`. PROTEUS does not mutate the shared `Config` object at runtime (the one documented exception is `config.orbit.module = 'dummy'` during the initial structure solve, restored in a `finally`). Where a temporary override is needed (for instance, SPIDER coupling forcing `adiabatic` mode for a T-dependent mantle EOS), the wrapper accepts a `temperature_mode_override` kwarg that takes effect for that single call only.

### 1. Initial structure solve

`solve_structure(dirs, config, hf_all, hf_row, outdir)` dispatches on `config.interior_struct.module`. For `zalmoxis`, it calls `determine_interior_radius_with_zalmoxis`, which temporarily sets `config.orbit.module = 'dummy'` for the duration of the structure solve so orbital-feedback hooks do not fire on a planet whose radius has not yet converged. (This single mutation of the shared `Config` object is a known pattern documented in the project rules; the wrapper restores `config.orbit.module` in a `finally` block.) The result is a `hf_row` populated with `R_int`, `R_core`, `M_int`, `M_core`, `gravity`, `P_center`, `P_cmb`, `core_density`, `core_heatcap`, and the EOS-consistent T(r) anchors.

### 2. Init equilibration loop

`equilibrate_initial_state(dirs, config, hf_row, outdir)` iterates CALLIOPE outgassing and Zalmoxis structure until the interior radius and surface pressure are mutually consistent. Each iteration calls `calc_target_elemental_inventories`, then `run_outgassing`, then `zalmoxis_solver`, and checks $|\Delta R / R| < $ tolerance and $|\Delta P / P| <$ tolerance (`equilibrate_tol`, default $0.01$). After convergence (or `equilibrate_max_iter` exhaustion, default 15), `generate_spider_tables()` regenerates the SPIDER P-S lookup tables against the final composition. See [the dedicated section](#initial-condition-equilibration-loop).

### 3. Per-iteration update

`update_structure_from_interior(dirs, config, hf_row, interior_o, last_struct_time, last_Tmagma, last_Phi)` is the function called from the main loop. It evaluates the trigger conditions (see [Dynamic update triggers](#dynamic-update-triggers)). When fired, it does the following in order.

1. Build an interpolating `temperature_function(r, P)` from `interior_o.radius` and `interior_o.temp` (staggered-node T from the energetics solver), holding T constant at `T_cmb` for $r \le R_\mathrm{cmb}$ to cover the core region.
2. Build a sorted r-indexed `temperature_arrays = (r_arr, T_arr)` tuple by reversing `interior_o.radius` to ascending order and applying an explicit `argsort` so `jnp.interp` sees a strictly increasing $r$.
3. Atomically snapshot `zalmoxis_output.dat` to `.prev`, snapshot `dirs['spider_mesh']` to `.prev`, and snapshot the current `hf_row` structure scalars (`R_int`, `M_int`, `M_core`, `M_mantle`, `P_surf`, `R_core`, `P_center`, `rho_avg`) into a local `_saved_structure` dict.
4. Call `zalmoxis_solver(config, outdir, hf_row, num_spider_nodes=..., temperature_function=temperature_function, temperature_arrays=(r_arr, T_arr))`. Zalmoxis's JAX path consumes the arrays, its numpy Picard path consumes the callable, and they do not contaminate each other (see [Outer solver and JAX path](#outer-solver-and-jax-path)).
5. On success, run the mass-anchor check `|M_int / M_int_target - 1| < 3e-3` (`_ZALMOXIS_MASS_ANCHOR_TOL`). The check tightens Zalmoxis's internal `solver_tol_outer` (default $3 \times 10^{-3}$, a numerical tolerance) to a coupling contract: the wrapper raises `RuntimeError` on violation and routes through the same fall-back path as a non-converged solve, so a too-loose-converged result never reaches the next iteration.
6. Clear `hf_row['_structure_stale']` and update `interior_o.last_successful_struct_time` (the anchor for `update_stale_ceiling`).

On any `RuntimeError` (Zalmoxis non-convergence, schema violation, mass-anchor violation), the wrapper restores the saved structure scalars, restores `zalmoxis_output.dat` and the SPIDER mesh from their `.prev` files, sets `hf_row['_structure_stale'] = True`, and increments a module-level `_zalmoxis_fail_count`. After 8 consecutive failures (`_ZALMOXIS_MAX_CONSECUTIVE_FAILS`) the run aborts. On the first success, the streak resets to 0 and is logged for post-hoc analysis.

### 4. Mesh blending

After a successful re-solve, `blend_mesh_files` (in `interior_energetics/spider.py`, used for both SPIDER and Aragog mesh files) caps the per-update fractional radius shift at `mesh_max_shift` (default 0.05 = 5%). When the requested shift exceeds the cap, the file is post-modified in place to a partial blend, and `dirs['mesh_shift_active']` is set to `True` so the [mesh-convergence trigger](#dynamic-update-triggers) refires on the next iteration with the floor bypassed. See [Mesh handover and blending](#mesh-handover-and-blending).

## State exchange (`hf_row` reads and writes)

Zalmoxis reads the current planet state from `hf_row`, runs its structure solve, and writes back the converged structure quantities. Two separate kwarg surfaces feed Zalmoxis: the dict built by `load_zalmoxis_configuration` (TOML-derived inputs) and the runtime-derived T(r) profile. The runtime-derived T(r) is supplied by the main-loop path only.

### Reads from `hf_row`

| Key | Meaning | Used for |
|---|---|---|
| `T_magma` | Current magma-ocean T [K] | Trigger decision (`update_dtmagma_frac`); not passed to Zalmoxis directly. |
| `Phi_global` | Global melt fraction [0, 1] | Trigger decision (`update_dphi_abs`). |
| `M_int` | Interior mass [kg] | Used in fall-back accounting and mass-anchor check. |
| `M_core` | Core mass [kg] | Same. |
| `R_int` | Interior radius [m] | Equilibration convergence, scalar override for atmosphere BC. |
| `P_cmb` | CMB pressure [Pa] | Resolves $T_\mathrm{cmb}$ in `liquidus_super` mode (Fei+2021 liquidus at $P_\mathrm{cmb}$). On the very first call uses a Noack & Lasbleis (2020) mass-aware fallback. |
| `P_surf` | Surface pressure [bar in `hf_row`, scaled to Pa internally] | Atmospheric BC for the structure solve, when finite and positive. |
| `H2O_kg_total`, `CO2_kg_total`, ... | Volatile inventory [kg] | Subtracted from `mass_tot * M_earth` to give the dry interior `planet_mass` passed to Zalmoxis. |
| `H2O_kg_liquid`, `H2_kg_liquid` | Dissolved volatile masses [kg] | Build the `VolatileProfile` when `dry_mantle = false`. |
| `T_eqm`, `f_accretion`, `f_differentiation` | Accretion-mode IC inputs | Drive White & Li initial thermal state when `temperature_mode = "accretion"`. |

### Writes to `hf_row`

| Key | Meaning |
|---|---|
| `R_int` | Top-of-mantle (interior surface) radius [m] |
| `R_core` | CMB radius [m] |
| `M_int` | Total interior mass [kg], reported by Zalmoxis |
| `M_core` | Core mass enclosed at CMB [kg] |
| `M_int_target` | The dry mass Zalmoxis was instructed to converge toward (= `mass_tot * M_earth - M_volatiles`) |
| `gravity` | Surface gravity [m s$^{-2}$] |
| `P_surf` | Atmospheric pressure at the structure-side top-of-mantle [bar] |
| `P_center` | Center pressure [Pa] |
| `P_cmb` | Core-mantle boundary pressure [Pa] |
| `core_density` | Mass-enclosed-at-CMB / volume-enclosed-at-CMB [kg m$^{-3}$], the self-consistent core density |
| `core_heatcap` | Either 450 J kg$^{-1}$ K$^{-1}$ (Dulong-Petit iron, when `core_heatcap = "self"`) or the configured numeric value |
| `rho_avg` | Volume-averaged interior density [kg m$^{-3}$] |
| `_structure_stale` | Flag consumed by `Aragog.setup_or_update_solver` to detect fall-back state |
| `T_cmb_initial`, `T_surf_accr`, `T_surface_initial` | Accretion-mode thermal-state outputs (only when `temperature_mode = "accretion"`) |
| `_initial_T_profile`, `_initial_T_radii`, `_initial_T_pressure` | Adiabatic T(r) profile arrays for the energetics solver IC |
| `R_solvus`, `T_solvus`, `P_solvus`, `X_H2_int` | Global-miscibility binodal outputs (only when `global_miscibility = true`) |

The full set of TOML-derived inputs the wrapper translates is in `load_zalmoxis_configuration` (see `interior_struct/zalmoxis.py` lines ~420-564). Every TOML key from `[interior_struct]` and `[interior_struct.zalmoxis]` lands either in the Zalmoxis dict, in a derived helper (`mushy_zone_factors`, `_resolve_zalmoxis_temperature_mode`, `_resolve_zalmoxis_cmb_temperature`), or in a runtime gate (the JAX-disable path on init / equilibration calls is gated on whether the caller supplied `temperature_arrays`).

## Initial-condition anchors

A coupled run's first iteration is the hardest. The structure solve and the energetics initial entropy must agree on $T(r)$, $P(r)$, $R_\mathrm{cmb}$, and $M_\mathrm{core}$ before SPIDER or Aragog takes its first time step. PROTEUS supports several IC modes; the production default is `adiabatic_from_cmb` (Stage 1a lock for the UnifyCoupling paper), with `liquidus_super` as the most physically grounded option for hot magma-ocean ICs.

`liquidus_super` mode anchors the CMB temperature to the MgSiO$_3$ liquidus at the converged $P_\mathrm{cmb}$, plus a configurable excess representing super-liquidus thermal state:

$$
T_\mathrm{cmb} = T_\mathrm{liq}^\mathrm{Fei2021}(P_\mathrm{cmb}) + \Delta T_\mathrm{super}
$$

where $T_\mathrm{liq}^\mathrm{Fei2021}$ is the Belonoshko+2005 / Fei+2021 piecewise Simon-Glatzel liquidus exposed via `zalmoxis.melting_curves.paleos_liquidus`, and `delta_T_super` defaults to 500 K (set to 0 to anchor the IC adiabat exactly at the liquidus).
Zalmoxis then integrates the adiabat *upward* from $(R_\mathrm{cmb}, T_\mathrm{cmb})$ rather than downward from a surface anchor.

Why this matters: a surface-anchored adiabat gets the deep-mantle $T$ wrong by hundreds of K when the surface $T$ is uncertain (atmospheres at 2000 to 3500 K), but the deep-mantle $T_\mathrm{cmb}$ is well-constrained by the liquidus plus a physical excess.
Anchoring upward fixes the well-known quantity and lets the surface $T$ float to whatever the radiative balance demands.

On the very first call, before Zalmoxis has populated `hf_row['P_cmb']`, the wrapper estimates $P_\mathrm{cmb}$ from the Noack & Lasbleis (2020) mass-aware scaling.
An Earth-only fixed $P_\mathrm{cmb}$ would bias the iter-0 $T_\mathrm{cmb}$ at 5 $M_\oplus$ by hundreds of K; the mass-aware fallback keeps super-Earth ICs anchored at the right deep-mantle temperature.
After one round-trip through Zalmoxis the converged $P_\mathrm{cmb}$ replaces the fallback estimate.

The other modes (`adiabatic`, `adiabatic_from_cmb`, `accretion`, `isentropic`) follow analogous strategies.
`adiabatic` is surface-anchored from `tsurf_init`.
`accretion` solves the structure adiabatically and then overlays White & Li (2025) accretion thermal state with `f_accretion`, `f_differentiation` partitioning, populating `T_cmb_initial`, `T_surf_accr`, `DeltaT_accretion`, `DeltaT_differentiation`, `DeltaT_adiabat`, and the `_initial_T_profile` array consumed by Aragog's `_set_entropy_ic`.
`isentropic` (CHILI protocol) means the energetics solver consumes `ini_entropy` directly, decoupling the structure-solve T(r) from the entropy IC.

## Initial-condition equilibration loop

A coupled run's first iteration is the hardest. The structure solve and the energetics initial entropy must agree on $T(r)$, $P(r)$, $R_\mathrm{cmb}$, and $M_\mathrm{core}$ before SPIDER or Aragog takes its first time step, and the volatile partitioning must agree with the structure-derived $M_\mathrm{mantle}$. PROTEUS runs a CALLIOPE + Zalmoxis loop in `proteus.interior_energetics.wrapper.equilibrate_initial_state()`:

```python
for i in range(equilibrate_max_iter):
    R_old, P_old = hf_row['R_int'], hf_row['P_surf']

    # 1. Volatile partitioning: recompute elemental targets and
    #    run CALLIOPE to get atmosphere/melt distribution.
    calc_target_elemental_inventories(dirs, config, hf_row)
    run_outgassing(dirs, config, hf_row)

    # 2. Re-compute structure with updated composition. The
    #    VolatileProfile is rebuilt inside zalmoxis_solver from hf_row.
    cmb_radius, spider_mesh_file = zalmoxis_solver(
        config, outdir, hf_row, num_spider_nodes=num_spider_nodes,
    )

    # 3. M_mantle is not written by zalmoxis_solver; recompute it so
    #    the next CALLIOPE call sees an up-to-date mantle reservoir.
    hf_row['M_mantle'] = hf_row['M_int'] - hf_row['M_core']

    # 4. Convergence check on the two outputs that close the
    #    structure-outgassing loop.
    delta_R = abs(R_new - R_old) / R_old
    delta_P = abs(P_new - P_old) / P_old
    if delta_R < tol and delta_P < tol:
        break
```

The reason this loop exists: the first CALLIOPE call needs $M_\mathrm{mantle}$ to solve volatile partitioning, but $M_\mathrm{mantle}$ comes from Zalmoxis.
The first Zalmoxis call needs $P_\mathrm{surf}$ and the dissolved-volatile inventory to solve structure, but those come from CALLIOPE.
The two converge by iteration; with `equilibrate_tol = 0.01`, $R_\mathrm{int}$ and $P_\mathrm{surf}$ each change by less than 1% per iteration at the fixed point.

After equilibration, `generate_spider_tables()` regenerates the SPIDER P-S lookup tables against the final composition, so SPIDER's first time step (and, when Aragog is the energetics module, Aragog's `_verify_entropy_ic` path) reads tables consistent with the equilibrated structure. The mesh path is also published into `dirs['spider_mesh']`, with a `.prev` copy created so that the very first dynamic-update fall-back has a valid baseline if Zalmoxis fails on iteration 1 of the main loop.

The equilibration loop runs only when `interior_struct.zalmoxis.equilibrate_init = true` (default). With `equilibrate_init = false`, the wrapper performs a single Zalmoxis call and trusts the user's TOML to be self-consistent. Practically, `equilibrate_init = false` is reserved for restart-style runs where the IC file already contains a converged structure and outgassing state, or for benchmark configs where bypassing the loop is part of the test protocol.

Two convergence pathologies show up at the equilibration stage and indicate a deeper config problem rather than a tolerance issue.

First, alternating $R_\mathrm{int}$ across iterations (oscillation rather than damped approach) usually means the volatile EOS extension and the dry mantle EOS produce inconsistent compressibility at the IC P-T, often when an ice-layer EOS is enabled with an inappropriate `mantle_mass_fraction`.

Second, $P_\mathrm{surf}$ stuck at a hard ceiling (typically 1 GPa from the fallback path inside `_get_target_surface_pressure`) means CALLIOPE is producing a runaway pressure that the Zalmoxis structure cannot accommodate; this typically points at a `volatile_mode = "elements"` budget inconsistent with the planet's temperature mode (for instance, a 100-Earth-ocean H budget at $T_\mathrm{surf} = 4000$ K with a non-soluble fO$_2$ buffer).

## Dynamic update triggers

Re-solving Zalmoxis costs 5 to 60 seconds depending on mass and EOS, which is non-negligible if the coupling loop runs $10^4$ to $10^5$ iterations. PROTEUS gates Zalmoxis re-solves on a small set of conditions evaluated at every iteration, composed with OR plus a time-floor AND:

```python
fire_resolve = (
    (mesh_shift_active AND elapsed >= mesh_convergence_interval)
    OR (elapsed >= update_interval)
    OR (Time - last_successful_struct_time >= update_stale_ceiling)
    OR (|Phi_global - last_Phi| >= update_dphi_abs)
    OR (|T_magma - last_Tmagma| / last_Tmagma >= update_dtmagma_frac)
    OR (|d w_volatile| / w_volatile >= 0.05)   # H2 / H2O dissolved fractions
) AND (elapsed >= update_min_interval)
```

| Trigger | Default | Purpose |
|---|---|---|
| `update_interval` | $10^9$ yr | Ceiling on time since last call. The default effectively disables the ceiling for paper runs that gate updates only on the dynamical triggers below; production CHILI runs use $10^4$ to $10^5$ yr. Setting `update_interval = 0` disables dynamic updates entirely. |
| `update_min_interval` | 0 yr | Floor on time since last call. Prevents thrashing when several triggers fire on consecutive iterations. The non-routine triggers (mesh convergence, stale ceiling) bypass this floor. |
| `update_dphi_abs` | 0.05 | Phase-change trigger on $\|\Delta\Phi_\mathrm{global}\|$. |
| `update_dtmagma_frac` | 0.05 | Thermal trigger on $\|\Delta T_\mathrm{magma}\| / T_\mathrm{magma}$. |
| `update_stale_ceiling` | $2.5 \times 10^4$ yr | Recovery trigger: if the time since the last *successful* re-solve (anchored on `interior_o.last_successful_struct_time`, not on every call) exceeds this, fire regardless of the dynamical state. Without this, a streak of fall-backs can keep `last_struct_time` advancing while the cached structure stretches across an entire `update_interval` window. Set to 0 to disable. |
| `mesh_convergence_interval` | 10 yr | When `mesh_max_shift` clamps the per-update radius shift, the resulting partial blend is itself a non-converged state. The wrapper sets `mesh_shift_active = True`, and on the next iteration this short-elapsed trigger fires (bypassing `update_min_interval`) so the mesh continues to converge. |
| Composition (H$_2$, H$_2$O) | $\|\Delta w / w\| \ge 0.05$ | Triggers when the dissolved mass fraction of H$_2$ or H$_2$O shifts by 5% of its previous value, since the volatile-profile-extended mantle EOS density depends on $w$. |

The triggers are evaluated only outside the init stage. During init equilibration, `equilibrate_initial_state` calls Zalmoxis directly and bypasses the gating logic entirely.

The trigger semantics deserve a worked example.
Imagine a 1 $M_\oplus$ wet run with `update_interval = 5e4` yr, `update_dphi_abs = 0.05`, `update_dtmagma_frac = 0.05`.
In the first $\sim 10^4$ yr after IC, $\Phi_\mathrm{global}$ drops fast (from 1.0 to roughly 0.7) and the $\mathrm{d}\Phi$ trigger fires every few hundred years.
As the rheological transition is approached (around $\Phi \sim 0.4$), $T_\mathrm{magma}$ falls steeply and the $\mathrm{d}T/T$ trigger takes over.
Below the rheological transition, the planet enters the long mushy plateau ($\Phi \sim 0.05$ to $0.1$) where neither $\Phi$ nor $T_\mathrm{magma}$ change much per iteration, and the `update_interval` ceiling (or `update_stale_ceiling` if a fall-back streak intervenes) becomes the dominant trigger.
This is the regime where the wall-time cost of Zalmoxis is most exposed; a too-tight `update_interval` can dominate the run wall, while a too-loose interval lets the structure drift away from the live $T(r)$.

## Mesh handover and blending

When `interior_energetics.module = "aragog"` (or `"spider"`) and `update_interval > 0`, Zalmoxis writes a fresh mesh file `zalmoxis_output.dat` (5-column TSV: `r`, `P`, `rho`, `g`, `T`) on each re-solve. Aragog reads this file inside `solver.reset()` to rebuild its mass-coordinate mesh. SPIDER reads a separately-written mesh file produced by `write_spider_mesh_file`, which includes the same columns sub-sampled to SPIDER's basic-node count.

Two contracts must hold at the file handover boundary, enforced by `validate_zalmoxis_output_schema`:

1. **Top-of-mantle radius equality**: `r_file[-1] == hf_row['R_int']` to relative tolerance $10^{-6}$. This is essentially exact because both come from the same `planet_radius` variable inside `zalmoxis_solver`; a violation indicates file corruption, column-swap, or truncation.
2. **Mantle mass conservation**: $\int 4 \pi r^2 \rho \, dr$ over the mantle = `M_int - M_core` to relative tolerance $5 \times 10^{-2}$. The 5% tolerance is loose because two legitimate noise sources stack: the integrator-method gap (RK45-with-substeps versus shell-trapezoid post-write reintegration, $\sim 1$%), and `blend_mesh_files` post-modifying the file when the per-call radius shift exceeds `mesh_max_shift` (up to 5% drift between the on-disk file and the unblended scalars in `hf_row`). The tight $< 0.1$% conservation contract lives elsewhere in the wrapper's mass-anchor check.

`mesh_max_shift` (default 0.05 = 5%) caps the fractional radius shift between consecutive Zalmoxis writes.
The motivation is that SPIDER's BDF integrator and Aragog's CVODE integrator both penalise abrupt mesh changes; a 30% radius update from one iteration to the next during the rheological transition can spike `T_magma` and `Phi_global` discontinuously, kicking the entropy solver onto a wrong P-S branch.
Capping the per-update shift forces Zalmoxis and the energetics solver to converge across several iterations.
The mesh-convergence trigger (above) ensures the unfinished blend gets refined on a fast cadence (default 10 yr) instead of waiting a full `update_interval`.

!!! note "Resume + ULP radius drift"
    Aragog's `_validate_eos_radius_range` in `aragog/src/aragog/solver/entropy_solver.py` (lines ~91-110) checks that the EOS-table radius range agrees with the live mesh `inner_radius` / `outer_radius`. The check uses a relative tolerance $\max(1\,\mathrm{m},\ 10^{-9} \cdot \mathrm{span})$ against floating-point comparisons. When resuming from a saved snapshot, Zalmoxis recomputes `inner` and `outer` from the live planet state while the EOS table radii are frozen at the launch-time mesh; single-ULP drift between the two used to trip a strict `<` / `>` comparison. The current code tolerates the drift, but if you see `External EOS radius range [..., ...] m inconsistent with mesh bounds [..., ...] m` on a resume, regenerate the mesh by re-running Zalmoxis with the current planet radius rather than reusing the stale table.

## EOS table generation

Zalmoxis writes EOS tables for both energetics modules as part of the structure solve.

**SPIDER P-S tables.** `generate_spider_tables(config, outdir)` in `interior_struct/zalmoxis.py` produces P-S lookup tables for density, temperature, heat capacity, thermal expansivity, and the adiabatic gradient, plus solidus/liquidus phase boundaries in S(P) form. The path supports two PALEOS layouts:

1. `paleos_unified` (for example `PALEOS:MgSiO3`): single P-T table covering both phases plus the mushy zone. Solidus is derived from `mushy_zone_factor * liquidus`.
2. `PALEOS-2phase:<solid>` (for example `PALEOS-2phase:MgSiO3`): separate solid + liquid tables. Phase boundaries are sampled at the PALEOS-liquidus temperature from each phase table directly; `mushy_zone_factor` is ignored (no analytic mushy zone exists for two-phase, the gap between solid-table-top and liquid-table-bottom defines the latent heat).

The default resolution `lookup_nP = 1350`, `lookup_nS = 280` is calibrated against SPIDER's spline tolerance; halving cuts table-generation wall by roughly $3\times$ but introduces visible interpolation artifacts in the SPIDER adiabat. The `_rectangularize_spider_ps_file` helper snaps the per-slice P drift in SPIDER's bundled tables to a strictly rectangular grid, because Aragog's loader uses `np.unique(P_all)` and scipy's `RegularGridInterpolator` rejects the $\sim 10^{-8}$ relative drift of the un-snapped tables.

**Aragog P-T tables.** `generate_aragog_pt_tables[_2phase]` (in `zalmoxis.eos_export`, called from `proteus/interior_energetics/aragog.py` lines ~485-530) writes the P-T tables Aragog reads in its non-uniform grid loader. The output goes to `<outdir>/data/aragog_pt/` as `{density,temperature,heat_capacity,adiabat_temp_grad,thermal_exp}_{melt,solid}.dat`, default resolution $200 \times 200$, P-range $[10^5\,\mathrm{Pa},\ \min(10^{13}\,\mathrm{Pa},\ 150 \cdot M_\oplus + 200\,\mathrm{GPa})]$. The two-phase variant takes separate solid and liquid PALEOS tables; the unified-fallback path uses a single PALEOS table for both phases.

!!! warning "Aragog tables must be strictly rectangular"
    Aragog's loader builds a `RegularGridInterpolator` over the (P, T) grid. Phase-filtering the input PALEOS table (for example, masking out points inside the mushy zone before writing) breaks the rectangularity assumption: scipy then silently falls back to unstructured (linear-ND) interpolation, which is roughly two orders of magnitude slower per call. The table generators in `zalmoxis.eos_export` write the full rectangle for both solid and melt files; do not inject a phase mask between them and disk.

## Volatile-profile coupling

When `dry_mantle = false`, each Zalmoxis re-solve builds a `VolatileProfile` from the dissolved-volatile masses in `hf_row`. The wrapper computes per-phase mass fractions for each species with a Zalmoxis EOS table (currently `Chabrier:H` for H$_2$ and `PALEOS:H2O` for water) and extends the configured mantle EOS string with placeholder fractions:

```text
"PALEOS:MgSiO3"  ->  "PALEOS:MgSiO3:0.97+PALEOS:H2O:0.02+Chabrier:H:0.01"
```

Inside Zalmoxis, `LayerMixture` mixes per-component density via the phase-aware suppressed harmonic mean (see [multi-material mixing](mixing.md)). At each radial shell the volatile profile re-evaluates per-phase mass fractions weighted by $\phi(r)$ from the phase routing, producing a $\phi$-aware structural density that smoothly transitions from a wet liquid mantle to a drier solid mantle as the planet crystallises.

The default `dry_mantle = true` skips `build_volatile_profile()` entirely. The structure is then determined by the canonical solid + liquid mantle tables alone, regardless of dissolved inventory. This is the production setting for paper runs and for cleanly-decoupled module comparisons; it isolates the volatile contribution to the structure from all other physics. Volatile partitioning still happens in the outgassing module, the flag only controls whether dissolved volatile mass shifts the structure-side EOS density.

When `global_miscibility = true`, the wrapper additionally calls `solve_miscible_interior()` instead of the standard structure solve, iterating the H$_2$-MgSiO$_3$ solvus radially until the bulk H$_2$ mass at the interior matches the dissolved inventory. The outputs `R_solvus`, `T_solvus`, `P_solvus`, `X_H2_int` are written to `hf_row`; AGNI subsequently reads the binodal surface as its lower boundary instead of the magma-ocean surface (see [coupling pitfalls](#coupling-pitfalls)).

## Outer solver and JAX path

Zalmoxis offers two outer mass-radius solvers, `picard` and `newton`, with `newton` as the default in PROTEUS coupled runs (since 2026-04-27). The Picard path is a damped fixed-point iteration $R_{n+1} = R_n \cdot (M_\mathrm{target} / M_n)^{1/3}$, clamped to $[0.5, 2.0]$ and damped by 0.5; it converges robustly on cool Earth-mass profiles but can hit a basin attractor on hot fully-molten profiles, where the iteration oscillates around the true $R$ without escaping. The Newton path uses a central-difference $\mathrm{d}M/\mathrm{d}R$ estimate combined with brentq bracketing on $f(R) = M(R) - M_\mathrm{target}$, and converges on the hot-profile cases that trap Picard.

Newton requires tighter integrator tolerances than Picard: the Picard default `relative_tolerance = 1e-5` produces $\sim 10^{-3}$ noise in $M(R)$ that swamps the central-difference estimate. The wrapper auto-applies `relative_tolerance = 1e-9` and `absolute_tolerance = 1e-10` whenever `outer_solver = "newton"` is selected, and only forwards the Newton-specific knobs to Zalmoxis when the Newton path is active, so a Picard run sees an unchanged config dict.

The structure solve runs through Zalmoxis's JAX + diffrax path by default (`use_jax = true`), with an opt-in Anderson Type-II Picard accelerator (`use_anderson = true`). The JAX path has a known subtlety in coupled mode: the JAX RHS is P-indexed for adiabat tabulation, but PROTEUS's `temperature_function(r, P)` ignores $P$ (it interpolates an interior-solver T(r) from r alone), and the P-indexed JAX path collapses on a P-ignoring callable. The wrapper detects two argument styles:

| Argument style | Caller | JAX behaviour |
|---|---|---|
| `temperature_function: f(r, P) -> T` only | `equilibrate_initial_state` and PROTEUS init | JAX RHS path collapses; wrapper auto-disables `use_jax` and `use_anderson` and falls back to numpy. The one-time cost (~70 s each, 2 to 4 calls) is negligible against a 3 to 4 h full run. |
| `temperature_arrays: (r_arr, T_arr)` | Main-loop `update_structure_from_interior` | r-indexed JAX RHS path, no collapse. The wrapper sorts `r_arr` strictly ascending before passing it, since `jnp.interp` requires monotonic `xp`. |

When `temperature_arrays` is supplied, the wrapper deliberately does *not* also pass `temperature_function`, even though Zalmoxis's solver accepts both. The numpy Picard density update inside `_solve` uses the callable for per-node temperature lookup, which trips PALEOS phase-boundary clamps and forces $\sim 75\times$ more inner-Picard iterations to converge. Dropping the callable lets Zalmoxis use its internal linear T profile for Picard while the JAX integration still uses the accurate array-based T(r). On a representative bench this collapsed the per-call wall from 156 s to 2 s for the same JAX arrays.

## Coupling pitfalls

The function-call coupling is robust at the file boundary, but several edges have known traps. New code paths should re-check these before claiming behavioural neutrality.

!!! warning "`prevent_warming` clamp is energy non-conserving"
    `config.planet.prevent_warming` (default `false`) gates an early ratchet `T_magma = min(new, prev)` in `interior_energetics/wrapper.py`. The clamp was originally intended for strictly-cooling regimes but **silently destroys the warming half of any heat-pump cycle** in a coupled magma-ocean run, producing an apparent "T_magma plateau" that is in fact an energy-leak bug. As of 2026-05-03 the clamp is documented as non-conserving and must remain at the `false` default for production runs. A separate runaway-T fallback (`interior_o.ic == 2` recovery path) remains active independently and is not affected. If you see `T_magma` byte-pinned across hundreds of consecutive iterations with `F_dil`-style residuals (or any per-call energy residual) growing without bound, check the `prevent_warming` flag first.

### `core_density` echo-back

When Zalmoxis runs as the structure module with SPIDER as the energetics module, Zalmoxis writes a self-consistent `hf_row['core_density'] = M_core / (4/3 \pi R_\mathrm{cmb}^3)` from the converged structure (`interior_struct/zalmoxis.py` line ~1830).
SPIDER's call sequence in `interior_energetics/spider.py` then passes `-rho_core <value>` to the SPIDER C binary (line ~1033), and SPIDER's internal core thermal-budget routine treats this density as authoritative.
The two values are usually consistent because both are derived from the same converged structure, but a divergence is possible if the SPIDER mesh and the Zalmoxis output drift (for instance, after a fall-back where `_saved_structure` has stale `M_core` while the on-disk mesh was rolled back to a `.prev` from one iteration earlier).
The wrapper-level mass-anchor check `|M_int / M_int_target - 1| < 3 \times 10^{-3}` (`wrapper.py` lines ~1752-1768) catches gross mismatches by raising `RuntimeError`, which routes through the fall-back path.
The check is loose enough to tolerate the $\sim 0.3$% margin built into Zalmoxis's outer solver tolerance but tight enough that a column-swap or mesh-blending bug shows up.

### `hf_row` temporary overrides for atmosphere BC

When `global_miscibility = true` and Zalmoxis writes a sub-surface solvus, `proteus.py` overrides `T_surf`, `P_surf`, `R_int`, and `T_magma` in `hf_row` for the duration of the atmosphere step (so AGNI sees the binodal surface as its lower boundary), and restores the originals immediately after `run_atmosphere` returns.
Any new submodule that reads these keys *during* the atmosphere step gets the binodal values; outside the atmosphere step it gets the magma-ocean values.
New code that reads structure quantities should read them where the override is not active, or at minimum check `R_solvus` to disambiguate.

### DELETED `dilatation` field

The `interior_energetics.aragog.dilatation` configuration field, the `F_dil` and `Q_dil_W` and `step_dE_Q_dil_J` helpfile columns, and the runtime gate that switched on volumetric-work heating were removed from PROTEUS in commits `706ff56f`, `b2241704`, and `3e5d7641` (all 2026-05-04).
The corresponding source term was also removed from Aragog.
The current code path has no `dilatation` schema field, no warning-on-deprecation shim, and no `Q_dil` column in the helpfile.
Do not document, configure, or pattern-match this field; any reference in old configs is silently ignored by the attrs schema since the field no longer exists.
If you see `dilatation = ...` in an old TOML, delete the line.

### Determinism

Coupled runs occasionally hit a class of numerical-fragility failures where the same config produces different results across launches: $\sim 10^{-7}$ floating-point noise in early helpfile rows compounds through Aragog's tight tolerances and lands the solver on a wrong P-S branch within $\sim 15$ iterations.
PROTEUS pins BLAS thread counts at `cli.py` import time (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`); BLAS is not the only source of reduction-order non-determinism, JAX/XLA has its own threading model independent of OpenBLAS.
The `--deterministic` flag intercepts itself in raw `sys.argv` *before* any heavy imports, sets `JAX_ENABLE_X64=1` and `XLA_FLAGS=--xla_cpu_enable_fast_math=false`, and self-re-execs once with a sentinel env var to prevent infinite re-exec.
Use sparingly: the flag has a small per-step cost (XLA fast-math disabled) and most coupled runs converge cleanly without it.
It is intended for configs that show noise-floor divergence between launches, typically wet 1 $M_\oplus$ at IW+4 or reduced 1 $M_\oplus$ at IW-2.

## See also

- [PROTEUS coupling how-to](../How-to/proteus_coupling.md): TOML recipe and parameter reference.
- [Process flow](process_flow.md): Zalmoxis's internal three-loop solver (structure ODE / density Picard / mass-radius outer).
- [Code architecture](code_architecture.md): Zalmoxis package layout.
- [Equations of state](eos_physics.md): PALEOS unified tables and the EOS dispatch model.
- [Multi-material mixing](mixing.md): phase-aware suppressed harmonic mean used inside the volatile-profile path.

The wrapper itself lives in the PROTEUS repository, not in Zalmoxis. The single source of truth for the symbol-level API is rendered from PROTEUS source via mkdocstrings:

- [`proteus.interior_struct.zalmoxis`](https://proteus-framework.org/PROTEUS/Reference/api/interior_struct_zalmoxis.html): `zalmoxis_solver`, `load_zalmoxis_configuration`, `validate_zalmoxis_output_schema`, `build_volatile_profile`, `generate_spider_tables`.
- [`proteus.interior_energetics.wrapper`](https://proteus-framework.org/PROTEUS/Reference/api/interior_energetics_wrapper.html): `equilibrate_initial_state`, `solve_structure`, `update_structure_from_interior`, the mass-anchor and fall-back logic.
- [`proteus.config._struct`](https://proteus-framework.org/PROTEUS/Reference/api/config_struct.html): the attrs schema for `[interior_struct]` and `[interior_struct.zalmoxis]`. Every TOML key from the [coupling how-to](../How-to/proteus_coupling.md) maps to a field here.
- [PROTEUS framework documentation](https://proteus-framework.org/PROTEUS): top-level entry.
