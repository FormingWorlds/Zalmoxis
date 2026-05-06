<h1 align="center">
    <a href="https://proteus-framework.org">
    <div>
        <img src="https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/PROTEUS_white.png#gh-light-mode-only" style="vertical-align: middle;" width="60%"/>
        <img src="https://raw.githubusercontent.com/FormingWorlds/PROTEUS/main/docs/assets/PROTEUS_black_nobkg.png#gh-dark-mode-only" style="vertical-align: middle;" width="60%"/>
    </div>
    </a>
</h1>

# Zalmoxis in the PROTEUS framework

Zalmoxis is the **interior structure module** of [PROTEUS](https://proteus-framework.org/PROTEUS) (/ˈproʊtiəs/, PROH-tee-əs), a modular Python framework for the coupled evolution of the atmospheres and interiors of rocky planets and exoplanets.
A schematic of PROTEUS components and corresponding modules is shown below.

<p align="center">
      <img src="assets/schematic_round.png" style="max-width: 90%; height: auto;"></br>
      <b>Schematic of PROTEUS components and corresponding modules.</b> </br>
</p>

You can find the documentation of each PROTEUS module in the sidebar.

---

## Where Zalmoxis sits in a coupled run

In a PROTEUS coupled run, three submodules deliver mantle physics in a fixed per-iteration order:

| Submodule | Role | State variable |
|---|---|---|
| **Outgassing** (CALLIOPE / atmodeller) | Volatile partitioning between magma ocean and atmosphere | $X_i^\mathrm{melt}$, $X_i^\mathrm{atm}$ |
| **Zalmoxis** | Static structure: hydrostatic equilibrium, mass-radius, density, gravity, $R_\mathrm{cmb}$, $P_\mathrm{cmb}$ | $\rho(r)$, $g(r)$, $P(r)$ |
| **Aragog** or **SPIDER** | Thermal evolution: entropy ODE, $T(r)$ trajectory, surface heat flux | $S(r)$, $T(r)$ |

Zalmoxis is the slowest of the three, so PROTEUS gates re-solves on physical-state-change triggers ($\Delta \phi$, $\Delta T_\mathrm{magma}$, time-since-call ceilings) rather than calling it every iteration.

The atmosphere is closed by AGNI, JANUS, or a dummy radiative-balance module, depending on the configuration; the surface temperature couples back into the entropy ODE.

---

## Quick links into the coupling docs

The two pages dedicated to PROTEUS coupling are:

<div class="grid cards" markdown>

-   :material-rocket-launch: **How to couple Zalmoxis to PROTEUS**

    Practical TOML recipe: minimal `[interior_struct]` block, IC modes (including the new `liquidus_super` anchor), Newton outer solver, equilibration, update triggers, mesh smoothing, JAX path, the `--deterministic` flag, common pitfalls.

    [Go to the how-to](How-to/proteus_coupling.md)

-   :material-book-open-variant: **Theory of the coupling**

    Per-iteration control flow with diagram, the dict the wrapper passes to `zalmoxis.solver.main()`, the Fei+2021-anchored `liquidus_super` IC and the Noack & Lasbleis (2020) mass-aware super-Earth fallback, the `zalmoxis_output.dat` schema contract, volatile-profile $\phi(r)$ blending, the Newton outer-solver default, the JAX path subtleties.

    [Go to the explainer](Explanations/proteus_coupling.md)

</div>

For the PROTEUS-side documentation (config schema, orchestrator behavior, atmosphere modules, etc.), see [proteus-framework.org/PROTEUS](https://proteus-framework.org/PROTEUS).

---

## Standalone vs PROTEUS-coupled

Zalmoxis is also a fully self-contained tool for one-off mass-radius modelling and parameter sweeps. The two modes are configured independently:

| Mode | Entry point | Configuration |
|---|---|---|
| **Standalone** | `python -m zalmoxis -c input/<cfg>.toml` | TOML sections `[InputParameter]`, `[AssumptionsAndInitialGuesses]`, `[EOS]`, `[Calculations]`, `[IterativeProcess]`, `[PressureAdjustment]`, `[Output]`. Documented under [Configuration](How-to/configuration.md). |
| **PROTEUS-coupled** | `proteus start -c <run>.toml` | PROTEUS-side TOML sections `[planet]`, `[interior_struct]`, `[interior_struct.zalmoxis]`, plus the relevant `[interior_energetics]`, `[outgas]`, `[atmos]` blocks. The Zalmoxis-side sections above are not read. Documented in [Coupling to PROTEUS](How-to/proteus_coupling.md). |

The EOS identifier strings (`PALEOS:iron`, `Seager2007:MgSiO3`, `Chabrier:H`, `Analytic:graphite`, etc.) are common to both modes; only the TOML scaffolding around them differs.

---

## Where the wrapper code lives

Zalmoxis is invoked from PROTEUS via:

- `proteus/interior_struct/zalmoxis.py`: the wrapper that builds the call-time dict, calls `zalmoxis.solver.main()`, writes the Aragog mesh file, and validates the schema contract on `zalmoxis_output.dat`.
- `proteus/config/_struct.py`: the attrs-based schema for `[interior_struct.zalmoxis]`. All knobs documented in the how-to map back to fields here.
- `proteus/interior_energetics/wrapper.py`: hosts `equilibrate_initial_state()`, the pre-main-loop CALLIOPE + Zalmoxis convergence loop.

These files live in the [PROTEUS repository](https://github.com/FormingWorlds/PROTEUS), not in Zalmoxis. Their per-symbol API documentation is rendered in the PROTEUS docs; this site documents the Zalmoxis side of the contract.
