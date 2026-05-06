# Configuration file

!!! info "Standalone vs PROTEUS-coupled mode"
    This page documents the TOML schema used by **standalone** Zalmoxis (`python -m zalmoxis`).
    Inside the PROTEUS framework, none of the sections below (`[InputParameter]`, `[AssumptionsAndInitialGuesses]`, `[EOS]`, `[Calculations]`, `[IterativeProcess]`, `[PressureAdjustment]`, `[Output]`) are read.
    PROTEUS configures Zalmoxis through its own `[interior_struct]` and `[interior_struct.zalmoxis]` blocks; see [Coupling to PROTEUS](proteus_coupling.md).
    The EOS-identifier table below (`PALEOS:iron`, `WolfBower2018:MgSiO3`, etc.) is still authoritative for both modes; only the surrounding TOML scaffolding differs.

Zalmoxis uses [TOML](https://toml.io/en/) to structure its configuration file. The default is `default.toml` in the `input/` directory.

The configuration file defines all parameters needed to run the planetary interior structure model: planet properties, equation of state (EOS) selection for each structural layer, numerical solver settings, and output options. The sections below document each parameter.

## Configuration sections

### `InputParameter`

Defines the basic input for the planetary model.

| Parameter | Type | Unit | Validated range | Description |
|---|---|---|---|---|
| `planet_mass` | float | Earth masses | 0.1 to 50 | Total planet mass ($M_\oplus = 5.972 \times 10^{24}$ kg). The solver has been validated across this range with all EOS options. PALEOS tables extend to 100 TPa, which corresponds to ~50 $M_\oplus$ central pressure. Masses outside this range produce a warning. |

### `AssumptionsAndInitialGuesses`

Initial guesses and assumptions for the planetary structure.

| Parameter | Type | Unit | Validated range | Description |
|---|---|---|---|---|
| `core_mass_fraction` | float | -- | 0.01 to 0.999 | Core mass as a fraction of total mass. Also used in the Seager et al. (2007) scaling relation for the initial radius guess. Earth: ~0.325. Extreme values (pure iron at 0.999, pure rock at 0.01) converge. |
| `mantle_mass_fraction` | float | -- | 0 to 0.99 | Mantle mass as a fraction of total mass. Set to 0 for a 2-layer model where the mantle fills the remainder (1 - `core_mass_fraction`). Must be > 0 when using a 3-layer model with an ice layer. |
| `temperature_mode` | string | -- | -- | Temperature profile type. One of: `"isothermal"`, `"linear"`, `"prescribed"`, `"adiabatic"`. See [Temperature profiles](#temperature-profiles). |
| `surface_temperature` | float | K | 300 to 5000 | Surface temperature. Used when `temperature_mode` is `"isothermal"`, `"linear"`, or `"adiabatic"`. Validated across 300-5000 K for all compositions. Values above 5000 K produce a warning. |
| `center_temperature` | float | K | -- | Central temperature. Used when `temperature_mode` is `"linear"` or `"adiabatic"` (initial guess). The adiabat solver determines the actual center temperature self-consistently. |
| `temperature_profile_file` | string | -- | -- | Filename (relative to `input/`) containing a prescribed radial temperature profile. Used when `temperature_mode` is `"prescribed"`. |

#### Temperature profiles

Temperature profiles are only relevant when using a temperature-dependent EOS (i.e., `WolfBower2018:MgSiO3`, `RTPress100TPa:MgSiO3`, `PALEOS-2phase:MgSiO3`, `PALEOS:iron`, `PALEOS:MgSiO3`, `PALEOS:H2O`, or `Chabrier:H`). For all other EOS choices, a fixed 300 K is assumed internally and these parameters are ignored.

- **`"isothermal"`**: Constant temperature equal to `surface_temperature` at all radii.
- **`"linear"`**: Linear interpolation from `center_temperature` at $r = 0$ to `surface_temperature` at $r = R$.
- **`"prescribed"`**: Reads a temperature profile from `temperature_profile_file`. The file must contain one temperature value per line (in K), ordered from center to surface, with the same number of entries as `num_layers`.
- **`"adiabatic"`**: Computes the adiabatic temperature profile by integrating adiabatic gradients from the EOS, starting at `surface_temperature` and integrating inward. The initial guess is a linear profile between `surface_temperature` and `center_temperature`. **Requires at least one T-dependent EOS layer.** For `WolfBower2018` and `RTPress100TPa`, pre-tabulated $(dT/dP)_S$ gradient files are used. For `PALEOS-2phase`, the dimensionless adiabatic gradient $\nabla_{\mathrm{ad}}$ is read from separate solid/liquid tables and converted via $dT/dP = \nabla_{\mathrm{ad}} \cdot T/P$, with phase-aware weighting using the configured solidus and liquidus curves. For unified PALEOS tables (`PALEOS:iron`, `PALEOS:MgSiO3`, `PALEOS:H2O`), $\nabla_{\mathrm{ad}}$ is read directly from the single table (all stable phases included), and no external melting curves are needed. The adiabat is parameterized as $T(P)$ internally, ensuring thermodynamic consistency during the Brent pressure solver's bracket search. For T-independent EOS layers (e.g., `Seager2007:iron` core), temperature is held constant (isothermal) through that layer. With `PALEOS:iron`, the core follows its own adiabat for the first time.

---

### `EOS`

Specifies the equation of state for each structural layer. Each layer is configured independently using a `"<source>:<composition>"` string.

```toml
[EOS]
core      = "Seager2007:iron"
mantle    = "WolfBower2018:MgSiO3"
ice_layer = ""  # Empty string = 2-layer model
```

**Fields:**

| Field | Required | Description |
|---|---|---|
| `core` | Yes | EOS for the innermost layer (iron core). |
| `mantle` | Yes | EOS for the mantle layer. |
| `ice_layer` | No | EOS for an outer volatile layer. If empty or omitted, the model uses a 2-layer structure (core + mantle). If set, the model uses a 3-layer structure and `mantle_mass_fraction` must be > 0 in `[AssumptionsAndInitialGuesses]`. |

#### Naming convention

All EOS identifiers follow the format `<source>:<composition>`, where:

- **`<source>`** identifies the data source or method: `Seager2007` (tabulated), `WolfBower2018` (tabulated, temperature-dependent), `RTPress100TPa` (extended T-dependent melt table), `PALEOS-2phase` (T-dependent with separate solid/liquid tables), `PALEOS` (unified T-dependent table with all stable phases), or `Analytic` (closed-form fit, no data files needed).
- **`<composition>`** identifies the material: `iron`, `MgSiO3`, `MgFeSiO3`, `H2O`, `graphite`, `SiC`.

#### Multi-material mixing

A single layer can contain multiple materials mixed by volume additivity. Use `+` to combine materials with mass fractions:

```toml
mantle = "PALEOS:MgSiO3:0.85+PALEOS:H2O:0.15"
```

Rules:

- Components are separated by `+`.
- Each component uses the format `<source>:<composition>:<mass_fraction>`.
- If the fraction is omitted (single-material backward compat), it defaults to 1.0.
- Mass fractions must sum to 1.0 (automatically normalized if not).
- Any EOS type can be mixed with any other (tabulated + analytic, T-dependent + T-independent).
- Density is computed via a **phase-aware suppressed harmonic mean**: each component is weighted by a smooth sigmoid function of its density before entering the harmonic mean. This prevents non-condensed volatiles (vapor, supercritical gas) from dominating the mixture. See the [mixing documentation](../Explanations/mixing.md) for the physics.
- For mixtures where all components are well above the sigmoid center (iron at $\rho > 7000$, MgSiO$_3$ at $\rho > 2500$ kg/m$^3$), the suppression is negligible and the result is identical to the standard harmonic mean.
- For adiabatic mode, $\nabla_{\mathrm{ad}}$ uses the same sigmoid-weighted average across components.
- Mixing fractions can be updated at runtime by PROTEUS/CALLIOPE without re-parsing the config.

!!! warning "Mixing T-dependent with T-independent EOS"
    Mixing a T-dependent EOS (e.g., `PALEOS:MgSiO3`) with a T-independent EOS (e.g., `Seager2007:H2O`) in the same layer is allowed but physically inconsistent: the T-independent component uses a fixed 300 K internally while the T-dependent component uses the adiabatic temperature. A warning is logged when this occurs.

!!! note "Phase-aware suppression of non-condensed volatiles"
    At high surface temperatures ($T_\mathrm{surf} > 2000$ K), H$_2$O in the mantle may be in vapor or low-density supercritical phase near the surface. The phase-aware suppression automatically excludes these low-density states from the structural density calculation, preventing unphysically inflated radii. The suppression is controlled by `condensed_rho_min` and `condensed_rho_scale` (see [Phase-aware mixing parameters](#phase-aware-mixing-parameters-multi-material-only) below).

#### Available EOS options

| EOS string | Source | Composition | Temperature | Data files required | Notes |
|---|---|---|---|---|---|
| `Seager2007:iron` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) | Fe (epsilon phase) | Fixed 300 K | Yes | Vinet EOS + DFT, tabulated $\rho(P)$. |
| `Seager2007:MgSiO3` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) | MgSiO3 perovskite | Fixed 300 K | Yes | 4th-order Birch-Murnaghan + DFT, tabulated $\rho(P)$. |
| `Seager2007:H2O` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) | Water ice (phases VII, VIII, X) | Fixed 300 K | Yes | Experimental + DFT, tabulated $\rho(P)$. |
| `WolfBower2018:MgSiO3` | [Wolf & Bower (2018)](https://www.sciencedirect.com/science/article/pii/S0031920117301449) | MgSiO3 (melt + solid) | T-dependent | Yes | RTpress EOS with phase-aware melting (solid EOS derived from [Mosenfelder et al. 2009](https://doi.org/10.1029/2008JB005900)). **Limited to $\leq 7\,M_\oplus$** (table max ~1 TPa; out-of-bounds pressures clamped). Requires `temperature_mode` configuration. Uses [Monteux et al.](https://doi.org/10.1016/j.epsl.2016.05.010) solidus/liquidus curves. |
| `RTPress100TPa:MgSiO3` | Extended RTpress melt table | MgSiO3 (melt + solid) | T-dependent | Yes | Extended melt EOS to 100 TPa ($P$: $10^3$ to $10^{14}$ Pa, $T$: 400 to 50000 K). Solid phase uses [Wolf & Bower (2018)](https://www.sciencedirect.com/science/article/pii/S0031920117301449) table (clamped at 1 TPa). **Limited to $\leq 50\,M_\oplus$**. Requires `temperature_mode` configuration. |
| `PALEOS-2phase:MgSiO3` | PALEOS (Zenodo 19680050, v1.1.0) | MgSiO3 (solid + liquid) | T-dependent | Yes | Separate solid and liquid tables providing density and $\nabla_{\mathrm{ad}}$ (P: 1 bar to 100 TPa, T: 300 to 100000 K, 150 ppd log-uniform grid). Enables phase-aware adiabatic temperature profiles using nabla_ad from both phases. **Limited to $\leq 50\,M_\oplus$**. Requires `temperature_mode` configuration. |
| `PALEOS:iron` | PALEOS (Zenodo 19000316) | Fe (5 phases) | T-dependent | Yes | Unified single-file table with all stable Fe phases (alpha-bcc, delta-bcc, gamma-fcc, epsilon-hcp, liquid). P: 1 bar to 100 TPa, T: 300 to 100000 K. Phase boundary encoded in table. **Limited to $\leq 50\,M_\oplus$**. Enables T-dependent core with its own adiabat. |
| `PALEOS:MgSiO3` | PALEOS (Zenodo 19000316) | MgSiO3 (6 phases) | T-dependent | Yes | Unified single-file table with all stable MgSiO3 phases (3 pyroxene, bridgmanite, postperovskite, liquid). P: 1 bar to 100 TPa, T: 300 to 100000 K. Phase boundary encoded in table. **Limited to $\leq 50\,M_\oplus$**. |
| `PALEOS:H2O` | PALEOS (Zenodo 19000316) | H2O (7 EOS) | T-dependent | Yes | Unified single-file table with all stable H2O phases (ice Ih to X, liquid, vapor, superionic). P: 1 bar to 100 TPa, T: 100 to 100000 K. Phase boundary encoded in table. **Limited to $\leq 50\,M_\oplus$**. Use as `ice_layer` for 3-layer models. |
| `Chabrier:H` | [Chabrier et al. (2019)](https://doi.org/10.3847/1538-4357/aaf99f), [Chabrier & Debras (2021)](https://doi.org/10.3847/1538-4357/abfc48) | Pure H$_2$ (molecular, atomic, ionized) | T-dependent | Yes | DirEOS2021 table for pure hydrogen. Grid: 121 $\times$ 441 ($\log T$, $\log P$), $T = 100$ to $10^8$ K, $P = 1$ Pa to $10^{22}$ Pa. Provides $\rho$, $\nabla_{\mathrm{ad}}$, $c_p$, $c_v$, $\alpha$, and phase identifier. Loaded through the same `paleos_unified` reader. Use as a mixing component in the mantle (e.g., `"PALEOS:MgSiO3:0.97+Chabrier:H:0.03"`) for sub-Neptune interiors. **No standalone mass limit** (pressure range far exceeds any planet). In H$_2$ dissociation zones ($T \sim 3000$ to $30000$ K), $\nabla_{\mathrm{ad}}$ is clamped at 0.100 in the source tables. |
| `Analytic:iron` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) Table 3 | Fe (epsilon) | Fixed 300 K | No | Modified polytrope: $\rho(P) = \rho_0 + c \cdot P^n$. |
| `Analytic:MgSiO3` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) Table 3 | MgSiO3 perovskite | Fixed 300 K | No | Modified polytrope. |
| `Analytic:MgFeSiO3` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) Table 3 | (Mg,Fe)SiO3 | Fixed 300 K | No | Modified polytrope. |
| `Analytic:H2O` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) Table 3 | Water ice | Fixed 300 K | No | Modified polytrope. |
| `Analytic:graphite` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) Table 3 | C (graphite) | Fixed 300 K | No | Modified polytrope. |
| `Analytic:SiC` | [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) Table 3 | Silicon carbide | Fixed 300 K | No | Modified polytrope. |

**Analytic EOS details.** The analytic options use the modified polytropic fit from Seager et al. (2007, Eq. 11, Table 3):

$$\rho(P) = \rho_0 + c \cdot P^n$$

where $\rho_0$ is the zero-pressure density, $c$ and $n$ are fitted constants. This approximation is valid for $P < 10^{16}$ Pa and reproduces the full tabulated EOS to 2 to 12% accuracy across all planetary pressures. The analytic EOS requires no external data files, making it useful for quick exploration and testing.

**Temperature-dependent EOS.** Three families of T-dependent EOS are available:

1. **Separate solid/liquid tables** (`WolfBower2018:MgSiO3`, `RTPress100TPa:MgSiO3`, `PALEOS-2phase:MgSiO3`): use separate $\rho(P, T)$ grids for solid and melt phases, with linear melt-fraction interpolation between external solidus and liquidus melting curves.
2. **Unified PALEOS tables** (`PALEOS:iron`, `PALEOS:MgSiO3`, `PALEOS:H2O`): single file per material containing all stable phases. The thermodynamically stable phase at each $(P, T)$ is encoded in the table, and the phase boundary (liquidus) is extracted automatically. No external melting curves needed. A configurable `mushy_zone_factor` controls the width of an artificial mushy zone below the liquidus.
3. **Chabrier H/He tables** (`Chabrier:H`): DirEOS2021 tables from [Chabrier et al. (2019)](https://doi.org/10.3847/1538-4357/aaf99f) and [Chabrier & Debras (2021)](https://doi.org/10.3847/1538-4357/abfc48). Same 10-column format as unified PALEOS. Intended for use as a mixing component in sub-Neptune mantles, combined with binodal (miscibility) suppression.

When any T-dependent EOS is assigned to any layer, the `temperature_mode`, `surface_temperature`, and `center_temperature` parameters in `[AssumptionsAndInitialGuesses]` become active.

#### Mushy zone factor (unified PALEOS only)

| Field | Required | Default | Description |
|---|---|---|---|
| `mushy_zone_factor` | No | `1.0` | Global default cryoscopic depression factor for all unified PALEOS tables. Controls the width of the artificial mushy (mixed solid+liquid) zone below the liquidus extracted from the table. Valid range: [0.7, 1.0]. `1.0` = no mushy zone (sharp phase boundary). `< 1.0` = solidus at this fraction of the liquidus temperature ($T_\mathrm{sol} = T_\mathrm{liq} \times f$). E.g., `0.8` gives a mushy zone similar to the Stixrude (2014) cryoscopic depression. Values below 0.7 are rejected (unphysically wide mushy zones cause solver instabilities). In the mushy zone, density is volume-averaged between solid-side and liquid-side table values. Only relevant for `PALEOS:iron`, `PALEOS:MgSiO3`, `PALEOS:H2O`. |
| `mushy_zone_factor_iron` | No | global value | Per-material override for `PALEOS:iron`. Same range and semantics as `mushy_zone_factor`. Only validated when `PALEOS:iron` is configured. |
| `mushy_zone_factor_MgSiO3` | No | global value | Per-material override for `PALEOS:MgSiO3`. |
| `mushy_zone_factor_H2O` | No | global value | Per-material override for `PALEOS:H2O`. |

Per-material overrides take precedence over the global default. If a per-material key is absent, the global `mushy_zone_factor` is used for that material. This allows independent control of phase boundary widths, for example using a wider mushy zone for silicate (`0.8`) while keeping a sharp boundary for iron (`1.0`):

```toml
mushy_zone_factor          = 1.0   # global default
mushy_zone_factor_MgSiO3   = 0.8   # wider mushy zone for silicate only
```

#### Phase-aware mixing parameters (multi-material only)

These parameters control the smooth sigmoid suppression that prevents non-condensed volatile components from dominating the harmonic-mean density in multi-material layers.
They have no effect on single-material layers.
See the [mixing documentation](../Explanations/mixing.md) for the physics.

| Field | Required | Default | Unit | Description |
|---|---|---|---|---|
| `condensed_rho_min` | No | `322.0` | kg/m$^3$ | Sigmoid center density. The default is the H$_2$O critical density (322 kg/m$^3$ at 647 K, 22.1 MPa). Components with density well below this are progressively excluded from the harmonic mean. For `Chabrier:H`, the code uses a per-component value of 30 kg/m$^3$ automatically (the H$_2$ critical density), so this global setting does not need to be changed for sub-Neptune models. For future volatiles (CO$_2$ ~470, NH$_3$ ~225, He ~70), this parameter serves as the fallback. |
| `condensed_rho_scale` | No | `50.0` | kg/m$^3$ | Sigmoid transition width. Smaller values produce a sharper cutoff; larger values a more gradual transition. The sigmoid goes from $\sigma = 0.02$ to $\sigma = 0.98$ over a density range of approximately $8 \times$ `condensed_rho_scale`. |

Both parameters must be positive. The validation rejects non-positive values at config load time.

#### Binodal (miscibility) suppression parameters (H2-containing mixtures only)

When `Chabrier:H` is mixed with silicate or water components, an additional binodal (miscibility) suppression determines whether H$_2$ participates in the harmonic mean at each $(P, T)$ point.
Two independent binodal models are evaluated automatically:

- **H$_2$-MgSiO$_3$**: [Rogers, Young & Schlichting (2025)](https://doi.org/10.1093/mnras/staf1940) analytic binodal fit.
- **H$_2$-H$_2$O**: [Gupta, Stixrude & Schlichting (2025)](https://doi.org/10.3847/2041-8213/adb631) Gibbs free energy model.

Above the binodal temperature, H$_2$ is miscible with its partner and participates fully in the density calculation.
Below the binodal, H$_2$ is suppressed via a smooth sigmoid transition.
See the [binodal physics documentation](../Explanations/binodal.md) for details.

| Field | Required | Default | Unit | Description |
|---|---|---|---|---|
| `binodal_T_scale` | No | `50.0` | K | Sigmoid width for binodal suppression. Controls how sharply H$_2$ is excluded below the miscibility boundary. With the default of 50 K, the sigmoid goes from $\sigma = 0.02$ to $\sigma = 0.98$ over a temperature range of approximately 400 K. Smaller values produce a sharper transition. Has no effect on layers without `Chabrier:H` components. |

!!! note "Per-component density suppression"
    When `Chabrier:H` is present in a mixture, the density-based sigmoid (controlled by `condensed_rho_min` and `condensed_rho_scale`) and the binodal sigmoid are combined multiplicatively: $\sigma_{\mathrm{total}} = \sigma_{\mathrm{density}} \times \sigma_{\mathrm{binodal}}$. H$_2$ uses a per-component `condensed_rho_min` of 30 kg/m$^3$ (near the H$_2$ critical density) and `condensed_rho_scale` of 10 kg/m$^3$ internally, regardless of the global config values. The global `condensed_rho_min` and `condensed_rho_scale` still apply to other volatile components (e.g., H$_2$O at 322 kg/m$^3$) in the same mixture.

#### Melting curve selection

The `rock_solidus` and `rock_liquidus` fields control which melting curves are used for phase routing (solid/mixed/liquid) in temperature-dependent EOS layers that use separate solid/liquid tables (`WolfBower2018`, `RTPress100TPa`, `PALEOS-2phase`). Unified PALEOS tables derive their phase boundary from the table itself and ignore these fields.

| Field | Required | Default | Description |
|---|---|---|---|
| `rock_solidus` | No | `"Monteux16-solidus"` | Solidus melting curve identifier. The `default.toml` sets Monteux16; if the key is absent, the code falls back to `"Stixrude14-solidus"` for backward compatibility with old TOML files. |
| `rock_liquidus` | No | `"Monteux16-liquidus-A-chondritic"` | Liquidus melting curve identifier. Same fallback logic: absent key defaults to `"Stixrude14-liquidus"`. |

**Available melting curves:**

| Identifier | Type | Source | Notes |
|---|---|---|---|
| `Monteux16-solidus` | Analytic | [Monteux et al. (2016)](https://doi.org/10.1016/j.epsl.2016.05.010) Eqs. 10/12 | Default. Piecewise Simon-Glatzel, valid to ~500 GPa. |
| `Stixrude14-solidus` | Analytic | [Stixrude (2014)](https://doi.org/10.1098/rsta.2013.0076) Eqs. 1.9+1.10 | Power law with cryoscopic depression ($x_0 = 0.79$). Valid at all $P$. |
| `Monteux16-liquidus-A-chondritic` | Analytic | [Monteux et al. (2016)](https://doi.org/10.1016/j.epsl.2016.05.010) Eqs. 11/13 | Default. A-chondritic composition. Valid to ~500 GPa. |
| `Monteux16-liquidus-F-peridotitic` | Analytic | [Monteux et al. (2016)](https://doi.org/10.1016/j.epsl.2016.05.010) Eqs. 11/13 | F-peridotitic composition. Valid to ~660 GPa. |
| `Stixrude14-liquidus` | Analytic | [Stixrude (2014)](https://doi.org/10.1098/rsta.2013.0076) Eq. 1.9 | Simon-like power law for pure MgSiO$_3$. Valid at all $P$. Liquidus always above solidus (constant ratio). |
| `PALEOS-liquidus` | Analytic | [Belonoshko et al. (2005)](https://doi.org/10.1103/PhysRevB.72.104107) / [Fei et al. (2021)](https://doi.org/10.1103/PhysRevLett.127.135701) | Piecewise Simon-Glatzel fit: Belonoshko+2005 ($P < 2.55$ GPa) / Fei+2021 ($P \geq 2.55$ GPa). This is the melting curve used internally by the PALEOS unified MgSiO$_3$ table. Use it with `PALEOS-2phase:MgSiO3` for consistency with the unified table's phase boundary. Valid at all $P$. |
| `Monteux600-solidus-tabulated` | Tabulated | Monteux et al. (2016) with fixed 600 K gap | Legacy. Returns NaN outside table range ($P > 999$ GPa). |
| `Monteux600-liquidus-tabulated` | Tabulated | Monteux et al. (2016) with fixed 600 K gap | Legacy. Returns NaN outside table range. |

The Monteux+2016 piecewise parameterization has a solidus/liquidus crossing above ~500 GPa (beyond the experimental calibration range). The Stixrude (2014) power laws avoid this issue entirely: the cryoscopic depression produces a constant solidus/liquidus ratio, guaranteeing liquidus > solidus at all pressures. Old TOML files without `rock_solidus`/`rock_liquidus` fields use the Stixrude14 defaults.

!!! warning "Mass limits for temperature-dependent EOS"
    **WolfBower2018:MgSiO3**: Tables cover pressures up to ~1 TPa. For planets above ~2 $M_\oplus$, deep-mantle pressures begin to exceed this limit. The Brent pressure solver with out-of-bounds clamping handles this gracefully up to $7\,M_\oplus$. Beyond $7\,M_\oplus$, clamped densities become unreliable and the code raises a `ValueError`. Use `RTPress100TPa:MgSiO3` or `PALEOS-2phase:MgSiO3` for higher-mass planets with a temperature-dependent mantle.

    **RTPress100TPa:MgSiO3**: The extended melt table covers pressures up to 100 TPa, enabling modeling of super-Earths up to ~$50\,M_\oplus$. The solid-phase table remains limited to 1 TPa (clamped at boundary), but at the high temperatures typical of massive planets the mantle is predominantly molten, so this limitation is less constraining.

    **PALEOS-2phase:MgSiO3**: Both solid and liquid tables extend to 100 TPa, limited to $\leq 50\,M_\oplus$. The tables have NaN gaps at certain (P, T) corners where the thermodynamic solver did not converge (53% fill for liquid, 75% for solid). Per-cell temperature clamping with nearest-neighbor fallback handles these gaps transparently. In adiabatic mode, the T(P) parameterization ensures the Brent pressure solver never queries outside the valid domain.

    **Unified PALEOS** (`PALEOS:iron`, `PALEOS:MgSiO3`, `PALEOS:H2O`): All extend to 100 TPa, limited to $\leq 50\,M_\oplus$. These are unified single-file tables that contain all stable phases for each material. The phase boundary is extracted from the table's phase column at load time. The same per-cell clamping and nearest-neighbor fallback apply. `PALEOS:iron` makes the iron core T-dependent for the first time, following its own adiabat.

For guidance on when to use tabulated vs. analytic EOS, see the [EOS physics explanation](../Explanations/eos_physics.md).

#### Configuration examples

**1. Full PALEOS adiabatic planet (recommended)**: T-dependent core + mantle with unified tables. This is the default configuration.
```toml
[AssumptionsAndInitialGuesses]
temperature_mode      = "adiabatic"
surface_temperature   = 3000
center_temperature    = 6000

[EOS]
core              = "PALEOS:iron"
mantle            = "PALEOS:MgSiO3"
ice_layer         = ""
mushy_zone_factor = 1.0   # 1.0 = sharp phase boundary from table
```
The iron core follows its own adiabat using $\nabla_{\mathrm{ad}}$ from the iron table. No external melting curves are needed.

**2. 3-layer water world with PALEOS** (T-dependent core + mantle + ice):
```toml
[AssumptionsAndInitialGuesses]
core_mass_fraction   = 0.25
mantle_mass_fraction = 0.50
temperature_mode     = "adiabatic"
surface_temperature  = 300       # Must be < 647 K for H2O ice layer
center_temperature   = 6000

[EOS]
core              = "PALEOS:iron"
mantle            = "PALEOS:MgSiO3"
ice_layer         = "PALEOS:H2O"
mushy_zone_factor = 1.0
```
Note: `surface_temperature` must be below 647 K (H$_2$O critical point) for three-layer models with H$_2$O ice. At higher temperatures, use multi-material mixing in the mantle instead (see example 3).

**3. Hydrated mantle** (iron core + mixed silicate-water mantle, phase-aware mixing):
```toml
[AssumptionsAndInitialGuesses]
temperature_mode      = "adiabatic"
surface_temperature   = 2000
center_temperature    = 6000

[EOS]
core                = "PALEOS:iron"
mantle              = "PALEOS:MgSiO3:0.85+PALEOS:H2O:0.15"
ice_layer           = ""
condensed_rho_min   = 322.0   # H2O critical density; adjust for other volatiles
condensed_rho_scale = 50.0
```
The mantle is 85% MgSiO$_3$ and 15% H$_2$O by mass. Each component's density and phase are evaluated independently from the PALEOS table. At depth (high $P$), both components are condensed and mixed via the standard harmonic mean. Near the surface (low $P$), H$_2$O transitions to vapor/supercritical gas and is smoothly suppressed by the sigmoid weighting, preventing unphysical radius inflation.

**4. Sub-Neptune with dissolved H$_2$** (iron core + mixed silicate-H$_2$ mantle, binodal suppression):
```toml
[InputParameter]
planet_mass               = 5

[AssumptionsAndInitialGuesses]
core_mass_fraction        = 0.25
temperature_mode          = "adiabatic"
surface_temperature       = 2000
center_temperature        = 10000

[EOS]
core                = "PALEOS:iron"
mantle              = "PALEOS:MgSiO3:0.97+Chabrier:H:0.03"
ice_layer           = ""
condensed_rho_min   = 322.0    # still needed for any H2O components in other layers
condensed_rho_scale = 50.0
binodal_T_scale     = 50.0     # H2-silicate miscibility sigmoid width (K)
```
The mantle is 97% MgSiO$_3$ and 3% H$_2$ by mass, representing a sub-Neptune with a miscible interior where H$_2$ is dissolved into the magma. The [Rogers+2025 binodal model](../Explanations/binodal.md) determines the miscibility boundary: above the binodal temperature (~4000 K at 1 GPa, decreasing to ~3000 K at 10 GPa), H$_2$ is miscible and contributes to the structural density. Below the binodal, H$_2$ is phase-separated and smoothly excluded. The per-component `condensed_rho_min` for H$_2$ (30 kg/m$^3$) is set automatically.

**5. Earth-like rocky planet** (Seager cold model, iron core + silicate mantle, 2-layer):
```toml
[EOS]
core      = "Seager2007:iron"
mantle    = "Seager2007:MgSiO3"
ice_layer = ""
```

**6. Hot rocky planet** (Wolf & Bower T-dependent mantle, 2-layer):
```toml
[EOS]
core      = "Seager2007:iron"
mantle    = "WolfBower2018:MgSiO3"
ice_layer = ""
```
Requires `temperature_mode`, `surface_temperature`, and `center_temperature` to be set in `[AssumptionsAndInitialGuesses]`.

**7. Water-rich planet** (3-layer Seager model, iron core + silicate mantle + water ice):
```toml
[AssumptionsAndInitialGuesses]
core_mass_fraction   = 0.25
mantle_mass_fraction = 0.50
# ...

[EOS]
core      = "Seager2007:iron"
mantle    = "Seager2007:MgSiO3"
ice_layer = "Seager2007:H2O"
```
Note: `mantle_mass_fraction` must be > 0 for a 3-layer model.

**8. Carbon planet / exotic compositions** (iron core + SiC or graphite mantle):
```toml
[EOS]
core      = "Analytic:iron"
mantle    = "Analytic:SiC"
ice_layer = ""
```

Other exotic mantles: `Analytic:graphite`, `Analytic:MgFeSiO3`.

**9. Mixed-EOS model** (tabulated core + analytic mantle):
```toml
[EOS]
core      = "Seager2007:iron"
mantle    = "Analytic:MgSiO3"
ice_layer = ""
```
Mixing tabulated and analytic EOS across layers is fully supported.

**10. Pure iron sphere** (single-composition edge case, core fills entire planet):
```toml
[AssumptionsAndInitialGuesses]
core_mass_fraction   = 1.0
mantle_mass_fraction = 0
# ...

[EOS]
core      = "Analytic:iron"
mantle    = "Analytic:iron"
ice_layer = ""
```
Both `core` and `mantle` must be set even for a single-composition model. The mantle EOS is still evaluated but occupies zero mass.

#### Validated parameter ranges

The solver has been systematically tested across 293 configurations spanning 9 test suites. The table below summarizes the validated ranges where convergence is reliable:

| Parameter | Validated range | Notes |
|---|---|---|
| Planet mass | 0.1 to 50 $M_\oplus$ | All standard compositions converge. EOS-specific limits apply (WolfBower2018 $\leq 7\,M_\oplus$, all others $\leq 50\,M_\oplus$). |
| Core mass fraction | 0.01 to 0.999 | Extreme values (pure iron, pure rock) converge at isothermal 300 K. |
| H$_2$O mixing fraction | 0 to 30% by mass | In mantle, using multi-material format. All fractions converge at 2000-3000 K surface temperature. Fractions above 30% produce a warning. |
| H$_2$ mixing fraction | 0 to 20% by mass | In mantle via `Chabrier:H`. Fractions above 20% produce a warning. Pure `Chabrier:H` mantle (without a condensed partner) is rejected. |
| Surface temperature | 300 to 5000 K | All compositions converge. 300 K uses isothermal mode; 1000-5000 K uses adiabatic mode. |
| Mushy zone factor | 0.7 to 1.0 | Per-material overrides validated independently. Values below 0.7 are rejected. |
| Three-layer ice fraction | any | **Only with $T_\mathrm{surf} < 647$ K** (H$_2$O critical point). See warning below. |

!!! note "Per-EOS table limits are the canonical reference"
    The mass limits in the **Available EOS options** table (`WolfBower2018:MgSiO3` $\leq 7\,M_\oplus$; all PALEOS, RTPress100TPa, and Chabrier:H entries up to $50\,M_\oplus$) are the canonical thermodynamic constraints, set by the pressure ceiling of each tabulated EOS. The validated-range row above ("Planet mass: 0.1 to 50 $M_\oplus$") is a downstream consequence of those table ceilings combined with solver convergence testing across 293 configurations. If a per-EOS limit and the validator-side limit ever appear to conflict (e.g., when a new EOS is added), the per-EOS table value takes precedence.

!!! warning "H$_2$O-dominated mantles"
    Mantles with more than 50% H$_2$O by mass and no silicate component are rejected in adiabatic and linear temperature modes. At these temperatures, H$_2$O is vapor at surface pressure and cannot support hydrostatic structure in the volume-additive mixing model. Options: add a silicate component, use a 3-layer model with a separate ice layer, or use isothermal mode with $T < 647$ K.

!!! warning "Pure Chabrier:H mantle"
    A mantle consisting entirely of `Chabrier:H` (100% H$_2$) is rejected. H$_2$ is a gas at surface pressure and cannot form a condensed layer on its own. Use H$_2$ as a mixing component in a silicate mantle, e.g., `mantle = "PALEOS:MgSiO3:0.97+Chabrier:H:0.03"`.

!!! danger "Three-layer models require cold surfaces"
    Three-layer models with an H$_2$O ice layer (`ice_layer = "PALEOS:H2O"`) are only valid when `surface_temperature` is below the H$_2$O critical temperature (647 K). At higher temperatures, H$_2$O is vapor or supercritical gas at the surface, not condensed ice. The solver will reject configurations with H$_2$O ice layers at $T_\mathrm{surf} \geq 647$ K.

    To model water-bearing planets at high surface temperatures (magma ocean stage), represent water as a **mixing component in the mantle** instead:
    ```toml
    mantle = "PALEOS:MgSiO3:0.85+PALEOS:H2O:0.15"
    ```
    The phase-aware suppression automatically handles the vapor-to-condensed transition with depth. This is the recommended approach for PROTEUS coupled simulations during the hot phase.

!!! note "Density-inverted architectures"
    Configurations where a lighter material sits below a denser one (e.g., `core = "PALEOS:MgSiO3"`, `mantle = "PALEOS:iron"`) are physically unstable and fail pressure convergence. The Brent solver cannot bracket 1 atm surface pressure for these configurations. Mass and density converge, but the surface pressure boundary condition cannot be satisfied.

#### Legacy configuration (backward compatibility)

The previous single-field `choice` syntax is still recognized and auto-expanded to the per-layer format:

| Legacy `choice` value | Expands to |
|---|---|
| `"Tabulated:iron/silicate"` | `core = "Seager2007:iron"`, `mantle = "Seager2007:MgSiO3"` |
| `"Tabulated:iron/Tdep_silicate"` | `core = "Seager2007:iron"`, `mantle = "WolfBower2018:MgSiO3"` |
| `"Tabulated:water"` | `core = "Seager2007:iron"`, `mantle = "Seager2007:MgSiO3"`, `ice_layer = "Seager2007:H2O"` |
| `"Analytic:Seager2007"` | Per-layer analytic EOS from `core_material`, `mantle_material`, `water_layer_material` fields |

Example of the old format (still works but deprecated):
```toml
[EOS]
choice = "Tabulated:iron/silicate"
```

Example of the old analytic format (still works but deprecated):
```toml
[EOS]
choice               = "Analytic:Seager2007"
core_material        = "iron"
mantle_material      = "MgSiO3"
water_layer_material = ""
```

New configurations should use the per-layer format. The legacy format may be removed in a future version.

---

### `Calculations`

Numerical grid settings.

| Parameter | Type | Unit | Description |
|---|---|---|---|
| `num_layers` | int | -- | Number of radial grid points. Higher values increase resolution and accuracy at the cost of computation time. Default: 150. |

---

### `IterativeProcess`

Controls the three nested iteration loops (outer mass convergence, inner density convergence, Brent pressure solver) and the ODE solver tolerances.

| Parameter | Type | Unit | Default | Description |
|---|---|---|---|---|
| `max_iterations_outer` | int | -- | 100 | Maximum iterations for the outer loop (total mass convergence). |
| `tolerance_outer` | float | -- | 3e-3 | Relative tolerance for outer loop convergence ($\lvert M_\mathrm{calc} - M_\mathrm{target} \rvert / M_\mathrm{target}$). |
| `max_iterations_inner` | int | -- | 100 | Maximum iterations for the inner loop (density profile convergence). |
| `tolerance_inner` | float | -- | 1e-4 | Relative tolerance for inner loop density convergence. |
| `relative_tolerance` | float | -- | 1e-5 | Relative tolerance for `solve_ivp` (ODE integrator). |
| `absolute_tolerance` | float | -- | 1e-6 | Absolute tolerance for `solve_ivp`. |
| `maximum_step` | float | m | 250000 | Maximum radial step size for `solve_ivp`. |
| `adaptive_radial_fraction` | float | -- | 0.98 | Fraction (0 to 1) of the radial domain where `solve_ivp` uses adaptive stepping before switching to fixed steps. Active for all T-dependent EOS. |
| `max_center_pressure_guess` | float | Pa | 10e12 | Upper bound on the Brent solver's pressure bracket ($P_{\mathrm{high}}$). This caps the *central* pressure (at the center of the iron core, which uses the Seager2007 EOS valid to $10^{16}$ Pa), not the mantle pressure directly. The default of 10 TPa is intentionally higher than the WolfBower2018 table ceiling (~1 TPa) because it limits the core center, not the mantle. However, higher central pressures produce higher mantle pressures at the CMB, so capping $P_c$ indirectly prevents deep-mantle pressures from exceeding the WB2018 table by too much. Only active when `WolfBower2018:MgSiO3` is used; not needed for pure-Seager runs. |

#### Guidance on reasonable parameter ranges

The default values work well for Earth-mass planets with the default EOS. For other configurations:

- **Super-Earths (1 to 10 $M_\oplus$):** The defaults generally suffice. For masses above ~5 $M_\oplus$, consider tightening `tolerance_outer` to 1e-4 and increasing `num_layers` to 200 to 300. The Brent pressure solver converges in 20 to 36 evaluations regardless of planet mass.
- **Sub-Earths (< 1 $M_\oplus$):** May converge faster. Defaults are conservative.
- **Temperature-dependent EOS (`WolfBower2018:MgSiO3`):** Limited to $\leq 7\,M_\oplus$. If convergence is slow, try reducing `maximum_step` (e.g., to 100000 m) and ensuring `adaptive_radial_fraction` is close to 1.0 (e.g., 0.98 to 0.99). The `max_center_pressure_guess` caps the Brent solver's upper bracket to prevent deep-mantle pressures from exceeding the WolfBower2018 table by too much. The default (10 TPa) is sufficient for planets up to 7 $M_\oplus$.
- **Extended T-dependent EOS (`RTPress100TPa:MgSiO3`):** Limited to $\leq 50\,M_\oplus$. The melt table extends to 100 TPa so `max_center_pressure_guess` capping is not applied. For very massive planets (>10 $M_\oplus$), consider increasing `num_layers` to 200 to 300 and tightening `tolerance_outer`.
- **Analytic EOS:** Convergence is typically fast and robust. Looser tolerances and fewer layers are usually sufficient for exploration.
- **3-layer models:** Adding an ice layer increases the number of density discontinuities. Consider increasing `num_layers` to 200+ for smooth profiles.

---

### `PressureAdjustment`

Controls the Brent root-finding solver that determines the central pressure.
The solver finds $P_c$ such that the surface pressure after ODE integration matches the target.
See the [process flow documentation](../Explanations/process_flow.md#pressure-solver-brents-method) for details on the algorithm.

| Parameter | Type | Unit | Default | Description |
|---|---|---|---|---|
| `target_surface_pressure` | float | Pa | 101325 | Target pressure at the planetary surface. Default is 1 atm. |
| `pressure_tolerance` | float | Pa | 1e9 | Convergence criterion: the solver iterates until $\lvert P_\mathrm{surface} - P_\mathrm{target} \rvert$ falls below this value. |
| `max_iterations_pressure` | int | -- | 200 | Maximum number of function evaluations for the Brent solver. Typical convergence requires 20 to 36 evaluations. |

---

### `Output`

Controls what the model writes after a run.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `data_enabled` | bool | true | Write the computed radial profiles (radius, density, gravity, pressure, temperature, enclosed mass) to a text file in `output/`. |
| `plots_enabled` | bool | false | Generate profile plots after the run. |

Convergence diagnostics are controlled by the Python logging level. Set `logging.getLogger('zalmoxis').setLevel(logging.DEBUG)` to see per-iteration convergence info and write iteration pressure/density profiles.
