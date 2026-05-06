# Equations of State

This page documents the physics of each EOS family available in Zalmoxis.
For a summary of EOS identifiers and their validity ranges, see the [model overview](model.md#valid-per-layer-eos-identifiers).
For configuration syntax and examples, see the [configuration guide](../How-to/configuration.md#eos).

---

## PALEOS (Iron, MgSiO3, H2O)

The PALEOS tables (`PALEOS:iron`, `PALEOS:MgSiO3`, `PALEOS:H2O`) are the standard EOS in Zalmoxis. PALEOS is described in [Attia et al. (2026)](https://ui.adsabs.harvard.edu/abs/2026arXiv260503741A/abstract) and developed at [github.com/maraattia/PALEOS](https://github.com/maraattia/PALEOS); the tables used by Zalmoxis are distributed via [Zenodo record 19000316](https://zenodo.org/records/19000316).
Each material is contained in a single file with all thermodynamically stable phases: iron has 5 phases (alpha-bcc, delta-bcc, gamma-fcc, epsilon-hcp, liquid), MgSiO$_3$ has 6 phases (3 pyroxene polymorphs, bridgmanite, postperovskite, liquid), and H$_2$O has 7 EOS (ice Ih through X, liquid, vapor, superionic).

The tables provide 10 columns in SI units: $P$, $T$, $\rho$, $u$, $s$, $c_p$, $c_v$, $\alpha$, $\nabla_{\mathrm{ad}}$, and the stable phase identifier at each grid point.
The grid is log-uniform with 150 points per decade in both $P$ (1 bar to 100 TPa) and $T$ (300 to 100,000 K for iron and MgSiO$_3$; 100 to 100,000 K for H$_2$O).

**Single-table architecture.** The density at any $(P, T)$ is obtained by a single interpolation on the table, which already provides the stable-phase value.
No external solidus/liquidus curves or separate solid/liquid files are needed.
This eliminates the phase-routing complexity required by the two-phase and WolfBower EOS families.

**Phase boundary.** The PALEOS MgSiO$_3$ liquidus has a well-defined analytic functional form: a piecewise Simon-Glatzel fit using [Belonoshko et al. (2005)](https://doi.org/10.1103/PhysRevB.72.104107) below 2.55 GPa and [Fei et al. (2021)](https://doi.org/10.1103/PhysRevLett.127.135701) above, with the crossover pressure chosen for continuity.
This analytic liquidus is implemented as `paleos_liquidus(P)` in `melting_curves.py` and is used in the vectorized density and $\nabla_{\mathrm{ad}}$ code paths.
At load time, the code also extracts a discrete liquidus boundary from the phase column (for each pressure row, the lowest temperature where the phase is `liquid`); this serves as a consistency reference and is used in the scalar code path.

The liquidus serves as the basis for an optional mushy zone controlled by the `mushy_zone_factor` parameter (global default) or per-material overrides (`mushy_zone_factor_iron`, `mushy_zone_factor_MgSiO3`, `mushy_zone_factor_H2O`):

- $f = 1.0$ (default): no mushy zone; the density and $\nabla_{\mathrm{ad}}$ come directly from the table (sharp phase transition).
- $f < 1.0$: a synthetic solidus is defined at $T_{\mathrm{sol}} = f \times T_{\mathrm{liq}}$. Between $T_{\mathrm{sol}}$ and $T_{\mathrm{liq}}$, density is computed from volume-additive mixing of specific volumes (density inverses) between the solid-side value at $T_{\mathrm{sol}}$ and the liquid-side value at $T_{\mathrm{liq}}$, weighted by the melt fraction $\phi = (T - T_{\mathrm{sol}}) / (T_{\mathrm{liq}} - T_{\mathrm{sol}})$. The adiabatic gradient $\nabla_{\mathrm{ad}}$ is linearly interpolated between the solid and liquid values with the same melt fraction.

Each PALEOS material can have its own mushy zone width. Per-material overrides take precedence over the global default. This is useful when mixing materials with different phase transition characteristics (e.g., a wider mushy zone for MgSiO$_3$ with a sharp boundary for iron).

**Adiabatic gradient.** All three tables include the dimensionless adiabatic gradient $\nabla_{\mathrm{ad}} = (d \ln T / d \ln P)_S$.
In adiabatic mode, the temperature profile is computed by integrating $dT/dP = \nabla_{\mathrm{ad}} \cdot T / P$ from the surface inward.
Each material follows its own adiabat: with `PALEOS:iron`, the iron core is temperature-dependent and follows its own adiabat using $\nabla_{\mathrm{ad}}$ from the iron table, instead of being isothermal at the CMB temperature.

**Grid coverage.** Some cells at corners of the $P$-$T$ domain are missing where the thermodynamic solver did not converge.
Per-cell temperature clamping restricts queries to the valid data range at each pressure, and a nearest-neighbor fallback handles remaining NaN results near the ragged domain boundary.
In adiabatic mode, the temperature is parameterized as $T(P)$ rather than $T(r)$, ensuring that the Brent pressure solver always evaluates thermodynamically consistent $(P, T)$ pairs within the valid table domain.

**Mass limit.** All three tables extend to 100 TPa, supporting planets up to ~50 $M_\oplus$.

---

## Chabrier H/He (Pure H$_2$)

The `Chabrier:H` EOS provides temperature- and pressure-dependent properties for pure molecular hydrogen from the DirEOS2021 tables of [Chabrier et al. (2019, ApJ 872, 51)](https://doi.org/10.3847/1538-4357/aaf99f) and [Chabrier & Debras (2021, ApJ 917, 4)](https://doi.org/10.3847/1538-4357/abfc48).
The table covers all H$_2$ regimes: molecular, dissociated atomic, and ionized.

**Grid.** 121 $\times$ 441 points ($\log T$, $\log P$): $T = 100$ to $10^8$ K, $P = 1$ Pa to $10^{22}$ Pa (53,361 grid points).

**Format.** Converted to the same 10-column PALEOS-compatible format (P, T, $\rho$, $u$, $s$, $c_p$, $c_v$, $\alpha$, $\nabla_{\mathrm{ad}}$, phase_id) and loaded through the same `paleos_unified` reader.
The derivations are:

- $\alpha$ (thermal expansivity): exact from the table.
- $c_p$: derived as $P \alpha / (\rho \, \nabla_{\mathrm{ad}})$.
- $c_v$: from the Mayer relation.

**$\nabla_{\mathrm{ad}}$ clamping.** In the H$_2$ dissociation zone ($T \sim 3000$ to $30{,}000$ K, $P \sim 0.1$ to $1000$ GPa), the source tables clamp $\nabla_{\mathrm{ad}}$ at a floor of 0.100.
This affects 17.6% of H table grid points.
The clamping makes $c_p$ unreliable in this region (it is derived from $\nabla_{\mathrm{ad}}$), but Zalmoxis uses $\nabla_{\mathrm{ad}}$ directly for adiabat integration, so this limitation does not affect the structure solver.

**Negative $\alpha$ and $c_p$.** Approximately 30% of the physical domain has negative thermal expansivity ($\alpha < 0$) and consequently negative $c_p$.
These are physical: H$_2$ contracts upon heating in regions where dissociation or ionization absorb enthalpy faster than thermal expansion adds volume.
Zalmoxis uses $\nabla_{\mathrm{ad}}$ for adiabat integration and $\rho(P, T)$ for structure, so negative $c_p$ does not affect the solver.

**Intended use.** `Chabrier:H` is designed as a mixing component in sub-Neptune mantle layers (e.g., `"PALEOS:MgSiO3:0.97+Chabrier:H:0.03"`), not as a standalone layer EOS.
When mixed with silicate or water, [binodal (miscibility) suppression](binodal.md) determines whether the H$_2$ component participates in the structural density at each $(P, T)$ point.

**Additional tables.** The data directory also contains tables for pure He and three H/He mixtures at different helium mass fractions ($Y = 0.275, 0.292, 0.297$).
These are not currently registered in the EOS registry but are available for future use.

---

### PALEOS-2phase (MgSiO3 only)

The `PALEOS-2phase:MgSiO3` EOS provides MgSiO$_3$ as separate solid and liquid table files generated by PALEOS ([Attia et al. (2026)](https://ui.adsabs.harvard.edu/abs/2026arXiv260503741A/abstract); [Zenodo record 19680050](https://zenodo.org/records/19680050), v1.1.0).
The table format and grid structure are identical to the main PALEOS tables.

Unlike the main PALEOS tables, `PALEOS-2phase` requires external melting curves for phase routing.
Configurable solidus and liquidus curves (see [Melting curve selection](../How-to/configuration.md#melting-curve-selection)) determine the phase at each $(P, T)$.

!!! info "Which EOS families need an external melting curve"
    The unified PALEOS tables (`PALEOS:iron`, `PALEOS:MgSiO3`, `PALEOS:H2O`) and `Chabrier:H` carry their own phase information in the table's stable-phase column and do **not** require an external solidus or liquidus file. They derive their phase boundary from the table at load time.

    The two-phase EOS families (`PALEOS-2phase:MgSiO3`, `WolfBower2018:MgSiO3`, `RTPress100TPa:MgSiO3`) ship as separate solid and liquid tables and **do** require external melting curves loaded via the configurable `rock_solidus` and `rock_liquidus` fields (or via `melting_dir` in the PROTEUS coupling layer). Phase routing uses these curves to compute melt fraction in the mushy zone.

    The Seager (2007) and Analytic families are evaluated at fixed 300 K and have no phase boundary at all.
Both density and $\nabla_{\mathrm{ad}}$ follow the same three-regime structure:

- **Below the solidus** ($T \le T_{\mathrm{sol}}$): density and $\nabla_{\mathrm{ad}}$ from the solid table.
- **Above the liquidus** ($T \ge T_{\mathrm{liq}}$): density and $\nabla_{\mathrm{ad}}$ from the liquid table.
- **Mushy zone** ($T_{\mathrm{sol}} < T < T_{\mathrm{liq}}$): density from volume-additive mixing of specific volumes (density inverses) evaluated at the solidus and liquidus temperatures respectively, weighted by $\phi = (T - T_{\mathrm{sol}}) / (T_{\mathrm{liq}} - T_{\mathrm{sol}})$. The adiabatic gradient is a melt-fraction-weighted average: $(1 - \phi) \nabla_{\mathrm{ad,solid}} + \phi \, \nabla_{\mathrm{ad,liquid}}$.

The default analytic Monteux+2016 curves are defined for all pressures, eliminating the NaN-at-boundary issue of the legacy tabulated curves.

**Grid coverage.** The PALEOS-2phase tables have missing cells at domain corners (solid table: 75% filled, liquid table: 53% filled), handled by the same per-cell clamping and nearest-neighbor fallback as the unified tables.

**Mass limit.** Both solid and liquid tables extend to 100 TPa, supporting planets up to ~50 $M_\oplus$.

**When to use.** `PALEOS:MgSiO3` is the default and recommended choice.
Use `PALEOS-2phase:MgSiO3` when you need explicit control over the solidus/liquidus curves independently of the table's internal phase boundary.

---

## Legacy EOS families

The following EOS families predate the PALEOS tables. They remain fully supported for backward compatibility and for specific use cases (cold 300 K models, comparison with published mass-radius relations, exotic compositions).

### Seager et al. (2007) Tabulated EOS

The tabulated EOS from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) provides $\rho(P)$ at 300 K for iron (Fe $\epsilon$-phase), MgSiO$_3$ perovskite, and water ice.
The tables are constructed by merging two regimes:

- **Low pressure** ($P \lesssim 200$ GPa): Vinet EOS for iron, fourth-order Birch-Murnaghan EOS for MgSiO$_3$ and water ice, fitted to experimental data and DFT calculations.
- **High pressure** ($P \gtrsim 10^4$ GPa): Thomas-Fermi-Dirac (TFD) theory, which becomes exact in the limit of fully ionized, degenerate electron gas.

The two regimes are smoothly joined in the intermediate range.

**Limitations.** The tabulated EOS is evaluated at a fixed temperature of 300 K and does not account for thermal pressure, phase transitions (e.g., post-perovskite), compositional gradients, or partial melting.
It is appropriate for cold, fully solidified structure models.

### Wolf & Bower (2018) Temperature-Dependent EOS

The `WolfBower2018:MgSiO3` EOS implements the RTpress (Reciprocal Transform pressure) framework from [Wolf & Bower (2018)](https://www.sciencedirect.com/science/article/pii/S0031920117301449), providing $\rho(P, T)$ for MgSiO$_3$ in both the solid and liquid phases.
The solid-phase EOS is derived from [Mosenfelder et al. (2009)](https://doi.org/10.1029/2008JB005900).

Phase boundaries are defined using solidus and liquidus melting curves derived from [Monteux et al. (2016)](https://www.sciencedirect.com/science/article/pii/S0012821X16302199).
At each radial shell, the code evaluates the local melt fraction:

$$
f_{\mathrm{melt}} = \frac{T - T_{\mathrm{sol}}}{T_{\mathrm{liq}} - T_{\mathrm{sol}}}
$$

Three phase regimes are distinguished:

- **Solid** ($T \le T_{\mathrm{sol}}$, $f_{\mathrm{melt}} \le 0$): density from the solid-state EOS table.
- **Melt** ($T \ge T_{\mathrm{liq}}$, $f_{\mathrm{melt}} \ge 1$): density from the liquid-state EOS table.
- **Mixed / mush** ($T_{\mathrm{sol}} < T < T_{\mathrm{liq}}$): the solid-side density is evaluated at $(P, T_{\mathrm{sol}})$ and the liquid-side density at $(P, T_{\mathrm{liq}})$. The mixed density is computed from volume-additive mixing of their specific volumes (density inverses), weighted by melt fraction:

$$
\rho_{\mathrm{mixed}} = \left[ (1 - f_{\mathrm{melt}}) \, \rho_{\mathrm{solid}}^{-1} + f_{\mathrm{melt}} \, \rho_{\mathrm{liquid}}^{-1} \right]^{-1}
$$

**Mass limit.** Tables cover pressures up to ~1 TPa.
Out-of-bounds pressures are clamped to the table edge, which is acceptable up to $7\,M_\oplus$.
Beyond $7\,M_\oplus$, the code raises a `ValueError`.
Use PALEOS or RTPress100TPa for higher-mass planets.

### RTPress100TPa Extended Melt EOS

The `RTPress100TPa:MgSiO3` EOS extends the WolfBower2018 melt phase coverage from 1 TPa to 100 TPa ($P$: $10^3$ to $10^{14}$ Pa, $T$: 400 to 50,000 K), enabling temperature-dependent modeling up to ~50 $M_\oplus$.

The solid-phase EOS remains the [Wolf & Bower (2018)](https://www.sciencedirect.com/science/article/pii/S0031920117301449) table (valid to 1 TPa, clamped at boundary).
At the high internal temperatures typical of massive rocky planets, the mantle is predominantly molten, so the solid table limitation is less constraining.
The same [Monteux et al. (2016)](https://www.sciencedirect.com/science/article/pii/S0012821X16302199) solidus/liquidus melting curves are used for phase determination.

**Mass limit.** 50 $M_\oplus$. Unlike WolfBower2018, the central pressure is not capped by `max_center_pressure_guess`.

### Analytic Modified Polytrope (Seager et al. 2007)

The analytic EOS implements the modified polytropic fit from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346), Table 3, Eq. 11:

$$
\rho(P) = \rho_0 + c \cdot P^n
$$

where $\rho$ is density in kg/m$^3$, $P$ is pressure in Pa, and $\rho_0$, $c$, $n$ are material-specific parameters.
This closed-form expression approximates the full merged Vinet/BME + TFD EOS without requiring tabulated data files.
Accuracy is 2 to 12% across all planetary pressures.
The fit is valid for $P < 10^{16}$ Pa.

Six materials are available:

| Material key | Compound | $\rho_0$ (kg/m$^3$) | $c$ (kg m$^{-3}$ Pa$^{-n}$) | $n$ |
|---|---|---|---|---|
| `iron` | Fe ($\epsilon$) | 8300 | 0.00349 | 0.528 |
| `MgSiO3` | MgSiO$_3$ perovskite | 4100 | 0.00161 | 0.541 |
| `MgFeSiO3` | (Mg,Fe)SiO$_3$ | 4260 | 0.00127 | 0.549 |
| `H2O` | Water ice (VII/VIII/X) | 1460 | 0.00311 | 0.513 |
| `graphite` | C (graphite) | 2250 | 0.00350 | 0.514 |
| `SiC` | Silicon carbide | 3220 | 0.00172 | 0.537 |

The analytic EOS assumes 300 K and no phase transitions.
It is useful for quick exploration, testing, and exotic compositions (graphite, SiC, MgFeSiO$_3$) that have no tabulated counterpart.
