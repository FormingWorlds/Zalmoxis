# Equations of State

This page documents the physics of each EOS family available in Zalmoxis.
For a summary of EOS identifiers and their validity ranges, see the [model overview](model.md#valid-per-layer-eos-identifiers).
For configuration syntax and examples, see the [configuration guide](../How-to/configuration.md#eos).

---

## PALEOS (Iron, MgSiO3, H2O)

The PALEOS tables (`PALEOS:iron`, `PALEOS:MgSiO3`, `PALEOS:H2O`) from Zenodo record 19000316 are the standard EOS in Zalmoxis.
Each material is contained in a single file with all thermodynamically stable phases: iron has 5 phases (alpha-bcc, delta-bcc, gamma-fcc, epsilon-hcp, liquid), MgSiO$_3$ has 6 phases (3 pyroxene polymorphs, bridgmanite, postperovskite, liquid), and H$_2$O has 7 EOS (ice Ih through X, liquid, vapor, superionic).

The tables provide 10 columns in SI units: $P$, $T$, $\rho$, $u$, $s$, $c_p$, $c_v$, $\alpha$, $\nabla_{\mathrm{ad}}$, and the stable phase identifier at each grid point.
The grid is log-uniform with 150 points per decade in both $P$ (1 bar to 100 TPa) and $T$ (300 to 100,000 K for iron and MgSiO$_3$; 100 to 100,000 K for H$_2$O).

**Single-table architecture.** The density at any $(P, T)$ is obtained by a single interpolation on the table, which already provides the stable-phase value.
No external solidus/liquidus curves or separate solid/liquid files are needed.
This eliminates the phase-routing complexity of earlier EOS families.

**Phase boundary extraction.** At load time, the code extracts the liquidus boundary from the phase column: for each pressure row, the lowest temperature where the phase is `liquid` defines the liquidus.
This extracted boundary serves as the basis for an optional mushy zone controlled by the `mushy_zone_factor` parameter:

- $f = 1.0$ (default): no mushy zone; the density comes directly from the table (sharp phase transition).
- $f < 1.0$: a synthetic solidus is defined at $T_{\mathrm{sol}} = f \times T_{\mathrm{liq}}$. Between $T_{\mathrm{sol}}$ and $T_{\mathrm{liq}}$, density is volume-averaged between solid-side and liquid-side table values.

**Adiabatic gradient.** All three tables include the dimensionless adiabatic gradient $\nabla_{\mathrm{ad}} = (d \ln T / d \ln P)_S$.
In adiabatic mode, the temperature profile is computed by integrating $dT/dP = \nabla_{\mathrm{ad}} \cdot T / P$ from the surface inward.
Each material follows its own adiabat: with `PALEOS:iron`, the iron core is temperature-dependent and follows its own adiabat using $\nabla_{\mathrm{ad}}$ from the iron table, instead of being isothermal at the CMB temperature.

**Grid coverage.** Some cells at corners of the $P$--$T$ domain are missing where the thermodynamic solver did not converge.
Per-cell temperature clamping restricts queries to the valid data range at each pressure, and a nearest-neighbor fallback handles remaining NaN results near the ragged domain boundary.
In adiabatic mode, the temperature is parameterized as $T(P)$ rather than $T(r)$, ensuring that the Brent pressure solver always evaluates thermodynamically consistent $(P, T)$ pairs within the valid table domain.

**Mass limit.** All three tables extend to 100 TPa, supporting planets up to ~50 $M_\oplus$.

---

### PALEOS-2phase (MgSiO3 only)

The `PALEOS-2phase:MgSiO3` EOS is an earlier variant of the PALEOS database (Zenodo record 18924171) that provides MgSiO$_3$ as separate solid and liquid table files.
The table format and grid structure are identical to the unified tables.

Unlike the unified PALEOS tables, `PALEOS-2phase` requires external melting curves for phase routing.
Configurable solidus and liquidus curves (see [Melting curve selection](../How-to/configuration.md#melting-curve-selection)) determine the phase at each $(P, T)$:

- Below the solidus: $\nabla_{\mathrm{ad}}$ from the solid table.
- Above the liquidus: $\nabla_{\mathrm{ad}}$ from the liquid table.
- In the mushy zone: melt-fraction-weighted average $(1 - \phi) \nabla_{\mathrm{ad,solid}} + \phi \, \nabla_{\mathrm{ad,liquid}}$.

The default analytic Monteux+2016 curves are defined for all pressures, eliminating the NaN-at-boundary issue of the legacy tabulated curves.

**Grid coverage.** The PALEOS-2phase tables have missing cells at domain corners (solid table: 75% filled, liquid table: 53% filled), handled by the same per-cell clamping and nearest-neighbor fallback as the unified tables.

**Mass limit.** Both solid and liquid tables extend to 100 TPa, supporting planets up to ~50 $M_\oplus$.

**When to use.** The unified `PALEOS:MgSiO3` is preferred for new work.
Use `PALEOS-2phase:MgSiO3` only if you need explicit control over the solidus/liquidus curves independently of the table's internal phase boundary.

---

## Legacy EOS families

The following EOS families predate the PALEOS tables. They remain fully supported for backward compatibility and for specific use cases (cold 300 K models, comparison with published mass-radius relations, exotic compositions).

### Seager et al. (2007) Tabulated EOS

The tabulated EOS from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) provides $\rho(P)$ at 300 K for iron (Fe $\epsilon$-phase), MgSiO$_3$ perovskite, and water ice.
The tables are constructed by merging two regimes:

- **Low pressure** ($P \lesssim 200$ GPa): Vinet EOS for iron, fourth-order Birch--Murnaghan EOS for MgSiO$_3$ and water ice, fitted to experimental data and DFT calculations.
- **High pressure** ($P \gtrsim 10^4$ GPa): Thomas--Fermi--Dirac (TFD) theory, which becomes exact in the limit of fully ionized, degenerate electron gas.

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
- **Mixed / mush** ($T_{\mathrm{sol}} < T < T_{\mathrm{liq}}$): density from volume-additive interpolation between solid and liquid densities:

$$
\rho_{\mathrm{mixed}} = \left[ (1 - f_{\mathrm{melt}}) \, \rho_{\mathrm{solid}}^{-1} + f_{\mathrm{melt}} \, \rho_{\mathrm{liquid}}^{-1} \right]^{-1}
$$

**Mass limit.** Tables cover pressures up to ~1 TPa.
Out-of-bounds pressures are clamped to the table edge, which is acceptable up to $7\,M_\oplus$.
Beyond $7\,M_\oplus$, the code raises a `ValueError`.
Use PALEOS or RTPress100TPa for higher-mass planets.

### RTPress100TPa Extended Melt EOS

The `RTPress100TPa:MgSiO3` EOS extends the WolfBower2018 melt phase coverage from 1 TPa to 100 TPa ($P$: $10^3$--$10^{14}$ Pa, $T$: 400--50000 K), enabling temperature-dependent modeling up to ~50 $M_\oplus$.

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
Accuracy is 2--12% across all planetary pressures.
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
