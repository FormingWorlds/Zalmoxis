# Equations of State

This page documents the physics of each EOS family available in Zalmoxis.
For a summary of EOS identifiers and their validity ranges, see the [model overview](model.md#valid-per-layer-eos-identifiers).
For configuration syntax and examples, see the [configuration guide](../How-to/configuration.md#eos).

---

## Seager et al. (2007) Tabulated EOS

The tabulated EOS from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) provides $\rho(P)$ at 300 K for iron (Fe $\epsilon$-phase), MgSiO$_3$ perovskite, and water ice.
The tables are constructed by merging two regimes:

- **Low pressure** ($P \lesssim 200$ GPa): Vinet EOS for iron, fourth-order Birch--Murnaghan EOS for MgSiO$_3$ and water ice, fitted to experimental data and DFT calculations.
- **High pressure** ($P \gtrsim 10^4$ GPa): Thomas--Fermi--Dirac (TFD) theory, which becomes exact in the limit of fully ionized, degenerate electron gas.

The two regimes are smoothly joined in the intermediate range.

**Limitations.** The tabulated EOS is evaluated at a fixed temperature of 300 K and does not account for thermal pressure, phase transitions (e.g., post-perovskite), compositional gradients, or partial melting.
It is appropriate for cold, fully solidified structure models.

---

## Wolf & Bower (2018) Temperature-Dependent EOS

The `WolfBower2018:MgSiO3` EOS implements the RTpress (Reciprocal Transform pressure) framework from [Wolf & Bower (2018)](https://www.sciencedirect.com/science/article/pii/S0031920117301449), providing $\rho(P, T)$ for MgSiO$_3$ in both the solid and liquid phases.
The solid-phase EOS is derived from [Mosenfelder et al. (2009)](https://doi.org/10.1029/2008JB005900).
This EOS captures the thermal expansivity and compressibility of silicate mantle material over a wide $P$--$T$ range.

Phase boundaries are defined using solidus and liquidus melting curves derived from [Monteux et al. (2016)](https://www.sciencedirect.com/science/article/pii/S0012821X16302199), with the solidus offset by $-600$ K from the liquidus.
At each radial shell, the code evaluates the local melt fraction:

$$
f_{\mathrm{melt}} = \frac{T - T_{\mathrm{sol}}}{T_{\mathrm{liq}} - T_{\mathrm{sol}}}
$$

where $T$ is the local temperature, $T_{\mathrm{sol}}$ is the solidus temperature, and $T_{\mathrm{liq}}$ is the liquidus temperature at the local pressure.
Three phase regimes are distinguished:

- **Solid** ($T \le T_{\mathrm{sol}}$, $f_{\mathrm{melt}} \le 0$): density from the solid-state EOS table.
- **Melt** ($T \ge T_{\mathrm{liq}}$, $f_{\mathrm{melt}} \ge 1$): density from the liquid-state EOS table.
- **Mixed / mush** ($T_{\mathrm{sol}} < T < T_{\mathrm{liq}}$): density from volume-additive interpolation between solid and liquid densities:

$$
\rho_{\mathrm{mixed}} = \left[ (1 - f_{\mathrm{melt}}) \, \rho_{\mathrm{solid}}^{-1} + f_{\mathrm{melt}} \, \rho_{\mathrm{liquid}}^{-1} \right]^{-1}
$$

This EOS is appropriate when modeling hot rocky planets whose mantles may be partially or fully molten, which is critical for accurately coupling to thermal evolution codes.
A temperature profile (isothermal, linear, prescribed from file, or adiabatic) must be supplied.
When this EOS is selected for any layer, the radial integration is split into two segments (controlled by `adaptive_radial_fraction`) to handle the steep density gradients near the surface.

**Mass limit.** The WolfBower2018 tables cover pressures up to ~1 TPa.
For planets above ~2 $M_\oplus$, deep-mantle pressures near the core-mantle boundary begin to exceed this table ceiling.
The Brent pressure solver (see [Pressure Solver](process_flow.md#pressure-solver-brents-method)) with out-of-bounds pressure clamping handles this gracefully up to $7\,M_\oplus$: pressures beyond the table boundary are clamped to the table edge, returning the boundary density.
This approximation is acceptable for planets up to ~7 $M_\oplus$, where the clamped region is a small fraction of the mantle.
Beyond $7\,M_\oplus$, the clamped densities diverge too far from reality and the code raises a `ValueError`.
For higher-mass planets, use `RTPress100TPa:MgSiO3` or `PALEOS-2phase:MgSiO3` (T-dependent, up to ~50 $M_\oplus$) or `Seager2007:MgSiO3` / `Analytic:MgSiO3` (300 K, up to ~50 $M_\oplus$).

---

## RTPress100TPa Extended Melt EOS

The `RTPress100TPa:MgSiO3` EOS extends the melt phase coverage from the WolfBower2018 1 TPa ceiling to 100 TPa ($P$: $10^3$--$10^{14}$ Pa, $T$: 400--50000 K).
This enables temperature-dependent modeling of much more massive rocky planets (up to ~50 $M_\oplus$).

The solid-phase EOS remains the [Wolf & Bower (2018)](https://www.sciencedirect.com/science/article/pii/S0031920117301449) / [Mosenfelder et al. (2009)](https://doi.org/10.1029/2008JB005900) table (valid to 1 TPa, clamped at boundary).
At the high internal temperatures typical of massive rocky planets, the mantle is predominantly molten, so the solid table limitation is less constraining than it would be for cooler planets.
The same [Monteux et al. (2016)](https://www.sciencedirect.com/science/article/pii/S0012821X16302199) solidus/liquidus melting curves are used for phase determination.

**Mass limit.** The melt table extends to 100 TPa, matching the Seager2007 iron EOS range.
The solid table is clamped at 1 TPa, but this primarily affects cold planets where a significant fraction of the mantle is solid.
The code allows planets up to 50 $M_\oplus$ with this EOS.
Unlike `WolfBower2018:MgSiO3`, the central pressure is not capped by `max_center_pressure_guess` since the melt table covers the full range.

---

## PALEOS MgSiO3 EOS

The `PALEOS-2phase:MgSiO3` EOS provides separate solid and liquid MgSiO$_3$ tables from the PALEOS thermodynamic database (Zenodo record 18924171).
Each table contains 10 columns in SI units: $P$, $T$, $\rho$, $u$, $s$, $c_p$, $c_v$, $\alpha$, $\nabla_{\mathrm{ad}}$, and phase identifier.
The grid is log-uniform in both $P$ (1 bar to 100 TPa) and $T$ (300 K to 100,000 K) with 150 points per decade.

Unlike WolfBower2018 and RTPress100TPa, the PALEOS tables include the dimensionless adiabatic gradient $\nabla_{\mathrm{ad}} = (d \ln T / d \ln P)_S$ for both solid and liquid phases.
This enables a phase-aware adiabatic temperature profile:

- Below the solidus: $\nabla_{\mathrm{ad}}$ from the solid table.
- Above the liquidus: $\nabla_{\mathrm{ad}}$ from the liquid table.
- In the mushy zone: melt-fraction-weighted average $(1 - \phi) \nabla_{\mathrm{ad,solid}} + \phi \, \nabla_{\mathrm{ad,liquid}}$.

The adiabatic gradient is converted to $dT/dP = \nabla_{\mathrm{ad}} \cdot T / P$ for integration from the surface inward.

**Grid coverage.** The PALEOS tables have missing cells at corners of the $P$--$T$ domain where the thermodynamic solver did not converge (solid table: 75% filled, liquid table: 53% filled).
Per-cell temperature clamping restricts queries to the valid data range at each pressure, and a nearest-neighbor fallback handles remaining NaN results near the ragged domain boundary.
In adiabatic mode, the temperature is parameterized as $T(P)$ rather than $T(r)$, ensuring that the Brent pressure solver always evaluates thermodynamically consistent $(P, T)$ pairs within the valid table domain.

**Melting curves.** Phase routing uses configurable solidus and liquidus curves (see [Melting curve selection](../How-to/configuration.md#melting-curve-selection)).
The default analytic Monteux+2016 curves are defined for all pressures, eliminating the NaN-at-boundary issue of the legacy tabulated curves.

**Mass limit.** Both solid and liquid tables extend to 100 TPa, supporting planets up to ~50 $M_\oplus$.

---

## Unified PALEOS Tables (Iron, MgSiO3, H2O)

The unified PALEOS tables (`PALEOS:iron`, `PALEOS:MgSiO3`, `PALEOS:H2O`) from Zenodo record 19000316 represent the next generation of the PALEOS EOS.
Each material is contained in a single file with all stable phases: iron has 5 phases (alpha-bcc, delta-bcc, gamma-fcc, epsilon-hcp, liquid), MgSiO$_3$ has 6 phases (3 pyroxene polymorphs, bridgmanite, postperovskite, liquid), and H$_2$O has 7 EOS (ice Ih through X, liquid, vapor, superionic).

The format is the same 10-column layout as PALEOS-2phase ($P$, $T$, $\rho$, $u$, $s$, $c_p$, $c_v$, $\alpha$, $\nabla_{\mathrm{ad}}$, phase), but the `phase` column now encodes the thermodynamically stable phase at each $(P, T)$ grid point.
The grid is log-uniform with 150 points per decade in both $P$ (1 bar to 100 TPa) and $T$ (300 to 100,000 K for iron and MgSiO$_3$; 100 to 100,000 K for H$_2$O).

**Key architectural difference from PALEOS-2phase.** Instead of splitting the EOS into separate solid and liquid files and routing through an external melting curve, the unified table contains all phases in one file.
The density at any $(P, T)$ is obtained by a single interpolation on the table, which already provides the stable-phase value.
No external solidus/liquidus curves are needed.

**Phase boundary extraction.** At load time, the code extracts the liquidus boundary from the phase column: for each pressure row, the lowest temperature where the phase is `liquid` defines the liquidus.
This extracted boundary serves as the basis for an optional mushy zone controlled by the `mushy_zone_factor` parameter:

- $f = 1.0$ (default): no mushy zone; the density comes directly from the table (sharp phase transition).
- $f < 1.0$: a synthetic solidus is defined at $T_{\mathrm{sol}} = f \times T_{\mathrm{liq}}$. Between $T_{\mathrm{sol}}$ and $T_{\mathrm{liq}}$, density is volume-averaged between solid-side and liquid-side table values using the same melt-fraction formula as for PALEOS-2phase.

**T-dependent iron core.** With `PALEOS:iron`, the iron core becomes temperature-dependent for the first time in Zalmoxis.
In adiabatic mode, the core follows its own adiabat using $\nabla_{\mathrm{ad}}$ from the iron table, instead of being isothermal at the CMB temperature.

**Grid coverage.** The unified tables have the same per-cell clamping and nearest-neighbor fallback infrastructure as the PALEOS-2phase tables, handling NaN gaps at domain corners transparently.

**Mass limit.** All three tables extend to 100 TPa, supporting planets up to ~50 $M_\oplus$.

---

## Analytic Modified Polytrope (Seager et al. 2007)

The analytic EOS implements the modified polytropic fit from [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346), Table 3, Eq. 11:

$$
\rho(P) = \rho_0 + c \cdot P^n
$$

where $\rho$ is density in kg/m$^3$, $P$ is pressure in Pa, and $\rho_0$, $c$, $n$ are material-specific parameters.
This closed-form expression approximates the full merged Vinet/BME + TFD EOS without requiring tabulated data files.

Accuracy is 2--5% at low ($P < 5$ GPa) and high ($P > 30$ TPa) pressures, and degrades to up to 12% at intermediate pressures where the transition between the low-pressure and TFD regimes occurs.
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

Like the Seager et al. (2007) tabulated EOS, this assumes 300 K and no phase transitions.
Because any of the six materials can be assigned to any structural layer, the analytic EOS enables modeling of exotic compositions (e.g., carbon planets with iron/SiC or iron/graphite layers) that are not available in the tabulated data.
