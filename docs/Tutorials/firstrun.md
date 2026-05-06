# First run

This tutorial walks a 1 $M_\oplus$ rocky planet through Zalmoxis end to end.
By the end you will have written a minimal config file, run the structure solver, plotted a radial density profile, and verified the result against the [Zeng et al. (2019)](https://ui.adsabs.harvard.edu/abs/2019PNAS..116.9723Z) Earth-like mass-radius curve and the [Seager et al. (2007)](https://ui.adsabs.harvard.edu/abs/2007ApJ...669.1279S) tabulated density profile.

You should already have Zalmoxis installed in a working conda environment.
If not, follow [Installation](../How-to/installation.md) first; the steps below assume `python -m zalmoxis` runs without an `ImportError`.

## Step 1: set up your environment

Activate the environment Zalmoxis was installed into:

```console
conda activate zalmoxis      # or: conda activate proteus
```

Verify the package imports and that the repository root resolves:

```console
python -c "import zalmoxis; print(zalmoxis.__version__, zalmoxis.get_zalmoxis_root())"
```

`get_zalmoxis_root()` walks up from the installed package until it finds the repository.
You normally do **not** need to set `ZALMOXIS_ROOT` by hand.
Set it explicitly only if auto-detection fails (non-standard installation layouts, frozen wheels, etc.); see the [installation troubleshooting](../How-to/installation.md#zalmoxis_root-not-set) section.

Download the equation-of-state tables (about 800 MB on disk):

```console
bash tools/setup/get_zalmoxis.sh
```

This populates `data/` inside the repository (Seager+2007 lookups, PALEOS unified tables, Zeng+2019 reference curves) and creates an empty `output/` directory.
You should now see populated subdirectories under `data/`, for example `data/EOS_Seager2007/` and `data/EOS_PALEOS/`.

!!! note "Within PROTEUS"
    When Zalmoxis runs inside PROTEUS, EOS data lives under `FWL_DATA` and the script above is not used.
    See [Coupling Zalmoxis to PROTEUS](../How-to/proteus_coupling.md) for that workflow.

## Step 2: build a minimal input file

Create `input/firstrun.toml` with the following contents.
Every key here also exists in `input/default.toml`; the version below is stripped to the smallest set that uniquely defines an Earth-like rocky planet.

```toml
# input/firstrun.toml
# Minimal 1 M_earth rocky-planet config.

[InputParameter]
planet_mass                = 1          # M_earth

[AssumptionsAndInitialGuesses]
core_mass_fraction         = 0.325      # Earth's core fraction
mantle_mass_fraction       = 0          # 0 = mantle fills the remainder (2-layer model)
temperature_mode           = "adiabatic"
surface_temperature        = 3000       # K
center_temperature         = 6000       # K, initial guess for the adiabat

[EOS]
core                       = "PALEOS:iron"
mantle                     = "PALEOS:MgSiO3"
ice_layer                  = ""         # empty = 2-layer model

[Calculations]
num_layers                 = 150        # radial grid points

[PressureAdjustment]
target_surface_pressure    = 101325     # Pa, 1 atm

[Output]
data_enabled               = true
plots_enabled              = false
```

A few notes on what each block does:

- `[InputParameter]` only carries `planet_mass`.
  Validated range is 0.1 to 50 $M_\oplus$; outside it the config loader raises before the solver starts.
- `[AssumptionsAndInitialGuesses]` sets the temperature treatment and the core/mantle split.
  `temperature_mode = "adiabatic"` integrates the layer adiabats with `surface_temperature` as the upper boundary and `center_temperature` as the initial guess for the inner boundary.
  Other choices are `"isothermal"`, `"linear"`, and `"prescribed"` (reads `temperature_profile_file`); see [Configuration](../How-to/configuration.md) for the full list.
- `[EOS]` selects the per-layer equation of state.
  `PALEOS:iron` and `PALEOS:MgSiO3` are unified multi-phase PALEOS tables ([Attia et al. (2026)](https://ui.adsabs.harvard.edu/abs/2026arXiv260503741A/abstract)) that handle pressures up to 100 TPa and masses up to 50 $M_\oplus$.
  `ice_layer = ""` keeps the model two-layer (core + mantle).
- `[Calculations]` controls the radial grid resolution.
  150 points is enough for a 3 % radius match; double it for tighter convergence at high mass.
- `[PressureAdjustment]` sets the target surface pressure that the outer mass-radius loop drives towards.
- `[Output]` turns on the CSV/text writers.
  Leave `plots_enabled = false` for now; we will plot the profile by hand in Step 4.

Numerical solver tolerances are intentionally absent.
The solver picks mass-adaptive defaults internally; override them only when debugging convergence (add an `[IterativeProcess]` block with the keys listed in the comments of `input/default.toml`).

## Step 3: run the solver

From the repository root:

```console
python -m zalmoxis -c input/firstrun.toml
```

Runtime on a recent laptop is roughly 30 to 90 seconds for this configuration.
Most of that goes into the outermost mass-radius loop; it iterates a Picard step on the central pressure until the surface mass and pressure both match their targets.
The solver has three nested loops: a structure ODE for `(M(r), g(r), P(r))` (innermost), a density Picard iteration that re-evaluates the EOS at the new `(P, T)`, and the outer mass-radius loop just mentioned.

Progress goes to `output/zalmoxis.log` (the path is hard-coded in `src/zalmoxis/__main__.py`).
On a successful run you should see lines along the lines of:

```console
INFO - Exoplanet Internal Structure Model Results:
INFO - Calculated Planet Mass: 5.97e+24 kg or 1.00 Earth masses
INFO - Calculated Planet Radius: 6.34e+06 m or 1.00 Earth radii
INFO - Core Radius: 3.48e+06 m
INFO - Pressure at Core-Mantle Boundary (CMB): 1.35e+11 Pa
INFO - Pressure at Center: 3.65e+11 Pa
INFO - Overall Convergence Status: True with Pressure: True, Density: True, Mass: True
```

Note that the "Earth radii" comparison uses Zalmoxis's reference value `earth_radius = 6.335e6` m (Seager+2007 volumetric mean), not the IAU 6.371e6 m.
That convention is intentional and matches the Seager reference data shipped in `data/`.

If the final line shows `Convergence Status: False`, see [Convergence failures](../How-to/installation.md#convergence-failures) for the standard remedies (looser tolerance, larger pressure bracket, or a different mantle EOS for high-mass configurations).

## Step 4: inspect the output

After a successful run the `output/` directory contains:

| File | Contents |
| --- | --- |
| `zalmoxis.log` | Run log (overwritten on every run, `filemode='w'`). |
| `planet_profile.txt` | Radial profile. Six tab-separated columns: radius (m), density (kg m$^{-3}$), gravity (m s$^{-2}$), pressure (Pa), temperature (K), enclosed mass (kg). Header line begins with `#`. |
| `calculated_planet_mass_radius.txt` | One row appended per run: total mass (kg) and total radius (m). |

To plot the density profile:

```python
# scripts/plot_firstrun.py
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('output/planet_profile.txt')
radii_km   = data[:, 0] / 1e3
density    = data[:, 1]
pressure_GPa = data[:, 3] / 1e9

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(radii_km, density, lw=2)
ax1.set_xlabel('Radius (km)')
ax1.set_ylabel(r'Density (kg m$^{-3}$)')
ax1.grid(alpha=0.3)

ax2.plot(radii_km, pressure_GPa, lw=2, color='C3')
ax2.set_xlabel('Radius (km)')
ax2.set_ylabel('Pressure (GPa)')
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig('firstrun_profile.pdf')
plt.show()
```

You should see a sharp density jump near $r \approx 3500$ km (the core-mantle boundary), an inner core density of order 13000 kg m$^{-3}$, and a central pressure near 365 GPa.

## Step 5: verify the result

The smoke test `tests/test_MR_rocky.py::test_rocky_1Mearth_vs_zeng_and_seager` exercises the same code path you just ran, with two reference checks bolted on.
You can reproduce them by hand.

**Mass-radius (Zeng+2019).**
The reference curve `data/Zeng2019/massradiusEarthlikeRocky.txt` is an Earth-like rocky composition (32.5 % iron core, 67.5 % MgSiO3 mantle, 300 K).
Interpolate it at your computed mass and compare:

```python
import numpy as np

zeng = np.loadtxt('data/Zeng2019/massradiusEarthlikeRocky.txt', skiprows=1)
zeng_mass, zeng_radius = zeng[:, 0], zeng[:, 1]   # M_earth, R_earth

mr = np.loadtxt('output/calculated_planet_mass_radius.txt', skiprows=1)
m_kg, r_m = mr[-1]                                # last row = this run
earth_mass, earth_radius = 5.972e24, 6.335439e6
m_e, r_e = m_kg / earth_mass, r_m / earth_radius

zeng_at_mass = 10 ** np.interp(np.log10(m_e),
                               np.log10(zeng_mass),
                               np.log10(zeng_radius))
print(f'M = {m_e:.3f} M_earth, R = {r_e:.3f} R_earth, '
      f'Zeng+2019 = {zeng_at_mass:.3f} R_earth, '
      f'rel. diff = {(r_e - zeng_at_mass) / zeng_at_mass:+.2%}')
```

The smoke test asserts `rtol = 0.03` (3 %).
Your run should be well inside that envelope.

**Density profile (Seager+2007).**
The file `data/Seager2007/radiusdensitySeagerEarthbymass.txt` carries tabulated Earth-mass density profiles from the same paper.
Loading the 1 $M_\oplus$ slice and overlaying it on your `planet_profile.txt` should match within 10 % everywhere except for a narrow band around the core-mantle boundary.
The CMB lies between Seager's grid points, so interpolating across the discontinuity inflates the residual artificially; the smoke test masks $\pm 3$ grid points around any density jump greater than 2000 kg m$^{-3}$ before comparing.
See `tests/test_MR_rocky.py` for the exact masking logic if you need to reproduce the assertion verbatim.

!!! info "Run the smoke test directly"
    `python -m pytest -o "addopts=" tests/test_MR_rocky.py -v` runs the same comparison Zalmoxis's CI uses on every push.
    The `-o "addopts="` override is needed because `pyproject.toml` defaults to `pytest-xdist` parallel execution.

## Step 6: sweep a parameter

The `tools.grids.run_grid` runner takes a small TOML, expands the Cartesian product of every list it finds in `[sweep]`, and runs each grid point with multiprocessing.
Save the following as `input/grids/firstrun_sweep.toml`:

```toml
# input/grids/firstrun_sweep.toml
[base]
config = "input/firstrun.toml"

[sweep]
planet_mass = [0.5, 1.0, 2.0]

[output]
dir = "output/grid_firstrun"
save_profiles = true
```

Run it with two workers:

```console
python -m tools.grids.run_grid input/grids/firstrun_sweep.toml -j 2
```

Outputs land in `output/grid_firstrun/`:

- `grid_summary.csv`: one row per grid point with the sweep parameters, the converged mass and radius, the convergence flags, and the wall time;
- `<label>.json`: per-run summary mirroring the CSV row;
- `<label>.csv`: per-run radial profile (because `save_profiles = true`); a six-column body with SI-unit columns plus a `# key: value` comment header that records the EOS strings, melting-curve ids, and convergence flag.

Filter to converged rows (`converged == True`) and overlay the three radii on the Zeng+2019 curve to check that you reproduce the expected shallow $R \propto M^{0.27}$ slope for rocky bodies in this mass range.
The `tools/grids/plot_grid.py` helper does this for you:

```console
python -m tools.grids.plot_grid output/grid_firstrun
```

Browse `input/grids/` for richer examples: `mass_radius.toml` is a 7-point version of the sweep above, `mass_temperature.toml` adds a temperature axis, and `h2o_mixing.toml` / `h2_mixing.toml` exercise multi-component mantle EOS mixing.

## Where to go next

- For the full set of TOML keys, validators, and defaults, read [Configuration](../How-to/configuration.md).
- For the equations behind the structure ODE, the EOS dispatch, and the mushy-zone mixing model, read [Model overview](../Explanations/model.md) and [Equation of state physics](../Explanations/eos_physics.md).
- For the function-call coupling used when Zalmoxis runs inside PROTEUS (the `python -m zalmoxis` invocation in this tutorial does *not* apply there), read [Coupling Zalmoxis to PROTEUS](../How-to/proteus_coupling.md) and [Coupling to PROTEUS (theory)](../Explanations/proteus_coupling.md).
- For first-principles verification cases (analytic polytropes, hydrostatic equilibrium tests, mass conservation), read [Verification](../Explanations/verification.md).
