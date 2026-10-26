# Check that a structure meets its surface pressure

This tutorial shows how to check that a Zalmoxis structure meets its surface boundary condition: the pressure at the outer radius $R$ must equal the target surface pressure $P_t$.
You will run a 1 $M_\oplus$ planet, read the pressure at its outer radius, look at the central pressures that the pressure solve tried, see how the solve refines a root whose integration stops below the surface, and compare a few solver setups.

You need a working installation with the PALEOS tables under `data/` (see [First run](firstrun.md), Step 1).
Run every block from the repository root, in one Python session, in the order of the steps.
The whole tutorial took about 10 minutes on a workstation; Step 4 takes about 5.5 minutes of it.

## Step 1: solve a structure and read its surface pressure

Zalmoxis integrates the structure equations from the centre, with a trial central pressure $P_c$, out to the outer radius $R$.
The pressure solve adjusts $P_c$ until the pressure at the outer radius, $P(R)$, equals the target $P_t$ (`target_surface_pressure`, 1 atm $= 101325$ Pa by default).
The function below solves `input/default.toml` with the changes you pass to it, and returns the configuration and the result.

```python
import os

import numpy as np

import zalmoxis
from zalmoxis import solver
from zalmoxis.config import (
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
)

root = zalmoxis.get_zalmoxis_root()
materials = load_material_dictionaries()


def solve(**changes):
    """Solve input/default.toml with the given changes; return the config and the result."""
    cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    cfg.update(data_output_enabled=False, plotting_enabled=False, **changes)
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        cfg['rock_solidus'],
        cfg['rock_liquidus'],
        cfg['mushy_zone_factor'],
    )
    return cfg, solver.main(cfg, materials, melt, os.path.join(root, 'input'))


def report(cfg, result):
    """Print how the result was accepted and how it meets the surface pressure."""
    P = np.asarray(result['pressure'])
    P_t = cfg['target_surface_pressure']
    print(f"accepted on {result['accepted_on']}, converged {result['converged']}")
    print(f'P(R) = {P[-1]:.4g} Pa, P_t = {P_t:.4g} Pa, miss {abs(P[-1] - P_t) / P_t:.3f} P_t')
    print(f"padded {result['padded']}, nodes with P = 0: {np.sum(P == 0)} of {len(P)}")


cfg, result = solve()
report(cfg, result)
```

On our run this printed:

```text
accepted on strict, converged True
P(R) = 0 Pa, P_t = 1.013e+05 Pa, miss 1.000 P_t
padded True, nodes with P = 0: 1 of 150
```

The solve accepted the profile, but its outer node has $P = 0$, not $P_t$: the profile is **padded**.
The integration from the accepted $P_c$ stopped at $P = 0$ just below the outer radius, and the node past the stop got $P = 0$.
The solver logs one warning for such a result (`The accepted profile ends in a surface stop: ...`).
The accept test of the pressure solve passes a profile when $\lvert P(R) - P_t \rvert$ is below `pressure_tolerance` ($10^9$ Pa by default), so it passes this one, whose miss is $P_t$.
`accepted_on` names the test that accepted the structure: `strict` means that the mass, density and pressure tests all passed ([Convergence Checks](../Explanations/process_flow.md#convergence-checks) lists every label).
Whether a profile ends padded depends on round-off, so your run can end reached; Step 2 shows the trials that decide it.

## Step 2: look at the central pressures the solve tried

The pressure solve looks for the root of the residual

$$
f(P_c) = P(R) - P_t,
$$

the pressure at the outer radius minus the target, as a function of the trial central pressure.
A trial $P_c$ that is too low lets the pressure reach 0 before $R$: the integration stops there (a **surface stop**), the profile is padded, and $f = -P_t$.
A trial $P_c$ whose integration reaches the outer radius gives a **reached** profile with $f = P(R) - P_t$.
Two neighbouring trial pressures with a negative residual at the lower one and a positive residual at the higher one form a **sign pair**: the root lies between them, and the higher one is the reached end.

The solver does not return its trials, but at the `DEBUG` log level it writes the pressure profile of every trial to `output/pressure_profiles.txt`.
The first row of each profile is $P_c$ and the last row is $P(R)$.
The block below solves the planet again at that level, which writes this file and `output/density_profiles.txt` (about 50 MB each), and reads the trials of the last pressure solve of the run.

```python
import logging

logging.getLogger('zalmoxis').setLevel(logging.DEBUG)
cfg, result = solve()
logging.getLogger('zalmoxis').setLevel(logging.WARNING)


def last_pressure_solve(path):
    """(P_c, P(R)) [Pa] of each trial of the last pressure solve in a DEBUG dump."""
    solves = []
    for block in open(path).read().split('# Pressure iteration ')[1:]:
        number, _header, *rows = block.strip().splitlines()
        if int(number) == 0:  # the count starts again for each pressure solve
            solves.append([])
        pressure = np.loadtxt(rows)[:, 1]
        solves[-1].append((pressure[0], pressure[-1]))
    return np.array(sorted(solves[-1]))


trials = last_pressure_solve(os.path.join(root, 'output', 'pressure_profiles.txt'))
P_t = cfg['target_surface_pressure']
f = trials[:, 1] - P_t
print(f'{len(trials)} trials; accepted P_c = {result["p_center"]:.6e} Pa')
for (p_c, p_r), fi in zip(trials, f):
    if abs(p_c - result['p_center']) < 2e7:
        label = 'padded' if p_r == 0 else 'reached'
        print(f'P_c = {p_c:.6e} Pa  P(R) = {p_r:10.4g} Pa  f = {fi:+.4g} Pa  {label}')
```

On our run this printed:

```text
33 trials; accepted P_c = 3.861929e+11 Pa
P_c = 3.861779e+11 Pa  P(R) =          0 Pa  f = -1.013e+05 Pa  padded
P_c = 3.861862e+11 Pa  P(R) =          0 Pa  f = -1.013e+05 Pa  padded
P_c = 3.861903e+11 Pa  P(R) =          0 Pa  f = -1.013e+05 Pa  padded
P_c = 3.861924e+11 Pa  P(R) =          0 Pa  f = -1.013e+05 Pa  padded
P_c = 3.861929e+11 Pa  P(R) =          0 Pa  f = -1.013e+05 Pa  padded
P_c = 3.861934e+11 Pa  P(R) =  4.169e+07 Pa  f = +4.159e+07 Pa  reached
P_c = 3.861945e+11 Pa  P(R) =  4.439e+07 Pa  f = +4.429e+07 Pa  reached
```

The loop prints only the trials within 20 MPa of the accepted $P_c$; the others are far from the root.
These trials show a jump of $f$: the padded trials sit at $f = -P_t$, and the first reached trial, only 0.5 MPa higher, has $P(R) = 4.2 \times 10^7$ Pa, about 400 times $P_t$.
These two trials are the sign pair around the root.
Brent's method, the root finder of the pressure solve, stops when its bracket is narrower than $10^6$ Pa and returns the end with the smaller $\lvert f \rvert$.
When the reached end has $P(R) > 2 P_t$, its $\lvert f \rvert$ is larger than $P_t$, so Brent's method returns the padded end.
Here the reached end misses $P_t$ by much more than $P_t$, so the refinement of Step 3 has no reached candidate within $P_t$, and the solve keeps the padded root.
The plot below shows the trials near the root.

```python
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

near = np.abs(trials[:, 0] - result['p_center']) < 2e7
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot((trials[near, 0] - result['p_center']) / 1e6, f[near] / 1e5, 'o')
ax.axhline(0.0, ls='--', color='grey')
ax.set_xlabel(r'$P_c$ - accepted $P_c$ [MPa]')
ax.set_ylabel(r'$f = P(R) - P_t$ [bar]')
fig.tight_layout()
fig.savefig(os.path.join(root, 'output', 'surface_pressure_residual.png'))
```

Open `output/surface_pressure_residual.png`: the padded trials lie at $-1.01$ bar, just below the dashed zero line, and the 2 reached trials lie at about 416 and 443 bar.

## Step 3: see how the solve refines a padded root

After Brent's method returns a root that misses $P_t$ by more than $0.1 P_t$, the solve can run a second, finer Brent solve on the nearest sign pair, and it keeps the candidate closest to the target ([Padded accepts](../Explanations/process_flow.md#padded-accepts) gives the full rule).
Whether a real solve needs this depends on round-off, so this step uses a residual written down by hand (and `P_t` from Step 2), the one of the validation case [Surface boundary condition of the pressure solve](../Validation/hydrostatic_structure.md#surface-boundary-condition-of-the-pressure-solve).
Below $x_0 = 2 \times 10^{11}$ Pa the integration stops; above it $P(R)$ rises from $0.95\,P_t$ with a slope $dP(R)/dP_c$ of 0.47.
The refinement is the private function `solver._refine_pressure_root`; the block calls it directly to show the rule, and a real solve calls it for you.

```python
x0, slope = 2.0e11, 0.47
evals = {}


def residual(p_c):
    """f(P_c) [Pa] of the hand-made residual; every call is recorded in evals."""
    p_r = 0.0 if p_c < x0 else 0.95 * P_t + slope * (p_c - x0)
    evals[p_c] = p_r - P_t
    return evals[p_c]


first, info = solver.brentq(residual, x0 - 4e6, x0 + 4e6, xtol=1e6, rtol=1e-10, full_output=True)
refined, _ = solver._refine_pressure_root(first, info, evals, residual, P_t, 100)
for name, p_c in (('Brent root', first), ('refined root', refined)):
    p_r = evals[p_c] + P_t
    print(f'{name}: P_c - x0 = {(p_c - x0) / 1e6:+.4f} MPa, P(R) = {p_r:.5g} Pa, miss {abs(evals[p_c]) / P_t:.3f} P_t')
```

On our run this printed:

```text
Brent root: P_c - x0 = -0.7362 MPa, P(R) = 0 Pa, miss 1.000 P_t
refined root: P_c - x0 = +0.0262 MPa, P(R) = 1.0857e+05 Pa, miss 0.071 P_t
```

The arguments of `_refine_pressure_root` are the Brent root, its Brent record `info`, the recorded residuals `evals`, the function `residual`, the target `P_t`, and 100, the iteration limit of the second Brent solve.
The Brent root is padded: $P(R) = 0$, a miss of the full $P_t$.
The second solve on the sign pair around $x_0$ finds a reached root that misses $P_t$ by less than $0.1 P_t$.

## Step 4: compare solver setups

The script `tools/validation/pressure_root_sweep.py` runs 26 solver setups built from `input/default.toml` and records how each accepted profile meets $P_t$.
The `case` command runs one setup and writes its record.
The setups are numbered from 0 to 25. Run 3 of them: 1 (the Analytic EOS, a closed-form equation of state without tables, at 1 $M_\oplus$ with the Picard outer solver), 8 (`input/default.toml` itself, PALEOS tables and the Picard outer solver) and 9 (the same planet with the Newton outer solver at `relative_tolerance` $10^{-9}$).
The outer solver finds the radius at which the computed mass equals the planet mass. The Picard outer solver, the default, rescales the radius after each structure solve; the Newton outer solver uses Newton's method, which follows the slope of the mass error with the radius. `relative_tolerance` is the relative tolerance of the ODE integrator, and the label `newton_tol` means that the Newton outer solver accepted the structure on its mass error.
Setup 9 takes about 4 minutes.

```console
mkdir -p output/sweep
python -m tools.validation.pressure_root_sweep case 1 output/sweep
python -m tools.validation.pressure_root_sweep case 8 output/sweep
python -m tools.validation.pressure_root_sweep case 9 output/sweep
```

Then read the records:

```python
import json

for i in (1, 8, 9):
    with open(os.path.join(root, 'output', 'sweep', f'setup_{i}.json')) as file:
        rec = json.load(file)
    miss = abs(rec['P_R'] - rec['target']) / rec['target']
    print(f"setup {i} {rec['name']}: {rec['accepted_on']}, P(R) = {rec['P_R']:.4g} Pa, miss {miss:.3f} P_t, padded nodes {rec['padded']}")
```

On our run this printed:

```text
setup 1 Analytic 1: strict, P(R) = 1.017e+05 Pa, miss 0.003 P_t, padded nodes 0
setup 8 PALEOS 1 (default.toml): strict, P(R) = 0 Pa, miss 1.000 P_t, padded nodes 1
setup 9 PALEOS 1, Newton: newton_tol, P(R) = 1.393e+05 Pa, miss 0.375 P_t, padded nodes 0
```

The command `python -m tools.validation.pressure_root_sweep run <dir> -j 2` runs all 26 setups (about 30 minutes with 2 processes), and `python -m tools.validation.pressure_root_sweep plot <dir> --assets <dir>` draws them into `<dir>` (it needs the `proteus_mpl` package and a record of every setup; without `--assets` it writes over the figures under `docs/assets/validation/`); the figure of all 26 setups is on the [validation page](../Validation/hydrostatic_structure.md#surface-boundary-condition-in-every-newton-setup).
In it, 15 of the 19 Picard setups that return a profile end padded, and 5 of the 6 Newton setups at `relative_tolerance` $10^{-9}$ reach the outer radius; setup 10 (PALEOS on the JAX path) ends padded.
Reached means that the integration reaches $R$, not that $P(R) = P_t$: setup 9 above misses $P_t$ by 0.375 $P_t$.

## Step 5: what to do with a padded or missing surface pressure

First decide whether the surface values matter for your use.
A padded result passed the accept tests of the solve, and in this tutorial only its outermost node has $P = 0$ instead of $P_t$.
If you need $P(R) = P_t$, these settings matter:

- **The outer solver.** `outer_solver = "newton"` with `relative_tolerance = 1e-9` reached the outer radius in setup 9 above, where the default Picard solve ended padded; it still missed $P_t$ by 0.375 $P_t$. PROTEUS uses the Newton outer solver by default. It takes longer (about 4 minutes here against 1.5).
- **`pressure_tolerance`.** This is the accept test of the pressure solve, $\lvert P(R) - P_t \rvert$ below it. A value below $P_t$ makes the test refuse a padded profile, but the outer loop can still accept the structure on its mass test, as the block below shows.
- **The integration tolerances** (`relative_tolerance`, `absolute_tolerance`). They change where the integration stops, and so whether the trials near the root pad. Setup 9 changes them together with the outer solver, so this tutorial does not separate the two effects.

```python
cfg, result = solve(pressure_tolerance=5e4)
report(cfg, result)
print(f"pressure test passed: {result['converged_pressure']}")
```

On our run this printed:

```text
accepted on mass_only, converged True
P(R) = 0 Pa, P_t = 1.013e+05 Pa, miss 1.000 P_t
padded True, nodes with P = 0: 1 of 150
pressure test passed: False
```

With `pressure_tolerance` below $P_t$ the pressure test fails, and the outer loop accepts the structure on its mass test alone (`mass_only`: the mass converged, the density or the pressure did not): the profile stays padded.
So check `padded` and $P(R)$ of every result whose surface values you use, as in Step 1.

## Summary

- `result['pressure'][-1]` is $P(R)$; compare it with `target_surface_pressure`.
- `result['padded']` is True when the outer node has $P = 0$; Zalmoxis then logs one warning.
- The pressure solve refines a root that misses $P_t$ by more than $0.1 P_t$, and keeps the candidate closest to $P_t$.
- The Newton outer solver at `relative_tolerance` $10^{-9}$ reached the outer radius in this tutorial, with a miss of 0.375 $P_t$; a lower `pressure_tolerance` alone did not reach it.

[Padded accepts](../Explanations/process_flow.md#padded-accepts) explains the full rule, and the [validation page](../Validation/hydrostatic_structure.md#surface-boundary-condition-of-the-pressure-solve) shows how it is tested.
