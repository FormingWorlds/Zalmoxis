# Integrate an adiabat and check its step size

This tutorial runs the adiabat integration of Zalmoxis on stored structures, without a structure solve.
You will compute the temperature profile of a default planet, see how the centre temperature changes with the substep limit `max_dlnp`, and see that the top temperature is continuous when the surface pressure crosses 1 bar.
[How the adiabat is integrated](../Explanations/adiabat_integration.md) explains the method.

You need a working installation with the PALEOS tables under `data/` (see [First run](firstrun.md), Step 1).
Run every block from the repository root, in one Python session, in the order of the steps; all 3 steps together take about 3 seconds.

## Step 1: integrate a stored profile

`tests/data/adiabat_profiles.npz` holds the structure (radius, pressure, enclosed mass) and the call arguments of 6 real solves.
`call_kwargs` rebuilds the arguments of `compute_adiabatic_temperature` for one of them; `default_k0` is `input/default.toml` at a surface temperature of 2515 K.

```python
import numpy as np

from tools.validation import adiabat_profiles as ap
from zalmoxis.eos.temperature import compute_adiabatic_temperature

profiles = ap.load('tests/data/adiabat_profiles.npz')
call = ap.call_kwargs(profiles['default_k0'])
T = compute_adiabatic_temperature(**call)
P = call['pressure']
print(f'{len(P)} nodes, centre {P[0] / 1e9:.1f} GPa, top node {P[-1]:.0f} Pa')
print(f'surface {T[-1]:.1f} K, centre {T[0]:.1f} K')
```

```text
150 nodes, centre 406.9 GPa, top node 0 Pa
surface 2515.0 K, centre 6237.9 K
```

The top node has $P = 0$: this structure ends in a padded top, which the integration takes as a top at 1 bar.

## Step 2: change the substep limit

`max_dlnp` is the largest change of $\ln P$ in one substep; the default is 0.0025.
The block compares 5 values with a run at 0.0005.

```python
fine = compute_adiabatic_temperature(**call, max_dlnp=0.0005)
for h in (0.04, 0.02, 0.01, 0.005, 0.0025):
    T_h = compute_adiabatic_temperature(**call, max_dlnp=h)
    print(f'h = {h:<6}  centre {T_h[0]:7.2f} K   largest difference {np.max(np.abs(T_h - fine)):.3f} K')
```

```text
h = 0.04    centre 6223.08 K   largest difference 14.506 K
h = 0.02    centre 6241.69 K   largest difference 4.297 K
h = 0.01    centre 6239.08 K   largest difference 1.493 K
h = 0.005   centre 6236.79 K   largest difference 0.799 K
h = 0.0025  centre 6237.94 K   largest difference 0.355 K
```

The difference falls by a factor of 1.9 to 3.4 per halving of $h$, not 4: the PALEOS gradient has kinks at the table grid lines and a steep step at the iron melting boundary, where a substep loses part of its second-order accuracy.
At the default 0.0025 the profile is within 0.36 K of the finer run. The finer run is close to the exact answer: it lies within 0.018 K of the DOP853 integration that [Adiabat integration on real profiles](../Validation/thermal_state.md#adiabat-integration-on-real-profiles) uses as its reference.

## Step 3: move the top pressure around 1 bar

`ic_dummy` is the PROTEUS initial structure of the `dummy` setup, with the CMB anchor and its top near 1 atm.
The block sets the top pressure to 5 values around 1 bar.

```python
dummy = ap.call_kwargs(profiles['ic_dummy'])
tops = (0.95e5, 0.99e5, 1.0e5, 1.01e5, 1.05e5)
top_T = []
for p_top in tops:
    P_d = dummy['pressure'].copy()
    P_d[-1] = p_top
    top_T.append(compute_adiabatic_temperature(**{**dummy, 'pressure': P_d})[-1])
for p_top, t in zip(tops, top_T):
    print(f'top {p_top / 1e5:.2f} bar: top node {t:.3f} K, {1e3 * (t - top_T[-1]):+.3f} mK from 1.05 bar')
```

```text
top 0.95 bar: top node 4422.885 K, -0.136 mK from 1.05 bar
top 0.99 bar: top node 4422.885 K, -0.136 mK from 1.05 bar
top 1.00 bar: top node 4422.885 K, -0.136 mK from 1.05 bar
top 1.01 bar: top node 4422.885 K, -0.109 mK from 1.05 bar
top 1.05 bar: top node 4422.885 K, +0.000 mK from 1.05 bar
```

Below 1 bar the temperature is held, so the 3 lowest values are equal; above it the top node changes by a fraction of a millikelvin.
The temperature therefore changes continuously with the top pressure.

**Where this is checked.** [Temperature across 1 bar](../Validation/thermal_state.md#temperature-across-1-bar) and [Adiabat integration on real profiles](../Validation/thermal_state.md#adiabat-integration-on-real-profiles).
