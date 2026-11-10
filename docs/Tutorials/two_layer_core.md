# A two-layer Earth core

This tutorial solves a 1 $M_\oplus$ planet whose core has a solid inner core and a liquid outer core with sulfur. You set the inner-core boundary (ICB) in 2 ways, by a given inner-core mass and by an iron melting curve, and read the core arrays of the result. The [two-layer core](../Explanations/mixing.md#two-layer-core) explanation describes the model.

You need:

- a working Zalmoxis environment with the PALEOS tables ([First run](firstrun.md), Step 1);
- the PALEOS Python package, installed from a git checkout that contains the tested PALEOS commit ([PALEOS-API live tabulation](../Reference/api/zalmoxis.eos.paleos_api.md)).

The PALEOS package makes the 2 iron side tables: one table of solid iron and one of liquid iron, each over the pressure and temperature range of the PALEOS iron table.

## Step 1: write the script

Make a folder `scripts/` in the repository root and save this script as `scripts/two_layer_core.py`. It starts from `input/default.toml`, puts the outer core on the liquid iron side with 13.85 wt% S and the inner core on the solid iron side, and solves the planet once for each ICB rule.

```python
# scripts/two_layer_core.py
"""Solve a 1 M_E planet with a two-layer core and print its inner-core boundary."""
import os

import numpy as np

from zalmoxis import get_zalmoxis_root
from zalmoxis.config import (
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
    validate_config,
)
from zalmoxis.mixing import parse_all_layer_mixtures
from zalmoxis.solver import main


def solve(**rule):
    """The default 1 M_E planet with a liquid S-bearing outer core and a solid iron inner core."""
    root = get_zalmoxis_root()
    cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    cfg['layer_eos_config'].update(
        core='PALEOS-API-2phase:iron-liquid:0.8615+CoreAlloy:S:0.1385',
        inner_core='PALEOS-API-2phase:iron-solid',
    )
    cfg.update(rule)
    validate_config(cfg)
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'], cfg['rock_solidus'], cfg['rock_liquidus'], mushy_zone_factor=1.0
    )
    mixes = parse_all_layer_mixtures(cfg['layer_eos_config'])
    return main(cfg, load_material_dictionaries(), melt, os.path.join(root, 'input'), layer_mixtures=mixes)


if __name__ == '__main__':
    for rule in ({'inner_core_mass_fraction': 0.0508}, {'core_liquidus': 'Anzellini13-iron'}):
        res = solve(**rule)
        core = res['core']
        inner = core['phase'] == 1
        print(rule)
        print(f"  converged {res['converged']}, radius {res['radii'][-1] / 1e3:.1f} km")
        print(f"  CMB {core['r'][-1] / 1e3:.1f} km, {core['P'][-1] / 1e9:.2f} GPa, {core['T'][-1]:.0f} K")
        print(f"  ICB {core['R_icb'] / 1e3:.1f} km, {core['P_icb'] / 1e9:.2f} GPa, M_ic/M_core {core['M_ic'] / res['cmb_mass']:.4f}")
        print(f"  centre {core['P'][0] / 1e9:.2f} GPa, {core['T'][0]:.0f} K, {core['rho'][0]:.0f} kg/m^3")
        if inner.any() and (~inner).any():
            print(f"  density step at the ICB {core['rho'][inner][-1]:.0f} -> {core['rho'][~inner][0]:.0f} kg/m^3")
        print(f"  solid above the ICB: {core['solid_above_icb_mass_fraction']:.3f} of the core mass")
```

The script sets the ICB with 2 rules. A two-layer core takes exactly one of the 2:

- `inner_core_mass_fraction = 0.0508` gives the inner core this fraction of the core mass, the inner-core mass fraction of PREM, the Preliminary Reference Earth Model of Dziewonski & Anderson (1981)[^cite-dziewonski1981].
- `core_liquidus = "Anzellini13-iron"` puts the ICB where the temperature profile reaches the melting curve of pure iron of Anzellini et al. (2013)[^cite-anzellini2013].

Mori et al. (2017)[^cite-mori2017] show that S lowers the melting point of iron. A Fe-S outer core can therefore melt below the pure-iron curve, and the solve warns that the inner core can come out too large; a caller with the liquidus of the alloy passes it as a callable.

## Step 2: run it

From the repository root:

```console
python scripts/two_layer_core.py
```

The first run makes the 2 iron side tables in `data/EOS_PALEOS_API/`, once per PALEOS version: 1351 by 379 nodes per side. Later runs read the cached tables. With the tables cached, the 2 solves took 3 min 12 s together on an Apple M1 Ultra, next to other jobs.

The script prints, for each rule:

```console
{'inner_core_mass_fraction': 0.0508}
  converged True, radius 6561.4 km
  CMB 3477.5 km, 128.22 GPa, 5045 K
  ICB 1211.6 km, 322.18 GPa, M_ic/M_core 0.0508
  centre 358.54 GPa, 7708 K, 13439 kg/m^3
  density step at the ICB 13164 -> 12049 kg/m^3
  solid above the ICB: 0.000 of the core mass
{'core_liquidus': 'Anzellini13-iron'}
  converged True, radius 6563.4 km
  CMB 3483.5 km, 127.78 GPa, 5040 K
  ICB 0.0 km, 349.00 GPa, M_ic/M_core 0.0000
  centre 349.00 GPa, 7735 K, 12282 kg/m^3
  solid above the ICB: 0.000 of the core mass
```

With no inner core, `R_icb` is 0 and `P_icb` is the centre pressure.

## Step 3: read the result

`res['core']` holds the core arrays. The script reads `r`, `P`, `T` and `rho` (m, Pa, K and kg/m$^3$) at the core nodes, whose last node is the core-mantle boundary (CMB); `phase` (1 inner core, 0 outer core); and the scalars `M_ic` \[kg\], `R_icb` \[m\], `P_icb` \[Pa\] and `solid_above_icb_mass_fraction`. `res['cmb_mass']` is the core mass. [Core arrays](../How-to/usage.md#core-arrays) lists every key with its unit. The 2 runs differ as follows:

1. With the inner-core mass of PREM, the ICB lies at 1212 km and 322 GPa. The density falls there by about 1100 kg/m$^3$, from the solid iron of the inner core to the liquid Fe-S of the outer core.
2. With the melting curve, the core has no inner core. The adiabat from 3000 K at the surface (the `surface_temperature` of `input/default.toml`) reaches 7735 K at the centre, above the 6403 K of the Anzellini curve at 349 GPa.
3. The inner core of the first run is solid at 7708 K at the centre, above the melting curve. The mass rule sets where the boundary is; it does not check that iron is solid there.

To see an inner core from the melting curve, cool the core: `solve(core_liquidus='Anzellini13-iron', temperature_mode='isothermal', surface_temperature=5000)` puts the ICB at 206.6 GPa, next to the 206.4 GPa where the Anzellini curve is at 5000 K.

To plot the density of the core, run these lines from the `scripts/` folder, in a Python session or a second script; the import reads `solve` from Step 1:

```python
import matplotlib.pyplot as plt
from two_layer_core import solve

res = solve(inner_core_mass_fraction=0.0508)
core = res['core']
plt.plot(core['r'] / 1e3, core['rho'])
plt.xlabel('Radius [km]')
plt.ylabel('Density [kg/m$^3$]')
plt.show()
```

## What to try next

- Put another light element in the outer core, for example `CoreAlloy:O:0.0865` with the liquid side at 0.9135.
- Give the inner core light elements: `inner_core = "PALEOS-API-2phase:iron-solid:0.99+CoreAlloy:Si:0.01"`. The light-element laws are calibrated on liquid iron, so the solve warns once per process that it applies them to solid iron; `CoreAlloy:H-Luo` is refused on the solid side.
- Pass a callable `T_liq(P)` \[K\] as `core_liquidus`, for example the liquidus of an alloy.

[^cite-anzellini2013]: S. Anzellini, A. Dewaele, M. Mezouar, P. Loubeyre, G. Morard, *[Melting of Iron at Earth's Inner Core Boundary Based on Fast X-ray Diffraction](https://doi.org/10.1126/science.1233514)*, Science, 340, 464-466, 2013. [SciX](https://scixplorer.org/abs/2013Sci...340..464A/abstract).
[^cite-mori2017]: Y. Mori, H. Ozawa, K. Hirose, R. Sinmyo, S. Tateno, G. Morard, Y. Ohishi, *[Melting experiments on Fe-Fe$_3$S system to 254 GPa](https://doi.org/10.1016/j.epsl.2017.02.021)*, Earth and Planetary Science Letters, 464, 135-141, 2017. [SciX](https://scixplorer.org/abs/2017E%26PSL.464..135M/abstract).
[^cite-dziewonski1981]: A. M. Dziewonski, D. L. Anderson, *[Preliminary reference Earth model](https://doi.org/10.1016/0031-9201(81)90046-7)*, Physics of the Earth and Planetary Interiors, 25, 297-356, 1981. [SciX](https://scixplorer.org/abs/1981PEPI...25..297D/abstract).
