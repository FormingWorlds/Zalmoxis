# API overview

This is a detailed overview of Zalmoxis' API for the user's reference. If you want to understand the underlying model, please visit the [model overview](../../Explanations/model.md).

## Module overview

```
src/zalmoxis/
├── config.py             # Config loading, parsing, validation, EOS setup
├── solver.py             # main() solver loop (3 nested iterations)
├── output.py             # post_processing(), file output
├── structure_model.py    # ODE system: coupled_odes(), solve_structure()
├── eos/                  # EOS package, organized by family
│   ├── interpolation.py  # Grid builders, bilinear interp, table loaders
│   ├── seager.py         # Seager2007 tabulated 1D P-rho lookups
│   ├── paleos.py         # Unified PALEOS density + nabla_ad
│   ├── tdep.py           # T-dependent EOS, melting curves, phase routing
│   ├── dispatch.py       # calculate_density/batch entry points
│   ├── temperature.py    # Adiabat computation, T profiles
│   └── output.py         # Profile file writing
├── eos_analytic.py       # Analytic modified polytrope
├── eos_properties.py     # Lazy EOS_REGISTRY
├── eos_export.py         # EOS table generation for SPIDER/Aragog
├── mixing.py             # Multi-material mixing, LayerMixture
├── melting_curves.py     # Solidus/liquidus curves
├── binodal.py            # H2 miscibility models
└── constants.py          # Physical constants
```

## API reference

### Core
- [`zalmoxis.config`](zalmoxis.config.md) - Configuration loading, parsing, validation
- [`zalmoxis.solver`](zalmoxis.solver.md) - Solver loop (`main()`)
- [`zalmoxis.output`](zalmoxis.output.md) - Post-processing and file output
- [`zalmoxis.structure_model`](zalmoxis.structure_model.md) - ODE system

### EOS
- [`zalmoxis.eos`](zalmoxis.eos.md) - EOS package (dispatch, interpolation, PALEOS, Seager, T-dep, temperature)
- [`zalmoxis.eos_analytic`](zalmoxis.eos_analytic.md) - Analytic modified polytrope
- [`zalmoxis.eos_properties`](zalmoxis.eos_properties.md) - EOS registry
- [`zalmoxis.melting_curves`](zalmoxis.melting_curves.md) - Melting curves

### Mixing
- [`zalmoxis.mixing`](zalmoxis.mixing.md) - Multi-material mixing
- [`zalmoxis.binodal`](zalmoxis.binodal.md) - H2 miscibility models

### Constants
- [`zalmoxis.constants`](zalmoxis.constants.md) - Physical constants
