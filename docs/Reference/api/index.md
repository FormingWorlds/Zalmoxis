# API overview

This is a detailed overview of Zalmoxis' API for the user's reference. If you want to understand the underlying model, please visit the [model overview](../../Explanations/model.md). 

## Module overview

```
src/zalmoxis/
├── zalmoxis.py          # Orchestration: config loading, iteration loops, Brent solver, output
├── structure_model.py   # ODE system: coupled_odes(), solve_structure(), terminal events
├── eos_functions.py     # EOS dispatch: calculate_density(), Tdep phase logic, temperature profiles
├── eos_analytic.py      # Analytic modified polytrope: get_analytic_density()
├── eos_properties.py    # Material property dictionaries (file paths, unit conversions)
├── constants.py         # Physical constants (G, earth_mass, earth_radius, etc.)
└── plots/               # Visualization (profile plots, P-T phase diagrams)
```

## API reference

### Core
- [`zalmoxis.zalmoxis`](zalmoxis.zalmoxis.md)
- [`zalmoxis.structure_model`](zalmoxis.structure_model.md)

### EOS
- [`zalmoxis.eos_analytic`](zalmoxis.eos_analytic.md)
- [`zalmoxis.eos_functions`](zalmoxis.eos_functions.md)
- [`zalmoxis.eos_properties`](zalmoxis.eos_properties.md)

### Constants
- [`zalmoxis.constants`](zalmoxis.constants.md)