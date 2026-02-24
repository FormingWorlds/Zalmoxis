# API overview

This is a detailed overview of Zalmoxis' API for the user's reference. If you want to understand the underlying model, please visit the [model overview](../../Explanations/model.md). 

## Project structure

```
src
    ├── get_zalmoxis.sh
    ├── __init__.py
    ├── setup_zalmoxis.py
    ├── tests
    │   ├── __init__.py
    │   ├── test_convergence.py
    │   ├── test_convergence_TdepEOS.py
    │   ├── test_MR.py
    │   └── test_Seager.py
    ├── tools
    │   ├── run_parallel.py
    │   ├── run_ternary.py
    │   ├── setup_tests.py
    │   └── setup_utils.py
    └── zalmoxis
        ├── constants.py
        ├── eos_functions.py
        ├── eos_properties.py
        ├── __init__.py
        ├── __main__.py
        ├── plots
        │   ├── plot_animated_pressure_density_profiles.py
        │   ├── plot_eos.py
        │   ├── plot_melting_curves.py
        │   ├── plot_MR.py
        │   ├── plot_phase_vs_radius.py
        │   ├── plot_profiles_all_in_one.py
        │   ├── plot_profiles.py
        │   └── plot_ternary.py
        ├── structure_model.py
        └── zalmoxis.py
```

## API reference

### Core
- [`zalmoxis.zalmoxis`](zalmoxis.zalmoxis.md)
- [`zalmoxis.structure_model`](zalmoxis.structure_model.md)

### EOS
- [`zalmoxis.eos_functions`](zalmoxis.eos_functions.md)
- [`zalmoxis.eos_properties`](zalmoxis.eos_properties.md)

### Constants
- [`zalmoxis.constants`](zalmoxis.constants.md)