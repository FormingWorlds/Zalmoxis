# Zalmoxis Validation Grid Report

## Summary

- Total runs: (pending)
- Converged: (pending)
- Failed: (pending)

## Suite Results

| Suite | Total | Converged | Failed | Error | Rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| (run analyze_results.py to populate) | | | | | |

## Plots

All plots are in `tools/validation_grid/plots/`.

### Dashboard
- `convergence_heatmap.png`: Convergence status for every run
- `runtime_histogram.png`: Distribution of computation times

### Mass-Radius Relations
- `mass_radius_baseline.png`: Adiabatic vs isothermal baseline (Suite 1)
- `mass_radius_mixing.png`: H2O mixing fraction effect (Suite 2)
- `mass_radius_temperature.png`: Surface temperature effect (Suite 3)
- `mass_radius_mushy.png`: Mushy zone factor effect (Suite 4)
- `mass_radius_3layer.png`: Three-layer models (Suite 5)
- `mass_radius_stress.png`: High-mass stress test (Suite 6)
- `mass_radius_exotic.png`: Exotic architectures (Suite 7)
- `mass_radius_legacy.png`: EOS comparison (Suite 9)

### Sensitivity
- `sensitivity_h2o_fraction.png`: Radius vs H2O fraction
- `sensitivity_temperature.png`: Radius vs surface temperature
- `sensitivity_mushy_zone.png`: Radius vs mushy zone factor

### Profile Comparisons
- `profiles_1Me_compositions.png`: 1 M_earth, varying H2O content
- `profiles_mass_sweep.png`: Adiabatic profiles at different masses
- `profiles_temperature_modes.png`: Temperature mode comparison at 1 M_earth

### Edge Cases
- `exotic_results_table.png`: Table of exotic architecture results

## Recommended Parameter Ranges

(To be filled based on convergence analysis.)

## Bugs Found

(To be filled based on analysis of failed runs.)
