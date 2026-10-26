# Thermal state

## What this page checks

This page shows how the initial thermal state of a planet depends on its mass and on the material properties. The initial thermal state is the temperature at the core-mantle boundary (CMB) and at the surface right after accretion, which `zalmoxis.energetics.initial_thermal_state` computes. Validation means that a test runs the code on a case with a known right answer and checks the result within a stated tolerance. The figures on this page come from scripts, not from tests: no test asserts them and CI does not run the scripts. Each case below says why the comparison exists, what it shows, what it compares with and why it matters, and ends with a box "Test details". The [overview](index.md) lists every case of the tab.

## Mass dependence with the Seager et al. (2007) EOS

**Why this comparison exists.** The initial temperatures depend on the planet mass through the gravitational energy released in accretion; the figures show that dependence.

**What it shows.** `tools/validation/validate_thermal_state.py` runs Zalmoxis with the [Seager et al. (2007)](https://iopscience.iop.org/article/10.1086/521346) analytic EOS for 5 planet masses (0.5, 1, 2, 3, 5 $M_\oplus$) and computes the initial thermal state for each.

![Initial thermal state against planet mass](../img/validation/validate_thermal_state_mass.png)

Plot 1: (a) $T_\mathrm{CMB}$ against planet mass, with the Boujibar et al. (2020) polynomial overlay; (b) $T_\mathrm{surface}$ against planet mass.

![Energy terms against planet mass](../img/validation/validate_thermal_state_energy.png)

Plot 2: $U_d$, $U_u$ and $\Delta U = U_d - U_u$ against planet mass.

**How we know the right answer.** Plot 1 overlays the polynomial of Boujibar et al. (2020) on the CMB temperature.

**Why it matters.** The initial thermal state sets the start temperatures of the interior.

**Known limit.** No test asserts these figures, and the script does not run with the current `initial_thermal_state`.

??? info "Test details"
    - References: Seager et al. (2007); Boujibar et al. (2020) and White & Li (2025), reference named in the script docstring.
    - Tolerance: none (no test).
    - Test id: none; status: script figure, no test, not run in CI.
    - Command: `python tools/validation/validate_thermal_state.py`. The script does not run with the current `initial_thermal_state`, which returns `T_surf_accr` and no `T_surface` key.

## Constant and PALEOS material properties

**Why this comparison exists.** Constant material properties and PALEOS properties give different initial temperatures; the comparison shows how large that difference is.

**What it shows.** `tools/validation/validate_thermal_state_paleos.py` compares the initial thermal state from the Seager et al. (2007) EOS with PALEOS material properties. It reproduces the mass-temperature trend from White & Li (2025, Fig. 3) using:

1. The Seager et al. (2007) structure with the default material properties of `initial_thermal_state`: heat capacities of 450 J/kg/K for iron and 1250 J/kg/K for silicate. The temperature follows the Gruneisen adiabat, an adiabat set by the Gruneisen parameter $\gamma$ = 1.3. The script docstring names other constants (C_Fe=840, C_sil=1200, nabla_ad=0.3), but the script does not pass them to `initial_thermal_state`.
2. The same structure with the adiabatic gradient `nabla_ad` from the PALEOS unified table and the default heat capacities. A step falls back to the Gruneisen step where the PALEOS change exceeds it by more than 50 %. The script defines a PALEOS heat-capacity function (`build_paleos_cp`) but does not call it.

The script also highlights the surface temperature issue: for 1 $M_\oplus$ planets, $T_\mathrm{surface}$ from the adiabatic integration can be well below the solidus, meaning the surface would accrete solid (not as a magma ocean).

![Initial thermal state, constant and PALEOS material properties](../img/validation/thermal_state_comparison.png)

![Energy budget, constant and PALEOS material properties](../img/validation/thermal_state_energy_comparison.png)

**How we know the right answer.** The script compares with the trend of White & Li (2025, Fig. 3) in style only, and overlays the Boujibar et al. (2020) polynomial; it plots no White & Li numbers.

**Why it matters.** It shows which material properties the initial thermal state needs.

**Known limit.** No test asserts these figures, and the script does not run with the current `initial_thermal_state`.

??? info "Test details"
    - References: White & Li (2025) and Boujibar et al. (2020), reference named in the script docstring.
    - Tolerance: none (no test).
    - Test id: none; status: script figure, no test, not run in CI.
    - Command: `python tools/validation/validate_thermal_state_paleos.py`. The script does not run with the current `initial_thermal_state`, which returns `T_surf_accr` and no `T_surface` key.

## Three tiers of material properties

**Why this comparison exists.** The structure and the material properties can each come from the simple or the PALEOS model; the comparison separates their effects.

**What it shows.** `tools/validation/validate_thermal_state_full_paleos.py` compares 3 tiers: the Seager et al. (2007) structure with constant properties, the same structure with the PALEOS adiabatic gradient, and the PALEOS structure with PALEOS properties at the CMB. It also sweeps the accretion heat retention efficiency $f_a$ from 0.01 to 0.40 for the full PALEOS case. It writes `thermal_state_three_tiers`, `thermal_state_fa_full_paleos` and `thermal_state_whiteli_comparison`; this page shows none of them.

**How we know the right answer.** The Boujibar et al. (2020) polynomial, and a run with the accretion efficiency of White & Li (2025), $f_a$ = 0.04 (with the differentiation efficiency $f_d$ = 0.50). The script also runs a case with the parameters of Boujibar et al. (2020): $f_a$ = $f_d$ = 0.40, with heat capacities of 840 and 1200 J/kg/K.

**Why it matters.** The script writes the effect of the accretion heat retention $f_a$ on the start temperatures to `thermal_state_fa_full_paleos`, which this page does not show.

**Known limit.** No test asserts these results, the page shows none of the figures, and the script does not run with the current `initial_thermal_state`.

??? info "Test details"
    - References: Boujibar et al. (2020) polynomial; White & Li (2025), accretion efficiency $f_a$ = 0.04.
    - Tolerance: none (no test).
    - Test id: none; status: script figure, no test, not run in CI.
    - Command: `python tools/validation/validate_thermal_state_full_paleos.py`. Like the 2 scripts above, it reads the `T_surface` key, which the current `initial_thermal_state` does not return.
