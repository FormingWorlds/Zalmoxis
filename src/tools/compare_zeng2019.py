"""Compare Zalmoxis EOS variants against Zeng et al. (2019) mass-radius data.

Runs Zalmoxis for 4 compositions (pure Fe, pure MgSiO3, Earth-like, pure H2O)
at several masses, using tabulated Seager2007, analytic Seager2007, and
(where applicable) temperature-dependent WolfBower2018 EOS.
Loads Zeng+2019 reference curves and plots the comparison on lin-lin axes.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from zalmoxis import zalmoxis
from zalmoxis.constants import earth_mass, earth_radius
from zalmoxis.zalmoxis import load_solidus_liquidus_functions

ZALMOXIS_ROOT = os.environ.get('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    sys.exit('ZALMOXIS_ROOT environment variable not set')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MASSES = [0.5, 1, 2, 3, 5, 7, 10]  # Earth masses
WOLFBOWER_MASSES = [0.5, 1, 2, 3, 4, 5, 7]  # WolfBower2018 limited to <= 7 M_earth

# Each composition defines EOS overrides for tabulated, analytic, and
# (optionally) WolfBower2018 variants.
COMPOSITIONS = {
    'Pure Fe': {
        'config_type': 'rocky',
        'cmf': 0.5,
        'immf': 0,
        'tabulated_eos': {'core': 'Seager2007:iron', 'mantle': 'Seager2007:iron'},
        'analytic_eos': {'core': 'Analytic:iron', 'mantle': 'Analytic:iron'},
        'wolfbower_eos': None,  # no MgSiO3 layer
        'zeng_file': 'massradiusFe.txt',
        'color': '#d62728',
    },
    'Pure MgSiO3': {
        'config_type': 'rocky',
        'cmf': 0.5,
        'immf': 0,
        'tabulated_eos': {
            'core': 'Seager2007:MgSiO3',
            'mantle': 'Seager2007:MgSiO3',
        },
        'analytic_eos': {'core': 'Analytic:MgSiO3', 'mantle': 'Analytic:MgSiO3'},
        'wolfbower_eos': {
            'core': 'WolfBower2018:MgSiO3',
            'mantle': 'WolfBower2018:MgSiO3',
        },
        'zeng_file': 'massradiusmgsio3.txt',
        'color': '#ff7f0e',
    },
    'Earth-like': {
        'config_type': 'rocky',
        'cmf': 0.325,
        'immf': 0,
        'tabulated_eos': {
            'core': 'Seager2007:iron',
            'mantle': 'Seager2007:MgSiO3',
        },
        'analytic_eos': {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'},
        'wolfbower_eos': {
            'core': 'Seager2007:iron',
            'mantle': 'WolfBower2018:MgSiO3',
        },
        'zeng_file': 'massradiusEarthlikeRocky.txt',
        'color': '#2ca02c',
    },
    'Pure H2O': {
        'config_type': 'rocky',
        'cmf': 0.5,
        'immf': 0,
        'tabulated_eos': {'core': 'Seager2007:H2O', 'mantle': 'Seager2007:H2O'},
        'analytic_eos': {'core': 'Analytic:H2O', 'mantle': 'Analytic:H2O'},
        'wolfbower_eos': None,  # no MgSiO3 layer
        'zeng_file': 'massradius_100percentH2O_300K_1mbar.txt',
        'color': '#1f77b4',
    },
}

ZENG_DIR = os.path.join(
    ZALMOXIS_ROOT, 'data', 'mass_radius_comparison_data', 'Zeng+2019_M-R_models'
)
OUTPUT_DIR = os.path.join(ZALMOXIS_ROOT, 'output_files', 'zeng_comparison')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Solver helper
# ---------------------------------------------------------------------------


def run_solver(id_mass, config_type, cmf, immf, layer_eos_override=None):
    """Run Zalmoxis and return (mass_earth, radius_earth)."""
    default_config_path = os.path.join(ZALMOXIS_ROOT, 'input', 'default.toml')
    config_params = zalmoxis.load_zalmoxis_config(default_config_path)
    config_params['planet_mass'] = id_mass * earth_mass
    config_params['core_mass_fraction'] = cmf
    config_params['mantle_mass_fraction'] = immf

    if config_type == 'rocky':
        config_params['layer_eos_config'] = {
            'core': 'Seager2007:iron',
            'mantle': 'Seager2007:MgSiO3',
        }
    elif config_type == 'water':
        config_params['layer_eos_config'] = {
            'core': 'Seager2007:iron',
            'mantle': 'Seager2007:MgSiO3',
            'ice_layer': 'Seager2007:H2O',
        }

    if layer_eos_override:
        config_params['layer_eos_config'] = layer_eos_override

    config_params['verbose'] = False

    layer_eos_config = config_params['layer_eos_config']
    model_results = zalmoxis.main(
        config_params,
        material_dictionaries=zalmoxis.load_material_dictionaries(),
        melting_curves_functions=load_solidus_liquidus_functions(layer_eos_config),
        input_dir=os.path.join(ZALMOXIS_ROOT, 'input'),
    )

    converged = model_results.get('converged', False)
    if not converged:
        print('  WARNING: solver did NOT converge')

    mass_kg = model_results['mass_enclosed'][-1]
    radius_m = model_results['radii'][-1]
    return mass_kg / earth_mass, radius_m / earth_radius


# ---------------------------------------------------------------------------
# Reference data helpers
# ---------------------------------------------------------------------------


def load_zeng_data(filename):
    """Load a Zeng+2019 mass-radius curve (tab-separated, M_earth vs R_earth)."""
    path = os.path.join(ZENG_DIR, filename)
    masses, radii = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m, r = map(float, line.split())
            masses.append(m)
            radii.append(r)
    return np.array(masses), np.array(radii)


def interpolate_zeng(zeng_masses, zeng_radii, target_mass):
    """Log-log interpolation of Zeng curve at a target mass."""
    log_m = np.log10(zeng_masses)
    log_r = np.log10(zeng_radii)
    return 10 ** np.interp(np.log10(target_mass), log_m, log_r)


# ---------------------------------------------------------------------------
# Run comparisons — tabulated, analytic, and WolfBower2018
# ---------------------------------------------------------------------------

print('=' * 70)
print('Zalmoxis vs. Zeng+2019 mass-radius comparison')
print('=' * 70)

results_tab = {}
results_ana = {}
results_wb = {}

for comp_name, cfg in COMPOSITIONS.items():
    zeng_m, zeng_r = load_zeng_data(cfg['zeng_file'])
    results_tab[comp_name] = {'masses': [], 'radii': [], 'zeng_radii': []}
    results_ana[comp_name] = {'masses': [], 'radii': [], 'zeng_radii': []}
    results_wb[comp_name] = {'masses': [], 'radii': [], 'zeng_radii': []}

    for mass in MASSES:
        r_zeng = interpolate_zeng(zeng_m, zeng_r, mass)

        # Tabulated EOS
        print(f'\n[Tabulated]    {comp_name} at {mass} M_earth ...')
        _, r_tab = run_solver(
            id_mass=mass,
            config_type=cfg['config_type'],
            cmf=cfg['cmf'],
            immf=cfg['immf'],
            layer_eos_override=cfg['tabulated_eos'],
        )
        diff_tab = (r_tab - r_zeng) / r_zeng * 100
        print(f'  R = {r_tab:.4f}, Zeng = {r_zeng:.4f}, diff = {diff_tab:+.2f}%')
        results_tab[comp_name]['masses'].append(mass)
        results_tab[comp_name]['radii'].append(r_tab)
        results_tab[comp_name]['zeng_radii'].append(r_zeng)

        # Analytic EOS
        print(f'[Analytic]     {comp_name} at {mass} M_earth ...')
        _, r_ana = run_solver(
            id_mass=mass,
            config_type=cfg['config_type'],
            cmf=cfg['cmf'],
            immf=cfg['immf'],
            layer_eos_override=cfg['analytic_eos'],
        )
        diff_ana = (r_ana - r_zeng) / r_zeng * 100
        print(f'  R = {r_ana:.4f}, Zeng = {r_zeng:.4f}, diff = {diff_ana:+.2f}%')
        results_ana[comp_name]['masses'].append(mass)
        results_ana[comp_name]['radii'].append(r_ana)
        results_ana[comp_name]['zeng_radii'].append(r_zeng)

    # WolfBower2018 EOS (only for compositions with MgSiO3, mass <= 7 Me)
    if cfg['wolfbower_eos'] is not None:
        for mass in WOLFBOWER_MASSES:
            r_zeng = interpolate_zeng(zeng_m, zeng_r, mass)
            print(f'[WolfBower18]  {comp_name} at {mass} M_earth ...')
            _, r_wb = run_solver(
                id_mass=mass,
                config_type=cfg['config_type'],
                cmf=cfg['cmf'],
                immf=cfg['immf'],
                layer_eos_override=cfg['wolfbower_eos'],
            )
            diff_wb = (r_wb - r_zeng) / r_zeng * 100
            print(f'  R = {r_wb:.4f}, Zeng = {r_zeng:.4f}, diff = {diff_wb:+.2f}%')
            results_wb[comp_name]['masses'].append(mass)
            results_wb[comp_name]['radii'].append(r_wb)
            results_wb[comp_name]['zeng_radii'].append(r_zeng)

# ---------------------------------------------------------------------------
# Save numerical results
# ---------------------------------------------------------------------------

results_file = os.path.join(OUTPUT_DIR, 'zeng_comparison_results.txt')
with open(results_file, 'w') as f:
    f.write(
        f'{"Composition":<16} {"Mass(Me)":>8} {"Tabulated(Re)":>14} '
        f'{"Analytic(Re)":>13} {"WB2018(Re)":>11} {"Zeng(Re)":>10} '
        f'{"Tab(%)":>8} {"Ana(%)":>8} {"WB(%)":>8}\n'
    )
    f.write('-' * 105 + '\n')
    for comp_name in COMPOSITIONS:
        tab = results_tab[comp_name]
        ana = results_ana[comp_name]
        wb = results_wb[comp_name]
        for i in range(len(MASSES)):
            d_tab = (tab['radii'][i] - tab['zeng_radii'][i]) / tab['zeng_radii'][i] * 100
            d_ana = (ana['radii'][i] - ana['zeng_radii'][i]) / ana['zeng_radii'][i] * 100
            # WolfBower2018 only for some compositions and masses
            wb_idx = None
            if MASSES[i] in wb['masses']:
                wb_idx = wb['masses'].index(MASSES[i])
            if wb_idx is not None:
                d_wb = (
                    (wb['radii'][wb_idx] - wb['zeng_radii'][wb_idx])
                    / wb['zeng_radii'][wb_idx]
                    * 100
                )
                f.write(
                    f'{comp_name:<16} {MASSES[i]:8.1f} {tab["radii"][i]:14.4f} '
                    f'{ana["radii"][i]:13.4f} {wb["radii"][wb_idx]:11.4f} '
                    f'{tab["zeng_radii"][i]:10.4f} '
                    f'{d_tab:+8.2f} {d_ana:+8.2f} {d_wb:+8.2f}\n'
                )
            else:
                f.write(
                    f'{comp_name:<16} {MASSES[i]:8.1f} {tab["radii"][i]:14.4f} '
                    f'{ana["radii"][i]:13.4f} {"---":>11} '
                    f'{tab["zeng_radii"][i]:10.4f} '
                    f'{d_tab:+8.2f} {d_ana:+8.2f} {"---":>8}\n'
                )

print(f'\nResults saved to {results_file}')

# ---------------------------------------------------------------------------
# Plot (lin-lin, lines between points, axes clipped to data range)
# ---------------------------------------------------------------------------

mass_min = min(MASSES)
mass_max = max(MASSES)
margin = 0.05

fig, ax = plt.subplots(figsize=(9, 6.5))

for comp_name, cfg in COMPOSITIONS.items():
    color = cfg['color']
    zeng_m, zeng_r = load_zeng_data(cfg['zeng_file'])
    mask = (zeng_m >= mass_min) & (zeng_m <= mass_max)

    # Zeng reference (solid line, semi-transparent)
    ax.plot(
        zeng_m[mask],
        zeng_r[mask],
        color=color,
        linewidth=2,
        linestyle='-',
        alpha=0.5,
        label=f'{comp_name} (Zeng+2019)',
    )

    tab = results_tab[comp_name]
    ana = results_ana[comp_name]
    wb = results_wb[comp_name]

    # Tabulated EOS (dashed)
    ax.plot(
        tab['masses'],
        tab['radii'],
        color=color,
        linewidth=1.5,
        linestyle='--',
        marker='o',
        markersize=5,
        markeredgecolor='black',
        markeredgewidth=0.5,
        zorder=5,
        label=f'{comp_name} (Tabulated)',
    )

    # Analytic EOS (dotted)
    ax.plot(
        ana['masses'],
        ana['radii'],
        color=color,
        linewidth=1.5,
        linestyle=':',
        marker='s',
        markersize=4,
        markeredgecolor='black',
        markeredgewidth=0.5,
        zorder=4,
        label=f'{comp_name} (Analytic)',
    )

    # WolfBower2018 EOS (dash-dot)
    if wb['masses']:
        ax.plot(
            wb['masses'],
            wb['radii'],
            color=color,
            linewidth=1.5,
            linestyle='-.',
            marker='D',
            markersize=4,
            markeredgecolor='black',
            markeredgewidth=0.5,
            zorder=6,
            label=f'{comp_name} (WB2018, T-dep)',
        )

ax.set_xlabel(r'Mass ($M_\oplus$)', fontsize=13)
ax.set_ylabel(r'Radius ($R_\oplus$)', fontsize=13)
ax.set_title('Zalmoxis vs. Zeng+2019', fontsize=14)

# Collect all plotted radii for axis limits
all_radii = []
for comp_name in COMPOSITIONS:
    all_radii.extend(results_tab[comp_name]['radii'])
    all_radii.extend(results_ana[comp_name]['radii'])
    all_radii.extend(results_tab[comp_name]['zeng_radii'])
    all_radii.extend(results_wb[comp_name]['radii'])
r_min, r_max = min(all_radii), max(all_radii)

x_pad = margin * (mass_max - mass_min)
r_pad = margin * (r_max - r_min)
ax.set_xlim(mass_min - x_pad, mass_max + x_pad)
ax.set_ylim(r_min - r_pad, r_max + r_pad)

ax.legend(fontsize=6.5, ncol=3, loc='upper left')
ax.grid(True, alpha=0.3)
ax.tick_params(labelsize=11)

plot_file = os.path.join(OUTPUT_DIR, 'zeng_comparison.pdf')
fig.savefig(plot_file, bbox_inches='tight', dpi=150)
print(f'Plot saved to {plot_file}')
plt.close(fig)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print('\n' + '=' * 70)
print('Summary of relative differences vs. Zeng+2019:')
print(f'{"Composition":<16} {"Mass":>5}  {"Tabulated":>10}  {"Analytic":>10}  {"WB2018":>10}')
print('-' * 60)
for comp_name in COMPOSITIONS:
    tab = results_tab[comp_name]
    ana = results_ana[comp_name]
    wb = results_wb[comp_name]
    for i in range(len(MASSES)):
        d_tab = (tab['radii'][i] - tab['zeng_radii'][i]) / tab['zeng_radii'][i] * 100
        d_ana = (ana['radii'][i] - ana['zeng_radii'][i]) / ana['zeng_radii'][i] * 100
        wb_str = '---'
        if MASSES[i] in wb['masses']:
            j = wb['masses'].index(MASSES[i])
            d_wb = (wb['radii'][j] - wb['zeng_radii'][j]) / wb['zeng_radii'][j] * 100
            wb_str = f'{d_wb:+9.2f}%'
        print(
            f'  {comp_name:<16} {MASSES[i]:4.1f}  {d_tab:+9.2f}%  {d_ana:+9.2f}%  {wb_str:>10}'
        )
