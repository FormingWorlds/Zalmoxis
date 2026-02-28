"""Validation plots for RTPress100TPa:MgSiO3 EOS.

Runs three EOS methods (RTPress100TPa, WolfBower2018, Seager2007) across
overlapping mass ranges and produces:
  1. Mass-radius comparison (all three EOS)
  2. Density profile comparison at 1 and 5 M_earth
  3. Pressure profile comparison at 1 and 5 M_earth
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure imports work from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.zalmoxis.zalmoxis import (
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
    main,
)
from zalmoxis.constants import earth_mass, earth_radius

ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')

OUTPUT_DIR = os.path.join(ZALMOXIS_ROOT, 'output_files')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_model(mass_earth, eos_config):
    """Run Zalmoxis for a given mass and EOS config, return results dict or None."""
    config = load_zalmoxis_config(os.path.join(ZALMOXIS_ROOT, 'input', 'default.toml'))
    config['planet_mass'] = mass_earth * earth_mass
    config['layer_eos_config'] = eos_config

    mat = load_material_dictionaries()
    melt = load_solidus_liquidus_functions(eos_config)
    input_dir = os.path.join(ZALMOXIS_ROOT, 'input')

    results = main(config, mat, melt, input_dir)
    if not results['converged']:
        print(f'  WARNING: did not converge for {mass_earth} M_earth with {eos_config}')
        return None
    return results


def collect_mass_radius(masses, eos_config, label):
    """Run models across a mass range and collect (mass, radius) pairs."""
    radii = []
    converged_masses = []
    all_results = {}
    for m in masses:
        print(f'  {label}: {m} M_earth ...', end=' ', flush=True)
        res = run_model(m, eos_config)
        if res is not None:
            r = res['radii'][-1] / earth_radius
            converged_masses.append(m)
            radii.append(r)
            all_results[m] = res
            print(f'R = {r:.4f} R_earth')
        else:
            print('FAILED')
    return np.array(converged_masses), np.array(radii), all_results


# --- EOS configurations ---
EOS_RTPRESS = {'core': 'Seager2007:iron', 'mantle': 'RTPress100TPa:MgSiO3'}
EOS_WB2018 = {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'}
EOS_SEAGER = {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'}

# --- Mass ranges ---
masses_all = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0]
masses_seager_extended = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

print('=' * 60)
print('Collecting RTPress100TPa mass-radius data')
print('=' * 60)
m_rt, r_rt, res_rt = collect_mass_radius(masses_all, EOS_RTPRESS, 'RTPress100TPa')

print()
print('=' * 60)
print('Collecting WolfBower2018 mass-radius data')
print('=' * 60)
m_wb, r_wb, res_wb = collect_mass_radius(masses_all, EOS_WB2018, 'WolfBower2018')

print()
print('=' * 60)
print('Collecting Seager2007 mass-radius data')
print('=' * 60)
m_sg, r_sg, res_sg = collect_mass_radius(masses_seager_extended, EOS_SEAGER, 'Seager2007')


# =====================================================================
# Plot 1: Mass-Radius comparison
# =====================================================================
fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(m_rt, r_rt, 'o-', color='C0', label='RTPress100TPa:MgSiO3', markersize=6, lw=1.5)
ax.plot(m_wb, r_wb, 's--', color='C1', label='WolfBower2018:MgSiO3', markersize=6, lw=1.5)
ax.plot(m_sg, r_sg, '^:', color='C2', label='Seager2007:MgSiO3 (300 K)', markersize=6, lw=1.5)

ax.set_xlabel(r'Planet mass [$M_\oplus$]', fontsize=12)
ax.set_ylabel(r'Planet radius [$R_\oplus$]', fontsize=12)
ax.set_title(
    'Mass–radius comparison: three EOS methods\n(core: Seager2007 iron, CMF = 0.325)',
    fontsize=11,
)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 11)

fname1 = os.path.join(OUTPUT_DIR, 'validation_RTPress100TPa_mass_radius.pdf')
fig.savefig(fname1, bbox_inches='tight', dpi=300)
fig.savefig(fname1.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
print(f'\nSaved: {fname1} (+png)')
plt.close(fig)


# =====================================================================
# Plot 2 & 3: Density and Pressure profiles at selected masses
# =====================================================================
profile_masses = [1.0, 5.0]

for pm in profile_masses:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    datasets = []
    if pm in res_rt:
        datasets.append(('RTPress100TPa', res_rt[pm], 'C0'))
    if pm in res_wb:
        datasets.append(('WolfBower2018', res_wb[pm], 'C1'))
    if pm in res_sg:
        datasets.append(('Seager2007 (300 K)', res_sg[pm], 'C2'))

    # Panel (a): Density
    ax = axes[0]
    for label, res, color in datasets:
        r_norm = res['radii'] / res['radii'][-1]
        ax.plot(r_norm, res['density'] / 1e3, color=color, label=label, lw=1.5)
    ax.set_xlabel(r'Normalised radius $r/R_p$', fontsize=11)
    ax.set_ylabel(r'Density [10$^3$ kg m$^{-3}$]', fontsize=11)
    ax.set_title(f'(a) Density — {pm:.0f} $M_\\oplus$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (b): Pressure
    ax = axes[1]
    for label, res, color in datasets:
        r_norm = res['radii'] / res['radii'][-1]
        ax.plot(r_norm, res['pressure'] / 1e9, color=color, label=label, lw=1.5)
    ax.set_xlabel(r'Normalised radius $r/R_p$', fontsize=11)
    ax.set_ylabel('Pressure [GPa]', fontsize=11)
    ax.set_title(f'(b) Pressure — {pm:.0f} $M_\\oplus$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel (c): Temperature
    ax = axes[2]
    for label, res, color in datasets:
        r_norm = res['radii'] / res['radii'][-1]
        ax.plot(r_norm, res['temperature'], color=color, label=label, lw=1.5)
    ax.set_xlabel(r'Normalised radius $r/R_p$', fontsize=11)
    ax.set_ylabel('Temperature [K]', fontsize=11)
    ax.set_title(f'(c) Temperature — {pm:.0f} $M_\\oplus$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f'Radial profiles at {pm:.0f} $M_\\oplus$ (CMF = 0.325, isothermal T = 3500 K)',
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()

    fname = os.path.join(OUTPUT_DIR, f'validation_RTPress100TPa_profiles_{pm:.0f}Mearth.pdf')
    fig.savefig(fname, bbox_inches='tight', dpi=300)
    fig.savefig(fname.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f'Saved: {fname} (+png)')
    plt.close(fig)


# =====================================================================
# Plot 4: Radius difference RTPress100TPa vs WolfBower2018
# =====================================================================
common_masses = sorted(set(m_rt) & set(m_wb))
if len(common_masses) > 1:
    r_rt_common = np.array([r_rt[list(m_rt).index(m)] for m in common_masses])
    r_wb_common = np.array([r_wb[list(m_wb).index(m)] for m in common_masses])
    pct_diff = (r_rt_common - r_wb_common) / r_wb_common * 100

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(common_masses, pct_diff, width=0.3, color='C3', alpha=0.8)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel(r'Planet mass [$M_\oplus$]', fontsize=12)
    ax.set_ylabel('Radius difference [%]\n(RTPress100TPa − WB2018) / WB2018', fontsize=10)
    ax.set_title('RTPress100TPa vs WolfBower2018 radius difference', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    fname4 = os.path.join(OUTPUT_DIR, 'validation_RTPress100TPa_vs_WB2018_diff.pdf')
    fig.savefig(fname4, bbox_inches='tight', dpi=300)
    fig.savefig(fname4.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
    print(f'Saved: {fname4} (+png)')
    plt.close(fig)

print('\nDone.')
