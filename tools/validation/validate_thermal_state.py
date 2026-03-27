"""Validation of the initial thermal state computation (energetics module).

Runs Zalmoxis with Seager2007 analytic EOS for 5 planet masses (0.5, 1, 2, 3, 5
M_Earth), computes the initial thermal state for each, and produces:

Plot 1 (validate_thermal_state_mass.pdf):
    (a) T_CMB vs planet mass, with Boujibar+2020 polynomial overlay
    (b) T_surface vs planet mass

Plot 2 (validate_thermal_state_energy.pdf):
    U_d, U_u, and Delta_U = U_d - U_u vs planet mass

References:
    Boujibar, A., Driscoll, P. & Fei, Y. (2020). JGRP, 125, e2019JE006124.
    White, N. I. & Li, J. (2025). JGRP, 130, e2024JE008550.
    Seager, S. et al. (2007). ApJ, 669, 1279.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure imports work from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from zalmoxis import get_zalmoxis_root
from zalmoxis.config import (
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
)
from zalmoxis.constants import earth_mass, earth_radius
from zalmoxis.energetics import initial_thermal_state
from zalmoxis.solver import main as zalmoxis_main

# ── Output directory ──────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(get_zalmoxis_root(), 'output', 'validation_thermal_state')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────
MASSES_EARTH = [0.5, 1.0, 2.0, 3.0, 5.0]
CMF = 0.32
EOS_CONFIG = {
    'core': 'Analytic:iron',
    'mantle': 'Analytic:MgSiO3',
}
F_ACCRETION = 0.04
F_DIFFERENTIATION = 0.50

# Colorblind-friendly palette (Tol bright)
COLORS = {
    'T_cmb': '#4477AA',
    'T_surface': '#EE6677',
    'U_d': '#228833',
    'U_u': '#CCBB44',
    'Delta_U': '#AA3377',
    'boujibar': '#BBBBBB',
}


def run_model(mass_earth):
    """Run Zalmoxis for a given mass and return (model_results, thermal_state).

    Parameters
    ----------
    mass_earth : float
        Planet mass in Earth masses.

    Returns
    -------
    tuple
        (model_results, thermal_state) dicts, or (None, None) on failure.
    """
    config_path = os.path.join(get_zalmoxis_root(), 'input', 'default.toml')
    config = load_zalmoxis_config(config_path)
    config['planet_mass'] = mass_earth * earth_mass
    config['core_mass_fraction'] = CMF
    config['layer_eos_config'] = EOS_CONFIG
    # Analytic EOS: isothermal mode, no melting curves needed
    config['temperature_mode'] = 'isothermal'
    config['surface_temperature'] = 3000
    config['data_output_enabled'] = False
    config['plotting_enabled'] = False
    config['verbose'] = False

    mat = load_material_dictionaries()
    melt = load_solidus_liquidus_functions(EOS_CONFIG)
    input_dir = os.path.join(get_zalmoxis_root(), 'input')

    results = zalmoxis_main(config, mat, melt, input_dir)
    if not results['converged']:
        print(f'  WARNING: did not converge for {mass_earth} M_earth')
        return None, None

    thermal = initial_thermal_state(
        results,
        core_mass_fraction=CMF,
        f_accretion=F_ACCRETION,
        f_differentiation=F_DIFFERENTIATION,
    )
    return results, thermal


def boujibar2020_tcmb(mass_earth):
    """Boujibar+2020 polynomial for T_CMB at CMF=0.32.

    From their Table 3: T_CMB = 24.6*(M/ME)^2 + 1695*(M/ME) + 6041

    Parameters
    ----------
    mass_earth : float or array-like
        Planet mass in Earth masses.

    Returns
    -------
    float or ndarray
        CMB temperature [K].
    """
    m = np.asarray(mass_earth, dtype=float)
    return 24.6 * m**2 + 1695.0 * m + 6041.0


# ── Run models ────────────────────────────────────────────────────────
print('=' * 60)
print('Validation: Initial thermal state vs planet mass')
print(f'EOS: Analytic Seager2007, CMF = {CMF}')
print(f'f_a = {F_ACCRETION}, f_d = {F_DIFFERENTIATION}')
print('=' * 60)

masses = []
T_cmb_arr = []
T_surf_arr = []
U_d_arr = []
U_u_arr = []
dU_arr = []
radii_arr = []

for m in MASSES_EARTH:
    print(f'  {m:.1f} M_earth ...', end=' ', flush=True)
    res, therm = run_model(m)
    if res is not None:
        masses.append(m)
        T_cmb_arr.append(therm['T_cmb'])
        T_surf_arr.append(therm['T_surface'])
        U_d_arr.append(therm['U_differentiated'])
        U_u_arr.append(therm['U_undifferentiated'])
        dU_arr.append(therm['U_differentiated'] - therm['U_undifferentiated'])
        radii_arr.append(res['radii'][-1] / earth_radius)
        print(
            f'R = {radii_arr[-1]:.3f} R_earth, '
            f'T_CMB = {therm["T_cmb"]:.0f} K, '
            f'T_surf = {therm["T_surface"]:.0f} K, '
            f'core: {therm["core_state"]}'
        )
    else:
        print('FAILED')

masses = np.array(masses)
T_cmb_arr = np.array(T_cmb_arr)
T_surf_arr = np.array(T_surf_arr)
U_d_arr = np.array(U_d_arr)
U_u_arr = np.array(U_u_arr)
dU_arr = np.array(dU_arr)

if len(masses) == 0:
    print('\nNo converged models. Aborting.')
    sys.exit(1)

# Save raw data
data_file = os.path.join(OUTPUT_DIR, 'thermal_state_data.txt')
header = (
    'mass_Mearth  T_cmb_K  T_surface_K  U_differentiated_J  U_undifferentiated_J  Delta_U_J'
)
np.savetxt(
    data_file,
    np.column_stack([masses, T_cmb_arr, T_surf_arr, U_d_arr, U_u_arr, dU_arr]),
    header=header,
    fmt='%.6e',
)
print(f'\nSaved raw data: {data_file}')


# ── Plot 1: Temperature vs mass ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel (a): T_CMB
ax = axes[0]
ax.plot(
    masses,
    T_cmb_arr,
    'o-',
    color=COLORS['T_cmb'],
    lw=1.5,
    markersize=7,
    label='Zalmoxis (Seager2007)',
)
# Boujibar+2020 overlay
m_fine = np.linspace(0.3, 5.5, 100)
ax.plot(
    m_fine,
    boujibar2020_tcmb(m_fine),
    '--',
    color=COLORS['boujibar'],
    lw=1.5,
    label='Boujibar+2020 (CMF=0.32)',
)
ax.set_xlabel(r'Planet mass [$M_\oplus$]', fontsize=12)
ax.set_ylabel(r'$T_\mathrm{CMB}$ [K]', fontsize=12)
ax.set_title('(a) CMB temperature', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Panel (b): T_surface
ax = axes[1]
ax.plot(masses, T_surf_arr, 's-', color=COLORS['T_surface'], lw=1.5, markersize=7)
ax.set_xlabel(r'Planet mass [$M_\oplus$]', fontsize=12)
ax.set_ylabel(r'$T_\mathrm{surface}$ [K]', fontsize=12)
ax.set_title('(b) Surface temperature', fontsize=12)
ax.grid(True, alpha=0.3)

fig.suptitle(
    f'Initial thermal state: $f_a$ = {F_ACCRETION}, $f_d$ = {F_DIFFERENTIATION}, CMF = {CMF}',
    fontsize=13,
    y=1.02,
)
fig.tight_layout()

fname1 = os.path.join(OUTPUT_DIR, 'validate_thermal_state_mass.pdf')
fig.savefig(fname1, bbox_inches='tight', dpi=300)
fig.savefig(fname1.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
print(f'Saved: {fname1} (+png)')
plt.close(fig)


# ── Plot 2: Binding energy vs mass ───────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

ax.semilogy(
    masses,
    U_d_arr,
    'o-',
    color=COLORS['U_d'],
    lw=1.5,
    markersize=7,
    label=r'$U_\mathrm{diff}$ (differentiated)',
)
ax.semilogy(
    masses,
    U_u_arr,
    's--',
    color=COLORS['U_u'],
    lw=1.5,
    markersize=7,
    label=r'$U_\mathrm{uni}$ (uniform)',
)
ax.semilogy(
    masses,
    dU_arr,
    '^:',
    color=COLORS['Delta_U'],
    lw=1.5,
    markersize=7,
    label=r'$\Delta U = U_\mathrm{diff} - U_\mathrm{uni}$',
)

# Overplot M^{5/3} scaling reference (anchored to 1 M_earth U_d)
if 1.0 in list(masses):
    idx_1 = list(masses).index(1.0)
    U_ref = U_d_arr[idx_1]
    ax.semilogy(
        m_fine,
        U_ref * m_fine ** (5.0 / 3.0),
        '-',
        color='gray',
        lw=1,
        alpha=0.5,
        label=r'$\propto M^{5/3}$ reference',
    )

ax.set_xlabel(r'Planet mass [$M_\oplus$]', fontsize=12)
ax.set_ylabel('Gravitational energy [J]', fontsize=12)
ax.set_title('Gravitational binding and differentiation energy', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, which='both')

fig.tight_layout()

fname2 = os.path.join(OUTPUT_DIR, 'validate_thermal_state_energy.pdf')
fig.savefig(fname2, bbox_inches='tight', dpi=300)
fig.savefig(fname2.replace('.pdf', '.png'), bbox_inches='tight', dpi=300)
print(f'Saved: {fname2} (+png)')
plt.close(fig)

print(f'\nAll output in: {OUTPUT_DIR}')
print('Done.')
