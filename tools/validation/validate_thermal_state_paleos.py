"""Validate initial thermal state: Seager+2007 vs PALEOS EOS comparison.

Reproduces the mass-temperature trend from White & Li (2025, Fig. 3) using:
1. Constant material properties (C_Fe=840, C_sil=1200, nabla_ad=0.3)
2. PALEOS-derived material properties (nabla_ad from unified table,
   C_p from table, pressure-dependent heat capacities)

Also highlights the surface temperature issue: for 1 M_Earth planets,
T_surface from the adiabatic integration can be well below the solidus,
meaning the surface would accrete solid (not as a magma ocean).

References:
    White, N. I. & Li, J. (2025). JGRP, 130, e2024JE008550.
    Boujibar, A., Driscoll, P. & Fei, Y. (2020). JGRP, 125, e2019JE006124.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from zalmoxis import get_zalmoxis_root
from zalmoxis.config import load_material_dictionaries, load_zalmoxis_config
from zalmoxis.constants import earth_mass, earth_radius
from zalmoxis.energetics import (
    gravitational_binding_energy,
    gravitational_binding_energy_uniform,
    initial_thermal_state,
)
from zalmoxis.melting_curves import iron_melting_anzellini13
from zalmoxis.solver import main as zalmoxis_main

# ── Output ───────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(get_zalmoxis_root(), 'output', 'validation_thermal_state_paleos')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Configuration ────────────────────────────────────────────────────
MASSES_EARTH = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
CMF = 0.32

# White & Li 2025 parameters
F_ACC = 0.04
F_DIFF = 0.50
T_EQ = 255.0

# Style
plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 15, 'axes.titlesize': 16,
    'legend.fontsize': 12, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'lines.linewidth': 2.5, 'lines.markersize': 8,
    'savefig.bbox': 'tight', 'savefig.dpi': 200,
})


def build_paleos_nabla_ad():
    """Build a nabla_ad(P, T) function from the PALEOS MgSiO3 unified table."""
    from zalmoxis.eos.interpolation import load_paleos_unified_table

    eos_file = os.path.join(
        get_zalmoxis_root(), 'data', 'EOS_PALEOS_MgSiO3_unified',
        'paleos_mgsio3_eos_table_pt.dat',
    )
    if not os.path.isfile(eos_file):
        print(f'PALEOS file not found: {eos_file}')
        return None

    cache = load_paleos_unified_table(eos_file)

    def nabla_ad_paleos(P_Pa, T_K):
        """Query PALEOS nabla_ad at (P, T)."""
        import math
        if P_Pa <= 0 or T_K <= 0:
            return 0.3
        log_p = math.log10(P_Pa)
        log_t = math.log10(T_K)
        # Clamp to table range
        log_p = max(cache['logp_min'], min(log_p, cache['logp_max']))
        log_t = max(cache['logt_min'], min(log_t, cache['logt_max']))
        try:
            val = float(cache['nabla_ad_interp']([[log_p, log_t]])[0])
            if np.isfinite(val) and val > 0:
                return val
        except Exception:
            pass
        return 0.3  # fallback

    return nabla_ad_paleos


def build_paleos_cp():
    """Build a cp(P, T) function from the PALEOS MgSiO3 unified table."""
    from zalmoxis.eos.interpolation import load_paleos_unified_table

    eos_file = os.path.join(
        get_zalmoxis_root(), 'data', 'EOS_PALEOS_MgSiO3_unified',
        'paleos_mgsio3_eos_table_pt.dat',
    )
    if not os.path.isfile(eos_file):
        return None

    # Load the full table to get cp
    data = np.genfromtxt(eos_file, usecols=range(9), comments='#')
    pressures = data[:, 0]
    temps = data[:, 1]
    cp_vals = data[:, 5]  # column 5 is cp

    valid = pressures > 0
    log_p = np.log10(pressures[valid])
    log_t = np.log10(temps[valid])
    cp = cp_vals[valid]

    from scipy.interpolate import LinearNDInterpolator
    interp = LinearNDInterpolator(list(zip(log_p, log_t)), cp)

    def cp_paleos(P_Pa, T_K):
        import math
        if P_Pa <= 0 or T_K <= 0:
            return 1200.0
        lp = math.log10(P_Pa)
        lt = math.log10(T_K)
        val = float(interp(lp, lt))
        if np.isfinite(val) and val > 0:
            return val
        return 1200.0

    return cp_paleos


def run_seager(mass_earth):
    """Run with Seager2007 analytic EOS + constant nabla_ad=0.3."""
    config_path = os.path.join(get_zalmoxis_root(), 'input', 'default.toml')
    config = load_zalmoxis_config(config_path)
    config['planet_mass'] = mass_earth * earth_mass
    config['core_mass_fraction'] = CMF
    config['layer_eos_config'] = {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'}
    config['temperature_mode'] = 'isothermal'
    config['surface_temperature'] = 3000
    config['data_output_enabled'] = False
    config['plotting_enabled'] = False
    config['verbose'] = False

    mat = load_material_dictionaries()
    input_dir = os.path.join(get_zalmoxis_root(), 'input')
    results = zalmoxis_main(config, mat, None, input_dir)
    if not results['converged']:
        return None, None

    thermal = initial_thermal_state(
        results, CMF, T_radiative_eq=T_EQ, f_accretion=F_ACC, f_differentiation=F_DIFF,
    )
    return results, thermal


def run_paleos(mass_earth, nabla_ad_func, cp_func=None):
    """Run with Seager2007 structure but PALEOS nabla_ad for adiabat."""
    config_path = os.path.join(get_zalmoxis_root(), 'input', 'default.toml')
    config = load_zalmoxis_config(config_path)
    config['planet_mass'] = mass_earth * earth_mass
    config['core_mass_fraction'] = CMF
    config['layer_eos_config'] = {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'}
    config['temperature_mode'] = 'isothermal'
    config['surface_temperature'] = 3000
    config['data_output_enabled'] = False
    config['plotting_enabled'] = False
    config['verbose'] = False

    mat = load_material_dictionaries()
    input_dir = os.path.join(get_zalmoxis_root(), 'input')
    results = zalmoxis_main(config, mat, None, input_dir)
    if not results['converged']:
        return None, None

    # Use PALEOS-derived nabla_ad for the adiabat integration
    thermal = initial_thermal_state(
        results, CMF, T_radiative_eq=T_EQ, f_accretion=F_ACC, f_differentiation=F_DIFF,
        nabla_ad_func=nabla_ad_func,
    )
    return results, thermal


def boujibar2020_tcmb(m):
    """Boujibar+2020 polynomial for T_CMB (Table 3, CMF=0.32)."""
    m = np.asarray(m, dtype=float)
    return 24.6 * m**2 + 1695.0 * m + 6041.0


def monteux2016_solidus(P_GPa):
    """Monteux+2016 silicate solidus (approximate, for reference)."""
    from zalmoxis.melting_curves import get_solidus_liquidus_functions
    sol, _ = get_solidus_liquidus_functions('Monteux16-solidus', 'Monteux16-liquidus-A-chondritic')
    return sol(P_GPa * 1e9)


def main():
    print('Building PALEOS nabla_ad interpolator...')
    nabla_ad_paleos = build_paleos_nabla_ad()
    if nabla_ad_paleos is None:
        print('PALEOS data not available, skipping PALEOS comparison')
        return

    print(f'Running {len(MASSES_EARTH)} masses with Seager2007 and PALEOS nabla_ad...\n')

    seager_data = []
    paleos_data = []

    for m in MASSES_EARTH:
        print(f'  {m} M_earth...')
        res_s, th_s = run_seager(m)
        res_p, th_p = run_paleos(m, nabla_ad_paleos)

        if th_s is not None:
            seager_data.append({
                'mass': m, 'R': res_s['radii'][-1] / earth_radius,
                **th_s,
            })
            print(f'    Seager: T_CMB={th_s["T_cmb"]:.0f} K, T_surf={th_s["T_surface"]:.0f} K, '
                  f'core={th_s["core_state"]}')

        if th_p is not None:
            paleos_data.append({
                'mass': m, 'R': res_p['radii'][-1] / earth_radius,
                **th_p,
            })
            print(f'    PALEOS: T_CMB={th_p["T_cmb"]:.0f} K, T_surf={th_p["T_surface"]:.0f} K, '
                  f'core={th_p["core_state"]}')

    # ── Plot 1: Reproduce White & Li 2025 Fig. 3 style ──────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    masses_s = [d['mass'] for d in seager_data]
    masses_p = [d['mass'] for d in paleos_data]
    T_cmb_s = [d['T_cmb'] for d in seager_data]
    T_cmb_p = [d['T_cmb'] for d in paleos_data]
    T_surf_s = [d['T_surface'] for d in seager_data]
    T_surf_p = [d['T_surface'] for d in paleos_data]

    # Panel (a): T_CMB and T_surface vs mass
    ax = axes[0]
    ax.plot(masses_s, T_cmb_s, 'o-', color='#4477AA', label=r'$T_\mathrm{CMB}$ (constant $\nabla_\mathrm{ad}=0.3$)')
    ax.plot(masses_p, T_cmb_p, 's--', color='#228833', label=r'$T_\mathrm{CMB}$ (PALEOS $\nabla_\mathrm{ad}$)')
    ax.plot(masses_s, T_surf_s, 'o-', color='#EE6677', label=r'$T_\mathrm{surf}$ (constant $\nabla_\mathrm{ad}=0.3$)')
    ax.plot(masses_p, T_surf_p, 's--', color='#AA3377', label=r'$T_\mathrm{surf}$ (PALEOS $\nabla_\mathrm{ad}$)')

    # Boujibar+2020 reference
    m_ref = np.linspace(0.5, 5, 50)
    ax.plot(m_ref, boujibar2020_tcmb(m_ref), ':', color='gray', linewidth=1.5,
            label='Boujibar+2020 $T_\\mathrm{CMB}$')

    # Iron melting at CMB pressure for reference
    P_cmb_s = [seager_data[i].get('P_cmb', None) for i in range(len(seager_data))]

    # Solidus reference line at 0 GPa (surface)
    ax.axhline(1400, color='orange', linestyle=':', linewidth=1, label='Approx. solidus ($T \\approx 1400$ K)')

    ax.set_xlabel(r'Planet mass [$M_\oplus$]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title('(a) Initial thermal state')
    ax.legend(fontsize=10, loc='upper left')
    ax.set_ylim(0, 14000)

    # Panel (b): Temperature difference (PALEOS - Seager)
    ax = axes[1]
    dT_cmb = [paleos_data[i]['T_cmb'] - seager_data[i]['T_cmb'] for i in range(len(seager_data))]
    dT_surf = [paleos_data[i]['T_surface'] - seager_data[i]['T_surface'] for i in range(len(seager_data))]
    ax.plot(masses_s, dT_cmb, 'o-', color='#4477AA', label=r'$\Delta T_\mathrm{CMB}$')
    ax.plot(masses_s, dT_surf, 's-', color='#EE6677', label=r'$\Delta T_\mathrm{surf}$')
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_xlabel(r'Planet mass [$M_\oplus$]')
    ax.set_ylabel(r'$T_\mathrm{PALEOS} - T_\mathrm{const}$ [K]')
    ax.set_title(r'(b) Effect of PALEOS $\nabla_\mathrm{ad}$')
    ax.legend()

    fig.suptitle(
        r'Initial thermal state: Seager+2007 structure, $f_c=0.32$, $f_a=0.04$, $f_d=0.50$',
    )
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'thermal_state_comparison.pdf')
    fig.savefig(fname)
    fig.savefig(fname.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'\nSaved: {fname}')

    # ── Plot 2: Energy budget comparison ─────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    U_d_s = [d['U_differentiated'] for d in seager_data]
    U_u_s = [d['U_undifferentiated'] for d in seager_data]
    dT_acc_s = [d['Delta_T_accretion'] for d in seager_data]
    dT_diff_s = [d['Delta_T_differentiation'] for d in seager_data]
    C_avg_s = [d['C_avg'] for d in seager_data]

    ax = axes[0]
    ax.semilogy(masses_s, U_d_s, 'o-', color='#228833', label='$U_d$ (differentiated)')
    ax.semilogy(masses_s, U_u_s, 's-', color='#CCBB44', label='$U_u$ (undifferentiated)')
    ax.semilogy(masses_s, [U_d_s[i] - U_u_s[i] for i in range(len(masses_s))],
                '^-', color='#AA3377', label=r'$\Delta E_D = U_d - U_u$')
    ax.set_xlabel(r'Planet mass [$M_\oplus$]')
    ax.set_ylabel('Energy [J]')
    ax.set_title('(a) Gravitational energy budget')
    ax.legend()

    ax = axes[1]
    ax.plot(masses_s, dT_acc_s, 'o-', color='#4477AA', label=r'$\Delta T_\mathrm{accretion}$')
    ax.plot(masses_s, dT_diff_s, 's-', color='#EE6677', label=r'$\Delta T_\mathrm{differentiation}$')
    ax.plot(masses_s, [dT_acc_s[i] + dT_diff_s[i] for i in range(len(masses_s))],
            '^-', color='#228833', label=r'$\Delta T_\mathrm{total}$')
    ax.set_xlabel(r'Planet mass [$M_\oplus$]')
    ax.set_ylabel(r'$\Delta T$ [K]')
    ax.set_title('(b) Temperature increments')
    ax.legend()

    fig.suptitle(
        f'Energy budget: C_avg = {C_avg_s[0]:.1f} J/kg/K (constant), '
        f'f_a = {F_ACC}, f_d = {F_DIFF}',
    )
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'thermal_state_energy_comparison.pdf')
    fig.savefig(fname)
    fig.savefig(fname.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'Saved: {fname}')

    # ── Save raw data ────────────────────────────────────────────────
    fname_data = os.path.join(OUTPUT_DIR, 'thermal_state_comparison_data.txt')
    with open(fname_data, 'w') as f:
        f.write('# M_earth  R_earth  T_CMB_seager  T_surf_seager  T_CMB_paleos  T_surf_paleos  '
                'U_d  U_u  Delta_E_D  C_avg  core_state_seager  core_state_paleos\n')
        for i in range(len(seager_data)):
            s = seager_data[i]
            p = paleos_data[i]
            f.write(f'{s["mass"]:.1f}  {s["R"]:.4f}  {s["T_cmb"]:.1f}  {s["T_surface"]:.1f}  '
                    f'{p["T_cmb"]:.1f}  {p["T_surface"]:.1f}  '
                    f'{s["U_differentiated"]:.4e}  {s["U_undifferentiated"]:.4e}  '
                    f'{s["U_differentiated"] - s["U_undifferentiated"]:.4e}  '
                    f'{s["C_avg"]:.1f}  {s["core_state"]}  {p["core_state"]}\n')
    print(f'Saved: {fname_data}')

    # ── Summary table ────────────────────────────────────────────────
    print('\n' + '=' * 90)
    print(f'{"M/ME":>5}  {"R/RE":>6}  {"T_CMB_s":>8}  {"T_surf_s":>9}  '
          f'{"T_CMB_p":>8}  {"T_surf_p":>9}  {"dT_surf":>8}  {"core_s":>7}  {"core_p":>7}')
    print('-' * 90)
    for i in range(len(seager_data)):
        s = seager_data[i]
        p = paleos_data[i]
        dt = p['T_surface'] - s['T_surface']
        print(f'{s["mass"]:5.1f}  {s["R"]:6.3f}  {s["T_cmb"]:8.0f}  {s["T_surface"]:9.0f}  '
              f'{p["T_cmb"]:8.0f}  {p["T_surface"]:9.0f}  {dt:8.0f}  '
              f'{s["core_state"]:>7}  {p["core_state"]:>7}')
    print('=' * 90)

    # Flag the 1 M_Earth issue
    for d in seager_data:
        if abs(d['mass'] - 1.0) < 0.01:
            print(f'\n** 1 M_Earth: T_surface = {d["T_surface"]:.0f} K (Seager)')
            if d['T_surface'] < 1400:
                print('   -> Below solidus (~1400 K): surface accretes SOLID, not as magma ocean')
            break

    print(f'\nAll output in: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
