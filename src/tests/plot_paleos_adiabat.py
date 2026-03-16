"""Plot T-P and rho-P profiles for PALEOS-2phase:MgSiO3 adiabatic mode at various masses.

Run from the Zalmoxis root:
    python src/tests/plot_paleos_adiabat.py

Produces three figures:
    output_files/paleos_adiabat_TP.png
    output_files/paleos_adiabat_rhoP.png
    output_files/paleos_adiabat_1Mearth_profiles.png

NOTE: The PALEOS liquid table has NaN gaps at low P / high T (unconverged
thermodynamic cells). Adiabatic mode works reliably for M <= 2 M_earth.
Higher masses are included in linear mode only for comparison.
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Suppress repetitive EOS warning spam
logging.getLogger('zalmoxis.eos_functions').setLevel(logging.CRITICAL)

# Ensure Zalmoxis is importable
ZALMOXIS_ROOT = os.environ.get(
    'ZALMOXIS_ROOT', os.path.dirname(os.path.abspath(__file__)) + '/../..'
)
sys.path.insert(0, os.path.join(ZALMOXIS_ROOT, 'src'))

from zalmoxis import zalmoxis as zal
from zalmoxis.constants import earth_mass, earth_radius
from zalmoxis.eos_functions import get_solidus_liquidus_functions
from zalmoxis.zalmoxis import load_material_dictionaries, load_solidus_liquidus_functions


# ── Configuration ──────────────────────────────────────────────────────
# Adiabatic mode works for <= 2 M_earth; above that the PALEOS liquid
# table has NaN holes at low P / high T that cause density failures.
MASSES_ADIABATIC = [0.5, 1.0, 2.0]
MASSES_LINEAR_ONLY = [5.0, 10.0]
ALL_MASSES = MASSES_ADIABATIC + MASSES_LINEAR_ONLY
CMF = 0.325
T_SURFACE = 3500.0
T_CENTER = 6000.0

OUTPUT_DIR = os.path.join(ZALMOXIS_ROOT, 'output_files')
os.makedirs(OUTPUT_DIR, exist_ok=True)

CMAP_MASS = plt.cm.viridis(np.linspace(0.15, 0.9, len(ALL_MASSES)))


def run_case(mass_earth, temperature_mode):
    """Run Zalmoxis for a single mass and temperature mode."""
    default_config_path = os.path.join(ZALMOXIS_ROOT, 'input', 'default.toml')
    config = zal.load_zalmoxis_config(default_config_path)

    config['planet_mass'] = mass_earth * earth_mass
    config['core_mass_fraction'] = CMF
    config['mantle_mass_fraction'] = 0
    config['temperature_mode'] = temperature_mode
    config['surface_temperature'] = T_SURFACE
    config['center_temperature'] = T_CENTER
    config['layer_eos_config'] = {
        'core': 'Seager2007:iron',
        'mantle': 'PALEOS-2phase:MgSiO3',
    }
    config['data_output_enabled'] = False
    config['plotting_enabled'] = False
    config['verbose'] = False

    layer_eos_config = config['layer_eos_config']
    results = zal.main(
        config,
        material_dictionaries=load_material_dictionaries(),
        melting_curves_functions=load_solidus_liquidus_functions(layer_eos_config),
        input_dir=os.path.join(ZALMOXIS_ROOT, 'input'),
    )
    return results


def main():
    print('PALEOS MgSiO3 adiabatic mode comparison\n')

    # ── Run all cases ──────────────────────────────────────────────────
    all_results = {}
    for mass in ALL_MASSES:
        modes = ['linear', 'adiabatic'] if mass in MASSES_ADIABATIC else ['linear']
        for mode in modes:
            label = f'{mass:.1f} M_E, {mode}'
            print(f'  Running {label} ...', end='', flush=True)
            res = run_case(mass, mode)
            conv = res.get('converged', False)
            R = res['radii'][-1] / earth_radius
            T_c = res['temperature'][0]
            print(f'  converged={conv}, R={R:.3f} R_E, T_center={T_c:.0f} K')
            all_results[(mass, mode)] = res

    # ── Load solidus/liquidus for reference ────────────────────────────
    solidus_func, liquidus_func = get_solidus_liquidus_functions()
    P_melt = np.logspace(9, 13, 200)  # 1 GPa to 10 TPa
    T_sol = np.array([solidus_func(p) for p in P_melt])
    T_liq = np.array([liquidus_func(p) for p in P_melt])
    valid_melt = np.isfinite(T_sol) & np.isfinite(T_liq)

    # ── Figure 1: T vs P ──────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(9, 6.5))

    ax1.fill_betweenx(
        P_melt[valid_melt] / 1e9,
        T_sol[valid_melt],
        T_liq[valid_melt],
        color='orange',
        alpha=0.15,
        label='Mushy zone',
    )
    ax1.plot(T_sol[valid_melt], P_melt[valid_melt] / 1e9, 'k--', lw=0.8, label='Solidus')
    ax1.plot(T_liq[valid_melt], P_melt[valid_melt] / 1e9, 'k:', lw=0.8, label='Liquidus')

    for i, mass in enumerate(ALL_MASSES):
        # Linear (always available)
        res_lin = all_results[(mass, 'linear')]
        P_lin = np.array(res_lin['pressure']) / 1e9
        T_lin = np.array(res_lin['temperature'])
        ax1.plot(T_lin, P_lin, color=CMAP_MASS[i], ls='--', lw=1.2, alpha=0.4)

        # Adiabatic (only for low masses)
        if (mass, 'adiabatic') in all_results:
            res_adi = all_results[(mass, 'adiabatic')]
            if res_adi.get('converged', False):
                P_adi = np.array(res_adi['pressure']) / 1e9
                T_adi = np.array(res_adi['temperature'])
                ax1.plot(
                    T_adi,
                    P_adi,
                    color=CMAP_MASS[i],
                    ls='-',
                    lw=2.2,
                    label=f'{mass:.1f} M$_\\oplus$',
                )
            else:
                ax1.plot([], [], color=CMAP_MASS[i], ls='-', lw=2.2, label=f'{mass:.1f} (failed)')
        else:
            ax1.plot(
                T_lin,
                P_lin,
                color=CMAP_MASS[i],
                ls='-',
                lw=1.6,
                alpha=0.6,
                label=f'{mass:.1f} M$_\\oplus$ (linear only)',
            )

    ax1.set_xlabel('Temperature (K)', fontsize=13)
    ax1.set_ylabel('Pressure (GPa)', fontsize=13)
    ax1.set_yscale('log')
    ax1.set_ylim(5e-2, 5e4)
    ax1.set_xlim(2000, 8000)
    ax1.invert_yaxis()
    ax1.set_title('PALEOS MgSiO3: T-P profiles (solid = adiabatic, dashed = linear)', fontsize=11)
    ax1.legend(fontsize=8.5, loc='upper left', ncol=2)
    ax1.grid(True, which='both', alpha=0.3)
    fig1.tight_layout()
    fname1 = os.path.join(OUTPUT_DIR, 'paleos_adiabat_TP.png')
    fig1.savefig(fname1, dpi=200)
    print(f'\nSaved: {fname1}')

    # ── Figure 2: rho vs P ────────────────────────────────────────────
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(14, 6))

    for i, mass in enumerate(ALL_MASSES):
        for mode in ['linear', 'adiabatic']:
            if (mass, mode) not in all_results:
                continue
            res = all_results[(mass, mode)]
            if not res.get('converged', False) and mode == 'adiabatic':
                continue

            P = np.array(res['pressure']) / 1e9
            rho = np.array(res['density'])
            M = np.array(res['mass_enclosed'])
            cmb_mass = res['cmb_mass']

            ls = '-' if mode == 'adiabatic' else '--'
            alpha = 1.0 if mode == 'adiabatic' else 0.4
            lbl = f'{mass:.1f} M$_\\oplus$' if mode == 'adiabatic' else None
            if mass in MASSES_LINEAR_ONLY and mode == 'linear':
                lbl = f'{mass:.1f} M$_\\oplus$ (lin)'
                alpha = 0.7
                ls = '-'

            cmb_idx = np.argmax(M >= cmb_mass) if cmb_mass > 0 else 0
            if cmb_idx == 0:
                cmb_idx = 1

            ax2a.plot(
                P[:cmb_idx],
                rho[:cmb_idx] / 1e3,
                color=CMAP_MASS[i],
                ls=ls,
                lw=1.8,
                alpha=alpha,
                label=lbl,
            )
            ax2b.plot(
                P[cmb_idx:],
                rho[cmb_idx:] / 1e3,
                color=CMAP_MASS[i],
                ls=ls,
                lw=1.8,
                alpha=alpha,
                label=lbl,
            )

    for ax, title in [(ax2a, 'Core (Fe)'), (ax2b, 'Mantle (MgSiO3)')]:
        ax.set_xlabel('Pressure (GPa)', fontsize=13)
        ax.set_title(title, fontsize=12)
        ax.set_xscale('log')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=8.5, loc='upper left')

    ax2a.set_ylabel(r'Density (10$^3$ kg m$^{-3}$)', fontsize=13)
    fig2.suptitle(
        r'PALEOS MgSiO3: $\rho$(P) profiles (solid = adiabatic, dashed = linear)',
        fontsize=12,
    )
    fig2.tight_layout()
    fname2 = os.path.join(OUTPUT_DIR, 'paleos_adiabat_rhoP.png')
    fig2.savefig(fname2, dpi=200, bbox_inches='tight')
    print(f'Saved: {fname2}')

    # ── Figure 3: Full radial profiles for 1 M_earth ─────────────────
    fig3, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_T, ax_rho, ax_P, ax_g = axes.flat

    res_lin = all_results[(1.0, 'linear')]
    res_adi = all_results[(1.0, 'adiabatic')]

    for res, mode, ls, c in [
        (res_lin, 'linear', '--', '#1f77b4'),
        (res_adi, 'adiabatic', '-', '#d62728'),
    ]:
        r = np.array(res['radii']) / 1e6
        ax_T.plot(r, res['temperature'], ls=ls, lw=2, color=c, label=mode)
        ax_rho.plot(r, np.array(res['density']) / 1e3, ls=ls, lw=2, color=c, label=mode)
        ax_P.plot(r, np.array(res['pressure']) / 1e9, ls=ls, lw=2, color=c, label=mode)
        ax_g.plot(r, res['gravity'], ls=ls, lw=2, color=c, label=mode)

    ax_T.set_ylabel('Temperature (K)')
    ax_rho.set_ylabel(r'Density (10$^3$ kg m$^{-3}$)')
    ax_P.set_ylabel('Pressure (GPa)')
    ax_g.set_ylabel(r'Gravity (m s$^{-2}$)')

    for ax in axes.flat:
        ax.set_xlabel('Radius (Mm)')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    R_lin = res_lin['radii'][-1] / earth_radius
    R_adi = res_adi['radii'][-1] / earth_radius
    fig3.suptitle(
        f'1 M$_\\oplus$ PALEOS MgSiO3: linear (R={R_lin:.3f} R$_\\oplus$) '
        f'vs adiabatic (R={R_adi:.3f} R$_\\oplus$)',
        fontsize=13,
    )
    fig3.tight_layout()
    fname3 = os.path.join(OUTPUT_DIR, 'paleos_adiabat_1Mearth_profiles.png')
    fig3.savefig(fname3, dpi=200)
    print(f'Saved: {fname3}')

    # ── Summary table ─────────────────────────────────────────────────
    print('\n' + '=' * 72)
    print(f'{"Mass (M_E)":>12} {"Mode":>10} {"R (R_E)":>10} {"T_c (K)":>10} {"Converged":>10}')
    print('-' * 72)
    for mass in ALL_MASSES:
        for mode in ['linear', 'adiabatic']:
            if (mass, mode) not in all_results:
                continue
            res = all_results[(mass, mode)]
            R = res['radii'][-1] / earth_radius
            Tc = res['temperature'][0]
            conv = res.get('converged', False)
            print(f'{mass:>12.1f} {mode:>10} {R:>10.4f} {Tc:>10.0f} {str(conv):>10}')
    print('=' * 72)

    plt.show()


if __name__ == '__main__':
    main()
