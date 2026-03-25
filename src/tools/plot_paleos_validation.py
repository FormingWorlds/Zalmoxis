"""Validation plots for PALEOS-2phase:MgSiO3 adiabatic mode.

Produces:
1. T-P profiles on PALEOS table background (density) with solidus/liquidus
2. Density vs pressure profiles (core + mantle)
3. Mass-radius comparison with Zeng+2019 Earth-like rocky curve

Run from the Zalmoxis root:
    python src/tests/plot_paleos_validation.py
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

logging.getLogger('zalmoxis.eos_functions').setLevel(logging.CRITICAL)
logging.getLogger('zalmoxis.zalmoxis').setLevel(logging.WARNING)

ZALMOXIS_ROOT = os.environ.get(
    'ZALMOXIS_ROOT', os.path.dirname(os.path.abspath(__file__)) + '/../..'
)
sys.path.insert(0, os.path.join(ZALMOXIS_ROOT, 'src'))

from zalmoxis import zalmoxis as zal  # noqa: E402
from zalmoxis.constants import earth_mass, earth_radius  # noqa: E402
from zalmoxis.zalmoxis import (  # noqa: E402
    load_material_dictionaries,
    load_solidus_liquidus_functions,
)

OUTPUT_DIR = os.path.join(ZALMOXIS_ROOT, 'output_files')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MASSES = [1.0, 3.0, 5.0, 10.0]
COLORS = {1.0: 'cyan', 3.0: 'lime', 5.0: 'magenta', 10.0: 'orange'}


# ── Helpers ────────────────────────────────────────────────────────────


def load_raw_paleos_grid(filepath):
    """Load PALEOS table into 2D grids for pcolormesh plotting."""
    data = np.genfromtxt(filepath, usecols=range(9), comments='#')
    P, T, rho = data[:, 0], data[:, 1], data[:, 2]
    valid = P > 0
    P, T, rho = P[valid], T[valid], rho[valid]
    logP, logT = np.log10(P), np.log10(T)
    ulogP, ulogT = np.unique(logP), np.unique(logT)
    grid = np.full((len(ulogP), len(ulogT)), np.nan)
    p_idx = {v: i for i, v in enumerate(ulogP)}
    t_idx = {v: i for i, v in enumerate(ulogT)}
    for k in range(len(P)):
        grid[p_idx[logP[k]], t_idx[logT[k]]] = rho[k]
    return ulogP, ulogT, grid


def run_case(mass_earth, temperature_mode):
    """Run Zalmoxis for a single case."""
    config = zal.load_zalmoxis_config(os.path.join(ZALMOXIS_ROOT, 'input', 'default.toml'))
    config['planet_mass'] = mass_earth * earth_mass
    config['core_mass_fraction'] = 0.325
    config['mantle_mass_fraction'] = 0
    config['temperature_mode'] = temperature_mode
    config['surface_temperature'] = 3500.0
    config['center_temperature'] = 6000.0
    config['layer_eos_config'] = {'core': 'Seager2007:iron', 'mantle': 'PALEOS-2phase:MgSiO3'}
    config['data_output_enabled'] = False
    config['plotting_enabled'] = False
    config['verbose'] = False
    layer_eos = config['layer_eos_config']
    return zal.main(
        config,
        material_dictionaries=load_material_dictionaries(),
        melting_curves_functions=load_solidus_liquidus_functions(layer_eos),
        input_dir=os.path.join(ZALMOXIS_ROOT, 'input'),
    )


def load_zeng_curve(filename):
    """Load Zeng+2019 mass-radius data."""
    path = os.path.join(ZALMOXIS_ROOT, 'data', 'mass_radius_curves', filename)
    data = np.loadtxt(path)
    return data[:, 0], data[:, 1]  # M_earth, R_earth


def make_pcolormesh_edges(ulog):
    """Build pixel edges from cell centers for pcolormesh."""
    d = np.diff(ulog)
    return np.concatenate(
        [
            [ulog[0] - d[0] / 2],
            (ulog[:-1] + ulog[1:]) / 2,
            [ulog[-1] + d[-1] / 2],
        ]
    )


# ── Main ───────────────────────────────────────────────────────────────


def main():
    print('Running PALEOS adiabatic validation cases...\n')

    # Run all cases
    results = {}
    for m in MASSES:
        for mode in ['linear', 'adiabatic']:
            print(f'  {m:.0f} M_E {mode}...', end='', flush=True)
            res = run_case(m, mode)
            conv = res.get('converged', False)
            R = res['radii'][-1] / earth_radius
            Tc = res['temperature'][0]
            print(f'  converged={conv}, R={R:.4f} R_E, T_c={Tc:.0f} K')
            results[(m, mode)] = res

    # Load PALEOS table backgrounds
    solid_file = os.path.join(
        ZALMOXIS_ROOT,
        'data',
        'EOS_PALEOS_MgSiO3',
        'paleos_mgsio3_tables_pt_proteus_solid.dat',
    )
    liquid_file = os.path.join(
        ZALMOXIS_ROOT,
        'data',
        'EOS_PALEOS_MgSiO3',
        'paleos_mgsio3_tables_pt_proteus_liquid.dat',
    )
    ulogP_s, ulogT_s, rho_s = load_raw_paleos_grid(solid_file)
    ulogP_l, ulogT_l, rho_l = load_raw_paleos_grid(liquid_file)

    # Melting curves (Monteux16 analytic, the new default)
    # Clip to the valid range where liquidus > solidus (~490 GPa for A-chondritic)
    from zalmoxis.melting_curves import monteux16_liquidus, monteux16_solidus

    P_melt = np.logspace(5, 13.5, 1000)
    T_sol_all = monteux16_solidus(P_melt)
    T_liq_all = monteux16_liquidus(P_melt, model='A-chondritic')
    valid_melt = T_liq_all > T_sol_all
    P_melt = P_melt[valid_melt]
    T_sol = T_sol_all[valid_melt]
    T_liq = T_liq_all[valid_melt]

    # ── Figure 1: T-P on PALEOS table background ──────────────────────
    fig1, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    log_rho_s = np.log10(np.where(np.isfinite(rho_s), rho_s, np.nan))
    log_rho_l = np.log10(np.where(np.isfinite(rho_l), rho_l, np.nan))
    all_log_rho = np.concatenate(
        [log_rho_s[np.isfinite(log_rho_s)], log_rho_l[np.isfinite(log_rho_l)]]
    )
    vmin, vmax = np.nanmin(all_log_rho), np.nanmax(all_log_rho)

    for ax, grid, ulogP, ulogT, title in [
        (ax_s, log_rho_s, ulogP_s, ulogT_s, 'Solid table'),
        (ax_l, log_rho_l, ulogP_l, ulogT_l, 'Liquid table'),
    ]:
        T_edges = 10.0 ** make_pcolormesh_edges(ulogT)
        P_edges = 10.0 ** make_pcolormesh_edges(ulogP) / 1e9
        im = ax.pcolormesh(
            T_edges,
            P_edges,
            grid,
            vmin=vmin,
            vmax=vmax,
            cmap='viridis',
            shading='flat',
        )

        # Melting curves
        ax.plot(T_sol, P_melt / 1e9, 'r-', lw=1.5, label='solidus')
        ax.plot(T_liq, P_melt / 1e9, 'r--', lw=1.5, label='liquidus')

        # Adiabatic T-P profiles
        for m in MASSES:
            res = results[(m, 'adiabatic')]
            conv = res.get('converged', False)
            P = np.array(res['pressure']) / 1e9
            T = np.array(res['temperature'])
            ls = '-' if conv else ':'
            lbl = f'{m:.0f} M$_\\oplus$'
            if not conv:
                lbl += ' (no conv.)'
            ax.plot(T, P, color=COLORS[m], ls=ls, lw=2.2, label=lbl)

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(5e2, 1e5)
        ax.set_ylim(1e-4, 1e5)
        ax.invert_yaxis()
        ax.set_xlabel('Temperature [K]', fontsize=13)
        ax.set_title(f'{title} (density)', fontsize=12)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, which='both', alpha=0.2, color='grey')

    ax_s.set_ylabel('Pressure [GPa]', fontsize=13)
    fig1.colorbar(
        im,
        ax=[ax_s, ax_l],
        label=r'log$_{10}$($\rho$ [kg m$^{-3}$])',
        pad=0.02,
        fraction=0.03,
    )
    fig1.tight_layout()
    fname1 = os.path.join(OUTPUT_DIR, 'paleos_validation_TP.png')
    fig1.savefig(fname1, dpi=200, bbox_inches='tight')
    print(f'\nSaved: {fname1}')

    # ── Figure 2: Density vs Pressure ─────────────────────────────────
    fig2, (ax_core, ax_mantle) = plt.subplots(1, 2, figsize=(14, 6))

    for m in MASSES:
        for mode in ['linear', 'adiabatic']:
            res = results[(m, mode)]
            if not res.get('converged', False) and mode == 'adiabatic':
                continue
            P = np.array(res['pressure']) / 1e9
            rho = np.array(res['density'])
            M = np.array(res['mass_enclosed'])
            cmb = res['cmb_mass']

            ls = '-' if mode == 'adiabatic' else '--'
            alpha = 1.0 if mode == 'adiabatic' else 0.35
            lbl = f'{m:.0f} M$_\\oplus$' if mode == 'adiabatic' else None

            cmb_idx = max(1, np.argmax(M >= cmb))

            ax_core.plot(
                P[:cmb_idx],
                rho[:cmb_idx] / 1e3,
                color=COLORS[m],
                ls=ls,
                lw=1.8,
                alpha=alpha,
                label=lbl,
            )
            ax_mantle.plot(
                P[cmb_idx:],
                rho[cmb_idx:] / 1e3,
                color=COLORS[m],
                ls=ls,
                lw=1.8,
                alpha=alpha,
                label=lbl,
            )

    for ax, title in [(ax_core, 'Core (Fe)'), (ax_mantle, 'Mantle (MgSiO3)')]:
        ax.set_xlabel('Pressure (GPa)', fontsize=13)
        ax.set_title(title, fontsize=12)
        ax.set_xscale('log')
        ax.grid(True, which='both', alpha=0.3)
        ax.legend(fontsize=9, loc='upper left')

    ax_core.set_ylabel(r'Density (10$^3$ kg m$^{-3}$)', fontsize=13)
    fig2.suptitle(
        r'PALEOS MgSiO3: $\rho$(P) profiles, solid = adiabatic, dashed = linear',
        fontsize=12,
    )
    fig2.tight_layout()
    fname2 = os.path.join(OUTPUT_DIR, 'paleos_validation_rhoP.png')
    fig2.savefig(fname2, dpi=200, bbox_inches='tight')
    print(f'Saved: {fname2}')

    # ── Figure 3: Mass-Radius comparison with Zeng ────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 6))

    # Zeng+2019 Earth-like rocky curve
    zeng_file = os.path.join(
        ZALMOXIS_ROOT, 'data', 'mass_radius_curves', 'massradiusEarthlikeRocky.txt'
    )
    if os.path.isfile(zeng_file):
        zM, zR = load_zeng_curve('massradiusEarthlikeRocky.txt')
        ax3.plot(zM, zR, 'k-', lw=2, label='Zeng+2019 Earth-like rocky', zorder=5)

    # Zalmoxis PALEOS results
    for mode, ls, marker, alpha, label_sfx in [
        ('linear', '--', 's', 0.5, 'linear'),
        ('adiabatic', '-', 'o', 1.0, 'adiabatic'),
    ]:
        masses_plot = []
        radii_plot = []
        for m in MASSES:
            res = results[(m, mode)]
            if not res.get('converged', False):
                continue
            masses_plot.append(m)
            radii_plot.append(res['radii'][-1] / earth_radius)
        if masses_plot:
            ax3.plot(
                masses_plot,
                radii_plot,
                ls=ls,
                marker=marker,
                ms=8,
                lw=1.5,
                alpha=alpha,
                label=f'PALEOS-2phase:MgSiO3 ({label_sfx})',
            )

    ax3.set_xlabel(r'Planet mass [M$_\oplus$]', fontsize=13)
    ax3.set_ylabel(r'Planet radius [R$_\oplus$]', fontsize=13)
    ax3.set_xscale('log')
    ax3.set_xlim(0.3, 15)
    ax3.set_ylim(0.7, 2.2)
    ax3.legend(fontsize=10, loc='upper left')
    ax3.grid(True, which='both', alpha=0.3)
    ax3.set_title('Mass-radius: PALEOS vs Zeng+2019', fontsize=12)
    fig3.tight_layout()
    fname3 = os.path.join(OUTPUT_DIR, 'paleos_validation_MR.png')
    fig3.savefig(fname3, dpi=200)
    print(f'Saved: {fname3}')

    # ── Figure 4: Radial profiles for 1 M_earth ──────────────────────
    fig4, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_T, ax_rho, ax_P, ax_g = axes.flat

    for mode, ls, c in [('linear', '--', '#1f77b4'), ('adiabatic', '-', '#d62728')]:
        res = results[(1.0, mode)]
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

    R_lin = results[(1.0, 'linear')]['radii'][-1] / earth_radius
    R_adi = results[(1.0, 'adiabatic')]['radii'][-1] / earth_radius
    fig4.suptitle(
        f'1 M$_\\oplus$ PALEOS: linear (R={R_lin:.3f}) vs adiabatic (R={R_adi:.3f})',
        fontsize=13,
    )
    fig4.tight_layout()
    fname4 = os.path.join(OUTPUT_DIR, 'paleos_validation_1Me_profiles.png')
    fig4.savefig(fname4, dpi=200)
    print(f'Saved: {fname4}')

    # ── Summary table ─────────────────────────────────────────────────
    print('\n' + '=' * 78)
    print(
        f'{"Mass":>8} {"Mode":>10} {"R (R_E)":>10} {"T_c (K)":>10} '
        f'{"P_c (GPa)":>10} {"Conv":>6}'
    )
    print('-' * 78)
    for m in MASSES:
        for mode in ['linear', 'adiabatic']:
            res = results[(m, mode)]
            R = res['radii'][-1] / earth_radius
            Tc = res['temperature'][0]
            Pc = res['pressure'][0] / 1e9
            conv = res.get('converged', False)
            print(f'{m:>8.1f} {mode:>10} {R:>10.4f} {Tc:>10.0f} {Pc:>10.1f} {str(conv):>6}')
    print('=' * 78)

    plt.show()


if __name__ == '__main__':
    main()
