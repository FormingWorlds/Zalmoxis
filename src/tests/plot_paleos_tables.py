"""Diagnostic plot: PALEOS table coverage (density + nabla_ad) with melting
curves and adiabatic T-P profiles overlaid.

Run from the Zalmoxis root:
    python src/tests/plot_paleos_tables.py

Produces:
    output_files/paleos_tables_density.png
    output_files/paleos_tables_nabla_ad.png
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Suppress EOS warning spam during convergence runs
logging.getLogger('zalmoxis.eos_functions').setLevel(logging.CRITICAL)
logging.getLogger('zalmoxis.zalmoxis').setLevel(logging.WARNING)

ZALMOXIS_ROOT = os.environ.get(
    'ZALMOXIS_ROOT', os.path.dirname(os.path.abspath(__file__)) + '/../..'
)
sys.path.insert(0, os.path.join(ZALMOXIS_ROOT, 'src'))

from zalmoxis import zalmoxis as zal
from zalmoxis.constants import earth_mass, earth_radius
from zalmoxis.eos_functions import get_solidus_liquidus_functions
from zalmoxis.zalmoxis import load_material_dictionaries, load_solidus_liquidus_functions

OUTPUT_DIR = os.path.join(ZALMOXIS_ROOT, 'output_files')
os.makedirs(OUTPUT_DIR, exist_ok=True)

solid_file = os.path.join(
    ZALMOXIS_ROOT, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_solid.dat'
)
liquid_file = os.path.join(
    ZALMOXIS_ROOT, 'data', 'EOS_PALEOS_MgSiO3', 'paleos_mgsio3_tables_pt_proteus_liquid.dat'
)


def load_raw_grid(filepath):
    """Load PALEOS table and return grid arrays for plotting."""
    data = np.genfromtxt(filepath, usecols=range(9), comments='#')
    P, T, rho, nabla = data[:, 0], data[:, 1], data[:, 2], data[:, 8]
    valid = P > 0
    P, T, rho, nabla = P[valid], T[valid], rho[valid], nabla[valid]

    logP = np.log10(P)
    logT = np.log10(T)
    ulogP = np.unique(logP)
    ulogT = np.unique(logT)
    nP, nT = len(ulogP), len(ulogT)

    rho_grid = np.full((nP, nT), np.nan)
    nabla_grid = np.full((nP, nT), np.nan)

    p_idx = {v: i for i, v in enumerate(ulogP)}
    t_idx = {v: i for i, v in enumerate(ulogT)}
    for k in range(len(P)):
        ip, it = p_idx[logP[k]], t_idx[logT[k]]
        rho_grid[ip, it] = rho[k]
        nabla_grid[ip, it] = nabla[k]

    return ulogP, ulogT, rho_grid, nabla_grid


def run_adiabatic(mass_earth):
    """Run Zalmoxis with PALEOS adiabatic mode."""
    root = ZALMOXIS_ROOT
    config = zal.load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    config['planet_mass'] = mass_earth * earth_mass
    config['core_mass_fraction'] = 0.325
    config['mantle_mass_fraction'] = 0
    config['temperature_mode'] = 'adiabatic'
    config['surface_temperature'] = 3500.0
    config['center_temperature'] = 6000.0
    config['layer_eos_config'] = {'core': 'Seager2007:iron', 'mantle': 'PALEOS:MgSiO3'}
    config['data_output_enabled'] = False
    config['plotting_enabled'] = False
    config['verbose'] = False
    config['max_iterations_outer'] = 60

    layer_eos = config['layer_eos_config']
    res = zal.main(
        config,
        material_dictionaries=load_material_dictionaries(),
        melting_curves_functions=load_solidus_liquidus_functions(layer_eos),
        input_dir=os.path.join(root, 'input'),
    )
    return res


def plot_table_pair(
    field_name, solid_grid, liquid_grid, ulogP_s, ulogT_s, ulogP_l, ulogT_l,
    solidus_func, liquidus_func, adiabat_results, fname,
):
    """Plot solid + liquid table side by side with melting curves and adiabats."""
    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(16, 8), sharey=True)

    # Prepare display grids
    if field_name == 'density':
        s_plot = np.log10(np.where(np.isfinite(solid_grid), solid_grid, np.nan))
        l_plot = np.log10(np.where(np.isfinite(liquid_grid), liquid_grid, np.nan))
        all_valid = np.concatenate([s_plot[np.isfinite(s_plot)], l_plot[np.isfinite(l_plot)]])
        vmin, vmax = np.nanmin(all_valid), np.nanmax(all_valid)
        cbar_label = r'log$_{10}$($\rho$ [kg m$^{-3}$])'
    else:
        s_plot = np.where(np.isfinite(solid_grid) & (solid_grid > 0), solid_grid, np.nan)
        l_plot = np.where(np.isfinite(liquid_grid) & (liquid_grid > 0), liquid_grid, np.nan)
        all_valid = np.concatenate([s_plot[np.isfinite(s_plot)], l_plot[np.isfinite(l_plot)]])
        vmin, vmax = np.nanpercentile(all_valid, [1, 99])
        cbar_label = r'$\nabla_{ad}$ = d ln T / d ln P'

    for ax, grid, ulogP, ulogT, title in [
        (ax_s, s_plot, ulogP_s, ulogT_s, f'Solid table \u2014 {field_name}'),
        (ax_l, l_plot, ulogP_l, ulogT_l, f'Liquid table \u2014 {field_name}'),
    ]:
        # Build pixel edges from cell centers
        dlogT = np.diff(ulogT)
        dlogP = np.diff(ulogP)
        T_edges = np.concatenate([
            [ulogT[0] - dlogT[0] / 2],
            (ulogT[:-1] + ulogT[1:]) / 2,
            [ulogT[-1] + dlogT[-1] / 2],
        ])
        P_edges = np.concatenate([
            [ulogP[0] - dlogP[0] / 2],
            (ulogP[:-1] + ulogP[1:]) / 2,
            [ulogP[-1] + dlogP[-1] / 2],
        ])

        T_edges_K = 10.0**T_edges
        P_edges_GPa = 10.0**P_edges / 1e9

        im = ax.pcolormesh(
            T_edges_K, P_edges_GPa, grid,
            vmin=vmin, vmax=vmax,
            cmap='viridis', shading='flat',
        )

        # Melting curves
        P_melt = np.logspace(8, 13.5, 500)
        T_sol = np.array([solidus_func(p) for p in P_melt])
        T_liq = np.array([liquidus_func(p) for p in P_melt])
        vm = np.isfinite(T_sol) & np.isfinite(T_liq)
        ax.plot(T_sol[vm], P_melt[vm] / 1e9, 'r-', lw=1.5, label='solidus')
        ax.plot(T_liq[vm], P_melt[vm] / 1e9, 'r--', lw=1.5, label='liquidus')

        # Adiabatic profiles
        colors_adi = ['cyan', 'lime', 'magenta', 'orange']
        for idx, (mass, res) in enumerate(adiabat_results.items()):
            conv = res.get('converged', False)
            P_adi = np.array(res['pressure']) / 1e9
            T_adi = np.array(res['temperature'])
            style = '-' if conv else ':'
            lbl = f'{mass:.0f} M$_\\oplus$'
            if not conv:
                lbl += ' (no conv.)'
            ax.plot(
                T_adi, P_adi,
                color=colors_adi[idx % len(colors_adi)],
                ls=style, lw=2, label=lbl,
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(5e2, 1e5)
        ax.set_ylim(1e-4, 1e5)
        ax.invert_yaxis()
        ax.set_xlabel('Temperature [K]', fontsize=13)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(True, which='both', alpha=0.2, color='grey')

    ax_s.set_ylabel('Pressure [GPa]', fontsize=13)
    fig.colorbar(im, ax=[ax_s, ax_l], label=cbar_label, pad=0.02, fraction=0.03)
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches='tight')
    print(f'Saved: {fname}')
    return fig


def main():
    print('Loading PALEOS tables...')
    ulogP_s, ulogT_s, rho_s, nabla_s = load_raw_grid(solid_file)
    ulogP_l, ulogT_l, rho_l, nabla_l = load_raw_grid(liquid_file)
    print(
        f'  Solid:  {len(ulogP_s)} P x {len(ulogT_s)} T, '
        f'{np.isfinite(rho_s).sum() * 100 / (rho_s.size):.1f}% filled'
    )
    print(
        f'  Liquid: {len(ulogP_l)} P x {len(ulogT_l)} T, '
        f'{np.isfinite(rho_l).sum() * 100 / (rho_l.size):.1f}% filled'
    )

    solidus_func, liquidus_func = get_solidus_liquidus_functions()

    # Run adiabatic cases
    adiabat_results = {}
    for mass in [1.0, 3.0, 5.0, 10.0]:
        print(f'  Running {mass:.0f} M_E adiabatic...', end='', flush=True)
        res = run_adiabatic(mass)
        conv = res.get('converged', False)
        R = res['radii'][-1] / earth_radius
        Tc = res['temperature'][0]
        print(f'  converged={conv}, R={R:.3f} R_E, T_center={Tc:.0f} K')
        adiabat_results[mass] = res

    plot_table_pair(
        'density', rho_s, rho_l,
        ulogP_s, ulogT_s, ulogP_l, ulogT_l,
        solidus_func, liquidus_func, adiabat_results,
        os.path.join(OUTPUT_DIR, 'paleos_tables_density.png'),
    )

    plot_table_pair(
        'nabla_ad', nabla_s, nabla_l,
        ulogP_s, ulogT_s, ulogP_l, ulogT_l,
        solidus_func, liquidus_func, adiabat_results,
        os.path.join(OUTPUT_DIR, 'paleos_tables_nabla_ad.png'),
    )

    plt.show()


if __name__ == '__main__':
    main()
