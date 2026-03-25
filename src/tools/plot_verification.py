"""Verification plots for 2-layer and 3-layer PALEOS models.

Generates diagnostic plots showing phase regimes, layer boundaries,
and structural profiles across temperature and composition space.

Usage:
    python -m src.tests.plot_verification
"""

from __future__ import annotations

import logging
import os
import sys
import time

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s  %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('verify')
logging.getLogger('zalmoxis').setLevel(logging.WARNING)

ZALMOXIS_ROOT = os.environ.get('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    sys.exit('ZALMOXIS_ROOT not set')

from zalmoxis.constants import earth_mass, earth_radius  # noqa: E402
from zalmoxis.eos_functions import load_paleos_unified_table  # noqa: E402
from zalmoxis.eos_properties import EOS_REGISTRY  # noqa: E402
from zalmoxis.melting_curves import monteux16_liquidus, monteux16_solidus  # noqa: E402
from zalmoxis.zalmoxis import (  # noqa: E402
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
    main,
)

OUT = os.path.join(ZALMOXIS_ROOT, 'output_files', 'verification_plots')
os.makedirs(OUT, exist_ok=True)

# Preload caches
MD = load_material_dictionaries()
IRON_CACHE = load_paleos_unified_table(EOS_REGISTRY['PALEOS:iron']['eos_file'])
MGSIO3_CACHE = load_paleos_unified_table(EOS_REGISTRY['PALEOS:MgSiO3']['eos_file'])
H2O_CACHE = load_paleos_unified_table(EOS_REGISTRY['PALEOS:H2O']['eos_file'])


def run(mass_e, core, mantle, mode, T_surf, T_center=6000, ice='', cmf=0.325, mmf=0, mushy=1.0):
    root = ZALMOXIS_ROOT
    c = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    c['planet_mass'] = mass_e * earth_mass
    lec = {'core': core, 'mantle': mantle}
    if ice:
        lec['ice_layer'] = ice
    c['layer_eos_config'] = lec
    c['core_mass_fraction'] = cmf
    c['mantle_mass_fraction'] = mmf
    c['temperature_mode'] = mode
    c['surface_temperature'] = T_surf
    c['center_temperature'] = T_center
    c['mushy_zone_factor'] = mushy
    c['data_output_enabled'] = False
    c['plotting_enabled'] = False
    c['verbose'] = False
    t0 = time.time()
    r = main(
        c,
        material_dictionaries=MD,
        melting_curves_functions=load_solidus_liquidus_functions(
            lec,
            c.get('rock_solidus', 'Stixrude14-solidus'),
            c.get('rock_liquidus', 'Stixrude14-liquidus'),
        ),
        input_dir=os.path.join(root, 'input'),
    )
    return r, time.time() - t0


def lookup_phases(cache, P_arr, T_arr):
    """Look up phase at each (P,T) from a unified PALEOS table."""
    ulp = cache['unique_log_p']
    ult = cache['density_interp'].grid[1]
    pg = cache['phase_grid']
    out = []
    for P, T in zip(P_arr, T_arr):
        if P <= 0 or T <= 0:
            out.append('')
            continue
        lp = np.log10(max(P, cache['p_min']))
        lt = np.log10(max(T, 1.0))
        ip = int(np.argmin(np.abs(ulp - lp)))
        it = int(np.argmin(np.abs(ult - lt)))
        ip = min(ip, pg.shape[0] - 1)
        it = min(it, pg.shape[1] - 1)
        ph = pg[ip, it]
        out.append(str(ph) if ph else '')
    return out


def phase_color_map():
    """Consistent color mapping for all phases."""
    return {
        # Iron
        'solid-alpha-bcc': '#1f77b4',
        'solid-delta-bcc': '#2ca02c',
        'solid-gamma-fcc': '#9467bd',
        'solid-epsilon-hcp': '#17becf',
        # MgSiO3
        'solid-lpcen': '#aec7e8',
        'solid-hpcen': '#98df8a',
        'solid-en': '#c5b0d5',
        'solid-brg': '#ff7f0e',
        'solid-ppv': '#d62728',
        # H2O
        'solid-iceIh': '#e6f2ff',
        'solid-iceII': '#cce5ff',
        'solid-iceIII': '#b3d9ff',
        'solid-iceV': '#99ccff',
        'solid-iceVI': '#80bfff',
        'solid-iceVII': '#66b3ff',
        'solid-iceVIII': '#4da6ff',
        'solid-iceX': '#3399ff',
        'superionic': '#ff9896',
        'vapor': '#ffbb78',
        # Common
        'liquid': '#e31a1c',
        '': '#cccccc',
    }


# ═══════════════════════════════════════════════════════════════════
# PLOT 1: Master T-P phase diagram with adiabats at multiple T_surf
# ═══════════════════════════════════════════════════════════════════


def plot1_tp_phase_diagram():
    logger.info('Plot 1: T-P phase diagram with adiabats')

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    for ax, cache, title, mat_label in [
        (axes[0], IRON_CACHE, 'PALEOS:iron', 'Fe'),
        (axes[1], MGSIO3_CACHE, 'PALEOS:MgSiO3', 'MgSiO3'),
        (axes[2], H2O_CACHE, 'PALEOS:H2O', 'H2O'),
    ]:
        # Background: density map
        ulp = cache['unique_log_p']
        ult = cache['density_interp'].grid[1]
        PP, TT = np.meshgrid(ulp, ult, indexing='ij')
        rho = cache['density_interp'](np.column_stack([PP.ravel(), TT.ravel()])).reshape(
            PP.shape
        )

        ax.pcolormesh(
            10.0**ulp / 1e9,
            10.0**ult,
            np.log10(rho).T,
            shading='auto',
            cmap='viridis',
            alpha=0.4,
            rasterized=True,
        )

        # Extracted liquidus
        if len(cache['liquidus_log_p']) > 0:
            ax.plot(
                10.0 ** cache['liquidus_log_p'] / 1e9,
                10.0 ** cache['liquidus_log_t'],
                'r-',
                lw=2.5,
                label='Liquidus (from table)',
            )

        # Monteux melting curves (for MgSiO3 comparison)
        if mat_label == 'MgSiO3':
            P_mc = np.logspace(8, 12.5, 300)
            ax.plot(
                P_mc / 1e9,
                [monteux16_solidus(p) for p in P_mc],
                'k--',
                lw=1.5,
                label='Monteux16 solidus',
            )
            ax.plot(
                P_mc / 1e9,
                [monteux16_liquidus(p) for p in P_mc],
                'k-.',
                lw=1.5,
                label='Monteux16 liquidus',
            )

        ax.set_xlabel('Pressure [GPa]')
        ax.set_ylabel('Temperature [K]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'{title} ({mat_label})')

    # Overlay adiabats at multiple T_surf for 1 M_earth
    colors_t = plt.cm.coolwarm(np.linspace(0, 1, 8))
    for idx, T_surf in enumerate([300, 500, 1000, 1500, 2000, 3000, 4000, 5000]):
        try:
            res, _ = run(
                1.0,
                'PALEOS:iron',
                'PALEOS:MgSiO3',
                'adiabatic',
                T_surf,
                T_center=max(T_surf + 2000, 6000),
            )
            if not res['converged']:
                continue
            P = res['pressure']
            T = res['temperature']
            M = res['mass_enclosed']
            cmb = res['cmb_mass']
            mask_valid = P > 1e5

            core_mask = (M < cmb) & mask_valid
            mantle_mask = (M >= cmb) & mask_valid

            c = colors_t[idx]
            if np.any(core_mask):
                axes[0].plot(
                    P[core_mask] / 1e9, T[core_mask], '-', color=c, lw=1.5, label=f'{T_surf} K'
                )
            if np.any(mantle_mask):
                axes[1].plot(
                    P[mantle_mask] / 1e9,
                    T[mantle_mask],
                    '-',
                    color=c,
                    lw=1.5,
                    label=f'{T_surf} K',
                )
        except Exception as e:
            logger.warning(f'  T_surf={T_surf}: {e}')

    # 3-layer: overlay H2O adiabat
    try:
        res3, _ = run(
            1.0,
            'PALEOS:iron',
            'PALEOS:MgSiO3',
            'adiabatic',
            300,
            T_center=3000,
            ice='PALEOS:H2O',
            cmf=0.25,
            mmf=0.50,
        )
        if res3['converged']:
            P = res3['pressure']
            T = res3['temperature']
            M = res3['mass_enclosed']
            cm_mass = res3['core_mantle_mass']
            ice_mask = (M >= cm_mass) & (P > 1e5)
            if np.any(ice_mask):
                axes[2].plot(
                    P[ice_mask] / 1e9, T[ice_mask], 'b-', lw=2, label='3-layer 300K adiabat'
                )
    except Exception as e:
        logger.warning(f'  3-layer: {e}')

    for ax in axes:
        ax.legend(fontsize=7, loc='upper left')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '01_tp_phase_diagram.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('  Saved 01_tp_phase_diagram.png')


# ═══════════════════════════════════════════════════════════════════
# PLOT 2: Radial profiles with phase-colored strips
# ═══════════════════════════════════════════════════════════════════


def plot2_radial_with_phases():
    logger.info('Plot 2: Radial profiles with phase identification')

    temps = [300, 1000, 2000, 3000, 5000]
    fig, axes = plt.subplots(len(temps), 4, figsize=(20, 4 * len(temps)), sharex=False)
    pcm = phase_color_map()

    for row, T_surf in enumerate(temps):
        try:
            res, dt = run(
                1.0,
                'PALEOS:iron',
                'PALEOS:MgSiO3',
                'adiabatic',
                T_surf,
                T_center=max(T_surf + 2000, 6000),
            )
            if not res['converged']:
                for ax in axes[row]:
                    ax.text(
                        0.5,
                        0.5,
                        'DID NOT CONVERGE',
                        transform=ax.transAxes,
                        ha='center',
                        fontsize=12,
                        color='red',
                    )
                continue

            r = res['radii'] / 1e6  # Mm
            P = res['pressure']
            T = res['temperature']
            rho = res['density']
            g = res['gravity']
            M = res['mass_enclosed']
            cmb = res['cmb_mass']
            cmb_idx = np.argmax(M >= cmb)

            # Get phases
            core_phases = lookup_phases(IRON_CACHE, P[:cmb_idx], T[:cmb_idx])
            mantle_phases = lookup_phases(MGSIO3_CACHE, P[cmb_idx:], T[cmb_idx:])
            all_phases = core_phases + mantle_phases

            # Plot profiles
            for col, (y, ylabel) in enumerate(
                [
                    (T, 'Temperature [K]'),
                    (rho, 'Density [kg/m$^3$]'),
                    (P / 1e9, 'Pressure [GPa]'),
                    (g, 'Gravity [m/s$^2$]'),
                ]
            ):
                ax = axes[row, col]
                # Color each segment by phase
                for i in range(len(r) - 1):
                    ph = all_phases[i] if i < len(all_phases) else ''
                    color = pcm.get(ph, '#888888')
                    ax.plot(r[i : i + 2], y[i : i + 2], '-', color=color, lw=2)

                # CMB line
                ax.axvline(r[cmb_idx], color='gray', ls=':', lw=1, alpha=0.7)
                ax.set_ylabel(ylabel)
                if row == len(temps) - 1:
                    ax.set_xlabel('Radius [Mm]')

            # Label row
            axes[row, 0].text(
                -0.25,
                0.5,
                f'T$_{{surf}}$ = {T_surf} K\n({dt:.0f}s)',
                transform=axes[row, 0].transAxes,
                fontsize=11,
                ha='center',
                va='center',
                rotation=90,
                fontweight='bold',
            )

        except Exception as e:
            logger.warning(f'  T_surf={T_surf}: {e}')

    # Legend
    unique_phases = sorted(set(ph for ph in pcm.keys() if ph and ph != ''))
    legend_handles = [
        plt.Line2D([0], [0], color=pcm[ph], lw=4, label=ph) for ph in unique_phases if ph in pcm
    ]
    fig.legend(
        handles=legend_handles,
        loc='lower center',
        ncol=6,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.02),
    )

    fig.suptitle(
        '1 M$_\\oplus$ PALEOS adiabatic: radial profiles colored by phase\n'
        '(vertical dotted line = CMB)',
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '02_radial_phase_profiles.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('  Saved 02_radial_phase_profiles.png')


# ═══════════════════════════════════════════════════════════════════
# PLOT 3: Phase transition map (T_surf vs radius, colored by phase)
# ═══════════════════════════════════════════════════════════════════


def plot3_phase_transition_map():
    logger.info('Plot 3: Phase transition map')

    temps = [300, 500, 800, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
    pcm = phase_color_map()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, cache, title in [
        (axes[0], IRON_CACHE, 'Iron core'),
        (axes[1], MGSIO3_CACHE, 'MgSiO3 mantle'),
    ]:
        for T_surf in temps:
            try:
                res, _ = run(
                    1.0,
                    'PALEOS:iron',
                    'PALEOS:MgSiO3',
                    'adiabatic',
                    T_surf,
                    T_center=max(T_surf + 2000, 6000),
                )
                if not res['converged']:
                    continue

                P = res['pressure']
                T = res['temperature']
                r = res['radii'] / 1e6
                M = res['mass_enclosed']
                cmb = res['cmb_mass']
                cmb_idx = np.argmax(M >= cmb)

                if cache is IRON_CACHE:
                    phases = lookup_phases(cache, P[:cmb_idx], T[:cmb_idx])
                    r_seg = r[:cmb_idx]
                else:
                    phases = lookup_phases(cache, P[cmb_idx:], T[cmb_idx:])
                    r_seg = r[cmb_idx:]

                # Plot each point colored by phase
                for i, (ri, ph) in enumerate(zip(r_seg, phases)):
                    color = pcm.get(ph, '#888888')
                    ax.scatter(ri, T_surf, c=color, s=3, marker='s', edgecolors='none')

            except Exception:
                pass

        ax.set_xlabel('Radius [Mm]')
        ax.set_ylabel('Surface temperature [K]')
        ax.set_title(title)

    # Legend
    seen = set()
    handles = []
    for ph, color in pcm.items():
        if ph and ph not in seen:
            seen.add(ph)
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker='s',
                    color='w',
                    markerfacecolor=color,
                    markersize=8,
                    label=ph,
                )
            )
    fig.legend(
        handles=handles, loc='lower center', ncol=5, fontsize=8, bbox_to_anchor=(0.5, -0.05)
    )
    fig.suptitle('1 M$_\\oplus$: phase along adiabat vs surface temperature', fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '03_phase_transition_map.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('  Saved 03_phase_transition_map.png')


# ═══════════════════════════════════════════════════════════════════
# PLOT 4: 3-layer model verification
# ═══════════════════════════════════════════════════════════════════


def plot4_three_layer():
    logger.info('Plot 4: 3-layer model profiles')

    pcm = phase_color_map()
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))

    configs = [
        ('Adiabatic 300 K', 'adiabatic', 300, 3000),
        ('Isothermal 300 K', 'isothermal', 300, 3000),
    ]

    for row, (label, mode, T_surf, T_center) in enumerate(configs):
        try:
            res, dt = run(
                1.0,
                'PALEOS:iron',
                'PALEOS:MgSiO3',
                mode,
                T_surf,
                T_center=T_center,
                ice='PALEOS:H2O',
                cmf=0.25,
                mmf=0.50,
            )

            r = res['radii'] / 1e6
            P = res['pressure']
            T = res['temperature']
            rho = res['density']
            g = res['gravity']
            M = res['mass_enclosed']
            cmb = res['cmb_mass']
            cm_mass = res['core_mantle_mass']
            cmb_idx = np.argmax(M >= cmb)
            ice_idx = np.argmax(M >= cm_mass)

            # Phases for each layer
            core_ph = lookup_phases(IRON_CACHE, P[:cmb_idx], T[:cmb_idx])
            mantle_ph = lookup_phases(MGSIO3_CACHE, P[cmb_idx:ice_idx], T[cmb_idx:ice_idx])
            ice_ph = lookup_phases(H2O_CACHE, P[ice_idx:], T[ice_idx:])
            all_ph = core_ph + mantle_ph + ice_ph

            for col, (y, ylabel) in enumerate(
                [
                    (T, 'Temperature [K]'),
                    (rho, 'Density [kg/m$^3$]'),
                    (P / 1e9, 'Pressure [GPa]'),
                    (g, 'Gravity [m/s$^2$]'),
                ]
            ):
                ax = axes[row, col]
                for i in range(len(r) - 1):
                    ph = all_ph[i] if i < len(all_ph) else ''
                    color = pcm.get(ph, '#888888')
                    ax.plot(r[i : i + 2], y[i : i + 2], '-', color=color, lw=2)

                # Layer boundaries
                ax.axvline(r[cmb_idx], color='gray', ls=':', lw=1, alpha=0.7)
                if ice_idx > 0 and ice_idx < len(r):
                    ax.axvline(r[ice_idx], color='blue', ls=':', lw=1, alpha=0.7)
                ax.set_ylabel(ylabel)
                if row == 1:
                    ax.set_xlabel('Radius [Mm]')

            axes[row, 0].text(
                -0.25,
                0.5,
                f'{label}\nconv={res["converged"]}\n{dt:.0f}s',
                transform=axes[row, 0].transAxes,
                fontsize=10,
                ha='center',
                va='center',
                rotation=90,
                fontweight='bold',
            )

        except Exception as e:
            logger.warning(f'  {label}: {e}')
            for ax in axes[row]:
                ax.text(
                    0.5,
                    0.5,
                    f'FAILED: {e}',
                    transform=ax.transAxes,
                    ha='center',
                    fontsize=10,
                    color='red',
                    wrap=True,
                )

    # Legend
    handles = [
        plt.Line2D([0], [0], color=pcm.get(ph, '#888'), lw=4, label=ph)
        for ph in sorted(pcm.keys())
        if ph
    ]
    fig.legend(
        handles=handles, loc='lower center', ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.04)
    )
    fig.suptitle(
        '3-layer 1 M$_\\oplus$: PALEOS:iron + PALEOS:MgSiO3 + PALEOS:H2O\n'
        '(gray dotted = CMB, blue dotted = mantle/ice boundary)',
        fontsize=13,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '04_three_layer.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('  Saved 04_three_layer.png')


# ═══════════════════════════════════════════════════════════════════
# PLOT 5: EOS comparison at T_surf where phases differ
# ═══════════════════════════════════════════════════════════════════


def plot5_eos_comparison():
    logger.info('Plot 5: EOS comparison across temperatures')

    fig, axes = plt.subplots(3, 4, figsize=(22, 14))

    for row, T_surf in enumerate([1000, 2000, 3000]):
        configs = [
            ('PALEOS unified', 'PALEOS:iron', 'PALEOS:MgSiO3', 'adiabatic', T_surf),
            ('PALEOS-2phase', 'Seager2007:iron', 'PALEOS-2phase:MgSiO3', 'adiabatic', T_surf),
            ('WolfBower2018', 'Seager2007:iron', 'WolfBower2018:MgSiO3', 'adiabatic', T_surf),
            ('Seager2007 300K', 'Seager2007:iron', 'Seager2007:MgSiO3', 'isothermal', 300),
        ]
        for label, core, mantle, mode, ts in configs:
            try:
                res, _ = run(1.0, core, mantle, mode, ts, T_center=max(ts + 2000, 6000))
                if not res['converged']:
                    continue
                r = res['radii'] / 1e6
                R_re = res['radii'][-1] / earth_radius
                lbl = f'{label} (R={R_re:.3f})'

                axes[row, 0].plot(r, res['temperature'], label=lbl)
                axes[row, 1].plot(r, res['density'], label=lbl)
                axes[row, 2].plot(r, res['pressure'] / 1e9, label=lbl)
                axes[row, 3].plot(r, res['gravity'], label=lbl)
            except Exception as e:
                logger.warning(f'  {label} T={T_surf}: {e}')

        axes[row, 0].set_ylabel('Temperature [K]')
        axes[row, 1].set_ylabel('Density [kg/m$^3$]')
        axes[row, 2].set_ylabel('Pressure [GPa]')
        axes[row, 3].set_ylabel('Gravity [m/s$^2$]')
        axes[row, 0].text(
            -0.2,
            0.5,
            f'T$_{{surf}}$ = {T_surf} K',
            transform=axes[row, 0].transAxes,
            fontsize=12,
            ha='center',
            va='center',
            rotation=90,
            fontweight='bold',
        )
        axes[row, 0].legend(fontsize=7)

    for ax in axes[-1]:
        ax.set_xlabel('Radius [Mm]')

    fig.suptitle(
        '1 M$_\\oplus$ EOS comparison: PALEOS vs PALEOS-2phase vs WB2018 vs Seager',
        fontsize=14,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '05_eos_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('  Saved 05_eos_comparison.png')


# ═══════════════════════════════════════════════════════════════════
# PLOT 6: Radius and T_center vs T_surf (summary)
# ═══════════════════════════════════════════════════════════════════


def plot6_summary():
    logger.info('Plot 6: Summary R and T_center vs T_surf')

    temps = [300, 500, 800, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
    masses = [1.0, 5.0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for mass_e in masses:
        ts_list, R_list, Tc_list, rho_c_list = [], [], [], []
        for T_surf in temps:
            try:
                res, _ = run(
                    mass_e,
                    'PALEOS:iron',
                    'PALEOS:MgSiO3',
                    'adiabatic',
                    T_surf,
                    T_center=max(T_surf + 2000, 6000),
                )
                if not res['converged']:
                    continue
                ts_list.append(T_surf)
                R_list.append(res['radii'][-1] / earth_radius)
                Tc_list.append(res['temperature'][0])
                rho_c_list.append(res['density'][0])
            except Exception:
                pass

        axes[0].plot(ts_list, R_list, 'o-', label=f'{mass_e} M$_\\oplus$')
        axes[1].plot(ts_list, Tc_list, 'o-', label=f'{mass_e} M$_\\oplus$')
        axes[2].plot(ts_list, rho_c_list, 'o-', label=f'{mass_e} M$_\\oplus$')

    axes[0].set_ylabel('Radius [R$_\\oplus$]')
    axes[1].set_ylabel('Center temperature [K]')
    axes[2].set_ylabel('Center density [kg/m$^3$]')
    for ax in axes:
        ax.set_xlabel('Surface temperature [K]')
        ax.legend()

    fig.suptitle('PALEOS unified adiabatic: structural parameters vs surface temperature')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, '06_summary.png'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    logger.info('  Saved 06_summary.png')


# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info(f'Output: {OUT}')
    plot1_tp_phase_diagram()
    plot2_radial_with_phases()
    plot3_phase_transition_map()
    plot4_three_layer()
    plot5_eos_comparison()
    plot6_summary()
    logger.info('All verification plots complete.')
