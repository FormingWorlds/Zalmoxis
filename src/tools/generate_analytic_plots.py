"""
Generate comparison plots for the Seager+2007 analytic EOS.

Produces 4 plots:
1. EOS comparison (P vs rho) — all 6 analytic materials + tabulated overlays
2. Density residuals (analytic vs tabulated) — iron, silicate, water
3. Mass-radius comparison — analytic vs tabulated for rocky and water planets
4. Interior density profiles — 1 M_earth Earth-like, analytic vs tabulated

Saves all plots as PDF to output_files/.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure src is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from zalmoxis.eos_analytic import SEAGER2007_MATERIALS, get_analytic_density

ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')

OUTPUT_DIR = os.path.join(ZALMOXIS_ROOT, 'output_files')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot1_eos_comparison():
    """Plot 1: EOS comparison (P vs rho) for all 6 materials."""
    # Tabulated data: open markers matching the analytic curve color
    tabulated_files = {
        'eos_seager07_iron.txt': ('iron', 'o', '#d62728'),
        'eos_seager07_silicate.txt': ('MgSiO3', 's', '#ff7f0e'),
        'eos_seager07_water.txt': ('H2O', '^', '#1f77b4'),
    }

    # Analytic curves: distinct colors, all solid for materials with tabulated data,
    # dashed for analytic-only materials
    analytic_styles = {
        'iron': ('#d62728', '-', 2.5),  # red
        'MgSiO3': ('#ff7f0e', '-', 2.5),  # orange
        'MgFeSiO3': ('#8c564b', '--', 1.8),  # brown
        'H2O': ('#1f77b4', '-', 2.5),  # blue
        'graphite': ('#7f7f7f', '--', 1.8),  # gray
        'SiC': ('#2ca02c', '--', 1.8),  # green
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot analytic curves first (underneath)
    pressures_pa = np.logspace(7, 21, 500)
    pressures_gpa = pressures_pa / 1e9

    for mat_key in SEAGER2007_MATERIALS:
        color, ls, lw = analytic_styles[mat_key]
        densities = np.array([get_analytic_density(p, mat_key) for p in pressures_pa])
        ax.plot(
            pressures_gpa,
            densities,
            color=color,
            linestyle=ls,
            linewidth=lw,
            label=f'{mat_key} (analytic)',
            zorder=2,
        )

    # Plot tabulated data on top as large open markers, subsampled for clarity
    data_folder = os.path.join(ZALMOXIS_ROOT, 'data', 'EOS_Seager2007')
    for filename, (mat_key, marker, color) in tabulated_files.items():
        filepath = os.path.join(data_folder, filename)
        if os.path.exists(filepath):
            data = np.loadtxt(filepath, delimiter=',', skiprows=1)
            pressure_gpa = data[:, 1]
            density = data[:, 0] * 1000  # g/cm^3 -> kg/m^3
            # Subsample every 5th point so markers don't overlap
            step = max(1, len(pressure_gpa) // 40)
            ax.scatter(
                pressure_gpa[::step],
                density[::step],
                facecolors='none',
                edgecolors=color,
                s=60,
                linewidths=1.5,
                marker=marker,
                label=f'{mat_key} (tabulated)',
                zorder=4,
            )

    ax.set_xlabel('Pressure (GPa)', fontsize=13)
    ax.set_ylabel('Density (kg/m³)', fontsize=13)
    ax.set_title('Seager+2007 EOS: Tabulated vs Analytic Modified Polytrope', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=9, ncol=2, loc='upper left')
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, 'plot1_eos_comparison.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')
    return path


def plot2_density_residuals():
    """Plot 2: Density residuals (analytic vs tabulated) for iron, silicate, water."""
    tabulated_files = {
        'iron': 'eos_seager07_iron.txt',
        'MgSiO3': 'eos_seager07_silicate.txt',
        'H2O': 'eos_seager07_water.txt',
    }

    colors = {'iron': 'red', 'MgSiO3': 'orange', 'H2O': 'blue'}

    fig, ax = plt.subplots(figsize=(10, 5))

    data_folder = os.path.join(ZALMOXIS_ROOT, 'data', 'EOS_Seager2007')
    has_data = False

    for mat_key, filename in tabulated_files.items():
        filepath = os.path.join(data_folder, filename)
        if not os.path.exists(filepath):
            continue
        has_data = True

        data = np.loadtxt(filepath, delimiter=',', skiprows=1)
        pressures_pa = data[:, 1] * 1e9  # GPa -> Pa
        densities_tab = data[:, 0] * 1000  # g/cm^3 -> kg/m^3

        # Filter to valid range
        mask = pressures_pa > 0
        pressures_pa = pressures_pa[mask]
        densities_tab = densities_tab[mask]

        densities_analytic = np.array([get_analytic_density(p, mat_key) for p in pressures_pa])
        residuals = (densities_analytic - densities_tab) / densities_tab * 100

        ax.plot(
            pressures_pa / 1e9,
            residuals,
            color=colors[mat_key],
            linewidth=1.5,
            label=mat_key,
        )

    if not has_data:
        ax.text(
            0.5,
            0.5,
            'Tabulated data files not available',
            transform=ax.transAxes,
            ha='center',
            fontsize=14,
        )

    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')
    ax.set_xlabel('Pressure (GPa)', fontsize=13)
    ax.set_ylabel('Relative difference (%)', fontsize=13)
    ax.set_title('Analytic vs Tabulated EOS: Density Residuals', fontsize=14)
    ax.set_xscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, 'plot2_density_residuals.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')
    return path


def plot3_mass_radius():
    """Plot 3: Mass-radius comparison (analytic vs tabulated)."""
    from tools.setup_tests import load_model_output, load_zeng_curve, run_zalmoxis_rocky_water

    masses = [0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Try to load Zeng reference curves
    for zeng_file, label, color in [
        ('Zeng_earthlike_rocky.txt', 'Zeng+2019 Earth-like', 'gray'),
        ('Zeng_50percH2O.txt', 'Zeng+2019 50% H2O', 'lightblue'),
    ]:
        try:
            zeng_m, zeng_r = load_zeng_curve(zeng_file)
            ax.plot(zeng_m, zeng_r, color=color, linewidth=2, linestyle=':', label=label)
        except FileNotFoundError:
            pass

    # Analytic iron/MgSiO3 (rocky)
    analytic_rocky_m, analytic_rocky_r = [], []
    for m in masses:
        try:
            analytic_mats = {'core': 'iron', 'mantle': 'MgSiO3'}
            out, _ = run_zalmoxis_rocky_water(
                m,
                'rocky',
                cmf=0.325,
                immf=0,
                eos_override='Analytic:Seager2007',
                analytic_materials=analytic_mats,
            )
            mass_out, radius_out = load_model_output(out)
            analytic_rocky_m.append(mass_out)
            analytic_rocky_r.append(radius_out)
        except Exception as e:
            print(f'  Analytic rocky {m} M_earth failed: {e}')

    if analytic_rocky_m:
        ax.plot(
            analytic_rocky_m,
            analytic_rocky_r,
            'o-',
            color='red',
            linewidth=2,
            markersize=6,
            label='Analytic: iron/MgSiO3',
        )

    # Tabulated iron/silicate (rocky)
    tab_rocky_m, tab_rocky_r = [], []
    for m in masses:
        try:
            out, _ = run_zalmoxis_rocky_water(m, 'rocky', cmf=0.325, immf=0)
            mass_out, radius_out = load_model_output(out)
            tab_rocky_m.append(mass_out)
            tab_rocky_r.append(radius_out)
        except Exception as e:
            print(f'  Tabulated rocky {m} M_earth failed: {e}')

    if tab_rocky_m:
        ax.plot(
            tab_rocky_m,
            tab_rocky_r,
            's--',
            color='darkred',
            linewidth=2,
            markersize=6,
            label='Tabulated: iron/silicate',
        )

    # Analytic iron/MgSiO3/H2O (water)
    analytic_water_m, analytic_water_r = [], []
    for m in masses:
        try:
            analytic_mats = {'core': 'iron', 'mantle': 'MgSiO3', 'water_ice_layer': 'H2O'}
            out, _ = run_zalmoxis_rocky_water(
                m,
                'water',
                cmf=0.065,
                immf=0.485,
                eos_override='Analytic:Seager2007',
                analytic_materials=analytic_mats,
            )
            mass_out, radius_out = load_model_output(out)
            analytic_water_m.append(mass_out)
            analytic_water_r.append(radius_out)
        except Exception as e:
            print(f'  Analytic water {m} M_earth failed: {e}')

    if analytic_water_m:
        ax.plot(
            analytic_water_m,
            analytic_water_r,
            'o-',
            color='blue',
            linewidth=2,
            markersize=6,
            label='Analytic: iron/MgSiO3/H2O',
        )

    # Tabulated water
    tab_water_m, tab_water_r = [], []
    for m in masses:
        try:
            out, _ = run_zalmoxis_rocky_water(m, 'water', cmf=0.065, immf=0.485)
            mass_out, radius_out = load_model_output(out)
            tab_water_m.append(mass_out)
            tab_water_r.append(radius_out)
        except Exception as e:
            print(f'  Tabulated water {m} M_earth failed: {e}')

    if tab_water_m:
        ax.plot(
            tab_water_m,
            tab_water_r,
            's--',
            color='darkblue',
            linewidth=2,
            markersize=6,
            label='Tabulated: water',
        )

    # Analytic iron/SiC (carbon planet)
    analytic_sic_m, analytic_sic_r = [], []
    for m in masses:
        try:
            analytic_mats = {'core': 'iron', 'mantle': 'SiC'}
            out, _ = run_zalmoxis_rocky_water(
                m,
                'rocky',
                cmf=0.325,
                immf=0,
                eos_override='Analytic:Seager2007',
                analytic_materials=analytic_mats,
            )
            mass_out, radius_out = load_model_output(out)
            analytic_sic_m.append(mass_out)
            analytic_sic_r.append(radius_out)
        except Exception as e:
            print(f'  Analytic SiC {m} M_earth failed: {e}')

    if analytic_sic_m:
        ax.plot(
            analytic_sic_m,
            analytic_sic_r,
            'D-',
            color='green',
            linewidth=2,
            markersize=6,
            label='Analytic: iron/SiC (carbon)',
        )

    ax.set_xlabel('Mass (Earth masses)', fontsize=13)
    ax.set_ylabel('Radius (Earth radii)', fontsize=13)
    ax.set_title('Mass-Radius Relations: Analytic vs Tabulated EOS', fontsize=14)
    ax.set_xscale('log')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, 'plot3_mass_radius.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')
    return path


def plot4_density_profiles():
    """Plot 4: Interior density profiles for 1 M_earth."""
    from tools.setup_tests import load_profile_output, run_zalmoxis_rocky_water
    from zalmoxis.constants import earth_radius

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Earth-like rocky (iron/MgSiO3)
    ax = axes[0]
    try:
        analytic_mats = {'core': 'iron', 'mantle': 'MgSiO3'}
        _, profile_ana = run_zalmoxis_rocky_water(
            1,
            'rocky',
            cmf=0.325,
            immf=0,
            eos_override='Analytic:Seager2007',
            analytic_materials=analytic_mats,
        )
        r_ana, d_ana = load_profile_output(profile_ana)
        ax.plot(
            np.array(r_ana) / earth_radius,
            d_ana,
            color='red',
            linewidth=2,
            label='Analytic: iron/MgSiO3',
        )
    except Exception as e:
        print(f'  Analytic rocky profile failed: {e}')

    try:
        _, profile_tab = run_zalmoxis_rocky_water(1, 'rocky', cmf=0.325, immf=0)
        r_tab, d_tab = load_profile_output(profile_tab)
        ax.plot(
            np.array(r_tab) / earth_radius,
            d_tab,
            color='darkred',
            linewidth=2,
            linestyle='--',
            label='Tabulated: iron/silicate',
        )
    except Exception as e:
        print(f'  Tabulated rocky profile failed: {e}')

    ax.set_xlabel('Radius (Earth radii)', fontsize=12)
    ax.set_ylabel('Density (kg/m³)', fontsize=12)
    ax.set_title('1 $M_\\oplus$ Earth-like', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Right panel: Water planet (iron/MgSiO3/H2O)
    ax = axes[1]
    try:
        analytic_mats = {'core': 'iron', 'mantle': 'MgSiO3', 'water_ice_layer': 'H2O'}
        _, profile_ana = run_zalmoxis_rocky_water(
            1,
            'water',
            cmf=0.065,
            immf=0.485,
            eos_override='Analytic:Seager2007',
            analytic_materials=analytic_mats,
        )
        r_ana, d_ana = load_profile_output(profile_ana)
        ax.plot(
            np.array(r_ana) / earth_radius,
            d_ana,
            color='blue',
            linewidth=2,
            label='Analytic: iron/MgSiO3/H2O',
        )
    except Exception as e:
        print(f'  Analytic water profile failed: {e}')

    try:
        _, profile_tab = run_zalmoxis_rocky_water(1, 'water', cmf=0.065, immf=0.485)
        r_tab, d_tab = load_profile_output(profile_tab)
        ax.plot(
            np.array(r_tab) / earth_radius,
            d_tab,
            color='darkblue',
            linewidth=2,
            linestyle='--',
            label='Tabulated: water',
        )
    except Exception as e:
        print(f'  Tabulated water profile failed: {e}')

    ax.set_xlabel('Radius (Earth radii)', fontsize=12)
    ax.set_ylabel('Density (kg/m³)', fontsize=12)
    ax.set_title('1 $M_\\oplus$ Water World', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Interior Density Profiles: Analytic vs Tabulated', fontsize=14)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'plot4_density_profiles.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')
    return path


if __name__ == '__main__':
    print('Generating comparison plots...')
    print()

    print('Plot 1: EOS comparison (P vs rho)')
    plot1_eos_comparison()

    print('\nPlot 2: Density residuals')
    plot2_density_residuals()

    print('\nPlot 3: Mass-radius comparison')
    plot3_mass_radius()

    print('\nPlot 4: Interior density profiles')
    plot4_density_profiles()

    print(f'\nAll plots saved to: {OUTPUT_DIR}')
