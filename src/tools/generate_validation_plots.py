"""
Generate physical validity plots for the per-layer EOS implementation.

Produces 3 plots:
5. Physical validity panel — density, pressure, gravity, mass profiles for 1 M_earth
   Earth-like planet, with monotonicity and range checks annotated.
6. Mixed EOS comparison — density profiles for 4 EOS combinations overlaid:
   pure tabulated, pure analytic, tab core + analytic mantle, analytic core + tab mantle.
7. Layer boundary verification — density profile color-coded by active layer EOS,
   confirming per-layer dispatch at boundaries.

Saves all plots as PDF to output_files/.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from zalmoxis.constants import earth_mass, earth_radius

ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')

OUTPUT_DIR = os.path.join(ZALMOXIS_ROOT, 'output_files')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def run_model(layer_eos, cmf=0.325, immf=0, config_type='rocky'):
    """Run a 1 M_earth model and return the full profile arrays.

    Returns
    -------
    dict
        Keys: radii, density, gravity, pressure, temperature, mass_enclosed
        All as numpy arrays.
    """
    from src.zalmoxis import zalmoxis
    from src.zalmoxis.zalmoxis import load_solidus_liquidus_functions

    default_config_path = os.path.join(ZALMOXIS_ROOT, 'input', 'default.toml')
    config_params = zalmoxis.load_zalmoxis_config(default_config_path)
    config_params['planet_mass'] = 1.0 * earth_mass
    config_params['core_mass_fraction'] = cmf
    config_params['mantle_mass_fraction'] = immf
    config_params['weight_iron_fraction'] = cmf
    config_params['layer_eos_config'] = layer_eos

    model_results = zalmoxis.main(
        config_params,
        material_dictionaries=zalmoxis.load_material_dictionaries(),
        melting_curves_functions=load_solidus_liquidus_functions(layer_eos),
        input_dir=os.path.join(ZALMOXIS_ROOT, 'input'),
    )

    return {
        'radii': np.array(model_results['radii']),
        'density': np.array(model_results['density']),
        'gravity': np.array(model_results['gravity']),
        'pressure': np.array(model_results['pressure']),
        'temperature': np.array(model_results['temperature']),
        'mass_enclosed': np.array(model_results['mass_enclosed']),
    }


def check_monotonic(arr, direction='decreasing'):
    """Check if array is monotonically increasing or decreasing.

    Parameters
    ----------
    arr : array-like
        Array to check.
    direction : str
        'increasing' or 'decreasing'.

    Returns
    -------
    bool
        True if monotonic in the given direction.
    """
    diff = np.diff(arr)
    if direction == 'decreasing':
        return np.all(diff <= 0)
    return np.all(diff >= 0)


def plot5_physical_validity():
    """Plot 5: Physical validity panel for a 1 M_earth Earth-like planet."""
    print('  Running tabulated iron/silicate model...')
    prof = run_model({'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'})

    r_re = prof['radii'] / earth_radius

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Density
    ax = axes[0, 0]
    ax.plot(r_re, prof['density'], 'k-', linewidth=1.5)
    ax.set_ylabel('Density (kg/m$^3$)')
    ax.set_title('Density profile')
    ax.grid(True, alpha=0.3)
    # Annotate range check
    rho_center = prof['density'][0]
    rho_surface = prof['density'][-1]
    ax.axhspan(12000, 14000, alpha=0.1, color='red', label=f'Fe expected range')
    ax.axhspan(4000, 5500, alpha=0.1, color='orange', label='MgSiO$_3$ expected range')
    ax.legend(fontsize=8, loc='upper right')
    mono_ok = '(monotonic)' if check_monotonic(prof['density']) else '(NOT monotonic!)'
    ax.text(0.02, 0.95, f'Center: {rho_center:.0f} kg/m$^3$\nSurface: {rho_surface:.0f} kg/m$^3$\n{mono_ok}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Pressure
    ax = axes[0, 1]
    ax.plot(r_re, prof['pressure'] / 1e9, 'k-', linewidth=1.5)
    ax.set_ylabel('Pressure (GPa)')
    ax.set_title('Pressure profile')
    ax.grid(True, alpha=0.3)
    p_center = prof['pressure'][0] / 1e9
    p_surface = prof['pressure'][-1] / 1e9
    mono_ok = '(monotonic)' if check_monotonic(prof['pressure']) else '(NOT monotonic!)'
    ax.text(0.02, 0.95, f'Center: {p_center:.1f} GPa\nSurface: {p_surface:.4f} GPa\n{mono_ok}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Gravity
    ax = axes[1, 0]
    ax.plot(r_re, prof['gravity'], 'k-', linewidth=1.5)
    ax.set_ylabel('Gravity (m/s$^2$)')
    ax.set_xlabel('Radius ($R_\\oplus$)')
    ax.set_title('Gravity profile')
    ax.grid(True, alpha=0.3)
    g_max = np.max(prof['gravity'])
    g_surface = prof['gravity'][-1]
    all_nonneg = np.all(prof['gravity'] >= 0)
    ax.text(0.02, 0.95, f'Max: {g_max:.1f} m/s$^2$\nSurface: {g_surface:.1f} m/s$^2$\nAll non-negative: {all_nonneg}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Enclosed mass
    ax = axes[1, 1]
    ax.plot(r_re, prof['mass_enclosed'] / earth_mass, 'k-', linewidth=1.5)
    ax.set_ylabel('Enclosed mass ($M_\\oplus$)')
    ax.set_xlabel('Radius ($R_\\oplus$)')
    ax.set_title('Enclosed mass profile')
    ax.grid(True, alpha=0.3)
    mono_ok = '(monotonic)' if check_monotonic(prof['mass_enclosed'], 'increasing') else '(NOT monotonic!)'
    m_total = prof['mass_enclosed'][-1] / earth_mass
    ax.text(0.02, 0.95, f'Total: {m_total:.4f} $M_\\oplus$\n{mono_ok}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.suptitle('Physical Validity: 1 $M_\\oplus$ Earth-like (Seager2007:iron + Seager2007:MgSiO$_3$)', fontsize=14)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'plot5_physical_validity.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


def plot6_mixed_eos_comparison():
    """Plot 6: Mixed EOS density profile comparison for 1 M_earth."""
    configs = {
        'Pure tabulated': {
            'eos': {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'},
            'color': 'black', 'ls': '-', 'lw': 2.5,
        },
        'Pure analytic': {
            'eos': {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'},
            'color': 'red', 'ls': '-', 'lw': 2,
        },
        'Tab core + analytic mantle': {
            'eos': {'core': 'Seager2007:iron', 'mantle': 'Analytic:MgSiO3'},
            'color': 'blue', 'ls': '--', 'lw': 2,
        },
        'Analytic core + tab mantle': {
            'eos': {'core': 'Analytic:iron', 'mantle': 'Seager2007:MgSiO3'},
            'color': 'green', 'ls': '--', 'lw': 2,
        },
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    results = {}
    for label, cfg in configs.items():
        print(f'  Running {label}...')
        prof = run_model(cfg['eos'])
        results[label] = prof
        r_re = prof['radii'] / earth_radius
        ax.plot(r_re, prof['density'], color=cfg['color'], linestyle=cfg['ls'],
                linewidth=cfg['lw'], label=label)

    ax.set_xlabel('Radius ($R_\\oplus$)', fontsize=13)
    ax.set_ylabel('Density (kg/m$^3$)', fontsize=13)
    ax.set_title('Mixed EOS Comparison: 1 $M_\\oplus$ Earth-like Density Profiles', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # Annotate total radius for each
    text_lines = []
    for label, prof in results.items():
        r_total = prof['radii'][-1] / earth_radius
        text_lines.append(f'{label}: R = {r_total:.4f} $R_\\oplus$')
    ax.text(0.98, 0.95, '\n'.join(text_lines),
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    path = os.path.join(OUTPUT_DIR, 'plot6_mixed_eos_comparison.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


def plot7_layer_boundary():
    """Plot 7: Layer boundary verification with color-coded EOS regions."""
    # Run a 3-layer mixed model: tab iron core + analytic silicate mantle + analytic water
    print('  Running 3-layer mixed model...')
    prof = run_model(
        {'core': 'Seager2007:iron', 'mantle': 'Analytic:MgSiO3', 'ice_layer': 'Analytic:H2O'},
        cmf=0.13, immf=0.485, config_type='water',
    )

    r_re = prof['radii'] / earth_radius
    density = prof['density']
    mass = prof['mass_enclosed']

    # Determine layer boundaries from mass fractions
    # cmf=0.13, mantle = 1 - 0.13 - 0.485 = 0.385, so core-mantle at 0.13*M_total
    m_total = mass[-1]
    cmb_mass = 0.13 * m_total
    core_mantle_mass = (1 - 0.485) * m_total  # core + mantle mass

    # Find boundary indices
    i_cmb = np.searchsorted(mass, cmb_mass)
    i_ice = np.searchsorted(mass, core_mantle_mass)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1]})

    # Top panel: density profile with color-coded regions
    ax1.fill_between(r_re[:i_cmb+1], 0, density[:i_cmb+1],
                     alpha=0.2, color='red', label='Core (Seager2007:iron)')
    ax1.fill_between(r_re[i_cmb:i_ice+1], 0, density[i_cmb:i_ice+1],
                     alpha=0.2, color='orange', label='Mantle (Analytic:MgSiO$_3$)')
    ax1.fill_between(r_re[i_ice:], 0, density[i_ice:],
                     alpha=0.2, color='blue', label='Ice layer (Analytic:H$_2$O)')
    ax1.plot(r_re, density, 'k-', linewidth=1.5)

    # Mark boundaries
    if i_cmb < len(r_re):
        ax1.axvline(r_re[i_cmb], color='red', linestyle=':', alpha=0.7, linewidth=1)
        ax1.text(r_re[i_cmb], ax1.get_ylim()[1]*0.9, ' CMB', fontsize=9, color='red')
    if i_ice < len(r_re):
        ax1.axvline(r_re[i_ice], color='blue', linestyle=':', alpha=0.7, linewidth=1)
        ax1.text(r_re[i_ice], ax1.get_ylim()[1]*0.85, ' Ice boundary', fontsize=9, color='blue')

    ax1.set_ylabel('Density (kg/m$^3$)', fontsize=13)
    ax1.set_title('Layer Boundary Verification: 1 $M_\\oplus$ 3-Layer Mixed EOS', fontsize=14)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Bottom panel: pressure profile
    ax2.plot(r_re, prof['pressure'] / 1e9, 'k-', linewidth=1.5)
    if i_cmb < len(r_re):
        ax2.axvline(r_re[i_cmb], color='red', linestyle=':', alpha=0.7, linewidth=1)
    if i_ice < len(r_re):
        ax2.axvline(r_re[i_ice], color='blue', linestyle=':', alpha=0.7, linewidth=1)
    ax2.set_xlabel('Radius ($R_\\oplus$)', fontsize=13)
    ax2.set_ylabel('Pressure (GPa)', fontsize=13)
    ax2.set_title('Pressure profile (continuous across boundaries)', fontsize=12)
    ax2.grid(True, alpha=0.3)

    r_total = r_re[-1]
    ax1.text(0.02, 0.95, f'Total radius: {r_total:.4f} $R_\\oplus$',
             transform=ax1.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'plot7_layer_boundary.pdf')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


if __name__ == '__main__':
    print('Generating physical validity plots...')
    print()

    print('Plot 5: Physical validity (monotonicity + range checks)')
    plot5_physical_validity()

    print('\nPlot 6: Mixed EOS density profile comparison')
    plot6_mixed_eos_comparison()

    print('\nPlot 7: Layer boundary verification (3-layer mixed)')
    plot7_layer_boundary()

    print(f'\nAll plots saved to: {OUTPUT_DIR}')
