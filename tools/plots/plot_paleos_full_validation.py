"""Validation plots for the full PALEOS EOS integration (iron + MgSiO3 + H2O).

Generates diagnostic plots for visual inspection:
1. T-P profiles on unified PALEOS table background with extracted liquidus
2. Density-pressure profiles comparing PALEOS vs Seager2007
3. Mass-radius comparison: PALEOS adiabatic vs Zeng+2019
4. Radial profiles (T, rho, P, g) for 1 M_earth: linear vs adiabatic
5. Phase diagram: which iron/MgSiO3 phases are sampled along the adiabat
6. mushy_zone_factor comparison: factor=1.0 vs 0.8

Usage:
    python -m src.tools.plot_paleos_full_validation
"""

from __future__ import annotations

import logging
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure ZALMOXIS_ROOT is set
ZALMOXIS_ROOT = os.environ.get('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    print('Error: ZALMOXIS_ROOT environment variable not set.')
    sys.exit(1)

from zalmoxis.constants import earth_mass, earth_radius  # noqa: E402
from zalmoxis.eos_functions import load_paleos_unified_table  # noqa: E402
from zalmoxis.zalmoxis import (  # noqa: E402
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
    main,
)

OUTPUT_DIR = os.path.join(ZALMOXIS_ROOT, 'output', 'paleos_full_validation')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _run_model(mass_earth, core_eos, mantle_eos, temperature_mode, mushy_zone_factor=1.0):
    """Run Zalmoxis with given EOS config."""
    root = os.environ['ZALMOXIS_ROOT']
    config_path = os.path.join(root, 'input', 'default.toml')
    config = load_zalmoxis_config(config_path)

    config['planet_mass'] = mass_earth * earth_mass
    config['layer_eos_config'] = {'core': core_eos, 'mantle': mantle_eos}
    config['temperature_mode'] = temperature_mode
    config['data_output_enabled'] = False
    config['plotting_enabled'] = False
    config['verbose'] = False
    config['mushy_zone_factor'] = mushy_zone_factor

    layer_eos_config = config['layer_eos_config']

    return main(
        config,
        material_dictionaries=load_material_dictionaries(),
        melting_curves_functions=load_solidus_liquidus_functions(
            layer_eos_config,
            config.get('rock_solidus', 'Stixrude14-solidus'),
            config.get('rock_liquidus', 'Stixrude14-liquidus'),
        ),
        input_dir=os.path.join(root, 'input'),
    )


def _check_data(eos_file):
    """Check if a data file exists."""
    return os.path.isfile(eos_file)


def plot_tp_profiles_on_table():
    """Plot T-P profiles overlaid on PALEOS table phase maps."""
    iron_file = os.path.join(
        ZALMOXIS_ROOT, 'data', 'EOS_PALEOS_iron', 'paleos_iron_eos_table_pt.dat'
    )
    mgsio3_file = os.path.join(
        ZALMOXIS_ROOT,
        'data',
        'EOS_PALEOS_MgSiO3_unified',
        'paleos_mgsio3_eos_table_pt.dat',
    )

    if not (_check_data(iron_file) and _check_data(mgsio3_file)):
        logger.warning('PALEOS unified data not found, skipping T-P plot.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, eos_file, title in [
        (axes[0], iron_file, 'PALEOS:iron'),
        (axes[1], mgsio3_file, 'PALEOS:MgSiO3'),
    ]:
        cache = load_paleos_unified_table(eos_file)

        # Plot density as background
        ulp = cache['unique_log_p']
        ult = np.linspace(cache['logt_valid_min'][0], cache['logt_valid_max'][-1], 200)
        PP, TT = np.meshgrid(ulp, ult, indexing='ij')
        pts = np.column_stack([PP.ravel(), TT.ravel()])
        rho = cache['density_interp'](pts).reshape(PP.shape)

        im = ax.pcolormesh(
            10.0**ulp / 1e9,
            10.0**ult,
            np.log10(rho).T,
            shading='auto',
            cmap='viridis',
            rasterized=True,
        )
        plt.colorbar(im, ax=ax, label='log10(rho [kg/m^3])')

        # Overlay extracted liquidus
        if len(cache['liquidus_log_p']) > 0:
            ax.plot(
                10.0 ** cache['liquidus_log_p'] / 1e9,
                10.0 ** cache['liquidus_log_t'],
                'r-',
                lw=2,
                label='Extracted liquidus',
            )

        ax.set_xlabel('Pressure [GPa]')
        ax.set_ylabel('Temperature [K]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(title)
        ax.legend()

    # Overlay adiabatic profiles
    for mass_earth in [1, 3, 5, 10]:
        try:
            result = _run_model(mass_earth, 'PALEOS:iron', 'PALEOS:MgSiO3', 'adiabatic')
            if not result['converged']:
                logger.warning(f'{mass_earth} M_earth did not converge, skipping.')
                continue
            P = result['pressure']
            T = result['temperature']
            cmb_mass = result['cmb_mass']
            M = result['mass_enclosed']

            # Core
            core_mask = M < cmb_mass
            if np.any(core_mask):
                axes[0].plot(
                    P[core_mask] / 1e9,
                    T[core_mask],
                    '-',
                    lw=1.5,
                    label=f'{mass_earth} M_E',
                )

            # Mantle
            mantle_mask = M >= cmb_mass
            if np.any(mantle_mask):
                axes[1].plot(
                    P[mantle_mask] / 1e9,
                    T[mantle_mask],
                    '-',
                    lw=1.5,
                    label=f'{mass_earth} M_E',
                )
        except Exception as e:
            logger.warning(f'{mass_earth} M_earth failed: {e}')

    for ax in axes:
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'tp_profiles_on_table.png'), dpi=150)
    plt.close(fig)
    logger.info('Saved tp_profiles_on_table.png')


def plot_density_pressure():
    """Plot density vs pressure comparing PALEOS to Seager2007."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for mass_earth in [1, 5, 10]:
        # PALEOS adiabatic
        try:
            result_paleos = _run_model(mass_earth, 'PALEOS:iron', 'PALEOS:MgSiO3', 'adiabatic')
            ax.plot(
                result_paleos['pressure'] / 1e9,
                result_paleos['density'],
                '-',
                lw=1.5,
                label=f'PALEOS adiabatic {mass_earth} M_E',
            )
        except Exception as e:
            logger.warning(f'PALEOS {mass_earth} M_earth failed: {e}')

        # Seager2007 isothermal reference
        try:
            result_seager = _run_model(
                mass_earth, 'Seager2007:iron', 'Seager2007:MgSiO3', 'isothermal'
            )
            ax.plot(
                result_seager['pressure'] / 1e9,
                result_seager['density'],
                '--',
                lw=1,
                alpha=0.7,
                label=f'Seager2007 300K {mass_earth} M_E',
            )
        except Exception as e:
            logger.warning(f'Seager2007 {mass_earth} M_earth failed: {e}')

    ax.set_xlabel('Pressure [GPa]')
    ax.set_ylabel('Density [kg/m^3]')
    ax.set_xscale('log')
    ax.set_title('Density vs Pressure: PALEOS adiabatic vs Seager2007')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'density_pressure.png'), dpi=150)
    plt.close(fig)
    logger.info('Saved density_pressure.png')


def plot_mass_radius():
    """Plot mass-radius relation comparing PALEOS to Zeng+2019."""
    masses = [0.5, 1, 2, 3, 5, 7, 10]
    radii_paleos = []
    radii_seager = []
    converged_masses_paleos = []
    converged_masses_seager = []

    for m in masses:
        try:
            result = _run_model(m, 'PALEOS:iron', 'PALEOS:MgSiO3', 'adiabatic')
            if result['converged']:
                radii_paleos.append(result['radii'][-1] / earth_radius)
                converged_masses_paleos.append(m)
        except Exception as e:
            logger.warning(f'PALEOS {m} M_earth failed: {e}')

        try:
            result = _run_model(m, 'Seager2007:iron', 'Seager2007:MgSiO3', 'isothermal')
            if result['converged']:
                radii_seager.append(result['radii'][-1] / earth_radius)
                converged_masses_seager.append(m)
        except Exception as e:
            logger.warning(f'Seager2007 {m} M_earth failed: {e}')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(converged_masses_paleos, radii_paleos, 'o-', label='PALEOS adiabatic')
    ax.plot(converged_masses_seager, radii_seager, 's--', label='Seager2007 300K')

    # Load Zeng+2019 if available
    zeng_file = os.path.join(
        ZALMOXIS_ROOT, 'data', 'mass_radius_curves', 'massradiusEarthlikeRocky.txt'
    )
    if os.path.isfile(zeng_file):
        zeng = np.loadtxt(zeng_file)
        ax.plot(zeng[:, 0], zeng[:, 1], 'k-', lw=2, alpha=0.5, label='Zeng+2019 rocky')

    ax.set_xlabel('Mass [M_earth]')
    ax.set_ylabel('Radius [R_earth]')
    ax.set_title('Mass-Radius: PALEOS adiabatic vs Seager2007 vs Zeng+2019')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'mass_radius.png'), dpi=150)
    plt.close(fig)
    logger.info('Saved mass_radius.png')


def plot_radial_profiles():
    """Plot radial profiles (T, rho, P, g) for 1 M_earth: linear vs adiabatic."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for mode, ls in [('linear', '--'), ('adiabatic', '-')]:
        try:
            result = _run_model(1.0, 'PALEOS:iron', 'PALEOS:MgSiO3', mode)
            r = result['radii'] / 1e6  # km

            axes[0, 0].plot(r, result['temperature'], ls, label=mode)
            axes[0, 0].set_ylabel('Temperature [K]')

            axes[0, 1].plot(r, result['density'], ls, label=mode)
            axes[0, 1].set_ylabel('Density [kg/m^3]')

            axes[1, 0].plot(r, result['pressure'] / 1e9, ls, label=mode)
            axes[1, 0].set_ylabel('Pressure [GPa]')

            axes[1, 1].plot(r, result['gravity'], ls, label=mode)
            axes[1, 1].set_ylabel('Gravity [m/s^2]')
        except Exception as e:
            logger.warning(f'1 M_earth {mode} failed: {e}')

    for ax in axes.flat:
        ax.set_xlabel('Radius [km]')
        ax.legend()

    fig.suptitle('1 M_earth radial profiles: PALEOS:iron + PALEOS:MgSiO3')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'radial_profiles_1ME.png'), dpi=150)
    plt.close(fig)
    logger.info('Saved radial_profiles_1ME.png')


def plot_mushy_zone_comparison():
    """Compare profiles at mushy_zone_factor=1.0 vs 0.8 for 1 M_earth."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for factor, ls in [(1.0, '-'), (0.8, '--')]:
        try:
            result = _run_model(
                1.0, 'PALEOS:iron', 'PALEOS:MgSiO3', 'adiabatic', mushy_zone_factor=factor
            )
            r = result['radii'] / 1e6

            axes[0].plot(r, result['temperature'], ls, label=f'f={factor}')
            axes[0].set_ylabel('Temperature [K]')

            axes[1].plot(r, result['density'], ls, label=f'f={factor}')
            axes[1].set_ylabel('Density [kg/m^3]')

            axes[2].plot(r, result['pressure'] / 1e9, ls, label=f'f={factor}')
            axes[2].set_ylabel('Pressure [GPa]')
        except Exception as e:
            logger.warning(f'mushy_zone_factor={factor} failed: {e}')

    for ax in axes:
        ax.set_xlabel('Radius [km]')
        ax.legend()

    fig.suptitle('1 M_earth: mushy_zone_factor comparison (PALEOS adiabatic)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'mushy_zone_comparison.png'), dpi=150)
    plt.close(fig)
    logger.info('Saved mushy_zone_comparison.png')


if __name__ == '__main__':
    logger.info(f'Output directory: {OUTPUT_DIR}')

    logger.info('=== Plot 1: T-P profiles on table ===')
    plot_tp_profiles_on_table()

    logger.info('=== Plot 2: Density vs Pressure ===')
    plot_density_pressure()

    logger.info('=== Plot 3: Mass-Radius ===')
    plot_mass_radius()

    logger.info('=== Plot 4: Radial profiles ===')
    plot_radial_profiles()

    logger.info('=== Plot 5: Mushy zone comparison ===')
    plot_mushy_zone_comparison()

    logger.info('All validation plots generated.')
