"""Comprehensive benchmark and verification script for PALEOS EOS integration.

Runs Zalmoxis simulations across multiple EOS configurations, masses,
and mushy zone factors, then produces comparison plots and runtime
documentation.

Usage
-----
    python -m src.tests.benchmark_paleos_eos

Output is written to ``output/benchmark_paleos/``.

Plots generated
---------------
1. T-P phase diagram comparison (1 M_earth, all EOS + melting curves)
2. Density-pressure comparison (1 and 5 M_earth, all EOS)
3. Radial profiles (T, rho, P, g) multi-panel (1 M_earth, all EOS)
4. Mass-radius diagram (all EOS, with Zeng+2019 reference)
5. Mushy zone factor effect (1 M_earth, factors 1.0/0.9/0.8/0.7)
6. Phase regime visualization (1 M_earth PALEOS unified)
7. Runtime comparison bar chart

See also
--------
- ``src/tests/plot_paleos_full_validation.py``: earlier validation script
- ``docs/Reference/data.md``: EOS data file inventory
"""

from __future__ import annotations

import logging
import os
import sys
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('benchmark_paleos')

# Suppress verbose EOS loader messages
logging.getLogger('zalmoxis.eos_functions').setLevel(logging.WARNING)
logging.getLogger('zalmoxis.zalmoxis').setLevel(logging.WARNING)

# ── Ensure ZALMOXIS_ROOT is set ──────────────────────────────────────
ZALMOXIS_ROOT = os.environ.get('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    print('Error: ZALMOXIS_ROOT environment variable not set.')
    sys.exit(1)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from zalmoxis.constants import earth_mass, earth_radius  # noqa: E402
from zalmoxis.eos_functions import load_paleos_unified_table  # noqa: E402
from zalmoxis.melting_curves import (  # noqa: E402
    monteux16_liquidus,
    monteux16_solidus,
)
from zalmoxis.zalmoxis import (  # noqa: E402
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
    main,
)

# ── Constants ────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(ZALMOXIS_ROOT, 'output', 'benchmark_paleos')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MASSES = [1.0, 3.0, 5.0, 10.0]

# (label, core_eos, mantle_eos, temperature_mode, T_surface)
EOS_CONFIGS = [
    (
        'PALEOS unified',
        'PALEOS:iron',
        'PALEOS:MgSiO3',
        'adiabatic',
        3000.0,
    ),
    (
        'PALEOS-2phase',
        'Seager2007:iron',
        'PALEOS-2phase:MgSiO3',
        'adiabatic',
        3000.0,
    ),
    (
        'WolfBower2018',
        'Seager2007:iron',
        'WolfBower2018:MgSiO3',
        'adiabatic',
        3000.0,
    ),
    (
        'Seager2007 cold',
        'Seager2007:iron',
        'Seager2007:MgSiO3',
        'isothermal',
        300.0,
    ),
]

MUSHY_FACTORS = [1.0, 0.9, 0.8, 0.7]
MUSHY_MASSES = [1.0, 5.0]

# WolfBower2018 is limited to <= 7 M_earth
WB2018_MAX_MASS = 7.0

# Colors for EOS labels (consistent across plots)
EOS_COLORS = {
    'PALEOS unified': 'C0',
    'PALEOS-2phase': 'C1',
    'WolfBower2018': 'C2',
    'Seager2007 cold': 'C3',
}

# Line styles for mushy zone factors
MUSHY_STYLES = {
    1.0: '-',
    0.9: '--',
    0.8: '-.',
    0.7: ':',
}


# ── Helpers ──────────────────────────────────────────────────────────


def _check_eos_files(core_eos, mantle_eos, ice_eos=''):
    """Return True if the data files for the requested EOS exist."""
    from zalmoxis.eos_properties import EOS_REGISTRY

    for eos_name in [core_eos, mantle_eos, ice_eos]:
        if not eos_name:
            continue
        entry = EOS_REGISTRY.get(eos_name)
        if entry is None:
            return False
        # Walk through the entry dict looking for eos_file keys
        files_to_check = []
        if 'eos_file' in entry:
            files_to_check.append(entry['eos_file'])
        for sub_key in ('core', 'mantle', 'ice_layer', 'melted_mantle', 'solid_mantle'):
            sub = entry.get(sub_key, {})
            if isinstance(sub, dict) and 'eos_file' in sub:
                files_to_check.append(sub['eos_file'])
        for f in files_to_check:
            if not os.path.isfile(f):
                return False
    return True


def _run_model(
    mass_earth,
    core_eos,
    mantle_eos,
    temperature_mode,
    surface_temperature=3000.0,
    mushy_zone_factor=1.0,
    ice_eos='',
    core_mass_fraction=0.325,
    mantle_mass_fraction=0.0,
):
    """Run Zalmoxis and return model_results dict.

    Parameters
    ----------
    mass_earth : float
        Planet mass in Earth masses.
    core_eos, mantle_eos : str
        EOS identifier strings for core and mantle.
    temperature_mode : str
        ``'isothermal'``, ``'linear'``, or ``'adiabatic'``.
    surface_temperature : float
        Surface temperature in K.
    mushy_zone_factor : float
        Mushy zone width factor (1.0 = no mushy zone).
    ice_eos : str
        EOS identifier for ice/water layer (empty = 2-layer).
    core_mass_fraction : float
        Core mass fraction.
    mantle_mass_fraction : float
        Mantle mass fraction (0 = fills remainder).

    Returns
    -------
    dict
        Model results from ``main()``.
    """
    config_path = os.path.join(ZALMOXIS_ROOT, 'input', 'default.toml')
    config = load_zalmoxis_config(config_path)

    config['planet_mass'] = mass_earth * earth_mass
    config['temperature_mode'] = temperature_mode
    config['surface_temperature'] = surface_temperature
    config['data_output_enabled'] = False
    config['plotting_enabled'] = False
    config['verbose'] = False
    config['mushy_zone_factor'] = mushy_zone_factor
    config['core_mass_fraction'] = core_mass_fraction
    config['mantle_mass_fraction'] = mantle_mass_fraction

    layer_eos = {'core': core_eos, 'mantle': mantle_eos}
    if ice_eos:
        layer_eos['ice_layer'] = ice_eos
    config['layer_eos_config'] = layer_eos

    melting_funcs = load_solidus_liquidus_functions(
        layer_eos,
        config.get('rock_solidus', 'Stixrude14-solidus'),
        config.get('rock_liquidus', 'Stixrude14-liquidus'),
    )

    return main(
        config,
        material_dictionaries=load_material_dictionaries(),
        melting_curves_functions=melting_funcs,
        input_dir=os.path.join(ZALMOXIS_ROOT, 'input'),
    )


def _timed_run(label, **kwargs):
    """Run _run_model and return (result, wall_seconds).

    If the run fails, logs the error and returns (None, wall_seconds).
    """
    t0 = time.perf_counter()
    try:
        result = _run_model(**kwargs)
        dt = time.perf_counter() - t0
        converged = result.get('converged', False)
        tag = 'OK' if converged else 'DID NOT CONVERGE'
        logger.info(f'  {label}: {dt:.2f}s [{tag}]')
        return result, dt
    except Exception as exc:
        dt = time.perf_counter() - t0
        logger.warning(f'  {label}: FAILED after {dt:.2f}s -- {exc}')
        return None, dt


# ── Run all configurations ───────────────────────────────────────────


def run_all_configs():
    """Execute all benchmark configurations.

    Returns
    -------
    dict
        Nested dict: results[label][mass] = (model_results, wall_time)
    dict
        Mushy zone results: mushy_results[mass][factor] = (result, dt)
    dict or None
        3-layer result: {"result": ..., "dt": ...} or None
    """
    results = {}

    logger.info('=' * 60)
    logger.info('Running main EOS configurations')
    logger.info('=' * 60)

    for label, core_eos, mantle_eos, tmode, t_surf in EOS_CONFIGS:
        if not _check_eos_files(core_eos, mantle_eos):
            logger.warning(f'Skipping {label}: data files not found.')
            continue

        results[label] = {}
        for mass in MASSES:
            # WolfBower2018 only up to 7 M_earth
            if label == 'WolfBower2018' and mass > WB2018_MAX_MASS:
                logger.info(f'  Skipping {label} at {mass} M_earth (max {WB2018_MAX_MASS})')
                continue

            run_label = f'{label} {mass:.0f}ME'
            res, dt = _timed_run(
                run_label,
                mass_earth=mass,
                core_eos=core_eos,
                mantle_eos=mantle_eos,
                temperature_mode=tmode,
                surface_temperature=t_surf,
            )
            results[label][mass] = (res, dt)

    # ── Mushy zone factor variations ─────────────────────────────────
    logger.info('=' * 60)
    logger.info('Running mushy zone factor variations')
    logger.info('=' * 60)

    mushy_results = {}
    paleos_ok = _check_eos_files('PALEOS:iron', 'PALEOS:MgSiO3')
    if paleos_ok:
        for mass in MUSHY_MASSES:
            mushy_results[mass] = {}
            for factor in MUSHY_FACTORS:
                run_label = f'PALEOS mushy f={factor:.1f} {mass:.0f}ME'
                res, dt = _timed_run(
                    run_label,
                    mass_earth=mass,
                    core_eos='PALEOS:iron',
                    mantle_eos='PALEOS:MgSiO3',
                    temperature_mode='adiabatic',
                    surface_temperature=3000.0,
                    mushy_zone_factor=factor,
                )
                mushy_results[mass][factor] = (res, dt)
    else:
        logger.warning('Skipping mushy zone tests: PALEOS unified data not found.')

    # ── 3-layer model ────────────────────────────────────────────────
    logger.info('=' * 60)
    logger.info('Running 3-layer model (iron + MgSiO3 + H2O)')
    logger.info('=' * 60)

    three_layer = None
    if _check_eos_files('PALEOS:iron', 'PALEOS:MgSiO3', 'PALEOS:H2O'):
        res, dt = _timed_run(
            '3-layer 1ME',
            mass_earth=1.0,
            core_eos='PALEOS:iron',
            mantle_eos='PALEOS:MgSiO3',
            temperature_mode='adiabatic',
            surface_temperature=3000.0,
            ice_eos='PALEOS:H2O',
            core_mass_fraction=0.25,
            mantle_mass_fraction=0.50,
        )
        three_layer = {'result': res, 'dt': dt}
    else:
        logger.warning('Skipping 3-layer model: PALEOS:H2O data not found.')

    return results, mushy_results, three_layer


# ── Plot 1: T-P phase diagram comparison ────────────────────────────


def plot_tp_phase_diagram(results):
    """T-P profiles from all EOS at 1 M_earth with melting curves."""
    fig, ax = plt.subplots(figsize=(9, 7))

    # Melting curves (Monteux16) as background reference
    P_range = np.logspace(8, 12.5, 500)  # 0.1 GPa to ~300 GPa
    ax.plot(
        P_range / 1e9,
        monteux16_solidus(P_range),
        'k--',
        lw=1.5,
        alpha=0.6,
        label='Monteux16 solidus',
    )
    ax.plot(
        P_range / 1e9,
        monteux16_liquidus(P_range),
        'k-.',
        lw=1.5,
        alpha=0.6,
        label='Monteux16 liquidus',
    )

    for label in EOS_CONFIGS:
        eos_label = label[0]
        if eos_label not in results:
            continue
        entry = results[eos_label].get(1.0)
        if entry is None or entry[0] is None:
            continue
        res = entry[0]
        P = res['pressure']
        T = res['temperature']
        valid = P > 0
        ax.plot(
            P[valid] / 1e9,
            T[valid],
            color=EOS_COLORS[eos_label],
            lw=2,
            label=eos_label,
        )

    ax.set_xlabel('Pressure [GPa]')
    ax.set_ylabel('Temperature [K]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('T-P profiles at 1 M_earth')
    ax.legend(fontsize=8, loc='upper left')
    ax.set_xlim(0.1, 500)
    ax.set_ylim(200, 15000)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, '01_tp_phase_diagram.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f'Saved {path}')


# ── Plot 2: Density-pressure comparison ─────────────────────────────


def plot_density_pressure(results):
    """Density vs pressure for 1 and 5 M_earth, all EOS."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, mass in zip(axes, [1.0, 5.0]):
        for label in EOS_CONFIGS:
            eos_label = label[0]
            if eos_label not in results:
                continue
            entry = results[eos_label].get(mass)
            if entry is None or entry[0] is None:
                continue
            res = entry[0]
            P = res['pressure']
            rho = res['density']
            valid = P > 0
            ax.plot(
                P[valid] / 1e9,
                rho[valid],
                color=EOS_COLORS[eos_label],
                lw=1.5,
                label=eos_label,
            )

        ax.set_xlabel('Pressure [GPa]')
        ax.set_ylabel('Density [kg/m^3]')
        ax.set_xscale('log')
        ax.set_title(f'{mass:.0f} M_earth')
        ax.legend(fontsize=8)

    fig.suptitle('Density vs Pressure comparison')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, '02_density_pressure.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f'Saved {path}')


# ── Plot 3: Radial profiles multi-panel ─────────────────────────────


def plot_radial_profiles(results):
    """4-panel radial profiles (T, rho, P, g) at 1 M_earth."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    labels_and_units = [
        ('temperature', 'Temperature [K]'),
        ('density', 'Density [kg/m^3]'),
        ('pressure', 'Pressure [GPa]'),
        ('gravity', 'Gravity [m/s^2]'),
    ]

    for label in EOS_CONFIGS:
        eos_label = label[0]
        if eos_label not in results:
            continue
        entry = results[eos_label].get(1.0)
        if entry is None or entry[0] is None:
            continue
        res = entry[0]
        r_km = res['radii'] / 1e6

        for ax, (key, ylabel) in zip(axes.flat, labels_and_units):
            vals = res[key]
            if key == 'pressure':
                vals = vals / 1e9
            ax.plot(
                r_km,
                vals,
                color=EOS_COLORS[eos_label],
                lw=1.5,
                label=eos_label,
            )

    for ax, (_, ylabel) in zip(axes.flat, labels_and_units):
        ax.set_xlabel('Radius [1000 km]')
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)

    fig.suptitle('Radial profiles at 1 M_earth')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, '03_radial_profiles.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f'Saved {path}')


# ── Plot 4: Mass-radius diagram ─────────────────────────────────────


def plot_mass_radius(results):
    """R/R_earth vs M/M_earth for all EOS, with Zeng+2019."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for label in EOS_CONFIGS:
        eos_label = label[0]
        if eos_label not in results:
            continue
        masses_plot = []
        radii_plot = []
        for mass in sorted(results[eos_label].keys()):
            entry = results[eos_label][mass]
            if entry[0] is None:
                continue
            res = entry[0]
            if not res.get('converged', False):
                continue
            masses_plot.append(mass)
            radii_plot.append(res['radii'][-1] / earth_radius)

        if masses_plot:
            ax.plot(
                masses_plot,
                radii_plot,
                'o-',
                color=EOS_COLORS[eos_label],
                lw=1.5,
                label=eos_label,
            )

    # Zeng+2019 reference
    zeng_file = os.path.join(
        ZALMOXIS_ROOT,
        'data',
        'mass_radius_curves',
        'massradiusEarthlikeRocky.txt',
    )
    if os.path.isfile(zeng_file):
        zeng = np.loadtxt(zeng_file)
        ax.plot(
            zeng[:, 0],
            zeng[:, 1],
            'k-',
            lw=2,
            alpha=0.5,
            label='Zeng+2019 rocky',
        )

    ax.set_xlabel('Mass [M_earth]')
    ax.set_ylabel('Radius [R_earth]')
    ax.set_title('Mass-Radius diagram')
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, '04_mass_radius.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f'Saved {path}')


# ── Plot 5: Mushy zone factor effect ────────────────────────────────


def plot_mushy_zone(mushy_results):
    """3-panel (T, rho, P) vs radius for different mushy_zone_factor."""
    if not mushy_results:
        logger.warning('No mushy zone results, skipping plot 5.')
        return

    for mass in MUSHY_MASSES:
        if mass not in mushy_results:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        keys_units = [
            ('temperature', 'Temperature [K]'),
            ('density', 'Density [kg/m^3]'),
            ('pressure', 'Pressure [GPa]'),
        ]

        for factor in MUSHY_FACTORS:
            entry = mushy_results[mass].get(factor)
            if entry is None or entry[0] is None:
                continue
            res = entry[0]
            r_km = res['radii'] / 1e6
            ls = MUSHY_STYLES[factor]

            for ax, (key, _) in zip(axes, keys_units):
                vals = res[key]
                if key == 'pressure':
                    vals = vals / 1e9
                ax.plot(
                    r_km,
                    vals,
                    ls,
                    lw=1.5,
                    label=f'f={factor:.1f}',
                )

        for ax, (_, ylabel) in zip(axes, keys_units):
            ax.set_xlabel('Radius [1000 km]')
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)

        fig.suptitle(
            f'Mushy zone factor comparison, {mass:.0f} M_earth (PALEOS unified adiabatic)'
        )
        fig.tight_layout()
        path = os.path.join(
            OUTPUT_DIR,
            f'05_mushy_zone_{mass:.0f}ME.png',
        )
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f'Saved {path}')


# ── Plot 6: Phase regime visualization ──────────────────────────────


def plot_phase_regime(results):
    """Show which phase the adiabat samples at each radius.

    Loads the unified PALEOS phase_grid for iron and MgSiO3 and
    determines the stable phase at each (P, T) along the converged
    1 M_earth adiabatic profile.
    """
    entry = results.get('PALEOS unified', {}).get(1.0)
    if entry is None or entry[0] is None:
        logger.warning('No PALEOS unified 1 M_earth result, skipping plot 6.')
        return

    res = entry[0]
    P_profile = res['pressure']
    T_profile = res['temperature']
    r_profile = res['radii']
    cmb_mass = res['cmb_mass']
    M_profile = res['mass_enclosed']

    iron_file = os.path.join(
        ZALMOXIS_ROOT,
        'data',
        'EOS_PALEOS_iron',
        'paleos_iron_eos_table_pt.dat',
    )
    mgsio3_file = os.path.join(
        ZALMOXIS_ROOT,
        'data',
        'EOS_PALEOS_MgSiO3_unified',
        'paleos_mgsio3_eos_table_pt.dat',
    )

    if not (os.path.isfile(iron_file) and os.path.isfile(mgsio3_file)):
        logger.warning('PALEOS unified table files not found, skipping plot 6.')
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, eos_file, title, layer_mask in [
        (
            axes[0],
            iron_file,
            'Iron core phases',
            M_profile < cmb_mass,
        ),
        (
            axes[1],
            mgsio3_file,
            'MgSiO3 mantle phases',
            M_profile >= cmb_mass,
        ),
    ]:
        cache = load_paleos_unified_table(eos_file)
        phase_grid = cache['phase_grid']
        unique_log_p = cache['unique_log_p']
        # Recover unique_log_t from the interpolator grid
        unique_log_t = cache['density_interp'].grid[1]

        P_layer = P_profile[layer_mask]
        T_layer = T_profile[layer_mask]
        r_layer = r_profile[layer_mask] / 1e6  # 1000 km

        # Look up phase at each (P, T) along the profile
        phases = []
        for p_val, t_val in zip(P_layer, T_layer):
            if p_val <= 0 or t_val <= 0:
                phases.append('N/A')
                continue
            lp = np.log10(p_val)
            lt = np.log10(t_val)
            ip = np.argmin(np.abs(unique_log_p - lp))
            it = np.argmin(np.abs(unique_log_t - lt))
            phase = phase_grid[ip, it]
            phases.append(phase if phase else 'N/A')

        # Map phase strings to integers for coloring
        unique_phases = sorted(set(phases) - {'N/A', ''})
        phase_to_int = {ph: i for i, ph in enumerate(unique_phases)}
        phase_to_int['N/A'] = -1
        phase_to_int[''] = -1
        phase_ints = np.array([phase_to_int.get(ph, -1) for ph in phases])

        cmap = plt.cm.Set1
        for ph in unique_phases:
            mask = np.array(phases) == ph
            if not np.any(mask):
                continue
            idx = phase_to_int[ph]
            ax.scatter(
                r_layer[mask],
                phase_ints[mask],
                c=[cmap(idx % 9)],
                label=ph,
                s=15,
                alpha=0.8,
            )

        ax.set_xlabel('Radius [1000 km]')
        ax.set_ylabel('Phase (categorical)')
        ax.set_yticks(range(len(unique_phases)))
        ax.set_yticklabels(unique_phases, fontsize=8)
        ax.set_title(title)
        ax.legend(fontsize=7, loc='best')

    fig.suptitle('Phase sampled along adiabat, 1 M_earth PALEOS unified')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, '06_phase_regime.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f'Saved {path}')


# ── Plot 7: Runtime comparison ──────────────────────────────────────


def plot_runtimes(results, mushy_results, three_layer):
    """Bar chart of wall-clock times for each configuration."""
    labels = []
    times = []

    for eos_label in [cfg[0] for cfg in EOS_CONFIGS]:
        if eos_label not in results:
            continue
        for mass in sorted(results[eos_label].keys()):
            _, dt = results[eos_label][mass]
            labels.append(f'{eos_label}\n{mass:.0f} ME')
            times.append(dt)

    for mass in sorted(mushy_results.keys()):
        for factor in MUSHY_FACTORS:
            entry = mushy_results[mass].get(factor)
            if entry is None:
                continue
            _, dt = entry
            labels.append(f'Mushy f={factor:.1f}\n{mass:.0f} ME')
            times.append(dt)

    if three_layer is not None:
        labels.append('3-layer\n1 ME')
        times.append(three_layer['dt'])

    if not labels:
        logger.warning('No runtime data, skipping plot 7.')
        return

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.8), 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, times, color='steelblue', edgecolor='navy')

    # Add time labels on top of bars
    for bar, t in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{t:.1f}s',
            ha='center',
            va='bottom',
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7, rotation=45, ha='right')
    ax.set_ylabel('Wall-clock time [s]')
    ax.set_title('Runtime comparison')
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, '07_runtimes.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f'Saved {path}')


# ── Runtime text report ─────────────────────────────────────────────


def write_runtimes_report(results, mushy_results, three_layer):
    """Write runtimes.txt with a table of all configurations."""
    path = os.path.join(OUTPUT_DIR, 'runtimes.txt')
    lines = []
    lines.append('Zalmoxis PALEOS EOS Benchmark Results')
    lines.append('=' * 70)
    lines.append('')
    lines.append(
        f'{"Configuration":<40s} {"Mass[ME]":>9s} '
        f'{"Time[s]":>9s} {"Converged":>10s} '
        f'{"R[R_E]":>9s}'
    )
    lines.append('-' * 80)

    for eos_label in [cfg[0] for cfg in EOS_CONFIGS]:
        if eos_label not in results:
            continue
        for mass in sorted(results[eos_label].keys()):
            res, dt = results[eos_label][mass]
            if res is not None:
                conv = 'Yes' if res.get('converged') else 'No'
                r_re = f'{res["radii"][-1] / earth_radius:.4f}'
            else:
                conv = 'FAILED'
                r_re = 'N/A'
            lines.append(f'{eos_label:<40s} {mass:>9.1f} {dt:>9.2f} {conv:>10s} {r_re:>9s}')

    lines.append('')
    lines.append('Mushy zone factor variations (PALEOS unified adiabatic)')
    lines.append('-' * 80)

    for mass in sorted(mushy_results.keys()):
        for factor in MUSHY_FACTORS:
            entry = mushy_results[mass].get(factor)
            if entry is None:
                continue
            res, dt = entry
            label = f'f={factor:.1f}'
            if res is not None:
                conv = 'Yes' if res.get('converged') else 'No'
                r_re = f'{res["radii"][-1] / earth_radius:.4f}'
            else:
                conv = 'FAILED'
                r_re = 'N/A'
            lines.append(f'{label:<40s} {mass:>9.1f} {dt:>9.2f} {conv:>10s} {r_re:>9s}')

    if three_layer is not None:
        lines.append('')
        lines.append('3-layer model (PALEOS iron + MgSiO3 + H2O)')
        lines.append('-' * 80)
        res = three_layer['result']
        dt = three_layer['dt']
        if res is not None:
            conv = 'Yes' if res.get('converged') else 'No'
            r_re = f'{res["radii"][-1] / earth_radius:.4f}'
        else:
            conv = 'FAILED'
            r_re = 'N/A'
        lines.append(
            f'{"PALEOS 3-layer 1ME":<40s} {"1.0":>9s} {dt:>9.2f} {conv:>10s} {r_re:>9s}'
        )

    lines.append('')

    text = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(text)
    logger.info(f'Saved {path}')
    print(text)


# ── Summary text report ─────────────────────────────────────────────


def write_summary_report(results, mushy_results, three_layer):
    """Write summary.txt with key quantities for each run."""
    path = os.path.join(OUTPUT_DIR, 'summary.txt')
    lines = []
    lines.append('Zalmoxis PALEOS Benchmark Summary')
    lines.append('=' * 90)
    lines.append('')
    lines.append(
        f'{"Configuration":<35s} {"M[ME]":>6s} '
        f'{"R[R_E]":>9s} {"P_c[GPa]":>10s} '
        f'{"rho_c[kg/m3]":>13s} {"T_c[K]":>9s} '
        f'{"Conv":>5s}'
    )
    lines.append('-' * 90)

    for eos_label in [cfg[0] for cfg in EOS_CONFIGS]:
        if eos_label not in results:
            continue
        for mass in sorted(results[eos_label].keys()):
            res, _ = results[eos_label][mass]
            if res is None:
                lines.append(f'{eos_label:<35s} {mass:>6.1f}  FAILED')
                continue
            r_re = res['radii'][-1] / earth_radius
            p_c = res['pressure'][0] / 1e9
            rho_c = res['density'][0]
            t_c = res['temperature'][0]
            conv = 'Y' if res.get('converged') else 'N'
            lines.append(
                f'{eos_label:<35s} {mass:>6.1f} '
                f'{r_re:>9.4f} {p_c:>10.1f} '
                f'{rho_c:>13.1f} {t_c:>9.1f} {conv:>5s}'
            )

    if three_layer is not None and three_layer['result'] is not None:
        res = three_layer['result']
        r_re = res['radii'][-1] / earth_radius
        p_c = res['pressure'][0] / 1e9
        rho_c = res['density'][0]
        t_c = res['temperature'][0]
        conv = 'Y' if res.get('converged') else 'N'
        lines.append(
            f'{"3-layer (Fe+MgSiO3+H2O)":<35s} {"1.0":>6s} '
            f'{r_re:>9.4f} {p_c:>10.1f} '
            f'{rho_c:>13.1f} {t_c:>9.1f} {conv:>5s}'
        )

    lines.append('')
    text = '\n'.join(lines)
    with open(path, 'w') as f:
        f.write(text)
    logger.info(f'Saved {path}')


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == '__main__':
    total_t0 = time.perf_counter()
    logger.info(f'Output directory: {OUTPUT_DIR}')

    results, mushy_results, three_layer = run_all_configs()

    logger.info('=' * 60)
    logger.info('Generating plots')
    logger.info('=' * 60)

    logger.info('Plot 1: T-P phase diagram')
    plot_tp_phase_diagram(results)

    logger.info('Plot 2: Density vs pressure')
    plot_density_pressure(results)

    logger.info('Plot 3: Radial profiles')
    plot_radial_profiles(results)

    logger.info('Plot 4: Mass-radius diagram')
    plot_mass_radius(results)

    logger.info('Plot 5: Mushy zone factor effect')
    plot_mushy_zone(mushy_results)

    logger.info('Plot 6: Phase regime visualization')
    plot_phase_regime(results)

    logger.info('Plot 7: Runtime comparison')
    plot_runtimes(results, mushy_results, three_layer)

    logger.info('=' * 60)
    logger.info('Writing reports')
    logger.info('=' * 60)

    write_runtimes_report(results, mushy_results, three_layer)
    write_summary_report(results, mushy_results, three_layer)

    total_dt = time.perf_counter() - total_t0
    logger.info(f'Benchmark complete in {total_dt:.1f}s. Output: {OUTPUT_DIR}')
