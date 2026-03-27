"""Phase regime benchmark: probes solid/mushy/liquid transitions across temperature.

Runs Zalmoxis at surface temperatures from 300 K to 5000 K to map which
phases are sampled along the adiabat in both the iron core and MgSiO3
mantle. Also tests 3-layer models with PALEOS:H2O and mushy zone factor
effects at temperatures where the adiabat crosses the liquidus.

Usage
-----
    python -m src.tests.benchmark_phase_regimes

Output: ``output_files/benchmark_phase_regimes/``
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s  %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger('phase_regimes')
logging.getLogger('zalmoxis.eos_functions').setLevel(logging.WARNING)
logging.getLogger('zalmoxis.zalmoxis').setLevel(logging.WARNING)

ZALMOXIS_ROOT = os.environ.get('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    print('Error: ZALMOXIS_ROOT environment variable not set.')
    sys.exit(1)

from zalmoxis.constants import earth_mass, earth_radius  # noqa: E402
from zalmoxis.eos_functions import load_paleos_unified_table  # noqa: E402
from zalmoxis.eos_properties import EOS_REGISTRY  # noqa: E402
from zalmoxis.zalmoxis import (  # noqa: E402
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
    main,
)

OUTPUT_DIR = os.path.join(ZALMOXIS_ROOT, 'output_files', 'benchmark_phase_regimes')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Helpers ──────────────────────────────────────────────────────────


def _run(
    mass_earth,
    core_eos,
    mantle_eos,
    mode,
    T_surf,
    T_center=6000,
    ice_eos='',
    cmf=0.325,
    mmf=0,
    mushy=1.0,
):
    """Run Zalmoxis and return (result_dict, wall_time)."""
    root = os.environ['ZALMOXIS_ROOT']
    config = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    config['planet_mass'] = mass_earth * earth_mass
    lec = {'core': core_eos, 'mantle': mantle_eos}
    if ice_eos:
        lec['ice_layer'] = ice_eos
    config['layer_eos_config'] = lec
    config['core_mass_fraction'] = cmf
    config['mantle_mass_fraction'] = mmf
    config['temperature_mode'] = mode
    config['surface_temperature'] = T_surf
    config['center_temperature'] = T_center
    config['mushy_zone_factor'] = mushy
    config['data_output_enabled'] = False
    config['plotting_enabled'] = False
    config['verbose'] = False

    t0 = time.time()
    result = main(
        config,
        material_dictionaries=load_material_dictionaries(),
        melting_curves_functions=load_solidus_liquidus_functions(
            lec,
            config.get('rock_solidus', 'Stixrude14-solidus'),
            config.get('rock_liquidus', 'Stixrude14-liquidus'),
        ),
        input_dir=os.path.join(root, 'input'),
    )
    dt = time.time() - t0
    return result, dt


def _lookup_phases(cache, P_arr, T_arr):
    """Look up phase string at each (P, T) point from a unified PALEOS cache."""
    ulp = cache['unique_log_p']
    # Extract unique_log_t from the density interpolator grid axes
    ult = cache['density_interp'].grid[1]

    phases = []
    for P, T in zip(P_arr, T_arr):
        if P <= 0 or T <= 0:
            phases.append('')
            continue
        lp = np.log10(max(P, cache['p_min']))
        lt = np.log10(max(T, 1.0))
        ip = int(np.argmin(np.abs(ulp - lp)))
        it = int(np.argmin(np.abs(ult - lt)))
        ip = min(ip, cache['phase_grid'].shape[0] - 1)
        it = min(it, cache['phase_grid'].shape[1] - 1)
        ph = cache['phase_grid'][ip, it]
        phases.append(str(ph) if ph else '')
    return phases


# ── Main benchmarks ──────────────────────────────────────────────────


def run_temperature_sweep():
    """Run 2-layer PALEOS at 1 and 5 M_earth across surface temperatures."""
    temperatures = [300, 500, 800, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
    masses = [1.0, 5.0]

    iron_cache = load_paleos_unified_table(EOS_REGISTRY['PALEOS:iron']['eos_file'])
    mgsio3_cache = load_paleos_unified_table(EOS_REGISTRY['PALEOS:MgSiO3']['eos_file'])

    results = {}
    for mass_e in masses:
        for T_surf in temperatures:
            label = f'{mass_e}ME_{T_surf}K'
            logger.info(f'  {label}...')
            try:
                T_center = max(T_surf + 2000, 6000)
                res, dt = _run(
                    mass_e,
                    'PALEOS:iron',
                    'PALEOS:MgSiO3',
                    'adiabatic',
                    T_surf,
                    T_center=T_center,
                )

                # Extract phase info
                cmb_idx = np.argmax(res['mass_enclosed'] >= res['cmb_mass'])
                P = res['pressure']
                T = res['temperature']

                core_phases = _lookup_phases(iron_cache, P[:cmb_idx], T[:cmb_idx])
                mantle_phases = _lookup_phases(mgsio3_cache, P[cmb_idx:], T[cmb_idx:])

                core_unique = sorted(set(p for p in core_phases if p))
                mantle_unique = sorted(set(p for p in mantle_phases if p))

                results[label] = {
                    'result': res,
                    'dt': dt,
                    'core_phases': core_unique,
                    'mantle_phases': mantle_unique,
                    'T_surf': T_surf,
                    'mass': mass_e,
                }
                logger.info(
                    f'    {dt:.1f}s conv={res["converged"]} '
                    f'R={res["radii"][-1] / earth_radius:.3f} '
                    f'core=[{",".join(core_unique)}] '
                    f'mantle=[{",".join(mantle_unique)}]'
                )
            except Exception as e:
                logger.warning(f'    FAILED: {e}')
                results[label] = None

    return results


def run_mushy_sweep():
    """Mushy zone sweep at T_surf=1500 K where adiabat may cross liquidus."""
    factors = [1.0, 0.9, 0.8, 0.7]
    results = {}
    for f in factors:
        label = f'mushy_{f}'
        logger.info(f'  {label}...')
        try:
            res, dt = _run(
                1.0,
                'PALEOS:iron',
                'PALEOS:MgSiO3',
                'adiabatic',
                1500,
                T_center=5000,
                mushy=f,
            )
            results[label] = {'result': res, 'dt': dt, 'factor': f}
            logger.info(
                f'    {dt:.1f}s conv={res["converged"]} R={res["radii"][-1] / earth_radius:.3f}'
            )
        except Exception as e:
            logger.warning(f'    FAILED: {e}')
            results[label] = None
    return results


def run_three_layer():
    """3-layer PALEOS models at various surface temperatures."""
    temperatures = [300, 500, 800, 1000]
    results = {}
    for T_surf in temperatures:
        label = f'3layer_{T_surf}K'
        logger.info(f'  {label}...')
        try:
            res, dt = _run(
                1.0,
                'PALEOS:iron',
                'PALEOS:MgSiO3',
                'adiabatic',
                T_surf,
                T_center=max(T_surf + 1000, 3000),
                ice_eos='PALEOS:H2O',
                cmf=0.25,
                mmf=0.50,
            )
            results[label] = {'result': res, 'dt': dt, 'T_surf': T_surf}
            logger.info(
                f'    {dt:.1f}s conv={res["converged"]} R={res["radii"][-1] / earth_radius:.3f}'
            )
        except Exception as e:
            logger.warning(f'    FAILED: {e}')
            results[label] = None
    return results


def run_eos_comparison_cold():
    """Compare all EOS at T_surf=1500 K (cold enough for partial solidification)."""
    configs = [
        ('PALEOS unified', 'PALEOS:iron', 'PALEOS:MgSiO3', 'adiabatic'),
        ('PALEOS-2phase', 'Seager2007:iron', 'PALEOS-2phase:MgSiO3', 'adiabatic'),
        ('WolfBower2018', 'Seager2007:iron', 'WolfBower2018:MgSiO3', 'adiabatic'),
        ('Seager2007', 'Seager2007:iron', 'Seager2007:MgSiO3', 'isothermal'),
    ]
    results = {}
    for label, core, mantle, mode in configs:
        logger.info(f'  {label}...')
        try:
            T_surf = 300 if mode == 'isothermal' else 1500
            res, dt = _run(1.0, core, mantle, mode, T_surf, T_center=5000)
            results[label] = {'result': res, 'dt': dt}
            logger.info(
                f'    {dt:.1f}s conv={res["converged"]} R={res["radii"][-1] / earth_radius:.3f}'
            )
        except Exception as e:
            logger.warning(f'    FAILED: {e}')
            results[label] = None
    return results


# ── Plotting ─────────────────────────────────────────────────────────


def plot_phase_regime_map(temp_results):
    """Phase regime map: which phases appear at each (mass, T_surf)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, material, title in [
        (axes[0], 'core_phases', 'Iron core phases'),
        (axes[1], 'mantle_phases', 'MgSiO3 mantle phases'),
    ]:
        for mass_e in [1.0, 5.0]:
            temps = []
            phase_sets = []
            for key, val in sorted(temp_results.items()):
                if val is None:
                    continue
                if val['mass'] != mass_e or not val['result']['converged']:
                    continue
                temps.append(val['T_surf'])
                phase_sets.append(val[material])

            if not temps:
                continue

            # Build a phase presence matrix
            all_phases = sorted(set(p for ps in phase_sets for p in ps))
            for ip, phase in enumerate(all_phases):
                presence = [1 if phase in ps else 0 for ps in phase_sets]
                ax.scatter(
                    [t for t, p in zip(temps, presence) if p],
                    [phase] * sum(presence),
                    s=80,
                    label=f'{mass_e} ME: {phase}' if mass_e == 1.0 else None,
                    marker='o' if mass_e == 1.0 else 's',
                    alpha=0.7,
                )

        ax.set_xlabel('Surface temperature [K]')
        ax.set_ylabel('Phase')
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'phase_regime_map.png'), dpi=150)
    plt.close(fig)
    logger.info('Saved phase_regime_map.png')


def plot_tp_profiles_temperature_sweep(temp_results):
    """T-P profiles at different surface temperatures."""
    from zalmoxis.melting_curves import monteux16_liquidus, monteux16_solidus

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for mass_e, ax in [(1.0, axes[0]), (5.0, axes[1])]:
        # Melting curves
        P_melt = np.logspace(8, 12.5, 500)
        ax.plot(
            P_melt / 1e9,
            [monteux16_solidus(p) for p in P_melt],
            'k--',
            lw=1,
            alpha=0.5,
            label='Monteux16 solidus',
        )
        ax.plot(
            P_melt / 1e9,
            [monteux16_liquidus(p) for p in P_melt],
            'k-.',
            lw=1,
            alpha=0.5,
            label='Monteux16 liquidus',
        )

        cmap = plt.cm.coolwarm
        for key, val in sorted(temp_results.items()):
            if val is None or val['mass'] != mass_e:
                continue
            if not val['result']['converged']:
                continue
            T_surf = val['T_surf']
            res = val['result']
            color = cmap((T_surf - 300) / (5000 - 300))
            mask = res['pressure'] > 1e5
            ax.plot(
                res['pressure'][mask] / 1e9,
                res['temperature'][mask],
                '-',
                color=color,
                lw=1.2,
                label=f'{T_surf} K',
            )

        ax.set_xlabel('Pressure [GPa]')
        ax.set_ylabel('Temperature [K]')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title(f'{mass_e} M_earth')
        ax.legend(fontsize=7, ncol=2)

    fig.suptitle('T-P profiles vs surface temperature (PALEOS unified adiabatic)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'tp_sweep.png'), dpi=150)
    plt.close(fig)
    logger.info('Saved tp_sweep.png')


def plot_radial_profiles_sweep(temp_results, mass_e=1.0):
    """4-panel radial profiles at various T_surf for a given mass."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    cmap = plt.cm.coolwarm

    for key, val in sorted(temp_results.items()):
        if val is None or val['mass'] != mass_e:
            continue
        if not val['result']['converged']:
            continue
        T_surf = val['T_surf']
        res = val['result']
        color = cmap((T_surf - 300) / (5000 - 300))
        r_km = res['radii'] / 1e6

        axes[0, 0].plot(r_km, res['temperature'], color=color, label=f'{T_surf} K')
        axes[0, 1].plot(r_km, res['density'], color=color)
        axes[1, 0].plot(r_km, res['pressure'] / 1e9, color=color)
        axes[1, 1].plot(r_km, res['gravity'], color=color)

    axes[0, 0].set_ylabel('Temperature [K]')
    axes[0, 1].set_ylabel('Density [kg/m^3]')
    axes[1, 0].set_ylabel('Pressure [GPa]')
    axes[1, 1].set_ylabel('Gravity [m/s^2]')
    for ax in axes.flat:
        ax.set_xlabel('Radius [Mm]')
    axes[0, 0].legend(fontsize=7, ncol=2)

    fig.suptitle(f'Radial profiles vs T_surf, {mass_e} M_earth PALEOS adiabatic')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f'radial_sweep_{mass_e}ME.png'), dpi=150)
    plt.close(fig)
    logger.info(f'Saved radial_sweep_{mass_e}ME.png')


def plot_mushy_comparison(mushy_results):
    """T, rho, P profiles comparing mushy zone factors at T_surf=1500 K."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for label, val in sorted(mushy_results.items()):
        if val is None or not val['result']['converged']:
            continue
        f = val['factor']
        res = val['result']
        r_km = res['radii'] / 1e6

        axes[0].plot(r_km, res['temperature'], label=f'f={f}')
        axes[1].plot(r_km, res['density'], label=f'f={f}')
        axes[2].plot(r_km, res['pressure'] / 1e9, label=f'f={f}')

    axes[0].set_ylabel('Temperature [K]')
    axes[1].set_ylabel('Density [kg/m^3]')
    axes[2].set_ylabel('Pressure [GPa]')
    for ax in axes:
        ax.set_xlabel('Radius [Mm]')
        ax.legend()

    fig.suptitle('Mushy zone factor comparison, 1 M_earth T_surf=1500 K')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'mushy_1500K.png'), dpi=150)
    plt.close(fig)
    logger.info('Saved mushy_1500K.png')


def plot_three_layer(three_layer_results):
    """Radial profiles for 3-layer models at various T_surf."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for label, val in sorted(three_layer_results.items()):
        if val is None or not val['result']['converged']:
            continue
        T_surf = val['T_surf']
        res = val['result']
        r_km = res['radii'] / 1e6

        axes[0, 0].plot(r_km, res['temperature'], label=f'{T_surf} K')
        axes[0, 1].plot(r_km, res['density'])
        axes[1, 0].plot(r_km, res['pressure'] / 1e9)
        axes[1, 1].plot(r_km, res['gravity'])

    axes[0, 0].set_ylabel('Temperature [K]')
    axes[0, 1].set_ylabel('Density [kg/m^3]')
    axes[1, 0].set_ylabel('Pressure [GPa]')
    axes[1, 1].set_ylabel('Gravity [m/s^2]')
    for ax in axes.flat:
        ax.set_xlabel('Radius [Mm]')
    axes[0, 0].legend(fontsize=8)

    fig.suptitle('3-layer PALEOS (iron + MgSiO3 + H2O) adiabatic')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'three_layer.png'), dpi=150)
    plt.close(fig)
    logger.info('Saved three_layer.png')


def plot_eos_comparison(eos_results):
    """4-panel comparison of all EOS at T_surf=1500 K."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for label, val in sorted(eos_results.items()):
        if val is None or not val['result']['converged']:
            continue
        res = val['result']
        r_km = res['radii'] / 1e6

        axes[0, 0].plot(r_km, res['temperature'], label=label)
        axes[0, 1].plot(r_km, res['density'], label=label)
        axes[1, 0].plot(r_km, res['pressure'] / 1e9, label=label)
        axes[1, 1].plot(r_km, res['gravity'], label=label)

    axes[0, 0].set_ylabel('Temperature [K]')
    axes[0, 1].set_ylabel('Density [kg/m^3]')
    axes[1, 0].set_ylabel('Pressure [GPa]')
    axes[1, 1].set_ylabel('Gravity [m/s^2]')
    for ax in axes.flat:
        ax.set_xlabel('Radius [Mm]')
        ax.legend(fontsize=8)

    fig.suptitle('EOS comparison at 1 M_earth, T_surf=1500 K (adiabatic)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'eos_comparison_1500K.png'), dpi=150)
    plt.close(fig)
    logger.info('Saved eos_comparison_1500K.png')


def write_report(temp_results, mushy_results, three_results, eos_results):
    """Write summary text file."""
    path = os.path.join(OUTPUT_DIR, 'phase_regime_report.txt')
    with open(path, 'w') as f:
        f.write('Phase Regime Benchmark Report\n')
        f.write('=' * 70 + '\n\n')

        f.write('Temperature sweep (PALEOS unified adiabatic)\n')
        f.write('-' * 70 + '\n')
        f.write(
            f'{"Mass":>5} {"T_surf":>6} {"conv":>5} {"R/R_E":>7} '
            f'{"T_ctr":>7} {"time":>6} {"core_phases":>25} '
            f'{"mantle_phases":>25}\n'
        )
        for key, val in sorted(temp_results.items()):
            if val is None:
                continue
            res = val['result']
            R = res['radii'][-1] / earth_radius
            Tc = res['temperature'][0]
            f.write(
                f'{val["mass"]:5.1f} {val["T_surf"]:6d} '
                f'{str(res["converged"]):>5} {R:7.4f} '
                f'{Tc:7.0f} {val["dt"]:6.1f} '
                f'{",".join(val["core_phases"]):>25} '
                f'{",".join(val["mantle_phases"]):>25}\n'
            )

        f.write('\nMushy zone factor sweep (1 ME, T_surf=1500 K)\n')
        f.write('-' * 70 + '\n')
        for label, val in sorted(mushy_results.items()):
            if val is None:
                continue
            res = val['result']
            R = res['radii'][-1] / earth_radius
            f.write(
                f'  f={val["factor"]:.1f}: conv={res["converged"]}, '
                f'R={R:.4f} R_E, time={val["dt"]:.1f}s\n'
            )

        f.write('\n3-layer models (iron + MgSiO3 + H2O)\n')
        f.write('-' * 70 + '\n')
        for label, val in sorted(three_results.items()):
            if val is None:
                continue
            res = val['result']
            R = res['radii'][-1] / earth_radius
            f.write(
                f'  T_surf={val["T_surf"]}K: conv={res["converged"]}, '
                f'R={R:.4f} R_E, time={val["dt"]:.1f}s\n'
            )

        f.write('\nEOS comparison (1 ME, T_surf=1500 K)\n')
        f.write('-' * 70 + '\n')
        for label, val in sorted(eos_results.items()):
            if val is None:
                continue
            res = val['result']
            R = res['radii'][-1] / earth_radius
            f.write(
                f'  {label}: conv={res["converged"]}, R={R:.4f} R_E, time={val["dt"]:.1f}s\n'
            )

    logger.info(f'Saved {path}')


# ── Main ─────────────────────────────────────────────────────────────


if __name__ == '__main__':
    logger.info(f'Output: {OUTPUT_DIR}')

    logger.info('=== Temperature sweep (2-layer) ===')
    temp_results = run_temperature_sweep()

    logger.info('=== Mushy zone sweep (T_surf=1500 K) ===')
    mushy_results = run_mushy_sweep()

    logger.info('=== 3-layer PALEOS models ===')
    three_results = run_three_layer()

    logger.info('=== EOS comparison at 1500 K ===')
    eos_results = run_eos_comparison_cold()

    logger.info('=== Generating plots ===')
    plot_phase_regime_map(temp_results)
    plot_tp_profiles_temperature_sweep(temp_results)
    plot_radial_profiles_sweep(temp_results, 1.0)
    plot_radial_profiles_sweep(temp_results, 5.0)
    plot_mushy_comparison(mushy_results)
    plot_three_layer(three_results)
    plot_eos_comparison(eos_results)

    write_report(temp_results, mushy_results, three_results, eos_results)
    logger.info('Done.')
