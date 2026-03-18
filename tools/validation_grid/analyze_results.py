#!/usr/bin/env python3
"""Analyze validation grid results and produce diagnostic plots.

Reads results/summary.csv and profile files. Produces plots in
tools/validation_grid/plots/.

Usage
-----
    python tools/validation_grid/analyze_results.py
"""

from __future__ import annotations

import csv
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'results'
PLOTS_DIR = SCRIPT_DIR / 'plots'
SUMMARY_PATH = RESULTS_DIR / 'summary.csv'

# Plot defaults
DPI = 200
FIGSIZE = (10, 7)
FIGSIZE_WIDE = (14, 7)


def load_summary() -> list[dict]:
    """Load summary.csv into a list of dicts with numeric conversion."""
    if not SUMMARY_PATH.exists():
        print(f'Summary not found: {SUMMARY_PATH}')
        print('Run collect_results.py first.')
        return []
    rows = []
    with open(SUMMARY_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in (
                'planet_mass',
                'surface_temperature',
                'center_temperature',
                'core_mass_fraction',
                'mantle_mass_fraction',
                'mushy_zone_factor',
                'condensed_rho_min',
                'condensed_rho_scale',
                'h2o_fraction',
                'radius_m',
                'radius_rearth',
                'mass_kg',
                'mass_mearth',
                'total_time_s',
            ):
                val = row.get(key, '')
                if val and val not in ('', 'None'):
                    try:
                        row[key] = float(val)
                    except ValueError:
                        row[key] = None
                else:
                    row[key] = None
            # Convert booleans
            for key in (
                'converged',
                'converged_pressure',
                'converged_density',
                'converged_mass',
            ):
                row[key] = str(row.get(key, '')).lower() == 'true'
            rows.append(row)
    return rows


def filter_suite(rows: list[dict], suite: str) -> list[dict]:
    """Filter rows to a specific suite."""
    return [r for r in rows if r.get('suite') == suite]


def converged_only(rows: list[dict]) -> list[dict]:
    """Filter to converged runs only."""
    return [r for r in rows if r.get('converged')]


def load_profile(suite: str, run_id: str) -> np.ndarray | None:
    """Load a planet profile from results.

    Returns
    -------
    np.ndarray or None
        Array with columns: radius, density, gravity, pressure, temperature,
        mass_enclosed. None if file not found.
    """
    path = RESULTS_DIR / suite / run_id / 'planet_profile.txt'
    if not path.exists():
        return None
    try:
        return np.loadtxt(str(path))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dashboard plots
# ---------------------------------------------------------------------------


def plot_convergence_heatmap(rows: list[dict]) -> None:
    """Convergence yes/no for each run, organized by suite."""
    suites = sorted(set(r['suite'] for r in rows))
    fig, ax = plt.subplots(figsize=(14, max(6, len(suites) * 0.8)), dpi=DPI)

    y_labels = []
    for i, suite in enumerate(suites):
        suite_rows = filter_suite(rows, suite)
        suite_rows.sort(key=lambda r: r.get('run_id', ''))
        for j, r in enumerate(suite_rows):
            color = (
                '#2ecc71'
                if r.get('converged')
                else ('#e74c3c' if not r.get('error') else '#f39c12')
            )
            ax.scatter(j, i, c=color, s=12, marker='s', edgecolors='none')
        y_labels.append(f'{suite} ({len(suite_rows)})')

    ax.set_yticks(range(len(suites)))
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel('Run index within suite')
    ax.set_title('Convergence Heatmap (green=converged, red=failed, orange=error)')
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'convergence_heatmap.png'), dpi=DPI)
    plt.close(fig)


def plot_runtime_histogram(rows: list[dict]) -> None:
    """Distribution of runtimes."""
    times = [r['total_time_s'] for r in rows if r.get('total_time_s') is not None]
    if not times:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.hist(times, bins=40, color='#3498db', edgecolor='white', alpha=0.85)
    ax.set_xlabel('Runtime (s)')
    ax.set_ylabel('Count')
    ax.set_title('Runtime Distribution Across All Validation Runs')
    ax.axvline(
        np.median(times),
        color='#e74c3c',
        linestyle='--',
        label=f'Median: {np.median(times):.1f} s',
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'runtime_histogram.png'), dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Mass-radius plots
# ---------------------------------------------------------------------------


def _mr_scatter(
    ax,
    rows,
    color_key,
    cmap_name='viridis',
    label_fmt=None,
    marker='o',
    size=50,
    colorbar_label='',
):
    """Helper: scatter M vs R colored by a parameter."""
    conv = converged_only(rows)
    if not conv:
        return
    masses = np.array([r['mass_mearth'] for r in conv if r['mass_mearth'] is not None])
    radii = np.array([r['radius_rearth'] for r in conv if r['radius_rearth'] is not None])
    if color_key:
        colors = np.array(
            [r.get(color_key, 0) or 0 for r in conv if r['mass_mearth'] is not None]
        )
        sc = ax.scatter(
            masses,
            radii,
            c=colors,
            cmap=cmap_name,
            s=size,
            marker=marker,
            edgecolors='k',
            linewidths=0.3,
        )
        if colorbar_label:
            plt.colorbar(sc, ax=ax, label=colorbar_label)
    else:
        ax.scatter(masses, radii, s=size, marker=marker, edgecolors='k', linewidths=0.3)
    ax.set_xlabel(r'Planet mass ($M_\oplus$)')
    ax.set_ylabel(r'Planet radius ($R_\oplus$)')
    ax.set_xscale('log')


def plot_mass_radius_baseline(rows: list[dict]) -> None:
    """Suite 1: M-R curve, adiabatic vs isothermal."""
    suite = filter_suite(rows, 'suite_01_mass_radius')
    if not suite:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    for t_mode, color, label in [
        ('adiabatic', '#e74c3c', 'Adiabatic 3000K'),
        ('isothermal', '#3498db', 'Isothermal 300K'),
    ]:
        sub = converged_only([r for r in suite if r.get('temperature_mode') == t_mode])
        if not sub:
            continue
        m = [r['mass_mearth'] for r in sub if r['mass_mearth']]
        r_vals = [r['radius_rearth'] for r in sub if r['radius_rearth']]
        if m:
            order = np.argsort(m)
            ax.plot(
                np.array(m)[order],
                np.array(r_vals)[order],
                'o-',
                color=color,
                label=label,
                markersize=6,
            )

    ax.set_xlabel(r'Planet mass ($M_\oplus$)')
    ax.set_ylabel(r'Planet radius ($R_\oplus$)')
    ax.set_xscale('log')
    ax.set_title('Mass-Radius: Adiabatic vs Isothermal (PALEOS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'mass_radius_baseline.png'), dpi=DPI)
    plt.close(fig)


def plot_mass_radius_mixing(rows: list[dict]) -> None:
    """Suite 2: M-R colored by H2O fraction."""
    suite = filter_suite(rows, 'suite_02_mixing')
    if not suite:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    _mr_scatter(
        ax, suite, 'h2o_fraction', cmap_name='coolwarm', colorbar_label='H2O mass fraction'
    )
    ax.set_title('Mass-Radius: H2O Mixing Fraction Sweep')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'mass_radius_mixing.png'), dpi=DPI)
    plt.close(fig)


def plot_mass_radius_temperature(rows: list[dict]) -> None:
    """Suite 3: M-R colored by T_surf."""
    suite = filter_suite(rows, 'suite_03_temperature')
    if not suite:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    _mr_scatter(
        ax,
        suite,
        'surface_temperature',
        cmap_name='hot',
        colorbar_label='Surface temperature (K)',
    )
    ax.set_title('Mass-Radius: Surface Temperature Sweep')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'mass_radius_temperature.png'), dpi=DPI)
    plt.close(fig)


def plot_mass_radius_mushy(rows: list[dict]) -> None:
    """Suite 4: M-R colored by mushy zone factor."""
    suite = filter_suite(rows, 'suite_04_mushy_zone')
    if not suite:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    _mr_scatter(
        ax, suite, 'mushy_zone_factor', cmap_name='plasma', colorbar_label='Mushy zone factor'
    )
    ax.set_title('Mass-Radius: Mushy Zone Factor Sweep')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'mass_radius_mushy.png'), dpi=DPI)
    plt.close(fig)


def plot_mass_radius_3layer(rows: list[dict]) -> None:
    """Suite 5: M-R for three-layer models."""
    suite = filter_suite(rows, 'suite_05_three_layer')
    if not suite:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    conv = converged_only(suite)
    if not conv:
        return

    # Group by layer split (using CMF+MMF as key)
    groups: dict[str, list] = {}
    for r in conv:
        cmf = r.get('core_mass_fraction')
        mmf = r.get('mantle_mass_fraction')
        if cmf is None or mmf is None:
            continue
        key = f'CMF={cmf:.2f}, MMF={mmf:.2f}'
        groups.setdefault(key, []).append(r)

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(groups), 1)))
    for (key, group), color in zip(sorted(groups.items()), colors):
        m = [r['mass_mearth'] for r in group if r['mass_mearth']]
        rv = [r['radius_rearth'] for r in group if r['radius_rearth']]
        if m:
            order = np.argsort(m)
            ax.plot(
                np.array(m)[order],
                np.array(rv)[order],
                'o-',
                color=color,
                label=key,
                markersize=6,
            )

    ax.set_xlabel(r'Planet mass ($M_\oplus$)')
    ax.set_ylabel(r'Planet radius ($R_\oplus$)')
    ax.set_xscale('log')
    ax.set_title('Mass-Radius: Three-Layer Models')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'mass_radius_3layer.png'), dpi=DPI)
    plt.close(fig)


def plot_mass_radius_stress(rows: list[dict]) -> None:
    """Suite 6: High-mass results."""
    suite = filter_suite(rows, 'suite_06_max_mass')
    if not suite:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    _mr_scatter(
        ax,
        suite,
        'surface_temperature',
        cmap_name='hot',
        colorbar_label='Surface temperature (K)',
    )
    ax.set_title('Mass-Radius: High-Mass Stress Test (20 to 50 M_earth)')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'mass_radius_stress.png'), dpi=DPI)
    plt.close(fig)


def plot_mass_radius_exotic(rows: list[dict]) -> None:
    """Suite 7: Exotic architectures."""
    suite = filter_suite(rows, 'suite_07_exotic')
    if not suite:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    conv = converged_only(suite)
    fail = [r for r in suite if not r.get('converged')]

    if conv:
        m = [r['mass_mearth'] for r in conv if r['mass_mearth']]
        rv = [r['radius_rearth'] for r in conv if r['radius_rearth']]
        ax.scatter(
            m,
            rv,
            c='#2ecc71',
            s=60,
            marker='o',
            edgecolors='k',
            linewidths=0.5,
            label='Converged',
            zorder=5,
        )
    if fail:
        m_f = [r.get('mass_mearth') or r.get('planet_mass') for r in fail]
        m_f = [x for x in m_f if x is not None]
        if m_f:
            ax.scatter(
                m_f,
                [0.5] * len(m_f),
                c='#e74c3c',
                s=60,
                marker='x',
                linewidths=2,
                label='Failed/Error',
                zorder=5,
            )

    ax.set_xlabel(r'Planet mass ($M_\oplus$)')
    ax.set_ylabel(r'Planet radius ($R_\oplus$)')
    ax.set_xscale('log')
    ax.set_title('Mass-Radius: Exotic Architectures')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'mass_radius_exotic.png'), dpi=DPI)
    plt.close(fig)


def plot_mass_radius_legacy(rows: list[dict]) -> None:
    """Suite 9: PALEOS vs legacy EOS comparison."""
    suite = filter_suite(rows, 'suite_09_legacy_eos')
    if not suite:
        return
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    conv = converged_only(suite)
    if not conv:
        return

    # Group by EOS label (extract from run_id after M{mass}_)
    groups: dict[str, list] = {}
    for r in conv:
        run_id = r.get('run_id', '')
        # Parse EOS label from run_id: run_NNN_M{mass}_{label}
        parts = run_id.split('_', 3)
        if len(parts) >= 4:
            label = parts[3]
        else:
            label = run_id
        groups.setdefault(label, []).append(r)

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
    for (key, group), color in zip(sorted(groups.items()), colors):
        m = [r['mass_mearth'] for r in group if r['mass_mearth']]
        rv = [r['radius_rearth'] for r in group if r['radius_rearth']]
        if m:
            order = np.argsort(m)
            ax.plot(
                np.array(m)[order],
                np.array(rv)[order],
                'o-',
                color=color,
                label=key.replace('_', ' '),
                markersize=6,
            )

    ax.set_xlabel(r'Planet mass ($M_\oplus$)')
    ax.set_ylabel(r'Planet radius ($R_\oplus$)')
    ax.set_xscale('log')
    ax.set_title('Mass-Radius: EOS Comparison')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'mass_radius_legacy.png'), dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Sensitivity plots
# ---------------------------------------------------------------------------


def plot_sensitivity_h2o(rows: list[dict]) -> None:
    """R vs H2O fraction at fixed masses and T_surf."""
    suite = filter_suite(rows, 'suite_02_mixing')
    conv = converged_only(suite)
    if not conv:
        return

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)
    for ax, t_surf_target in zip(axes, [2000.0, 3000.0]):
        sub = [r for r in conv if r.get('surface_temperature') == t_surf_target]
        masses_set = sorted(set(r['mass_mearth'] for r in sub if r['mass_mearth'] is not None))
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, max(len(masses_set), 1)))
        for mass, color in zip(masses_set, colors):
            m_sub = [
                r
                for r in sub
                if r.get('mass_mearth') is not None and abs(r['mass_mearth'] - mass) < 0.1
            ]
            h2o = [r['h2o_fraction'] for r in m_sub if r['h2o_fraction'] is not None]
            rv = [r['radius_rearth'] for r in m_sub if r['radius_rearth'] is not None]
            if h2o and len(h2o) == len(rv):
                order = np.argsort(h2o)
                ax.plot(
                    np.array(h2o)[order] * 100,
                    np.array(rv)[order],
                    'o-',
                    color=color,
                    label=f'{mass:.1f} M_earth',
                    markersize=5,
                )
        ax.set_xlabel('H2O mass fraction (%)')
        ax.set_ylabel(r'Radius ($R_\oplus$)')
        ax.set_title(f'T_surf = {int(t_surf_target)} K')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Sensitivity: Radius vs H2O Fraction', fontsize=13)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'sensitivity_h2o_fraction.png'), dpi=DPI)
    plt.close(fig)


def plot_sensitivity_temperature(rows: list[dict]) -> None:
    """R vs T_surf at fixed mass and composition."""
    suite = filter_suite(rows, 'suite_03_temperature')
    conv = converged_only(suite)
    if not conv:
        return

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE, dpi=DPI)
    compositions = [
        ('pure', 'Pure MgSiO3', lambda r: r.get('h2o_fraction', 0) in (0, 0.0, None)),
        (
            'h2o15',
            '15% H2O',
            lambda r: r.get('h2o_fraction') is not None
            and abs(r.get('h2o_fraction', 0) - 0.15) < 0.01,
        ),
    ]
    for ax, (comp_key, comp_label, comp_filter) in zip(axes, compositions):
        sub = [r for r in conv if comp_filter(r)]
        masses_set = sorted(set(r['mass_mearth'] for r in sub if r['mass_mearth'] is not None))
        colors = plt.cm.plasma(np.linspace(0.1, 0.9, max(len(masses_set), 1)))
        for mass, color in zip(masses_set, colors):
            m_sub = [
                r
                for r in sub
                if r.get('mass_mearth') is not None and abs(r['mass_mearth'] - mass) < 0.1
            ]
            t = [
                r['surface_temperature'] for r in m_sub if r['surface_temperature'] is not None
            ]
            rv = [r['radius_rearth'] for r in m_sub if r['radius_rearth'] is not None]
            if t and len(t) == len(rv):
                order = np.argsort(t)
                ax.plot(
                    np.array(t)[order],
                    np.array(rv)[order],
                    'o-',
                    color=color,
                    label=f'{mass:.1f} M_earth',
                    markersize=5,
                )
        ax.set_xlabel('Surface temperature (K)')
        ax.set_ylabel(r'Radius ($R_\oplus$)')
        ax.set_title(comp_label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Sensitivity: Radius vs Surface Temperature', fontsize=13)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'sensitivity_temperature.png'), dpi=DPI)
    plt.close(fig)


def plot_sensitivity_mushy_zone(rows: list[dict]) -> None:
    """R vs mushy zone factor at fixed mass."""
    suite = filter_suite(rows, 'suite_04_mushy_zone')
    conv = converged_only(suite)
    if not conv:
        return

    # Only global MZF runs (no per-EOS overrides)
    global_runs = [
        r
        for r in conv
        if not r.get('mushy_zone_factor_iron') and not r.get('mushy_zone_factor_MgSiO3')
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    masses_set = sorted(
        set(r['mass_mearth'] for r in global_runs if r['mass_mearth'] is not None)
    )
    colors = plt.cm.cool(np.linspace(0.1, 0.9, max(len(masses_set), 1)))
    for mass, color in zip(masses_set, colors):
        m_sub = [
            r
            for r in global_runs
            if r.get('mass_mearth') is not None and abs(r['mass_mearth'] - mass) < 0.1
        ]
        mzf = [r['mushy_zone_factor'] for r in m_sub if r['mushy_zone_factor'] is not None]
        rv = [r['radius_rearth'] for r in m_sub if r['radius_rearth'] is not None]
        if mzf and len(mzf) == len(rv):
            order = np.argsort(mzf)
            ax.plot(
                np.array(mzf)[order],
                np.array(rv)[order],
                'o-',
                color=color,
                label=f'{mass:.0f} M_earth',
                markersize=6,
            )

    ax.set_xlabel('Mushy zone factor')
    ax.set_ylabel(r'Radius ($R_\oplus$)')
    ax.set_title('Sensitivity: Radius vs Mushy Zone Factor')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'sensitivity_mushy_zone.png'), dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Profile comparison plots
# ---------------------------------------------------------------------------


def _find_run(rows: list[dict], suite: str, **filters) -> dict | None:
    """Find the first converged run in a suite matching filters."""
    for r in converged_only(filter_suite(rows, suite)):
        match = True
        for k, v in filters.items():
            rv = r.get(k)
            if rv is None or (isinstance(v, float) and abs(rv - v) > 0.01):
                match = False
                break
            elif isinstance(v, str) and rv != v:
                match = False
                break
        if match:
            return r
    return None


def plot_profiles_compositions(rows: list[dict]) -> None:
    """Density/P/T profiles at 1 M_earth for different compositions."""
    suite = 'suite_02_mixing'
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=DPI)

    h2o_fracs = [0.0, 0.10, 0.20, 0.30]
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(h2o_fracs)))

    for h2o_frac, color in zip(h2o_fracs, colors):
        run = _find_run(
            rows, suite, mass_mearth=1.0, h2o_fraction=h2o_frac, surface_temperature=3000.0
        )
        if run is None:
            continue
        profile = load_profile(suite, run['run_id'])
        if profile is None:
            continue

        r_norm = profile[:, 0] / profile[-1, 0]  # Normalized radius
        label = f'{int(h2o_frac * 100)}% H2O'

        axes[0].plot(r_norm, profile[:, 1], color=color, label=label)
        axes[1].plot(r_norm, profile[:, 3] / 1e9, color=color, label=label)
        axes[2].plot(r_norm, profile[:, 4], color=color, label=label)

    axes[0].set_ylabel('Density (kg/m^3)')
    axes[1].set_ylabel('Pressure (GPa)')
    axes[2].set_ylabel('Temperature (K)')
    for ax in axes:
        ax.set_xlabel('r / R_planet')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Profiles at 1 M_earth: Varying H2O Content (T_surf=3000K)', fontsize=13)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'profiles_1Me_compositions.png'), dpi=DPI)
    plt.close(fig)


def plot_profiles_mass_sweep(rows: list[dict]) -> None:
    """Profiles at different masses for the same composition."""
    suite = 'suite_01_mass_radius'
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=DPI)

    target_masses = [0.5, 1.0, 5.0, 10.0]
    colors = plt.cm.magma(np.linspace(0.2, 0.8, len(target_masses)))

    for mass, color in zip(target_masses, colors):
        run = _find_run(rows, suite, mass_mearth=mass, temperature_mode='adiabatic')
        if run is None:
            continue
        profile = load_profile(suite, run['run_id'])
        if profile is None:
            continue

        r_norm = profile[:, 0] / profile[-1, 0]
        label = f'{mass:.1f} M_earth'

        axes[0].plot(r_norm, profile[:, 1], color=color, label=label)
        axes[1].plot(r_norm, profile[:, 3] / 1e9, color=color, label=label)
        axes[2].plot(r_norm, profile[:, 4], color=color, label=label)

    axes[0].set_ylabel('Density (kg/m^3)')
    axes[1].set_ylabel('Pressure (GPa)')
    axes[2].set_ylabel('Temperature (K)')
    for ax in axes:
        ax.set_xlabel('r / R_planet')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Profiles: Mass Sweep (Adiabatic 3000K, PALEOS)', fontsize=13)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'profiles_mass_sweep.png'), dpi=DPI)
    plt.close(fig)


def plot_profiles_temperature_modes(rows: list[dict]) -> None:
    """Isothermal vs linear vs adiabatic at 1 M_earth."""
    suite = 'suite_08_temperature_modes'
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), dpi=DPI)

    modes = [
        ('isothermal', '#3498db', 'Isothermal 3000K'),
        ('linear', '#2ecc71', 'Linear 3000 to 6000K'),
        ('adiabatic', '#e74c3c', 'Adiabatic 3000K'),
    ]

    for t_mode, color, label in modes:
        run = _find_run(rows, suite, mass_mearth=1.0, temperature_mode=t_mode, h2o_fraction=0.0)
        if run is None:
            continue
        profile = load_profile(suite, run['run_id'])
        if profile is None:
            continue

        r_norm = profile[:, 0] / profile[-1, 0]
        axes[0].plot(r_norm, profile[:, 1], color=color, label=label)
        axes[1].plot(r_norm, profile[:, 3] / 1e9, color=color, label=label)
        axes[2].plot(r_norm, profile[:, 4], color=color, label=label)

    axes[0].set_ylabel('Density (kg/m^3)')
    axes[1].set_ylabel('Pressure (GPa)')
    axes[2].set_ylabel('Temperature (K)')
    for ax in axes:
        ax.set_xlabel('r / R_planet')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Profiles at 1 M_earth: Temperature Mode Comparison', fontsize=13)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'profiles_temperature_modes.png'), dpi=DPI)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Exotic results table
# ---------------------------------------------------------------------------


def plot_exotic_results_table(rows: list[dict]) -> None:
    """Table showing which exotic configs converged."""
    suite = filter_suite(rows, 'suite_07_exotic')
    if not suite:
        return

    fig, ax = plt.subplots(figsize=(12, max(4, len(suite) * 0.35)), dpi=DPI)
    ax.axis('off')

    table_data = []
    for r in sorted(suite, key=lambda x: x.get('run_id', '')):
        status = 'Converged' if r.get('converged') else 'FAILED'
        if r.get('error'):
            status = f'ERROR: {str(r["error"])[:50]}'
        r_str = f'{r["radius_rearth"]:.4f}' if r.get('radius_rearth') else 'N/A'
        t_str = f'{r["total_time_s"]:.1f}' if r.get('total_time_s') else 'N/A'
        table_data.append(
            [
                r.get('run_id', '')[:40],
                f'{r.get("planet_mass", "N/A")}',
                status,
                r_str,
                f'{t_str} s',
            ]
        )

    if table_data:
        table = ax.table(
            cellText=table_data,
            colLabels=['Run', 'Mass (M_e)', 'Status', 'R (R_e)', 'Time'],
            loc='center',
            cellLoc='left',
        )
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.auto_set_column_width(range(5))

        # Color cells by status
        for i, row_data in enumerate(table_data):
            status = row_data[2]
            if 'Converged' in status:
                color = '#d5f5e3'
            elif 'ERROR' in status:
                color = '#fdebd0'
            else:
                color = '#fadbd8'
            for j in range(5):
                table[i + 1, j].set_facecolor(color)

    ax.set_title('Exotic Architecture Results', fontsize=12, pad=20)
    fig.tight_layout()
    fig.savefig(str(PLOTS_DIR / 'exotic_results_table.png'), dpi=DPI, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(rows: list[dict]) -> None:
    """Write a markdown validation report."""
    report_path = SCRIPT_DIR / 'validation_report.md'

    total = len(rows)
    conv = len(converged_only(rows))
    errors = len([r for r in rows if r.get('error')])
    failed = total - conv - errors

    suites = sorted(set(r['suite'] for r in rows))

    lines = [
        '# Zalmoxis Validation Grid Report',
        '',
        '## Summary',
        '',
        f'- Total runs: {total}',
        f'- Converged: {conv} ({conv / total * 100:.1f}%)' if total else '- Converged: 0',
        f'- Failed (non-converged): {failed}',
        f'- Errors (exceptions): {errors}',
        '',
        '## Suite Results',
        '',
        '| Suite | Total | Converged | Failed | Error | Rate |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]

    for suite in suites:
        s_rows = filter_suite(rows, suite)
        s_total = len(s_rows)
        s_conv = len(converged_only(s_rows))
        s_err = len([r for r in s_rows if r.get('error')])
        s_fail = s_total - s_conv - s_err
        s_rate = f'{s_conv / s_total * 100:.1f}%' if s_total else 'N/A'
        lines.append(f'| {suite} | {s_total} | {s_conv} | {s_fail} | {s_err} | {s_rate} |')

    lines.extend(
        [
            '',
            '## Plots',
            '',
            'All plots are in `tools/validation_grid/plots/`.',
            '',
            '### Dashboard',
            '- `convergence_heatmap.png`: Convergence status for every run',
            '- `runtime_histogram.png`: Distribution of computation times',
            '',
            '### Mass-Radius Relations',
            '- `mass_radius_baseline.png`: Adiabatic vs isothermal baseline (Suite 1)',
            '- `mass_radius_mixing.png`: H2O mixing fraction effect (Suite 2)',
            '- `mass_radius_temperature.png`: Surface temperature effect (Suite 3)',
            '- `mass_radius_mushy.png`: Mushy zone factor effect (Suite 4)',
            '- `mass_radius_3layer.png`: Three-layer models (Suite 5)',
            '- `mass_radius_stress.png`: High-mass stress test (Suite 6)',
            '- `mass_radius_exotic.png`: Exotic architectures (Suite 7)',
            '- `mass_radius_legacy.png`: EOS comparison (Suite 9)',
            '',
            '### Sensitivity',
            '- `sensitivity_h2o_fraction.png`: Radius vs H2O fraction',
            '- `sensitivity_temperature.png`: Radius vs surface temperature',
            '- `sensitivity_mushy_zone.png`: Radius vs mushy zone factor',
            '',
            '### Profile Comparisons',
            '- `profiles_1Me_compositions.png`: 1 M_earth, varying H2O content',
            '- `profiles_mass_sweep.png`: Adiabatic profiles at different masses',
            '- `profiles_temperature_modes.png`: Temperature mode comparison at 1 M_earth',
            '',
            '### Edge Cases',
            '- `exotic_results_table.png`: Table of exotic architecture results',
            '',
            '## Recommended Parameter Ranges',
            '',
            '(To be filled based on convergence analysis.)',
            '',
            '## Bugs Found',
            '',
            '(To be filled based on analysis of failed runs.)',
            '',
        ]
    )

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f'Report written to: {report_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Run all analysis and produce plots."""
    rows = load_summary()
    if not rows:
        print('No data to analyze.')
        return

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Producing plots in {PLOTS_DIR} ...')

    # Dashboard
    plot_convergence_heatmap(rows)
    plot_runtime_histogram(rows)
    print('  Dashboard plots done.')

    # Mass-radius plots
    plot_mass_radius_baseline(rows)
    plot_mass_radius_mixing(rows)
    plot_mass_radius_temperature(rows)
    plot_mass_radius_mushy(rows)
    plot_mass_radius_3layer(rows)
    plot_mass_radius_stress(rows)
    plot_mass_radius_exotic(rows)
    plot_mass_radius_legacy(rows)
    print('  Mass-radius plots done.')

    # Sensitivity plots
    plot_sensitivity_h2o(rows)
    plot_sensitivity_temperature(rows)
    plot_sensitivity_mushy_zone(rows)
    print('  Sensitivity plots done.')

    # Profile comparisons
    plot_profiles_compositions(rows)
    plot_profiles_mass_sweep(rows)
    plot_profiles_temperature_modes(rows)
    print('  Profile plots done.')

    # Exotic results table
    plot_exotic_results_table(rows)
    print('  Exotic results table done.')

    # Report
    generate_report(rows)

    n_plots = len(list(PLOTS_DIR.glob('*.png')))
    print(f'\nDone. {n_plots} plots written to {PLOTS_DIR}')


if __name__ == '__main__':
    main()
