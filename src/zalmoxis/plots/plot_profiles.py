"""Planet interior structure profile plots.

Generates a 6-panel figure showing density, pressure, temperature,
gravity, phase state, and mass enclosed as a function of radius.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from zalmoxis.constants import earth_mass, earth_radius

ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')

# Soft pastel colors for profile lines
_LINE_COLORS = {
    'density': '#60a5fa',
    'pressure': '#f87171',
    'gravity': '#4ade80',
    'temperature': '#c084fc',
    'mass': '#22d3ee',
}

# Phase colors (pastel, matching PALEOS phase_id strings)
_PHASE_COLORS = {
    # Iron phases
    'solid-epsilon-hcp': '#93c5fd',
    'solid-gamma-fcc': '#bfdbfe',
    'solid-delta-bcc': '#dbeafe',
    'solid-alpha-bcc': '#eff6ff',
    # MgSiO3 phases
    'solid-brg': '#fdba74',
    'solid-ppv': '#fb923c',
    'solid-lpcen': '#bbf7d0',
    'solid-hpcen': '#86efac',
    'solid-en': '#4ade80',
    # H2O phases
    'solid-ice-Ih': '#e0f2fe',
    'solid-ice-VII': '#bae6fd',
    'solid-ice-X': '#7dd3fc',
    'vapour': '#fef3c7',
    'supercritical': '#fde68a',
    # Shared
    'liquid': '#fca5a5',
}

_PHASE_PRETTY = {
    'solid-epsilon-hcp': 'Fe epsilon-hcp',
    'solid-gamma-fcc': 'Fe gamma-fcc',
    'solid-delta-bcc': 'Fe delta-bcc',
    'solid-alpha-bcc': 'Fe alpha-bcc',
    'solid-brg': 'Bridgmanite',
    'solid-ppv': 'Postperovskite',
    'solid-lpcen': 'LP clinoenstatite',
    'solid-hpcen': 'HP clinoenstatite',
    'solid-en': 'Enstatite',
    'solid-ice-Ih': 'Ice Ih',
    'solid-ice-VII': 'Ice VII',
    'solid-ice-X': 'Ice X',
    'vapour': 'Vapour',
    'supercritical': 'Supercritical',
    'liquid': 'Liquid',
}


def _lookup_phases(radii, pressure, density, temperature, mass_enclosed, layer_eos_config):
    """Look up the PALEOS phase at each radial shell.

    Parameters
    ----------
    radii : numpy.ndarray
        Radial grid in m.
    pressure, density, temperature, mass_enclosed : numpy.ndarray
        Profile arrays.
    layer_eos_config : dict
        Per-layer EOS config dict.

    Returns
    -------
    list of str
        Phase label at each shell.
    """
    from zalmoxis.eos_functions import _ensure_unified_cache
    from zalmoxis.eos_properties import EOS_REGISTRY
    from zalmoxis.mixing import parse_layer_components

    interp_cache = {}
    phase_labels = []

    # Determine which EOS each layer uses (primary component)
    core_eos = layer_eos_config.get('core', '')
    mantle_eos = layer_eos_config.get('mantle', '')
    # Get primary component for each layer
    core_primary = parse_layer_components(core_eos).primary() if core_eos else None
    mantle_primary = parse_layer_components(mantle_eos).primary() if mantle_eos else None
    # Find CMB mass
    total_mass = mass_enclosed[-1]
    # Read core_mass_fraction from mass profile (first shell where mass starts growing)
    # Use the midpoint between min and max density as a heuristic for CMB
    cmb_mass_est = 0.325 * total_mass  # default

    for i in range(len(radii)):
        P, T = pressure[i], temperature[i]
        if P <= 0 or density[i] <= 0:
            phase_labels.append('none')
            continue

        # Determine which EOS this shell uses
        if mass_enclosed[i] < cmb_mass_est:
            eos_name = core_primary
        else:
            eos_name = mantle_primary

        if eos_name is None or eos_name not in EOS_REGISTRY:
            phase_labels.append('unknown')
            continue

        mat = EOS_REGISTRY[eos_name]
        if mat.get('format') != 'paleos_unified':
            phase_labels.append('unknown')
            continue

        cached = _ensure_unified_cache(mat['eos_file'], interp_cache)
        log_p = np.log10(np.clip(P, cached['p_min'], cached['p_max']))
        log_t = np.log10(max(T, 1.0))
        ip = int(np.argmin(np.abs(cached['unique_log_p'] - log_p)))
        it = int(np.argmin(np.abs(cached['unique_log_t'] - log_t)))
        phase = cached['phase_grid'][ip, it]
        phase_labels.append(str(phase) if phase else 'unknown')

    return phase_labels


def plot_planet_profile_single(
    radii,
    density,
    gravity,
    pressure,
    temperature,
    cmb_radius,
    cmb_mass,
    average_density,
    mass_enclosed,
    id_mass,
    layer_eos_config=None,
):
    """Generate a 6-panel interior structure plot.

    Panels: density, pressure, temperature (top row);
    gravity, phase state, mass enclosed (bottom row).

    Parameters
    ----------
    radii : numpy.ndarray
        Radial grid in m.
    density : numpy.ndarray
        Density profile in kg/m^3.
    gravity : numpy.ndarray
        Gravity profile in m/s^2.
    pressure : numpy.ndarray
        Pressure profile in Pa.
    temperature : numpy.ndarray
        Temperature profile in K.
    cmb_radius : float
        Core-mantle boundary radius in m.
    cmb_mass : float
        Mass enclosed at CMB in kg.
    average_density : float
        Average planet density in kg/m^3.
    mass_enclosed : numpy.ndarray
        Enclosed mass profile in kg.
    id_mass : int or None
        Planet mass identifier for filename.
    layer_eos_config : dict or None
        Per-layer EOS config for phase lookup. If None, phase panel
        shows "unknown" throughout.
    """
    radii_re = radii / earth_radius
    cmb_r = cmb_radius / earth_radius
    R_earth_val = radii[-1] / earth_radius
    M_earth_val = mass_enclosed[-1] / earth_mass
    xlim = (0, R_earth_val * 1.02)

    # Phase lookup
    if layer_eos_config is not None:
        phase_labels = _lookup_phases(
            radii, pressure, density, temperature, mass_enclosed, layer_eos_config
        )
    else:
        phase_labels = ['unknown'] * len(radii)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    def add_cmb(ax):
        ax.axvline(cmb_r, color='#cbd5e1', ls='--', lw=1)

    # Density
    ax = axes[0, 0]
    ax.plot(radii_re, density / 1000, color=_LINE_COLORS['density'], lw=2)
    add_cmb(ax)
    ax.set_ylabel(r'Density (10$^3$ kg m$^{-3}$)', fontsize=11)
    ax.set_title('Density', fontsize=12, fontweight='bold')
    ax.set_xlim(xlim)
    ax.grid(alpha=0.15)

    # Pressure
    ax = axes[0, 1]
    ax.semilogy(radii_re, pressure / 1e9, color=_LINE_COLORS['pressure'], lw=2)
    add_cmb(ax)
    ax.set_ylabel('Pressure (GPa)', fontsize=11)
    ax.set_title('Pressure', fontsize=12, fontweight='bold')
    ax.set_xlim(xlim)
    ax.grid(alpha=0.15)

    # Temperature
    ax = axes[0, 2]
    ax.plot(radii_re, temperature, color=_LINE_COLORS['temperature'], lw=2)
    add_cmb(ax)
    ax.set_ylabel('Temperature (K)', fontsize=11)
    ax.set_title('Temperature', fontsize=12, fontweight='bold')
    ax.set_xlim(xlim)
    ax.grid(alpha=0.15)

    # Gravity
    ax = axes[1, 0]
    ax.plot(radii_re, gravity, color=_LINE_COLORS['gravity'], lw=2)
    add_cmb(ax)
    ax.set_ylabel(r'Gravity (m s$^{-2}$)', fontsize=11)
    ax.set_xlabel(r'Radius ($R_\oplus$)', fontsize=11)
    ax.set_title('Gravity', fontsize=12, fontweight='bold')
    ax.set_xlim(xlim)
    ax.grid(alpha=0.15)

    # Phase state
    ax = axes[1, 1]
    for i in range(len(radii_re) - 1):
        c = _PHASE_COLORS.get(phase_labels[i], '#f3f4f6')
        ax.axvspan(radii_re[i], radii_re[i + 1], color=c, alpha=0.5, lw=0)
    add_cmb(ax)
    patches = []
    seen = []
    for p in phase_labels:
        if p not in seen and p in _PHASE_COLORS:
            patches.append(
                Patch(color=_PHASE_COLORS[p], alpha=0.5, label=_PHASE_PRETTY.get(p, p))
            )
            seen.append(p)
    if patches:
        ax.legend(handles=patches, fontsize=9, loc='center left')
    else:
        ax.text(
            0.5,
            0.5,
            'Phase data not available\nfor this EOS',
            transform=ax.transAxes,
            ha='center',
            va='center',
            fontsize=11,
            color='#9ca3af',
            style='italic',
        )
    ax.set_xlabel(r'Radius ($R_\oplus$)', fontsize=11)
    ax.set_title('Phase State', fontsize=12, fontweight='bold')
    ax.set_xlim(xlim)
    ax.set_yticks([])

    # Mass enclosed
    ax = axes[1, 2]
    ax.plot(radii_re, mass_enclosed / earth_mass, color=_LINE_COLORS['mass'], lw=2)
    add_cmb(ax)
    ax.set_ylabel(r'Mass enclosed ($M_\oplus$)', fontsize=11)
    ax.set_xlabel(r'Radius ($R_\oplus$)', fontsize=11)
    ax.set_title('Mass Enclosed', fontsize=12, fontweight='bold')
    ax.set_xlim(xlim)
    ax.grid(alpha=0.15)

    fig.suptitle(
        rf'{M_earth_val:.2f} $M_\oplus$,  '
        rf'R = {R_earth_val:.3f} $R_\oplus$,  '
        rf'$\bar{{\rho}}$ = {average_density:.0f} kg m$^{{-3}}$',
        fontsize=13,
        y=0.99,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fname = 'planet_profile.png' if id_mass is None else f'planet_profile{id_mass}.png'
    fig.savefig(
        os.path.join(ZALMOXIS_ROOT, 'output_files', fname), dpi=200, bbox_inches='tight'
    )
    plt.close(fig)
