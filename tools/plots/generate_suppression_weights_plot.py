"""Generate the 3-panel suppression weight figure for documentation.

Creates docs/img/suppression_weights.png showing:
  Left:   density-based sigmoid for H2
  Center: binodal sigmoid at fixed P for different T_scale values
  Right:  combined sigma_total along a representative planetary adiabat

Usage:
    get_zalmoxis_root()=/path/to/Zalmoxis python -m src.tools.generate_suppression_weights_plot
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from zalmoxis import get_zalmoxis_root

# ── H2 density sigmoid parameters ──
H2_RHO_MIN = 30.0  # kg/m^3
H2_RHO_SCALE = 10.0  # kg/m^3


def density_sigmoid(rho, rho_min, rho_scale):
    """Condensed weight sigmoid (vectorized)."""
    arg = np.clip(-(rho - rho_min) / rho_scale, -500, 500)
    return 1.0 / (1.0 + np.exp(arg))


def binodal_sigmoid(T, T_binodal, T_scale):
    """Binodal suppression sigmoid (vectorized)."""
    if T_binodal <= 0:
        return np.ones_like(T)
    arg = np.clip((T - T_binodal) / T_scale, -500, 500)
    return 1.0 / (1.0 + np.exp(-arg))


def rogers2025_binodal_T(P_GPa, x_H2=0.60):
    """Simplified Rogers+2025 binodal temperature at fixed composition.

    Parameters
    ----------
    P_GPa : float
        Pressure in GPa.
    x_H2 : float
        H2 mole fraction. Default 0.60 (3% by mass).

    Returns
    -------
    float
        Binodal temperature in K. 0 if always miscible.
    """
    T_c = 4223.0 * (1.0 - P_GPa / 35.0)
    if T_c <= 0:
        return 0.0
    # Rough approximation of the binodal at fixed x
    # At x=0.60, T_b is close to T_c (near peak)
    return T_c * 0.95


def main():
    """Generate the 3-panel suppression weight plot."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Left panel: density sigmoid for H2 ──
    ax = axes[0]
    rho = np.linspace(0, 200, 1000)
    sigma_d = density_sigmoid(rho, H2_RHO_MIN, H2_RHO_SCALE)
    ax.plot(rho, sigma_d, 'k-', lw=2)

    # Shaded regions on correct sides of rho_min
    ax.axvspan(0, H2_RHO_MIN, alpha=0.08, color='#ef4444')
    ax.axvspan(H2_RHO_MIN, 200, alpha=0.08, color='#22c55e')
    ax.text(
        H2_RHO_MIN * 0.35,
        0.5,
        'Gas-like',
        fontsize=10,
        color='#dc2626',
        ha='center',
        va='center',
        style='italic',
    )
    ax.text(
        H2_RHO_MIN + (200 - H2_RHO_MIN) * 0.5,
        0.5,
        'Condensed',
        fontsize=10,
        color='#16a34a',
        ha='center',
        va='center',
        style='italic',
    )

    # rho_min marker
    ax.axvline(H2_RHO_MIN, color='#475569', ls=':', lw=1)
    ax.annotate(
        rf'$\rho_{{\mathrm{{min}}}}$ = {H2_RHO_MIN:.0f} kg/m$^3$',
        xy=(H2_RHO_MIN, 0.5),
        xytext=(H2_RHO_MIN + 50, 0.35),
        fontsize=9,
        ha='left',
        arrowprops=dict(arrowstyle='->', color='#475569', lw=1),
    )

    ax.set_xlabel(r'Density (kg/m$^3$)', fontsize=11)
    ax.set_ylabel(r'$\sigma_{\mathrm{density}}$', fontsize=12)
    ax.set_title(r'Density suppression $\sigma_{\mathrm{density}}(\rho)$', fontsize=11)
    ax.set_xlim(0, 200)
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.15)

    # ── Center panel: binodal sigmoid at fixed P ──
    ax = axes[1]
    T_arr = np.linspace(1500, 5500, 1000)
    P_fixed = 5.0  # GPa
    T_b = rogers2025_binodal_T(P_fixed)

    colors_scales = [
        ('#2563eb', 50.0, r'$T_{\mathrm{scale}}$ = 50 K (default)'),
        ('#f59e0b', 100.0, r'$T_{\mathrm{scale}}$ = 100 K'),
        ('#22c55e', 200.0, r'$T_{\mathrm{scale}}$ = 200 K'),
    ]
    for color, T_scale, label in colors_scales:
        sigma_b = binodal_sigmoid(T_arr, T_b, T_scale)
        ax.plot(T_arr, sigma_b, color=color, lw=2, label=label)

    # Binodal temperature marker
    ax.axvline(T_b, color='#475569', ls=':', lw=1)
    ax.annotate(
        rf'$T_{{\mathrm{{b}}}}$ = {T_b:.0f} K',
        xy=(T_b, 0.5),
        xytext=(T_b - 600, 0.3),
        fontsize=9,
        ha='right',
        arrowprops=dict(arrowstyle='->', color='#475569', lw=1),
    )

    # Region labels
    ax.text(2000, 0.15, 'Immiscible', fontsize=10, color='#6b7280', style='italic')
    ax.text(4800, 0.85, 'Miscible', fontsize=10, color='#6b7280', style='italic')

    # Shaded regions
    ax.axvspan(T_arr[0], T_b, alpha=0.05, color='#ef4444')
    ax.axvspan(T_b, T_arr[-1], alpha=0.05, color='#22c55e')

    ax.set_xlabel('Temperature (K)', fontsize=11)
    ax.set_ylabel(r'$\sigma_{\mathrm{binodal}}$', fontsize=12)
    ax.set_title(
        rf'Binodal suppression $\sigma_{{\mathrm{{binodal}}}}(T)$ at P = {P_fixed:.0f} GPa',
        fontsize=11,
    )
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim(-0.05, 1.1)
    ax.grid(alpha=0.15)

    # ── Right panel: combined weight along a schematic sub-Neptune adiabat ──
    ax = axes[2]

    # Schematic adiabat with T_surf = 1500 K to clearly separate the
    # density transition (~0.1 GPa) from the binodal transition (~10 GPa).
    # Between these two pressures, H2 is dense enough to count but still
    # immiscible with silicate — this is the "dense but immiscible" regime.
    P_GPa = np.logspace(-3, 1.8, 800)  # 1 MPa to ~60 GPa
    P_Pa = P_GPa * 1e9
    T_surf = 1500.0
    T_adiabat = T_surf * (P_GPa / P_GPa[0]) ** 0.28

    # H2 density (ideal gas: rho = P mu / R T)
    rho_H2 = P_Pa * 2e-3 / (8.314 * T_adiabat)

    sigma_density = density_sigmoid(rho_H2, H2_RHO_MIN, H2_RHO_SCALE)
    sigma_binodal = np.array(
        [
            binodal_sigmoid(np.array([T]), rogers2025_binodal_T(P), 50.0)[0]
            for T, P in zip(T_adiabat, P_GPa)
        ]
    )
    sigma_total = sigma_density * sigma_binodal

    ax.plot(
        P_GPa,
        sigma_density,
        '--',
        color='#2563eb',
        lw=2,
        label=r'$\sigma_{\mathrm{density}}$',
    )
    ax.plot(
        P_GPa,
        sigma_binodal,
        '--',
        color='#f59e0b',
        lw=2,
        label=r'$\sigma_{\mathrm{binodal}}$',
    )
    ax.plot(P_GPa, sigma_total, 'k-', lw=2.5, label=r'$\sigma_{\mathrm{total}}$')

    # Shade the three regimes
    p_dens_half = P_GPa[np.argmin(np.abs(sigma_density - 0.5))]
    p_bino_half = P_GPa[np.argmin(np.abs(sigma_binodal - 0.5))]
    ax.axvspan(P_GPa[0], p_dens_half, alpha=0.06, color='#ef4444')
    ax.axvspan(p_dens_half, p_bino_half, alpha=0.06, color='#fbbf24')
    ax.axvspan(p_bino_half, P_GPa[-1], alpha=0.06, color='#22c55e')

    # Region labels
    ax.text(
        0.003,
        0.5,
        'H$_2$\nexcluded',
        fontsize=8,
        color='#dc2626',
        ha='center',
        va='center',
        style='italic',
    )
    ax.text(
        np.sqrt(p_dens_half * p_bino_half),
        0.5,
        'dense but\nimmiscible',
        fontsize=8,
        color='#b45309',
        ha='center',
        va='center',
        style='italic',
    )
    ax.text(
        25,
        0.5,
        'H$_2$\nincluded',
        fontsize=8,
        color='#16a34a',
        ha='center',
        va='center',
        style='italic',
    )

    ax.set_xscale('log')
    ax.set_xlabel('Pressure (GPa)', fontsize=11)
    ax.set_ylabel('Suppression weight', fontsize=12)
    ax.set_title(
        r'$\sigma_{\mathrm{total}}$ along a schematic adiabat',
        fontsize=11,
    )
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlim(P_GPa[0], P_GPa[-1])
    ax.grid(alpha=0.15)

    fig.suptitle(
        r'Suppression weight behavior for H$_2$ in a silicate mantle',
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    outpath = os.path.join(get_zalmoxis_root(), 'docs', 'img', 'suppression_weights.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {outpath}')


if __name__ == '__main__':
    main()
