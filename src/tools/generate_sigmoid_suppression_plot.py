"""Generate the sigmoid suppression figure for documentation.

Creates docs/img/sigmoid_suppression.png showing the condensed weight
sigma as a function of component density, with annotated reference points.

Usage:
    get_zalmoxis_root()=/path/to/Zalmoxis python -m src.tools.generate_sigmoid_suppression_plot
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np

from zalmoxis import get_zalmoxis_root

# Sigmoid parameters (H2O defaults)
RHO_MIN = 322.0  # kg/m^3, H2O critical density
RHO_SCALE = 50.0  # kg/m^3


def sigmoid(rho, rho_min=RHO_MIN, rho_scale=RHO_SCALE):
    """Condensed weight sigmoid."""
    arg = np.clip(-(rho - rho_min) / rho_scale, -500, 500)
    return 1.0 / (1.0 + np.exp(arg))


# Reference points: (label_top, label_bottom, density, sigma_approx)
# Format: component phase (P, T) on top, rho = value on bottom
REFERENCE_POINTS = [
    (r'H$_2$O vapor (1 bar, 3000 K)', r'$\rho$ = 0.1 kg/m$^3$', 0.1),
    (r'H$_2$O critical (647 K, 22.1 MPa)', r'$\rho$ = 322 kg/m$^3$', 322.0),
    (r'MgSiO$_3$ solid (1 GPa, 500 K)', r'$\rho$ = 4100 kg/m$^3$', 4100.0),
    (r'Fe solid (100 GPa, 5000 K)', r'$\rho$ = 13000 kg/m$^3$', 13000.0),
]


def main():
    """Generate the sigmoid suppression plot."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sigmoid curve
    rho = np.logspace(-1, 4.5, 2000)
    sigma = sigmoid(rho)
    ax.plot(rho, sigma, 'k-', lw=2.5)

    # Shaded regions
    ax.axvspan(rho[0], RHO_MIN, alpha=0.08, color='#ef4444')
    ax.axvspan(RHO_MIN, rho[-1], alpha=0.08, color='#22c55e')

    # Region labels
    ax.text(
        3,
        0.15,
        'Suppressed\n(vapor)',
        fontsize=11,
        color='#dc2626',
        ha='center',
        style='italic',
    )
    ax.text(
        12000,
        0.35,
        'Included\n(condensed)',
        fontsize=11,
        color='#16a34a',
        ha='center',
        style='italic',
    )

    # Reference point annotations with manually tuned positions
    arrow_props = dict(arrowstyle='->', color='#475569', lw=1.2)
    # (label_top, label_bottom, rho_val, text_x, text_y, ha, va)
    annotation_layout = [
        (REFERENCE_POINTS[0], 1.5, 0.28, 'center', 'bottom'),  # H2O vapor
        (REFERENCE_POINTS[1], 60, 0.72, 'center', 'bottom'),  # H2O critical
        (REFERENCE_POINTS[2], 2500, 1.05, 'center', 'bottom'),  # MgSiO3
        (REFERENCE_POINTS[3], 15000, 0.72, 'center', 'top'),  # Fe
    ]
    for (label_top, label_bottom, rho_val), tx, ty, ha, va in annotation_layout:
        sig = float(sigmoid(np.array([rho_val]))[0])
        combined = f'{label_top}\n{label_bottom}'
        ax.annotate(
            combined,
            xy=(rho_val, sig),
            xytext=(tx, ty),
            fontsize=9,
            ha=ha,
            va=va,
            arrowprops=arrow_props,
        )
        ax.plot(rho_val, sig, 'o', color='#475569', ms=5, zorder=5)

    # Parameters annotation
    ax.text(
        0.98,
        0.02,
        rf'$\rho_{{\mathrm{{min}}}}$ = {RHO_MIN} kg/m$^3$,  '
        rf'$\rho_{{\mathrm{{scale}}}}$ = {RHO_SCALE} kg/m$^3$',
        transform=ax.transAxes,
        fontsize=10,
        ha='right',
        va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#cbd5e1', alpha=0.9),
    )

    ax.set_xscale('log')
    ax.set_xlabel(r'Component density $\rho_i$ (kg/m$^3$)', fontsize=12)
    ax.set_ylabel(r'Condensed weight $\sigma_i$', fontsize=12)
    ax.set_xlim(0.05, 3e4)
    ax.set_ylim(-0.05, 1.15)
    ax.grid(alpha=0.15)
    ax.tick_params(labelsize=10)

    fig.tight_layout()
    outpath = os.path.join(get_zalmoxis_root(), 'docs', 'img', 'sigmoid_suppression.png')
    fig.savefig(outpath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {outpath}')


if __name__ == '__main__':
    main()
