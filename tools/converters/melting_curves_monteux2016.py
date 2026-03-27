"""Analytic solidus and liquidus curves from Monteux et al. (2016, EPSL 448, 140-149).

Implements Equations 10-13, which use modified Simon-Glatzel parameterizations
fitted to experimental data:
  - P <= 20 GPa: Herzberg & Zhang (1996) chondritic mantle experiments
  - P >  20 GPa solidus: Andrault et al. (2011) A-chondritic solidus
  - P >  20 GPa liquidus: composition-dependent (F-peridotitic or A-chondritic)

Two composition models are provided:
  - F-peridotitic (33% enstatite + 56% forsterite + 7% fayalite + 3% anorthite + 0.7% diopside)
  - A-chondritic  (62% enstatite + 24% forsterite + 8% fayalite + 4% anorthite + 2% diopside)

Usage
-----
    from tools.melting_curves_monteux2016 import solidus, liquidus

    T_sol = solidus(P)             # P in Pa, returns T in K
    T_liq = liquidus(P, model='A-chondritic')

Or run directly to generate comparison plots:
    python src/tools/melting_curves_monteux2016.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Transition pressure between low-P and high-P parameterizations [Pa]
P_TRANSITION = 20.0e9  # 20 GPa


# ---------------------------------------------------------------------------
# Eq. 10-12: Solidus
# ---------------------------------------------------------------------------


def _solidus_low(P: np.ndarray) -> np.ndarray:
    """Solidus for P <= 20 GPa (Herzberg & Zhang 1996, Eq. 10).

    Parameters
    ----------
    P : array-like
        Pressure in Pa.

    Returns
    -------
    T : ndarray
        Temperature in K.
    """
    return 1661.2 * (P / 1.336e9 + 1.0) ** (1.0 / 7.437)


def _solidus_high(P: np.ndarray) -> np.ndarray:
    """Solidus for P > 20 GPa (Andrault et al. 2011 A-chondritic, Eq. 12).

    Parameters
    ----------
    P : array-like
        Pressure in Pa.

    Returns
    -------
    T : ndarray
        Temperature in K.
    """
    return 2081.8 * (P / (101.69e9) + 1.0) ** (1.0 / 1.226)


def solidus(P: float | np.ndarray) -> np.ndarray:
    """Solidus temperature as a function of pressure.

    Piecewise: Eq. 10 (P <= 20 GPa) and Eq. 12 (P > 20 GPa).
    Both composition models share the same solidus.

    Parameters
    ----------
    P : float or array-like
        Pressure in Pa (must be >= 0).

    Returns
    -------
    T : ndarray
        Solidus temperature in K.
    """
    P = np.atleast_1d(np.asarray(P, dtype=float))
    T = np.empty_like(P)
    lo = P <= P_TRANSITION
    hi = ~lo
    T[lo] = _solidus_low(P[lo])
    T[hi] = _solidus_high(P[hi])
    return T


# ---------------------------------------------------------------------------
# Eq. 11, 13: Liquidus
# ---------------------------------------------------------------------------


def _liquidus_low(P: np.ndarray) -> np.ndarray:
    """Liquidus for P <= 20 GPa (Herzberg & Zhang 1996, Eq. 11).

    Parameters
    ----------
    P : array-like
        Pressure in Pa.

    Returns
    -------
    T : ndarray
        Temperature in K.
    """
    return 1982.1 * (P / 6.594e9 + 1.0) ** (1.0 / 5.374)


def _liquidus_high(P: np.ndarray, model: str = 'A-chondritic') -> np.ndarray:
    """Liquidus for P > 20 GPa (Eq. 13).

    Parameters
    ----------
    P : array-like
        Pressure in Pa.
    model : str
        'F-peridotitic' (Fiquet et al. 2010) or 'A-chondritic' (Andrault et al. 2011).

    Returns
    -------
    T : ndarray
        Temperature in K.
    """
    if model == 'F-peridotitic':
        c1, c2, c3 = 78.74, 4.054e6, 2.44
    elif model == 'A-chondritic':
        c1, c2, c3 = 2006.8, 34.65e9, 1.844
    else:
        raise ValueError(f"Unknown model '{model}'. Use 'F-peridotitic' or 'A-chondritic'.")
    return c1 * (P / c2 + 1.0) ** (1.0 / c3)


def liquidus(P: float | np.ndarray, model: str = 'A-chondritic') -> np.ndarray:
    """Liquidus temperature as a function of pressure.

    Piecewise: Eq. 11 (P <= 20 GPa) and Eq. 13 (P > 20 GPa).

    Parameters
    ----------
    P : float or array-like
        Pressure in Pa (must be >= 0).
    model : str
        'F-peridotitic' or 'A-chondritic' (default).

    Returns
    -------
    T : ndarray
        Liquidus temperature in K.
    """
    P = np.atleast_1d(np.asarray(P, dtype=float))
    T = np.empty_like(P)
    lo = P <= P_TRANSITION
    hi = ~lo
    T[lo] = _liquidus_low(P[lo])
    T[hi] = _liquidus_high(P[hi], model=model)
    return T


# ---------------------------------------------------------------------------
# Convenience: both curves at once
# ---------------------------------------------------------------------------


def melting_curves(
    P: float | np.ndarray, model: str = 'A-chondritic'
) -> tuple[np.ndarray, np.ndarray]:
    """Return (T_solidus, T_liquidus) at given pressure(s).

    Parameters
    ----------
    P : float or array-like
        Pressure in Pa.
    model : str
        'F-peridotitic' or 'A-chondritic'.

    Returns
    -------
    T_sol, T_liq : ndarray
        Solidus and liquidus temperatures in K.
    """
    return solidus(P), liquidus(P, model=model)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _load_existing_table(filepath):
    """Load an existing tabulated melting curve (P [Pa], T [K])."""
    data = np.loadtxt(filepath, comments='#')
    return data[:, 0], data[:, 1]


def plot_melting_curves(output_dir: str | None = None):
    """Plot Monteux+2016 melting curves and compare with existing Zalmoxis tables.

    Generates two figures:
    1. T vs P for both composition models (F-peridotitic and A-chondritic)
    2. Comparison of analytic curves with existing tabulated curves in Zalmoxis
    """
    if output_dir is None:
        zalmoxis_root = os.environ.get(
            'ZALMOXIS_ROOT',
            os.path.dirname(os.path.abspath(__file__)) + '/../..',
        )
        output_dir = os.path.join(zalmoxis_root, 'output')
    os.makedirs(output_dir, exist_ok=True)

    P_Pa = np.linspace(0, 140e9, 2000)  # 0 to 140 GPa
    P_GPa = P_Pa / 1e9

    T_sol = solidus(P_Pa)
    T_liq_A = liquidus(P_Pa, model='A-chondritic')
    T_liq_F = liquidus(P_Pa, model='F-peridotitic')

    # ── Figure 1: Both composition models ──────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # A-chondritic
    ax1.plot(P_GPa, T_sol, 'g-', lw=2, label='Solidus')
    ax1.plot(P_GPa, T_liq_A, 'r-', lw=2, label='Liquidus (A-chondritic)')
    ax1.fill_between(P_GPa, T_sol, T_liq_A, alpha=0.15, color='orange', label='Mushy zone')
    ax1.axvline(20, color='grey', ls=':', lw=0.8, alpha=0.6)
    ax1.text(21, 1500, '20 GPa\ntransition', fontsize=8, color='grey')
    ax1.set_xlabel('Pressure [GPa]', fontsize=12)
    ax1.set_ylabel('Temperature [K]', fontsize=12)
    ax1.set_title('A-chondritic model', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xlim(0, 140)
    ax1.grid(True, alpha=0.3)

    # F-peridotitic
    ax2.plot(P_GPa, T_sol, 'g-', lw=2, label='Solidus')
    ax2.plot(P_GPa, T_liq_F, 'r-', lw=2, label='Liquidus (F-peridotitic)')
    ax2.fill_between(P_GPa, T_sol, T_liq_F, alpha=0.15, color='orange', label='Mushy zone')
    ax2.axvline(20, color='grey', ls=':', lw=0.8, alpha=0.6)
    ax2.text(21, 1500, '20 GPa\ntransition', fontsize=8, color='grey')
    ax2.set_xlabel('Pressure [GPa]', fontsize=12)
    ax2.set_title('F-peridotitic model', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_xlim(0, 140)
    ax2.grid(True, alpha=0.3)

    fig.suptitle('Monteux et al. (2016) melting curves, Eqs. 10-13', fontsize=14, y=1.02)
    fig.tight_layout()
    fname1 = os.path.join(output_dir, 'monteux2016_melting_curves.png')
    fig.savefig(fname1, dpi=200, bbox_inches='tight')
    print(f'Saved: {fname1}')

    # ── Figure 2: Comparison with existing Zalmoxis tables ─────────────
    zalmoxis_root = os.environ.get(
        'ZALMOXIS_ROOT',
        os.path.dirname(os.path.abspath(__file__)) + '/../..',
    )
    sol_file = os.path.join(zalmoxis_root, 'data', 'melting_curves_Monteux-600', 'solidus.dat')
    liq_file = os.path.join(zalmoxis_root, 'data', 'melting_curves_Monteux-600', 'liquidus.dat')

    if os.path.isfile(sol_file) and os.path.isfile(liq_file):
        tab_P_sol, tab_T_sol = _load_existing_table(sol_file)
        tab_P_liq, tab_T_liq = _load_existing_table(liq_file)

        fig2, ax = plt.subplots(figsize=(8, 6))

        # existing tables
        mask_sol = tab_P_sol <= 140e9
        mask_liq = tab_P_liq <= 140e9
        ax.plot(
            tab_P_sol[mask_sol] / 1e9,
            tab_T_sol[mask_sol],
            'g--',
            lw=2,
            alpha=0.7,
            label='Solidus (Zalmoxis table)',
        )
        ax.plot(
            tab_P_liq[mask_liq] / 1e9,
            tab_T_liq[mask_liq],
            'r--',
            lw=2,
            alpha=0.7,
            label='Liquidus (Zalmoxis table)',
        )

        # analytic (A-chondritic)
        ax.plot(P_GPa, T_sol, 'g-', lw=2, label='Solidus (Monteux+2016 Eq. 10/12)')
        ax.plot(P_GPa, T_liq_A, 'r-', lw=2, label='Liquidus A-chon. (Eq. 11/13)')
        ax.plot(P_GPa, T_liq_F, 'b-', lw=1.5, label='Liquidus F-peri. (Eq. 11/13)')

        ax.axvline(20, color='grey', ls=':', lw=0.8, alpha=0.6)
        ax.set_xlabel('Pressure [GPa]', fontsize=12)
        ax.set_ylabel('Temperature [K]', fontsize=12)
        ax.set_title('Monteux+2016 analytic vs. existing Zalmoxis melting curves', fontsize=12)
        ax.legend(fontsize=9, loc='upper left')
        ax.set_xlim(0, 140)
        ax.grid(True, alpha=0.3)

        fig2.tight_layout()
        fname2 = os.path.join(output_dir, 'monteux2016_vs_zalmoxis_tables.png')
        fig2.savefig(fname2, dpi=200, bbox_inches='tight')
        print(f'Saved: {fname2}')
    else:
        print('Warning: existing table files not found, skipping comparison plot.')

    print('Done.')


if __name__ == '__main__':
    plot_melting_curves()
