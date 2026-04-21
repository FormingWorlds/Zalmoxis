"""Plot radial interior profiles from a Zalmoxis parameter grid.

Reads the ``<label>.npz`` files produced by ``run_grid`` when
``[output].save_profiles = true`` and overlays density, pressure,
temperature, and gravity vs. radius across all converged grid points,
coloured by the primary sweep parameter.

Usage
-----
From the terminal::

    python -m src.tools.plot_grid_profiles output_files/grid_mass_radius
    python -m src.tools.plot_grid_profiles output_files/grid_mass_radius -o profiles.pdf
    python -m src.tools.plot_grid_profiles output_files/grid_mass_radius --colour-by surface_temperature

The script is also importable from Python / a notebook::

    from src.tools.plot_grid_profiles import plot_grid_profiles
    plot_grid_profiles("output_files/grid_mass_radius")
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm, Normalize

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that are always present in grid_summary.csv (not sweep parameters).
# Kept in sync with src/tools/plot_grid.py::_FIXED_COLUMNS.
_FIXED_COLUMNS = {
    'label',
    'R_earth',
    'M_earth',
    'converged',
    'converged_pressure',
    'converged_density',
    'converged_mass',
    'time_s',
    'error',
}

_AXIS_LABELS = {
    'planet_mass': r'Planet mass ($M_\oplus$)',
    'surface_temperature': r'Surface temperature (K)',
    'center_temperature': r'Centre temperature (K)',
    'core_mass_fraction': 'Core mass fraction',
    'mantle_mass_fraction': 'Mantle mass fraction',
    'mantle': 'Mantle EOS',
    'core': 'Core EOS',
    'num_layers': 'Number of radial layers',
    'condensed_rho_min': r'Condensed $\rho_\mathrm{min}$ (kg m$^{-3}$)',
    'mushy_zone_factor': 'Mushy zone factor',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_grid_dir(path):
    """Accept either a grid output directory or its grid_summary.csv file."""
    if os.path.isdir(path):
        return path
    if os.path.isfile(path) and os.path.basename(path) == 'grid_summary.csv':
        return os.path.dirname(os.path.abspath(path))
    raise ValueError(f'Expected a grid output directory or grid_summary.csv file, got: {path}')


def _load_summary(grid_dir):
    """Load grid_summary.csv as a list of dict rows."""
    csv_path = os.path.join(grid_dir, 'grid_summary.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'grid_summary.csv not found in {grid_dir}')
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f'grid_summary.csv in {grid_dir} is empty')
    return rows


def _detect_sweep_params(rows):
    """Return the sorted list of column names that are sweep parameters."""
    return sorted(set(rows[0].keys()) - _FIXED_COLUMNS)


def _try_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _choose_colour_param(sweep_params, rows, explicit=None):
    """Pick a sweep parameter to colour lines by."""
    if explicit is not None:
        if explicit not in sweep_params:
            raise ValueError(
                f"Colour parameter '{explicit}' not in sweep parameters {sweep_params}"
            )
        return explicit
    # Prefer the first numerically-valued sweep parameter.
    for p in sweep_params:
        if _try_float(rows[0].get(p)) is not None:
            return p
    return sweep_params[0] if sweep_params else None


def _converged(row):
    return str(row.get('converged', '')).strip().lower() == 'true'


def _load_profile(grid_dir, label):
    """Load a profile archive into a plain dict, closing the NpzFile.

    Keeping the ``np.load(...)`` handle open for the lifetime of the
    caller leaks one file descriptor per grid point; on 1000+ point
    sweeps that can exhaust the process limit. Copy arrays into memory
    and let the context manager close the archive immediately.
    """
    npz_path = os.path.join(grid_dir, f'{label}.npz')
    if not os.path.isfile(npz_path):
        return None
    with np.load(npz_path) as data:
        return {key: np.array(data[key], copy=True) for key in data.files}


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------
def plot_grid_profiles(
    grid_dir,
    out=None,
    dpi=200,
    colour_by=None,
    log_pressure=False,
):
    """Overlay radial profiles of every converged grid point in a 2x2 figure.

    Parameters
    ----------
    grid_dir : str
        Path to the grid output directory (or directly to grid_summary.csv).
    out : str, optional
        Output image path. Extension determines format (e.g. ``.pdf``,
        ``.png``). Defaults to ``profiles_vs_radius.pdf`` next to the CSV.
    dpi : int
        Resolution for raster outputs. Vector formats (PDF, SVG) ignore this.
    colour_by : str, optional
        Sweep-parameter column to colour lines by. Defaults to the first
        numeric sweep parameter (for the mass-radius grid: ``planet_mass``).
    log_pressure : bool
        If True, plot pressure on a log y-axis. Default False (linear).
    """
    grid_dir = _resolve_grid_dir(grid_dir)
    rows = _load_summary(grid_dir)
    sweep_params = _detect_sweep_params(rows)
    if not sweep_params:
        raise RuntimeError('No sweep parameters detected in grid_summary.csv')
    colour_param = _choose_colour_param(sweep_params, rows, colour_by)

    # Collect converged profiles along with their colour-parameter value.
    profiles = []
    skipped = []
    for row in rows:
        label = row['label']
        if not _converged(row):
            skipped.append((label, 'not converged'))
            continue
        data = _load_profile(grid_dir, label)
        if data is None:
            skipped.append((label, 'missing .npz (run with save_profiles = true)'))
            continue
        c_val = _try_float(row.get(colour_param))
        profiles.append((label, row, data, c_val))

    if not profiles:
        raise RuntimeError(
            f'No converged profiles with saved .npz files found in {grid_dir}. '
            f'Re-run run_grid with [output].save_profiles = true.'
        )

    # Colour mapping over the sweep values (numeric if possible).
    c_vals = [p[3] for p in profiles if p[3] is not None]
    if c_vals and len(set(c_vals)) > 1:
        if min(c_vals) > 0 and max(c_vals) / min(c_vals) >= 10:
            norm = LogNorm(vmin=min(c_vals), vmax=max(c_vals))
        else:
            norm = Normalize(vmin=min(c_vals), vmax=max(c_vals))
        cmap = plt.get_cmap('viridis')
        numeric_colour = True
    else:
        # Fall back to a discrete palette indexed by grid-point order.
        norm = None
        cmap = plt.get_cmap('tab10')
        numeric_colour = False

    def _colour(val, idx):
        if numeric_colour and val is not None:
            return cmap(norm(val))
        return cmap(idx % cmap.N)

    # 2x2 figure.
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    (ax_rho, ax_P), (ax_T, ax_g) = axes
    panels = [
        (ax_rho, 'density', 1.0, r'Density (kg m$^{-3}$)', '(a)'),
        (ax_P, 'pressure', 1e-9, 'Pressure (GPa)', '(b)'),
        (ax_T, 'temperature', 1.0, 'Temperature (K)', '(c)'),
        (ax_g, 'gravity', 1.0, r'Gravity (m s$^{-2}$)', '(d)'),
    ]

    for idx, (label, row, data, c_val) in enumerate(profiles):
        r_km = np.asarray(data['radii']) / 1e3
        col = _colour(c_val, idx)
        for ax, key, scale, _, _ in panels:
            y = np.asarray(data[key]) * scale
            # Log axes cannot render P <= 0; padded surface shells can
            # hit exactly zero. Mask those so matplotlib does not warn
            # and leave the rest of the trajectory plotted cleanly.
            if key == 'pressure' and log_pressure:
                mask = y > 0
                ax.plot(r_km[mask], y[mask], color=col, lw=1.2)
            else:
                ax.plot(r_km, y, color=col, lw=1.2)

    for ax, _, _, ylabel, tag in panels:
        ax.set_xlabel('Radius (km)')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.text(
            0.03,
            0.95,
            tag,
            transform=ax.transAxes,
            fontsize=12,
            fontweight='bold',
            va='top',
            ha='left',
        )

    if log_pressure:
        ax_P.set_yscale('log')

    # Colour bar or legend.
    colour_label = _AXIS_LABELS.get(colour_param, colour_param)
    if numeric_colour:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=axes, location='right', shrink=0.8, pad=0.02)
        cbar.set_label(colour_label)
    else:
        handles = []
        for idx, (label, row, _data, _c) in enumerate(profiles):
            value = row.get(colour_param, label)
            handles.append(
                plt.Line2D([0], [0], color=_colour(None, idx), lw=1.5, label=str(value))
            )
        fig.legend(
            handles=handles,
            title=colour_label,
            loc='center right',
            bbox_to_anchor=(1.0, 0.5),
            frameon=False,
        )

    n_total = len(rows)
    n_plotted = len(profiles)
    fig.suptitle(
        f'Zalmoxis grid profiles, {n_plotted}/{n_total} converged '
        f'({os.path.basename(grid_dir.rstrip("/"))})'
    )

    if out is None:
        out = os.path.join(grid_dir, 'profiles_vs_radius.pdf')
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved {out}')
    if skipped:
        print(f'Skipped {len(skipped)} grid point(s):')
        for label, reason in skipped:
            print(f'  {label}: {reason}')

    return out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _build_parser():
    parser = argparse.ArgumentParser(
        description=(
            'Overlay radial interior profiles (density, pressure, '
            'temperature, gravity) across a Zalmoxis parameter grid.'
        )
    )
    parser.add_argument(
        'path',
        help='Grid output directory, or path to grid_summary.csv.',
    )
    parser.add_argument(
        '-o',
        '--output',
        default=None,
        help=(
            'Output image path (extension selects format, e.g. .pdf, .png). '
            'Default: <grid_dir>/profiles_vs_radius.pdf'
        ),
    )
    parser.add_argument(
        '--colour-by',
        '--color-by',
        dest='colour_by',
        default=None,
        help=(
            'Sweep parameter to colour lines by. Defaults to the first '
            'numeric sweep parameter in the grid (e.g. planet_mass).'
        ),
    )
    parser.add_argument(
        '--log-pressure',
        action='store_true',
        help='Plot pressure on a logarithmic y-axis (default: linear).',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='Raster DPI (ignored for vector formats). Default 200.',
    )
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    plot_grid_profiles(
        grid_dir=args.path,
        out=args.output,
        dpi=args.dpi,
        colour_by=args.colour_by,
        log_pressure=args.log_pressure,
    )
