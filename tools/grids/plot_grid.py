"""Plot results from a Zalmoxis parameter grid.

Reads the ``grid_summary.csv`` produced by ``run_grid`` and generates
publication-quality figures. The default is a mass-radius diagram, but
any pair of columns can be selected for the axes.

Usage
-----
From the terminal::

    python -m src.tools.plot_grid output/grid_mass_radius
    python -m src.tools.plot_grid output/grid_mass_radius/grid_summary.csv
    python -m src.tools.plot_grid grid_summary.csv -x surface_temperature -y R_earth
    python -m src.tools.plot_grid grid_summary.csv --single-panel

The script is also importable and runnable from an IDE: call
``plot_grid_summary()`` directly.
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns that are always present in grid_summary.csv (not sweep parameters)
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

# Soft color palette (consistent with plot_profiles.py style)
_COLORS = [
    '#60a5fa',  # blue
    '#f87171',  # red
    '#4ade80',  # green
    '#c084fc',  # purple
    '#22d3ee',  # cyan
    '#facc15',  # yellow
    '#fb923c',  # orange
    '#a78bfa',  # violet
    '#f472b6',  # pink
    '#34d399',  # emerald
]

# Pretty axis labels for known columns
_AXIS_LABELS = {
    'R_earth': r'Radius ($R_\oplus$)',
    'M_earth': r'Mass ($M_\oplus$)',
    'planet_mass': r'Planet mass ($M_\oplus$)',
    'surface_temperature': r'Surface temperature (K)',
    'center_temperature': r'Center temperature (K)',
    'core_mass_fraction': 'Core mass fraction',
    'mantle_mass_fraction': 'Mantle mass fraction',
    'num_layers': 'Number of radial layers',
    'time_s': 'Wall time (s)',
    'condensed_rho_min': r'Condensed $\rho_\mathrm{min}$ (kg m$^{-3}$)',
    'condensed_rho_scale': r'Condensed $\rho$ scale (kg m$^{-3}$)',
    'binodal_T_scale': r'Binodal $T$ scale (K)',
    'mushy_zone_factor': 'Mushy zone factor',
}


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------
def _load_grid_csv(csv_path):
    """Load a grid_summary.csv into a list of dicts.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file.

    Returns
    -------
    rows : list[dict]
        Each row as a dict with string keys and auto-typed values.
    sweep_params : list[str]
        Names of the sweep parameters (columns that are not fixed outputs).
    """
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        raw_rows = list(reader)

    if not raw_rows:
        raise ValueError(f'No data rows in {csv_path}')

    # Identify sweep parameters: any column not in _FIXED_COLUMNS
    all_columns = list(raw_rows[0].keys())
    sweep_params = [c for c in all_columns if c not in _FIXED_COLUMNS]

    # Auto-type values
    rows = []
    for raw in raw_rows:
        row = {}
        for k, v in raw.items():
            if v in ('', None):
                row[k] = None
            elif v in ('True', 'true'):
                row[k] = True
            elif v in ('False', 'false'):
                row[k] = False
            else:
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
        rows.append(row)

    return rows, sweep_params


def _resolve_csv_path(path):
    """Accept either a directory or a CSV file path.

    Parameters
    ----------
    path : str
        Path to a directory containing grid_summary.csv, or to the CSV itself.

    Returns
    -------
    str
        Absolute path to the CSV file.
    """
    if os.path.isdir(path):
        csv_path = os.path.join(path, 'grid_summary.csv')
    else:
        csv_path = path
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'Grid summary CSV not found: {csv_path}')
    return os.path.abspath(csv_path)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_grid_summary(
    csv_path,
    x='planet_mass',
    y='R_earth',
    group_by=None,
    single_panel=False,
    output=None,
    dpi=200,
):
    """Plot grid results from a Zalmoxis grid_summary.csv.

    Parameters
    ----------
    csv_path : str
        Path to ``grid_summary.csv`` or a directory containing it.
    x : str
        Column name for the x-axis. Default: ``'planet_mass'``.
    y : str
        Column name for the y-axis. Default: ``'R_earth'``.
    group_by : str or None
        Sweep parameter to group by (separate subplots, or color-coded
        series if ``single_panel=True``). If None, auto-detected from
        sweep parameters (first sweep param that is not ``x``).
    single_panel : bool
        If True, plot all groups on a single panel with a legend.
        If False (default), use one subplot per group value.
    output : str or None
        Output file path. If None, saves to the same directory as the
        CSV with a descriptive filename.
    dpi : int
        Resolution for the saved figure.

    Returns
    -------
    str
        Path to the saved figure.
    """
    csv_path = _resolve_csv_path(csv_path)
    rows, sweep_params = _load_grid_csv(csv_path)

    # Validate x and y columns
    sample = rows[0]
    if x not in sample:
        raise ValueError(
            f"x-axis column '{x}' not found in CSV. Available columns: {sorted(sample.keys())}"
        )
    if y not in sample:
        raise ValueError(
            f"y-axis column '{y}' not found in CSV. Available columns: {sorted(sample.keys())}"
        )

    # Identify all sweep parameters that are not the x-axis
    other_sweeps = [p for p in sweep_params if p != x]

    # Auto-detect group_by if not specified
    if group_by is None and other_sweeps:
        group_by = other_sweeps[0]

    # Build series: each unique combination of non-x sweep parameters
    # becomes its own line. group_by controls subplot assignment;
    # remaining sweeps split lines within each subplot.
    def _series_key(row):
        """Return a tuple of all non-x sweep parameter values for this row."""
        return tuple(row.get(p) for p in other_sweeps)

    def _series_label(key):
        """Build a human-readable label from sweep parameter values."""
        parts = []
        for p, v in zip(other_sweeps, key):
            parts.append(f'{p} = {v}')
        return ', '.join(parts) if parts else None

    def _group_val_from_key(key):
        """Extract the group_by value from a series key."""
        if group_by and group_by in other_sweeps:
            idx = other_sweeps.index(group_by)
            return key[idx]
        return None

    # Collect all unique series keys (preserving order)
    all_keys = []
    for r in rows:
        k = _series_key(r)
        if k not in all_keys:
            all_keys.append(k)

    # Map each series key to its group_by value (for subplot assignment)
    if group_by and group_by in sample:
        group_values = []
        for k in all_keys:
            gv = _group_val_from_key(k)
            if gv not in group_values:
                group_values.append(gv)
    else:
        group_values = [None]

    n_groups = len(group_values)

    if single_panel or n_groups == 1:
        fig, ax = plt.subplots(figsize=(8, 6))
        axes_map = {gv: ax for gv in group_values}
    else:
        ncols = min(n_groups, 3)
        nrows = (n_groups + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
        axes_flat = list(np.array(axes).flat)
        axes_map = {gv: axes_flat[i] for i, gv in enumerate(group_values)}

    # Assign a color to each series
    color_idx = 0
    for series_key in all_keys:
        gv = _group_val_from_key(series_key)
        ax = axes_map[gv]
        color = _COLORS[color_idx % len(_COLORS)]
        color_idx += 1
        label = _series_label(series_key)

        # Collect rows matching this series key
        series_rows = [r for r in rows if _series_key(r) == series_key]
        conv = [r for r in series_rows if r.get('converged') is True]
        unconv = [r for r in series_rows if r.get('converged') is not True]

        # Extract and sort (x, y) pairs
        def _extract_sorted(subset):
            pairs = []
            for r in subset:
                xval, yval = r.get(x), r.get(y)
                if (
                    xval is not None
                    and yval is not None
                    and isinstance(xval, (int, float))
                    and isinstance(yval, (int, float))
                ):
                    pairs.append((xval, yval))
            pairs.sort()
            if not pairs:
                return [], []
            return [p[0] for p in pairs], [p[1] for p in pairs]

        x_conv, y_conv = _extract_sorted(conv)
        x_unconv, y_unconv = _extract_sorted(unconv)

        # Plot converged points
        if x_conv:
            ax.plot(
                x_conv,
                y_conv,
                'o-',
                color=color,
                markersize=6,
                lw=1.5,
                label=label,
            )

        # Plot unconverged points as X markers
        if x_unconv:
            ax.plot(
                x_unconv,
                y_unconv,
                'x',
                color=color,
                markersize=8,
                markeredgewidth=2,
                label=f'{label} (unconverged)' if label else 'Unconverged',
            )

    # Format all axes
    for gv in group_values:
        ax = axes_map[gv]
        ax.set_xlabel(_AXIS_LABELS.get(x, x), fontsize=11)
        ax.set_ylabel(_AXIS_LABELS.get(y, y), fontsize=11)
        ax.grid(alpha=0.15)

        if not single_panel and gv is not None:
            ax.set_title(f'{group_by} = {gv}', fontsize=11)

        # Show legend only when there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(fontsize=9)

    # Hide unused subplots
    if not single_panel and n_groups > 1:
        for idx in range(n_groups, len(axes_flat)):
            axes_flat[idx].set_visible(False)

    # Figure title
    title = f'{y} vs {x}'
    if group_by and n_groups > 1:
        title += f' (grouped by {group_by})'
    fig.suptitle(title, fontsize=13, fontweight='bold', y=1.01)
    fig.tight_layout()

    # Output path
    if output is None:
        csv_dir = os.path.dirname(csv_path)
        fname = f'grid_{y}_vs_{x}.png'
        output = os.path.join(csv_dir, fname)

    fig.savefig(output, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {output}')
    return output


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _main():
    """CLI entry point for plot_grid."""
    parser = argparse.ArgumentParser(
        description='Plot Zalmoxis grid results from grid_summary.csv',
    )
    parser.add_argument(
        'path',
        help='Path to grid output directory or grid_summary.csv file',
    )
    parser.add_argument(
        '-x',
        '--x-axis',
        default='planet_mass',
        help='Column for x-axis (default: planet_mass)',
    )
    parser.add_argument(
        '-y',
        '--y-axis',
        default='R_earth',
        help='Column for y-axis (default: R_earth)',
    )
    parser.add_argument(
        '-g',
        '--group-by',
        default=None,
        help='Sweep parameter to group by (auto-detected if omitted)',
    )
    parser.add_argument(
        '--single-panel',
        action='store_true',
        help='Plot all groups on one panel instead of subplots',
    )
    parser.add_argument(
        '-o',
        '--output',
        default=None,
        help='Output file path (default: grid_<y>_vs_<x>.png in CSV directory)',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='Figure resolution (default: 200)',
    )

    args = parser.parse_args()

    plot_grid_summary(
        csv_path=args.path,
        x=args.x_axis,
        y=args.y_axis,
        group_by=args.group_by,
        single_panel=args.single_panel,
        output=args.output,
        dpi=args.dpi,
    )


if __name__ == '__main__':
    _main()
