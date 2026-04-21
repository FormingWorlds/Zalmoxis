"""Plot layer composition (core / mantle / ice) across a Zalmoxis parameter grid.

Reads the ``<label>.npz`` files produced by ``run_grid`` when
``[output].save_profiles = true`` and draws two horizontal stacked-bar
panels: mass fractions on the left and radius fractions on the right,
one bar per converged grid point.

Core / mantle / ice fractions are derived from the ``cmb_mass``,
``core_mantle_mass``, ``mass_enclosed``, and ``radii`` arrays stored in
each archive. For 2-layer planets the ice segment is absent; for 3-layer
models it is shown explicitly.

Usage
-----
From the terminal::

    python -m src.tools.plot_grid_composition output_files/grid_mass_radius
    python -m src.tools.plot_grid_composition output_files/grid_mass_radius -o comp.pdf
    python -m src.tools.plot_grid_composition output_files/grid_h2_mixing --label-by mantle

Python API::

    from src.tools.plot_grid_composition import plot_grid_composition
    plot_grid_composition("output_files/grid_mass_radius")
"""

from __future__ import annotations

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants (kept in sync with plot_grid_profiles.py)
# ---------------------------------------------------------------------------
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
    'ice_layer': 'Ice layer EOS',
    'num_layers': 'Number of radial layers',
}

# Earthy / geophysical palette: iron core, silicate mantle, icy envelope.
_LAYER_COLOURS = {
    'core': '#8c6b4f',  # warm brown (iron)
    'mantle': '#d98750',  # ochre / silicate orange
    'ice': '#8ec5fc',  # pale blue (H2O ice / envelope)
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _resolve_grid_dir(path):
    if os.path.isdir(path):
        return path
    if os.path.isfile(path) and os.path.basename(path) == 'grid_summary.csv':
        return os.path.dirname(os.path.abspath(path))
    raise ValueError(f'Expected a grid output directory or grid_summary.csv file, got: {path}')


def _load_summary(grid_dir):
    csv_path = os.path.join(grid_dir, 'grid_summary.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f'grid_summary.csv not found in {grid_dir}')
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f'grid_summary.csv in {grid_dir} is empty')
    return rows


def _detect_sweep_params(rows):
    return sorted(set(rows[0].keys()) - _FIXED_COLUMNS)


def _converged(row):
    return str(row.get('converged', '')).strip().lower() == 'true'


def _load_profile(grid_dir, label):
    npz_path = os.path.join(grid_dir, f'{label}.npz')
    if not os.path.isfile(npz_path):
        return None
    return np.load(npz_path)


def _choose_label_param(sweep_params, explicit=None):
    if explicit is not None:
        if explicit not in sweep_params:
            raise ValueError(
                f"Label parameter '{explicit}' not in sweep parameters {sweep_params}"
            )
        return explicit
    return sweep_params[0] if sweep_params else None


def _layer_fractions(data):
    """Return (core, mantle, ice) fractions for both mass and radius.

    Derived from the per-planet arrays in the archive. Handles two
    conventions the solver may have written:

    - 2-layer models (no ice layer): ``core_mantle_mass`` equals
      ``cmb_mass`` because Zalmoxis stores ``(core_mass_fraction +
      mantle_mass_fraction) * M_total`` and the default config sets
      ``mantle_mass_fraction = 0``. The mantle then fills to the surface
      and there is no ice segment.
    - 3-layer models (``ice_layer`` EOS set, ``mantle_mass_fraction > 0``):
      ``core_mantle_mass`` is the cumulative mass through the mantle
      outer boundary and the ice layer fills the remainder.
    """
    radii = np.asarray(data['radii'])
    mass = np.asarray(data['mass_enclosed'])
    cmb_mass = float(np.asarray(data['cmb_mass']))
    cm_mass = float(np.asarray(data['core_mantle_mass']))

    M_total = float(mass[-1])
    R_total = float(radii[-1])

    if M_total <= 0 or R_total <= 0:
        return None

    # Detect 2-layer (no ice) by checking whether core_mantle_mass equals
    # the core mass within numerical tolerance. The alternative would be
    # to inspect the run config, which is not stored in the .npz.
    is_two_layer = abs(cm_mass - cmb_mass) < 1e-3 * M_total

    # Mass fractions, clamped to [0, 1] to guard against tiny Picard residuals.
    f_core_m = max(0.0, min(1.0, cmb_mass / M_total))
    if is_two_layer:
        f_mantle_m = max(0.0, 1.0 - f_core_m)
        f_ice_m = 0.0
    else:
        f_cm_m = max(f_core_m, min(1.0, cm_mass / M_total))
        f_mantle_m = f_cm_m - f_core_m
        f_ice_m = max(0.0, 1.0 - f_cm_m)

    # Radius fractions: find the outermost shell whose enclosed mass is
    # still <= the target, then interpolate linearly into the next cell so
    # a 150-layer grid resolves the CMB to ~1% of R.
    def _r_at_mass(target_mass):
        if target_mass <= 0:
            return 0.0
        if target_mass >= M_total:
            return R_total
        i = int(np.searchsorted(mass, target_mass))
        if i <= 0:
            return float(radii[0])
        if i >= len(mass):
            return R_total
        m_lo, m_hi = mass[i - 1], mass[i]
        if m_hi == m_lo:
            return float(radii[i])
        w = (target_mass - m_lo) / (m_hi - m_lo)
        return float(radii[i - 1] + w * (radii[i] - radii[i - 1]))

    r_cmb = _r_at_mass(cmb_mass)
    f_core_r = r_cmb / R_total
    if is_two_layer:
        f_mantle_r = max(0.0, 1.0 - f_core_r)
        f_ice_r = 0.0
    else:
        r_cm = _r_at_mass(cm_mass)
        f_mantle_r = max(0.0, (r_cm - r_cmb) / R_total)
        f_ice_r = max(0.0, 1.0 - r_cm / R_total)

    return {
        'mass': (f_core_m, f_mantle_m, f_ice_m),
        'radius': (f_core_r, f_mantle_r, f_ice_r),
        'M_total': M_total,
        'R_total': R_total,
    }


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------
def plot_grid_composition(
    grid_dir,
    out=None,
    dpi=200,
    label_by=None,
):
    """Draw stacked bar charts of mass and radius fractions per grid point.

    Parameters
    ----------
    grid_dir : str
        Path to the grid output directory (or directly to grid_summary.csv).
    out : str, optional
        Output image path. Defaults to ``composition.pdf`` next to the CSV.
    dpi : int
        Raster DPI (ignored for vector formats).
    label_by : str, optional
        Sweep-parameter column whose value labels each bar. Defaults to
        the first sweep parameter. Non-numeric sweeps (e.g. ``mantle``)
        are labelled as-is.
    """
    grid_dir = _resolve_grid_dir(grid_dir)
    rows = _load_summary(grid_dir)
    sweep_params = _detect_sweep_params(rows)
    if not sweep_params:
        raise RuntimeError('No sweep parameters detected in grid_summary.csv')
    label_param = _choose_label_param(sweep_params, label_by)

    entries = []
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
        fracs = _layer_fractions(data)
        if fracs is None:
            skipped.append((label, 'zero mass or radius'))
            continue
        entries.append((row.get(label_param, label), row, fracs))

    if not entries:
        raise RuntimeError(f'No converged profiles with saved .npz files found in {grid_dir}.')

    # Sort by the numeric label value when possible so bars stack in a
    # natural reading order (bottom = smallest).
    def _sort_key(e):
        try:
            return (0, float(e[0]))
        except (TypeError, ValueError):
            return (1, str(e[0]))

    entries.sort(key=_sort_key)

    y_positions = np.arange(len(entries))
    y_labels = [str(e[0]) for e in entries]

    # Whether any grid point has a nonzero ice segment; controls legend.
    any_ice = any(e[2]['mass'][2] > 1e-6 or e[2]['radius'][2] > 1e-6 for e in entries)

    fig, (ax_m, ax_r) = plt.subplots(
        1,
        2,
        figsize=(11, max(3.5, 0.35 * len(entries) + 2)),
        sharey=True,
        constrained_layout=True,
    )

    def _draw_stacked(ax, panel_key):
        core_vals = np.array([e[2][panel_key][0] for e in entries])
        mantle_vals = np.array([e[2][panel_key][1] for e in entries])
        ice_vals = np.array([e[2][panel_key][2] for e in entries])
        ax.barh(
            y_positions,
            core_vals,
            color=_LAYER_COLOURS['core'],
            edgecolor='black',
            linewidth=0.3,
            label='Core',
        )
        ax.barh(
            y_positions,
            mantle_vals,
            left=core_vals,
            color=_LAYER_COLOURS['mantle'],
            edgecolor='black',
            linewidth=0.3,
            label='Mantle',
        )
        if any_ice:
            ax.barh(
                y_positions,
                ice_vals,
                left=core_vals + mantle_vals,
                color=_LAYER_COLOURS['ice'],
                edgecolor='black',
                linewidth=0.3,
                label='Ice / envelope',
            )
        ax.set_xlim(0, 1)
        ax.set_yticks(y_positions)
        ax.set_yticklabels(y_labels)
        ax.grid(True, axis='x', alpha=0.25, linestyle='--')

    _draw_stacked(ax_m, 'mass')
    _draw_stacked(ax_r, 'radius')

    ax_m.set_xlabel('Fraction of total mass')
    ax_r.set_xlabel('Fraction of total radius')
    ax_m.text(
        0.02,
        0.98,
        '(a)',
        transform=ax_m.transAxes,
        fontsize=12,
        fontweight='bold',
        va='top',
        ha='left',
    )
    ax_r.text(
        0.02,
        0.98,
        '(b)',
        transform=ax_r.transAxes,
        fontsize=12,
        fontweight='bold',
        va='top',
        ha='left',
    )

    label_axis = _AXIS_LABELS.get(label_param, label_param)
    ax_m.set_ylabel(label_axis)

    # Single legend at the bottom so it does not collide with the bars.
    handles, labels = ax_m.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(handles),
        frameon=False,
    )

    n_total = len(rows)
    n_plotted = len(entries)
    fig.suptitle(
        f'Zalmoxis grid composition, {n_plotted}/{n_total} converged '
        f'({os.path.basename(grid_dir.rstrip("/"))})'
    )

    if out is None:
        out = os.path.join(grid_dir, 'composition.pdf')
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
            'Plot core / mantle / ice mass and radius fractions across a '
            'Zalmoxis parameter grid as stacked horizontal bars.'
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
        help='Output image path (extension selects format). Default: <grid_dir>/composition.pdf',
    )
    parser.add_argument(
        '--label-by',
        default=None,
        help=(
            'Sweep parameter used to label each bar. Defaults to the first '
            'sweep parameter in the grid (e.g. planet_mass).'
        ),
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
    plot_grid_composition(
        grid_dir=args.path,
        out=args.output,
        dpi=args.dpi,
        label_by=args.label_by,
    )
