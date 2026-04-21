"""Plot pressure-temperature trajectories from a Zalmoxis parameter grid.

Reads the ``<label>.npz`` files produced by ``run_grid`` when
``[output].save_profiles = true`` and plots the interior (P, T) trajectory
of each converged grid point as one line, coloured by the primary sweep
parameter. Useful for diagnosing which phase regimes a grid traverses.

By convention the trajectory runs from the planet centre (high P, high T)
to the surface (low P, low T). Pressure is on the y-axis in GPa and is
log-scaled by default because interior pressures span several decades.

Mantle solidus and liquidus reference curves are overlaid as dashed and
dotted lines, but only when the mantle EOS actually used by the run
takes external melting curves (``WolfBower2018:MgSiO3``,
``RTPress100TPa:MgSiO3``, ``PALEOS-2phase:MgSiO3``). For those runs the
curves match whatever ``rock_solidus`` / ``rock_liquidus`` the solver
was given, read from metadata embedded in the ``.npz``. For unified
PALEOS tables, where the phase boundary is embedded in the table itself
rather than an external curve, the overlay is suppressed by default and
a note is printed. Explicit CLI flags ``--solidus`` / ``--liquidus``
force a specific overlay regardless, and ``--no-melting-curves``
disables the overlay entirely.

Usage
-----
From the terminal::

    python -m src.tools.plot_grid_pt output_files/grid_mass_radius
    python -m src.tools.plot_grid_pt output_files/grid_mass_radius -o pt.pdf
    python -m src.tools.plot_grid_pt output_files/grid_mass_radius --linear-pressure
    python -m src.tools.plot_grid_pt output_files/grid_mass_radius \\
        --solidus Stixrude14-solidus --liquidus Stixrude14-liquidus
    python -m src.tools.plot_grid_pt output_files/grid_mass_radius --no-melting-curves

The script is also importable::

    from src.tools.plot_grid_pt import plot_grid_pt
    plot_grid_pt("output_files/grid_mass_radius")
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
# Constants (kept in sync with src/tools/plot_grid_profiles.py)
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
    'num_layers': 'Number of radial layers',
    'condensed_rho_min': r'Condensed $\rho_\mathrm{min}$ (kg m$^{-3}$)',
    'mushy_zone_factor': 'Mushy zone factor',
}

# EOS components for which the solver uses *external* solidus / liquidus
# curves. Unified PALEOS tables encode the phase boundary in the table
# itself and are NOT in this set. Kept in sync with
# ``src/zalmoxis/zalmoxis.py::_NEEDS_MELTING_CURVES``.
_EOS_USES_EXTERNAL_CURVES = {
    'WolfBower2018:MgSiO3',
    'RTPress100TPa:MgSiO3',
    'PALEOS-2phase:MgSiO3',
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


def _try_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _choose_colour_param(sweep_params, rows, explicit=None):
    if explicit is not None:
        if explicit not in sweep_params:
            raise ValueError(
                f"Colour parameter '{explicit}' not in sweep parameters {sweep_params}"
            )
        return explicit
    for p in sweep_params:
        if _try_float(rows[0].get(p)) is not None:
            return p
    return sweep_params[0] if sweep_params else None


def _converged(row):
    return str(row.get('converged', '')).strip().lower() == 'true'


def _load_profile(grid_dir, label):
    npz_path = os.path.join(grid_dir, f'{label}.npz')
    if not os.path.isfile(npz_path):
        return None
    return np.load(npz_path)


def _read_str(data, key):
    """Return a stored string field from a .npz, or an empty string if absent."""
    if key not in data.files:
        return ''
    value = data[key]
    try:
        return str(value.item()) if hasattr(value, 'item') else str(value)
    except Exception:
        return str(value)


def _mantle_uses_external_curves(mantle_eos):
    """True iff any component of the mantle EOS string uses external curves.

    The mantle string may be a single component (``"PALEOS:MgSiO3"``) or a
    multi-component mixture with mass fractions
    (``"PALEOS:MgSiO3:0.9+Chabrier:H:0.1"``). Any component in the
    ``_EOS_USES_EXTERNAL_CURVES`` set triggers the overlay.
    """
    if not mantle_eos:
        return False
    # Mixture separator in Zalmoxis is '+'; strip any mass-fraction suffix
    # (``:<float>``) before checking membership.
    components = []
    for part in mantle_eos.split('+'):
        tokens = part.split(':')
        # Rejoin the first two tokens (source:composition); drop any
        # trailing numeric mass fraction.
        if len(tokens) >= 2:
            components.append(':'.join(tokens[:2]))
    return any(c in _EOS_USES_EXTERNAL_CURVES for c in components)


# ---------------------------------------------------------------------------
# Main plot function
# ---------------------------------------------------------------------------
def plot_grid_pt(
    grid_dir,
    out=None,
    dpi=200,
    colour_by=None,
    linear_pressure=False,
    solidus=None,
    liquidus=None,
    show_melting_curves=True,
):
    """Plot interior P-T trajectories for every converged grid point.

    Parameters
    ----------
    grid_dir : str
        Path to the grid output directory (or directly to grid_summary.csv).
    out : str, optional
        Output image path. Extension determines format. Defaults to
        ``pt_trajectories.pdf`` next to the CSV.
    dpi : int
        Resolution for raster outputs.
    colour_by : str, optional
        Sweep-parameter column to colour trajectories by. Defaults to the
        first numeric sweep parameter.
    linear_pressure : bool
        Use a linear pressure axis instead of the default log y-axis.
    solidus : str, optional
        Force a specific solidus identifier (from
        ``zalmoxis.melting_curves``). When ``None`` the tool reads
        ``rock_solidus_id`` from the ``.npz`` metadata and overlays only
        if the mantle EOS is one that uses external curves.
    liquidus : str, optional
        Force a specific liquidus identifier; same auto-detection rules
        as ``solidus``.
    show_melting_curves : bool
        Master switch. If False, skip the overlay entirely.
    """
    grid_dir = _resolve_grid_dir(grid_dir)
    rows = _load_summary(grid_dir)
    sweep_params = _detect_sweep_params(rows)
    if not sweep_params:
        raise RuntimeError('No sweep parameters detected in grid_summary.csv')
    colour_param = _choose_colour_param(sweep_params, rows, colour_by)

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

    c_vals = [p[3] for p in profiles if p[3] is not None]
    if c_vals and len(set(c_vals)) > 1:
        if min(c_vals) > 0 and max(c_vals) / min(c_vals) >= 10:
            norm = LogNorm(vmin=min(c_vals), vmax=max(c_vals))
        else:
            norm = Normalize(vmin=min(c_vals), vmax=max(c_vals))
        cmap = plt.get_cmap('viridis')
        numeric_colour = True
    else:
        norm = None
        cmap = plt.get_cmap('tab10')
        numeric_colour = False

    def _colour(val, idx):
        if numeric_colour and val is not None:
            return cmap(norm(val))
        return cmap(idx % cmap.N)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # Collect global P- and T-ranges across all trajectories so the
    # overlaid melting curves span the same domain and do not stretch the
    # plot when an analytic curve extrapolates to unphysical T at very
    # high P (e.g. Monteux+16 is documented valid only to ~500 GPa).
    p_min = float('inf')
    p_max = 0.0
    t_max_data = 0.0
    for _, _, data, _ in profiles:
        P = np.asarray(data['pressure'])
        T = np.asarray(data['temperature'])
        P_pos = P[P > 0]
        if P_pos.size:
            p_min = min(p_min, float(P_pos.min()))
            p_max = max(p_max, float(P_pos.max()))
        if T.size:
            t_max_data = max(t_max_data, float(np.nanmax(T)))
    if not np.isfinite(p_min) or p_max <= p_min:
        p_min, p_max = 1e5, 1e12  # fallback: 0.1 kPa to 1 TPa
    if t_max_data <= 0:
        t_max_data = 1e4

    for idx, (label, row, data, c_val) in enumerate(profiles):
        # Log pressure axis cannot show P = 0 at the surface cell. Clip the
        # surface pad so we plot only shells with P > 0. The physically
        # interesting trajectory (centre -> surface mantle) is preserved.
        P = np.asarray(data['pressure'])
        T = np.asarray(data['temperature'])
        keep = P > 0
        if not keep.any():
            continue
        ax.plot(T[keep], P[keep] / 1e9, color=_colour(c_val, idx), lw=1.3)

        # Mark the CMB if we can find it (first node where mass_enclosed
        # exceeds cmb_mass). Small dot, no legend entry.
        mass = np.asarray(data['mass_enclosed'])
        cmb_mass = float(np.asarray(data['cmb_mass']))
        if cmb_mass > 0 and mass.max() > cmb_mass:
            i_cmb = int(np.argmax(mass >= cmb_mass))
            if keep[i_cmb]:
                ax.scatter(
                    [T[i_cmb]],
                    [P[i_cmb] / 1e9],
                    color=_colour(c_val, idx),
                    edgecolors='black',
                    s=30,
                    zorder=5,
                )

    ax.set_xlabel('Temperature (K)')
    ax.set_ylabel('Pressure (GPa)')
    if not linear_pressure:
        ax.set_yscale('log')
    ax.grid(True, alpha=0.25, linestyle='--', which='both')

    # Overlay mantle solidus / liquidus curves, but only if they actually
    # correspond to the curves the solver used for this grid.
    curve_handles = []
    overlay_note = None
    if show_melting_curves:
        # Collect mantle-EOS / curve-id metadata per grid point. For a
        # fixed-composition sweep there will be one unique set; for a
        # mantle-varying sweep we skip the overlay (would need per-run
        # curves) and print a note.
        mantle_set = {_read_str(p[2], 'mantle_eos') for p in profiles}
        solidus_set = {_read_str(p[2], 'rock_solidus_id') for p in profiles}
        liquidus_set = {_read_str(p[2], 'rock_liquidus_id') for p in profiles}

        # Explicit CLI override wins unconditionally.
        if solidus is not None or liquidus is not None:
            chosen_solidus = solidus or 'Monteux16-solidus'
            chosen_liquidus = liquidus or 'Monteux16-liquidus-A-chondritic'
            overlay_reason = 'forced via CLI'
        elif len(mantle_set) > 1:
            overlay_note = (
                'mantle EOS differs across grid points; melting-curve overlay '
                'suppressed. Force with --solidus / --liquidus.'
            )
            chosen_solidus = chosen_liquidus = None
            overlay_reason = None
        elif not any(_mantle_uses_external_curves(m) for m in mantle_set if m):
            mantle_eos = next(iter(mantle_set - {''}), '<unknown>')
            overlay_note = (
                f"mantle EOS '{mantle_eos}' embeds its phase boundary in the "
                f'table itself (no external curves used); no overlay.'
            )
            chosen_solidus = chosen_liquidus = None
            overlay_reason = None
        elif len(solidus_set) == 1 and len(liquidus_set) == 1:
            chosen_solidus = next(iter(solidus_set))
            chosen_liquidus = next(iter(liquidus_set))
            overlay_reason = 'from solver metadata'
            if not chosen_solidus or not chosen_liquidus:
                overlay_note = (
                    'missing rock_solidus_id / rock_liquidus_id metadata in '
                    '.npz (pre-metadata grid run?); no overlay.'
                )
                chosen_solidus = chosen_liquidus = None
                overlay_reason = None
        else:
            overlay_note = (
                'rock_solidus / rock_liquidus differ across grid points; '
                'overlay suppressed. Force with --solidus / --liquidus.'
            )
            chosen_solidus = chosen_liquidus = None
            overlay_reason = None

        if chosen_solidus and chosen_liquidus:
            try:
                from zalmoxis.melting_curves import get_melting_curve_function

                sol_fn = get_melting_curve_function(chosen_solidus)
                liq_fn = get_melting_curve_function(chosen_liquidus)
                # Sample on a log grid across the trajectory P-range; P >= 1 MPa
                # avoids the low-P end of piecewise forms.
                p_sample = np.logspace(
                    np.log10(max(p_min, 1e6)),
                    np.log10(p_max),
                    200,
                )
                t_sol = np.asarray(sol_fn(p_sample))
                t_liq = np.asarray(liq_fn(p_sample))
                # Clip to 1.3 x trajectory T-max so analytic extrapolation
                # beyond the curves' validity range cannot stretch the axes.
                t_cap = 1.3 * t_max_data
                valid_sol = np.isfinite(t_sol) & (t_sol <= t_cap)
                valid_liq = np.isfinite(t_liq) & (t_liq <= t_cap)
                (sol_line,) = ax.plot(
                    t_sol[valid_sol],
                    p_sample[valid_sol] / 1e9,
                    color='0.25',
                    linestyle='--',
                    lw=1.4,
                    label=f'Solidus ({chosen_solidus}, {overlay_reason})',
                    zorder=3,
                )
                (liq_line,) = ax.plot(
                    t_liq[valid_liq],
                    p_sample[valid_liq] / 1e9,
                    color='0.55',
                    linestyle=':',
                    lw=1.4,
                    label=f'Liquidus ({chosen_liquidus}, {overlay_reason})',
                    zorder=3,
                )
                curve_handles = [sol_line, liq_line]
            except Exception as exc:
                overlay_note = f'could not load melting curves ({exc})'
                curve_handles = []

    if overlay_note:
        print(f'Note: {overlay_note}')
        # Also put a short hint on the figure itself so readers of the
        # PDF do not misinterpret a missing overlay as a bug.
        ax.text(
            0.98,
            0.97,
            f'Melting curves: {overlay_note}',
            transform=ax.transAxes,
            fontsize=7,
            ha='right',
            va='top',
            color='0.35',
            wrap=True,
        )

    # Add a small in-axes annotation to explain the CMB markers.
    # Lower-left to leave the lower-right free for the melting-curves legend.
    ax.text(
        0.02,
        0.03,
        'Markers: core-mantle boundary',
        transform=ax.transAxes,
        fontsize=8,
        ha='left',
        va='bottom',
        color='0.3',
    )

    colour_label = _AXIS_LABELS.get(colour_param, colour_param)
    if numeric_colour:
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, shrink=0.9, pad=0.02)
        cbar.set_label(colour_label)
        if curve_handles:
            # Separate legend for the reference melting curves; placed
            # inside the axes so it does not push the colourbar around.
            ax.legend(handles=curve_handles, frameon=False, loc='lower right')
    else:
        handles = []
        for idx, (label, row, _data, _c) in enumerate(profiles):
            value = row.get(colour_param, label)
            handles.append(
                plt.Line2D([0], [0], color=_colour(None, idx), lw=1.5, label=str(value))
            )
        ax.legend(
            handles=handles + curve_handles,
            title=colour_label,
            frameon=False,
            loc='best',
        )

    n_total = len(rows)
    n_plotted = len(profiles)
    fig.suptitle(
        f'Zalmoxis grid interior P-T trajectories, {n_plotted}/{n_total} converged '
        f'({os.path.basename(grid_dir.rstrip("/"))})'
    )

    if out is None:
        out = os.path.join(grid_dir, 'pt_trajectories.pdf')
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
            'Plot interior pressure-temperature trajectories for every '
            'converged grid point from a Zalmoxis parameter grid.'
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
            'Default: <grid_dir>/pt_trajectories.pdf'
        ),
    )
    parser.add_argument(
        '--colour-by',
        '--color-by',
        dest='colour_by',
        default=None,
        help=(
            'Sweep parameter to colour trajectories by. Defaults to the '
            'first numeric sweep parameter (e.g. planet_mass).'
        ),
    )
    parser.add_argument(
        '--linear-pressure',
        action='store_true',
        help='Use a linear pressure y-axis (default: log).',
    )
    parser.add_argument(
        '--solidus',
        default=None,
        help=(
            'Force a specific solidus identifier (from zalmoxis.melting_curves) '
            'regardless of what the solver used. Default: auto-detect from '
            '.npz metadata; overlay only if the mantle EOS uses external curves.'
        ),
    )
    parser.add_argument(
        '--liquidus',
        default=None,
        help='Force a specific liquidus identifier (see --solidus).',
    )
    parser.add_argument(
        '--no-melting-curves',
        dest='show_melting_curves',
        action='store_false',
        help='Disable the solidus/liquidus reference overlay.',
    )
    parser.set_defaults(show_melting_curves=True)
    parser.add_argument(
        '--dpi',
        type=int,
        default=200,
        help='Raster DPI (ignored for vector formats). Default 200.',
    )
    return parser


if __name__ == '__main__':
    args = _build_parser().parse_args()
    plot_grid_pt(
        grid_dir=args.path,
        out=args.output,
        dpi=args.dpi,
        colour_by=args.colour_by,
        linear_pressure=args.linear_pressure,
        solidus=args.solidus,
        liquidus=args.liquidus,
        show_melting_curves=args.show_melting_curves,
    )
