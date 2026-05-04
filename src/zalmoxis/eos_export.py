"""Export Zalmoxis/PALEOS EOS data to SPIDER-compatible P-S format.

SPIDER uses entropy-pressure (P-S) coordinates for its 2D EOS lookup
tables and 1D phase boundaries. Zalmoxis and the PALEOS unified tables
use pressure-temperature (P-T) coordinates. This module converts
between the two coordinate systems.

Phase boundary conversion:
    T(P) melting curves -> S(P) phase boundaries via direct PALEOS
    S(P,T) lookup. No inversion needed.

Full EOS table generation:
    PALEOS P-T tables -> SPIDER P-S tables for density, temperature,
    heat capacity, thermal expansion, and adiabatic gradient, split
    into separate solid and melt files.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

logger = logging.getLogger(__name__)


# ── PALEOS unified table loader ─────────────────────────────────────


def load_paleos_all_properties(eos_file):
    """Load all properties from a PALEOS unified table on a regular (logP, logT) grid.

    Unlike ``eos_functions.load_paleos_unified_table`` (which only builds
    density and nabla_ad interpolators), this function retains all 9
    numeric columns for EOS export.

    Parameters
    ----------
    eos_file : str or Path
        Path to the PALEOS unified table (10-column format).

    Returns
    -------
    dict
        Keys:
        - ``'unique_log_p'``, ``'unique_log_t'``: 1D grid axes (log10 Pa, log10 K)
        - ``'rho'``: density grid [kg/m^3], shape (nP, nT)
        - ``'u'``: internal energy grid [J/kg]
        - ``'s'``: specific entropy grid [J/(kg*K)]
        - ``'cp'``: isobaric heat capacity grid [J/(kg*K)]
        - ``'cv'``: isochoric heat capacity grid [J/(kg*K)]
        - ``'alpha'``: thermal expansion coefficient grid [1/K]
        - ``'nabla_ad'``: adiabatic gradient grid [dimensionless]
        - ``'phase'``: phase identifier grid (string, object dtype)
        - ``'p_min'``, ``'p_max'``: pressure bounds [Pa]
        - ``'t_min'``, ``'t_max'``: temperature bounds [K]
    """
    eos_file = str(eos_file)
    data = np.genfromtxt(eos_file, usecols=range(9), comments='#')
    phase_strings = np.genfromtxt(eos_file, usecols=(9,), dtype=str, comments='#')

    pressures = data[:, 0]
    temps = data[:, 1]

    # Filter out P=0 rows (present in some tables as padding)
    valid = pressures > 0
    pressures = pressures[valid]
    temps = temps[valid]
    data = data[valid]
    phase_strings = np.char.strip(phase_strings[valid])

    log_p = np.log10(pressures)
    log_t = np.log10(temps)
    unique_log_p = np.unique(log_p)
    unique_log_t = np.unique(log_t)
    n_p = len(unique_log_p)
    n_t = len(unique_log_t)

    # Column names: P(0), T(1), rho(2), u(3), s(4), cp(5), cv(6), alpha(7), nabla_ad(8)
    prop_names = ['rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad']
    prop_cols = [2, 3, 4, 5, 6, 7, 8]

    grids = {}
    for name, col in zip(prop_names, prop_cols):
        grids[name] = np.full((n_p, n_t), np.nan)

    phase_grid = np.full((n_p, n_t), '', dtype=object)

    p_idx_map = {v: i for i, v in enumerate(unique_log_p)}
    t_idx_map = {v: i for i, v in enumerate(unique_log_t)}

    for k in range(len(pressures)):
        ip = p_idx_map[log_p[k]]
        it = t_idx_map[log_t[k]]
        for name, col in zip(prop_names, prop_cols):
            grids[name][ip, it] = data[k, col]
        phase_grid[ip, it] = phase_strings[k]

    result = {
        'unique_log_p': unique_log_p,
        'unique_log_t': unique_log_t,
        'phase': phase_grid,
        'p_min': 10.0 ** unique_log_p[0],
        'p_max': 10.0 ** unique_log_p[-1],
        't_min': 10.0 ** unique_log_t[0],
        't_max': 10.0 ** unique_log_t[-1],
    }
    result.update(grids)
    return result


def _build_interpolator(unique_log_p, unique_log_t, grid):
    """Build a RegularGridInterpolator for a 2D property grid.

    Parameters
    ----------
    unique_log_p : ndarray
        Unique log10(P) values.
    unique_log_t : ndarray
        Unique log10(T) values.
    grid : ndarray
        2D array of shape (nP, nT).

    Returns
    -------
    RegularGridInterpolator
    """
    return RegularGridInterpolator(
        (unique_log_p, unique_log_t),
        grid,
        bounds_error=False,
        fill_value=np.nan,
    )


def _fill_nan_nearest(grid):
    """Fill NaN cells in a 2D grid using nearest valid neighbor.

    Operates in-place. For each NaN cell, copies the value from the
    nearest cell (by Manhattan distance) that has a finite value.

    Parameters
    ----------
    grid : ndarray, shape (nS, nP)
        2D array with NaN gaps to fill.
    """
    from scipy.ndimage import distance_transform_edt

    mask = np.isnan(grid)
    if not mask.any():
        return

    # distance_transform_edt returns distances and indices of nearest valid cell
    _, indices = distance_transform_edt(mask, return_distances=True, return_indices=True)
    grid[mask] = grid[tuple(indices[:, mask])]


# ── Phase boundary generation ───────────────────────────────────────


def generate_spider_phase_boundaries(
    solidus_func,
    liquidus_func,
    eos_file,
    P_range=(1e5, 200e9),
    n_P=1000,
    output_dir=None,
    solid_eos_file=None,
    liquid_eos_file=None,
):
    """Generate SPIDER S(P) phase boundary files from T(P) melting curves.

    For each pressure on a log-spaced grid, the solidus and liquidus
    temperatures are computed from the provided melting curve functions,
    then the corresponding entropy is looked up directly from the PALEOS
    unified table's entropy column. This avoids the numerical inversion
    required by the legacy approach (tools/generate_spider_phase_boundaries.py).

    Parameters
    ----------
    solidus_func : callable
        P [Pa] -> T_solidus [K]. Must accept scalar or array.
    liquidus_func : callable
        P [Pa] -> T_liquidus [K]. Must accept scalar or array.
    eos_file : str or Path
        Path to the PALEOS unified EOS table.
    P_range : tuple of float
        (P_min, P_max) in Pa. Default: (1e5, 200 GPa).
    n_P : int
        Number of pressure points (log-spaced). Default: 1000.
    output_dir : str or Path or None
        Directory where ``solidus_P-S.dat`` and ``liquidus_P-S.dat``
        are written. If None, files are not written and results are
        only returned.

    Returns
    -------
    dict
        Keys:
        - ``'P_Pa'``: pressure grid [Pa], shape (n_valid,)
        - ``'S_solidus'``: solidus entropy [J/(kg*K)], shape (n_valid,)
        - ``'S_liquidus'``: liquidus entropy [J/(kg*K)], shape (n_valid,)
        - ``'solidus_path'``: path to written solidus file (or None)
        - ``'liquidus_path'``: path to written liquidus file (or None)
    """
    # Load PALEOS table and build entropy interpolators.
    # When 2-phase tables are provided, use phase-specific entropy to
    # avoid interpolation across the melting curve discontinuity.
    logger.info('Loading PALEOS table for phase boundary generation: %s', eos_file)
    table = load_paleos_all_properties(eos_file)
    s_interp = _build_interpolator(table['unique_log_p'], table['unique_log_t'], table['s'])

    # Phase-specific entropy interpolators (if 2-phase tables provided)
    s_sol_interp = s_interp  # default: use unified table for both
    s_liq_interp = s_interp
    if solid_eos_file and liquid_eos_file:
        solid_tab = load_paleos_all_properties(solid_eos_file)
        liquid_tab = load_paleos_all_properties(liquid_eos_file)
        if solid_tab is not None and liquid_tab is not None:
            s_sol_interp = _build_interpolator(
                solid_tab['unique_log_p'], solid_tab['unique_log_t'], solid_tab['s']
            )
            s_liq_interp = _build_interpolator(
                liquid_tab['unique_log_p'], liquid_tab['unique_log_t'], liquid_tab['s']
            )
            logger.info('Using 2-phase PALEOS tables for phase boundary entropy')

    # Build log-spaced pressure grid
    P_grid = np.logspace(np.log10(P_range[0]), np.log10(P_range[1]), n_P)
    log_P = np.log10(P_grid)

    # Compute melting curve temperatures
    T_sol = np.atleast_1d(solidus_func(P_grid))
    T_liq = np.atleast_1d(liquidus_func(P_grid))

    # Look up entropy at (P, T_solidus) from solid table
    # and at (P, T_liquidus) from liquid table
    S_solidus = np.full(n_P, np.nan)
    S_liquidus = np.full(n_P, np.nan)

    for i in range(n_P):
        if np.isnan(T_sol[i]) or T_sol[i] <= 0:
            continue
        S_solidus[i] = float(s_sol_interp((log_P[i], np.log10(T_sol[i]))))

        if np.isnan(T_liq[i]) or T_liq[i] <= 0:
            continue
        S_liquidus[i] = float(s_liq_interp((log_P[i], np.log10(T_liq[i]))))

    # Filter to valid points
    valid = np.isfinite(S_solidus) & np.isfinite(S_liquidus)
    n_valid = np.sum(valid)
    logger.info(
        'Phase boundary conversion: %d/%d valid points (P = %.1e - %.1e Pa)',
        n_valid,
        n_P,
        P_grid[valid][0] if n_valid > 0 else 0,
        P_grid[valid][-1] if n_valid > 0 else 0,
    )

    if n_valid == 0:
        logger.error(
            'No valid phase boundary points. Check that the melting curve '
            'temperature range overlaps the PALEOS table temperature range '
            '(%.0f - %.0f K).',
            table['t_min'],
            table['t_max'],
        )
        return {
            'P_Pa': np.array([]),
            'S_solidus': np.array([]),
            'S_liquidus': np.array([]),
            'solidus_path': None,
            'liquidus_path': None,
        }

    P_valid = P_grid[valid]
    S_sol_valid = S_solidus[valid]
    S_liq_valid = S_liquidus[valid]

    # Smooth S(P) phase boundaries. The PALEOS entropy surface has fine
    # structure that produces wild dS/dP oscillations (100+ sign changes)
    # when sampled along the liquidus T(P). These oscillations produce
    # extreme mixing fluxes that crash SPIDER's CVode solver.
    #
    # Fix: fit a monotonic PCHIP spline through the S(P) data, using a
    # coarsened version (every Nth point) to eliminate oscillations, then
    # evaluate on the full P grid. PCHIP preserves monotonicity.
    if n_valid > 10:
        from scipy.interpolate import PchipInterpolator

        # Coarsen to ~20 anchor points to remove oscillations
        n_anchor = min(20, n_valid)
        indices = np.linspace(0, n_valid - 1, n_anchor, dtype=int)

        log_P_anchors = np.log10(P_valid[indices])
        S_sol_anchors = S_sol_valid[indices]
        S_liq_anchors = S_liq_valid[indices]

        # PCHIP interpolation on coarsened anchors -> smooth curve
        sol_pchip = PchipInterpolator(log_P_anchors, S_sol_anchors)
        liq_pchip = PchipInterpolator(log_P_anchors, S_liq_anchors)

        S_sol_smooth = sol_pchip(np.log10(P_valid))
        S_liq_smooth = liq_pchip(np.log10(P_valid))

        # Log improvement
        dliq_raw = np.diff(S_liq_valid)
        dliq_smooth = np.diff(S_liq_smooth)
        n_raw = np.sum((dliq_raw[:-1] > 0) != (dliq_raw[1:] > 0))
        n_smooth = np.sum((dliq_smooth[:-1] > 0) != (dliq_smooth[1:] > 0))
        logger.info(
            'Smoothed phase boundaries (PCHIP, %d anchors): '
            'liquidus dS/dP sign changes %d -> %d',
            n_anchor,
            n_raw,
            n_smooth,
        )
        S_sol_valid = S_sol_smooth
        S_liq_valid = S_liq_smooth

    # Enforce non-decreasing S_solidus(P) and S_liquidus(P) via
    # cumulative maximum (super-Earth pressure range fix).
    #
    # Motivation. When P_max is raised above ~200 GPa to cover super-Earth
    # mantles, both phase-boundary entropies show a peak near 175-200 GPa
    # and decrease thereafter. The peak is a feature of the WB2018 liquid
    # EOS entropy lookup at (P, T_liq(P)): at high P the density
    # contribution to S dominates and S(P, T_liq(P)) flattens and then
    # decreases, even though T_liq(P) is strictly monotone in P via
    # Belonoshko+2005 / Fei+2021.
    #
    # The lever rule Phi = (S_node - S_sol) / (S_liq - S_sol) becomes
    # nearly singular near the peak, because dS_liq/dP goes through zero
    # there. A fully-molten IC with S_node comparable to S_liq_peak ends
    # up with a narrow mushy band at ~175 GPa where Phi ~ 0.995, and the
    # resulting RHS is stiff enough that CVODE collapses to micro-year
    # timesteps and hits its 100k-step cap. Observed on the first 5 M_E
    # dry CHILI full runs at P_max=950 GPa
    # (see stage1c_5me_super_earth_progress memory for the full trace).
    #
    # Fix. Apply np.maximum.accumulate (from low P to high P) to both
    # arrays. Below the peak nothing changes (the arrays are already
    # monotone there). Above the peak the entropy is pinned to the peak
    # value, which is the defensible choice for a lever-rule consumer:
    # anything at least as hot as the peak should register as fully
    # liquid, and the flat tail removes the near-singular derivative.
    if n_valid > 1:
        S_sol_before = S_sol_valid.copy()
        S_liq_before = S_liq_valid.copy()
        S_sol_valid = np.maximum.accumulate(S_sol_valid)
        S_liq_valid = np.maximum.accumulate(S_liq_valid)
        n_sol_clipped = int(np.sum(S_sol_valid > S_sol_before + 1e-6))
        n_liq_clipped = int(np.sum(S_liq_valid > S_liq_before + 1e-6))
        if n_sol_clipped > 0 or n_liq_clipped > 0:
            logger.info(
                'Monotonised phase boundaries (cumulative max): %d solidus '
                'points clipped, %d liquidus points clipped; peaks at '
                'S_sol=%.1f J/kg/K and S_liq=%.1f J/kg/K',
                n_sol_clipped,
                n_liq_clipped,
                float(S_sol_valid[-1]),
                float(S_liq_valid[-1]),
            )

    # Write files
    solidus_path = None
    liquidus_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        solidus_path = str(output_dir / 'solidus_P-S.dat')
        liquidus_path = str(output_dir / 'liquidus_P-S.dat')
        _write_spider_1d(solidus_path, P_valid, S_sol_valid)
        _write_spider_1d(liquidus_path, P_valid, S_liq_valid)
        logger.info('Wrote solidus_P-S.dat  (%d points)', n_valid)
        logger.info('Wrote liquidus_P-S.dat (%d points)', n_valid)

    return {
        'P_Pa': P_valid,
        'S_solidus': S_sol_valid,
        'S_liquidus': S_liq_valid,
        'solidus_path': solidus_path,
        'liquidus_path': liquidus_path,
    }


# ── SPIDER file writers ──────────────────────────────────────────────

# Default scaling factors (matching SPIDER's internal conventions)
_P_SCALE = 1e9  # Pa
_S_SCALE = 4824266.84604467  # J/(kg*K), SPIDER's entropy scale


def _write_spider_1d(filepath, P_Pa, S_SI, P_scale=_P_SCALE, S_scale=_S_SCALE):
    """Write a SPIDER-format 1D phase boundary file (P, S).

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    P_Pa : ndarray
        Pressure values [Pa].
    S_SI : ndarray
        Entropy values [J/(kg*K)].
    P_scale : float
        Pressure scaling factor.
    S_scale : float
        Entropy scaling factor.
    """
    N = len(P_Pa)
    P_nd = P_Pa / P_scale
    S_nd = S_SI / S_scale

    with open(filepath, 'w') as f:
        f.write(f'# 5 {N}\n')
        f.write('# Pressure [nondim], Entropy [nondim]\n')
        f.write('# column * scaling factor = SI units: Pressure [Pa], Entropy [J/kg/K]\n')
        f.write('# scaling factors (constant) for each column given on line below\n')
        f.write(f'# {P_scale} {S_scale}\n')
        for p, s in zip(P_nd, S_nd):
            f.write(f'{p:.18e} {s:.18e}\n')


def _write_spider_2d(
    filepath, P_Pa, S_SI, values, quantity_scale, P_scale=_P_SCALE, S_scale=_S_SCALE
):
    """Write a SPIDER-format 2D lookup table file (P, S, quantity).

    The grid layout is: S is the slow (outer) index, P is the fast
    (inner) index. This matches SPIDER's expected format.

    Parameters
    ----------
    filepath : str or Path
        Output file path.
    P_Pa : ndarray
        Pressure grid [Pa], shape (nP,).
    S_SI : ndarray
        Entropy grid [J/(kg*K)], shape (nS,).
    values : ndarray
        Property values, shape (nS, nP).
    quantity_scale : float
        Scaling factor for the quantity.
    P_scale : float
        Pressure scaling factor.
    S_scale : float
        Entropy scaling factor.
    """
    nP = len(P_Pa)
    nS = len(S_SI)
    P_nd = P_Pa / P_scale
    S_nd = S_SI / S_scale
    Q_nd = values / quantity_scale

    with open(filepath, 'w') as f:
        f.write(f'# 5 {nP} {nS}\n')
        f.write('# Pressure [nondim], Entropy [nondim], Quantity [nondim]\n')
        f.write('# column * scaling factor = SI units\n')
        f.write('# scaling factors (constant) for each column given on line below\n')
        f.write(f'# {P_scale} {S_scale} {quantity_scale}\n')
        for j in range(nS):
            for i in range(nP):
                f.write(f'{P_nd[i]:.18e} {S_nd[j]:.18e} {Q_nd[j, i]:.18e}\n')


# ── Full EOS table generation ───────────────────────────────────────


def _find_valid_T_bounds(logP, T_lo, T_hi, s_interp, n_probe=50):
    """Find the valid temperature bounds where S(P,T) is finite.

    The PALEOS table has NaN entries at extreme temperatures. This
    function narrows [T_lo, T_hi] to the largest sub-interval where
    entropy lookups return finite values at both endpoints.

    Parameters
    ----------
    logP : float
        log10(pressure in Pa).
    T_lo, T_hi : float
        Initial temperature bounds [K].
    s_interp : RegularGridInterpolator
        Entropy interpolator on (log10P, log10T) grid.
    n_probe : int
        Number of probe points to test.

    Returns
    -------
    tuple or (None, None)
        (T_lo_valid, T_hi_valid) or (None, None) if no valid range found.
    """
    T_probes = np.geomspace(T_lo, T_hi, n_probe)
    valid_T = []
    for T in T_probes:
        S_val = float(s_interp((logP, np.log10(T))))
        if np.isfinite(S_val):
            valid_T.append(T)

    if len(valid_T) < 2:
        return None, None

    return valid_T[0], valid_T[-1]


def generate_spider_eos_tables(
    eos_file,
    solidus_func,
    liquidus_func,
    P_range=(1e5, 200e9),
    n_P=150,
    n_S=150,
    output_dir=None,
    solid_eos_file=None,
    liquid_eos_file=None,
):
    """Generate full SPIDER P-S EOS lookup tables from a PALEOS unified table.

    For each phase (solid, melt), generates 5 files:
    - density_{phase}.dat
    - temperature_{phase}.dat
    - heat_capacity_{phase}.dat
    - thermal_exp_{phase}.dat
    - adiabat_temp_grad_{phase}.dat

    The algorithm:
    1. Load PALEOS P-T table with all properties (rho, s, cp, alpha, nabla_ad)
    2. At each (P, T) grid point, we have S directly from the table
    3. Classify each point as solid (T < T_solidus) or melt (T > T_liquidus)
    4. For each phase, build a P-S grid by interpolation
    5. Write 2D tables in SPIDER format

    Parameters
    ----------
    eos_file : str or Path
        Path to the PALEOS unified EOS table.
    solidus_func : callable
        P [Pa] -> T_solidus [K].
    liquidus_func : callable
        P [Pa] -> T_liquidus [K].
    P_range : tuple of float
        (P_min, P_max) in Pa.
    n_P : int
        Number of pressure points in the output grid.
    n_S : int
        Number of entropy points in the output grid.
    output_dir : str or Path or None
        Directory for output files. If None, returns data without writing.

    Returns
    -------
    dict
        Keys: ``'P_Pa'``, ``'S_solid'``, ``'S_melt'``, and property
        grids for each phase. Also ``'output_dir'`` if files were written.
    """
    logger.info('Generating SPIDER P-S EOS tables from %s', eos_file)
    table = load_paleos_all_properties(eos_file)

    log_p_grid = table['unique_log_p']
    log_t_grid = table['unique_log_t']

    # Build interpolators for all needed properties.
    # Default: unified table for everything.
    s_interp = _build_interpolator(log_p_grid, log_t_grid, table['s'])
    rho_interp = _build_interpolator(log_p_grid, log_t_grid, table['rho'])
    cp_interp = _build_interpolator(log_p_grid, log_t_grid, table['cp'])
    alpha_interp = _build_interpolator(log_p_grid, log_t_grid, table['alpha'])
    nad_interp = _build_interpolator(log_p_grid, log_t_grid, table['nabla_ad'])

    # When 2-phase tables are available, use phase-specific interpolators
    # for ALL lookups (entropy, density, cp, alpha, nabla_ad), not just
    # entropy ranges. This avoids interpolation artifacts across the
    # thermodynamically inconsistent melting curve in the unified table,
    # and ensures SPIDER uses the same phase-specific properties as Aragog.
    s_solid_interp = s_interp
    s_melt_interp = s_interp
    # Phase-specific property interpolators (default to unified)
    rho_solid_interp = rho_interp
    rho_melt_interp = rho_interp
    cp_solid_interp = cp_interp
    cp_melt_interp = cp_interp
    alpha_solid_interp = alpha_interp
    alpha_melt_interp = alpha_interp
    nad_solid_interp = nad_interp
    nad_melt_interp = nad_interp

    if solid_eos_file and liquid_eos_file:
        solid_tab = load_paleos_all_properties(solid_eos_file)
        liquid_tab = load_paleos_all_properties(liquid_eos_file)
        if solid_tab is not None and liquid_tab is not None:
            sol_lp = solid_tab['unique_log_p']
            sol_lt = solid_tab['unique_log_t']
            liq_lp = liquid_tab['unique_log_p']
            liq_lt = liquid_tab['unique_log_t']

            s_solid_interp = _build_interpolator(sol_lp, sol_lt, solid_tab['s'])
            s_melt_interp = _build_interpolator(liq_lp, liq_lt, liquid_tab['s'])
            rho_solid_interp = _build_interpolator(sol_lp, sol_lt, solid_tab['rho'])
            rho_melt_interp = _build_interpolator(liq_lp, liq_lt, liquid_tab['rho'])
            cp_solid_interp = _build_interpolator(sol_lp, sol_lt, solid_tab['cp'])
            cp_melt_interp = _build_interpolator(liq_lp, liq_lt, liquid_tab['cp'])
            alpha_solid_interp = _build_interpolator(sol_lp, sol_lt, solid_tab['alpha'])
            alpha_melt_interp = _build_interpolator(liq_lp, liq_lt, liquid_tab['alpha'])
            nad_solid_interp = _build_interpolator(sol_lp, sol_lt, solid_tab['nabla_ad'])
            nad_melt_interp = _build_interpolator(liq_lp, liq_lt, liquid_tab['nabla_ad'])
            logger.info('Using 2-phase PALEOS tables for SPIDER (all properties)')

    # Output pressure grid. SPIDER's Interp2d assumes uniform spacing
    # for the P (x) coordinate: indx = floor((P - P_min) / dP). Log-spaced
    # pressure causes indx to overflow for large P and segfault SPIDER.
    P_out = np.linspace(P_range[0], P_range[1], n_P)

    # For each pressure, determine the entropy range for solid and melt phases
    # by scanning T from T_min to T_solidus (solid) and T_liquidus to T_max (melt)
    S_min_solid = np.full(n_P, np.inf)
    S_max_solid = np.full(n_P, -np.inf)
    S_min_melt = np.full(n_P, np.inf)
    S_max_melt = np.full(n_P, -np.inf)

    # Sample T finely to determine S ranges (vectorized per pressure)
    n_T_sample = 500
    T_lo_global = max(table['t_min'], 300.0)
    T_hi_global = min(table['t_max'], 1e5)

    for ip in range(n_P):
        P_Pa = P_out[ip]
        T_sol = solidus_func(P_Pa)
        T_liq = liquidus_func(P_Pa)

        if np.isnan(T_sol) or np.isnan(T_liq) or T_sol <= 0 or T_liq <= 0:
            continue

        logP = np.log10(P_Pa)

        # Solid phase: vectorized S lookup over T range
        if T_sol > T_lo_global:
            T_arr = np.linspace(T_lo_global, T_sol, n_T_sample)
            pts = np.column_stack([np.full(n_T_sample, logP), np.log10(T_arr)])
            S_arr = s_solid_interp(pts)
            finite = np.isfinite(S_arr)
            if finite.any():
                S_min_solid[ip] = np.min(S_arr[finite])
                S_max_solid[ip] = np.max(S_arr[finite])

        # Melt phase: vectorized S lookup over T range
        if T_liq < T_hi_global:
            T_arr = np.linspace(T_liq, T_hi_global, n_T_sample)
            pts = np.column_stack([np.full(n_T_sample, logP), np.log10(T_arr)])
            S_arr = s_melt_interp(pts)
            finite = np.isfinite(S_arr)
            if finite.any():
                S_min_melt[ip] = np.min(S_arr[finite])
                S_max_melt[ip] = np.max(S_arr[finite])

    # Build global S ranges for each phase
    valid_solid = np.isfinite(S_min_solid) & np.isfinite(S_max_solid)
    valid_melt = np.isfinite(S_min_melt) & np.isfinite(S_max_melt)

    if not valid_solid.any() or not valid_melt.any():
        logger.error('Cannot determine entropy ranges for solid and/or melt phases.')
        return {}

    S_global_min_solid = np.min(S_min_solid[valid_solid])
    S_global_max_solid = np.max(S_max_solid[valid_solid])
    S_global_min_melt = np.min(S_min_melt[valid_melt])
    S_global_max_melt = np.max(S_max_melt[valid_melt])

    # SPIDER queries both solid and melt tables near the phase boundary.
    # When the initial adiabat entropy (e.g. 3000 J/kg/K) exceeds the
    # solid-phase table range, SPIDER clamps to the table edge, producing
    # unphysical material properties and convergence failures.
    # Fix: extend the solid-phase entropy range to cover at least the
    # melt-phase maximum. The solid-phase properties beyond the actual
    # solidus are extrapolated (constant from the edge value), which is
    # acceptable because SPIDER only queries those cells when the material
    # is fully molten and the solid contribution is zero.
    S_global_max_solid_ext = max(S_global_max_solid, S_global_max_melt)
    if S_global_max_solid_ext > S_global_max_solid:
        logger.info(
            'Extending solid entropy range from %.0f to %.0f J/(kg*K) '
            'to match melt range (SPIDER compatibility)',
            S_global_max_solid,
            S_global_max_solid_ext,
        )
        S_global_max_solid = S_global_max_solid_ext

    S_solid_grid = np.linspace(S_global_min_solid, S_global_max_solid, n_S)
    S_melt_grid = np.linspace(S_global_min_melt, S_global_max_melt, n_S)

    logger.info(
        'Entropy ranges: solid [%.0f, %.0f], melt [%.0f, %.0f] J/(kg*K)',
        S_global_min_solid,
        S_global_max_solid,
        S_global_min_melt,
        S_global_max_melt,
    )

    # For each (P, S) in the output grid, invert S(P, T) to find T,
    # then look up rho, cp, alpha, nabla_ad at that (P, T).
    # Inversion: at fixed P, S(T) is monotonically increasing, so we
    # can use a simple bisection over the T range.

    def _fill_phase_grid(
        P_grid,
        S_grid,
        T_lo_func,
        T_hi_func,
        phase_name,
        s_phase_interp,
        rho_phase_interp,
        cp_phase_interp,
        alpha_phase_interp,
        nad_phase_interp,
    ):
        """Fill 2D grids for a single phase by inverting S(P,T) -> T(P,S).

        Parameters
        ----------
        P_grid : ndarray, shape (nP,)
        S_grid : ndarray, shape (nS,)
        T_lo_func : callable
            P [Pa] -> T_min for this phase
        T_hi_func : callable
            P [Pa] -> T_max for this phase
        phase_name : str
            'solid' or 'melt', for logging.
        s_phase_interp, rho_phase_interp, cp_phase_interp,
        alpha_phase_interp, nad_phase_interp : callable
            Phase-specific interpolators for S(logP,logT) -> value.

        Returns
        -------
        dict of ndarray
            Keys: 'rho', 'temperature', 'cp', 'alpha', 'nabla_ad',
            each shape (nS, nP).
        """
        nP_out = len(P_grid)
        nS_out = len(S_grid)
        result = {
            'rho': np.full((nS_out, nP_out), np.nan),
            'temperature': np.full((nS_out, nP_out), np.nan),
            'cp': np.full((nS_out, nP_out), np.nan),
            'alpha': np.full((nS_out, nP_out), np.nan),
            'nabla_ad': np.full((nS_out, nP_out), np.nan),
        }

        n_filled = 0
        n_total = nP_out * nS_out

        for ip in range(nP_out):
            P_Pa = P_grid[ip]
            logP = np.log10(P_Pa)
            T_lo = T_lo_func(P_Pa)
            T_hi = T_hi_func(P_Pa)

            if np.isnan(T_lo) or np.isnan(T_hi) or T_lo <= 0 or T_hi <= 0:
                continue
            if T_lo >= T_hi:
                continue

            T_lo_valid, T_hi_valid = _find_valid_T_bounds(logP, T_lo, T_hi, s_phase_interp)
            if T_lo_valid is None:
                continue

            Sa = float(s_phase_interp((logP, np.log10(T_lo_valid))))
            Sb = float(s_phase_interp((logP, np.log10(T_hi_valid))))

            if np.isnan(Sa) or np.isnan(Sb):
                continue

            S_lo, S_hi = min(Sa, Sb), max(Sa, Sb)

            # Vectorized bisection: invert S(P,T) -> T for all S values at once
            valid_s = (S_grid >= S_lo) & (S_grid <= S_hi)
            idx_s = np.where(valid_s)[0]
            if len(idx_s) == 0:
                continue

            S_batch = S_grid[idx_s]
            Ta_arr = np.full(len(idx_s), T_lo_valid)
            Tb_arr = np.full(len(idx_s), T_hi_valid)

            for _ in range(40):
                Tm_arr = 0.5 * (Ta_arr + Tb_arr)
                logT_arr = np.log10(Tm_arr)
                pts_arr = np.column_stack([np.full(len(idx_s), logP), logT_arr])
                Sm_arr = s_phase_interp(pts_arr)

                below = Sm_arr < S_batch
                Ta_arr = np.where(below, Tm_arr, Ta_arr)
                Tb_arr = np.where(below, Tb_arr, Tm_arr)

                if np.max(Tb_arr - Ta_arr) < 0.01:
                    break

            T_found = 0.5 * (Ta_arr + Tb_arr)
            logT_arr = np.log10(T_found)
            pts = np.column_stack([np.full(len(idx_s), logP), logT_arr])

            result['temperature'][idx_s, ip] = T_found
            result['rho'][idx_s, ip] = rho_phase_interp(pts)
            result['cp'][idx_s, ip] = cp_phase_interp(pts)
            result['alpha'][idx_s, ip] = alpha_phase_interp(pts)
            nad_vals = nad_phase_interp(pts)
            if P_Pa > 0:
                result['nabla_ad'][idx_s, ip] = np.where(
                    np.isfinite(nad_vals), nad_vals * T_found / P_Pa, 0.0
                )
            n_filled += len(idx_s)

        fill_frac = n_filled / n_total if n_total > 0 else 0
        logger.info(
            '%s phase: filled %d/%d cells (%.1f%%)',
            phase_name,
            n_filled,
            n_total,
            100 * fill_frac,
        )
        return result

    # Fill solid and melt phase grids in parallel (independent computations)
    def _T_lo_solid(P):
        return max(table['t_min'], 300.0)

    def _T_hi_melt(P):
        return min(table['t_max'], 1e5)

    from concurrent.futures import ThreadPoolExecutor

    def _fill_solid():
        return _fill_phase_grid(
            P_out,
            S_solid_grid,
            _T_lo_solid,
            solidus_func,
            'solid',
            s_phase_interp=s_solid_interp,
            rho_phase_interp=rho_solid_interp,
            cp_phase_interp=cp_solid_interp,
            alpha_phase_interp=alpha_solid_interp,
            nad_phase_interp=nad_solid_interp,
        )

    def _fill_melt():
        return _fill_phase_grid(
            P_out,
            S_melt_grid,
            liquidus_func,
            _T_hi_melt,
            'melt',
            s_phase_interp=s_melt_interp,
            rho_phase_interp=rho_melt_interp,
            cp_phase_interp=cp_melt_interp,
            alpha_phase_interp=alpha_melt_interp,
            nad_phase_interp=nad_melt_interp,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_solid = pool.submit(_fill_solid)
        fut_melt = pool.submit(_fill_melt)
        solid_grids = fut_solid.result()
        melt_grids = fut_melt.result()

    # Fill NaN cells by nearest-neighbor extrapolation. SPIDER does
    # not handle NaN in its lookup tables.
    for phase_name, grids in [('solid', solid_grids), ('melt', melt_grids)]:
        for prop in ['rho', 'temperature', 'cp', 'alpha', 'nabla_ad']:
            grid = grids[prop]
            n_nan_before = np.sum(np.isnan(grid))
            if n_nan_before > 0:
                _fill_nan_nearest(grid)
                n_nan_after = np.sum(np.isnan(grid))
                if n_nan_after > 0:
                    logger.warning(
                        '%s %s: %d NaN remain after fill (of %d total)',
                        phase_name,
                        prop,
                        n_nan_after,
                        grid.size,
                    )

    # Write files
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Scaling factors for each property
        scales = {
            'rho': 1000.0,  # kg/m^3
            'temperature': 1000.0,  # K
            'cp': 1000.0,  # J/(kg*K)
            'alpha': 1e-5,  # 1/K
            'nabla_ad': 1e-9,  # K/Pa (dT/dP_s, converted from dimensionless nabla_ad)
        }
        spider_names = {
            'rho': 'density',
            'temperature': 'temperature',
            'cp': 'heat_capacity',
            'alpha': 'thermal_exp',
            'nabla_ad': 'adiabat_temp_grad',
        }

        for prop in ['rho', 'temperature', 'cp', 'alpha', 'nabla_ad']:
            spider_name = spider_names[prop]
            scale = scales[prop]

            # Solid
            fpath = str(output_dir / f'{spider_name}_solid.dat')
            _write_spider_2d(fpath, P_out, S_solid_grid, solid_grids[prop], scale)
            logger.info('Wrote %s', fpath)

            # Melt
            fpath = str(output_dir / f'{spider_name}_melt.dat')
            _write_spider_2d(fpath, P_out, S_melt_grid, melt_grids[prop], scale)
            logger.info('Wrote %s', fpath)

    return {
        'P_Pa': P_out,
        'S_solid': S_solid_grid,
        'S_melt': S_melt_grid,
        'solid': solid_grids,
        'melt': melt_grids,
        'output_dir': str(output_dir) if output_dir else None,
    }


# ── Aragog P-T format table generation ─────────────────────────────


def generate_aragog_pt_tables(
    eos_file,
    P_range=(1e5, 200e9),
    n_P=1000,
    n_T=1000,
    output_dir=None,
    solidus_func=None,
    liquidus_func=None,
):
    """Generate Aragog-format P-T lookup tables from a PALEOS unified table.

    Writes the full PALEOS P-T table as both solid-phase and melt-phase
    files for density, heat capacity, and thermal expansivity. Each file
    has 3 columns (pressure, temperature, value) in tab-separated format
    with a 1-line header, matching the format Aragog reads from
    ``interior_lookup_tables/EOS/dynamic/<eos_dir>/P-T/``.

    Both solid and melt files contain the same complete rectangular grid.
    Aragog handles the solid-liquid transition internally via its own
    solidus/liquidus curves and PhaseMixedParameters.

    Parameters
    ----------
    eos_file : str or Path
        Path to the PALEOS unified EOS table.
    P_range : tuple of float
        (P_min, P_max) in Pa.
    n_P : int
        Number of pressure points.
    n_T : int
        Number of temperature points.
    output_dir : str or Path or None
        Directory for output files. Creates it if needed.
    solidus_func : callable or None
        Unused. Kept for backward compatibility.
    liquidus_func : callable or None
        Unused. Kept for backward compatibility.

    Returns
    -------
    dict or None
        Keys: ``'output_dir'``, ``'properties'`` list. None if eos_file
        not found.
    """
    table = load_paleos_all_properties(eos_file)
    if table is None:
        return None

    log_p_grid = table['unique_log_p']
    log_t_grid = table['unique_log_t']

    # Build interpolators
    rho_interp = _build_interpolator(log_p_grid, log_t_grid, table['rho'])
    cp_interp = _build_interpolator(log_p_grid, log_t_grid, table['cp'])
    alpha_interp = _build_interpolator(log_p_grid, log_t_grid, table['alpha'])
    s_interp = _build_interpolator(log_p_grid, log_t_grid, table['s'])

    # P and T grids (linear, matching WB2018 format)
    P_arr = np.linspace(P_range[0], P_range[1], n_P)
    T_arr = np.linspace(max(table['t_min'], 300.0), min(table['t_max'], 1e5), n_T)

    # File names use short form (density, heat_capacity, thermal_exp) to
    # match the existing WB2018 file names that Aragog's PROTEUS wrapper
    # references. The header column name (3rd column) must match Aragog's
    # _ScalingsParameters attribute name for nondimensionalization.
    properties = {
        'density': ('density', rho_interp),
        'heat_capacity': ('heat_capacity', cp_interp),
        'thermal_exp': ('thermal_expansivity', alpha_interp),
        'entropy': ('entropy', s_interp),
    }

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Build full P-T meshgrid for vectorized interpolation.
    # The PALEOS unified table covers the entire P-T rectangle (no gaps
    # except at high T / low P corners for MgSiO3). Write the COMPLETE
    # table for both "melt" and "solid" files: Aragog handles the
    # solid-liquid transition internally via its own solidus/liquidus
    # curves and PhaseMixedParameters. Writing incomplete tables (with
    # phase filtering) creates an irregular grid that forces scipy to
    # use slow unstructured interpolation instead of fast regular-grid
    # interpolation.
    PP, TT = np.meshgrid(P_arr, T_arr, indexing='ij')  # (n_P, n_T)
    logPP = np.log10(np.maximum(PP, 1.0))
    logTT = np.log10(np.maximum(TT, 1.0))
    pts = np.column_stack([logPP.ravel(), logTT.ravel()])

    for file_name, (header_name, interp) in properties.items():
        # Evaluate property on full rectangular grid (vectorized)
        vals = interp(pts).reshape(PP.shape)
        vals = np.where(np.isfinite(vals) & (vals > 0), vals, 1e-15)

        # Write both melt and solid files with the identical full table.
        # PALEOS unified tables do not distinguish phases; Aragog's
        # mushy zone model handles the transition via solidus/liquidus.
        data_out = np.column_stack([PP.ravel(), TT.ravel(), vals.ravel()])

        for phase in ('melt', 'solid'):
            if output_dir is not None:
                fname = output_dir / f'{file_name}_{phase}.dat'
                with open(fname, 'w') as f:
                    f.write(f'#pressure\ttemperature\t{header_name}\n')
                    np.savetxt(f, data_out, fmt='%.10e', delimiter='\t')
                logger.info('Wrote %s (%d rows)', fname, len(data_out))

    return {
        'output_dir': str(output_dir) if output_dir else None,
        'properties': list(properties.keys()),
    }


def generate_aragog_pt_tables_2phase(
    solid_eos_file,
    liquid_eos_file,
    P_range=(1e5, 200e9),
    n_P=1000,
    n_T=1000,
    output_dir=None,
):
    """Generate Aragog-format P-T lookup tables from PALEOS 2-phase tables.

    Unlike ``generate_aragog_pt_tables`` (which uses the unified table and
    writes identical solid/melt files), this function reads from separate
    solid-phase and liquid-phase PALEOS tables. Each Aragog output file
    contains the correct phase-specific properties, giving clean values
    at the solidus and liquidus without interpolation across the melting
    curve discontinuity.

    This is essential for:
    - Correct Delta_S = S_liq(T_liq) - S_sol(T_sol) in the mixing flux
    - Entropy-conserving adiabatic IC via S(P,T) inversion
    - Matching SPIDER's entropy-based formulation

    Parameters
    ----------
    solid_eos_file : str or Path
        Path to the PALEOS solid-phase table (10-column format).
    liquid_eos_file : str or Path
        Path to the PALEOS liquid-phase table (10-column format).
    P_range : tuple of float
        (P_min, P_max) in Pa.
    n_P : int
        Number of pressure points.
    n_T : int
        Number of temperature points.
    output_dir : str or Path or None
        Directory for output files.

    Returns
    -------
    dict or None
        Keys: ``'output_dir'``, ``'properties'`` list.
    """
    solid_table = load_paleos_all_properties(solid_eos_file)
    liquid_table = load_paleos_all_properties(liquid_eos_file)
    if solid_table is None or liquid_table is None:
        return None

    # Properties to export: (file_name_stem, header_column_name)
    prop_map = {
        'density': 'density',
        'heat_capacity': 'heat_capacity',
        'thermal_exp': 'thermal_expansivity',
        'entropy': 'entropy',
    }
    # Map file stems to PALEOS table column names
    paleos_keys = {
        'density': 'rho',
        'heat_capacity': 'cp',
        'thermal_exp': 'alpha',
        'entropy': 's',
    }

    # P and T grids (intersection of solid and liquid T ranges)
    P_arr = np.linspace(P_range[0], P_range[1], n_P)
    T_min = max(solid_table['t_min'], liquid_table['t_min'], 300.0)
    T_max = min(solid_table['t_max'], liquid_table['t_max'], 1e5)
    T_arr = np.linspace(T_min, T_max, n_T)

    logger.info(
        '2-phase table T intersection: [%.0f, %.0f] K '
        '(solid: [%.0f, %.0f], liquid: [%.0f, %.0f])',
        T_min,
        T_max,
        solid_table['t_min'],
        solid_table['t_max'],
        liquid_table['t_min'],
        liquid_table['t_max'],
    )
    if T_max < 6000:
        logger.warning(
            '2-phase T_max=%.0f K is below typical CMB temperatures for '
            'super-Earths. Entropy and density lookups above this T will '
            'be extrapolated.',
            T_max,
        )

    PP, TT = np.meshgrid(P_arr, T_arr, indexing='ij')
    logPP = np.log10(np.maximum(PP, 1.0))
    logTT = np.log10(np.maximum(TT, 1.0))
    pts = np.column_stack([logPP.ravel(), logTT.ravel()])

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    for file_stem, header_name in prop_map.items():
        paleos_key = paleos_keys[file_stem]

        for phase_tag, table in [('solid', solid_table), ('melt', liquid_table)]:
            interp = _build_interpolator(
                table['unique_log_p'], table['unique_log_t'], table[paleos_key]
            )
            vals = interp(pts).reshape(PP.shape)
            vals = np.where(np.isfinite(vals) & (vals > 0), vals, 1e-15)

            data_out = np.column_stack([PP.ravel(), TT.ravel(), vals.ravel()])

            if output_dir is not None:
                fname = output_dir / f'{file_stem}_{phase_tag}.dat'
                with open(fname, 'w') as f:
                    f.write(f'#pressure\ttemperature\t{header_name}\n')
                    np.savetxt(f, data_out, fmt='%.10e', delimiter='\t')
                logger.info('Wrote %s (%d rows)', fname, len(data_out))

    return {
        'output_dir': str(output_dir) if output_dir else None,
        'properties': list(prop_map.keys()),
    }


def compute_surface_entropy(
    eos_file,
    T_surface,
    P_surface=1e5,
    solidus_func=None,
    liquidus_func=None,
    solid_eos_file=None,
    liquid_eos_file=None,
):
    """Compute the entropy of a single (P, T) point using PALEOS tables.

    Thin wrapper that performs just the surface-entropy lookup used to
    initialise an isentropic adiabat, without integrating the full T(P)
    profile down to the CMB. Needed by the entropy-IC cross-check paths
    (``AragogRunner._verify_entropy_ic`` and
    ``common._verify_initial_entropy``) which only need the scalar
    ``S(P_surface, T_surface)`` for comparison against the primary
    ``EntropyEOS.invert_temperature`` inversion.

    The full ``compute_entropy_adiabat`` routine integrates an adiabat
    from the surface to P_cmb and during bracket expansion can reach
    (P_surface, 2*T_surface) tuples that land in the PALEOS "low-P
    high-T" non-converged region (~100% NaN at P<1e7 Pa, T>5000 K for
    MgSiO3 liquid). That region is physically vapour/plasma which the
    liquid EOS fit does not cover. Since the cross-check only needs
    entropy at the surface, this helper avoids the problematic bracket
    expansion entirely.

    Parameters
    ----------
    eos_file : str or Path
        Path to the PALEOS unified EOS table (used when 2-phase tables
        are not provided).
    T_surface : float
        Surface temperature [K].
    P_surface : float
        Surface pressure [Pa]. Default 1 bar = 1e5 Pa.
    solidus_func, liquidus_func : callable or None
        Optional T_solidus(P) and T_liquidus(P) functions. Used only when
        (P_surface, T_surface) falls inside the mushy zone, in which
        case S is computed via phase-weighted blending.
    solid_eos_file, liquid_eos_file : str or Path or None
        Optional PALEOS 2-phase tables. When provided and the point is
        inside the mushy zone, phase-specific entropy values are used
        instead of the unified-table lookup. Avoids the Clausius-
        Clapeyron inconsistency at the melting curve.

    Returns
    -------
    dict
        Keys:
            ``'S_target'`` : float
                Entropy at (P_surface, T_surface) [J/(kg*K)].
            ``'P_surface'`` : float (echoed back).
            ``'T_surface'`` : float (echoed back).

    Raises
    ------
    FileNotFoundError
        If ``eos_file`` cannot be loaded.
    ValueError
        If the entropy lookup at (P_surface, T_surface) returns NaN.
        This is a genuine out-of-table call (rare at surface conditions)
        and indicates a config mismatch or a corrupt table.
    """
    table = load_paleos_all_properties(eos_file)
    if table is None:
        raise FileNotFoundError(f'PALEOS table not found: {eos_file}')

    s_interp = _build_interpolator(table['unique_log_p'], table['unique_log_t'], table['s'])

    s_solid_interp = None
    s_liquid_interp = None
    if solid_eos_file and liquid_eos_file:
        solid_tab = load_paleos_all_properties(solid_eos_file)
        liquid_tab = load_paleos_all_properties(liquid_eos_file)
        if solid_tab is not None and liquid_tab is not None:
            s_solid_interp = _build_interpolator(
                solid_tab['unique_log_p'], solid_tab['unique_log_t'], solid_tab['s']
            )
            s_liquid_interp = _build_interpolator(
                liquid_tab['unique_log_p'], liquid_tab['unique_log_t'], liquid_tab['s']
            )

    def _lookup_unified(P, T):
        pt = np.array([[np.log10(max(P, 1.0)), np.log10(max(T, 300.0))]])
        return float(s_interp(pt).item())

    def _lookup_phase_weighted(P, T):
        if solidus_func is not None and liquidus_func is not None:
            # Coerce to scalar floats: ``solidus_func`` / ``liquidus_func`` may
            # return 1-element ndarrays from a vectorised callable, which would
            # break the chained comparison and the array constructor below.
            # Going via ``.item()`` accepts scalars and 1-element arrays but
            # raises on truly multi-element input, which is the right contract
            # for a single-point lookup.
            T_sol = float(np.asarray(solidus_func(P)).item())
            T_liq = float(np.asarray(liquidus_func(P)).item())
            if T_sol < T < T_liq and T_liq > T_sol:
                phi = (T - T_sol) / (T_liq - T_sol)
                pt_sol = np.array([[np.log10(max(P, 1.0)), np.log10(max(T_sol, 300.0))]])
                pt_liq = np.array([[np.log10(max(P, 1.0)), np.log10(max(T_liq, 300.0))]])
                if s_solid_interp is not None and s_liquid_interp is not None:
                    S_sol = float(s_solid_interp(pt_sol).item())
                    S_liq = float(s_liquid_interp(pt_liq).item())
                else:
                    S_sol = float(s_interp(pt_sol).item())
                    S_liq = float(s_interp(pt_liq).item())
                return phi * S_liq + (1 - phi) * S_sol
        return _lookup_unified(P, T)

    S_target = _lookup_phase_weighted(P_surface, T_surface)

    if not np.isfinite(S_target):
        raise ValueError(
            f'PALEOS entropy lookup at (P={P_surface:.2e} Pa, '
            f'T={T_surface:.1f} K) returned NaN. This should not happen '
            f'at surface conditions. Check eos_file={eos_file}.'
        )

    logger.info(
        'Surface entropy: T_surf=%.1f K, P_surf=%.2e Pa, S_target=%.2f J/(kg*K)',
        T_surface,
        P_surface,
        S_target,
    )

    return {
        'S_target': S_target,
        'P_surface': float(P_surface),
        'T_surface': float(T_surface),
    }


def compute_entropy_adiabat(
    eos_file,
    T_surface,
    P_surface=1e5,
    P_cmb=135e9,
    n_points=500,
    solidus_func=None,
    liquidus_func=None,
    solid_eos_file=None,
    liquid_eos_file=None,
):
    """Compute an entropy-conserving adiabatic T(P) profile from PALEOS.

    Starting from T_surface at P_surface, computes the target entropy
    S_target = S(P_surface, T_surface) and then inverts S(P, T) = S_target
    at each pressure to find T. This correctly handles phase boundaries
    (solidus/liquidus) where the Clausius-Clapeyron slope differs from
    the single-phase adiabatic gradient.

    When ``solid_eos_file`` and ``liquid_eos_file`` are provided (2-phase
    PALEOS tables), uses phase-specific entropy values for the mushy zone.
    This gives clean Delta_S without interpolation across the melting curve
    discontinuity. Falls back to the unified table if 2-phase tables are
    not provided.

    Parameters
    ----------
    eos_file : str or Path
        Path to the PALEOS unified EOS table (used for single-phase regions
        and as fallback).
    T_surface : float
        Surface temperature [K].
    P_surface : float
        Surface pressure [Pa].
    P_cmb : float
        CMB pressure [Pa].
    n_points : int
        Number of pressure points in the profile.
    solidus_func : callable or None
        P [Pa] -> T_solidus [K]. Required for mixed-phase entropy.
    liquidus_func : callable or None
        P [Pa] -> T_liquidus [K]. Required for mixed-phase entropy.
    solid_eos_file : str or Path or None
        Path to PALEOS solid-phase table. When provided with
        ``liquid_eos_file``, uses phase-specific entropy.
    liquid_eos_file : str or Path or None
        Path to PALEOS liquid-phase table.

    Returns
    -------
    dict
        Keys: ``'P'`` [Pa], ``'T'`` [K], ``'S_target'`` [J/(kg*K)],
        ``'S_profile'`` [J/(kg*K)] (entropy at each point for verification).
    """
    from scipy.optimize import brentq

    table = load_paleos_all_properties(eos_file)
    if table is None:
        raise FileNotFoundError(f'PALEOS table not found: {eos_file}')

    log_p_grid = table['unique_log_p']
    log_t_grid = table['unique_log_t']
    s_interp = _build_interpolator(log_p_grid, log_t_grid, table['s'])

    # Build phase-specific entropy interpolators if 2-phase tables available
    s_solid_interp = None
    s_liquid_interp = None
    if solid_eos_file and liquid_eos_file:
        solid_tab = load_paleos_all_properties(solid_eos_file)
        liquid_tab = load_paleos_all_properties(liquid_eos_file)
        if solid_tab is not None and liquid_tab is not None:
            s_solid_interp = _build_interpolator(
                solid_tab['unique_log_p'], solid_tab['unique_log_t'], solid_tab['s']
            )
            s_liquid_interp = _build_interpolator(
                liquid_tab['unique_log_p'], liquid_tab['unique_log_t'], liquid_tab['s']
            )
            logger.info('Using PALEOS-2phase tables for entropy adiabat')

    def entropy_single_phase(P, T):
        """Evaluate single-phase entropy from PALEOS."""
        pt = np.array([[np.log10(max(P, 1.0)), np.log10(max(T, 300.0))]])
        return float(s_interp(pt).item())

    def entropy_total(P, T):
        """Evaluate total entropy including mixed-phase contribution.

        In the mushy zone, S = phi * S_liq(P, T_liq) + (1-phi) * S_sol(P, T_sol).
        Uses phase-specific tables when available for clean values at
        the solidus/liquidus.
        """
        if solidus_func is not None and liquidus_func is not None:
            # Coerce to scalar floats: ``solidus_func`` / ``liquidus_func`` may
            # return 1-element ndarrays from a vectorised callable, which would
            # break the chained comparison and the array constructor below.
            # Going via ``.item()`` accepts scalars and 1-element arrays but
            # raises on truly multi-element input, which is the right contract
            # for a single-point lookup.
            T_sol = float(np.asarray(solidus_func(P)).item())
            T_liq = float(np.asarray(liquidus_func(P)).item())
            if T_sol < T < T_liq and T_liq > T_sol:
                phi = (T - T_sol) / (T_liq - T_sol)
                pt_sol = np.array([[np.log10(max(P, 1.0)), np.log10(max(T_sol, 300.0))]])
                pt_liq = np.array([[np.log10(max(P, 1.0)), np.log10(max(T_liq, 300.0))]])
                if s_solid_interp is not None and s_liquid_interp is not None:
                    S_sol = float(s_solid_interp(pt_sol).item())
                    S_liq = float(s_liquid_interp(pt_liq).item())
                else:
                    S_sol = float(s_interp(pt_sol).item())
                    S_liq = float(s_interp(pt_liq).item())
                return phi * S_liq + (1 - phi) * S_sol
        return entropy_single_phase(P, T)

    # Compute target entropy at the surface
    S_target = entropy_total(P_surface, T_surface)
    logger.info(
        'Entropy adiabat: T_surf=%.1f K, P_surf=%.2e Pa, S_target=%.2f J/(kg*K)',
        T_surface,
        P_surface,
        S_target,
    )

    # Build pressure grid (log-spaced for better resolution at low P)
    P_grid = np.logspace(np.log10(P_surface * 1.001), np.log10(P_cmb * 0.999), n_points)
    T_profile = np.zeros(n_points)
    S_profile = np.zeros(n_points)

    T_prev = T_surface
    n_nan_bracket = 0
    for i, P_i in enumerate(P_grid):

        def residual(T_cand):
            return entropy_total(P_i, T_cand) - S_target

        # Bracket: expand around previous T. If either endpoint lands
        # in a non-converged PALEOS cell (NaN), shrink it toward T_prev
        # rather than expanding, because the PALEOS tables have a 100%
        # NaN region at low P / high T (MgSiO3 vapour regime). Expanding
        # into that region propagates NaN into brentq and crashes the
        # whole run (was a historical bug: `NaN > 0` is False, `NaN *
        # NaN > 0` is False, so the bracket-expansion loop exited
        # immediately with NaN endpoints).
        T_lo = T_prev * 0.8
        T_hi = T_prev * 2.0
        s_lo = residual(T_lo)
        s_hi = residual(T_hi)

        n_expand = 0
        while (
            not np.isfinite(s_lo) or not np.isfinite(s_hi) or s_lo * s_hi > 0
        ) and n_expand < 30:
            if not np.isfinite(s_lo):
                # Shrink T_lo back toward T_prev instead of expanding
                T_lo = 0.5 * (T_lo + T_prev)
                s_lo = residual(T_lo)
            elif not np.isfinite(s_hi):
                # Shrink T_hi back toward T_prev instead of expanding.
                # This is the critical case: at P=P_surface the default
                # T_hi = 2*T_surface can land in the vapour/plasma region
                # where the PALEOS liquid table has no data.
                T_hi = 0.5 * (T_hi + T_prev)
                s_hi = residual(T_hi)
            elif s_lo > 0:
                T_lo *= 0.5
                s_lo = residual(T_lo)
            else:
                T_hi *= 2.0
                s_hi = residual(T_hi)
            n_expand += 1

        bracket_ok = np.isfinite(s_lo) and np.isfinite(s_hi) and s_lo * s_hi <= 0
        if not bracket_ok:
            n_nan_bracket += 1
            logger.warning(
                'Could not bracket S root at P=%.2e Pa '
                '(s_lo=%s, s_hi=%s). Using previous T=%.1f K.',
                P_i,
                s_lo,
                s_hi,
                T_prev,
            )
            T_profile[i] = T_prev
        else:
            T_profile[i] = brentq(residual, T_lo, T_hi, rtol=1e-10)

        S_profile[i] = entropy_total(P_i, T_profile[i])
        T_prev = T_profile[i]

    if n_nan_bracket > 0:
        logger.warning(
            'Entropy adiabat: %d/%d bracket-expansion failures due to '
            'PALEOS non-converged cells. Profile may be flat in those '
            'regions; treat with caution.',
            n_nan_bracket,
            n_points,
        )

    S_drift = abs(S_profile[-1] - S_target) / abs(S_target) * 100
    logger.info('Entropy adiabat: T_cmb=%.1f K, S_drift=%.4f%%', T_profile[-1], S_drift)

    return {
        'P': P_grid,
        'T': T_profile,
        'S_target': S_target,
        'S_profile': S_profile,
    }
