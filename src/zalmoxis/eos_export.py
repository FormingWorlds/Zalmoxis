"""Export Zalmoxis/PALEOS EOS data to SPIDER-compatible P-S format.

SPIDER uses entropy-pressure (P-S) coordinates for its 2D EOS lookup
tables and 1D phase boundaries. Zalmoxis and the PALEOS unified tables
use pressure-temperature (P-T) coordinates. This module converts
between the two coordinate systems.

Phase boundary conversion (Phase 1B):
    T(P) melting curves -> S(P) phase boundaries via direct PALEOS
    S(P,T) lookup. No inversion needed.

Full EOS table generation (Phase 1A):
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


# ── Phase boundary generation (Phase 1B) ────────────────────────────


def generate_spider_phase_boundaries(
    solidus_func,
    liquidus_func,
    eos_file,
    P_range=(1e5, 200e9),
    n_P=1000,
    output_dir=None,
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
    # Load PALEOS table and build entropy interpolator
    logger.info('Loading PALEOS table for phase boundary generation: %s', eos_file)
    table = load_paleos_all_properties(eos_file)
    s_interp = _build_interpolator(table['unique_log_p'], table['unique_log_t'], table['s'])

    # Build log-spaced pressure grid
    P_grid = np.logspace(np.log10(P_range[0]), np.log10(P_range[1]), n_P)
    log_P = np.log10(P_grid)

    # Compute melting curve temperatures
    T_sol = np.atleast_1d(solidus_func(P_grid))
    T_liq = np.atleast_1d(liquidus_func(P_grid))

    # Look up entropy at (P, T_solidus) and (P, T_liquidus)
    S_solidus = np.full(n_P, np.nan)
    S_liquidus = np.full(n_P, np.nan)

    for i in range(n_P):
        if np.isnan(T_sol[i]) or T_sol[i] <= 0:
            continue
        S_solidus[i] = float(s_interp((log_P[i], np.log10(T_sol[i]))))

        if np.isnan(T_liq[i]) or T_liq[i] <= 0:
            continue
        S_liquidus[i] = float(s_interp((log_P[i], np.log10(T_liq[i]))))

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


# ── Full EOS table generation (Phase 1A) ────────────────────────────


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

    # Build interpolators for all needed properties
    s_interp = _build_interpolator(log_p_grid, log_t_grid, table['s'])
    rho_interp = _build_interpolator(log_p_grid, log_t_grid, table['rho'])
    cp_interp = _build_interpolator(log_p_grid, log_t_grid, table['cp'])
    alpha_interp = _build_interpolator(log_p_grid, log_t_grid, table['alpha'])
    nad_interp = _build_interpolator(log_p_grid, log_t_grid, table['nabla_ad'])

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

    # Sample T finely to determine S ranges
    n_T_sample = 500
    for ip in range(n_P):
        P_Pa = P_out[ip]
        T_sol = solidus_func(P_Pa)
        T_liq = liquidus_func(P_Pa)

        if np.isnan(T_sol) or np.isnan(T_liq) or T_sol <= 0 or T_liq <= 0:
            continue

        # Solid phase: T from T_min to T_solidus
        T_lo = max(table['t_min'], 300.0)
        if T_sol > T_lo:
            T_solid_range = np.linspace(T_lo, T_sol, n_T_sample)
            for T in T_solid_range:
                S_val = float(s_interp((np.log10(P_Pa), np.log10(T))))
                if np.isfinite(S_val):
                    S_min_solid[ip] = min(S_min_solid[ip], S_val)
                    S_max_solid[ip] = max(S_max_solid[ip], S_val)

        # Melt phase: T from T_liquidus to T_max
        T_hi = min(table['t_max'], 1e5)
        if T_liq < T_hi:
            T_melt_range = np.linspace(T_liq, T_hi, n_T_sample)
            for T in T_melt_range:
                S_val = float(s_interp((np.log10(P_Pa), np.log10(T))))
                if np.isfinite(S_val):
                    S_min_melt[ip] = min(S_min_melt[ip], S_val)
                    S_max_melt[ip] = max(S_max_melt[ip], S_val)

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

    def _fill_phase_grid(P_grid, S_grid, T_lo_func, T_hi_func, phase_name):
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

            # Narrow T bounds to the valid data region at this pressure.
            # The PALEOS table has NaN entries at extreme temperatures, so
            # we search inward from both ends to find the last valid T.
            T_lo_valid, T_hi_valid = _find_valid_T_bounds(logP, T_lo, T_hi, s_interp)
            if T_lo_valid is None:
                continue

            Sa = float(s_interp((logP, np.log10(T_lo_valid))))
            Sb = float(s_interp((logP, np.log10(T_hi_valid))))

            if np.isnan(Sa) or np.isnan(Sb):
                continue

            for js in range(nS_out):
                S_target = S_grid[js]

                if S_target < min(Sa, Sb) or S_target > max(Sa, Sb):
                    continue

                # Bisect to find T such that S(P, T) = S_target
                Ta_bs, Tb_bs = T_lo_valid, T_hi_valid

                # Bisection (40 iterations gives ~1e-12 relative precision)
                for _ in range(40):
                    Tm = 0.5 * (Ta_bs + Tb_bs)
                    Sm = float(s_interp((logP, np.log10(Tm))))
                    if np.isnan(Sm):
                        break
                    if Sm < S_target:
                        Ta_bs = Tm
                    else:
                        Tb_bs = Tm
                    if abs(Tb_bs - Ta_bs) < 0.01:  # 0.01 K precision
                        break

                T_found = 0.5 * (Ta_bs + Tb_bs)
                logT = np.log10(T_found)
                pt = (logP, logT)

                result['temperature'][js, ip] = T_found
                result['rho'][js, ip] = float(rho_interp(pt))
                result['cp'][js, ip] = float(cp_interp(pt))
                result['alpha'][js, ip] = float(alpha_interp(pt))
                result['nabla_ad'][js, ip] = float(nad_interp(pt))
                n_filled += 1

        fill_frac = n_filled / n_total if n_total > 0 else 0
        logger.info(
            '%s phase: filled %d/%d cells (%.1f%%)',
            phase_name,
            n_filled,
            n_total,
            100 * fill_frac,
        )
        return result

    # Fill solid phase grids
    def _T_lo_solid(P):
        return max(table['t_min'], 300.0)

    solid_grids = _fill_phase_grid(
        P_out,
        S_solid_grid,
        _T_lo_solid,
        solidus_func,
        'solid',
    )

    # Fill melt phase grids
    def _T_hi_melt(P):
        return min(table['t_max'], 1e5)

    melt_grids = _fill_phase_grid(
        P_out,
        S_melt_grid,
        liquidus_func,
        _T_hi_melt,
        'melt',
    )

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
            'nabla_ad': 1.0,  # dimensionless
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
