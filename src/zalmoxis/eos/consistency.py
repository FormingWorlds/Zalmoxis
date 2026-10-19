"""Thermodynamic consistency residuals of a regular (log10 P, log10 T) EOS table.

Each identity is written as a relative residual that is 0 for an exact EOS. The derivatives are
central differences in x = ln P and y = ln T with a step of k grid cells; a stencil counts only
when all its cells hold numbers of one phase, so a phase boundary never enters a derivative.
"""

from __future__ import annotations

import numpy as np

IDENTITIES = ('cp', 'maxwell', 'alpha', 'dudt', 'nabla_ad', 'cp_cv')
EPS_FLOOR = 5e-9  # 9 significant digits, the precision of the shipped tables


def segments(table, axis):
    """Label runs of valid cells of one phase along ``axis``; invalid cells get -1."""
    cols = ('rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad')
    valid = np.logical_and.reduce([np.isfinite(table[c]) for c in cols])
    _, code = np.unique(table['phase'].astype(str), return_inverse=True)
    code = np.where(valid, code.reshape(valid.shape), -1)
    start = np.ones(code.shape, bool)
    inner = tuple(slice(1, None) if a == axis else slice(None) for a in range(code.ndim))
    start[inner] = np.diff(code, axis=axis) != 0
    return np.where(valid, np.cumsum(start, axis=axis), -1)


def _stencil(f, seg, axis, k, h, sign):
    """(f[i+k] + sign f[i-k]) / (2 k h) along ``axis``, NaN where the stencil leaves one segment."""
    n = f.shape[axis]
    out = np.full(f.shape, np.nan)
    if n <= 2 * k:
        return out

    def cut(a, b):
        return tuple(slice(a, b) if ax == axis else slice(None) for ax in range(f.ndim))

    mid, lo, hi = seg[cut(k, n - k)], seg[cut(0, n - 2 * k)], seg[cut(2 * k, n)]
    ok = (mid >= 0) & (lo == mid) & (hi == mid)
    out[cut(k, n - k)] = np.where(
        ok, (f[cut(2 * k, n)] + sign * f[cut(0, n - 2 * k)]) / (2 * k * h), np.nan
    )
    return out


def derivative(f, seg, axis, k, h):
    """d f / d(axis) with step ``k h``, NaN where the stencil leaves one segment."""
    return _stencil(f, seg, axis, k, h, -1.0)


def _grid(table):
    lp, lt = np.asarray(table['unique_log_p'], float), np.asarray(table['unique_log_t'], float)
    P, T = np.meshgrid(10.0**lp, 10.0**lt, indexing='ij')
    cols = (
        np.asarray(table[c], float) for c in ('rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad')
    )
    return np.log(10.0) * np.diff(lp).mean(), np.log(10.0) * np.diff(lt).mean(), P, T, *cols


def _parts(table, k):
    """Per derivative identity (D, E, Q, q, axis) with R + 1 = D / Q: E bounds the error of D per
    unit of stored precision, q the relative error of Q per unit of stored precision."""
    hx, hy, P, T, rho, u, s, cp, cv, alpha, _ = _grid(table)
    seg = {0: segments(table, 0), 1: segments(table, 1)}
    h = {0: hx, 1: hy}
    one = np.ones_like(rho)
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_rho, a = np.log(rho), P * alpha / rho

        def d(f, ax, log=False):  # a stored log has an absolute error of 1 per unit precision
            ends = one if log else np.abs(f)
            return derivative(f, seg[ax], ax, k, h[ax]), _stencil(
                ends, seg[ax], ax, k, h[ax], 1.0
            )

        cpcv = (np.abs(cp) + np.abs(cv)) / np.abs(cp - cv) + 3
        return {
            'cp': (*d(s, 1), cp, 1.0, 1),
            'maxwell': (*d(s, 0), -a, 2.0, 0),
            'alpha': (*d(ln_rho, 1, log=True), -T * alpha, 1.0, 1),
            'dudt': (*d(u, 1), T * (cp - a), (np.abs(cp) + 2 * np.abs(a)) / np.abs(cp - a), 1),
            'cp_cv': (*d(ln_rho, 0, log=True), P * T * alpha**2 / ((cp - cv) * rho), cpcv, 0),
        }


def residuals(table, k=1):
    """Relative residuals of the identities at stencil step ``k`` cells.

    Parameters
    ----------
    table : dict
        ``unique_log_p``, ``unique_log_t`` (uniform, log10 Pa and K) and the columns rho, u, s,
        cp, cv, alpha, nabla_ad (SI) and phase on the (P, T) grid, as from
        ``zalmoxis.eos_export.load_paleos_all_properties``.
    k : int
        Stencil step in grid cells.

    Returns
    -------
    dict
        Identity name -> residual array on the grid (NaN where it cannot be formed):
        ``cp``: (ds/dlnT)_P / cp - 1; ``maxwell``: (ds/dlnP)_T / (-P alpha / rho) - 1;
        ``alpha``: -(dln rho/dlnT)_P / (T alpha) - 1; ``dudt``: (du/dlnT)_P / T / (cp - P alpha
        / rho) - 1; ``nabla_ad``: nabla_ad rho cp / (P alpha) - 1 (no derivative);
        ``cp_cv``: (cp - cv) rho kappa_T / (T alpha^2) - 1 with kappa_T = (dln rho/dlnP)_T / P.
    """
    _, _, P, _, rho, _, _, cp, _, alpha, nab = _grid(table)
    parts = _parts(table, k)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = {n: D / Q - 1 for n, (D, _, Q, _, _) in parts.items()}
        out['nabla_ad'] = nab * rho * cp / (P * alpha) - 1
    return {n: out[n] for n in IDENTITIES}


def floors(table, eps, k=1):
    """Round-off bound of each residual at step ``k`` for columns stored to relative ``eps``.

    For R + 1 = D / Q the bound is (E / |Q| + |D / Q| q) eps, with E the error of the central
    difference from its two stored ends and q the relative error of Q; nabla_ad, with no
    derivative, gets |R + 1| 4 eps from its four stored factors.
    """
    _, _, P, _, rho, _, _, cp, _, alpha, nab = _grid(table)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = {
            n: (E / np.abs(Q) + np.abs(D / Q) * q) * eps
            for n, (D, E, Q, q, _) in _parts(table, k).items()
        }
        out['nabla_ad'] = np.abs(nab * rho * cp / (P * alpha)) * 4 * eps
    return {n: out[n] for n in IDENTITIES}


def _number(name, value, ok, rule):
    """``value`` as a float when it is a finite real number with ``ok(value)``, else ValueError."""
    real = isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, (bool, np.bool_)
    )
    if not (real and np.isfinite(value) and ok(value)):
        raise ValueError(f'{name} must be a finite number {rule}, got {value!r}')
    return float(value)


def _eps_factor(eps, factor):
    """The checked eps, raised to ``EPS_FLOOR``, and the checked factor."""
    eps = _number('eps', eps, lambda x: 0 <= x < 1, 'in [0, 1)')
    return max(eps, EPS_FLOOR), _number('factor', factor, lambda x: x > 0, 'above 0')


def check(table, eps, factor=2.0):
    """Test every cell against the Richardson bound of each identity.

    A derivative identity fails where |R_x(h)| > floor_x + factor * est, with R_x(kh) =
    (4 R_k - R_2k) / 3, floor_x = (4 floor_1 + floor_2) / 3 and est = |R_x(2h) - R_x(h)| / 15,
    the measured h^4 term of R_x(h) at the cell. nabla_ad fails where |R| > floor. At 9 stored
    digits (eps 5e-9) the exact ideal gas and Murnaghan tables give no failing cell, and one
    planted value is found at the cells one and two steps from it.

    The floor uses eps_eff = max(eps, EPS_FLOOR): every table is checked at the 9-digit
    floor at best, so faults below about 2e-8 (relative) are out of reach on any table.

    Parameters
    ----------
    table : dict
        As for ``residuals``.
    eps : float
        Relative precision of the stored columns (0.5 x 10^(1 - n) for n significant digits).
    factor : float
        Allowed multiple of the h^4 term.

    Returns
    -------
    dict
        Identity name -> dict with ``rx`` (the tested residual), ``bound``, ``tested`` (both
        finite) and ``fail`` (tested and above the bound) arrays.
    Raises
    ------
    ValueError
        If eps is not a finite number in [0, 1) or factor is not a finite number above 0.
    """
    eps, factor = _eps_factor(eps, factor)
    r = {k: residuals(table, k) for k in (1, 2, 4)}
    f1, f2 = floors(table, eps, 1), floors(table, eps, 2)
    out = {}
    with np.errstate(invalid='ignore'):
        for n in IDENTITIES:
            if n == 'nabla_ad':
                rx, bound = r[1][n], f1[n]
            else:
                rx = (4 * r[1][n] - r[2][n]) / 3
                est = np.abs((4 * r[2][n] - r[4][n]) / 3 - rx) / 15
                bound = (4 * f1[n] + f2[n]) / 3 + factor * est
            tested = np.isfinite(rx) & np.isfinite(bound)
            out[n] = {
                'rx': rx,
                'bound': bound,
                'tested': tested,
                'fail': tested & (np.abs(rx) > bound),
            }
    return out


def signs(table, eps):
    """Cells that break a sign condition beyond the stored precision ``eps``.

    Parameters
    ----------
    table : dict
        As for ``residuals``.
    eps : float
        Relative precision of the stored columns.

    Returns
    -------
    dict
        Condition name -> boolean array, True where the cell breaks it (a missing value breaks
        none): ``rho``, ``cp`` and ``cv`` not positive; ``cp_ge_cv``: cv above cp by more than eps (|cp| + |cv|);
        ``kappa_T``: rho falls with P at fixed T by more than the rounding of its two
        neighbours (central difference over one cell on each side, one phase).
    Raises
    ------
    ValueError
        If eps is not a finite number in [0, 1).
    """
    eps = _number('eps', eps, lambda x: 0 <= x < 1, 'in [0, 1)')
    rho, cp, cv = (np.asarray(table[c], float) for c in ('rho', 'cp', 'cv'))
    seg = segments(table, 0)
    rise = _stencil(rho, seg, 0, 1, 0.5, -1.0)  # rho[i+1] - rho[i-1]
    noise = _stencil(np.abs(rho), seg, 0, 1, 0.5, 1.0) * eps
    with np.errstate(invalid='ignore'):
        return {
            'rho': rho <= 0,
            'cp': cp <= 0,
            'cv': cv <= 0,
            'cp_ge_cv': cv - cp > eps * (np.abs(cp) + np.abs(cv)),
            'kappa_T': rise < -noise,
        }


PT_IDENTITIES = ('cp', 'maxwell', 'alpha')
PT_QUANTITIES = ('density', 'heat_capacity', 'thermal_exp', 'entropy')
PT_FILL = 1e-15  # the value eos_export writes for a cell without data


def _coord_stencil(f, x, axis, k, sign):
    """(f[i+k] + sign f[i-k]) / (x[i+k] - x[i-k]), NaN where a stencil cell is missing."""
    n = f.shape[axis]
    out = np.full(f.shape, np.nan)
    if n <= 2 * k:
        return out
    run = np.cumsum(~np.isfinite(f), axis=axis)

    def cut(a, b):
        return tuple(slice(a, b) if ax == axis else slice(None) for ax in range(f.ndim))

    shape = [1] * f.ndim
    shape[axis] = n - 2 * k
    dx = (x[2 * k :] - x[: n - 2 * k]).reshape(shape)
    ok = np.isfinite(f[cut(0, n - 2 * k)]) & (run[cut(2 * k, n)] == run[cut(0, n - 2 * k)])
    out[cut(k, n - k)] = np.where(
        ok, (f[cut(2 * k, n)] + sign * f[cut(0, n - 2 * k)]) / dx, np.nan
    )
    return out


def _pt_parts(tables, k):
    """(D, E, Q, q) per identity of the Aragog P-T tables, with R + 1 = D / Q."""
    P, T = np.asarray(tables['density']['P'], float), np.asarray(tables['density']['T'], float)
    rho, cp, alpha, s = (
        np.where(np.asarray(tables[q]['values'], float) == PT_FILL, np.nan, tables[q]['values'])
        for q in PT_QUANTITIES
    )
    TT = np.broadcast_to(T[None, :], rho.shape)
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_rho = np.log(rho)
        one = np.where(np.isfinite(ln_rho), 1.0, np.nan)

        def d(f, x, ax, log=False):
            ends = one if log else np.abs(f)
            return _coord_stencil(f, x, ax, k, -1.0), _coord_stencil(ends, x, ax, k, 1.0)

        return {
            'cp': (*d(s, T, 1), cp / TT, 1.0),
            'maxwell': (*d(s, P, 0), -alpha / rho, 2.0),
            'alpha': (*d(ln_rho, T, 1, log=True), -alpha, 1.0),
        }


def check_pt(tables, eps, factor=2.0):
    """``check`` for the Aragog P-T tables that eos_export writes (no u, cv or nabla_ad).

    Parameters
    ----------
    tables : dict
        ``PT_QUANTITIES`` name -> dict with ``P`` (Pa) and ``T`` (K) grid vectors and ``values``
        of shape (n_P, n_T); the fill value ``PT_FILL`` counts as a missing cell.
    eps : float
        Relative precision of the stored values; raised to ``EPS_FLOOR`` at least.
    factor : float
        Allowed multiple of the h^4 term.

    Returns
    -------
    dict
        As for ``check``, for the identities cp = T (ds/dT)_P, (ds/dP)_T = -alpha / rho and
        alpha = -(dln rho/dT)_P.
    Raises
    ------
    ValueError
        If eps is not a finite number in [0, 1) or factor is not a finite number above 0.
    """
    eps, factor = _eps_factor(eps, factor)
    parts = {k: _pt_parts(tables, k) for k in (1, 2, 4)}
    out = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        for n in PT_IDENTITIES:
            r = {k: parts[k][n][0] / parts[k][n][2] - 1 for k in (1, 2, 4)}
            fl = {
                k: (parts[k][n][1] / np.abs(parts[k][n][2]) + np.abs(r[k] + 1) * parts[k][n][3])
                * eps
                for k in (1, 2)
            }
            rx = (4 * r[1] - r[2]) / 3
            est = np.abs((4 * r[2] - r[4]) / 3 - rx) / 15
            bound = (4 * fl[1] + fl[2]) / 3 + factor * est
            tested = np.isfinite(rx) & np.isfinite(bound)
            out[n] = {
                'rx': rx,
                'bound': bound,
                'tested': tested,
                'fail': tested & (np.abs(rx) > bound),
            }
    return out
