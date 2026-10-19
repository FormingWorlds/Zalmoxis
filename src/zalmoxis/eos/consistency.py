"""Thermodynamic consistency residuals of a regular (log10 P, log10 T) EOS table.

Each identity is written as a relative residual that is 0 for an exact EOS. The derivatives are
central differences in x = ln P and y = ln T with a step of k grid cells; a stencil counts only
when all its cells hold numbers of one phase, so a phase boundary never enters a derivative.
"""

from __future__ import annotations

import numpy as np

IDENTITIES = ('cp', 'maxwell', 'alpha', 'dudt', 'nabla_ad', 'cp_cv')


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


def check(table, eps, factor=2.0):
    """Test every cell against the Richardson bound of each identity.

    A derivative identity fails where |R_x(h)| > floor_x + factor * est, with R_x(kh) =
    (4 R_k - R_2k) / 3, floor_x = (4 floor_1 + floor_2) / 3 and est = |R_x(2h) - R_x(h)| / 15,
    the measured h^4 term of R_x(h) at the cell. nabla_ad fails where |R| > floor. At 9 stored
    digits (eps 5e-9) the exact ideal gas and Murnaghan tables give no failing cell, and one
    planted value is found at the cells one and two steps from it.

    Limits for tables in full double precision (eps 4 x machine eps, 150 points per decade):
    one exact Murnaghan cell fails cp_cv at 0.67 MPa, where R_x is round-off scatter at the
    floor and the h^4 estimate of that cell is small by chance. The exact ideal gas fails cp_cv
    in 33,686 cells: its truncation is zero, and the steps of ln(10**log10 P) depart from the
    mean step by up to 3e-13, beyond the floor, which models only the stored values.

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
    """
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
        Condition name -> boolean array, True where the cell breaks it: ``rho``, ``cp`` and
        ``cv`` not positive; ``cp_ge_cv``: cv above cp by more than eps (|cp| + |cv|);
        ``kappa_T``: rho falls with P at fixed T by more than the rounding of its two
        neighbours (central difference over one cell on each side, one phase).
    """
    rho, cp, cv = (np.asarray(table[c], float) for c in ('rho', 'cp', 'cv'))
    seg = segments(table, 0)
    rise = _stencil(rho, seg, 0, 1, 0.5, -1.0)  # rho[i+1] - rho[i-1]
    noise = _stencil(np.abs(rho), seg, 0, 1, 0.5, 1.0) * eps
    with np.errstate(invalid='ignore'):
        return {
            'rho': ~(rho > 0),
            'cp': ~(cp > 0),
            'cv': ~(cv > 0),
            'cp_ge_cv': cv - cp > eps * (np.abs(cp) + np.abs(cv)),
            'kappa_T': rise < -noise,
        }
