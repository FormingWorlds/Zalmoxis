"""Thermodynamic consistency residuals of a regular (log10 P, log10 T) EOS table.

Each identity is written as a relative residual that is 0 for an exact EOS. The derivatives are
central differences in x = ln P and y = ln T with a step of k grid cells; a stencil counts only
when all its cells hold numbers of one phase, so a phase boundary never enters a derivative.
"""

from __future__ import annotations

import numbers

import numpy as np

IDENTITIES = ('cp', 'maxwell', 'alpha', 'dudt', 'nabla_ad', 'cp_cv')
COLUMNS = ('rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad')
INPUTS = {
    'cp': ('s', 'cp'),
    'maxwell': ('s', 'rho', 'alpha'),
    'alpha': ('rho', 'alpha'),
    'dudt': ('u', 'cp', 'rho', 'alpha'),
    'nabla_ad': ('nabla_ad', 'rho', 'cp', 'alpha'),
    'cp_cv': ('rho', 'alpha', 'cp', 'cv'),
}
EPS_FLOOR = 5e-9  # 9 significant digits, the precision of the shipped tables
FACTOR = 2.0  # multiple of the h^4 term; a fault 2 cells away is found only below 3


def _phase_code(table):
    """Integer code of the phase of each cell."""
    _, code = np.unique(table['phase'].astype(str), return_inverse=True)
    return code.reshape(np.shape(table['phase']))


def segments(table, axis, cols=COLUMNS, valid=np.isfinite, code=None):
    """Label runs of cells of one phase along ``axis``; cells where ``valid`` (a function of a
    column) is false for one of ``cols`` get -1. ``code`` is ``_phase_code(table)`` if given."""
    ok = np.logical_and.reduce([valid(np.asarray(table[c], float)) for c in cols])
    code = np.where(ok, _phase_code(table) if code is None else code, -1)
    start = np.ones(code.shape, bool)
    inner = tuple(slice(1, None) if a == axis else slice(None) for a in range(code.ndim))
    start[inner] = np.diff(code, axis=axis) != 0
    return np.where(ok, np.cumsum(start, axis=axis), -1)


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


GRID_FIT = 4e-9  # log10 departure from a uniform grid; 9-digit coordinates give up to ~2e-9


def _departure(x):
    """Absolute departure of each point of the axis ``x`` from its linear fit."""
    i = np.arange(x.size)
    return np.abs(x - np.polyval(np.polyfit(i, x, 1), i))


def _axes(table):
    """The two log10 axes, checked: at least 2 points, finite, increasing, within GRID_FIT of
    a uniform grid, and every column on their shape."""
    lp, lt = np.asarray(table['unique_log_p'], float), np.asarray(table['unique_log_t'], float)
    for name, x in (('unique_log_p', lp), ('unique_log_t', lt)):
        if x.size < 2:
            raise ValueError(f'{name} needs at least 2 points')
        if not (np.isfinite(x).all() and (np.diff(x) > 0).all()):
            raise ValueError(f'{name} must be finite and strictly increasing')
        off = _departure(x).max()
        if off > GRID_FIT:
            raise ValueError(f'{name} is not a uniform grid: {off:.1e} off a linear fit')
    shape = (lp.size, lt.size)
    if any(np.shape(table[c]) != shape for c in (*COLUMNS, 'phase')):
        raise ValueError('a column does not have the shape of the (P, T) grid')
    return lp, lt


def _grid(table):
    lp, lt = _axes(table)
    P, T = np.meshgrid(10.0**lp, 10.0**lt, indexing='ij')
    cols = [np.asarray(table[c], float) for c in COLUMNS]
    return np.log(10.0) * np.diff(lp).mean(), np.log(10.0) * np.diff(lt).mean(), P, T, *cols


def _parts(table, k):
    """Per derivative identity (D, E, Q, q) with R + 1 = D / Q: E bounds the error of D per
    unit of stored precision, q the relative error of Q per unit of stored precision."""
    hx, hy, P, T, rho, u, s, cp, cv, alpha, _ = _grid(table)
    h = {0: hx, 1: hy}
    one = np.ones_like(rho)
    code = _phase_code(table)
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_rho, a = np.log(rho), P * alpha / rho

        def d(f, col, ax, log=False):  # the stencil needs only ``col``; a stored log errs by 1
            seg = segments(table, ax, (col,), lambda v: ~np.isnan(v), code)
            ends = one if log else np.abs(f)
            return derivative(f, seg, ax, k, h[ax]), _stencil(ends, seg, ax, k, h[ax], 1.0)

        cpcv = (np.abs(cp) + np.abs(cv)) / np.abs(cp - cv) + 3
        return {
            'cp': (*d(s, 's', 1), cp, 1.0),
            'maxwell': (*d(s, 's', 0), -a, 2.0),
            'alpha': (*d(ln_rho, 'rho', 1, log=True), -T * alpha, 1.0),
            'dudt': (
                *d(u, 'u', 1),
                T * (cp - a),
                (np.abs(cp) + 2 * np.abs(a)) / np.abs(cp - a),
            ),
            'cp_cv': (
                *d(ln_rho, 'rho', 0, log=True),
                P * T * alpha**2 / ((cp - cv) * rho),
                cpcv,
            ),
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
        out = {n: D / Q - 1 for n, (D, _, Q, _) in parts.items()}
        out['nabla_ad'] = nab * rho * cp / (P * alpha) - 1
    return {n: out[n] for n in IDENTITIES}


def floors(table, eps, k=1):
    """Round-off bound of each residual at step ``k`` for columns stored to relative ``eps``.

    For R + 1 = D / Q the bound is (E / |Q| + |D / Q| q) eps + |D / Q| c, with E the error of
    the central difference from its two stored ends, q the relative error of Q, and c the
    relative error of Q from the stored P and T coordinates (e_P and e_T: ln 10 times the
    departure of log10 P and log10 T from their linear fit). nabla_ad, with no derivative, gets
    |R + 1| (4 eps + e_P) from its four stored factors and P.
    """
    _, _, P, _, rho, _, _, cp, _, alpha, nab = _grid(table)
    eP = np.log(10.0) * _departure(np.asarray(table['unique_log_p'], float))[:, None]
    eT = np.log(10.0) * _departure(np.asarray(table['unique_log_t'], float))[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        a = P * alpha / rho
        c = {
            'cp': 0.0,
            'maxwell': eP,
            'alpha': eT,
            'dudt': eT + eP * np.abs(a / (cp - a)),
            'cp_cv': eP + eT,
        }
        out = {
            n: (E / np.abs(Q) + np.abs(D / Q) * q) * eps + np.abs(D / Q) * c[n]
            for n, (D, E, Q, q) in _parts(table, k).items()
        }
        out['nabla_ad'] = np.abs(nab * rho * cp / (P * alpha)) * (4 * eps + eP)
    return {n: out[n] for n in IDENTITIES}


def _eps(eps):
    """``eps`` raised to ``EPS_FLOOR``; ValueError unless it is a real number in [0, 1)."""
    real = isinstance(eps, numbers.Real) and not isinstance(eps, (bool, np.timedelta64))
    try:
        x = float(eps) if real else np.nan
    except OverflowError:
        x = np.nan
    if not 0 <= x < 1:
        raise ValueError(f'eps must be a real number in [0, 1), got {eps!r}')
    return max(x, EPS_FLOOR)


# Column values that make every residual finite; a table of them marks where values are present.
_PRESENT = {
    'rho': 1.0,
    'u': 1.0,
    's': 1.0,
    'cp': 1.0,
    'cv': 0.5,
    'alpha': 1e-30,
    'nabla_ad': 1.0,
}


def _verdict(rx, bound, present):
    """The result of one identity: ``rx`` tested against ``bound`` where both are finite.

    A cell where every input is ``present`` (on the full stencil of a derivative identity) but
    ``rx`` or ``bound`` is not finite fails too; the identity is ``skipped`` when no cell is.
    """
    tested = np.isfinite(rx) & np.isfinite(bound)
    skipped = not present.any()
    fail = (tested & (np.abs(rx) > bound)) | (present & ~tested)
    return {'rx': rx, 'bound': bound, 'tested': tested, 'fail': fail, 'skipped': skipped}


def _richardson(r1, r2, r4, f1, f2, p1, p2, p4):
    """``_verdict`` of the Richardson residual from steps 1, 2, 4 with floors ``f1``, ``f2``;
    ``p1``, ``p2``, ``p4`` are the residuals of the present-values table at those steps."""
    rx = (4 * r1 - r2) / 3
    est = np.abs((4 * r2 - r4) / 3 - rx) / 15
    return _verdict(rx, (4 * f1 + f2) / 3 + FACTOR * est, np.isfinite(p1 + p2 + p4))


class NoCellTested(ValueError):
    """Every identity of a check is skipped."""


def _nonempty(out):
    """``out``, or NoCellTested if every identity is skipped."""
    if all(v['skipped'] for v in out.values()):
        raise NoCellTested(f'no cell tested for {", ".join(out)}')
    return out


def check(table, eps):
    """Test each cell against the Richardson bound of each identity.

    A derivative identity fails where |R_x(h)| > floor_x + FACTOR * est, with R_x(kh) =
    (4 R_k - R_2k) / 3, floor_x = (4 floor_1 + floor_2) / 3 and est = |R_x(2h) - R_x(h)| / 15,
    the measured h^4 term of R_x(h) at the cell. nabla_ad fails where |R| > floor. At 9 stored
    digits (eps 5e-9) the exact ideal gas and Murnaghan tables give no failing cell, and one
    planted value is found at the cells one and two steps from it.

    The floor uses eps_eff = max(eps, EPS_FLOOR): every table is checked at the 9-digit
    floor at best, so a single value off by about 1e-8 (relative) or less is out of reach on any
    table. Each identity reads only its own columns (``INPUTS``). A derivative identity is
    tested only where 4 cells on each side along its axis hold its differentiated column in one
    phase; NaN marks a missing value, and a stored 0 or inf is a value. Where all values an
    identity reads are present but its residual or bound is not finite, the cell fails; an
    identity is skipped only when no cell holds all values it reads, on the full stencil of a
    derivative identity (4 cells on each side in one phase). On a coarse grid a cell can lie
    outside the range where the Richardson estimate holds (R_x(2h) / R_x(h) far from 16), and
    an exact table can fail there: 1 cp cell of the Murnaghan table at 10 points per decade over
    P 1e5 to 10^11.5 Pa and T 300 to 7.5e3 K, where cp and cv are positive.

    Parameters
    ----------
    table : dict
        As for ``residuals``, on a grid uniform in log10 P and log10 T: each axis within
        ``GRID_FIT`` of a linear fit, as 9-digit coordinates are.
    eps : float
        Relative precision of the stored columns (0.5 x 10^(1 - n) for n significant digits).

    Returns
    -------
    dict
        Identity name -> dict with ``rx`` (the Richardson residual, defined also at untested
        cells), ``bound``, ``tested`` (both finite), ``fail`` (tested and above the bound, or
        with all values present and a residual or bound that is not finite) arrays, and
        ``skipped``, True when no cell holds all values the identity reads on its stencil.

    Raises
    ------
    ValueError
        If eps is not a real number in [0, 1), an axis has fewer than 2 points or is not
        finite, increasing and uniform, a column is off the shape of the grid, or every
        identity is skipped.
    """
    eps = _eps(eps)
    r = {k: residuals(table, k) for k in (1, 2, 4)}
    const = {**table}
    for col, value in _PRESENT.items():
        const[col] = np.where(np.isnan(np.asarray(table[col], float)), np.nan, value)
    p = {k: residuals(const, k) for k in (1, 2, 4)}
    f1, f2 = floors(table, eps, 1), floors(table, eps, 2)
    with np.errstate(invalid='ignore'):
        return _nonempty(
            {
                n: _verdict(r[1][n], f1[n], np.isfinite(p[1][n]))
                if n == 'nabla_ad'
                else _richardson(
                    r[1][n], r[2][n], r[4][n], f1[n], f2[n], p[1][n], p[2][n], p[4][n]
                )
                for n in IDENTITIES
            }
        )


def signs(table, eps):
    """Cells that break a sign condition beyond the stored precision ``eps``.

    Parameters
    ----------
    table : dict
        As for ``residuals``.
    eps : float
        Relative precision of the stored columns; raised to ``EPS_FLOOR`` at least.

    Returns
    -------
    dict
        Condition name -> dict with ``tested`` and ``fail`` boolean arrays. ``rho``, ``cp``
        and ``cv`` not positive; ``cp_ge_cv``: cv above cp by more than eps (|cp| + |cv|);
        ``kappa_T``: rho falls with P at fixed T by more than the rounding of its two
        neighbours (central difference over one cell on each side, one phase). A condition
        tests the cells where its values are not NaN; an infinite value breaks the rho, cp, cv
        and cp_ge_cv conditions, and kappa_T does not test it.

    Raises
    ------
    ValueError
        If eps is not a real number in [0, 1), or the axes or shapes fail as for ``check``.
    """
    eps = _eps(eps)
    _axes(table)
    rho, cp, cv = (np.asarray(table[c], float) for c in ('rho', 'cp', 'cv'))
    seg = segments(table, 0, ('rho',))
    rise = _stencil(rho, seg, 0, 1, 0.5, -1.0)  # rho[i+1] - rho[i-1]
    noise = _stencil(np.abs(rho), seg, 0, 1, 0.5, 1.0) * eps
    with np.errstate(invalid='ignore'):
        cond = {
            'rho': (rho, rho <= 0),
            'cp': (cp, cp <= 0),
            'cv': (cv, cv <= 0),
            'cp_ge_cv': (cp - cv, cv - cp > eps * (np.abs(cp) + np.abs(cv))),
            'kappa_T': (rise, rise < -noise),
        }
    return {
        n: {'tested': ~np.isnan(v), 'fail': ~np.isnan(v) & (f | np.isinf(v))}
        for n, (v, f) in cond.items()
    }


PT_IDENTITIES = ('cp', 'maxwell', 'alpha')
PT_QUANTITIES = ('density', 'heat_capacity', 'thermal_exp', 'entropy')
PT_FILL = 1e-15  # the value eos_export writes for a cell without data
PT_GRID_FIT = 1e-6  # allowed departure from a uniform or geometric axis, in units of its step


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


def _axis_error(x, eps):
    """Rounding error of each stored coordinate of an axis that is uniform, or uniform in log,
    within PT_GRID_FIT of its step plus its precision ``eps``: its departure from that fit plus
    4 ulp. None for any other axis."""
    i = np.arange(x.size)
    tiny = 4 * np.finfo(float).eps * np.abs(x)
    lin = np.polyval(np.polyfit(i, x, 1), i)
    if (np.abs(x - lin) <= PT_GRID_FIT * np.abs(np.diff(x)).mean() + eps * np.abs(x)).all():
        return np.abs(x - lin) + tiny
    if (x > 0).all():
        ly = np.log(x)
        fit = np.polyval(np.polyfit(i, ly, 1), i)
        if (np.abs(ly - fit) <= PT_GRID_FIT * np.abs(np.diff(ly)).mean() + eps).all():
            return np.abs(x - np.exp(fit)) + tiny
    return None


def _pt_axes(tables, eps):
    """The P and T axes of density, checked: 1-D, at least 2 points, finite, increasing, T
    positive, P not negative, and uniform or geometric."""
    P, T = np.asarray(tables['density']['P'], float), np.asarray(tables['density']['T'], float)
    for name, x in (('P', P), ('T', T)):
        if x.ndim != 1:
            raise ValueError(f'{name} must be one-dimensional')
        if x.size < 2:
            raise ValueError(f'{name} needs at least 2 points')
        if not (np.isfinite(x).all() and (np.diff(x) > 0).all()):
            raise ValueError(f'{name} must be finite and strictly increasing')
        if name == 'T' and x[0] <= 0:
            raise ValueError('T must be positive')
        if name == 'P' and x[0] < 0:
            raise ValueError('P must not be negative')
        if _axis_error(x, eps) is None:
            raise ValueError(f'{name} is neither uniform nor geometric')
    return P, T


def _pt_parts(tables, k, eps):
    """(D, E, Q, q, A) per identity of the Aragog P-T tables, with R + 1 = D / Q, E and q per
    unit of stored precision as in ``_parts``, and A the relative error of D from the rounding
    of its two coordinates."""
    P, T = _pt_axes(tables, eps)
    for q in PT_QUANTITIES:
        t = tables[q]
        if not (np.array_equal(t['P'], P) and np.array_equal(t['T'], T)):
            raise ValueError(f'{q} is not on the (P, T) grid of density')
        if np.shape(t['values']) != (P.size, T.size):
            raise ValueError(f'{q} values do not have the shape of the (P, T) grid')
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
            err = np.broadcast_to(
                _axis_error(x, eps).reshape((-1, 1) if ax == 0 else (1, -1)), f.shape
            )
            A = _coord_stencil(err, x, ax, k, 1.0)
            return _coord_stencil(f, x, ax, k, -1.0), _coord_stencil(ends, x, ax, k, 1.0), A

        cp_d, mx_d, al_d = d(s, T, 1), d(s, P, 0), d(ln_rho, T, 1, log=True)
        return {
            'cp': (*cp_d[:2], cp / TT, 2.0, cp_d[2]),
            'maxwell': (*mx_d[:2], -alpha / rho, 2.0, mx_d[2]),
            'alpha': (*al_d[:2], -alpha, 1.0, al_d[2]),
        }


def check_pt(tables, eps):
    """``check`` for the Aragog P-T tables that eos_export writes (no u, cv or nabla_ad).

    Parameters
    ----------
    tables : dict
        ``PT_QUANTITIES`` name -> dict with ``P`` (Pa) and ``T`` (K) grid vectors and ``values``
        of shape (n_P, n_T); the fill value ``PT_FILL`` counts as a missing cell.
    eps : float
        Relative precision of the stored values; raised to ``EPS_FLOOR`` at least.

    Returns
    -------
    dict
        As for ``check``, for the identities cp = T (ds/dT)_P, (ds/dP)_T = -alpha / rho and
        alpha = -(dln rho/dT)_P, each with ``fill``: quantity -> number of ``PT_FILL`` cells.

    Raises
    ------
    ValueError
        If eps is not a real number in [0, 1), an axis is not 1-D, has fewer than 2 points, is
        not finite and increasing, or departs from a uniform or geometric grid by more than
        ``PT_GRID_FIT`` of its step plus its stored precision, T is not positive or P is
        negative, the four tables are not on one grid or off its shape, or every identity is
        skipped.

    Notes
    -----
    The floor includes the rounding of the stored P and T coordinates: a central difference
    over x[i-k], x[i+k] gains |D| (e[i+k] + e[i-k]) / (x[i+k] - x[i-k]), with e the departure
    of each coordinate from the uniform or geometric fit of its axis plus 4 ulp.
    """
    eps = _eps(eps)
    parts = {k: _pt_parts(tables, k, eps) for k in (1, 2, 4)}
    const = {}
    for q, t in tables.items():
        v = np.asarray(t['values'], float)
        const[q] = {**t, 'values': np.where(np.isnan(v) | (v == PT_FILL), PT_FILL, 1.0)}
    present = {k: _pt_parts(const, k, eps) for k in (1, 2, 4)}
    fill = {
        q: int((np.asarray(tables[q]['values'], float) == PT_FILL).sum()) for q in PT_QUANTITIES
    }
    out = {}
    with np.errstate(divide='ignore', invalid='ignore'):
        for n in PT_IDENTITIES:
            p = [present[k][n][0] / present[k][n][2] for k in (1, 2, 4)]
            r = {k: parts[k][n][0] / parts[k][n][2] - 1 for k in (1, 2, 4)}
            fl = {
                k: (parts[k][n][1] / np.abs(parts[k][n][2]) + np.abs(r[k] + 1) * parts[k][n][3])
                * eps
                + np.abs(r[k] + 1) * parts[k][n][4]
                for k in (1, 2)
            }
            out[n] = {**_richardson(r[1], r[2], r[4], fl[1], fl[2], *p), 'fill': dict(fill)}
    return _nonempty(out)
