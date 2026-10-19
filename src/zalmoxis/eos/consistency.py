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


def derivative(f, seg, axis, k, h):
    """d f / d(axis) with step ``k h``, NaN where the stencil leaves one segment."""
    n = f.shape[axis]
    out = np.full(f.shape, np.nan)
    if n <= 2 * k:
        return out

    def cut(a, b):
        return tuple(slice(a, b) if ax == axis else slice(None) for ax in range(f.ndim))

    mid, lo, hi = seg[cut(k, n - k)], seg[cut(0, n - 2 * k)], seg[cut(2 * k, n)]
    ok = (mid >= 0) & (lo == mid) & (hi == mid)
    out[cut(k, n - k)] = np.where(
        ok, (f[cut(2 * k, n)] - f[cut(0, n - 2 * k)]) / (2 * k * h), np.nan
    )
    return out


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
    lp, lt = np.asarray(table['unique_log_p'], float), np.asarray(table['unique_log_t'], float)
    hx, hy = np.log(10.0) * np.diff(lp).mean(), np.log(10.0) * np.diff(lt).mean()
    P, T = np.meshgrid(10.0**lp, 10.0**lt, indexing='ij')
    rho, u, s, cp, cv, alpha, nab = (
        np.asarray(table[c], float) for c in ('rho', 'u', 's', 'cp', 'cv', 'alpha', 'nabla_ad')
    )
    sx, sy = segments(table, 0), segments(table, 1)
    with np.errstate(divide='ignore', invalid='ignore'):
        ln_rho = np.log(rho)
        kappa = derivative(ln_rho, sx, 0, k, hx) / P
        return {
            'cp': derivative(s, sy, 1, k, hy) / cp - 1,
            'maxwell': derivative(s, sx, 0, k, hx) / (-P * alpha / rho) - 1,
            'alpha': -derivative(ln_rho, sy, 1, k, hy) / (T * alpha) - 1,
            'dudt': derivative(u, sy, 1, k, hy) / T / (cp - P * alpha / rho) - 1,
            'nabla_ad': nab * rho * cp / (P * alpha) - 1,
            'cp_cv': (cp - cv) * rho * kappa / (T * alpha**2) - 1,
        }
