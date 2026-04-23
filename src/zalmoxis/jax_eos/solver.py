"""diffrax-based solver for the coupled structure ODEs.

Replaces ``scipy.integrate.solve_ivp(method='RK45')`` used inside
``zalmoxis.structure_model.solve_structure`` with
``diffrax.diffeqsolve(Tsit5())`` so the full ODE integration (RHS +
step management + error control) runs inside one JIT-compiled kernel.

Tsit5 is a 5th-order explicit Runge-Kutta method, comparable to
scipy's RK45 (Dormand-Prince 5(4)). Step trajectories differ at the
solver-tolerance level (rtol/atol) but integrated results agree to
within those tolerances.

The numpy ``solve_structure`` has a pressure-zero terminal event that
stops integration when P crosses zero. In the JAX path we handle this
by zeroing the derivative inside ``coupled_odes_jax`` when P <= 0, so
the state freezes and remaining t_eval points carry the frozen value.
A cleaner port using ``diffrax.Event`` can be added later if needed.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from .rhs import coupled_odes_jax


def _build_diffeqsolve_jit():
    """Build a jitted diffeqsolve closure, imported lazily.

    Must be top-level so jax.jit can cache the compiled kernel across
    wrapper calls. Closure over diffrax so the numpy-path import cost
    stays zero.
    """
    import diffrax

    def _ode_rhs(t, y, args):
        return coupled_odes_jax(t, y, **args)

    term = diffrax.ODETerm(_ode_rhs)
    solver = diffrax.Tsit5()

    @jax.jit
    def _solve(radii, y0, rtol, atol, rhs_args):
        controller = diffrax.PIDController(rtol=rtol, atol=atol)
        saveat = diffrax.SaveAt(ts=radii)
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=radii[0],
            t1=radii[-1],
            dt0=radii[1] - radii[0],
            y0=y0,
            saveat=saveat,
            stepsize_controller=controller,
            args=rhs_args,
            max_steps=200000,
            throw=False,
        )
        return sol.ys

    return _solve


_SOLVE_CACHE = None


def _get_solve():
    global _SOLVE_CACHE
    if _SOLVE_CACHE is None:
        _SOLVE_CACHE = _build_diffeqsolve_jit()
    return _SOLVE_CACHE


def solve_structure_jax(
    radii,
    y0,
    rtol=1e-5,
    atol=1e-6,
    **rhs_kwargs,
):
    """Integrate the structure ODE from radii[0] to radii[-1].

    Parameters
    ----------
    radii : array of shape (n_layers,)
        Radial grid points to save at. Monotone increasing.
    y0 : array of shape (3,)
        Initial conditions [mass, gravity, pressure] at radii[0].
    rtol, atol : float
        diffrax PIDController tolerances. Match scipy RK45 defaults by
        default (1e-5 / 1e-6).
    **rhs_kwargs :
        All the cache + adiabat + Stixrude14 parameters that
        coupled_odes_jax needs. Passed through as a dict pytree.

    Returns
    -------
    ys : array of shape (n_layers, 3)
        State [M, g, P] at each radii.
    """
    solve = _get_solve()
    return solve(
        jnp.asarray(radii, dtype=jnp.float64),
        jnp.asarray(y0, dtype=jnp.float64),
        jnp.asarray(rtol, dtype=jnp.float64),
        jnp.asarray(atol, dtype=jnp.float64),
        rhs_kwargs,
    )
