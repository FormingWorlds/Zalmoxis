"""diffrax-based solver for the coupled structure ODEs.

Replaces ``scipy.integrate.solve_ivp(method='RK45')`` used inside
``zalmoxis.structure_model.solve_structure`` with
``diffrax.diffeqsolve(Tsit5())`` so the full ODE integration (RHS +
step management + error control) runs inside one JIT-compiled kernel.

Tsit5 is a 5th-order explicit Runge-Kutta method, comparable to
scipy's RK45 (Dormand-Prince 5(4)). Step trajectories differ at the
solver-tolerance level (rtol/atol) but integrated results agree to
within those tolerances.

Pressure-zero terminal event: numpy's ``solve_structure`` stops
integration via a scipy event when P crosses zero. This module uses
``diffrax.Event`` with an ``optimistix.Newton`` root finder to
localize the crossing, matching numpy's physics. After the event
fires, saveat points beyond the crossing are returned as ``inf`` by
diffrax; the wrapper (``jax_eos/wrapper.py``) detects these and pads
pressure to 0 and mass/gravity to their last-valid values.

As a defensive belt-and-suspenders, ``coupled_odes_jax`` still zeroes
its RHS when P<=0 so the state is bounded even if the integrator
briefly overshoots into negative P between step and event-localize.
The event itself provides the physics-faithful termination.
"""
from __future__ import annotations

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
    import optimistix as optx

    def _ode_rhs(t, y, args):
        return coupled_odes_jax(t, y, **args)

    def _pressure_cond(t, y, args, **kwargs):
        # Event fires when pressure crosses zero. direction=False tells
        # diffrax to trigger only on the downcrossing (the physical
        # outer-surface case). A tiny positive offset keeps the root
        # finder away from exact y[2]=0 where the EOS tables can NaN.
        return y[2]

    term = diffrax.ODETerm(_ode_rhs)
    solver = diffrax.Tsit5()
    # Newton is the natural choice for a scalar condition on the
    # solver's own interpolation; tolerances 1e-6 match the structure
    # ODE's default rtol/atol and are finer than scipy's internal
    # event-bracket tolerance (1e-8 of the step; scipy uses brentq).
    root_finder = optx.Newton(rtol=1e-6, atol=1e-6)
    event = diffrax.Event(
        cond_fn=_pressure_cond,
        root_finder=root_finder,
        direction=False,
    )

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
            event=event,
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
