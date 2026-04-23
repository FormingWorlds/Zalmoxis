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
        coupled_odes_jax needs. Passed through unchanged.

    Returns
    -------
    ys : array of shape (n_layers, 3)
        State [M, g, P] at each radii.
    """
    # Local imports to keep the numpy-path cost low at module load.
    import diffrax

    def _ode_rhs(t, y, args):
        # args is the pytree dict of cache data; unpack via kwargs
        return coupled_odes_jax(t, y, **args)

    term = diffrax.ODETerm(_ode_rhs)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    saveat = diffrax.SaveAt(ts=radii)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=float(radii[0]),
        t1=float(radii[-1]),
        dt0=float(radii[1] - radii[0]) if len(radii) > 1 else 1.0,
        y0=jnp.asarray(y0),
        saveat=saveat,
        stepsize_controller=controller,
        args=rhs_kwargs,
        max_steps=200000,
        throw=False,  # match scipy's behaviour: return best-effort on failure
    )
    return sol.ys
