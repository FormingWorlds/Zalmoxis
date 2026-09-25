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
fires, or after a failed step, saveat points beyond the stop are
returned as ``inf`` by diffrax; the wrapper (``jax_eos/wrapper.py``)
pads them with ``structure_model.pad_after_stop`` from the state where
the solve stopped: mass/gravity at the stop and zero pressure at the
surface, NaN for a stop deep inside.

``coupled_odes_jax`` zeroes its RHS only for a non-finite density, not
for P <= 0, so that the event sees the pressure-zero downcrossing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .rhs import coupled_odes_jax


def _build_diffeqsolve_jit(
    T_axis_is_radius: bool, has_volatile: bool = False, mantle_is_unified: bool = False
):
    """Build a jitted diffeqsolve closure for one axis convention.

    Must be top-level so jax.jit can cache the compiled kernel across
    wrapper calls. Closure over diffrax so the numpy-path import cost
    stays zero. ``T_axis_is_radius``, ``has_volatile``, and
    ``mantle_is_unified`` are closed over (not traced args) so the
    corresponding branches of ``coupled_odes_jax`` are specialised at
    compile time. A separate compiled variant is cached per flag triple
    in ``_SOLVE_CACHE`` below.
    """
    import diffrax
    import optimistix as optx

    def _ode_rhs(t, y, args):
        # The static flags are closed over (not in args) so they stay
        # static for JIT; this keeps coupled_odes_jax single-variant
        # per _solve closure.
        return coupled_odes_jax(
            t,
            y,
            T_axis_is_radius=T_axis_is_radius,
            has_volatile=has_volatile,
            mantle_is_unified=mantle_is_unified,
            **args,
        )

    def _pressure_cond(t, y, args, **kwargs):
        # Event fires when pressure crosses zero. direction=False tells
        # diffrax to trigger only on the downcrossing (the physical
        # outer-surface case).
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
        controller = diffrax.PIDController(
            rtol=rtol, atol=atol, dtmin=1e-12 * (radii[-1] - radii[0]), force_dtmin=False
        )
        saveat = diffrax.SaveAt(subs=[diffrax.SubSaveAt(ts=radii), diffrax.SubSaveAt(t1=True)])
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
        return sol.ys[0], sol.ys[1][0]

    return _solve


# Separate compiled closure per (temperature-axis, wet-mantle,
# mantle-representation) triple.
_SOLVE_CACHE: dict[tuple[bool, bool, bool], object] = {}


def _get_solve(
    T_axis_is_radius: bool, has_volatile: bool = False, mantle_is_unified: bool = False
):
    key = (T_axis_is_radius, has_volatile, mantle_is_unified)
    if key not in _SOLVE_CACHE:
        _SOLVE_CACHE[key] = _build_diffeqsolve_jit(
            T_axis_is_radius, has_volatile, mantle_is_unified
        )
    return _SOLVE_CACHE[key]


def solve_structure_jax(
    radii,
    y0,
    rtol=1e-5,
    atol=1e-6,
    T_axis_is_radius: bool = False,
    has_volatile: bool = False,
    mantle_is_unified: bool = False,
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
    T_axis_is_radius : bool
        Selects the temperature-axis convention consumed by
        ``coupled_odes_jax``. False (default): ``T_axis_grid`` is
        ``log10(P)``. True: ``T_axis_grid`` is a radial grid. Routes to
        a separately-compiled diffeqsolve variant.
    **rhs_kwargs :
        All the cache + adiabat + Stixrude14 parameters that
        coupled_odes_jax needs. Passed through as a dict pytree.

    Returns
    -------
    ys : array of shape (n_layers, 3)
        State [M, g, P] at each radii.
    y_end : array of shape (3,)
        State where the integration stopped: at the pressure-zero event,
        at ``radii[-1]``, or at the last accepted step if the solve failed.

    Notes
    -----
    A step size below ``1e-12`` of the grid span (e.g. after NaN derivatives
    from a failed EOS lookup) ends the solve (``force_dtmin=False``,
    ``throw=False``): ``y_end`` is the last accepted state and the later save
    points are not finite.
    """
    solve = _get_solve(T_axis_is_radius, has_volatile, mantle_is_unified)
    return solve(
        jnp.asarray(radii, dtype=jnp.float64),
        jnp.asarray(y0, dtype=jnp.float64),
        jnp.asarray(rtol, dtype=jnp.float64),
        jnp.asarray(atol, dtype=jnp.float64),
        rhs_kwargs,
    )
