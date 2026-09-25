"""Main function to solve the coupled ODEs for the structure model.

!!! Imports
    - [`constants`](zalmoxis.constants.md): G
    - [`mixing`](zalmoxis.mixing.md): LayerMixture, calculate_mixed_density, any_component_is_tdep

"""

# This file contains the main function that solves the coupled ODEs for the structure model.
from __future__ import annotations

import logging

import numpy as np
from scipy.integrate import solve_ivp

from .constants import CONDENSED_RHO_MIN_DEFAULT, CONDENSED_RHO_SCALE_DEFAULT, G
from .mixing import BINODAL_T_SCALE_DEFAULT, any_component_is_tdep, calculate_mixed_density

# Set up logging
logger = logging.getLogger(__name__)

# A stop in the integration at or below this fraction of the central pressure
# is the planet's surface; a stop at a higher pressure is a failed solve.
SURFACE_STOP_P_FRACTION = 1e-6


def pad_after_stop(radii, mass, gravity, pressure, y_stop, p_center, p_surface=0.0):
    """Extend profiles cut short by a stop in the integration to the full grid.

    The integration stops between ``radii[n - 1]`` and ``radii[n]``, with ``n``
    the length of the profiles. A stop at a pressure of at most
    ``SURFACE_STOP_P_FRACTION * p_center``, or below the target surface
    pressure ``p_surface``, is padded as the surface: the remaining nodes take
    mass and gravity at the stop and zero pressure. Below the target the
    pressure has fallen short before the outer radius, so the zero surface
    pressure gives the pressure solve the right sign (central pressure too
    low). A stop at a higher pressure, or with a non-finite state, is a failed
    solve: the remaining nodes are NaN, which the callers treat as a failed
    evaluation, and a WARNING names the stop.

    Parameters
    ----------
    radii : numpy.ndarray
        Full radial grid [m].
    mass, gravity, pressure : numpy.ndarray
        Profiles up to the last grid node before the stop.
    y_stop : array_like
        State [m, g, P] where the integration stopped.
    p_center : float
        Central pressure of the solve [Pa].
    p_surface : float, optional
        Target surface pressure [Pa].

    Returns
    -------
    tuple of numpy.ndarray
        Mass, gravity and pressure on the full grid.
    """
    n = len(mass)
    m_stop, g_stop, p_stop = (float(v) for v in y_stop)
    if np.all(np.isfinite(y_stop)) and p_stop <= max(
        SURFACE_STOP_P_FRACTION * p_center, p_surface
    ):
        fill = (m_stop, g_stop, 0.0)
    else:
        where = f'between r = {radii[n - 1]:.6e} and {radii[n]:.6e} m' if n else 'at r = 0'
        logger.warning(
            'Structure integration stopped at P = %.3e Pa (P_c = %.3e Pa), %s; '
            'treating the solve as failed.',
            p_stop,
            p_center,
            where,
        )
        fill = (np.nan, np.nan, np.nan)
    return tuple(
        np.concatenate([a, np.full(len(radii) - n, f)])
        for a, f in zip((mass, gravity, pressure), fill)
    )


def get_layer_mixture(mass, cmb_mass, core_mantle_mass, layer_mixtures):
    """Determine the per-layer mixture based on enclosed mass (purely geometric).

    Parameters
    ----------
    mass : float
        Enclosed mass at the current radial shell [kg].
    cmb_mass : float
        Core-mantle boundary mass [kg].
    core_mantle_mass : float
        Core + mantle mass [kg].
    layer_mixtures : dict
        Per-layer LayerMixture objects, e.g.
        ``{"core": LayerMixture(...), "mantle": LayerMixture(...)}``.

    Returns
    -------
    LayerMixture
        Mixture for this shell.
    """
    # Use a small relative tolerance for boundary comparisons to avoid
    # float-precision misassignment when a mesh node lands exactly at
    # the CMB or core-mantle boundary.
    rtol = 1e-12
    if mass < cmb_mass * (1.0 - rtol):
        return layer_mixtures['core']
    elif 'ice_layer' in layer_mixtures and mass >= core_mantle_mass * (1.0 - rtol):
        return layer_mixtures['ice_layer']
    else:
        return layer_mixtures['mantle']


# Define the coupled ODEs for the structure model
def coupled_odes(
    radius,
    y,
    cmb_mass,
    core_mantle_mass,
    layer_mixtures,
    interpolation_cache,
    material_dictionaries,
    temperature,
    solidus_func,
    liquidus_func,
    mushy_zone_factors=None,
    condensed_rho_min=CONDENSED_RHO_MIN_DEFAULT,
    condensed_rho_scale=CONDENSED_RHO_SCALE_DEFAULT,
    binodal_T_scale=BINODAL_T_SCALE_DEFAULT,
    volatile_profile=None,
):
    """Calculate derivatives of mass, gravity, and pressure w.r.t. radius.

    Parameters
    ----------
    radius : float
        Current radius [m].
    y : array-like
        State vector [mass, gravity, pressure].
    cmb_mass : float
        Core-mantle boundary mass [kg].
    core_mantle_mass : float
        Core + mantle mass [kg].
    layer_mixtures : dict
        Per-layer LayerMixture objects.
    interpolation_cache : dict
        Cache for interpolation functions.
    material_dictionaries : dict
        EOS registry dict keyed by EOS identifier string.
    temperature : float
        Temperature at current radius [K].
    solidus_func : callable or None
        Solidus melting curve interpolation function.
    liquidus_func : callable or None
        Liquidus melting curve interpolation function.
    mushy_zone_factors : dict or float or None
        Per-EOS mushy zone factors. Dict keyed by EOS name, a single
        float (applied to all), or None (default 1.0 for all).
    condensed_rho_min : float
        Sigmoid center for phase-aware suppression (kg/m^3).
    condensed_rho_scale : float
        Sigmoid width for phase-aware suppression (kg/m^3).
    binodal_T_scale : float
        Binodal sigmoid width in K for H2 miscibility suppression.
    volatile_profile : VolatileProfile or None
        Phi-aware mantle profile from the partition-law hook. Applied
        only to the mantle layer; the core and ice layers ignore it.

    Returns
    -------
    list
        Derivatives [dM/dr, dg/dr, dP/dr].
    """
    # Unpack the state vector
    mass, gravity, pressure = y

    # Determine per-layer mixture for the current enclosed mass
    mixture = get_layer_mixture(mass, cmb_mass, core_mantle_mass, layer_mixtures)

    # Return zero derivatives for non-physical pressure.  When the RHS
    # returns zeros, the ODE state freezes (mass, gravity, pressure stop
    # changing).  The terminal event (_pressure_zero, direction=-1) then
    # fires when pressure crosses zero, stopping the integration.
    # Note: zero derivatives do NOT cause RK45 to reject the step; the
    # solver accepts them and advances with frozen state until the
    # terminal event triggers.
    if pressure <= 0 or np.isnan(pressure):
        logger.debug(f'Nonphysical pressure encountered: P={pressure} Pa at radius={radius} m')
        return [0.0, 0.0, 0.0]

    # Apply the phi-aware profile only inside the mantle: the core and
    # ice layers do not partition volatiles. get_layer_mixture returns the
    # 'mantle' entry for mantle shells, so gate on that same key.
    profile_for_shell = (
        volatile_profile
        if (volatile_profile is not None and mixture is layer_mixtures.get('mantle'))
        else None
    )

    # Calculate density at the current radius, using pressure from y
    current_density = calculate_mixed_density(
        pressure,
        temperature,
        mixture,
        material_dictionaries,
        solidus_func,
        liquidus_func,
        interpolation_cache,
        mushy_zone_factors,
        condensed_rho_min,
        condensed_rho_scale,
        binodal_T_scale,
        volatile_profile=profile_for_shell,
    )

    # An EOS failure (None or non-finite density) at P > 0 stops the integration:
    # NaN derivatives make the step fail, and the stop is a failed solve.
    if current_density is None or not np.isfinite(current_density):
        return [np.nan, np.nan, np.nan]

    # Define the ODEs for mass, gravity and pressure
    dMdr = 4 * np.pi * radius**2 * current_density
    # At r=0 the 2g/r term is singular; use the analytic limit dg/dr = 4πGρ/3
    # (L'Hopital on g(r) = GM(r)/r^2 with M ~ r^3 near the center).
    dgdr = (
        4 * np.pi * G * current_density - 2 * gravity / radius
        if radius > 0
        else (4.0 / 3.0) * np.pi * G * current_density
    )
    dPdr = -current_density * gravity

    # Return the derivatives
    return [dMdr, dgdr, dPdr]


def solve_structure(
    layer_mixtures,
    cmb_mass,
    core_mantle_mass,
    radii,
    adaptive_radial_fraction,
    relative_tolerance,
    absolute_tolerance,
    maximum_step,
    material_dictionaries,
    interpolation_cache,
    y0,
    solidus_func,
    liquidus_func,
    temperature_function=None,
    mushy_zone_factors=None,
    condensed_rho_min=CONDENSED_RHO_MIN_DEFAULT,
    condensed_rho_scale=CONDENSED_RHO_SCALE_DEFAULT,
    binodal_T_scale=BINODAL_T_SCALE_DEFAULT,
    use_jax=False,
    temperature_arrays=None,
    volatile_profile=None,
    surface_pressure=0.0,
):
    """Solve the coupled ODEs for the planetary structure model.

    Handles the special case for temperature-dependent EOS where the radial
    grid is split into two parts for better handling of large step sizes
    towards the surface.

    Parameters
    ----------
    layer_mixtures : dict
        Per-layer LayerMixture objects, e.g.
        ``{"core": LayerMixture(...), "mantle": LayerMixture(...)}``.
    cmb_mass : float
        Mass at the core-mantle boundary [kg].
    core_mantle_mass : float
        Core + mantle mass [kg].
    radii : numpy.ndarray
        Radial grid points [m].
    adaptive_radial_fraction : float
        Fraction of radial domain for adaptive-to-fixed step transition.
    relative_tolerance : float
        Relative tolerance for solve_ivp.
    absolute_tolerance : float
        Absolute tolerance for solve_ivp.
    maximum_step : float
        Maximum integration step size [m].
    material_dictionaries : dict
        EOS registry dict keyed by EOS identifier string.
    interpolation_cache : dict
        Cache for interpolation functions.
    y0 : array-like
        Initial conditions [mass, gravity, pressure] at center.
    solidus_func : callable or None
        Solidus melting curve interpolation function.
    liquidus_func : callable or None
        Liquidus melting curve interpolation function.
    temperature_function : callable or None
        Function returning temperature [K]. Signature: ``f(r, P) -> T``
        where ``r`` is radius in m and ``P`` is pressure in Pa. For
        non-adiabatic modes the pressure argument is ignored.
    temperature_arrays : tuple[ndarray, ndarray] or None
        Optional ``(r_arr, T_arr)`` for an explicit r-indexed T profile.
        Only consumed by the JAX path (``use_jax=True``); the numpy path
        still uses ``temperature_function``. See ``jax_eos.wrapper``
        docstring for when to prefer this over the callable form.
    mushy_zone_factors : dict or float or None
        Per-EOS mushy zone factors. Dict keyed by EOS name, a single
        float (applied to all), or None (default 1.0 for all).
    condensed_rho_min : float
        Sigmoid center for phase-aware suppression (kg/m^3).
    condensed_rho_scale : float
        Sigmoid width for phase-aware suppression (kg/m^3).
    binodal_T_scale : float
        Binodal sigmoid width in K for H2 miscibility suppression.
    surface_pressure : float, optional
        Target surface pressure [Pa], passed to ``pad_after_stop``.

    Returns
    -------
    tuple
        (mass_enclosed, gravity, pressure) arrays at each radial grid point.
        Past a stop in the integration they are padded by ``pad_after_stop``.
    """
    # JAX fast path — dispatch to the diffrax-based implementation when
    # requested. Falls back to numpy path on any ValueError (unsupported
    # config: 3-layer ice, unsupported mixtures, a mantle in neither the
    # unified nor the 2-phase PALEOS representation, unsupported
    # volatile profiles, etc.). Logged at warning; callers observe the
    # same return contract either way. A volatile_profile is supported
    # when it carries exactly one active paleos_unified volatile (the
    # phi-blended wet mantle); profiles outside that envelope (H2
    # binodal, miscibility, multi-volatile) raise ValueError inside the
    # wrapper and land on numpy.
    if use_jax:
        try:
            from .jax_eos.wrapper import solve_structure_via_jax

            return solve_structure_via_jax(
                layer_mixtures=layer_mixtures,
                cmb_mass=cmb_mass,
                core_mantle_mass=core_mantle_mass,
                radii=radii,
                adaptive_radial_fraction=adaptive_radial_fraction,
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                maximum_step=maximum_step,
                material_dictionaries=material_dictionaries,
                interpolation_cache=interpolation_cache,
                y0=y0,
                solidus_func=solidus_func,
                liquidus_func=liquidus_func,
                temperature_function=temperature_function,
                temperature_arrays=temperature_arrays,
                mushy_zone_factors=mushy_zone_factors,
                condensed_rho_min=condensed_rho_min,
                condensed_rho_scale=condensed_rho_scale,
                binodal_T_scale=binodal_T_scale,
                volatile_profile=volatile_profile,
                surface_pressure=surface_pressure,
            )
        except ValueError as exc:
            logger.warning(
                'JAX solve_structure fell back to numpy path: %s',
                exc,
            )

    uses_Tdep = any_component_is_tdep(layer_mixtures)

    # Terminal event: stop integration when pressure crosses zero.
    # Without this, the ODE solver grinds with tiny step sizes in the
    # zero-derivative region returned by coupled_odes() for P <= 0.
    def _pressure_zero(r, y, *args):
        return y[2]  # pressure component

    _pressure_zero.terminal = True
    _pressure_zero.direction = -1  # trigger on positive → negative crossing

    def _ode_rhs(r, y):
        return coupled_odes(
            r,
            y,
            cmb_mass,
            core_mantle_mass,
            layer_mixtures,
            interpolation_cache,
            material_dictionaries,
            temperature_function(r, y[2]) if temperature_function else 300,
            solidus_func,
            liquidus_func,
            mushy_zone_factors,
            condensed_rho_min,
            condensed_rho_scale,
            binodal_T_scale,
            volatile_profile=volatile_profile,
        )

    # scipy takes a NaN first step, and then never ends, if the RHS fails at the centre.
    if not np.all(np.isfinite(_ode_rhs(radii[0], y0))):
        empty = np.empty(0)
        return pad_after_stop(radii, empty, empty, empty, y0, y0[2], surface_pressure)

    if uses_Tdep:
        # Split the radial grid into two parts for better handling of large step sizes
        radial_split_index = max(
            1, min(len(radii) - 1, int(adaptive_radial_fraction * len(radii)))
        )

        # Solve the ODEs in two parts, first part with default max_step (adaptive)
        sol1 = solve_ivp(
            _ode_rhs,
            (radii[0], radii[radial_split_index - 1]),
            y0,
            t_eval=radii[:radial_split_index],
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            method='RK45',
            events=_pressure_zero,
        )

        sol_end, max_step_end = sol1, np.inf
        # If sol1 stopped (pressure-zero event or step-size failure), skip sol2
        if sol1.status != 0:
            mass_enclosed, gravity, pressure = np.reshape(sol1.y, (3, -1))
        else:
            # Second part with user-defined max_step
            sol2 = solve_ivp(
                _ode_rhs,
                (radii[radial_split_index - 1], radii[-1]),
                sol1.y[:, -1],
                t_eval=radii[radial_split_index - 1 :],
                rtol=relative_tolerance,
                atol=absolute_tolerance,
                max_step=maximum_step,
                method='RK45',
                events=_pressure_zero,
            )
            sol_end, max_step_end = sol2, maximum_step

            # Concatenate the two solutions
            y2 = np.reshape(sol2.y, (3, -1))  # empty if the first step fails
            mass_enclosed, gravity, pressure = np.concatenate(
                [sol1.y[:, :-1], y2 if y2.size else sol1.y[:, -1:]], axis=1
            )
    else:
        # Single integration with fixed temperature (300 K for Seager+2007)
        sol = solve_ivp(
            _ode_rhs,
            (radii[0], radii[-1]),
            y0,
            t_eval=radii,
            rtol=relative_tolerance,
            atol=absolute_tolerance,
            method='RK45',
            events=_pressure_zero,
        )
        sol_end, max_step_end = sol, np.inf

        # scipy returns empty lists when the first step fails before radii[1].
        mass_enclosed, gravity, pressure = np.reshape(sol.y, (3, -1))

    # Pad to full length if the integration stopped before the outermost radial
    # grid point (pressure-zero event, or a step-size failure).
    n = len(mass_enclosed)
    if n < len(radii):
        if sol_end.status == 1:
            y_stop = sol_end.y_events[0][-1]
        else:
            y_stop = [mass_enclosed[-1], gravity[-1], pressure[-1]] if n else y0
            # The failure lies between radii[n - 1] and radii[n]; re-integrate only that shell
            # to find where it stops (scipy never ends a step from a non-finite RHS).
            if n and np.all(np.isfinite(_ode_rhs(radii[n - 1], y_stop))):
                tail = solve_ivp(
                    _ode_rhs,
                    (radii[n - 1], radii[n]),
                    y_stop,
                    rtol=relative_tolerance,
                    atol=absolute_tolerance,
                    max_step=max_step_end,
                    method='RK45',
                    events=_pressure_zero,
                )
                y_stop = tail.y[:, -1]  # at a terminal event this is the event state
                if tail.status == 0 and n == len(radii) - 1:
                    # The re-integration passed the last shell: its end state completes the profile.
                    return tuple(
                        np.append(a, v)
                        for a, v in zip((mass_enclosed, gravity, pressure), y_stop)
                    )
        mass_enclosed, gravity, pressure = pad_after_stop(
            radii, mass_enclosed, gravity, pressure, y_stop, y0[2], surface_pressure
        )

    return mass_enclosed, gravity, pressure
