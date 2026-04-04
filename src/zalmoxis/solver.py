"""Core solver loop for the Zalmoxis interior structure model.

Contains the main() function that implements the three nested iterative
procedures: mass-radius convergence (outer), Picard density iteration
(middle), and Brent root-finding on central pressure with RK45 ODE
integration (inner).

Numerical solver parameters (tolerances, iteration limits, step sizes)
are set internally via `_default_solver_params()` with mass-adaptive
scaling. Callers do not need to provide them. If a first solve attempt
fails to converge, `main()` automatically retries with tighter settings.
"""

from __future__ import annotations

import copy
import logging
import time

import numpy as np
from scipy.optimize import brentq

from .config import (
    PALEOS_MAX_MASS_EARTH,
    PALEOS_UNIFIED_MAX_MASS_EARTH,
    RTPRESS100TPA_MAX_MASS_EARTH,
    WOLFBOWER2018_MAX_MASS_EARTH,
)
from .constants import (
    CONDENSED_RHO_MIN_DEFAULT,
    CONDENSED_RHO_SCALE_DEFAULT,
    TDEP_EOS_NAMES,
    earth_center_pressure,
    earth_mass,
    earth_radius,
)
from .eos import (
    calculate_temperature_profile,
    compute_adiabatic_temperature,
    create_pressure_density_files,
)
from .mixing import (
    BINODAL_T_SCALE_DEFAULT,
    any_component_is_tdep,
    calculate_mixed_density_batch,
    parse_all_layer_mixtures,
)
from .structure_model import solve_structure

logger = logging.getLogger(__name__)

# Module-level EOS interpolation cache. Persists across multiple main() calls
# within the same Python process (e.g., PROTEUS coupling loop), avoiding
# repeated parsing of large PALEOS text tables (~3s per table per reload).
_interpolation_cache = {}


def _default_solver_params(planet_mass):
    """Compute default numerical solver parameters with mass-adaptive scaling.

    Parameters
    ----------
    planet_mass : float
        Planet mass in kg.

    Returns
    -------
    dict
        Numerical solver parameters. Keys match what `_solve()` expects.
    """
    mass_in_earth = max(planet_mass, 0.01 * earth_mass) / earth_mass

    # Estimated planet radius from mass-radius scaling (Seager+2007)
    r_est = earth_radius * mass_in_earth**0.282

    # Estimated central pressure from scaling law
    p_center_est = earth_center_pressure * mass_in_earth**0.87

    # Ensure ~5 steps in the outer 2% of the radial domain
    maximum_step = r_est * 0.004

    # Brent bracket cap: 10x the estimated central pressure.
    # Only active for WolfBower2018 in _solve(), but set for all cases.
    max_center_pressure_guess = 10.0 * p_center_est

    return {
        'max_iterations_outer': 50,
        'tolerance_outer': 1e-3,
        'max_iterations_inner': 50,
        'tolerance_inner': 1e-4,
        'relative_tolerance': 1e-5,
        'absolute_tolerance': 1e-6,
        'maximum_step': maximum_step,
        'adaptive_radial_fraction': 0.98,
        'max_center_pressure_guess': max_center_pressure_guess,
        'pressure_tolerance': 1e9,
        'max_iterations_pressure': 100,
    }


def _tighten_solver_params(params):
    """Return a copy of solver params with tighter settings for retry.

    Parameters
    ----------
    params : dict
        Original solver parameters.

    Returns
    -------
    dict
        Tightened parameters: doubled iteration counts, 10x tighter
        tolerances, halved maximum step.
    """
    tightened = dict(params)
    tightened['max_iterations_outer'] = params['max_iterations_outer'] * 2
    tightened['max_iterations_inner'] = params['max_iterations_inner'] * 2
    tightened['max_iterations_pressure'] = params['max_iterations_pressure'] * 2
    tightened['tolerance_outer'] = params['tolerance_outer'] * 0.1
    tightened['tolerance_inner'] = params['tolerance_inner'] * 0.1
    tightened['relative_tolerance'] = params['relative_tolerance'] * 0.1
    tightened['absolute_tolerance'] = params['absolute_tolerance'] * 0.1
    tightened['maximum_step'] = params['maximum_step'] * 0.5
    tightened['pressure_tolerance'] = params['pressure_tolerance'] * 0.1
    return tightened


def main(
    config_params,
    material_dictionaries,
    melting_curves_functions,
    input_dir,
    layer_mixtures=None,
    volatile_profile=None,
    temperature_function=None,
):
    """Run the exoplanet internal structure model with automatic retry.

    Calls `_solve()` with mass-adaptive default parameters. If the first
    attempt does not converge, automatically retries once with tighter
    tolerances and doubled iteration limits.

    Parameters
    ----------
    config_params : dict
        Configuration parameters for the model. Numerical solver
        parameters (tolerances, iteration limits, step sizes) are
        optional; sensible mass-adaptive defaults are used when absent.
    material_dictionaries : dict
        EOS registry dict keyed by EOS identifier string.
    melting_curves_functions : tuple or None
        (solidus_func, liquidus_func) for EOS needing external melting curves.
    input_dir : str
        Directory containing input files.
    layer_mixtures : dict or None, optional
        Per-layer LayerMixture objects. If None, parsed from
        ``config_params['layer_eos_config']``. PROTEUS/CALLIOPE can provide
        pre-built mixtures with runtime-updated fractions.
    volatile_profile : VolatileProfile or None, optional
        Volatile profile for binodal-aware structure. Passed through to
        ``_solve()`` but not used directly by the retry logic.
    temperature_function : callable or None, optional
        External temperature function with signature ``f(r, P) -> T``
        where ``r`` is radius in m, ``P`` is pressure in Pa, and ``T``
        is temperature in K. When provided, bypasses the internal
        temperature mode dispatch (isothermal/linear/prescribed/adiabatic).
        Used by PROTEUS to pass SPIDER/Aragog T(r) profiles directly
        in memory.

    Returns
    -------
    dict
        Model results including radii, density, gravity, pressure, temperature,
        mass enclosed, convergence status, and timing.
    """
    result = _solve(
        config_params,
        material_dictionaries,
        melting_curves_functions,
        input_dir,
        layer_mixtures=layer_mixtures,
        volatile_profile=volatile_profile,
        temperature_function=temperature_function,
    )

    if not result['converged']:
        # Build tightened params for retry
        planet_mass = config_params['planet_mass']
        defaults = _default_solver_params(planet_mass)
        # Merge: explicit caller values override defaults
        effective = dict(defaults)
        for key in defaults:
            if key in config_params:
                effective[key] = config_params[key]
        tightened = _tighten_solver_params(effective)

        logger.warning(
            'Structure solve did not converge (mass=%s, density=%s, pressure=%s). '
            'Retrying with tighter parameters.',
            result.get('converged_mass', False),
            result.get('converged_density', False),
            result.get('converged_pressure', False),
        )

        # Seed retry with the first attempt's radius estimate, if plausible
        retry_params = copy.copy(config_params)
        retry_params.update(tightened)
        if 'radii' in result and result['radii'] is not None and len(result['radii']) > 0:
            r_last = result['radii'][-1]
            mass_in_earth = max(planet_mass, 0.01 * earth_mass) / earth_mass
            r_seager = earth_radius * mass_in_earth**0.282
            if np.isfinite(r_last) and r_last > 0 and 0.2 * r_seager < r_last < 5.0 * r_seager:
                retry_params['_initial_radius_guess'] = r_last

        result = _solve(
            retry_params,
            material_dictionaries,
            melting_curves_functions,
            input_dir,
            layer_mixtures=layer_mixtures,
            volatile_profile=volatile_profile,
            temperature_function=temperature_function,
        )

        if not result['converged']:
            logger.warning(
                'Structure solve did not converge after retry. '
                'Returning best result with converged=False.'
            )

    return result


def _solve(
    config_params,
    material_dictionaries,
    melting_curves_functions,
    input_dir,
    layer_mixtures=None,
    volatile_profile=None,
    temperature_function=None,
):
    """Internal solver: single attempt at the structure solve.

    Parameters
    ----------
    config_params : dict
        Configuration parameters. Numerical solver params are optional;
        mass-adaptive defaults are used when absent.
    material_dictionaries : dict
        EOS registry dict keyed by EOS identifier string.
    melting_curves_functions : tuple or None
        (solidus_func, liquidus_func) for EOS needing external melting curves.
    input_dir : str
        Directory containing input files.
    layer_mixtures : dict or None, optional
        Per-layer LayerMixture objects.
    temperature_function : callable or None, optional
        External temperature function ``f(r, P) -> T``. When provided,
        bypasses internal temperature mode dispatch and adiabat blending.

    Returns
    -------
    dict
        Model results including convergence status.
    """
    # Initialize convergence flags
    converged = False
    converged_pressure = False
    converged_density = False
    converged_mass = False

    # Unpack physics parameters (required)
    planet_mass = config_params['planet_mass']
    core_mass_fraction = config_params['core_mass_fraction']
    mantle_mass_fraction = config_params['mantle_mass_fraction']
    temperature_mode = config_params['temperature_mode']
    surface_temperature = config_params['surface_temperature']
    center_temperature = config_params['center_temperature']
    temp_profile_file = config_params['temp_profile_file']
    layer_eos_config = config_params['layer_eos_config']
    num_layers = config_params['num_layers']
    target_surface_pressure = config_params['target_surface_pressure']

    # Unpack numerical solver parameters (optional, with mass-adaptive defaults)
    defaults = _default_solver_params(planet_mass)
    max_iterations_outer = config_params.get(
        'max_iterations_outer', defaults['max_iterations_outer']
    )
    tolerance_outer = config_params.get('tolerance_outer', defaults['tolerance_outer'])
    max_iterations_inner = config_params.get(
        'max_iterations_inner', defaults['max_iterations_inner']
    )
    tolerance_inner = config_params.get('tolerance_inner', defaults['tolerance_inner'])
    relative_tolerance = config_params.get(
        'relative_tolerance', defaults['relative_tolerance']
    )
    absolute_tolerance = config_params.get(
        'absolute_tolerance', defaults['absolute_tolerance']
    )
    maximum_step = config_params.get('maximum_step', defaults['maximum_step'])
    adaptive_radial_fraction = config_params.get(
        'adaptive_radial_fraction', defaults['adaptive_radial_fraction']
    )
    max_center_pressure_guess = config_params.get(
        'max_center_pressure_guess', defaults['max_center_pressure_guess']
    )
    pressure_tolerance = config_params.get(
        'pressure_tolerance', defaults['pressure_tolerance']
    )
    max_iterations_pressure = config_params.get(
        'max_iterations_pressure', defaults['max_iterations_pressure']
    )
    # Optional initial radius guess from a previous failed attempt
    initial_radius_guess = config_params.get('_initial_radius_guess', None)

    # Build per-EOS mushy_zone_factors dict. Prefer the dict if present
    # (set by load_zalmoxis_config). Fall back to building one from the
    # single float for backward compat with callers that only set the
    # global 'mushy_zone_factor' key.
    if 'mushy_zone_factors' in config_params:
        mushy_zone_factors = config_params['mushy_zone_factors']
    else:
        _global_mzf = config_params.get('mushy_zone_factor', 1.0)
        mushy_zone_factors = {
            'PALEOS:iron': _global_mzf,
            'PALEOS:MgSiO3': _global_mzf,
            'PALEOS:H2O': _global_mzf,
        }
    condensed_rho_min = config_params.get('condensed_rho_min', CONDENSED_RHO_MIN_DEFAULT)
    condensed_rho_scale = config_params.get('condensed_rho_scale', CONDENSED_RHO_SCALE_DEFAULT)
    binodal_T_scale = config_params.get('binodal_T_scale', BINODAL_T_SCALE_DEFAULT)

    # Parse layer mixtures if not provided externally (PROTEUS/CALLIOPE)
    if layer_mixtures is None:
        layer_mixtures = parse_all_layer_mixtures(layer_eos_config)

    # Check if any component in any layer uses T-dependent EOS
    uses_Tdep = any_component_is_tdep(layer_mixtures)

    # Enforce per-EOS mass limits for T-dependent tables
    mass_in_earth = planet_mass / earth_mass
    all_comps = set()
    for mix in layer_mixtures.values():
        all_comps.update(mix.components)
    tdep_comps = all_comps & TDEP_EOS_NAMES
    if tdep_comps:
        for eos_name in tdep_comps:
            if eos_name == 'WolfBower2018:MgSiO3':
                max_mass = WOLFBOWER2018_MAX_MASS_EARTH
                reason = (
                    'Deep-mantle pressures far exceed the 1 TPa table boundary '
                    'at higher masses, making clamped densities unreliable. '
                    'Use RTPress100TPa:MgSiO3 for higher masses.'
                )
            elif eos_name == 'RTPress100TPa:MgSiO3':
                max_mass = RTPRESS100TPA_MAX_MASS_EARTH
                reason = (
                    'The RTPress100TPa melt table extends to 100 TPa but '
                    'the solid table is limited to 1 TPa.'
                )
            elif eos_name == 'PALEOS-2phase:MgSiO3':
                max_mass = PALEOS_MAX_MASS_EARTH
                reason = (
                    'The PALEOS MgSiO3 tables extend to 100 TPa for both '
                    'solid and liquid phases.'
                )
            elif eos_name in ('PALEOS:iron', 'PALEOS:MgSiO3', 'PALEOS:H2O'):
                max_mass = PALEOS_UNIFIED_MAX_MASS_EARTH
                reason = 'The unified PALEOS tables extend to 100 TPa (P: 1 bar to 100 TPa).'
            elif eos_name == 'Chabrier:H':
                max_mass = PALEOS_UNIFIED_MAX_MASS_EARTH
                reason = (
                    'The Chabrier H table extends to 10^22 Pa but '
                    'has only been validated up to ~50 M_earth.'
                )
            else:
                continue
            if mass_in_earth > max_mass:
                raise ValueError(
                    f'{eos_name} EOS is limited to planets <= '
                    f'{max_mass} M_earth (requested {mass_in_earth:.2f} M_earth). '
                    f'{reason}'
                )

    # Setup initial guesses
    if initial_radius_guess is not None:
        radius_guess = initial_radius_guess
    else:
        radius_guess = (
            1000 * (7030 - 1840 * core_mass_fraction) * (planet_mass / earth_mass) ** 0.282
        )
    cmb_mass = 0
    core_mantle_mass = 0

    logger.info(
        f'Starting structure model for a {planet_mass / earth_mass} Earth masses planet '
        f"with EOS config {layer_eos_config} and temperature mode '{temperature_mode}'."
    )

    start_time = time.time()

    # Use module-level cache (persists across coupling steps)
    interpolation_cache = _interpolation_cache

    # Load solidus and liquidus functions if the caller provided them.
    # Unified PALEOS tables don't need external melting curves, so
    # melting_curves_functions may be None even when uses_Tdep is True.
    if melting_curves_functions is not None:
        solidus_func, liquidus_func = melting_curves_functions
    else:
        solidus_func, liquidus_func = None, None

    # --- Adiabatic temperature mode (standalone Zalmoxis) ----------------
    #
    # When temperature_mode='adiabatic', Zalmoxis computes a self-consistent
    # T(r) from EOS adiabat gradient tables. The transition from the initial
    # linear-T guess to the full adiabat is GRADUAL via a blending parameter:
    #
    #   blend = 0.0   (iteration 0: pure linear T)
    #   blend = 0.5   (first post-convergence iteration: half adiabat)
    #   blend = 1.0   (second post-convergence iteration: full adiabat)
    #
    # This blending prevents the solver from diverging when the temperature
    # profile changes abruptly from linear to adiabatic.
    #
    # In the PROTEUS-SPIDER coupling, temperature_mode is typically
    # 'adiabatic' in the config, but the blend never activates because
    # the initial linear-T structure converges and the mass break fires
    # with blend=0. This is correct: SPIDER provides its own T(r).
    # -------------------------------------------------------------------

    # Storage for the previous iteration's converged profiles.
    # Used by adiabatic mode to compute T(r) from the last P(r) and M(r).
    prev_radii = None
    prev_pressure = None
    prev_mass_enclosed = None

    # Adiabat blending state.
    _using_adiabat = False
    _adiabat_blend = 0.0
    _ADIABAT_BLEND_STEP = 0.25

    # For adiabatic mode, cap the initial center_temperature guess to
    # prevent the linear-T initial profile from being far above the
    # actual adiabat. A typical rocky planet adiabat at 1 M_earth has
    # T_center ~ 3 * T_surface. If center_temperature >> this, the
    # blend from linear to adiabat causes a huge density perturbation
    # that destabilizes convergence. This is conservative (adiabats
    # can be steeper for massive planets), so we use 5 * T_surface.
    if temperature_mode == 'adiabatic':
        max_reasonable_T_center = max(5.0 * surface_temperature, 3000.0)
        if center_temperature > max_reasonable_T_center:
            center_temperature = max_reasonable_T_center
            logger.debug(
                'Adiabatic mode: capped center_temperature initial guess '
                'to %.0f K (5x surface or 3000 K).', center_temperature,
            )

    # Solve the interior structure
    for outer_iter in range(max_iterations_outer):
        # Reset per-iteration convergence flags (prevent stale True from
        # a previous outer iteration masking failure in the current one).
        converged_pressure = False
        converged_density = False

        radii = np.linspace(0, radius_guess, num_layers)

        density = np.zeros(num_layers)
        mass_enclosed = np.zeros(num_layers)
        gravity = np.zeros(num_layers)
        pressure = np.zeros(num_layers)

        if temperature_function is not None:
            # External T(r,P) provided (e.g. from SPIDER/Aragog in memory).
            # Skip internal mode dispatch and adiabat blending entirely.
            _ext_tf = temperature_function  # avoid shadowing in nested defs

            def _temperature_func(r, P, _f=_ext_tf):
                return _f(r, P)

            temperatures = np.array(
                [_temperature_func(radii[i], pressure[i]) for i in range(num_layers)]
            )
        elif uses_Tdep:
            # Compute the linear (initial guess) temperature profile.
            # This is a function of radius only; wrap it to accept (r, P).
            _linear_tf = calculate_temperature_profile(
                radii,
                'linear',
                surface_temperature,
                center_temperature,
                input_dir,
                temp_profile_file,
            )

            if _using_adiabat and prev_pressure is not None:
                # Bump blend toward full adiabat
                _adiabat_blend = min(1.0, _adiabat_blend + _ADIABAT_BLEND_STEP)
                logger.debug(
                    'Outer iter %d: adiabat blend = %.2f', outer_iter, _adiabat_blend,
                )

                # Recompute adiabat from previous iteration's converged structure
                adiabat_T = compute_adiabatic_temperature(
                    prev_radii,
                    prev_pressure,
                    prev_mass_enclosed,
                    surface_temperature,
                    cmb_mass,
                    core_mantle_mass,
                    layer_mixtures,
                    material_dictionaries,
                    interpolation_cache,
                    solidus_func,
                    liquidus_func,
                    mushy_zone_factors,
                    condensed_rho_min,
                    condensed_rho_scale,
                    binodal_T_scale,
                )

                # Build T(P) interpolator from the previous iteration's
                # pressure profile.  This ensures that during the Brent
                # bracket search, the adiabatic temperature tracks the
                # ODE's actual pressure rather than a fixed radial mapping.
                # This prevents unphysical (low P, high T) queries that
                # hit NaN gaps in the PALEOS tables.
                _sort = np.argsort(prev_pressure)
                _P_sorted = prev_pressure[_sort]
                _T_sorted = adiabat_T[_sort]
                # Remove P <= 0 entries (surface padding)
                _valid = _P_sorted > 0
                _P_sorted = _P_sorted[_valid]
                _T_sorted = _T_sorted[_valid]
                _logP_sorted = np.log10(_P_sorted)

                # Clamp adiabat T to a physically reasonable range.
                # np.interp flat-extrapolates at the edges, but the Brent
                # solver may query at extreme P values far beyond the
                # converged profile, producing huge T. Cap at 100,000 K
                # (PALEOS table maximum) and floor at 100 K.
                _T_MAX_CLAMP = 100000.0
                _T_MIN_CLAMP = 100.0

                if _adiabat_blend < 1.0:
                    _blend = _adiabat_blend

                    def _temperature_func(
                        r, P, _b=_blend, _lp=_logP_sorted, _ts=_T_sorted, _ltf=_linear_tf
                    ):
                        T_lin = _ltf(r)
                        if P <= 0:
                            return T_lin
                        T_adi = float(np.interp(np.log10(P), _lp, _ts))
                        T_adi = max(_T_MIN_CLAMP, min(T_adi, _T_MAX_CLAMP))
                        return (1.0 - _b) * T_lin + _b * T_adi

                else:

                    def _temperature_func(r, P, _lp=_logP_sorted, _ts=_T_sorted):
                        if P <= 0:
                            return surface_temperature
                        T_val = float(np.interp(np.log10(P), _lp, _ts))
                        return max(_T_MIN_CLAMP, min(T_val, _T_MAX_CLAMP))

                # Pre-compute temperatures array for the density update loop
                # (uses the converged pressure from the previous iteration)
                temperatures = np.array(
                    [
                        _temperature_func(radii[i], prev_pressure[i])
                        for i in range(num_layers)
                    ]
                )
            else:

                def _temperature_func(r, P, _tf=_linear_tf):
                    return _tf(r)

                temperatures = _linear_tf(radii)
        else:
            _temperature_func = None
            temperatures = np.ones(num_layers) * 300

        cmb_mass = core_mass_fraction * planet_mass
        core_mantle_mass = (core_mass_fraction + mantle_mass_fraction) * planet_mass

        pressure[0] = earth_center_pressure

        for inner_iter in range(max_iterations_inner):
            old_density = density.copy()

            # Scaling-law estimate for central pressure
            pressure_guess = (
                earth_center_pressure
                * (planet_mass / earth_mass) ** 2
                * (radius_guess / earth_radius) ** (-4)
            )
            # Cap the central pressure guess for WolfBower2018 (1 TPa table)
            # but not for RTPress100TPa (100 TPa melt table)
            uses_WB2018 = 'WolfBower2018:MgSiO3' in all_comps
            if uses_WB2018:
                pressure_guess = min(pressure_guess, max_center_pressure_guess)

            # Mutable state to capture the last ODE solution from inside
            # the residual function (brentq doesn't return intermediate
            # results, only the root)
            _state = {'mass_enclosed': None, 'gravity': None, 'pressure': None, 'n_evals': 0}

            def _pressure_residual(p_center):
                """Surface pressure residual f(P_c) = P_surface(P_c) - P_target.

                When the ODE integration terminates early (pressure hit zero
                before reaching the planet surface), P_center is too low and
                the residual is negative.
                """
                y0 = [0, 0, p_center]
                m, g, p = solve_structure(
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
                    _temperature_func,
                    mushy_zone_factors,
                    condensed_rho_min,
                    condensed_rho_scale,
                    binodal_T_scale,
                )
                if logger.isEnabledFor(logging.DEBUG):
                    create_pressure_density_files(
                        outer_iter, inner_iter, _state['n_evals'], radii, p, density
                    )
                _state['mass_enclosed'] = m
                _state['gravity'] = g
                _state['pressure'] = p
                _state['n_evals'] += 1

                # Early termination: pressure reached zero before the
                # surface (padded with zeros) → P_center is too low
                if p[-1] <= 0:
                    return -target_surface_pressure

                return p[-1] - target_surface_pressure

            # Bracket: P_center must be positive and wide enough to
            # straddle the root (where surface P = target P)
            p_low = max(1e6, 0.1 * pressure_guess)
            p_high = 10.0 * pressure_guess
            if uses_WB2018:
                p_high = min(p_high, max_center_pressure_guess)

            try:
                # Pre-validate that the bracket straddles the root.
                # This gives a clearer error than brentq's generic ValueError,
                # and the except handler below gracefully falls back to the
                # last evaluated solution.
                f_low = _pressure_residual(p_low)
                f_high = _pressure_residual(p_high)
                if f_low * f_high > 0:
                    raise ValueError(
                        f'Brent bracket does not straddle the root: '
                        f'f({p_low:.2e})={f_low:.2e}, f({p_high:.2e})={f_high:.2e}.'
                    )
                p_solution, root_info = brentq(
                    _pressure_residual,
                    p_low,
                    p_high,
                    xtol=1e6,
                    rtol=1e-10,
                    maxiter=max_iterations_pressure,
                    full_output=True,
                )
                # Re-run solve_structure at the exact root to get clean profiles
                # (brentq may have evaluated _state at a slightly different P)
                y0_root = [0, 0, p_solution]
                mass_enclosed, gravity, pressure = solve_structure(
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
                    y0_root,
                    solidus_func,
                    liquidus_func,
                    _temperature_func,
                    mushy_zone_factors,
                    condensed_rho_min,
                    condensed_rho_scale,
                    binodal_T_scale,
                )

                surface_residual = abs(pressure[-1] - target_surface_pressure)
                # Allow zero pressure at the surface: the terminal event
                # pads truncated points with P=0, so check >= 0
                if (
                    root_info.converged
                    and surface_residual < pressure_tolerance
                    and np.min(pressure) >= 0
                ):
                    converged_pressure = True
                    logger.debug(
                        'Surface pressure converged after '
                        '%d evaluations (Brent method).', root_info.function_calls,
                    )
                else:
                    converged_pressure = False
                    logger.debug(
                        'Brent method: converged=%s, residual=%.2e Pa, min_P=%.2e Pa.',
                        root_info.converged, surface_residual, np.min(pressure),
                    )
            except ValueError:
                # f(p_low) and f(p_high) have the same sign — bracket
                # invalid.  Use the last evaluated solution if available.
                logger.debug(
                    'Could not bracket pressure root in [%.2e, %.2e] Pa.',
                    p_low, p_high,
                )
                if _state['mass_enclosed'] is not None:
                    mass_enclosed = _state['mass_enclosed']
                    gravity = _state['gravity']
                    pressure = _state['pressure']
                else:
                    # No evaluations succeeded — keep profiles from previous
                    # outer iteration (already initialised above).
                    logger.debug(
                        'No valid ODE solutions obtained during bracket search. '
                        'Keeping previous profiles.'
                    )
                converged_pressure = False

            # Update density grid using vectorized EOS lookups.
            # Partition shells by layer (all shells in a layer share one EOS),
            # then batch-call calculate_mixed_density_batch per layer.
            n_valid = min(num_layers, len(mass_enclosed))
            new_density = np.full(num_layers, np.nan)

            # Zero-pressure shells get zero density
            p_valid = pressure[:n_valid] > 0

            # Compute temperatures for valid shells
            if _temperature_func is not None:
                T_arr = np.array(
                    [
                        _temperature_func(radii[i], pressure[i])
                        for i in range(n_valid)
                        if p_valid[i]
                    ]
                )
            else:
                T_arr = np.full(int(np.sum(p_valid)), 300.0)

            # Partition shells by layer using mass boundaries
            valid_indices = np.where(p_valid[:n_valid])[0]
            m_valid = mass_enclosed[valid_indices]
            rtol = 1e-12
            in_core = m_valid < cmb_mass * (1.0 - rtol)
            in_ice = ('ice_layer' in layer_mixtures) & (
                m_valid >= core_mantle_mass * (1.0 - rtol)
            )
            in_mantle = ~in_core & ~in_ice

            for layer_name, mask in [
                ('core', in_core),
                ('mantle', in_mantle),
                ('ice_layer', in_ice),
            ]:
                if not np.any(mask) or layer_name not in layer_mixtures:
                    continue
                idx = valid_indices[mask]
                mix = layer_mixtures[layer_name]
                rho_batch = calculate_mixed_density_batch(
                    pressure[idx],
                    T_arr[np.where(mask)[0]]
                    if len(T_arr) == len(valid_indices)
                    else T_arr[mask],
                    mix,
                    material_dictionaries,
                    solidus_func,
                    liquidus_func,
                    interpolation_cache,
                    mushy_zone_factors,
                    condensed_rho_min,
                    condensed_rho_scale,
                    binodal_T_scale,
                )
                new_density[idx] = rho_batch

            # Fill NaN entries with last valid density (walking outward)
            last_valid = None
            for i in range(n_valid):
                if not p_valid[i]:
                    new_density[i] = 0.0
                elif np.isnan(new_density[i]):
                    new_density[i] = last_valid if last_valid is not None else old_density[i]
                else:
                    last_valid = new_density[i]

            # Picard blend
            density[:n_valid] = 0.5 * (new_density[:n_valid] + old_density[:n_valid])

            # Check density convergence
            relative_diff_inner = np.max(
                np.abs((density - old_density) / (old_density + 1e-20))
            )
            if relative_diff_inner < tolerance_inner:
                logger.debug(
                    'Inner loop converged after %d iterations.', inner_iter + 1,
                )
                converged_density = True
                break

            if inner_iter == max_iterations_inner - 1:
                logger.debug(
                    'Maximum inner iterations (%d) reached. '
                    'Density may not be fully converged.', max_iterations_inner,
                )

        # Recompute the temperatures array from the converged pressure profile
        # so model_results['temperature'] reflects actual T(P), not the pre-Brent estimate.
        if _temperature_func is not None:
            temperatures = np.array(
                [_temperature_func(radii[i], pressure[i]) for i in range(num_layers)]
            )

        # Save converged profiles for the next outer iteration's adiabat
        prev_radii = radii.copy()
        prev_pressure = np.asarray(pressure).copy()
        prev_mass_enclosed = np.asarray(mass_enclosed).copy()

        # Update radius guess with damped scaling to prevent oscillation.
        # The cube-root scaling is correct in direction but can overshoot
        # wildly when calculated_mass << planet_mass, catapulting radius
        # to unphysical values and trapping the solver in a cycle.
        calculated_mass = mass_enclosed[-1]
        if calculated_mass <= 0 or not np.isfinite(calculated_mass):
            radius_guess *= 0.8
            logger.debug(
                'Outer iter %d: calculated_mass=%.2e, shrinking radius_guess to %.0f m.',
                outer_iter, calculated_mass, radius_guess,
            )
        else:
            scale = (planet_mass / calculated_mass) ** (1.0 / 3.0)
            scale = max(0.5, min(scale, 2.0))
            radius_guess *= scale
        cmb_mass = core_mass_fraction * calculated_mass
        core_mantle_mass = (core_mass_fraction + mantle_mass_fraction) * calculated_mass

        relative_diff_outer_mass = np.abs((calculated_mass - planet_mass) / planet_mass)

        # MASS CONVERGENCE CHECK
        # When temperature_mode='adiabatic' and the blend has not yet reached
        # 1.0, mass convergence triggers the adiabat transition instead of
        # breaking. The blend ramps 0 -> 0.5 -> 1.0 over successive mass
        # convergences, preventing solver divergence.
        if relative_diff_outer_mass < tolerance_outer:
            if temperature_mode == 'adiabatic' and _adiabat_blend < 1.0:
                if not _using_adiabat:
                    _using_adiabat = True
                    logger.info(
                        f'Outer iter {outer_iter}: mass converged with linear T, '
                        f'activating adiabat blend.'
                    )
                # Continue iterating to let the blend ramp up
                continue
            logger.info(f'Outer loop (total mass) converged after {outer_iter + 1} iterations.')
            converged_mass = True
            break

        if outer_iter == max_iterations_outer - 1:
            logger.debug(
                'Maximum outer iterations (%d) reached. '
                'Total mass may not be fully converged.', max_iterations_outer,
            )

    if converged_mass and converged_density and converged_pressure:
        converged = True

    end_time = time.time()
    total_time = end_time - start_time

    # Clean up surface artifacts. The Picard iteration can produce
    # density jumps at the outermost shells where the pressure drops
    # rapidly and the EOS lookup is sensitive to small P changes.
    # Detect shells where the density gradient suddenly steepens
    # (more than 3x the running average) and replace them with a
    # linear extrapolation from the smooth interior.
    pressure = np.asarray(pressure)
    density = np.asarray(density)

    # Zero out padded (P=0) shells
    density[pressure <= 0] = 0.0

    # Find the last shell with positive density
    i_surf = len(density) - 1
    while i_surf > 0 and density[i_surf] <= 0:
        i_surf -= 1

    if i_surf > 10:
        # Compute density gradient in the outer mantle.
        # Use the last 20 shells before the surface edge, but do not
        # cross layer boundaries (CMB, ice-layer transition) where
        # real density jumps exist.
        # Find outermost layer boundary by detecting large density
        # ratios (> 1.5x) walking inward from surface.
        i_layer_base = max(0, i_surf - 40)
        for _k in range(i_surf - 1, i_layer_base, -1):
            if density[_k] > 0 and density[_k + 1] > 0:
                ratio = density[_k] / density[_k + 1]
                if ratio > 1.5 or ratio < 1.0 / 1.5:
                    i_layer_base = _k + 1
                    break
        n_check = min(20, i_surf - i_layer_base)
        if n_check < 6:
            n_check = min(20, i_surf - 5)
        i_start = i_surf - n_check
        grads = np.diff(density[i_start : i_surf + 1])
        # Find where the gradient suddenly changes
        # (the smooth interior has nearly constant negative gradient).
        if len(grads) > 5:
            median_grad = np.median(grads[: n_check // 2])
            if median_grad < 0:  # density decreasing outward (normal)
                for j in range(len(grads)):
                    i_global = i_start + j + 1
                    # Catch both sudden drops (3x steeper) and sudden
                    # increases (sign reversal). Both indicate the Picard
                    # iteration produced unreliable density at the surface.
                    is_sudden_drop = grads[j] < 3 * median_grad
                    is_sudden_rise = grads[j] > 0 and abs(grads[j]) > abs(median_grad)
                    if is_sudden_drop or is_sudden_rise:
                        # Extrapolate linearly from the smooth region
                        for k in range(i_global, i_surf + 1):
                            extrap = density[i_global - 1] + median_grad * (k - i_global + 1)
                            density[k] = max(extrap, 0.0)
                        break

    model_results = {
        'layer_eos_config': layer_eos_config,
        'radii': radii,
        'density': density,
        'gravity': gravity,
        'pressure': pressure,
        'temperature': temperatures,
        'mass_enclosed': mass_enclosed,
        'cmb_mass': cmb_mass,
        'core_mantle_mass': core_mantle_mass,
        'total_time': total_time,
        'converged': converged,
        'converged_pressure': converged_pressure,
        'converged_density': converged_density,
        'converged_mass': converged_mass,
    }
    return model_results


def solve_miscible_interior(
    config_params,
    material_dictionaries,
    melting_curves_functions,
    input_dir,
    layer_mixtures=None,
    volatile_profile=None,
    temperature_function=None,
    h2_mass_targets=None,
    max_iterations=10,
    mass_tolerance=0.01,
):
    """Run Zalmoxis with mass-conservation iteration for binodal-controlled species.

    Wraps ``main()`` in an outer loop that adjusts the interior mass fractions
    (``volatile_profile.x_interior``) for each binodal-controlled species until
    the radially integrated mass of each species matches the target.

    Parameters
    ----------
    config_params : dict
        Configuration parameters for the model.
    material_dictionaries : dict
        EOS registry dict keyed by EOS identifier string.
    melting_curves_functions : tuple or None
        (solidus_func, liquidus_func) for EOS needing external melting curves.
    input_dir : str
        Directory containing input files.
    layer_mixtures : dict or None, optional
        Per-layer LayerMixture objects.
    volatile_profile : VolatileProfile or None, optional
        Must have ``global_miscibility = True`` and ``x_interior`` set with
        initial guesses.
    h2_mass_targets : dict or None
        Target masses for binodal-controlled species in kg, keyed by EOS name.
        Example: ``{"Chabrier:H": 1.5e22, "PALEOS:H2O": 3e23}``.
        If None or empty, runs ``main()`` once without iteration.
    max_iterations : int
        Maximum iterations for the mass conservation loop.
    mass_tolerance : float
        Relative mass tolerance for convergence. Default 0.01 (1%).

    Returns
    -------
    dict
        Model results from ``main()`` plus additional keys:

        - ``solvus_radius``: radius where binodal is crossed [m], or None
        - ``solvus_temperature``: temperature at solvus [K], or None
        - ``solvus_pressure``: pressure at solvus [Pa], or None
        - ``x_interior_converged``: converged interior mass fractions dict
        - ``h2_mass_integrated``: integrated masses of binodal species [kg]
        - ``miscibility_converged``: True if mass conservation converged
        - ``miscibility_iterations``: number of iterations used
    """
    # No iteration needed if no targets
    if (
        volatile_profile is None
        or not volatile_profile.global_miscibility
        or not h2_mass_targets
    ):
        result = main(
            config_params,
            material_dictionaries,
            melting_curves_functions,
            input_dir,
            layer_mixtures,
            volatile_profile,
            temperature_function=temperature_function,
        )
        result['solvus_radius'] = None
        result['solvus_temperature'] = None
        result['solvus_pressure'] = None
        result['x_interior_converged'] = {}
        result['h2_mass_integrated'] = {}
        result['miscibility_converged'] = True
        result['miscibility_iterations'] = 0
        return result

    species_list = list(h2_mass_targets.keys())
    misc_converged = False
    result = None

    for iteration in range(max_iterations):
        # Run Zalmoxis structure solve
        result = main(
            config_params,
            material_dictionaries,
            melting_curves_functions,
            input_dir,
            layer_mixtures,
            volatile_profile,
            temperature_function=temperature_function,
        )

        if not result.get('converged', False):
            logger.warning(
                'Miscibility iteration %d: Zalmoxis did not converge. '
                'Skipping mass conservation check.',
                iteration,
            )
            break

        radii = result['radii']
        density = result['density']
        pressure = result['pressure']
        temperature = result['temperature']

        # Integrate mass of each binodal-controlled species
        # M_species = integral(w_species(r) * rho(r) * 4*pi*r^2 dr)
        # over shells where species is above the binodal
        integrated_masses = {}
        solvus_info = {'radius': None, 'temperature': None, 'pressure': None}

        for species in species_list:
            x_int = volatile_profile.x_interior.get(species, 0.0)
            m_species = 0.0
            solvus_found = False

            for i in range(len(radii) - 1):
                r_mid = 0.5 * (radii[i] + radii[i + 1])
                dr = radii[i + 1] - radii[i]
                rho_mid = 0.5 * (density[i] + density[i + 1])
                P_mid = 0.5 * (pressure[i] + pressure[i + 1])
                T_mid = 0.5 * (temperature[i] + temperature[i + 1])

                if rho_mid <= 0 or P_mid <= 0 or dr <= 0:
                    continue

                # Check if this shell is above the binodal
                if volatile_profile._is_above_binodal(species, P_mid, T_mid):
                    shell_mass = rho_mid * 4.0 * np.pi * r_mid**2 * dr
                    m_species += x_int * shell_mass
                elif not solvus_found:
                    # First shell below binodal (scanning outward from center):
                    # this is the solvus crossing
                    solvus_found = True
                    if species == 'Chabrier:H':
                        solvus_info['radius'] = r_mid
                        solvus_info['temperature'] = T_mid
                        solvus_info['pressure'] = P_mid

            integrated_masses[species] = m_species

        # Check convergence
        all_converged = True
        for species in species_list:
            target = h2_mass_targets[species]
            if target <= 0:
                continue
            integrated = integrated_masses.get(species, 0.0)
            rel_error = abs(integrated - target) / target
            logger.info(
                'Miscibility iter %d: %s integrated=%.3e kg, ' 'target=%.3e kg, rel_error=%.4f',
                iteration,
                species,
                integrated,
                target,
                rel_error,
            )
            if rel_error > mass_tolerance:
                all_converged = False
                # Update x_interior using secant-like scaling:
                # new_x = old_x * (target / integrated)
                old_x = volatile_profile.x_interior.get(species, 0.01)
                if integrated > 0:
                    new_x = old_x * (target / integrated)
                else:
                    # No mass integrated: double the guess
                    new_x = old_x * 2.0
                # Clamp to physical range
                new_x = max(1e-6, min(0.5, new_x))
                volatile_profile.x_interior[species] = new_x
                logger.info(
                    'Miscibility iter %d: %s x_interior %.6f -> %.6f',
                    iteration,
                    species,
                    old_x,
                    new_x,
                )

        if all_converged:
            misc_converged = True
            logger.info('Miscibility converged after %d iterations.', iteration + 1)
            break

    if not misc_converged:
        logger.warning(
            'Miscibility mass conservation did not converge after %d '
            'iterations. Proceeding with current x_interior values.',
            max_iterations,
        )

    # Add solvus and convergence info to results
    result['solvus_radius'] = solvus_info['radius']
    result['solvus_temperature'] = solvus_info['temperature']
    result['solvus_pressure'] = solvus_info['pressure']
    result['x_interior_converged'] = dict(volatile_profile.x_interior)
    result['h2_mass_integrated'] = integrated_masses
    result['miscibility_converged'] = misc_converged
    result['miscibility_iterations'] = iteration + 1

    return result


