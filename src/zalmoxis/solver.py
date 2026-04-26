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
import math
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


def _anderson_mix(x_hist, f_hist, x_k, f_k, m_max=5, beta=1.0):
    """Type-II Anderson mixing step for a fixed-point iteration.

    Given history of iterates ``x_hist`` (list of 1-D ``np.ndarray``, each
    the density profile from a previous Picard iter) and residuals
    ``f_hist`` (same length, with ``f_i = g(x_i) - x_i``), plus the
    current iterate ``x_k`` and its residual ``f_k``, compute the next
    iterate by least-squares-combining previous residuals.

    Anderson Type-II update (following Walker & Ni 2011 Eq. 5):
        F_k = [f_k - f_{k-m}, ..., f_k - f_{k-1}]
        X_k = [x_k - x_{k-m}, ..., x_k - x_{k-1}]
        gamma_k = argmin_gamma || f_k - F_k gamma ||_2  (least squares)
        x_{k+1} = x_k + beta * f_k - (X_k + beta * F_k) gamma_k

    Returns the next iterate, or ``None`` if Anderson fails
    (matrix singular, non-finite result, too-short history). Caller
    must fall back to damped Picard when None is returned.

    Notes
    -----
    - ``m_max`` is the maximum history window (typically 3-5 for
      convergence problems in R^n). More history = more accelerative
      but also more risk of ill-conditioning.
    - beta=1.0 is Anderson without additional damping. beta<1.0 mixes
      Anderson with the plain residual step (conservative).
    """
    if len(f_hist) < 1 or len(x_hist) != len(f_hist):
        return None

    m = min(len(f_hist), m_max - 1)
    x_list = list(x_hist[-m:]) + [x_k]
    f_list = list(f_hist[-m:]) + [f_k]

    # Build delta columns (columns m: each col = f_k - f_{k-i}, X analogously)
    F_cols = []
    X_cols = []
    for i in range(m):
        F_cols.append(f_list[-1] - f_list[i])
        X_cols.append(x_list[-1] - x_list[i])
    F = np.column_stack(F_cols)
    X = np.column_stack(X_cols)

    # Early bail on non-finite inputs. LAPACK lstsq prints DLASCL warnings
    # on NaN/Inf and would still return (we always check isfinite on the
    # result), but checking up front avoids the spurious warning spam.
    if not (np.all(np.isfinite(F)) and np.all(np.isfinite(f_k))):
        return None

    # Least-squares: gamma = argmin || F gamma - f_k ||_2
    try:
        gamma, *_ = np.linalg.lstsq(F, f_k, rcond=None)
    except np.linalg.LinAlgError:
        return None

    x_next = x_k + beta * f_k - (X + beta * F) @ gamma

    if not np.all(np.isfinite(x_next)):
        return None
    return x_next


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
        # Outer mass-radius solver: 'picard' (damped fixed-point, default,
        # historically used) or 'newton' (Newton + brentq bracketing).
        # Newton was introduced 2026-04-26 (T2.1) to escape the basin
        # attractor in damped Picard's R-search on hot fully-molten
        # mantle profiles. See session_2026_04_26_t2_1a_newton_prototype.md.
        'outer_solver': 'picard',
    }


_VALID_OUTER_SOLVERS = ('picard', 'newton')


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
    tightened['wall_timeout'] = params.get('wall_timeout', 300.0) * 2
    return tightened


def main(
    config_params,
    material_dictionaries,
    melting_curves_functions,
    input_dir,
    layer_mixtures=None,
    volatile_profile=None,
    temperature_function=None,
    temperature_arrays=None,
    p_center_hint=None,
    initial_density=None,
    initial_radii=None,
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
    temperature_arrays : tuple[ndarray, ndarray] or None, optional
        Explicit r-indexed T profile ``(r_arr, T_arr)``. Only consumed
        by the JAX path (``config_params['use_jax']=True``). Preferred
        over ``temperature_function`` when the caller's T is naturally
        r-indexed (e.g. SPIDER/Aragog-coupled runs): the P-indexed
        tabulation inside ``jax_eos.wrapper`` collapses to a constant
        for P-ignoring callables and breaks convergence. See
        ``tools/benchmarks/bench_coupled_tempfunc.py``.
    initial_density : numpy.ndarray or None, optional
        Density profile from a previous solve, used to seed the Picard
        iteration. Must be paired with ``initial_radii``. Interpolated
        onto the current radial grid. Dramatically accelerates convergence
        when the structure changes incrementally between coupling steps.
    initial_radii : numpy.ndarray or None, optional
        Radial grid corresponding to ``initial_density``.

    Returns
    -------
    dict
        Model results including radii, density, gravity, pressure, temperature,
        mass enclosed, convergence status, and timing.
    """
    # Validate outer-solver choice. Default is 'picard' (the damped
    # fixed-point loop inside `_solve()`); 'newton' dispatches to
    # `_solve_newton_outer()` (T2.1).
    outer_solver = config_params.get('outer_solver', 'picard')
    if outer_solver not in _VALID_OUTER_SOLVERS:
        raise ValueError(
            f"outer_solver must be one of {_VALID_OUTER_SOLVERS!r}, "
            f"got {outer_solver!r}."
        )
    if outer_solver == 'newton':
        return _solve_newton_outer(
            config_params,
            material_dictionaries,
            melting_curves_functions,
            input_dir,
            layer_mixtures=layer_mixtures,
            volatile_profile=volatile_profile,
            temperature_function=temperature_function,
            temperature_arrays=temperature_arrays,
            p_center_hint=p_center_hint,
            initial_density=initial_density,
            initial_radii=initial_radii,
        )

    result = _solve(
        config_params,
        material_dictionaries,
        melting_curves_functions,
        input_dir,
        layer_mixtures=layer_mixtures,
        volatile_profile=volatile_profile,
        temperature_function=temperature_function,
        temperature_arrays=temperature_arrays,
        p_center_hint=p_center_hint,
        initial_density=initial_density,
        initial_radii=initial_radii,
    )

    # Skip retry when the first attempt is "good enough": density and
    # pressure converged, mass error within `_RETRY_SKIP_MASS_ERR`. On
    # hot fully-molten T(r) profiles (early CHILI coupled iters) the
    # retry path doubles every iteration cap and wall_timeout, takes
    # 600+ s, and routinely lands at the SAME or SLIGHTLY-WORSE mass
    # error as the first attempt. Empirical (live no-Anderson bench
    # iters from chili_dry_coupled_stage2_ab_postfix_b_noanderson):
    #   - iter 23869 first attempt mass err <5%, retry 600s, ends at 2.6%
    #   - iter 27657 first attempt mass err >5%, retry 600s timeout,
    #     ends at 3.37% (still > 3%, would still trigger retry on
    #     subsequent calls!) -- retry is genuinely useless here.
    # Threshold 7% catches both. The outer mass-radius loop will
    # converge across PROTEUS iters anyway; one resolve ending at 5-7%
    # mass error is fine when the next resolve corrects it.
    _RETRY_SKIP_MASS_ERR = 0.07
    _first_mass_err = result.get('best_mass_error')
    _retry_skipped = (
        not result['converged']
        and result.get('converged_density', False)
        and result.get('converged_pressure', False)
        and _first_mass_err is not None
        and _first_mass_err < _RETRY_SKIP_MASS_ERR
    )
    if _retry_skipped:
        logger.info(
            'Skipping retry: density and pressure converged, mass error '
            '%.4f%% < %.2f%% threshold. Accepting first-attempt solution.',
            _first_mass_err * 100, _RETRY_SKIP_MASS_ERR * 100,
        )
        result['converged'] = True
        result['converged_mass'] = True

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

        # Seed retry with the first attempt's results
        retry_params = copy.copy(config_params)
        retry_params.update(tightened)
        retry_density = None
        retry_radii = None
        if 'radii' in result and result['radii'] is not None and len(result['radii']) > 0:
            r_last = result['radii'][-1]
            mass_in_earth = max(planet_mass, 0.01 * earth_mass) / earth_mass
            r_seager = earth_radius * mass_in_earth**0.282
            if np.isfinite(r_last) and r_last > 0 and 0.2 * r_seager < r_last < 5.0 * r_seager:
                retry_params['_initial_radius_guess'] = r_last
            # Use first attempt's density as seed for retry
            if result.get('density') is not None and np.any(result['density'] > 0):
                retry_density = result['density']
                retry_radii = result['radii']

        result = _solve(
            retry_params,
            material_dictionaries,
            melting_curves_functions,
            input_dir,
            layer_mixtures=layer_mixtures,
            volatile_profile=volatile_profile,
            temperature_function=temperature_function,
            temperature_arrays=temperature_arrays,
            p_center_hint=p_center_hint,
            initial_density=retry_density,
            initial_radii=retry_radii,
        )

        if not result['converged']:
            logger.warning(
                'Structure solve did not converge after retry. '
                'Returning best result with converged=False.'
            )

    return result


# Newton outer-loop defaults (T2.1c). These are the validated values
# from the script-level prototype (scripts/probe_zalmoxis_newton_prototype.py)
# which converged 12/12 G4 starts to dM/M < 1e-4 on the failure-dump
# T(r) in 3-6 Newton iterations.
_NEWTON_DEFAULT_MAX_ITER = 30
_NEWTON_DEFAULT_TOL = 1.0e-4
_NEWTON_DEFAULT_DR_REL = 1.0e-3
# Trust region: cap each Newton step to 10% of current R. The M(R)
# map has internal nonlinearity from the integrator's terminal-event
# tolerance and EOS phase boundaries, so a generous trust region tends
# to overshoot on the first 1-2 iters from a poor R0.
_NEWTON_DEFAULT_TRUST_REGION_FRAC = 0.1
# Hard physical bounds on R; Newton step gets clamped, brentq hits
# error if any bracket lookup escapes these.
_NEWTON_DEFAULT_R_MIN = 2.0e6
_NEWTON_DEFAULT_R_MAX = 3.0e7
# Required minimum integrator tolerance for Newton: with looser tols
# the M(R) map has noise > 1e-3 which Newton cannot beat. Validated
# in T2.1a.
_NEWTON_REQUIRED_REL_TOL = 1.0e-7


def _solve_newton_outer(
    config_params,
    material_dictionaries,
    melting_curves_functions,
    input_dir,
    layer_mixtures=None,
    volatile_profile=None,
    temperature_function=None,
    temperature_arrays=None,
    p_center_hint=None,
    initial_density=None,
    initial_radii=None,
):
    """Newton outer mass-radius loop on ``f(R) = M(R) - M_target`` (T2.1).

    Replaces the damped-Picard outer loop in ``_solve()`` with a Newton
    iteration. The inner Picard density loop is untouched: each Newton
    step calls ``_solve()`` with ``max_iterations_outer=1`` and a
    forced ``_initial_radius_guess`` to evaluate ``M(R)``.

    Algorithm (validated in T2.1a, see
    ``session_2026_04_26_t2_1a_newton_prototype.md``):

    1. Evaluate ``f(R_k) = M(R_k) - M_target`` via inner Picard.
    2. If ``|f|/M_target < tol``, return.
    3. Estimate ``dM/dR`` via central difference at ``R_k +/- dR``.
    4. ``R_{k+1} = R_k - f / (dM/dR)``, clamped to a 10 %-of-R trust
       region.
    5. Damping: if previous step did not reduce ``|f|``, halve the
       proposed step.

    Brentq fall-back (T2.1d) is deferred to a follow-on commit. This
    commit ships pure Newton; degenerate cases (vanishing derivative,
    out-of-bounds step, max-iter without convergence) raise
    ``RuntimeError`` with the best result attached. Production deploy
    requires T2.1d's brentq fall-back to recover from these.

    Parameters
    ----------
    Same as :func:`main`. ``config_params['outer_solver']`` must be
    ``'newton'`` to dispatch here. Recognised numerical knobs:

    - ``newton_max_iter`` (default 30): Newton iter cap.
    - ``newton_tol`` (default 1e-4): convergence on
      ``|M-M_target|/M_target``.
    - ``newton_dR_rel`` (default 1e-3): central-difference step as
      fraction of R.
    - ``newton_trust_region_frac`` (default 0.1): max ``|dR|/R`` per step.
    - ``newton_R_min``, ``newton_R_max`` (defaults 2e6, 3e7 m): hard
      bounds on R.

    The integrator tolerances (``relative_tolerance``,
    ``absolute_tolerance``) MUST be at least 1e-7 / 1e-8: looser tols
    leave ~1e-3 noise on M(R) which dominates Newton's central diff.
    Caller will get a ``ValueError`` on entry if this is violated.

    Returns
    -------
    dict
        Same shape as ``_solve()`` for caller compatibility:
        ``{radii, density, gravity, pressure, temperature, mass_enclosed,
        cmb_mass, core_mantle_mass, total_time, converged,
        converged_pressure, converged_density, converged_mass,
        best_mass_error, p_center, layer_eos_config}``.

    Raises
    ------
    ValueError
        If integrator tolerances are too loose for Newton to converge.
    RuntimeError
        If Newton fails to converge within ``newton_max_iter`` and
        brentq fall-back is not yet implemented (T2.1d).
    """
    M_target = float(config_params['planet_mass'])
    defaults = _default_solver_params(M_target)

    # Newton numerical knobs (use defaults if absent).
    max_iter = int(config_params.get('newton_max_iter', _NEWTON_DEFAULT_MAX_ITER))
    tol = float(config_params.get('newton_tol', _NEWTON_DEFAULT_TOL))
    dR_rel = float(config_params.get('newton_dR_rel', _NEWTON_DEFAULT_DR_REL))
    trust_region_frac = float(
        config_params.get('newton_trust_region_frac', _NEWTON_DEFAULT_TRUST_REGION_FRAC)
    )
    R_min = float(config_params.get('newton_R_min', _NEWTON_DEFAULT_R_MIN))
    R_max = float(config_params.get('newton_R_max', _NEWTON_DEFAULT_R_MAX))

    # Integrator-tolerance precondition: enforce rel_tol <= 1e-7 so M(R)
    # is smooth enough for central-difference dM/dR.
    rel_tol_eff = float(config_params.get('relative_tolerance')
                        or defaults['relative_tolerance'])
    if rel_tol_eff > _NEWTON_REQUIRED_REL_TOL:
        raise ValueError(
            f'Newton outer requires relative_tolerance <= {_NEWTON_REQUIRED_REL_TOL!r}; '
            f'got {rel_tol_eff!r}. With looser tols, M(R) noise (~1e-3) '
            f"dominates Newton's central-difference dM/dR. Tighten the "
            'integrator (recommended: relative_tolerance=1e-9, '
            'absolute_tolerance=1e-10) when using outer_solver="newton". '
            'See session_2026_04_26_t2_1a_newton_prototype.md.'
        )

    # Initial radius guess. Same Seager+2007 estimate that _solve uses,
    # unless caller supplied _initial_radius_guess.
    initial_radius_guess = config_params.get('_initial_radius_guess', None)
    if initial_radius_guess is not None:
        R = float(initial_radius_guess)
    else:
        # Mass-adaptive initial guess (matches _solve line ~541).
        core_mass_fraction = float(config_params.get('core_mass_fraction', 0.0))
        R = (
            1000 * (7030 - 1840 * core_mass_fraction)
            * (M_target / earth_mass) ** 0.282
        )
    R = float(max(R_min, min(R_max, R)))

    # ----- Inner-Picard-at-fixed-R helper -----
    # Each Newton step calls _solve(max_iterations_outer=1, _initial_radius_guess=R)
    # to get M(R) via one full inner Picard density iteration.
    def _M_at_R(R_query, density_seed=None, radii_seed=None):
        cp = dict(config_params)
        cp['max_iterations_outer'] = 1
        cp['_initial_radius_guess'] = float(R_query)
        # outer_solver knob would route us back here; force picard for
        # the inner _solve call so we hit the actual inner Picard.
        cp['outer_solver'] = 'picard'
        sub_result = _solve(
            cp,
            material_dictionaries,
            melting_curves_functions,
            input_dir,
            layer_mixtures=layer_mixtures,
            volatile_profile=volatile_profile,
            temperature_function=temperature_function,
            temperature_arrays=temperature_arrays,
            p_center_hint=p_center_hint,
            initial_density=density_seed if density_seed is not None else initial_density,
            initial_radii=radii_seed if radii_seed is not None else initial_radii,
        )
        mass_arr = np.asarray(sub_result.get('mass_enclosed', [np.nan]))
        M_val = float(mass_arr[-1]) if mass_arr.size > 0 else float('nan')
        return M_val, sub_result

    # ----- Newton iteration -----
    history = []  # list of (R, M, |f|/M_target) per outer iter
    last_result = None

    start_time = time.time()
    logger.info(
        'Newton outer: M_target=%.4e kg, R0=%.4e m, tol=%.1e, max_iter=%d',
        M_target, R, tol, max_iter,
    )

    for k in range(max_iter):
        M_k, last_result = _M_at_R(R)
        if not np.isfinite(M_k):
            raise RuntimeError(
                f'Newton iter {k}: M(R={R:.4e}) is not finite. '
                'Brentq fall-back not yet implemented (T2.1d).'
            )
        f_k = M_k - M_target
        rel = abs(f_k) / M_target
        history.append((R, M_k, rel))
        logger.info(
            'Newton iter %2d: R=%.5e M=%.5e f=%+.3e rel=%.3e',
            k, R, M_k, f_k, rel,
        )
        if rel < tol:
            logger.info(
                'Newton converged in %d iter(s): dM/M=%.3e <= tol=%.3e',
                k + 1, rel, tol,
            )
            # Attach Newton bookkeeping to the inner result.
            last_result['converged'] = True
            last_result['converged_mass'] = True
            last_result['best_mass_error'] = float(rel)
            last_result['total_time'] = time.time() - start_time
            last_result['newton_n_iter'] = k + 1
            last_result['newton_history'] = history
            return last_result

        # Central-difference dM/dR.
        dR = dR_rel * R
        M_plus, _ = _M_at_R(R + dR)
        M_minus, _ = _M_at_R(R - dR)
        if not (np.isfinite(M_plus) and np.isfinite(M_minus)):
            raise RuntimeError(
                f'Newton iter {k}: M(R+/-dR) not finite at R={R:.4e}. '
                'Brentq fall-back not yet implemented (T2.1d).'
            )
        dMdR = (M_plus - M_minus) / (2.0 * dR)

        # Reject vanishing derivative. M_target/earth_radius ~ 1e18 kg/m;
        # below 1e15 the slope is effectively zero and Newton step blows up.
        if abs(dMdR) < 1.0e15 or not np.isfinite(dMdR):
            raise RuntimeError(
                f'Newton iter {k}: |dM/dR|={dMdR:.3e} below 1e15. '
                'Brentq fall-back not yet implemented (T2.1d).'
            )

        R_new = R - f_k / dMdR
        # Clamp to physical bounds. Newton step OUTSIDE [R_min, R_max]
        # signals a bad derivative or a poor R0; route to brentq.
        if R_new < R_min or R_new > R_max:
            raise RuntimeError(
                f'Newton iter {k}: R_new={R_new:.4e} out of bounds [{R_min:.2e}, '
                f'{R_max:.2e}]. Brentq fall-back not yet implemented (T2.1d).'
            )

        # Trust-region: cap step.
        max_step = trust_region_frac * R
        step = R_new - R
        if abs(step) > max_step:
            step = float(np.sign(step) * max_step)
            R_new = R + step
            logger.debug(
                'Newton iter %d: step capped at %.0f%% of R (was %.3e -> %.3e).',
                k, trust_region_frac * 100, R - f_k / dMdR, R_new,
            )

        # Damping: if previous step did not reduce |f|, halve this one.
        if len(history) >= 2:
            f_prev = history[-2][1] - M_target
            if abs(f_k) >= abs(f_prev):
                R_new = R + 0.5 * (R_new - R)
                logger.debug(
                    'Newton iter %d: previous step did not improve |f|; '
                    'halving step to R=%.4e.', k, R_new,
                )

        R = R_new

    # Out of iters without convergence. Without brentq fall-back (T2.1d),
    # raise so caller knows.
    raise RuntimeError(
        f'Newton failed to converge in {max_iter} iter(s). '
        f'Best |dM/M| = {min(h[2] for h in history):.3e} at '
        f'R = {history[int(np.argmin([h[2] for h in history]))][0]:.4e} m. '
        'Brentq fall-back not yet implemented (T2.1d).'
    )


def _solve(
    config_params,
    material_dictionaries,
    melting_curves_functions,
    input_dir,
    layer_mixtures=None,
    volatile_profile=None,
    temperature_function=None,
    temperature_arrays=None,
    p_center_hint=None,
    initial_density=None,
    initial_radii=None,
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
    initial_density : numpy.ndarray or None, optional
        Density seed from a previous solve. Interpolated onto the current
        radial grid to accelerate Picard convergence.
    initial_radii : numpy.ndarray or None, optional
        Radial grid corresponding to ``initial_density``.

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
    cmb_temperature = config_params.get('cmb_temperature', None)
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
    # JAX fast path: when True, solve_structure dispatches to the diffrax-
    # based implementation (jax_eos/wrapper.py) inside the Picard loop.
    # Supported configs: 2-layer single-component (Stage-1b PROTEUS).
    # Unsupported configs fall back to the numpy path automatically.
    use_jax = bool(config_params.get('use_jax', False))
    # Anderson acceleration for the density Picard loop: when True,
    # replaces the damped fixed-point update (density = alpha * new + (1-alpha) * old)
    # with a Walker & Ni 2011 Type-II Anderson step that least-squares-combines
    # previous residuals. Falls back to damped Picard on numerical failure or
    # when history is shorter than one step.
    use_anderson = bool(config_params.get('use_anderson', False))
    anderson_m_max = int(config_params.get('anderson_m_max', 5))

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
    if temperature_mode in ('adiabatic', 'adiabatic_from_cmb'):
        max_reasonable_T_center = max(5.0 * surface_temperature, 3000.0)
        if center_temperature > max_reasonable_T_center:
            center_temperature = max_reasonable_T_center
            logger.debug(
                'Adiabatic mode: capped center_temperature initial guess '
                'to %.0f K (5x surface or 3000 K).', center_temperature,
            )

    # Wall-clock timeout: bail out with best solution if solver takes
    # too long (prevents indefinite hangs with volatile-extended mantles).
    wall_timeout = config_params.get('wall_timeout', 300.0)  # seconds
    wall_start = time.time()

    # Outer-loop oscillation tracking (local to this call, not function-level).
    best_mass_error = float('inf')
    best_profiles = None
    oscillation_count = 0

    # Initialize arrays to safe defaults. These are overwritten on the
    # first outer iteration, but must exist in case the wall-clock timeout
    # fires at iteration 0 (before the arrays are assigned in the loop).
    radii = np.linspace(0, radius_guess, num_layers)
    density = np.zeros(num_layers)
    mass_enclosed = np.zeros(num_layers)
    gravity = np.zeros(num_layers)
    pressure = np.zeros(num_layers)
    temperatures = np.full(num_layers, surface_temperature)

    # Solve the interior structure
    for outer_iter in range(max_iterations_outer):
        # Reset per-iteration convergence flags (prevent stale True from
        # a previous outer iteration masking failure in the current one).
        converged_pressure = False
        converged_density = False

        # Wall-clock timeout check
        if time.time() - wall_start > wall_timeout:
            logger.warning(
                'Wall-clock timeout (%.0fs) reached at outer iter %d. '
                'Returning best solution (mass error %.4f%%).',
                wall_timeout, outer_iter,
                best_mass_error * 100 if np.isfinite(best_mass_error) else float('inf'),
            )
            if best_profiles is not None:
                converged_mass = best_mass_error < 3 * tolerance_outer
                # Fresh writable copies: `pressure` / `mass_enclosed`
                # carry over from the previous outer iter and may be
                # read-only views of JAX buffers (jax_eos/wrapper.py
                # returns np.asarray(...) for speed; the host-device
                # sync penalty of a writable np.array on every solve
                # is 1 ms/call ~ 88 s/main() in coupled PROTEUS).
                # Converting only on the rare timeout path keeps the
                # hot loop fast.
                radii = np.array(best_profiles['radii'])
                density = np.array(best_profiles['density'])
                pressure = np.array(best_profiles['pressure'])
                mass_enclosed = np.array(best_profiles['mass_enclosed'])
                if best_profiles['temperatures'] is not None:
                    temperatures = np.array(best_profiles['temperatures'])
            break

        radii = np.linspace(0, radius_guess, num_layers)

        mass_enclosed = np.zeros(num_layers)
        gravity = np.zeros(num_layers)
        pressure = np.zeros(num_layers)

        # Density seeding: interpolate previous density onto current grid.
        # Only seed on the first outer iteration; subsequent iterations use
        # the Picard-updated density from the previous outer iteration.
        if (
            outer_iter == 0
            and initial_density is not None
            and initial_radii is not None
            and len(initial_density) > 1
            and np.any(initial_density > 0)
        ):
            # Linear interpolation onto new grid, clamped to [0, max_old]
            valid = initial_density > 0
            if np.sum(valid) > 1:
                density = np.interp(
                    radii,
                    initial_radii[valid],
                    initial_density[valid],
                    left=initial_density[valid][0],
                    right=0.0,
                )
                # Zero out shells beyond the old surface radius
                density[radii > initial_radii[valid][-1] * 1.05] = 0.0
                logger.info(
                    'Density seeded from previous solve: '
                    'rho_mean=%.0f kg/m^3, %d/%d shells seeded.',
                    np.mean(density[density > 0]),
                    np.sum(density > 0),
                    num_layers,
                )
            else:
                density = np.zeros(num_layers)
        else:
            density = np.zeros(num_layers)

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
            # For 'adiabatic_from_cmb', pass cmb_temperature so the linear
            # seed anchors at the CMB value rather than center_temperature,
            # giving the first density iteration a more realistic mantle T.
            _linear_mode = (
                'adiabatic_from_cmb' if temperature_mode == 'adiabatic_from_cmb' else 'linear'
            )
            _linear_tf = calculate_temperature_profile(
                radii,
                _linear_mode,
                surface_temperature,
                center_temperature,
                input_dir,
                temp_profile_file,
                cmb_temperature=cmb_temperature,
            )

            if _using_adiabat and prev_pressure is not None:
                # Bump blend toward full adiabat
                _adiabat_blend = min(1.0, _adiabat_blend + _ADIABAT_BLEND_STEP)
                logger.debug(
                    'Outer iter %d: adiabat blend = %.2f', outer_iter, _adiabat_blend,
                )

                # Recompute adiabat from previous iteration's converged
                # structure. For 'adiabatic_from_cmb', anchor at the CMB
                # and integrate outward; otherwise anchor at the surface
                # and integrate inward (original behaviour).
                _anchor = 'cmb' if temperature_mode == 'adiabatic_from_cmb' else 'surface'
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
                    anchor=_anchor,
                    cmb_temperature=cmb_temperature,
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
                        # W5: math.log10 avoids numpy scalar dispatch (~1 us/call)
                        T_adi = float(np.interp(math.log10(P), _lp, _ts))
                        T_adi = max(_T_MIN_CLAMP, min(T_adi, _T_MAX_CLAMP))
                        return (1.0 - _b) * T_lin + _b * T_adi

                else:

                    def _temperature_func(r, P, _lp=_logP_sorted, _ts=_T_sorted):
                        if P <= 0:
                            return surface_temperature
                        # W5: math.log10 avoids numpy scalar dispatch (~1 us/call)
                        T_val = float(np.interp(math.log10(P), _lp, _ts))
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

        # Adaptive Picard blending: start at 0.5, reduce if mass oscillates
        _picard_alpha = 0.5
        _prev_mass_error = None
        _prev_brent_solution = None  # Persist Brent bracket (Fix 3)
        _frozen_sigma = {}  # Freeze suppression weights after first iteration (Fix 2)

        # Inner-loop density oscillation tracking
        _inner_alpha = 0.5  # Separate damping for density Picard
        _inner_prev_diff = None  # Previous density change direction
        _inner_osc_count = 0  # Consecutive oscillation count
        _inner_best_diff = float('inf')
        _inner_best_density = None
        # Stuck-bail counter: how many inner iters since the best
        # max-residual last improved by at least _STUCK_REL_IMPROVE
        # (5%). On hot fully-molten profiles the existing 15-oscillation
        # early-bail at line ~1156 doesn't fire because best_diff stays
        # > 0.1, AND a strict "any improvement" stuck counter doesn't
        # work either because Picard makes slow incremental progress
        # (each iter trims best_diff by < 1%). The relative-improvement
        # criterion counts a marginal step toward best_diff as "stuck."
        # On a real CHILI iter 23869 (hot, T_surf=2820K) cProfile pre-
        # bail showed 7900+ inner iters per main() totalling 949 s; with
        # this bail the outer mass-radius loop iterates the structure
        # to recover convergence on the next outer pass.
        _inner_stuck_count = 0
        _inner_last_breakthrough_diff = float('inf')
        _STUCK_BAIL_LIMIT = 15
        _STUCK_REL_IMPROVE = 0.05  # need 5% drop to reset counter

        # Anderson acceleration history (only used when use_anderson=True).
        # Cleared on (a) shape change (n_valid changes between iters) and
        # (b) inner-loop oscillation (signals that current residual landscape
        # is not locally linear; damped Picard is safer until things settle).
        _anderson_x_hist = []
        _anderson_f_hist = []

        for inner_iter in range(max_iterations_inner):
            # Wall-clock timeout check (inner loop)
            if time.time() - wall_start > wall_timeout:
                if _inner_best_density is not None:
                    density[:] = _inner_best_density
                    converged_density = _inner_best_diff < max(100 * tolerance_inner, 0.1)
                logger.warning(
                    'Wall-clock timeout in inner loop (outer=%d, inner=%d).',
                    outer_iter, inner_iter,
                )
                break

            old_density = density.copy()

            # Central pressure estimate: use cached hint if available,
            # otherwise fall back to scaling law
            if p_center_hint is not None and p_center_hint > 0:
                pressure_guess = p_center_hint
            else:
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
                    use_jax=use_jax,
                    temperature_arrays=temperature_arrays,
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

            # Bracket: use previous solution to narrow the search (Fix 3).
            # If the narrow bracket from the previous solve does not
            # straddle the root (e.g. when Aragog hands a hot-shifted T(r)
            # that pushes the new root outside [0.5, 2.0] x P_prev),
            # widen progressively before falling back. Each widened
            # attempt costs one extra _pressure_residual call (= one
            # solve_structure). 2026-04-26: added to harden coupled-run
            # robustness on hot fully-molten profiles.
            if _prev_brent_solution is not None and _prev_brent_solution > 0:
                bracket_attempts = [
                    (max(1e6, 0.5 * _prev_brent_solution), 2.0 * _prev_brent_solution),
                    (max(1e6, 0.1 * _prev_brent_solution), 10.0 * _prev_brent_solution),
                    (max(1e6, 0.01 * _prev_brent_solution), 100.0 * _prev_brent_solution),
                ]
            else:
                bracket_attempts = [
                    (max(1e6, 0.1 * pressure_guess), 10.0 * pressure_guess),
                    (max(1e6, 0.01 * pressure_guess), 100.0 * pressure_guess),
                ]
            if uses_WB2018:
                bracket_attempts = [
                    (lo, min(hi, max_center_pressure_guess))
                    for lo, hi in bracket_attempts
                ]

            try:
                # Pre-validate that the bracket straddles the root.
                # This gives a clearer error than brentq's generic ValueError,
                # and the except handler below gracefully falls back to the
                # last evaluated solution.
                p_low = p_high = None
                f_low = f_high = None
                for _bi, (_pl, _ph) in enumerate(bracket_attempts):
                    _fl = _pressure_residual(_pl)
                    _fh = _pressure_residual(_ph)
                    if _fl * _fh <= 0:
                        p_low, p_high = _pl, _ph
                        f_low, f_high = _fl, _fh
                        if _bi > 0:
                            logger.debug(
                                'Bracket widened to [%.2e, %.2e] Pa on attempt %d',
                                _pl, _ph, _bi + 1,
                            )
                        break
                if p_low is None:
                    _last_pl, _last_ph = bracket_attempts[-1]
                    raise ValueError(
                        f'Brent bracket does not straddle the root after '
                        f'{len(bracket_attempts)} widening attempts: '
                        f'final f({_last_pl:.2e})={_fl:.2e}, '
                        f'f({_last_ph:.2e})={_fh:.2e}.'
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
                _prev_brent_solution = p_solution  # Persist for next iteration
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
                    use_jax=use_jax,
                    temperature_arrays=temperature_arrays,
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
                T_batch = (
                    T_arr[np.where(mask)[0]]
                    if len(T_arr) == len(valid_indices)
                    else T_arr[mask]
                )
                rho_batch = calculate_mixed_density_batch(
                    pressure[idx],
                    T_batch,
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

            # Note: sigma freezing (Fix 2) removed due to shape mismatch when
            # valid shell counts change between Picard iterations. The adaptive
            # Picard alpha (Fix 1) handles the oscillation suppression instead.

            # Fill NaN entries with last valid density (walking outward)
            last_valid = None
            for i in range(n_valid):
                if not p_valid[i]:
                    new_density[i] = 0.0
                elif np.isnan(new_density[i]):
                    new_density[i] = last_valid if last_valid is not None else old_density[i]
                else:
                    last_valid = new_density[i]

            # Adaptive Picard blend: use inner-loop alpha for density damping
            alpha = min(_picard_alpha, _inner_alpha)

            # Anderson acceleration attempt (opt-in via use_anderson).
            # Residual f_k = g(x_k) - x_k = new_density - old_density.
            # Clear history on shape change (n_valid differs from last iter).
            x_next_anderson = None
            if use_anderson:
                if _anderson_x_hist and len(_anderson_x_hist[-1]) != n_valid:
                    _anderson_x_hist.clear()
                    _anderson_f_hist.clear()
                f_k = new_density[:n_valid] - old_density[:n_valid]
                x_next_anderson = _anderson_mix(
                    _anderson_x_hist,
                    _anderson_f_hist,
                    old_density[:n_valid],
                    f_k,
                    m_max=anderson_m_max,
                    beta=1.0,
                )
                # Always push the current (x_k, f_k) pair to history,
                # regardless of whether the Anderson step itself succeeded,
                # so the next iter has fresh data.
                _anderson_x_hist.append(old_density[:n_valid].copy())
                _anderson_f_hist.append(f_k.copy())
                if len(_anderson_x_hist) > anderson_m_max:
                    _anderson_x_hist.pop(0)
                    _anderson_f_hist.pop(0)

            if x_next_anderson is not None:
                density[:n_valid] = x_next_anderson
            else:
                density[:n_valid] = (
                    alpha * new_density[:n_valid]
                    + (1.0 - alpha) * old_density[:n_valid]
                )

            # Check density convergence
            relative_diff_inner = np.max(
                np.abs((density - old_density) / (old_density + 1e-20))
            )

            # Inner-loop oscillation detection: track mean density change direction
            mean_change = np.mean(density[:n_valid] - old_density[:n_valid])
            if _inner_prev_diff is not None and mean_change * _inner_prev_diff < 0:
                _inner_osc_count += 1
                _inner_alpha = max(0.1, _inner_alpha * 0.6)
                if _inner_osc_count % 5 == 0:
                    logger.debug(
                        'Inner iter %d: density oscillation #%d, alpha=%.2f, diff=%.2e',
                        inner_iter, _inner_osc_count, _inner_alpha, relative_diff_inner,
                    )
                # Oscillation = local residual landscape not well-approximated
                # by the affine model Anderson assumes. Drop history and let
                # damped Picard take over until things settle.
                if use_anderson:
                    _anderson_x_hist.clear()
                    _anderson_f_hist.clear()
            else:
                _inner_osc_count = max(0, _inner_osc_count - 1)
                _inner_alpha = min(0.5, _inner_alpha * 1.05)
            _inner_prev_diff = mean_change

            # Track best density for bailout. Reset stuck counter only
            # on a "breakthrough" improvement (5% relative drop from the
            # last breakthrough). Marginal trim improvements still update
            # best_diff but do NOT reset the stuck counter — this is what
            # lets the bail fire on hot profiles where Picard slowly
            # creeps toward 0.1 over 100 iters but never breaks through.
            if relative_diff_inner < _inner_best_diff:
                _inner_best_diff = relative_diff_inner
                _inner_best_density = density.copy()
            if relative_diff_inner < (1.0 - _STUCK_REL_IMPROVE) * _inner_last_breakthrough_diff:
                _inner_last_breakthrough_diff = relative_diff_inner
                _inner_stuck_count = 0
            else:
                _inner_stuck_count += 1

            if relative_diff_inner < tolerance_inner:
                logger.debug(
                    'Inner loop converged after %d iterations.', inner_iter + 1,
                )
                converged_density = True
                break

            # Stuck-bail: no improvement in best max-residual for too
            # many iters. Hot fully-molten T(r) profiles (early CHILI
            # coupled iters) drive Picard into a state where best_diff
            # plateaus above 0.1 (so the oscillation-bail below cannot
            # fire) and the loop runs all 100 inner iters per outer.
            # Bailing after no improvement saves the bulk of the wall.
            if (
                _inner_stuck_count >= _STUCK_BAIL_LIMIT
                and _inner_best_density is not None
            ):
                logger.info(
                    'Inner loop: stuck-bail after %d iters w/o improvement '
                    '(best diff=%.2e, target=%.2e). Outer loop will iterate.',
                    _inner_stuck_count, _inner_best_diff, tolerance_inner,
                )
                density[:] = _inner_best_density
                converged_density = _inner_best_diff < max(100 * tolerance_inner, 0.1)
                break

            # Bailout: after many oscillations, accept best density.
            # Use absolute floor (0.1) so volatile-extended mantles with external
            # T(r) profiles (from Aragog coupling) can still converge. The outer
            # mass loop corrects residual density error; density self-consistency
            # improves across outer iterations.
            osc_threshold = max(10 * tolerance_inner, 0.1)
            if _inner_osc_count >= 15 and _inner_best_diff < osc_threshold:
                logger.info(
                    'Inner loop: accepting best density after %d oscillations '
                    '(best diff=%.2e, target=%.2e)',
                    _inner_osc_count, _inner_best_diff, tolerance_inner,
                )
                density[:] = _inner_best_density
                converged_density = True
                break

            if inner_iter == max_iterations_inner - 1:
                # If best density is within reasonable range, use it.
                # Absolute floor (0.1) ensures retry with external T(r) profiles
                # (volatile mantles from Aragog coupling) doesn't reject workable
                # solutions. The outer mass loop handles residual density error.
                maxiter_threshold = max(100 * tolerance_inner, 0.1)
                if _inner_best_diff < maxiter_threshold and _inner_best_density is not None:
                    logger.info(
                        'Inner loop: max iterations, using best density '
                        '(diff=%.2e, target=%.2e)',
                        _inner_best_diff, tolerance_inner,
                    )
                    density[:] = _inner_best_density
                    converged_density = True
                else:
                    logger.warning(
                        'Maximum inner iterations (%d) reached. '
                        'Best density diff=%.2e, threshold=%.2e. '
                        'Density not converged.', max_iterations_inner,
                        _inner_best_diff, maxiter_threshold,
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

        # Adaptive Picard alpha: detect mass oscillation (Fix 1)
        mass_error_signed = (calculated_mass - planet_mass) / planet_mass
        if _prev_mass_error is not None:
            if mass_error_signed * _prev_mass_error < 0:
                # Sign changed: oscillating. Reduce alpha (stronger damping)
                _picard_alpha = max(0.2, _picard_alpha * 0.7)
                logger.debug(
                    'Outer iter %d: mass oscillation detected, reducing Picard alpha to %.2f',
                    outer_iter, _picard_alpha,
                )
            elif relative_diff_outer_mass < abs(_prev_mass_error):
                # Converging monotonically: relax alpha slightly
                _picard_alpha = min(0.7, _picard_alpha * 1.1)
        _prev_mass_error = mass_error_signed

        # Track best solution for oscillation bailout (local variables,
        # reset per _solve() call to avoid stale state between calls)
        if relative_diff_outer_mass < best_mass_error:
            best_mass_error = relative_diff_outer_mass
            best_profiles = {
                'radii': radii.copy(), 'density': density.copy(),
                'gravity': np.asarray(gravity).copy() if gravity is not None else None,
                'pressure': np.asarray(pressure).copy(),
                'mass_enclosed': np.asarray(mass_enclosed).copy(),
                'temperatures': temperatures.copy() if temperatures is not None else None,
            }
            oscillation_count = 0
        elif _prev_mass_error is not None and mass_error_signed * _prev_mass_error < 0:
            oscillation_count += 1

        # Bailout: if oscillating for 10+ iterations, accept best solution
        if oscillation_count >= 10 and best_mass_error < 3 * tolerance_outer:
            logger.warning(
                'Accepting best solution after %d oscillations (mass error %.4f%%, '
                'target %.4f%%)',
                oscillation_count,
                best_mass_error * 100,
                tolerance_outer * 100,
            )
            # Restore best profiles. Fresh writable copies — pressure
            # and mass_enclosed may be read-only views of JAX buffers
            # from prior solve_structure returns (see wrapper.py note
            # on np.asarray vs np.array cost).
            radii = np.array(best_profiles['radii'])
            density = np.array(best_profiles['density'])
            pressure = np.array(best_profiles['pressure'])
            mass_enclosed = np.array(best_profiles['mass_enclosed'])
            if best_profiles['temperatures'] is not None:
                temperatures = np.array(best_profiles['temperatures'])
            converged_mass = True
            break

        # Reset frozen sigma for next outer iteration
        _frozen_sigma = {}

        # MASS CONVERGENCE CHECK
        # When temperature_mode='adiabatic' and the blend has not yet reached
        # 1.0, mass convergence triggers the adiabat transition instead of
        # breaking. The blend ramps 0 -> 0.5 -> 1.0 over successive mass
        # convergences, preventing solver divergence.
        if relative_diff_outer_mass < tolerance_outer:
            if temperature_mode in ('adiabatic', 'adiabatic_from_cmb') and _adiabat_blend < 1.0:
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
    elif converged_mass and best_mass_error < tolerance_outer:
        # Mass has converged to within tolerance. The density/pressure
        # Picard iteration may not reach strict convergence when EOS
        # tables have interpolation noise near their boundaries (e.g.,
        # PALEOS at high T), but the integrated structure (mass, radius)
        # is physically valid. Accept the solution.
        logger.info(
            'Accepting solution: mass converged (%.4f%%), '
            'density_converged=%s, pressure_converged=%s.',
            best_mass_error * 100, converged_density, converged_pressure,
        )
        converged = True
    elif best_mass_error < 0.03 and best_profiles is not None:
        # Timeout or max-iterations with < 3% mass error. The structure
        # is usable (radius accurate to < 1%, pressure to < 3%).
        # Better to return a slightly loose solution than crash the
        # calling code. The caller can check converged_mass/density/
        # pressure flags for diagnostics.
        logger.warning(
            'Accepting timeout solution: mass error %.4f%% (threshold 2%%), '
            'density_converged=%s, pressure_converged=%s.',
            best_mass_error * 100, converged_density, converged_pressure,
        )
        converged = True
        converged_mass = True

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
        'best_mass_error': float(best_mass_error) if np.isfinite(best_mass_error) else None,
        'p_center': pressure[0] if len(pressure) > 0 else None,
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
    temperature_arrays=None,
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
            temperature_arrays=temperature_arrays,
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
            temperature_arrays=temperature_arrays,
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


