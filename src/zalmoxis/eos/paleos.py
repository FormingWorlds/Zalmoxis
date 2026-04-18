"""Unified PALEOS density and nabla_ad lookups.

Handles single-file-per-material PALEOS tables where all stable phases
are encoded in one file with a phase column.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from .interpolation import (
    _ensure_unified_cache,
    _fast_bilinear,
    _paleos_clamp_temperature,
    _paleos_clamp_warned,
    fast_bilinear_batch,
)

logger = logging.getLogger(__name__)

# Phase-boundary guard offset (K).  When evaluating endpoint properties in
# the mushy zone, solid-side queries are clamped to at most T_melt - dT and
# liquid-side to at least T_melt + dT.  This prevents cross-phase lookups
# when external solidus/liquidus curves disagree with PALEOS's own melting
# curve at high pressures (plausible for super-Earth conditions).
_DT_PHASE_GUARD = 1.0

_paleos_phase_guard_warned = set()


def get_paleos_unified_density(
    pressure, temperature, material_dict, mushy_zone_factor, interpolation_functions
):
    """Look up density from a unified PALEOS table.

    When ``mushy_zone_factor == 1.0`` (no mushy zone), the density is read
    directly from the table (the stable phase at each (P, T) is already
    encoded). When ``mushy_zone_factor < 1.0``, a synthetic solidus is
    derived as ``T_sol = T_liq * mushy_zone_factor`` and the density in the
    mushy zone is volume-averaged between the solid-side and liquid-side
    table values.

    Parameters
    ----------
    pressure : float
        Pressure in Pa.
    temperature : float
        Temperature in K.
    material_dict : dict
        Material properties dict with 'eos_file' and 'format' keys.
    mushy_zone_factor : float
        Cryoscopic depression factor. 1.0 = no mushy zone (sharp boundary).
        < 1.0 = solidus at this fraction of the extracted liquidus.
    interpolation_functions : dict
        Shared interpolation cache.

    Returns
    -------
    float or None
        Density in kg/m^3, or None on failure.
    """
    eos_file = material_dict['eos_file']
    try:
        cached = _ensure_unified_cache(eos_file, interpolation_functions)

        p_min, p_max = cached['p_min'], cached['p_max']
        if pressure < p_min:
            pressure = p_min
        elif pressure > p_max:
            pressure = p_max

        log_p = np.log10(pressure)
        log_t = np.log10(temperature if temperature > 1.0 else 1.0)

        # Per-cell clamping
        log_t_clamped, was_clamped = _paleos_clamp_temperature(log_p, log_t, cached)
        if was_clamped and eos_file not in _paleos_clamp_warned:
            _paleos_clamp_warned.add(eos_file)
            logger.warning(
                f'PALEOS unified per-cell clamping active for '
                f'{os.path.basename(eos_file)}: '
                f'T={temperature:.0f} K clamped to {10.0**log_t_clamped:.0f} K '
                f'at P={pressure:.2e} Pa.'
            )

        if mushy_zone_factor >= 1.0 or len(cached['liquidus_log_p']) == 0:
            # Direct lookup: no mushy zone (fast bilinear path)
            density = _fast_bilinear(log_p, log_t_clamped, cached['density_grid'], cached)
            if density != density:  # fast NaN check (NaN != NaN)
                density = float(cached['density_nn']((log_p, log_t_clamped)))
            return density if density == density else None

        # Mushy zone: interpolate liquidus T at this P.
        # If query pressure is outside the liquidus coverage (e.g. at
        # pressures where no liquid phase exists), fall back to direct lookup.
        liq_lp = cached['liquidus_log_p']
        if log_p < liq_lp[0] or log_p > liq_lp[-1]:
            density = _fast_bilinear(log_p, log_t_clamped, cached['density_grid'], cached)
            if not np.isfinite(density):
                density = float(cached['density_nn']((log_p, log_t_clamped)))
            return density if np.isfinite(density) else None

        # PALEOS's own melting curve at this pressure
        log_t_melt = float(np.interp(log_p, liq_lp, cached['liquidus_log_t']))
        T_melt = 10.0**log_t_melt

        # Derive mushy zone boundaries (currently from PALEOS liquidus,
        # but may come from external melting curves in future).
        T_liq = T_melt
        T_sol = T_liq * mushy_zone_factor

        # Clamp endpoints against PALEOS's internal phase boundary so that
        # solid-side queries never land on the liquid side and vice versa.
        # The guard offset (_DT_PHASE_GUARD) always shifts T_liq up and may
        # shift T_sol down; only warn when the clamp corrects a genuine
        # cross-boundary incursion (T_sol above T_melt or T_liq below it).
        sol_crossed = T_sol > T_melt
        liq_crossed = T_liq < T_melt
        T_sol = min(T_sol, T_melt - _DT_PHASE_GUARD)
        T_liq = max(T_liq, T_melt + _DT_PHASE_GUARD)

        if (sol_crossed or liq_crossed):
            if eos_file not in _paleos_phase_guard_warned:
                _paleos_phase_guard_warned.add(eos_file)
                logger.warning(
                    'Mushy zone endpoints crossed PALEOS melting curve '
                    'for %s at P=%.2e Pa (T_melt=%.1f K): '
                    'T_sol=%.1f K %s, T_liq=%.1f K %s. '
                    'Clamped to safe side.',
                    os.path.basename(eos_file),
                    pressure,
                    T_melt,
                    T_sol,
                    '(was above T_melt)' if sol_crossed else '(ok)',
                    T_liq,
                    '(was below T_melt)' if liq_crossed else '(ok)',
                )
        log_t_sol = np.log10(max(T_sol, 1.0))
        log_t_liq = np.log10(T_liq)

        if temperature >= T_liq:
            # Above liquidus: direct lookup
            density = _fast_bilinear(log_p, log_t_clamped, cached['density_grid'], cached)
            if not np.isfinite(density):
                density = float(cached['density_nn']((log_p, log_t_clamped)))
            return density if np.isfinite(density) else None

        if temperature <= T_sol:
            # Below solidus: direct lookup
            density = _fast_bilinear(log_p, log_t_clamped, cached['density_grid'], cached)
            if not np.isfinite(density):
                density = float(cached['density_nn']((log_p, log_t_clamped)))
            return density if np.isfinite(density) else None

        # In mushy zone: volume-average between solid-side and liquid-side
        phi = (temperature - T_sol) / (T_liq - T_sol)

        # Solid-side: density at T_sol
        log_t_sol_c, _ = _paleos_clamp_temperature(log_p, log_t_sol, cached)
        rho_sol = _fast_bilinear(log_p, log_t_sol_c, cached['density_grid'], cached)
        if not np.isfinite(rho_sol):
            rho_sol = float(cached['density_nn']((log_p, log_t_sol_c)))

        # Liquid-side: density at T_liq
        log_t_liq_c, _ = _paleos_clamp_temperature(log_p, log_t_liq, cached)
        rho_liq = _fast_bilinear(log_p, log_t_liq_c, cached['density_grid'], cached)
        if not np.isfinite(rho_liq):
            rho_liq = float(cached['density_nn']((log_p, log_t_liq_c)))

        if not (np.isfinite(rho_sol) and np.isfinite(rho_liq)):
            return None

        # Volume additivity
        specific_volume = phi * (1.0 / rho_liq) + (1.0 - phi) * (1.0 / rho_sol)
        return 1.0 / specific_volume

    except Exception as e:
        logger.error(
            f'Error in PALEOS unified density at P={pressure:.2e} Pa, '
            f'T={temperature:.1f} K: {e}'
        )
        return None


def get_paleos_unified_density_batch(
    pressures, temperatures, material_dict, mushy_zone_factor, interpolation_functions
):
    """Vectorized density lookup from a unified PALEOS table.

    Parameters
    ----------
    pressures : numpy.ndarray
        1D array of pressures in Pa.
    temperatures : numpy.ndarray
        1D array of temperatures in K.
    material_dict : dict
        Material properties dict with 'eos_file' and 'format' keys.
    mushy_zone_factor : float
        Cryoscopic depression factor. 1.0 = no mushy zone.
    interpolation_functions : dict
        Shared interpolation cache.

    Returns
    -------
    numpy.ndarray
        1D array of densities in kg/m^3. NaN where lookup fails.
    """
    eos_file = material_dict['eos_file']
    cached = _ensure_unified_cache(eos_file, interpolation_functions)

    p_clamped = np.clip(pressures, cached['p_min'], cached['p_max'])
    log_p = np.log10(p_clamped)
    log_t = np.log10(np.maximum(temperatures, 1.0))

    # Per-cell clamping (vectorized)
    ulp = cached['unique_log_p']
    lt_min = cached['logt_valid_min']
    lt_max = cached['logt_valid_max']
    local_tmin = np.interp(log_p, ulp, lt_min)
    local_tmax = np.interp(log_p, ulp, lt_max)

    valid_bounds = np.isfinite(local_tmin) & np.isfinite(local_tmax)
    log_t_clamped = log_t.copy()
    log_t_clamped = np.where(valid_bounds & (log_t < local_tmin), local_tmin, log_t_clamped)
    log_t_clamped = np.where(valid_bounds & (log_t > local_tmax), local_tmax, log_t_clamped)

    if mushy_zone_factor >= 1.0 or len(cached['liquidus_log_p']) == 0:
        # Direct lookup: fast vectorized bilinear interpolation
        result = fast_bilinear_batch(log_p, log_t_clamped, cached['density_grid'], cached)
        # NN fallback for NaN entries
        nan_mask = ~np.isfinite(result)
        if np.any(nan_mask):
            pts_nn = np.column_stack([log_p[nan_mask], log_t_clamped[nan_mask]])
            result[nan_mask] = cached['density_nn'](pts_nn)
        return result

    # Mushy zone path (vectorized).
    # Compute T_melt from the PALEOS analytic melting curve rather than
    # interpolating the extracted liquidus grid. This is faster and avoids
    # branching per-element for the "outside liquidus coverage" case.
    from ..melting_curves import paleos_liquidus

    T_melt = paleos_liquidus(pressures)

    # Derive mushy zone boundaries (currently from PALEOS liquidus,
    # but may come from external melting curves in future).
    T_liq = T_melt.copy()
    T_sol = T_liq * mushy_zone_factor

    # Clamp endpoints against PALEOS's own melting curve
    T_sol = np.minimum(T_sol, T_melt - _DT_PHASE_GUARD)
    T_liq = np.maximum(T_liq, T_melt + _DT_PHASE_GUARD)

    log_t_sol = np.log10(np.maximum(T_sol, 1.0))
    log_t_liq = np.log10(np.maximum(T_liq, 1.0))

    # Classify shells: above liquidus, below solidus, or in mushy zone
    above = temperatures >= T_liq
    below = temperatures <= T_sol
    mushy = ~above & ~below

    # Direct lookup for above-liquidus and below-solidus shells
    result = fast_bilinear_batch(log_p, log_t_clamped, cached['density_grid'], cached)
    nan_mask = ~np.isfinite(result)
    if np.any(nan_mask):
        pts_nn = np.column_stack([log_p[nan_mask], log_t_clamped[nan_mask]])
        result[nan_mask] = cached['density_nn'](pts_nn)

    # Mushy zone shells: volume-average between solid-side and liquid-side
    if np.any(mushy):
        m_idx = np.where(mushy)[0]
        phi = (temperatures[m_idx] - T_sol[m_idx]) / (T_liq[m_idx] - T_sol[m_idx])

        # Solid-side density at T_sol
        log_t_sol_c = log_t_sol[m_idx].copy()
        sol_tmin = np.interp(log_p[m_idx], ulp, lt_min)
        sol_tmax = np.interp(log_p[m_idx], ulp, lt_max)
        sol_valid = np.isfinite(sol_tmin) & np.isfinite(sol_tmax)
        log_t_sol_c = np.where(sol_valid & (log_t_sol_c < sol_tmin), sol_tmin, log_t_sol_c)
        log_t_sol_c = np.where(sol_valid & (log_t_sol_c > sol_tmax), sol_tmax, log_t_sol_c)
        rho_sol = fast_bilinear_batch(log_p[m_idx], log_t_sol_c, cached['density_grid'], cached)
        nn_sol = ~np.isfinite(rho_sol)
        if np.any(nn_sol):
            pts_sol_nn = np.column_stack([log_p[m_idx][nn_sol], log_t_sol_c[nn_sol]])
            rho_sol[nn_sol] = cached['density_nn'](pts_sol_nn)

        # Liquid-side density at T_liq
        log_t_liq_c = log_t_liq[m_idx].copy()
        liq_tmin = np.interp(log_p[m_idx], ulp, lt_min)
        liq_tmax = np.interp(log_p[m_idx], ulp, lt_max)
        liq_valid = np.isfinite(liq_tmin) & np.isfinite(liq_tmax)
        log_t_liq_c = np.where(liq_valid & (log_t_liq_c < liq_tmin), liq_tmin, log_t_liq_c)
        log_t_liq_c = np.where(liq_valid & (log_t_liq_c > liq_tmax), liq_tmax, log_t_liq_c)
        rho_liq = fast_bilinear_batch(log_p[m_idx], log_t_liq_c, cached['density_grid'], cached)
        nn_liq = ~np.isfinite(rho_liq)
        if np.any(nn_liq):
            pts_liq_nn = np.column_stack([log_p[m_idx][nn_liq], log_t_liq_c[nn_liq]])
            rho_liq[nn_liq] = cached['density_nn'](pts_liq_nn)

        # Volume additivity
        both_ok = np.isfinite(rho_sol) & np.isfinite(rho_liq) & (rho_sol > 0) & (rho_liq > 0)
        spec_vol = phi * (1.0 / np.where(both_ok, rho_liq, 1.0)) + (1.0 - phi) * (
            1.0 / np.where(both_ok, rho_sol, 1.0)
        )
        result[m_idx] = np.where(both_ok, 1.0 / spec_vol, np.nan)

    return result


def _get_paleos_unified_nabla_ad(pressure, temperature, material_dict, interpolation_functions):
    """Look up nabla_ad from a unified PALEOS cache entry.

    Parameters
    ----------
    pressure : float
        Pressure in Pa.
    temperature : float
        Temperature in K.
    material_dict : dict
        Material properties dict with 'eos_file' key.
    interpolation_functions : dict
        Shared interpolation cache.

    Returns
    -------
    float or None
        Dimensionless adiabatic gradient, or None if lookup fails.
    """
    eos_file = material_dict['eos_file']
    cached = _ensure_unified_cache(eos_file, interpolation_functions)

    p_clamped = np.clip(pressure, cached['p_min'], cached['p_max'])
    log_p = np.log10(p_clamped)
    log_t = np.log10(max(temperature, 1.0))

    log_t_clamped, was_clamped = _paleos_clamp_temperature(log_p, log_t, cached)
    if was_clamped and eos_file not in _paleos_clamp_warned:
        _paleos_clamp_warned.add(eos_file)
        logger.warning(
            f'PALEOS unified per-cell clamping active for nabla_ad in '
            f'{os.path.basename(eos_file)}: '
            f'T={temperature:.0f} K clamped to {10.0**log_t_clamped:.0f} K '
            f'at P={pressure:.2e} Pa.'
        )

    val = _fast_bilinear(log_p, log_t_clamped, cached['nabla_ad_grid'], cached)
    if not np.isfinite(val):
        val = float(cached['nabla_ad_nn']((log_p, log_t_clamped)))

    return val if np.isfinite(val) else None
