"""JAX-native coupled_odes RHS for the 2-layer (core + mantle) planet.

Port of ``zalmoxis.structure_model.coupled_odes`` for the Stage-1b
configuration: a two-layer planet with PALEOS:iron (paleos_unified) core
and PALEOS-2phase:MgSiO3 (Tdep) mantle. Returns [dM/dr, dg/dr, dP/dr].

Layer dispatch is the only data-dependent branch. In numpy:

    if mass < cmb_mass:
        rho = paleos_unified(P, T, core_table, mzf_core)
    else:
        rho = tdep(P, T, T_sol, T_liq, solid_table, liquid_table)

In JAX this becomes: evaluate BOTH density functions, select the right
one with jnp.where. Both evaluations are cheap compiled kernels, so the
wasted work is small and the branching is trace-friendly.

Temperature is supplied via a pre-tabulated (log_P, T) array built by
the caller each Picard iteration (matching numpy's _temperature_func
closure). At query time we do jnp.interp on log10(P).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .paleos import get_paleos_unified_density_jax
from .tdep import get_tdep_density_jax


@jax.jit
def coupled_odes_jax(
    radius: jnp.ndarray,
    y: jnp.ndarray,          # shape (3,): [mass, gravity, pressure]
    cmb_mass: float,
    # --- temperature lookup (pre-tabulated log_P -> T) ---
    T_logP_grid: jnp.ndarray,  # monotone increasing log10(P)
    T_values: jnp.ndarray,     # matching T values
    T_surface: float,          # fallback when P <= 0
    # --- core (paleos_unified) table ---
    mushy_zone_factor_core: jnp.ndarray,
    core_density_grid: jnp.ndarray,
    core_unique_log_p: jnp.ndarray,
    core_unique_log_t: jnp.ndarray,
    core_logp_min: float,
    core_logt_min: float,
    core_dlog_p: float,
    core_dlog_t: float,
    core_n_p: int,
    core_n_t: int,
    core_p_min: float,
    core_p_max: float,
    core_lt_min_per_p: jnp.ndarray,
    core_lt_max_per_p: jnp.ndarray,
    core_liquidus_log_p: jnp.ndarray,
    core_liquidus_log_t: jnp.ndarray,
    core_liquidus_min_log_p: float,
    core_liquidus_max_log_p: float,
    core_has_liquidus_f: jnp.ndarray,
    # --- mantle Tdep (solid + liquid sub-tables + melting curves) ---
    sol_density_grid: jnp.ndarray,
    sol_unique_log_p: jnp.ndarray,
    sol_unique_log_t: jnp.ndarray,
    sol_logp_min: float,
    sol_logt_min: float,
    sol_dlog_p: float,
    sol_dlog_t: float,
    sol_n_p: int,
    sol_n_t: int,
    sol_p_min: float,
    sol_p_max: float,
    sol_lt_min_per_p: jnp.ndarray,
    sol_lt_max_per_p: jnp.ndarray,
    liq_density_grid: jnp.ndarray,
    liq_unique_log_p: jnp.ndarray,
    liq_unique_log_t: jnp.ndarray,
    liq_logp_min: float,
    liq_logt_min: float,
    liq_dlog_p: float,
    liq_dlog_t: float,
    liq_n_p: int,
    liq_n_t: int,
    liq_p_min: float,
    liq_p_max: float,
    liq_lt_min_per_p: jnp.ndarray,
    liq_lt_max_per_p: jnp.ndarray,
    # Stixrude14 melting-curve parameters (analytic power law, exact).
    # T_liq(P) = stix_T_ref * (P/stix_P_ref)^stix_exponent
    # T_sol(P) = T_liq(P) * stix_cryo_factor
    # For the standard Stage-1b Stixrude14 setup these are provided by the
    # caller from melting_curves.py; pre-tabulated grids introduce linear-
    # interp error ~O(1e-5) that drifts the RHS above parity tolerance.
    stix_T_ref: float = 5400.0,
    stix_P_ref: float = 140e9,
    stix_exponent: float = 0.480,
    stix_cryo_factor: float = 0.8086,  # matches melting_curves._STIX14_CRYO_FACTOR
    # physical constant G (SI)
    G: float = 6.6743e-11,
):
    """Return dy/dr = [dM/dr, dg/dr, dP/dr] at (radius, y)."""
    mass, gravity, pressure = y[0], y[1], y[2]

    # Temperature at this (r, P). Matches numpy _temperature_func closure
    # (fallback to surface T when P <= 0, log10 via jnp).
    log_p_for_T = jnp.log10(jnp.maximum(pressure, 1.0))
    T_interp = jnp.interp(log_p_for_T, T_logP_grid, T_values)
    temperature = jnp.where(pressure > 0, T_interp, T_surface)

    # Melting curves at this pressure via analytic Stixrude14 power law
    # (exact; avoids pre-tabulation interp error ~O(1e-5)).
    T_liq = jnp.where(
        pressure > 0,
        stix_T_ref * (pressure / stix_P_ref) ** stix_exponent,
        0.0,
    )
    T_sol = T_liq * stix_cryo_factor

    # Core density (paleos_unified)
    rho_core = get_paleos_unified_density_jax(
        pressure, temperature, mushy_zone_factor_core,
        core_density_grid, core_unique_log_p, core_unique_log_t,
        core_logp_min, core_logt_min, core_dlog_p, core_dlog_t,
        core_n_p, core_n_t, core_p_min, core_p_max,
        core_lt_min_per_p, core_lt_max_per_p,
        core_liquidus_log_p, core_liquidus_log_t,
        core_liquidus_min_log_p, core_liquidus_max_log_p,
        core_has_liquidus_f,
    )

    # Mantle density (Tdep 2-phase)
    rho_mantle = get_tdep_density_jax(
        pressure, temperature, T_sol, T_liq,
        sol_density_grid, sol_unique_log_p, sol_unique_log_t,
        sol_logp_min, sol_logt_min, sol_dlog_p, sol_dlog_t,
        sol_n_p, sol_n_t, sol_p_min, sol_p_max,
        sol_lt_min_per_p, sol_lt_max_per_p,
        liq_density_grid, liq_unique_log_p, liq_unique_log_t,
        liq_logp_min, liq_logt_min, liq_dlog_p, liq_dlog_t,
        liq_n_p, liq_n_t, liq_p_min, liq_p_max,
        liq_lt_min_per_p, liq_lt_max_per_p,
    )

    # Layer selection
    rho = jnp.where(mass < cmb_mass, rho_core, rho_mantle)

    # Numpy's coupled_odes returns zeros when P<=0 or density non-finite
    # to freeze the state until the terminal event fires. Mirror that.
    valid = (pressure > 0) & jnp.isfinite(rho)
    rho = jnp.where(valid, rho, 1.0)  # placeholder; output is zeroed below

    # Standard structure ODEs
    dMdr = 4.0 * jnp.pi * radius ** 2 * rho
    # At r=0, dgdr singular; use analytic limit dg/dr = 4 pi G rho / 3
    dgdr_gen = 4.0 * jnp.pi * G * rho - 2.0 * gravity / jnp.where(radius > 0, radius, 1.0)
    dgdr_r0 = (4.0 / 3.0) * jnp.pi * G * rho
    dgdr = jnp.where(radius > 0, dgdr_gen, dgdr_r0)
    dPdr = -rho * gravity

    # Zero derivatives if invalid (freeze state for terminal event)
    zero3 = jnp.zeros(3, dtype=dMdr.dtype)
    return jnp.where(valid, jnp.stack([dMdr, dgdr, dPdr]), zero3)
