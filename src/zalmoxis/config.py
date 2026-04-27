"""Configuration loading, parsing, and validation for Zalmoxis.

Handles TOML config file selection, EOS config parsing (new and legacy formats),
parameter validation, material dictionary loading, and melting curve setup.
"""

from __future__ import annotations

import logging
import os
import sys

import toml

from . import get_zalmoxis_root
from .constants import (
    CONDENSED_RHO_MIN_DEFAULT,
    CONDENSED_RHO_SCALE_DEFAULT,
    TDEP_EOS_NAMES,
    earth_mass,
)
from .eos import get_solidus_liquidus_functions
from .eos_analytic import VALID_MATERIAL_KEYS
from .eos_properties import EOS_REGISTRY
from .eos_vinet import VALID_VINET_KEYS
from .mixing import (
    BINODAL_T_SCALE_DEFAULT,
    parse_layer_components,
)

logger = logging.getLogger(__name__)

# Mapping from legacy global EOS choice strings to per-layer config dicts
LEGACY_EOS_MAP = {
    'Tabulated:iron/silicate': {
        'core': 'Seager2007:iron',
        'mantle': 'Seager2007:MgSiO3',
    },
    'Tabulated:iron/Tdep_silicate': {
        'core': 'Seager2007:iron',
        'mantle': 'WolfBower2018:MgSiO3',
    },
    'Tabulated:water': {
        'core': 'Seager2007:iron',
        'mantle': 'Seager2007:MgSiO3',
        'ice_layer': 'Seager2007:H2O',
    },
}

VALID_TABULATED_EOS = {
    'Seager2007:iron',
    'Seager2007:MgSiO3',
    'WolfBower2018:MgSiO3',
    'RTPress100TPa:MgSiO3',
    'PALEOS-2phase:MgSiO3',
    'PALEOS-2phase:MgSiO3-highres',
    'Seager2007:H2O',
    'PALEOS:iron',
    'PALEOS:MgSiO3',
    'PALEOS:H2O',
    'Chabrier:H',
}

# WolfBower2018 tables are valid up to ~1 TPa. The Brent pressure solver with
# out-of-bounds clamping handles mantle pressures exceeding the table boundary
# for planets up to ~7 M_earth. Beyond this mass, deep-mantle pressures are far
# enough above the table ceiling that clamped densities become unreliable.
WOLFBOWER2018_MAX_MASS_EARTH = 7.0

# RTPress100TPa melt table extends to 100 TPa, matching the Seager2007 iron
# range. The solid table is still WolfBower2018 (1 TPa, clamped), but at high
# temperatures the mantle is predominantly molten, so the solid table limitation
# is less constraining. Safe up to ~50 M_earth.
RTPRESS100TPA_MAX_MASS_EARTH = 50.0

# PALEOS MgSiO3 tables extend to 100 TPa for both solid and liquid phases.
# Safe up to ~50 M_earth (same pressure ceiling as RTPress100TPa).
PALEOS_MAX_MASS_EARTH = 50.0

# Unified PALEOS tables (iron, MgSiO3, H2O) extend to 100 TPa.
PALEOS_UNIFIED_MAX_MASS_EARTH = 50.0


def parse_eos_config(eos_section):
    """Parse [EOS] TOML section into per-layer EOS dict.

    Info
    -----
    Supports new per-layer format (core/mantle/ice_layer fields) and legacy
    format (choice field) for backward compatibility.

    Parameters
    ----------
    eos_section : dict
        The [EOS] section from the TOML config.

    Returns
    -------
    dict
        Per-layer EOS config, e.g.
        {"core": "Seager2007:iron", "mantle": "WolfBower2018:MgSiO3"}.
    """
    # New format: per-layer fields present
    if 'core' in eos_section:
        if 'mantle' not in eos_section:
            raise ValueError(
                "EOS config has 'core' but missing 'mantle'. "
                "Both 'core' and 'mantle' are required."
            )
        layer_eos = {
            'core': eos_section['core'],
            'mantle': eos_section['mantle'],
        }
        water = eos_section.get('ice_layer', '')
        if water:
            layer_eos['ice_layer'] = water
        return layer_eos

    # Legacy format: expand 'choice' field
    choice = eos_section.get('choice', '')
    if choice in LEGACY_EOS_MAP:
        return dict(LEGACY_EOS_MAP[choice])

    if choice == 'Analytic:Seager2007':
        layer_eos = {
            'core': f'Analytic:{eos_section.get("core_material", "iron")}',
            'mantle': f'Analytic:{eos_section.get("mantle_material", "MgSiO3")}',
        }
        water_mat = eos_section.get('water_layer_material', '')
        if water_mat:
            layer_eos['ice_layer'] = f'Analytic:{water_mat}'
        return layer_eos

    raise ValueError(
        f"Unknown EOS config. Set per-layer fields (core, mantle) or legacy 'choice'. "
        f'Got: {eos_section}'
    )


def validate_layer_eos(layer_eos_config):
    """Validate all per-layer EOS strings, including multi-material mixtures.

    Parameters
    ----------
    layer_eos_config : dict
        Per-layer EOS config from parse_eos_config().

    Raises
    ------
    ValueError
        If any layer EOS string or component is invalid.
    """
    for layer, eos_str in layer_eos_config.items():
        mixture = parse_layer_components(eos_str)
        for comp in mixture.components:
            if comp in VALID_TABULATED_EOS:
                continue
            if comp.startswith('Analytic:'):
                material_key = comp.split(':', 1)[1]
                if material_key not in VALID_MATERIAL_KEYS:
                    raise ValueError(
                        f"Invalid analytic material '{material_key}' "
                        f"in layer '{layer}'. "
                        f'Valid keys: {sorted(VALID_MATERIAL_KEYS)}'
                    )
                continue
            if comp.startswith('Vinet:'):
                material_key = comp.split(':', 1)[1]
                if material_key not in VALID_VINET_KEYS:
                    raise ValueError(
                        f"Invalid Vinet material '{material_key}' "
                        f"in layer '{layer}'. "
                        f'Valid keys: {sorted(VALID_VINET_KEYS)}'
                    )
                continue
            raise ValueError(
                f"Invalid EOS component '{comp}' in layer '{layer}'. "
                f'Valid tabulated: {sorted(VALID_TABULATED_EOS)}. '
                f'Valid analytic: Analytic:<material> with '
                f'{sorted(VALID_MATERIAL_KEYS)}. '
                f'Valid Vinet: Vinet:<material> with '
                f'{sorted(VALID_VINET_KEYS)}.'
            )


def validate_config(config_params):
    """Validate all configuration parameters for physical and logical consistency.

    Checks parameter types, ranges, and cross-parameter constraints. Raises
    ValueError with a clear message explaining the problem and how to fix it,
    rather than letting the solver produce incorrect results.

    Parameters
    ----------
    config_params : dict
        Configuration parameters from load_zalmoxis_config().

    Raises
    ------
    ValueError
        If any parameter is invalid or parameters are incompatible.
    """
    # ── Planet mass ──────────────────────────────────────────────────
    planet_mass = config_params['planet_mass']
    planet_mass_mearth = planet_mass / earth_mass
    if planet_mass <= 0:
        raise ValueError(
            f'planet_mass must be positive, got {planet_mass_mearth:.4f} M_earth. '
            f'Set planet_mass > 0 in [InputParameter].'
        )

    if planet_mass_mearth < 0.1:
        logger.warning(
            f'planet_mass = {planet_mass_mearth:.4f} M_earth is below the validated '
            f'range (0.1-50 M_earth). Results may be unreliable at very low masses.'
        )
    if planet_mass_mearth > 50:
        logger.warning(
            f'planet_mass = {planet_mass_mearth:.1f} M_earth exceeds the validated '
            f'range (0.1-50 M_earth). PALEOS tables extend to 100 TPa '
            f'(~50 M_earth). Beyond this, EOS extrapolation may be unreliable.'
        )

    # ── Mass fractions ──────────────────────────────────────────────
    cmf = config_params['core_mass_fraction']
    mmf = config_params['mantle_mass_fraction']
    layer_eos_config = config_params['layer_eos_config']

    if cmf <= 0 or cmf > 1:
        raise ValueError(
            f'core_mass_fraction must be in (0, 1], got {cmf}. '
            f'A planet requires a nonzero core.'
        )

    if mmf < 0 or mmf >= 1:
        raise ValueError(
            f'mantle_mass_fraction must be in [0, 1), got {mmf}. '
            f'Use 0 for a 2-layer model where the mantle fills the remainder.'
        )

    if cmf + mmf > 1.0:
        raise ValueError(
            f'core_mass_fraction ({cmf}) + mantle_mass_fraction ({mmf}) = {cmf + mmf} > 1.0. '
            f'The sum of mass fractions cannot exceed 1.'
        )

    # 3-layer model requires mantle_mass_fraction > 0
    has_ice = 'ice_layer' in layer_eos_config
    if has_ice and mmf <= 0:
        raise ValueError(
            f'3-layer model (ice_layer = "{layer_eos_config["ice_layer"]}") requires '
            f'mantle_mass_fraction > 0, but got {mmf}. '
            f'Set mantle_mass_fraction > 0 in [AssumptionsAndInitialGuesses], '
            f'or remove ice_layer from [EOS] for a 2-layer model.'
        )

    if not has_ice and mmf > 0:
        raise ValueError(
            f'mantle_mass_fraction = {mmf} > 0 but no ice_layer EOS is set. '
            f'For a 2-layer model, set mantle_mass_fraction = 0 '
            f'(mantle fills the remainder 1 - core_mass_fraction). '
            f'For a 3-layer model, set ice_layer to a valid EOS string.'
        )

    # 3-layer models with H2O ice at high surface temperature
    if has_ice:
        ice_eos = layer_eos_config.get('ice_layer', '')
        ice_has_h2o = 'H2O' in ice_eos
        temperature_mode_raw = config_params['temperature_mode']
        surface_temp_raw = config_params['surface_temperature']
        if ice_has_h2o and temperature_mode_raw != 'isothermal' and surface_temp_raw >= 647:
            ice_frac = 1.0 - cmf - mmf
            raise ValueError(
                f'3-layer model with H2O ice layer at surface_temperature = '
                f'{surface_temp_raw} K >= 647 K (H2O critical point). '
                f'Pure H2O at T >= T_crit is vapor/supercritical at low pressure '
                f'and cannot support a hydrostatic ice shell. '
                f'The solver will diverge to unphysical radii '
                f'(ice fraction = {ice_frac:.1%}). '
                f'Options: (1) use isothermal mode with T_surf < 647 K for '
                f'3-layer models, or (2) represent water as a mixing component '
                f'in the mantle (e.g., mantle = "PALEOS:MgSiO3:0.85+PALEOS:H2O:0.15") '
                f'where the phase-aware suppression handles the vapor phase.'
            )

    # ── Temperature parameters ──────────────────────────────────────
    temperature_mode = config_params['temperature_mode']
    valid_modes = ('isothermal', 'linear', 'prescribed', 'adiabatic', 'adiabatic_from_cmb')
    if temperature_mode not in valid_modes:
        raise ValueError(
            f"Unknown temperature_mode '{temperature_mode}'. Valid options: {valid_modes}."
        )

    surface_temp = config_params['surface_temperature']
    center_temp = config_params['center_temperature']

    if surface_temp <= 0:
        raise ValueError(
            f'surface_temperature must be positive, got {surface_temp} K. '
            f'Temperatures must be in Kelvin.'
        )

    if surface_temp > 5000:
        logger.warning(
            f'surface_temperature = {surface_temp} K exceeds the validated range '
            f'(300-5000 K). The PALEOS tables extend to 100,000 K, but the '
            f'adiabatic solver has only been validated up to 5000 K surface temperature.'
        )

    if temperature_mode in ('linear', 'adiabatic', 'adiabatic_from_cmb') and center_temp <= 0:
        raise ValueError(
            f"center_temperature must be positive for '{temperature_mode}' mode, "
            f'got {center_temp} K. Temperatures must be in Kelvin.'
        )

    if temperature_mode == 'adiabatic_from_cmb':
        cmb_temp = config_params.get('cmb_temperature', 0)
        if cmb_temp <= 0:
            raise ValueError(
                "cmb_temperature must be positive (in K) for 'adiabatic_from_cmb' mode, "
                f'got {cmb_temp}. Set planet.tcmb_init in PROTEUS config.'
            )

    if temperature_mode == 'linear' and center_temp <= surface_temp:
        logger.warning(
            f'center_temperature ({center_temp} K) <= surface_temperature ({surface_temp} K) '
            f'in linear mode. The temperature gradient will be zero or negative '
            f'(temperature decreases inward). This is physically unusual.'
        )

    # Adiabatic mode requires at least one T-dependent EOS component
    all_components = set()
    for v in layer_eos_config.values():
        if v:
            m = parse_layer_components(v)
            all_components.update(m.components)
    uses_Tdep = bool(all_components & TDEP_EOS_NAMES)

    if temperature_mode in ('adiabatic', 'adiabatic_from_cmb') and not uses_Tdep:
        raise ValueError(
            f"'{temperature_mode}' temperature mode requires at least one T-dependent EOS layer, "
            f'but none found in {layer_eos_config}. '
            f'T-dependent EOS options: {sorted(TDEP_EOS_NAMES)}. '
            f"Use 'isothermal' or 'linear' mode with T-independent EOS."
        )

    # ── Mushy zone factor ───────────────────────────────────────────
    mushy_zone_factor = config_params.get('mushy_zone_factor', 1.0)

    if mushy_zone_factor < 0 or mushy_zone_factor > 1.0:
        raise ValueError(
            f'mushy_zone_factor must be in [0, 1.0], got {mushy_zone_factor}. '
            f'1.0 = no mushy zone (sharp phase boundary). '
            f'< 1.0 = solidus at this fraction of the liquidus temperature.'
        )

    if mushy_zone_factor < 0.7:
        raise ValueError(
            f'mushy_zone_factor = {mushy_zone_factor} is below the minimum of 0.7. '
            f'Values below 0.7 produce unphysically wide mushy zones that can '
            f'cause solver instabilities. Use a value in [0.7, 1.0].'
        )

    # mushy_zone_factor < 1.0 only makes sense with unified PALEOS tables
    has_unified_paleos = bool(all_components & {'PALEOS:iron', 'PALEOS:MgSiO3', 'PALEOS:H2O'})
    if mushy_zone_factor < 1.0 and not has_unified_paleos:
        raise ValueError(
            f'mushy_zone_factor = {mushy_zone_factor} < 1.0 but no unified PALEOS '
            f'EOS is configured. The mushy zone factor only applies to unified '
            f'PALEOS tables (PALEOS:iron, PALEOS:MgSiO3, PALEOS:H2O). '
            f'For PALEOS-2phase or WolfBower2018, phase routing is controlled '
            f'by the rock_solidus/rock_liquidus melting curves instead.'
        )

    # ── Per-EOS mushy zone factors ──────────────────────────────────
    mushy_zone_factors = config_params.get('mushy_zone_factors', {})
    _eos_to_key = {
        'PALEOS:iron': 'mushy_zone_factor_iron',
        'PALEOS:MgSiO3': 'mushy_zone_factor_MgSiO3',
        'PALEOS:H2O': 'mushy_zone_factor_H2O',
    }
    for eos_name, config_key in _eos_to_key.items():
        mzf = mushy_zone_factors.get(eos_name, 1.0)
        if mzf < 0 or mzf > 1.0:
            raise ValueError(
                f'{config_key} must be in [0, 1.0], got {mzf}. '
                f'1.0 = no mushy zone. '
                f'< 1.0 = solidus at this fraction of the liquidus temperature.'
            )
        if mzf < 0.7:
            raise ValueError(
                f'{config_key} = {mzf} is below the minimum of 0.7. '
                f'Values below 0.7 produce unphysically wide mushy zones that can '
                f'cause solver instabilities. Use a value in [0.7, 1.0].'
            )
        if mzf < 1.0 and eos_name not in all_components:
            raise ValueError(
                f'{config_key} = {mzf} < 1.0 but {eos_name} is not configured '
                f'in any layer. Per-material mushy zone overrides only apply '
                f'to materials that are actually used.'
            )

    # ── Condensed-phase suppression ─────────────────────────────────
    condensed_rho_min = config_params.get('condensed_rho_min', CONDENSED_RHO_MIN_DEFAULT)
    condensed_rho_scale = config_params.get('condensed_rho_scale', CONDENSED_RHO_SCALE_DEFAULT)

    if condensed_rho_min <= 0:
        raise ValueError(
            f'condensed_rho_min must be positive, got {condensed_rho_min} kg/m^3. '
            f'This is the sigmoid center for phase-aware mixing suppression. '
            f'Typical value: 300 (near H2O critical density 322 kg/m^3).'
        )

    if condensed_rho_scale <= 0:
        raise ValueError(
            f'condensed_rho_scale must be positive, got {condensed_rho_scale} kg/m^3. '
            f'This is the sigmoid width for phase-aware mixing suppression. '
            f'Typical value: 50 kg/m^3.'
        )

    binodal_T_scale = config_params.get('binodal_T_scale', BINODAL_T_SCALE_DEFAULT)
    if binodal_T_scale <= 0:
        raise ValueError(
            f'binodal_T_scale must be positive, got {binodal_T_scale} K. '
            f'This is the sigmoid width for H2 miscibility suppression. '
            f'Typical value: 50 K.'
        )

    # ── Numerical parameters ────────────────────────────────────────
    num_layers = config_params['num_layers']
    if num_layers < 10:
        raise ValueError(
            f'num_layers must be >= 10, got {num_layers}. '
            f'Fewer than 10 radial grid points cannot resolve the structure.'
        )

    if num_layers > 10000:
        raise ValueError(
            f'num_layers = {num_layers} is excessively large. '
            f'Values above 10000 cause very slow convergence with no accuracy gain. '
            f'Typical values: 100-500.'
        )

    # ── Numerical solver params (optional, validated only if provided) ──
    # These parameters have mass-adaptive defaults inside solver._solve().
    # When provided explicitly, validate that they are physically sensible.

    if 'adaptive_radial_fraction' in config_params:
        arf = config_params['adaptive_radial_fraction']
        if arf <= 0 or arf >= 1:
            raise ValueError(
                f'adaptive_radial_fraction must be in (0, 1), got {arf}. '
                f'Typical value: 0.98. This fraction of the radial domain uses '
                f'adaptive ODE stepping; the remainder uses fixed steps.'
            )

    for param in ('tolerance_outer', 'tolerance_inner',
                  'relative_tolerance', 'absolute_tolerance'):
        if param in config_params:
            val = config_params[param]
            if val <= 0:
                raise ValueError(f'{param} must be positive, got {val}.')

    for param in ('max_iterations_outer', 'max_iterations_inner',
                  'max_iterations_pressure'):
        if param in config_params:
            val = config_params[param]
            if val < 1:
                raise ValueError(f'{param} must be >= 1, got {val}.')

    if 'maximum_step' in config_params:
        maximum_step = config_params['maximum_step']
        if maximum_step <= 0:
            raise ValueError(
                f'maximum_step must be positive, got {maximum_step} m. '
                f'This is the max radial step size for the ODE integrator.'
            )

    # ── Pressure solver ─────────────────────────────────────────────
    target_sp = config_params['target_surface_pressure']
    if target_sp < 0:
        raise ValueError(
            f'target_surface_pressure must be >= 0, got {target_sp} Pa. '
            f'Default is 101325 Pa (1 atm).'
        )

    if 'pressure_tolerance' in config_params:
        ptol = config_params['pressure_tolerance']
        if ptol <= 0:
            raise ValueError(f'pressure_tolerance must be positive, got {ptol} Pa.')

    if 'max_center_pressure_guess' in config_params:
        max_pcg = config_params['max_center_pressure_guess']
        if max_pcg <= 0:
            raise ValueError(
                f'max_center_pressure_guess must be positive, got {max_pcg} Pa.'
            )

    # ── EOS-specific cross-checks ───────────────────────────────────
    # Melting curves only needed for EOS that use external phase routing
    needs_mc = bool(all_components & _NEEDS_MELTING_CURVES)
    if needs_mc:
        rock_solidus = config_params.get('rock_solidus', '')
        rock_liquidus = config_params.get('rock_liquidus', '')
        if not rock_solidus or not rock_liquidus:
            raise ValueError(
                'The configured EOS requires melting curves for phase routing, '
                'but rock_solidus or rock_liquidus is empty. '
                "Set both in [EOS]. Example: rock_solidus = 'Monteux16-solidus', "
                "rock_liquidus = 'Monteux16-liquidus-A-chondritic'."
            )

    # ── Multi-material mixing checks ────────────────────────────────
    for layer, eos_str in layer_eos_config.items():
        mixture = parse_layer_components(eos_str)

        # Fraction validation
        if len(mixture.components) > 1:
            for i, frac in enumerate(mixture.fractions):
                if frac < 0:
                    raise ValueError(
                        f'Negative mass fraction {frac} for component '
                        f"'{mixture.components[i]}' in layer '{layer}'."
                    )
            total = sum(mixture.fractions)
            if abs(total - 1.0) > 1e-4:
                raise ValueError(
                    f"Mass fractions in layer '{layer}' sum to {total:.6f}, "
                    f'not 1.0. Components: {mixture.components}, '
                    f'fractions: {mixture.fractions}.'
                )

        # Warn about mixing T-dependent and T-independent EOS
        tdep_comps_in_layer = [c for c in mixture.components if c in TDEP_EOS_NAMES]
        tindep_comps = [c for c in mixture.components if c not in TDEP_EOS_NAMES]
        if tdep_comps_in_layer and tindep_comps:
            # Only warn for non-Analytic T-independent components
            tindep_tabulated = [c for c in tindep_comps
                                if not c.startswith('Analytic:') and not c.startswith('Vinet:')]
            if tindep_tabulated:
                logger.warning(
                    f"Layer '{layer}' mixes T-dependent EOS "
                    f'({tdep_comps_in_layer}) with T-independent EOS '
                    f'({tindep_tabulated}). The T-independent components '
                    f'will use a fixed 300 K internally, which may be '
                    f'inconsistent with the adiabatic temperature profile.'
                )

    # ── EOS-layer compatibility ─────────────────────────────────────
    # Core should use iron-type EOS
    core_eos = layer_eos_config.get('core', '')
    core_mix = parse_layer_components(core_eos) if core_eos else None
    if core_mix:
        iron_keywords = {'iron', 'Fe'}
        for comp in core_mix.components:
            comp_material = comp.split(':')[-1] if ':' in comp else comp
            if not any(kw in comp_material for kw in iron_keywords):
                if not comp.startswith('Analytic:') and not comp.startswith('Vinet:'):
                    logger.warning(
                        f"Core EOS component '{comp}' does not appear to be "
                        f'an iron EOS. Core is typically iron (Fe). '
                        f'Proceeding anyway.'
                    )

    # H2O mixing fraction checks
    for layer in ('mantle', 'core'):
        eos_str = layer_eos_config.get(layer, '')
        if not eos_str:
            continue
        mix = parse_layer_components(eos_str)
        h2o_in_layer = any('H2O' in c for c in mix.components)
        if not h2o_in_layer:
            continue
        h2o_frac = sum(f for c, f in zip(mix.components, mix.fractions) if 'H2O' in c)
        if h2o_frac > 0.30:
            logger.warning(
                f"Layer '{layer}' has {h2o_frac * 100:.0f}% H2O, which exceeds "
                f'the validated range (0-30%). The solver has been tested up to '
                f'30% H2O by mass. Higher fractions may converge but are not '
                f'validated. Consider using a 3-layer model with a separate '
                f'ice layer for water-dominated compositions.'
            )

    # ── H2O-dominated mantle at low temperature ─────────────────────
    # H2O is vapor/supercritical at surface pressure when T > 647 K,
    # and the volume-additive mixing model cannot handle an all-vapor
    # mantle. Reject H2O-dominated mantles at adiabatic T_surf where
    # the solver will diverge.
    for layer in ('mantle',):
        eos_str = layer_eos_config.get(layer, '')
        if not eos_str:
            continue
        mix = parse_layer_components(eos_str)
        h2o_frac = sum(f for c, f in zip(mix.components, mix.fractions) if 'H2O' in c)
        has_silicate = any(
            c
            in {
                'PALEOS:MgSiO3',
                'WolfBower2018:MgSiO3',
                'RTPress100TPa:MgSiO3',
                'PALEOS-2phase:MgSiO3',
                'PALEOS-2phase:MgSiO3-highres',
            }
            for c in mix.components
        )
        if h2o_frac > 0.5 and not has_silicate and temperature_mode != 'isothermal':
            raise ValueError(
                f'Mantle is {h2o_frac * 100:.0f}% H2O with no silicate component. '
                f'At adiabatic/linear temperatures, H2O is vapor at surface '
                f'pressure and cannot support hydrostatic structure in the '
                f'volume-additive mixing model. Options: (1) add a silicate '
                f'component (e.g., "PALEOS:MgSiO3:0.50+PALEOS:H2O:0.50"), '
                f'(2) use a 3-layer model with a separate ice layer, or '
                f'(3) use isothermal mode with T < 647 K.'
            )

    # ── Pure Chabrier:H mantle (no condensed anchor) ────────────────
    for layer in ('mantle',):
        eos_str = layer_eos_config.get(layer, '')
        if not eos_str:
            continue
        mix = parse_layer_components(eos_str)
        if mix.is_single() and mix.components[0] == 'Chabrier:H':
            raise ValueError(
                'Pure Chabrier:H mantle is not supported. H2 is a gas at '
                'surface pressure and cannot form a condensed layer on its '
                'own. Use H2 as a mixing component in a silicate mantle, '
                'e.g., mantle = "PALEOS:MgSiO3:0.97+Chabrier:H:0.03".'
            )

    # ── H2 fraction warnings ───────────────────────────────────────
    for layer in ('mantle',):
        eos_str = layer_eos_config.get(layer, '')
        if not eos_str:
            continue
        mix = parse_layer_components(eos_str)
        h2_frac = sum(f for c, f in zip(mix.components, mix.fractions) if c == 'Chabrier:H')
        if h2_frac > 0.20:
            logger.warning(
                f"Layer '{layer}' has {h2_frac * 100:.0f}% H2 by mass. "
                f'The solver has been validated up to 20% H2. Higher '
                f'fractions may converge but are outside the tested range.'
            )


def choose_config_file(temp_config_path=None):
    """
    Function to choose the configuration file to run the main function.

    Info
    -----
    The function will first check if a temporary configuration file is provided.
    If not, it will check if the -c flag is provided in the command line arguments.
    If the -c flag is provided, the function will read the configuration file path from the next argument.
    If no temporary configuration file or -c flag is provided, the function will read the default configuration file.

    Parameters
    ----------
    temp_config_path : str, optional
        Path to a temporary configuration file. If provided, this file will be used instead of the default or -c specified config.

    Returns
    -------
    dict
        The loaded configuration parameters from the chosen config file.
    """

    # Load the configuration file either from terminal (-c flag) or default path
    if temp_config_path:
        try:
            config = toml.load(temp_config_path)
            logger.info(f'Reading temporary config file from: {temp_config_path}')
        except FileNotFoundError:
            logger.error(f'Error: Temporary config file not found at {temp_config_path}')
            sys.exit(1)
    elif '-c' in sys.argv:
        index = sys.argv.index('-c')
        try:
            config_file_path = sys.argv[index + 1]
            config = toml.load(config_file_path)
            logger.info(f'Reading config file from: {config_file_path}')
        except IndexError:
            logger.error('Error: -c flag provided but no config file path specified.')
            sys.exit(1)  # Exit with error code
        except FileNotFoundError:
            logger.error(f'Error: Config file not found at {config_file_path}')
            sys.exit(1)
    else:
        config_default_path = os.path.join(get_zalmoxis_root(), 'input', 'default.toml')
        try:
            config = toml.load(config_default_path)
            logger.info(f'Reading default config file from {config_default_path}')
        except FileNotFoundError:
            logger.info(f'Error: Default config file not found at {config_default_path}')
            sys.exit(1)

    return config


def load_zalmoxis_config(temp_config_path=None):
    """Load and return configuration parameters for the Zalmoxis model.

    Returns
    -------
    dict
        All relevant configuration parameters.
    """
    config = choose_config_file(temp_config_path)

    # Parse per-layer EOS config (supports both new and legacy formats)
    layer_eos_config = parse_eos_config(config['EOS'])
    validate_layer_eos(layer_eos_config)

    # Melting curve config (defaults for backward compat with old TOML files)
    eos_section = config['EOS']
    rock_solidus = eos_section.get('rock_solidus', 'Stixrude14-solidus')
    rock_liquidus = eos_section.get('rock_liquidus', 'Stixrude14-liquidus')
    mushy_zone_factor = eos_section.get('mushy_zone_factor', 1.0)
    condensed_rho_min = eos_section.get('condensed_rho_min', CONDENSED_RHO_MIN_DEFAULT)
    condensed_rho_scale = eos_section.get('condensed_rho_scale', CONDENSED_RHO_SCALE_DEFAULT)
    binodal_T_scale = eos_section.get('binodal_T_scale', BINODAL_T_SCALE_DEFAULT)

    # Build per-EOS mushy_zone_factors dict. Only include materials that are
    # actually configured in a layer; unused materials default to 1.0 so that
    # a global mushy_zone_factor < 1.0 does not trigger the validation check
    # for materials absent from the model.
    _paleos_materials = {
        'PALEOS:iron': 'mushy_zone_factor_iron',
        'PALEOS:MgSiO3': 'mushy_zone_factor_MgSiO3',
        'PALEOS:H2O': 'mushy_zone_factor_H2O',
    }
    # Collect all EOS component strings from all layers
    _all_eos_strings = ' '.join(v for v in layer_eos_config.values() if v)
    mushy_zone_factors = {}
    for paleos_name, toml_key in _paleos_materials.items():
        if paleos_name in _all_eos_strings:
            # Material is in use: apply per-material override or global default
            mushy_zone_factors[paleos_name] = eos_section.get(toml_key, mushy_zone_factor)
        else:
            # Material not in use: default to 1.0 (no mushy zone)
            mushy_zone_factors[paleos_name] = eos_section.get(toml_key, 1.0)

    config_params = {
        'planet_mass': config['InputParameter']['planet_mass'] * earth_mass,
        'core_mass_fraction': config['AssumptionsAndInitialGuesses']['core_mass_fraction'],
        'mantle_mass_fraction': config['AssumptionsAndInitialGuesses']['mantle_mass_fraction'],
        'temperature_mode': config['AssumptionsAndInitialGuesses']['temperature_mode'],
        'surface_temperature': config['AssumptionsAndInitialGuesses']['surface_temperature'],
        'center_temperature': config['AssumptionsAndInitialGuesses']['center_temperature'],
        'temp_profile_file': config['AssumptionsAndInitialGuesses']['temperature_profile_file'],
        'layer_eos_config': layer_eos_config,
        'rock_solidus': rock_solidus,
        'rock_liquidus': rock_liquidus,
        'mushy_zone_factor': mushy_zone_factor,
        'mushy_zone_factors': mushy_zone_factors,
        'condensed_rho_min': condensed_rho_min,
        'condensed_rho_scale': condensed_rho_scale,
        'binodal_T_scale': binodal_T_scale,
        'num_layers': config['Calculations']['num_layers'],
        'target_surface_pressure': config.get('PressureAdjustment', {}).get(
            'target_surface_pressure', 101325,
        ),
        'data_output_enabled': config['Output'].get('data_enabled', True),
        'plotting_enabled': config['Output'].get('plots_enabled', False),
    }

    # Numerical solver params are optional in the TOML. If present, include
    # them so they override the mass-adaptive defaults in solver._solve().
    _iter = config.get('IterativeProcess', {})
    _pres = config.get('PressureAdjustment', {})
    _optional_iter_keys = [
        'max_iterations_outer', 'tolerance_outer',
        'max_iterations_inner', 'tolerance_inner',
        'relative_tolerance', 'absolute_tolerance',
        'maximum_step', 'adaptive_radial_fraction',
        'max_center_pressure_guess',
    ]
    for key in _optional_iter_keys:
        if key in _iter:
            config_params[key] = _iter[key]
    _optional_pres_keys = ['pressure_tolerance', 'max_iterations_pressure']
    for key in _optional_pres_keys:
        if key in _pres:
            config_params[key] = _pres[key]

    # Validate all parameters for physical and logical consistency
    validate_config(config_params)

    return config_params


def load_material_dictionaries():
    """Load and return the material properties dictionaries.

    Returns
    -------
    dict
        EOS registry keyed by EOS identifier string (e.g.
        ``"Seager2007:iron"``, ``"PALEOS:MgSiO3"``).
    """
    return dict(EOS_REGISTRY)


# EOS names that need external melting curves for solid/liquid phase routing.
# Unified PALEOS tables do not need external melting curves (phase boundary
# is extracted from the table's phase column).
_NEEDS_MELTING_CURVES = {
    'WolfBower2018:MgSiO3',
    'RTPress100TPa:MgSiO3',
    'PALEOS-2phase:MgSiO3',
    'PALEOS-2phase:MgSiO3-highres',
}


def load_solidus_liquidus_functions(
    layer_eos_config,
    solidus_id='Stixrude14-solidus',
    liquidus_id='Stixrude14-liquidus',
):
    """Load solidus and liquidus functions if any layer uses an EOS that needs them.

    Unified PALEOS tables (PALEOS:iron, PALEOS:MgSiO3, PALEOS:H2O) are
    T-dependent but derive their phase boundary from the table itself, so
    they do not need external melting curves.

    Parameters
    ----------
    layer_eos_config : dict
        Per-layer EOS config.
    solidus_id : str
        Solidus melting curve identifier.
    liquidus_id : str
        Liquidus melting curve identifier.

    Returns
    -------
    tuple or None
        (solidus_func, liquidus_func) if needed, else None.
    """
    all_comps = set()
    for v in layer_eos_config.values():
        if v:
            m = parse_layer_components(v)
            all_comps.update(m.components)
    if all_comps & _NEEDS_MELTING_CURVES:
        solidus_func, liquidus_func = get_solidus_liquidus_functions(solidus_id, liquidus_id)
        return (solidus_func, liquidus_func)
    return None
