"""
Zalmoxis core module containing the main functions to run the exoplanet internal structure model.

Imports
-------
- [`constants`](zalmoxis.constants.md): TDEP_EOS_NAMES, earth_center_pressure, earth_mass, earth_radius
- [`eos_analytic`](zalmoxis.eos_analytic.md): VALID_MATERIAL_KEYS
- [`eos_functions`](zalmoxis.eos_functions.md): calculate_density, calculate_temperature_profile, create_pressure_density_files, get_solidus_liquidus_functions, get_Tdep_material
- [`eos_properties`](zalmoxis.eos_properties.md): material_properties_iron_PALEOS_silicate_planets, material_properties_iron_RTPress100TPa_silicate_planets,
        material_properties_iron_silicate_planets, material_properties_iron_Tdep_silicate_planets, material_properties_water_planets
- [`structure_model`](zalmoxis.structure_model.md): get_layer_eos, solve_structure
"""

from __future__ import annotations

import logging
import math
import os
import sys
import time

import numpy as np
import toml
from scipy.optimize import brentq

from .constants import (
    CONDENSED_RHO_MIN_DEFAULT,
    CONDENSED_RHO_SCALE_DEFAULT,
    TDEP_EOS_NAMES,
    earth_center_pressure,
    earth_mass,
    earth_radius,
)
from .eos_analytic import VALID_MATERIAL_KEYS
from .eos_functions import (
    calculate_temperature_profile,
    compute_adiabatic_temperature,
    create_pressure_density_files,
    get_solidus_liquidus_functions,
    get_Tdep_material,
)
from .eos_properties import EOS_REGISTRY
from .mixing import (
    any_component_is_tdep,
    calculate_mixed_density,
    parse_all_layer_mixtures,
    parse_layer_components,
)
from .plots.plot_phase_vs_radius import plot_PT_with_phases
from .plots.plot_profiles import plot_planet_profile_single
from .structure_model import get_layer_eos, solve_structure

# Run file via command line with default configuration file: python -m zalmoxis -c input/default.toml

# Read the environment variable for ZALMOXIS_ROOT
ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    raise RuntimeError('ZALMOXIS_ROOT environment variable not set')

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
    'Seager2007:H2O',
    'PALEOS:iron',
    'PALEOS:MgSiO3',
    'PALEOS:H2O',
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
            raise ValueError(
                f"Invalid EOS component '{comp}' in layer '{layer}'. "
                f'Valid tabulated: {sorted(VALID_TABULATED_EOS)}. '
                f'Valid analytic: Analytic:<material> with '
                f'{sorted(VALID_MATERIAL_KEYS)}.'
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
    if planet_mass <= 0:
        raise ValueError(
            f'planet_mass must be positive, got {planet_mass / earth_mass:.4f} M_earth. '
            f'Set planet_mass > 0 in [InputParameter].'
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

    # ── Temperature parameters ──────────────────────────────────────
    temperature_mode = config_params['temperature_mode']
    valid_modes = ('isothermal', 'linear', 'prescribed', 'adiabatic')
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

    if temperature_mode in ('linear', 'adiabatic') and center_temp <= 0:
        raise ValueError(
            f"center_temperature must be positive for '{temperature_mode}' mode, "
            f'got {center_temp} K. Temperatures must be in Kelvin.'
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

    if temperature_mode == 'adiabatic' and not uses_Tdep:
        raise ValueError(
            f'Adiabatic temperature mode requires at least one T-dependent EOS layer, '
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
        # Skip validation for entries that equal the global default
        # (inherited, not explicitly overridden per material).
        if mzf == mushy_zone_factor:
            continue
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

    arf = config_params['adaptive_radial_fraction']
    if arf <= 0 or arf >= 1:
        raise ValueError(
            f'adaptive_radial_fraction must be in (0, 1), got {arf}. '
            f'Typical value: 0.98. This fraction of the radial domain uses '
            f'adaptive ODE stepping; the remainder uses fixed steps.'
        )

    # ── Solver tolerances ───────────────────────────────────────────
    for param, name in [
        ('tolerance_outer', 'tolerance_outer'),
        ('tolerance_inner', 'tolerance_inner'),
        ('relative_tolerance', 'relative_tolerance'),
        ('absolute_tolerance', 'absolute_tolerance'),
    ]:
        val = config_params[param]
        if val <= 0:
            raise ValueError(f'{name} must be positive, got {val}.')

    for param, name in [
        ('max_iterations_outer', 'max_iterations_outer'),
        ('max_iterations_inner', 'max_iterations_inner'),
        ('max_iterations_pressure', 'max_iterations_pressure'),
    ]:
        val = config_params[param]
        if val < 1:
            raise ValueError(f'{name} must be >= 1, got {val}.')

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

    ptol = config_params['pressure_tolerance']
    if ptol <= 0:
        raise ValueError(f'pressure_tolerance must be positive, got {ptol} Pa.')

    max_pcg = config_params['max_center_pressure_guess']
    if max_pcg <= 0:
        raise ValueError(f'max_center_pressure_guess must be positive, got {max_pcg} Pa.')

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
            tindep_tabulated = [c for c in tindep_comps if not c.startswith('Analytic:')]
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
                if not comp.startswith('Analytic:'):
                    logger.warning(
                        f"Core EOS component '{comp}' does not appear to be "
                        f'an iron EOS. Core is typically iron (Fe). '
                        f'Proceeding anyway.'
                    )

    # H2O as mantle component at high T is physically questionable
    if temperature_mode == 'adiabatic':
        for layer in ('mantle', 'core'):
            eos_str = layer_eos_config.get(layer, '')
            if not eos_str:
                continue
            mix = parse_layer_components(eos_str)
            h2o_in_layer = any('H2O' in c for c in mix.components)
            if h2o_in_layer and surface_temp > 2000:
                h2o_frac = sum(f for c, f in zip(mix.components, mix.fractions) if 'H2O' in c)
                if h2o_frac > 0.3:
                    logger.warning(
                        f"Layer '{layer}' has {h2o_frac * 100:.0f}% H2O at "
                        f'T_surf={surface_temp} K. At high temperatures, '
                        f'water is supercritical vapor, which may cause '
                        f'convergence difficulties or unphysically large radii.'
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
        config_default_path = os.path.join(ZALMOXIS_ROOT, 'input', 'default.toml')
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

    # Build per-EOS mushy_zone_factors dict. Start from the global default,
    # then apply per-material overrides if present in the TOML.
    mushy_zone_factors = {
        'PALEOS:iron': eos_section.get('mushy_zone_factor_iron', mushy_zone_factor),
        'PALEOS:MgSiO3': eos_section.get('mushy_zone_factor_MgSiO3', mushy_zone_factor),
        'PALEOS:H2O': eos_section.get('mushy_zone_factor_H2O', mushy_zone_factor),
    }

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
        'num_layers': config['Calculations']['num_layers'],
        'max_iterations_outer': config['IterativeProcess']['max_iterations_outer'],
        'tolerance_outer': config['IterativeProcess']['tolerance_outer'],
        'max_iterations_inner': config['IterativeProcess']['max_iterations_inner'],
        'tolerance_inner': config['IterativeProcess']['tolerance_inner'],
        'relative_tolerance': config['IterativeProcess']['relative_tolerance'],
        'absolute_tolerance': config['IterativeProcess']['absolute_tolerance'],
        'maximum_step': config['IterativeProcess']['maximum_step'],
        'adaptive_radial_fraction': config['IterativeProcess']['adaptive_radial_fraction'],
        'max_center_pressure_guess': config['IterativeProcess']['max_center_pressure_guess'],
        'target_surface_pressure': config['PressureAdjustment']['target_surface_pressure'],
        'pressure_tolerance': config['PressureAdjustment']['pressure_tolerance'],
        'max_iterations_pressure': config['PressureAdjustment']['max_iterations_pressure'],
        'data_output_enabled': config['Output']['data_enabled'],
        'plotting_enabled': config['Output']['plots_enabled'],
        'verbose': config['Output']['verbose'],
        'iteration_profiles_enabled': config['Output']['iteration_profiles_enabled'],
    }

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


def main(
    config_params,
    material_dictionaries,
    melting_curves_functions,
    input_dir,
    layer_mixtures=None,
):
    """Run the exoplanet internal structure model.

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
        Per-layer LayerMixture objects. If None, parsed from
        ``config_params['layer_eos_config']``. PROTEUS/CALLIOPE can provide
        pre-built mixtures with runtime-updated fractions.

    Returns
    -------
    dict
        Model results including radii, density, gravity, pressure, temperature,
        mass enclosed, convergence status, and timing.
    """
    # Initialize convergence flags
    converged = False
    converged_pressure = False
    converged_density = False
    converged_mass = False

    # Unpack configuration parameters
    planet_mass = config_params['planet_mass']
    core_mass_fraction = config_params['core_mass_fraction']
    mantle_mass_fraction = config_params['mantle_mass_fraction']
    temperature_mode = config_params['temperature_mode']
    surface_temperature = config_params['surface_temperature']
    center_temperature = config_params['center_temperature']
    temp_profile_file = config_params['temp_profile_file']
    layer_eos_config = config_params['layer_eos_config']
    num_layers = config_params['num_layers']
    max_iterations_outer = config_params['max_iterations_outer']
    tolerance_outer = config_params['tolerance_outer']
    max_iterations_inner = config_params['max_iterations_inner']
    tolerance_inner = config_params['tolerance_inner']
    relative_tolerance = config_params['relative_tolerance']
    absolute_tolerance = config_params['absolute_tolerance']
    maximum_step = config_params['maximum_step']
    adaptive_radial_fraction = config_params['adaptive_radial_fraction']
    max_center_pressure_guess = config_params['max_center_pressure_guess']
    target_surface_pressure = config_params['target_surface_pressure']
    pressure_tolerance = config_params['pressure_tolerance']
    max_iterations_pressure = config_params['max_iterations_pressure']
    verbose = config_params['verbose']
    iteration_profiles_enabled = config_params['iteration_profiles_enabled']
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
            else:
                continue
            if mass_in_earth > max_mass:
                raise ValueError(
                    f'{eos_name} EOS is limited to planets <= '
                    f'{max_mass} M_earth (requested {mass_in_earth:.2f} M_earth). '
                    f'{reason}'
                )

    # Setup initial guesses
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

    # Initialize empty cache for interpolation functions
    interpolation_cache = {}

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
            verbose and logger.info(
                f'Adiabatic mode: capped center_temperature initial guess '
                f'to {center_temperature:.0f} K (5x surface or 3000 K).'
            )

    # Solve the interior structure
    for outer_iter in range(max_iterations_outer):
        radii = np.linspace(0, radius_guess, num_layers)

        density = np.zeros(num_layers)
        mass_enclosed = np.zeros(num_layers)
        gravity = np.zeros(num_layers)
        pressure = np.zeros(num_layers)

        if uses_Tdep:
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
                verbose and logger.info(
                    f'Outer iter {outer_iter}: adiabat blend = {_adiabat_blend:.2f}'
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

                    def temperature_function(
                        r, P, _b=_blend, _lp=_logP_sorted, _ts=_T_sorted, _ltf=_linear_tf
                    ):
                        T_lin = _ltf(r)
                        if P <= 0:
                            return T_lin
                        T_adi = float(np.interp(np.log10(P), _lp, _ts))
                        T_adi = max(_T_MIN_CLAMP, min(T_adi, _T_MAX_CLAMP))
                        return (1.0 - _b) * T_lin + _b * T_adi

                else:

                    def temperature_function(r, P, _lp=_logP_sorted, _ts=_T_sorted):
                        if P <= 0:
                            return surface_temperature
                        T_val = float(np.interp(np.log10(P), _lp, _ts))
                        return max(_T_MIN_CLAMP, min(T_val, _T_MAX_CLAMP))

                # Pre-compute temperatures array for the density update loop
                # (uses the converged pressure from the previous iteration)
                temperatures = np.array(
                    [
                        temperature_function(radii[i], prev_pressure[i])
                        for i in range(num_layers)
                    ]
                )
            else:

                def temperature_function(r, P, _tf=_linear_tf):
                    return _tf(r)

                temperatures = _linear_tf(radii)
        else:
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
                    temperature_function if uses_Tdep else None,
                    mushy_zone_factors,
                    condensed_rho_min,
                    condensed_rho_scale,
                )
                if iteration_profiles_enabled:
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
                    temperature_function if uses_Tdep else None,
                    mushy_zone_factors,
                    condensed_rho_min,
                    condensed_rho_scale,
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
                    verbose and logger.info(
                        f'Surface pressure converged after '
                        f'{root_info.function_calls} evaluations (Brent method).'
                    )
                else:
                    converged_pressure = False
                    verbose and logger.warning(
                        f'Brent method: converged={root_info.converged}, '
                        f'residual={surface_residual:.2e} Pa, '
                        f'min_P={np.min(pressure):.2e} Pa.'
                    )
            except ValueError:
                # f(p_low) and f(p_high) have the same sign — bracket
                # invalid.  Use the last evaluated solution if available.
                verbose and logger.warning(
                    f'Could not bracket pressure root in [{p_low:.2e}, {p_high:.2e}] Pa.'
                )
                if _state['mass_enclosed'] is not None:
                    mass_enclosed = _state['mass_enclosed']
                    gravity = _state['gravity']
                    pressure = _state['pressure']
                else:
                    # No evaluations succeeded — keep profiles from previous
                    # outer iteration (already initialised above).
                    verbose and logger.warning(
                        'No valid ODE solutions obtained during bracket search. '
                        'Keeping previous profiles.'
                    )
                converged_pressure = False

            # Update density grid (solve_structure may return fewer points
            # than num_layers if the ODE solver terminated early)
            for i in range(min(num_layers, len(mass_enclosed))):
                mixture = get_layer_eos(
                    mass_enclosed[i],
                    cmb_mass,
                    core_mantle_mass,
                    layer_mixtures,
                )

                # Use T(r, P) with the actual converged pressure so
                # temperature is consistent with the ODE solution.
                T_i = temperature_function(radii[i], pressure[i]) if uses_Tdep else 300
                new_density = calculate_mixed_density(
                    pressure[i],
                    T_i,
                    mixture,
                    material_dictionaries,
                    solidus_func,
                    liquidus_func,
                    interpolation_cache,
                    mushy_zone_factors,
                    condensed_rho_min,
                    condensed_rho_scale,
                )

                if new_density is None:
                    if not mixture.is_single():
                        verbose and logger.warning(
                            f'All mixture components suppressed at r={radii[i]:.0f} m, '
                            f'P={pressure[i]:.2e} Pa. Using previous density.'
                        )
                    else:
                        verbose and logger.warning(
                            f'Density lookup failed at r={radii[i]:.0f} m, '
                            f'P={pressure[i]:.2e} Pa. Using previous density.'
                        )
                    new_density = old_density[i]

                density[i] = 0.5 * (new_density + old_density[i])

            # Check density convergence
            relative_diff_inner = np.max(
                np.abs((density - old_density) / (old_density + 1e-20))
            )
            if relative_diff_inner < tolerance_inner:
                verbose and logger.info(
                    f'Inner loop converged after {inner_iter + 1} iterations.'
                )
                converged_density = True
                break

            if inner_iter == max_iterations_inner - 1:
                verbose and logger.warning(
                    f'Maximum inner iterations ({max_iterations_inner}) reached. '
                    'Density may not be fully converged.'
                )

        # Recompute the temperatures array from the converged pressure profile
        # so model_results['temperature'] reflects actual T(P), not the pre-Brent estimate.
        if uses_Tdep:
            temperatures = np.array(
                [temperature_function(radii[i], pressure[i]) for i in range(num_layers)]
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
            verbose and logger.warning(
                f'Outer iter {outer_iter}: calculated_mass={calculated_mass:.2e}, '
                f'shrinking radius_guess to {radius_guess:.0f} m.'
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
            verbose and logger.warning(
                f'Maximum outer iterations ({max_iterations_outer}) reached. '
                'Total mass may not be fully converged.'
            )

    if converged_mass and converged_density and converged_pressure:
        converged = True

    end_time = time.time()
    total_time = end_time - start_time

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


def post_processing(config_params, id_mass=None, output_file=None):
    """Post-process model results by saving output data and plotting.

    Parameters
    ----------
    config_params : dict
        Configuration parameters for the model.
    id_mass : str or None
        Identifier for the planet mass, used in output file naming.
    output_file : str or None
        Path to the output file for calculated mass and radius.
    """
    data_output_enabled = config_params['data_output_enabled']
    plotting_enabled = config_params['plotting_enabled']

    layer_eos_config = config_params['layer_eos_config']
    solidus_id = config_params.get('rock_solidus', 'Stixrude14-solidus')
    liquidus_id = config_params.get('rock_liquidus', 'Stixrude14-liquidus')

    model_results = main(
        config_params,
        material_dictionaries=load_material_dictionaries(),
        melting_curves_functions=load_solidus_liquidus_functions(
            layer_eos_config, solidus_id, liquidus_id
        ),
        input_dir=os.path.join(ZALMOXIS_ROOT, 'input'),
    )

    # Extract results
    radii = model_results['radii']
    density = model_results['density']
    gravity = model_results['gravity']
    pressure = model_results['pressure']
    temperature = model_results['temperature']
    mass_enclosed = model_results['mass_enclosed']
    cmb_mass = model_results['cmb_mass']
    core_mantle_mass = model_results['core_mantle_mass']
    total_time = model_results['total_time']
    converged = model_results['converged']
    converged_pressure = model_results['converged_pressure']
    converged_density = model_results['converged_density']
    converged_mass = model_results['converged_mass']

    cmb_index = np.argmax(mass_enclosed >= cmb_mass)

    average_density = mass_enclosed[-1] / (4 / 3 * math.pi * radii[-1] ** 3)

    # Check if mantle uses a Tdep EOS that needs external melting curves
    # for phase detection. Unified PALEOS tables derive phases from the
    # table itself and do not need (or support) get_Tdep_material().
    mantle_str = layer_eos_config.get('mantle', '')
    if mantle_str:
        _mantle_mix = parse_layer_components(mantle_str)
        uses_phase_detection = bool(set(_mantle_mix.components) & _NEEDS_MELTING_CURVES)
    else:
        uses_phase_detection = False

    if uses_phase_detection:
        mantle_pressures = pressure[cmb_index:]
        mantle_temperatures = temperature[cmb_index:]
        mantle_radii = radii[cmb_index:]

        solidus_func, liquidus_func = load_solidus_liquidus_functions(
            layer_eos_config, solidus_id, liquidus_id
        )

        mantle_phases = get_Tdep_material(
            mantle_pressures, mantle_temperatures, solidus_func, liquidus_func
        )

    logger.info('Exoplanet Internal Structure Model Results:')
    logger.info('----------------------------------------------------------------------')
    logger.info(
        f'Calculated Planet Mass: {mass_enclosed[-1]:.2e} kg or '
        f'{mass_enclosed[-1] / earth_mass:.2f} Earth masses'
    )
    logger.info(
        f'Calculated Planet Radius: {radii[-1]:.2e} m or '
        f'{radii[-1] / earth_radius:.2f} Earth radii'
    )
    logger.info(f'Core Radius: {radii[cmb_index]:.2e} m')
    logger.info(f'Mantle Density (at CMB): {density[cmb_index]:.2f} kg/m^3')
    logger.info(f'Core Density (at CMB): {density[cmb_index - 1]:.2f} kg/m^3')
    logger.info(f'Pressure at Core-Mantle Boundary (CMB): {pressure[cmb_index]:.2e} Pa')
    logger.info(f'Pressure at Center: {pressure[0]:.2e} Pa')
    logger.info(f'Average Density: {average_density:.2f} kg/m^3')
    logger.info(f'CMB Mass Fraction: {mass_enclosed[cmb_index] / mass_enclosed[-1]:.3f}')
    logger.info(
        f'Core+Mantle Mass Fraction: '
        f'{(core_mantle_mass - mass_enclosed[cmb_index]) / mass_enclosed[-1]:.3f}'
    )
    logger.info(f'Calculated Core Radius Fraction: {radii[cmb_index] / radii[-1]:.2f}')
    logger.info(
        f'Calculated Core+Mantle Radius Fraction: '
        f'{(radii[np.argmax(mass_enclosed >= core_mantle_mass)] / radii[-1]):.2f}'
    )
    logger.info(f'Total Computation Time: {total_time:.2f} seconds')
    logger.info(
        f'Overall Convergence Status: {converged} with Pressure: {converged_pressure}, '
        f'Density: {converged_density}, Mass: {converged_mass}'
    )

    if data_output_enabled:
        output_data = np.column_stack(
            (radii, density, gravity, pressure, temperature, mass_enclosed)
        )
        header = (
            'Radius (m)\tDensity (kg/m^3)\tGravity (m/s^2)\t'
            'Pressure (Pa)\tTemperature (K)\tMass Enclosed (kg)'
        )
        if id_mass is None:
            np.savetxt(
                os.path.join(ZALMOXIS_ROOT, 'output_files', 'planet_profile.txt'),
                output_data,
                header=header,
            )
        else:
            np.savetxt(
                os.path.join(ZALMOXIS_ROOT, 'output_files', f'planet_profile{id_mass}.txt'),
                output_data,
                header=header,
            )
        if output_file is None:
            output_file = os.path.join(
                ZALMOXIS_ROOT, 'output_files', 'calculated_planet_mass_radius.txt'
            )
        if not os.path.exists(output_file):
            header = 'Calculated Mass (kg)\tCalculated Radius (m)'
            with open(output_file, 'w') as file:
                file.write(header + '\n')
        with open(output_file, 'a') as file:
            file.write(f'{mass_enclosed[-1]}\t{radii[-1]}\n')

    if plotting_enabled:
        plot_planet_profile_single(
            radii,
            density,
            gravity,
            pressure,
            temperature,
            radii[np.argmax(mass_enclosed >= cmb_mass)],
            cmb_mass,
            mass_enclosed[-1] / (4 / 3 * math.pi * radii[-1] ** 3),
            mass_enclosed,
            id_mass,
        )

        if uses_phase_detection:
            plot_PT_with_phases(
                mantle_pressures,
                mantle_temperatures,
                mantle_radii,
                mantle_phases,
                radii[cmb_index],
            )
