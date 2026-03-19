"""Multi-material volume-additive mixing for layered planet models.

Provides:
- ``LayerMixture``: container for a layer's EOS components and mass fractions.
- ``parse_layer_components()``: parse config strings into LayerMixture.
- ``parse_all_layer_mixtures()``: parse all layers.
- ``calculate_mixed_density()``: harmonic-mean density from multiple EOS.
- ``get_mixed_nabla_ad()``: mass-weighted adiabatic gradient from multiple EOS.
- ``any_component_is_tdep()``: check if any component uses T-dependent EOS.

Imports
-------
- ``eos_functions``: ``calculate_density``, ``_get_paleos_unified_nabla_ad``,
  ``_get_paleos_nabla_ad``, ``_compute_paleos_dtdp``
- ``constants``: ``TDEP_EOS_NAMES``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from .constants import CONDENSED_RHO_MIN_DEFAULT, CONDENSED_RHO_SCALE_DEFAULT, TDEP_EOS_NAMES

logger = logging.getLogger(__name__)

# All unified PALEOS EOS names that support mushy_zone_factor
_PALEOS_UNIFIED_NAMES = frozenset({'PALEOS:iron', 'PALEOS:MgSiO3', 'PALEOS:H2O', 'Chabrier:H'})

# Component-type sets for binodal matching
_SILICATE_EOS_NAMES = frozenset(
    {
        'PALEOS:MgSiO3',
        'WolfBower2018:MgSiO3',
        'RTPress100TPa:MgSiO3',
        'PALEOS-2phase:MgSiO3',
    }
)
_H2_EOS_NAMES = frozenset({'Chabrier:H'})
_H2O_EOS_NAMES = frozenset({'PALEOS:H2O', 'Seager2007:H2O'})

# Default binodal sigmoid width (K). Controls the smooth transition at
# the miscibility boundary. 50 K gives a ~200 K transition zone.
BINODAL_T_SCALE_DEFAULT = 50.0


def _get_mushy_zone_factor(eos_name, mushy_zone_factors):
    """Look up the mushy zone factor for a specific EOS.

    Parameters
    ----------
    eos_name : str
        EOS identifier string, e.g. ``"PALEOS:iron"``.
    mushy_zone_factors : dict or float or None
        Per-EOS mushy zone factors. If a dict, looks up by ``eos_name``
        and falls back to 1.0 for missing keys. If a float, returns
        that value directly (backward compat). If None, returns 1.0.

    Returns
    -------
    float
        Mushy zone factor for this EOS. Always 1.0 for non-PALEOS EOS.
    """
    if eos_name not in _PALEOS_UNIFIED_NAMES:
        return 1.0
    if mushy_zone_factors is None:
        return 1.0
    if isinstance(mushy_zone_factors, (int, float)):
        return float(mushy_zone_factors)
    return mushy_zone_factors.get(eos_name, 1.0)


@dataclass
class LayerMixture:
    """A layer composed of one or more EOS materials with mass fractions.

    Parameters
    ----------
    components : list of str
        EOS identifier strings, e.g. ``["PALEOS:MgSiO3", "PALEOS:H2O"]``.
    fractions : list of float
        Mass fractions corresponding to each component. Must sum to 1.0.
    """

    components: list[str] = field(default_factory=list)
    fractions: list[float] = field(default_factory=list)

    def __post_init__(self):
        if len(self.components) == 0:
            raise ValueError('LayerMixture: must have at least one component.')
        if len(self.components) != len(self.fractions):
            raise ValueError(
                f'LayerMixture: components ({len(self.components)}) and '
                f'fractions ({len(self.fractions)}) must have the same length.'
            )
        total = sum(self.fractions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'LayerMixture: fractions sum to {total:.6f}, not 1.0.')

    def is_single(self) -> bool:
        """True if this mixture contains exactly one component."""
        return len(self.components) == 1

    def primary(self) -> str:
        """Return the dominant (highest fraction) EOS identifier."""
        idx = int(np.argmax(self.fractions))
        return self.components[idx]

    def update_fractions(self, new_fractions: list[float]) -> None:
        """Update mass fractions at runtime (called by PROTEUS/CALLIOPE).

        Parameters
        ----------
        new_fractions : list of float
            New mass fractions, must have same length as components
            and sum to 1.0 (within tolerance 1e-6).

        Raises
        ------
        ValueError
            If length mismatch or fractions don't sum to 1.0.
        """
        if len(new_fractions) != len(self.components):
            raise ValueError(
                f'Expected {len(self.components)} fractions, got {len(new_fractions)}.'
            )
        if any(f < 0 for f in new_fractions):
            raise ValueError(f'Mass fractions must be non-negative, got {new_fractions}.')
        total = sum(new_fractions)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f'Fractions must sum to 1.0, got {total:.8f}.')
        self.fractions = list(new_fractions)

    def has_tdep(self) -> bool:
        """True if any component uses a T-dependent EOS."""
        return any(c in TDEP_EOS_NAMES for c in self.components)


def _parse_single_component(s: str) -> tuple[str, float]:
    """Parse one component string into (eos_name, fraction).

    Examples
    --------
    >>> _parse_single_component("PALEOS:MgSiO3:0.85")
    ('PALEOS:MgSiO3', 0.85)
    >>> _parse_single_component("PALEOS:iron")
    ('PALEOS:iron', 1.0)
    >>> _parse_single_component("Analytic:SiC")
    ('Analytic:SiC', 1.0)
    """
    parts = s.strip().split(':')
    if len(parts) >= 3:
        try:
            frac = float(parts[-1])
            eos_name = ':'.join(parts[:-1])
            return eos_name, frac
        except ValueError:
            pass
    return s.strip(), 1.0


def parse_layer_components(config_value: str) -> LayerMixture:
    """Parse a layer EOS config string into a LayerMixture.

    Formats
    -------
    Single material (backward compat)::

        "PALEOS:iron"  ->  LayerMixture(["PALEOS:iron"], [1.0])

    Multi-material with mass fractions::

        "PALEOS:MgSiO3:0.85+PALEOS:H2O:0.15"
        ->  LayerMixture(["PALEOS:MgSiO3", "PALEOS:H2O"], [0.85, 0.15])

    Parameters
    ----------
    config_value : str
        Layer EOS config string from TOML.

    Returns
    -------
    LayerMixture

    Raises
    ------
    ValueError
        If no components found or fractions are invalid.
    """
    if not config_value or not config_value.strip():
        raise ValueError('Empty EOS config string.')

    segments = config_value.split('+')
    components = []
    fractions = []

    for seg in segments:
        eos_name, frac = _parse_single_component(seg)
        if not eos_name:
            raise ValueError(f'Empty EOS name in component: {seg!r}')
        components.append(eos_name)
        fractions.append(frac)

    if not components:
        raise ValueError(f'No components parsed from: {config_value!r}')

    # Normalize fractions if needed
    total = sum(fractions)
    if len(components) > 1 and abs(total - 1.0) > 1e-6:
        logger.warning(
            f'Mass fractions sum to {total:.6f}, normalizing to 1.0. '
            f'Components: {components}, fractions: {fractions}'
        )
        fractions = [f / total for f in fractions]
    elif len(components) == 1:
        fractions = [1.0]

    if any(f < 0 for f in fractions):
        raise ValueError(f'Negative mass fraction in {config_value!r}: {fractions}')

    return LayerMixture(components=components, fractions=fractions)


def parse_all_layer_mixtures(
    layer_eos_config: dict[str, str],
) -> dict[str, LayerMixture]:
    """Parse all layers in layer_eos_config into LayerMixture objects.

    Parameters
    ----------
    layer_eos_config : dict
        Per-layer EOS config, e.g.
        ``{"core": "PALEOS:iron", "mantle": "PALEOS:MgSiO3:0.85+PALEOS:H2O:0.15"}``.

    Returns
    -------
    dict
        Per-layer LayerMixture objects, keyed by layer name.
    """
    return {
        layer: parse_layer_components(eos_str)
        for layer, eos_str in layer_eos_config.items()
        if eos_str
    }


def any_component_is_tdep(
    layer_mixtures: dict[str, LayerMixture],
) -> bool:
    """Check if any component in any layer uses a T-dependent EOS.

    Parameters
    ----------
    layer_mixtures : dict
        Per-layer LayerMixture objects.

    Returns
    -------
    bool
    """
    return any(mix.has_tdep() for mix in layer_mixtures.values())


def _condensed_weight(
    rho, rho_min=CONDENSED_RHO_MIN_DEFAULT, rho_scale=CONDENSED_RHO_SCALE_DEFAULT
):
    """Smooth sigmoid weight: 1 for dense (condensed), ~0 for light (vapor).

    Used in multi-component mixtures to suppress gas-like contributions
    to the harmonic-mean density and nabla_ad. Single-component layers
    are not affected (they bypass this function via the fast path).

    Parameters
    ----------
    rho : float
        Component density in kg/m^3.
    rho_min : float
        Sigmoid center in kg/m^3. Default 322 (H2O critical density).
    rho_scale : float
        Sigmoid transition width in kg/m^3. Default 50. Must be positive;
        zero produces NaN at rho=rho_min (validated at config load time).

    Returns
    -------
    float
        Weight in [0, 1]. ~1 for condensed phases, ~0 for vapor.
    """
    return 1.0 / (1.0 + np.exp(-(rho - rho_min) / rho_scale))


def _binodal_factor(eos_name, w_i, mixture, pressure, temperature, T_scale):
    """Compute combined binodal suppression for an H2 component.

    Checks the H2 component against all relevant partner components
    in the mixture:

    - H2 + silicate (MgSiO3): Rogers+2025 binodal
    - H2 + H2O: Gupta+2025 critical curve

    Returns the minimum (most restrictive) suppression weight across
    all applicable binodals. For non-H2 components, returns 1.0.

    Parameters
    ----------
    eos_name : str
        EOS identifier of the component being evaluated.
    w_i : float
        Mass fraction of the component.
    mixture : LayerMixture
        Full layer mixture (needed to identify partner components).
    pressure : float
        Pressure in Pa.
    temperature : float
        Temperature in K.
    T_scale : float
        Binodal sigmoid width in K.

    Returns
    -------
    float
        Suppression weight in [0, 1]. 1.0 if no binodal applies.
    """
    if eos_name not in _H2_EOS_NAMES:
        return 1.0

    from .binodal import gupta2025_suppression_weight, rogers2025_suppression_weight

    sigma = 1.0
    for partner, w_p in zip(mixture.components, mixture.fractions):
        if w_p <= 0:
            continue
        if partner in _SILICATE_EOS_NAMES:
            sigma = min(
                sigma,
                rogers2025_suppression_weight(pressure, temperature, w_i, w_p, T_scale),
            )
        if partner in _H2O_EOS_NAMES:
            sigma = min(
                sigma,
                gupta2025_suppression_weight(pressure, temperature, w_i, w_p, T_scale),
            )
    return sigma


def calculate_mixed_density(
    pressure,
    temperature,
    mixture,
    material_dictionaries,
    solidus_func,
    liquidus_func,
    interpolation_functions,
    mushy_zone_factors=None,
    condensed_rho_min=CONDENSED_RHO_MIN_DEFAULT,
    condensed_rho_scale=CONDENSED_RHO_SCALE_DEFAULT,
    binodal_T_scale=BINODAL_T_SCALE_DEFAULT,
):
    """Compute volume-additive mixed density for a layer mixture.

    For a single component, delegates directly to ``calculate_density()``.
    For multiple components, evaluates each independently at (P, T) and
    returns a suppressed harmonic mean where each component is weighted
    by the product of:

    1. A density-based sigmoid (``_condensed_weight``): suppresses
       gas-like components by density.
    2. A binodal sigmoid (``_binodal_factor``): suppresses H2 when it
       is thermodynamically immiscible with its partner (below the
       binodal temperature). Only applies to H2 components in mixtures
       with silicate or water.

    Notes
    -----
    The suppressed harmonic mean does not conserve mass: when a component
    is suppressed (sigma near 0), its mass is excluded from the structural
    calculation. This is physically equivalent to treating vapor-phase
    volatiles as having negligible structural contribution. The
    approximation is valid when suppressed components are at low density
    (vapor/gas) and do not contribute meaningfully to hydrostatic support.

    Parameters
    ----------
    pressure : float
        Pressure in Pa.
    temperature : float
        Temperature in K.
    mixture : LayerMixture
        Layer mixture specification.
    material_dictionaries : dict
        EOS registry.
    solidus_func : callable or None
        Solidus melting curve function.
    liquidus_func : callable or None
        Liquidus melting curve function.
    interpolation_functions : dict
        Interpolation cache.
    mushy_zone_factors : dict or float or None
        Per-EOS mushy zone factors. Dict keyed by EOS name, a single
        float (applied to all), or None (default 1.0 for all).
    condensed_rho_min : float
        Sigmoid center for phase-aware suppression (kg/m^3).
    condensed_rho_scale : float
        Sigmoid width for phase-aware suppression (kg/m^3).
    binodal_T_scale : float
        Binodal sigmoid width in K. Controls the smooth transition at
        the miscibility boundary. Default 50 K.

    Returns
    -------
    float or None
        Mixed density in kg/m^3, or None if all components are gas-like.
    """
    from .eos_functions import calculate_density

    # Fast path: single component (no suppression)
    if mixture.is_single():
        mzf = _get_mushy_zone_factor(mixture.components[0], mushy_zone_factors)
        return calculate_density(
            pressure,
            material_dictionaries,
            mixture.components[0],
            temperature,
            solidus_func,
            liquidus_func,
            interpolation_functions,
            mzf,
        )

    # Multi-component suppressed harmonic mean
    w_eff_sum = 0.0
    inv_rho_sum = 0.0
    for eos_name, w_i in zip(mixture.components, mixture.fractions):
        if w_i <= 0:
            continue
        mzf = _get_mushy_zone_factor(eos_name, mushy_zone_factors)
        rho_i = calculate_density(
            pressure,
            material_dictionaries,
            eos_name,
            temperature,
            solidus_func,
            liquidus_func,
            interpolation_functions,
            mzf,
        )
        if rho_i is None or not np.isfinite(rho_i) or rho_i <= 0:
            # Abort the entire mixture rather than skipping this component.
            # A None from calculate_density indicates an EOS table coverage
            # gap, not a vapor-phase state (vapor has low but finite density).
            # Treating it as suppressed (continue) would silently hide table
            # errors. The Picard loop falls back to old_density for this shell.
            return None
        sigma_i = _condensed_weight(rho_i, condensed_rho_min, condensed_rho_scale)
        sigma_i *= _binodal_factor(
            eos_name, w_i, mixture, pressure, temperature, binodal_T_scale
        )
        w_eff = w_i * sigma_i
        if w_eff <= 0:
            continue
        w_eff_sum += w_eff
        inv_rho_sum += w_eff / rho_i

    if w_eff_sum <= 0 or inv_rho_sum <= 0:
        return None
    return w_eff_sum / inv_rho_sum


def get_mixed_nabla_ad(
    pressure,
    temperature,
    mixture,
    material_dictionaries,
    interpolation_functions,
    solidus_func=None,
    liquidus_func=None,
    mushy_zone_factors=None,
    condensed_rho_min=CONDENSED_RHO_MIN_DEFAULT,
    condensed_rho_scale=CONDENSED_RHO_SCALE_DEFAULT,
    binodal_T_scale=BINODAL_T_SCALE_DEFAULT,
):
    """Compute mass-fraction-weighted nabla_ad for a mixture.

    For a single component, returns that component's nabla_ad directly.
    For multiple components, computes a weighted average where each
    component's contribution is scaled by the same sigmoid suppression
    used for density. This prevents vapor-phase components from
    contributing their (typically large) nabla_ad to the mixture.

    Components that do not support nabla_ad (e.g., Seager2007) are skipped
    and their fraction is redistributed among T-dependent components.

    Parameters
    ----------
    pressure : float
        Pressure in Pa.
    temperature : float
        Temperature in K.
    mixture : LayerMixture
        Layer mixture specification.
    material_dictionaries : dict
        EOS registry.
    interpolation_functions : dict
        Interpolation cache.
    solidus_func : callable or None
        Solidus function (for PALEOS-2phase).
    liquidus_func : callable or None
        Liquidus function (for PALEOS-2phase).
    mushy_zone_factors : dict or float or None
        Per-EOS mushy zone factors. Dict keyed by EOS name, a single
        float (applied to all), or None (default 1.0 for all).
    condensed_rho_min : float
        Sigmoid center for phase-aware suppression (kg/m^3).
    condensed_rho_scale : float
        Sigmoid width for phase-aware suppression (kg/m^3).

    Returns
    -------
    float or None
        Dimensionless adiabatic gradient, or None if no component provides it.

    Notes
    -----
    For multi-component mixtures, this calls ``calculate_density()`` per
    component to obtain the sigmoid weight. This is a separate EOS
    evaluation from the one in ``calculate_mixed_density()``. Since PALEOS
    tables are pure functions of (P, T), the values are identical, but the
    double call has a performance cost. A future optimization could pass
    pre-computed per-component densities.

    Suppressing a vapor component's nabla_ad means the temperature profile
    ignores the volatile. This is consistent with the density suppression
    (both treat the vapor as structurally absent) but differs from reality
    where vapor affects heat transport. This is a known limitation.
    """
    from .eos_functions import calculate_density

    if mixture.is_single():
        return _nabla_ad_for_component(
            mixture.components[0],
            pressure,
            temperature,
            material_dictionaries,
            interpolation_functions,
            solidus_func,
            liquidus_func,
        )

    # Multi-component weighted average with condensed-phase suppression
    weighted_sum = 0.0
    weight_total = 0.0

    for eos_name, w_i in zip(mixture.components, mixture.fractions):
        if w_i <= 0:
            continue
        # Compute density to determine condensed-phase weight
        mzf = _get_mushy_zone_factor(eos_name, mushy_zone_factors)
        rho_i = calculate_density(
            pressure,
            material_dictionaries,
            eos_name,
            temperature,
            solidus_func,
            liquidus_func,
            interpolation_functions,
            mzf,
        )
        if rho_i is not None and np.isfinite(rho_i) and rho_i > 0:
            sigma_i = _condensed_weight(rho_i, condensed_rho_min, condensed_rho_scale)
            sigma_i *= _binodal_factor(
                eos_name, w_i, mixture, pressure, temperature, binodal_T_scale
            )
        else:
            sigma_i = 0.0
        w_eff = w_i * sigma_i
        if w_eff <= 0:
            continue
        nabla = _nabla_ad_for_component(
            eos_name,
            pressure,
            temperature,
            material_dictionaries,
            interpolation_functions,
            solidus_func,
            liquidus_func,
        )
        if nabla is not None and np.isfinite(nabla) and nabla >= 0:
            weighted_sum += w_eff * nabla
            weight_total += w_eff

    if weight_total <= 0:
        return None

    # Renormalize by actual weight of contributing components
    return weighted_sum / weight_total


def _nabla_ad_for_component(
    eos_name,
    pressure,
    temperature,
    material_dictionaries,
    interpolation_functions,
    solidus_func,
    liquidus_func,
):
    """Get nabla_ad for a single EOS component.

    Routes to the appropriate lookup based on EOS type.

    Parameters
    ----------
    eos_name : str
        EOS identifier string (e.g. ``"PALEOS:MgSiO3"``).
    pressure : float
        Pressure in Pa.
    temperature : float
        Temperature in K.
    material_dictionaries : dict
        EOS registry.
    interpolation_functions : dict
        Interpolation cache.
    solidus_func : callable or None
        Solidus function (for PALEOS-2phase). None for other EOS types.
    liquidus_func : callable or None
        Liquidus function (for PALEOS-2phase). None for other EOS types.

    Returns
    -------
    float or None
        Dimensionless adiabatic gradient, or None if the EOS does not
        support nabla_ad (e.g. Seager2007, Analytic).
    """
    from .eos_functions import (
        _compute_paleos_dtdp,
        _get_paleos_unified_nabla_ad,
    )

    if eos_name not in TDEP_EOS_NAMES:
        return None

    mat = material_dictionaries.get(eos_name, {})

    if mat.get('format') == 'paleos_unified':
        return _get_paleos_unified_nabla_ad(pressure, temperature, mat, interpolation_functions)

    if eos_name == 'PALEOS-2phase:MgSiO3':
        # Convert dT/dP back to nabla_ad = (dT/dP) * P / T
        if pressure <= 0 or temperature <= 0:
            return None
        dtdp = _compute_paleos_dtdp(
            pressure,
            temperature,
            mat,
            solidus_func,
            liquidus_func,
            interpolation_functions,
        )
        if dtdp is not None and dtdp > 0:
            return dtdp * pressure / temperature
        return None

    # WolfBower2018 / RTPress100TPa: use adiabat_grad_file
    # These provide dT/dP, not nabla_ad. Convert.
    from .eos_functions import get_tabulated_eos

    grad_file = mat.get('melted_mantle', {}).get('adiabat_grad_file')
    if grad_file is None:
        return None
    dtdp_dict = {'melted_mantle': {'eos_file': grad_file}}
    dtdp = get_tabulated_eos(
        pressure, dtdp_dict, 'melted_mantle', temperature, interpolation_functions
    )
    if dtdp is not None and dtdp > 0 and pressure > 0 and temperature > 0:
        return dtdp * pressure / temperature
    return None
