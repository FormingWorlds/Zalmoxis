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

from .constants import TDEP_EOS_NAMES

logger = logging.getLogger(__name__)


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
        if len(self.components) != len(self.fractions):
            raise ValueError(
                f'LayerMixture: components ({len(self.components)}) and '
                f'fractions ({len(self.fractions)}) must have the same length.'
            )
        if len(self.components) > 1:
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


def calculate_mixed_density(
    pressure,
    temperature,
    mixture,
    material_dictionaries,
    solidus_func,
    liquidus_func,
    interpolation_functions,
    mushy_zone_factor=1.0,
):
    """Compute volume-additive mixed density for a layer mixture.

    For a single component, delegates directly to ``calculate_density()``.
    For multiple components, evaluates each independently at (P, T) and
    returns the harmonic mean: ``rho_mix = (sum w_i / rho_i)^{-1}``.

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
    mushy_zone_factor : float
        Mushy zone factor for unified PALEOS.

    Returns
    -------
    float or None
        Mixed density in kg/m^3, or None if any component fails.
    """
    from .eos_functions import calculate_density

    # Fast path: single component
    if mixture.is_single():
        return calculate_density(
            pressure,
            material_dictionaries,
            mixture.components[0],
            temperature,
            solidus_func,
            liquidus_func,
            interpolation_functions,
            mushy_zone_factor,
        )

    # Multi-component harmonic mean
    inv_rho_sum = 0.0
    for eos_name, w_i in zip(mixture.components, mixture.fractions):
        if w_i <= 0:
            continue
        rho_i = calculate_density(
            pressure,
            material_dictionaries,
            eos_name,
            temperature,
            solidus_func,
            liquidus_func,
            interpolation_functions,
            mushy_zone_factor,
        )
        if rho_i is None or not np.isfinite(rho_i) or rho_i <= 0:
            return None
        inv_rho_sum += w_i / rho_i

    if inv_rho_sum <= 0:
        return None
    return 1.0 / inv_rho_sum


def get_mixed_nabla_ad(
    pressure,
    temperature,
    mixture,
    material_dictionaries,
    interpolation_functions,
    solidus_func=None,
    liquidus_func=None,
):
    """Compute mass-fraction-weighted nabla_ad for a mixture.

    For a single component, returns that component's nabla_ad directly.
    For multiple components, returns ``sum(w_i * nabla_ad_i)``.

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

    Returns
    -------
    float or None
        Dimensionless adiabatic gradient, or None if no component provides it.
    """

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

    # Multi-component weighted average
    weighted_sum = 0.0
    weight_total = 0.0

    for eos_name, w_i in zip(mixture.components, mixture.fractions):
        if w_i <= 0:
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
            weighted_sum += w_i * nabla
            weight_total += w_i

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

    Returns
    -------
    float or None
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
