"""Per-radial-shell main-component and phase labelling.

Helpers used by the standalone grid runner to enrich each per-cell
profile CSV with two extra columns:

- ``main_component``: the dominant (highest mass-fraction) chemistry of
  the layer the shell sits in, as a short label (``'Fe'``, ``'MgSiO3'``,
  ``'H2O'``, ``'H2'``).
- ``phase``: the phase of that component at the shell's (P, T), as one
  of ``'solid'``, ``'liquid'``, ``'gas'``, ``'supercritical'``,
  ``'mixed'``, or ``'unknown'``.

Phase routing
-------------
- Unified PALEOS tables (``PALEOS:iron``, ``PALEOS:MgSiO3``,
  ``PALEOS:H2O``): read the per-cell phase string out of the table grid
  and canonicalise (e.g. ``'solid-ice-X'`` -> ``'solid'``,
  ``'vapor'`` -> ``'gas'``).
- ``Chabrier:H``: the table column carries chemistry-state labels
  (``'molecular'``, ``'atomic'``, ``'dissociating'``, ``'unphysical'``)
  rather than phase. Hardcode ``'supercritical'`` whenever the shell is
  above the critical point of H2 (P > 1.3 MPa, T > 33 K), which is
  always satisfied for planetary interior conditions.
- 2-phase silicate EOS (``PALEOS-2phase:*``, ``WolfBower2018:*``,
  ``RTPress100TPa:*``): use the configured solidus / liquidus to label
  ``'solid'`` / ``'mixed'`` / ``'liquid'``.
- ``Seager2007:*``: 300 K static lookup tables for solid-phase Fe /
  silicate / water; emit ``'solid'``.
- Anything else: ``'unknown'``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from .mixing import parse_layer_components

logger = logging.getLogger(__name__)

PHASE_UNKNOWN = 'unknown'


def eos_to_chemistry(eos_name: str) -> str:
    """Map an EOS identifier to a short chemistry label.

    Parameters
    ----------
    eos_name : str
        EOS identifier as it appears in ``[EOS]`` config (e.g.
        ``'PALEOS:iron'``, ``'Chabrier:H'``,
        ``'PALEOS-2phase:MgSiO3-highres'``).

    Returns
    -------
    str
        Short chemistry label: ``'Fe'``, ``'MgSiO3'``, ``'H2O'``,
        ``'H2'``, or the trimmed material suffix as a last resort
        (``'SiC'``, ``'C'``, etc.). Returns ``'unknown'`` for empty or
        unparseable inputs.
    """
    if not eos_name or ':' not in eos_name:
        return PHASE_UNKNOWN
    suffix = eos_name.split(':', 1)[1]
    # Strip resolution / variant suffixes like '-highres'.
    base = suffix.split('-', 1)[0]
    # Defensive against 3-segment EOS names that ``parse_layer_components``
    # passes through unchanged when the third segment is not a float
    # mass-fraction (e.g. a hypothetical ``PALEOS:MgSiO3:phaseA``); strip
    # any trailing ``:...`` so the chemistry token stays clean.
    base = base.split(':', 1)[0]
    base_lower = base.lower()
    if base_lower in ('iron', 'fe'):
        return 'Fe'
    if base == 'H' or base_lower == 'h':
        # Chabrier hydrogen tables are H2 in the molecular regime; the
        # caller wants a single chemistry label across all (P, T).
        return 'H2'
    return base


def _canonicalize_paleos_phase(raw: str) -> str:
    """Map a PALEOS table phase string to one of the canonical labels.

    Parameters
    ----------
    raw : str
        The phase column entry from a unified PALEOS table (e.g.
        ``'solid-ice-X'``, ``'solid-epsilon-hcp'``, ``'liquid'``,
        ``'vapor'``, ``'supercritical'``).

    Returns
    -------
    str
        One of ``'solid'``, ``'liquid'``, ``'gas'``, ``'supercritical'``,
        or ``'unknown'``.
    """
    if not raw:
        return PHASE_UNKNOWN
    low = raw.strip().lower()
    if low.startswith('solid'):
        return 'solid'
    if low == 'liquid':
        return 'liquid'
    if low in ('vapor', 'gas'):
        return 'gas'
    if low.startswith('supercritical'):
        return 'supercritical'
    return PHASE_UNKNOWN


# Per-process cache for unified PALEOS phase grids. Keyed by absolute
# path of the table file. Built once on first hit and reused for every
# subsequent radial shell across every grid cell run by this worker.
# Entries are the full ``load_paleos_unified_table`` cache so that this
# loader hits the same ``.pkl`` binary cache that the solver path uses
# via ``_ensure_unified_cache``: cold ``.pkl`` skips the slow text reparse
# of the 50-140 MB ``.dat`` file, and an installation that has only the
# ``.pkl`` (no ``.dat``) still yields phase labels rather than ``unknown``.
_PHASE_GRID_CACHE: dict[str, Optional[dict]] = {}


def _load_phase_grid(eos_file: str) -> Optional[dict]:
    """Load (and cache) the (P-major, T-minor) phase string grid.

    Routes through the shared ``_ensure_unified_cache`` so the lookup
    benefits from the existing ``.pkl`` binary cache. Falls back to
    ``None`` if neither the ``.pkl`` nor the ``.dat`` file is reachable
    (e.g. on a CI runner without the bootstrapped ``data/`` tree).

    Parameters
    ----------
    eos_file : str
        Absolute path to a unified PALEOS table (``.dat`` filename; the
        cached ``.pkl`` is derived from it by ``_ensure_unified_cache``).

    Returns
    -------
    dict or None
        The shared cache entry, or ``None`` if no source was found.
    """
    if eos_file in _PHASE_GRID_CACHE:
        return _PHASE_GRID_CACHE[eos_file]

    from .eos.interpolation import _ensure_unified_cache

    try:
        cached = _ensure_unified_cache(eos_file, _PHASE_GRID_CACHE)
    except (FileNotFoundError, OSError) as exc:
        logger.debug('phase_columns: no PALEOS source for %s (%s)', eos_file, exc)
        _PHASE_GRID_CACHE[eos_file] = None
        return None

    return cached


def _phase_from_unified_grid(eos_file: str, P: float, T: float) -> str:
    """Nearest-cell phase string lookup in a unified PALEOS table."""
    grid = _load_phase_grid(eos_file)
    if grid is None or not (P > 0 and T > 0):
        return PHASE_UNKNOWN
    log_p = float(np.log10(P))
    log_t = float(np.log10(T))
    log_p_clamped = min(max(log_p, grid['logp_min']), grid['logp_max'])
    log_t_clamped = min(max(log_t, grid['logt_min']), grid['logt_max'])
    if grid['dlog_p'] > 0 and grid['n_p'] > 1:
        ip = int(round((log_p_clamped - grid['logp_min']) / grid['dlog_p']))
    else:  # pragma: no cover - degenerate single-row pressure grid; defensive
        ip = 0
    if grid['dlog_t'] > 0 and grid['n_t'] > 1:
        it = int(round((log_t_clamped - grid['logt_min']) / grid['dlog_t']))
    else:  # pragma: no cover - degenerate single-row temperature grid; defensive
        it = 0
    ip = min(max(ip, 0), grid['n_p'] - 1)
    it = min(max(it, 0), grid['n_t'] - 1)
    return _canonicalize_paleos_phase(str(grid['phase_grid'][ip, it]))


def _phase_from_melting_curves(P, T, melting_curves_functions) -> str:
    """Solid / mixed / liquid label from solidus & liquidus at (P, T)."""
    if not melting_curves_functions:
        return PHASE_UNKNOWN
    solidus_func, liquidus_func = melting_curves_functions
    if (
        solidus_func is None or liquidus_func is None
    ):  # pragma: no cover - tuple with None inner curves; defensive
        return PHASE_UNKNOWN
    try:
        T_sol = float(solidus_func(P))
        T_liq = float(liquidus_func(P))
    except Exception:  # pragma: no cover - melting-curve evaluation error; defensive
        return PHASE_UNKNOWN
    if not (np.isfinite(T_sol) and np.isfinite(T_liq)):
        return PHASE_UNKNOWN
    if T_liq <= T_sol:
        return 'liquid' if T >= T_sol else 'solid'
    if T < T_sol:
        return 'solid'
    if T <= T_liq:
        return 'mixed'
    return 'liquid'


# Critical-point thresholds for hardcoded Chabrier:H phase. H2 critical
# point is (~33 K, ~1.3 MPa); planetary interiors are far above both.
_H2_T_CRIT = 33.0
_H2_P_CRIT = 1.3e6


def phase_for_component(
    eos_name: str,
    P: float,
    T: float,
    *,
    melting_curves_functions=None,
) -> str:
    """Phase label for a single EOS component at (P, T).

    See module docstring for the full routing. Returns ``'unknown'``
    when no rule applies, when (P, T) is non-finite, or when a needed
    EOS file is missing on disk.
    """
    if not eos_name:
        return PHASE_UNKNOWN
    if not (np.isfinite(P) and np.isfinite(T)):
        return PHASE_UNKNOWN

    if eos_name == 'Chabrier:H':
        if P > _H2_P_CRIT and T > _H2_T_CRIT:
            return 'supercritical'
        return 'gas'

    from .eos_properties import EOS_REGISTRY

    entry = EOS_REGISTRY.get(eos_name)
    if entry is None:
        if eos_name.startswith('Seager2007:'):
            return 'solid'
        return PHASE_UNKNOWN

    if isinstance(entry, dict):
        fmt = entry.get('format')
        eos_file = entry.get('eos_file')

        if fmt == 'paleos_unified' and eos_file:
            return _phase_from_unified_grid(eos_file, P, T)

        if 'melted_mantle' in entry and 'solid_mantle' in entry:
            return _phase_from_melting_curves(P, T, melting_curves_functions)

        # PALEOS-API: producer cache lookup happens elsewhere; phase is
        # not currently surfaced through that path.
        if fmt == 'paleos_api':
            return PHASE_UNKNOWN

    if eos_name.startswith('Seager2007:'):
        return 'solid'

    return PHASE_UNKNOWN


def compute_layer_phase_columns(
    *,
    pressure: np.ndarray,
    temperature: np.ndarray,
    mass_enclosed: np.ndarray,
    cmb_mass: float,
    core_mantle_mass: float,
    layer_eos_config: dict,
    melting_curves_functions=None,
) -> tuple[list[str], list[str]]:
    """Compute per-shell ``main_component`` and ``phase`` columns.

    Each shell is assigned to a layer (``core``, ``mantle``,
    ``ice_layer``) by comparing ``mass_enclosed`` against ``cmb_mass``
    and ``core_mantle_mass``. The dominant component of that layer's
    EOS mixture (highest mass fraction) drives both columns.

    Parameters
    ----------
    pressure, temperature, mass_enclosed : numpy.ndarray
        Per-shell SI quantities (Pa, K, kg).
    cmb_mass, core_mantle_mass : float
        Layer-boundary masses in kg.
    layer_eos_config : dict
        Mapping of layer name to EOS config string (``'PALEOS:iron'``,
        ``'PALEOS:MgSiO3:0.85+PALEOS:H2O:0.15'``, ...).
    melting_curves_functions : tuple or None
        ``(solidus_func, liquidus_func)`` for 2-phase silicate EOS;
        ignored for unified-PALEOS / Chabrier / Seager paths.

    Returns
    -------
    components : list of str
    phases : list of str
        Both have the same length as ``mass_enclosed``.
    """
    layer_eos_config = layer_eos_config or {}

    primary_eos: dict[str, str] = {}
    for layer in ('core', 'mantle', 'ice_layer'):
        eos_str = layer_eos_config.get(layer, '')
        if not eos_str:
            continue
        try:
            primary_eos[layer] = parse_layer_components(eos_str).primary()
        except ValueError as exc:
            logger.debug('phase_columns: skipping layer %r (%s)', layer, exc)

    chem_per_layer = {layer: eos_to_chemistry(eos) for layer, eos in primary_eos.items()}

    n = len(mass_enclosed)
    components = [PHASE_UNKNOWN] * n
    phases = [PHASE_UNKNOWN] * n

    core_eos = primary_eos.get('core', '')
    mantle_eos = primary_eos.get('mantle', '')
    ice_eos = primary_eos.get('ice_layer', '')

    for i in range(n):
        m_i = float(mass_enclosed[i])
        if m_i < cmb_mass and core_eos:
            layer, eos_name = 'core', core_eos
        elif m_i < core_mantle_mass and mantle_eos:
            layer, eos_name = 'mantle', mantle_eos
        elif ice_eos:
            layer, eos_name = 'ice_layer', ice_eos
        elif mantle_eos:
            # 2-layer convention: top shell sits exactly at
            # ``core_mantle_mass == M_total`` and would otherwise fall
            # through; route to mantle.
            layer, eos_name = 'mantle', mantle_eos
        else:
            continue
        components[i] = chem_per_layer.get(layer, PHASE_UNKNOWN)
        phases[i] = phase_for_component(
            eos_name,
            float(pressure[i]),
            float(temperature[i]),
            melting_curves_functions=melting_curves_functions,
        )

    return components, phases
