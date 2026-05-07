"""Unit tests for ``zalmoxis.phase_columns``.

Verifies the chemistry-label mapping, the PALEOS phase-string
canonicalisation, the per-component phase router (Chabrier hardcoding,
melting-curve route, Seager fallback), and the per-shell column builder
across single-component, multi-component, and 2-vs-3-layer planet
configs.
"""

from __future__ import annotations

import numpy as np
import pytest

from zalmoxis.phase_columns import (
    PHASE_UNKNOWN,
    _canonicalize_paleos_phase,
    _load_phase_grid,
    compute_layer_phase_columns,
    eos_to_chemistry,
    phase_for_component,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# eos_to_chemistry
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    'eos_name, expected',
    [
        ('PALEOS:iron', 'Fe'),
        ('PALEOS:MgSiO3', 'MgSiO3'),
        ('PALEOS:H2O', 'H2O'),
        ('Chabrier:H', 'H2'),
        ('PALEOS-API:iron', 'Fe'),
        ('PALEOS-2phase:MgSiO3', 'MgSiO3'),
        ('PALEOS-2phase:MgSiO3-highres', 'MgSiO3'),
        ('Seager2007:iron', 'Fe'),
        ('Seager2007:H2O', 'H2O'),
        ('WolfBower2018:MgSiO3', 'MgSiO3'),
        ('RTPress100TPa:MgSiO3', 'MgSiO3'),
        # Edge cases.
        ('', PHASE_UNKNOWN),
        ('NoColon', PHASE_UNKNOWN),
        # Unrecognised but parseable: pass through the suffix.
        ('Foo:SiC', 'SiC'),
        # Three-segment EOS names whose last segment is NOT a float mass
        # fraction are passed through unchanged by parse_layer_components;
        # the chemistry label must still be the bare material token.
        # Discriminates a 'split-on-first-colon-only' bug.
        ('PALEOS:MgSiO3:phaseA', 'MgSiO3'),
        ('PALEOS-2phase:MgSiO3-highres:phaseA', 'MgSiO3'),
    ],
)
def test_eos_to_chemistry_known_and_edge_cases(eos_name, expected):
    """Maps every shipped EOS string to the canonical 4-element chemistry
    set, leaves unrecognised suffixes intact, and returns the unknown
    sentinel for malformed input."""
    assert eos_to_chemistry(eos_name) == expected


# ---------------------------------------------------------------------------
# _canonicalize_paleos_phase
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    'raw, expected',
    [
        # Solid variants (real labels from the shipped tables).
        ('solid-ice-X', 'solid'),
        ('solid-ice-Ih', 'solid'),
        ('solid-epsilon-hcp', 'solid'),
        ('solid-alpha-bcc', 'solid'),
        ('solid-ppv', 'solid'),
        ('solid-brg', 'solid'),
        # Other phases.
        ('liquid', 'liquid'),
        ('vapor', 'gas'),
        ('gas', 'gas'),
        ('supercritical', 'supercritical'),
        # Whitespace + case insensitivity.
        ('  LIQUID  ', 'liquid'),
        ('Solid-ice-VII', 'solid'),
        # Unknown / malformed.
        ('', PHASE_UNKNOWN),
        ('bogus', PHASE_UNKNOWN),
        # Chabrier-style chemistry-state labels are NOT valid phase
        # strings; the helper must reject them rather than masquerading
        # them as 'solid' or 'liquid'.
        ('atomic', PHASE_UNKNOWN),
        ('molecular', PHASE_UNKNOWN),
        ('dissociating:cv_unre', PHASE_UNKNOWN),
    ],
)
def test_canonicalize_paleos_phase_covers_all_table_labels(raw, expected):
    """Every phase string actually emitted by the shipped PALEOS unified
    tables maps into the canonical set, and out-of-band labels return
    the unknown sentinel rather than silently coercing."""
    assert _canonicalize_paleos_phase(raw) == expected


# ---------------------------------------------------------------------------
# phase_for_component: Chabrier:H hardcoding
# ---------------------------------------------------------------------------
def test_chabrier_hydrogen_supercritical_above_critical_point():
    """Chabrier:H above the H2 critical point (33 K, 1.3 MPa) must
    return 'supercritical', regardless of EOS-table phase strings.
    Planetary interior conditions almost always satisfy this."""
    assert phase_for_component('Chabrier:H', P=1.0e9, T=2000.0) == 'supercritical'
    assert phase_for_component('Chabrier:H', P=1.0e10, T=10000.0) == 'supercritical'


def test_chabrier_hydrogen_gas_below_critical_pressure():
    """Below the H2 critical pressure, return 'gas'. Discriminates the
    'always-supercritical' bug where the threshold is missing."""
    assert phase_for_component('Chabrier:H', P=1.0e5, T=1000.0) == 'gas'
    assert phase_for_component('Chabrier:H', P=1.0e6, T=1000.0) == 'gas'


def test_chabrier_hydrogen_gas_below_critical_temperature():
    """Below the H2 critical temperature (33 K), return 'gas' even at
    very high pressure (would otherwise hide a missing T-threshold)."""
    assert phase_for_component('Chabrier:H', P=1.0e10, T=20.0) == 'gas'


def test_phase_for_component_rejects_nonfinite_PT():
    """Non-finite (P, T) returns the unknown sentinel rather than
    propagating NaN / inf into the lookup."""
    assert phase_for_component('Chabrier:H', P=np.nan, T=300.0) == PHASE_UNKNOWN
    assert phase_for_component('Chabrier:H', P=1.0e9, T=np.inf) == PHASE_UNKNOWN


def test_phase_for_component_unknown_eos_returns_unknown():
    """An EOS identifier not in the registry (and not a Seager2007
    variant) returns the unknown sentinel."""
    assert phase_for_component('NotARealEOS:foo', P=1.0e9, T=1000.0) == PHASE_UNKNOWN


def test_load_phase_grid_missing_file_returns_none(tmp_path):
    """When neither the ``.pkl`` binary cache nor the ``.dat`` text
    table exists for an EOS file path, ``_load_phase_grid`` must
    return None and cache that result, rather than raising on the
    second call."""
    import zalmoxis.phase_columns as pc

    bogus = str(tmp_path / 'definitely_not_a_real_paleos_table.dat')
    pc._PHASE_GRID_CACHE.pop(bogus, None)

    first = _load_phase_grid(bogus)
    second = _load_phase_grid(bogus)
    assert first is None
    assert second is None
    # The cache entry must be the None sentinel, not a missing key, so
    # the second call hits the early-return path rather than re-trying
    # the disk read.
    assert bogus in pc._PHASE_GRID_CACHE
    assert pc._PHASE_GRID_CACHE[bogus] is None


def test_load_phase_grid_uses_pkl_cache_when_dat_absent(tmp_path, monkeypatch):
    """``_load_phase_grid`` must route through ``_ensure_unified_cache``
    so a pickle-only installation (``.pkl`` present, ``.dat`` absent)
    still produces phase labels rather than ``unknown``. This is the
    Codex P2 fix: the prior implementation gated on ``os.path.isfile``
    on the ``.dat`` path and returned None whenever the text table
    was missing, even when the ``.pkl`` binary cache existed."""
    import pickle

    import zalmoxis.phase_columns as pc

    eos_file = str(tmp_path / 'pkl_only.dat')  # NB: never created
    pkl_path = eos_file.replace('.dat', '.pkl')

    # Hand-built minimal cache entry covering the keys
    # ``_phase_from_unified_grid`` reads.
    payload = {
        'logp_min': 9.0,
        'logp_max': 11.0,
        'dlog_p': 0.5,
        'n_p': 5,
        'logt_min': 3.0,
        'logt_max': 4.0,
        'dlog_t': 0.25,
        'n_t': 5,
        'phase_grid': np.array([['solid'] * 5 for _ in range(5)], dtype=object),
    }
    with open(pkl_path, 'wb') as fh:
        pickle.dump(payload, fh, protocol=4)

    pc._PHASE_GRID_CACHE.pop(eos_file, None)
    cached = _load_phase_grid(eos_file)

    # Cache hit must look like the pickled payload (not None and not the
    # text-loaded shape).
    assert cached is not None
    assert cached['n_p'] == 5
    assert cached['phase_grid'][0, 0] == 'solid'

    # Clean up so subsequent tests don't see a stale cached entry.
    pc._PHASE_GRID_CACHE.pop(eos_file, None)


def test_phase_for_component_seager_returns_solid():
    """Seager2007:* tables are 300 K static lookups for solid-phase Fe /
    silicate / water; emit 'solid' regardless of the queried (P, T)."""
    assert phase_for_component('Seager2007:iron', P=1.0e11, T=5000.0) == 'solid'
    assert phase_for_component('Seager2007:MgSiO3', P=1.0e10, T=2000.0) == 'solid'
    assert phase_for_component('Seager2007:H2O', P=1.0e9, T=500.0) == 'solid'


# ---------------------------------------------------------------------------
# phase_for_component: 2-phase melting-curve path
# ---------------------------------------------------------------------------
def _curves(T_solidus, T_liquidus):
    """Return constant solidus / liquidus callables for routing tests."""
    return (lambda _P: T_solidus, lambda _P: T_liquidus)


def test_phase_from_melting_curves_solid_below_solidus():
    curves = _curves(2500.0, 4000.0)
    assert (
        phase_for_component(
            'PALEOS-2phase:MgSiO3',
            P=1.0e11,
            T=2000.0,
            melting_curves_functions=curves,
        )
        == 'solid'
    )


def test_phase_from_melting_curves_mixed_in_mushy_zone():
    """Discriminates between 'solid' (below T_sol) and 'liquid' (above
    T_liq); the mushy regime returns the dedicated 'mixed' label."""
    curves = _curves(2500.0, 4000.0)
    # Right at the solidus.
    assert (
        phase_for_component(
            'PALEOS-2phase:MgSiO3', P=1.0e11, T=2500.0, melting_curves_functions=curves
        )
        == 'mixed'
    )
    # Mid-mushy.
    assert (
        phase_for_component(
            'PALEOS-2phase:MgSiO3', P=1.0e11, T=3250.0, melting_curves_functions=curves
        )
        == 'mixed'
    )
    # Just below liquidus.
    assert (
        phase_for_component(
            'PALEOS-2phase:MgSiO3', P=1.0e11, T=3999.0, melting_curves_functions=curves
        )
        == 'mixed'
    )


def test_phase_from_melting_curves_liquid_above_liquidus():
    curves = _curves(2500.0, 4000.0)
    assert (
        phase_for_component(
            'WolfBower2018:MgSiO3',
            P=1.0e11,
            T=5000.0,
            melting_curves_functions=curves,
        )
        == 'liquid'
    )


def test_phase_from_melting_curves_degenerate_solidus_equals_liquidus():
    """When T_solidus >= T_liquidus (degenerate input), the helper must
    not divide by zero; it returns 'solid' below and 'liquid' at/above.
    Discriminates from the 'always-mixed' bug at the boundary."""
    curves = _curves(3000.0, 3000.0)
    assert (
        phase_for_component(
            'RTPress100TPa:MgSiO3', P=1.0e11, T=2999.0, melting_curves_functions=curves
        )
        == 'solid'
    )
    assert (
        phase_for_component(
            'RTPress100TPa:MgSiO3', P=1.0e11, T=3000.0, melting_curves_functions=curves
        )
        == 'liquid'
    )


def test_phase_from_melting_curves_missing_curves_returns_unknown():
    """No solidus / liquidus passed: the 2-phase route cannot decide;
    must return 'unknown' rather than silently defaulting to 'solid'."""
    assert (
        phase_for_component(
            'PALEOS-2phase:MgSiO3', P=1.0e11, T=3000.0, melting_curves_functions=None
        )
        == PHASE_UNKNOWN
    )


def test_phase_from_melting_curves_nan_solidus_returns_unknown():
    """If the solidus or liquidus function returns NaN (e.g. queried
    out of its valid pressure range), return 'unknown' rather than
    making a phase claim from garbage."""
    curves = (lambda _P: np.nan, lambda _P: 4000.0)
    assert (
        phase_for_component(
            'PALEOS-2phase:MgSiO3', P=1.0e11, T=3000.0, melting_curves_functions=curves
        )
        == PHASE_UNKNOWN
    )


# ---------------------------------------------------------------------------
# compute_layer_phase_columns: layer assignment
# ---------------------------------------------------------------------------
def _three_layer_profile(n=10):
    """Synthetic monotone (P, T, M) arrays for layer-assignment tests."""
    M_total = 5.972e24
    return {
        'pressure': np.linspace(3.0e11, 1.0e5, n),
        'temperature': np.linspace(6000.0, 1000.0, n),
        'mass_enclosed': np.linspace(0.0, M_total, n),
        'cmb_mass': 0.30 * M_total,
        'core_mantle_mass': 0.80 * M_total,
    }


def test_compute_layer_phase_columns_three_layer_assignment():
    """Each shell must be assigned to the correct layer's chemistry:
    core for m < cmb_mass, mantle for m < core_mantle_mass, ice
    otherwise. Discriminates the off-by-one boundary bugs."""
    prof = _three_layer_profile(n=10)
    components, phases = compute_layer_phase_columns(
        pressure=prof['pressure'],
        temperature=prof['temperature'],
        mass_enclosed=prof['mass_enclosed'],
        cmb_mass=prof['cmb_mass'],
        core_mantle_mass=prof['core_mantle_mass'],
        layer_eos_config={
            'core': 'Seager2007:iron',
            'mantle': 'Seager2007:MgSiO3',
            'ice_layer': 'Seager2007:H2O',
        },
    )
    M_total = prof['mass_enclosed'][-1]
    expected_components = []
    for m in prof['mass_enclosed']:
        if m < 0.30 * M_total:
            expected_components.append('Fe')
        elif m < 0.80 * M_total:
            expected_components.append('MgSiO3')
        else:
            expected_components.append('H2O')
    assert components == expected_components

    # Seager2007:* always emits 'solid' for any (P, T).
    assert all(p == 'solid' for p in phases)

    # The set of distinct components must match the configured layers.
    assert set(components) == {'Fe', 'MgSiO3', 'H2O'}


def test_compute_layer_phase_columns_two_layer_top_shell_routes_to_mantle():
    """In a 2-layer config, ``core_mantle_mass == M_total`` and the top
    shell sits exactly at that mass. Without the fall-through to mantle
    it would be labelled 'unknown'; this test pins the convention."""
    n = 6
    M = 5.972e24
    components, phases = compute_layer_phase_columns(
        pressure=np.linspace(3.0e11, 1.0e5, n),
        temperature=np.linspace(6000.0, 1000.0, n),
        mass_enclosed=np.linspace(0.0, M, n),
        cmb_mass=0.325 * M,
        core_mantle_mass=M,
        layer_eos_config={
            'core': 'Seager2007:iron',
            'mantle': 'Seager2007:MgSiO3',
            'ice_layer': '',
        },
    )
    assert components[0] == 'Fe'
    # Top shell must NOT be 'unknown' (would happen if the topmost mass
    # ever-so-slightly exceeds core_mantle_mass and ice_layer is empty).
    assert components[-1] == 'MgSiO3'
    assert PHASE_UNKNOWN not in components
    assert all(p == 'solid' for p in phases)


def test_compute_layer_phase_columns_multicomponent_picks_dominant():
    """A multi-component layer ``MgSiO3:0.6+H2O:0.4`` must surface
    'MgSiO3' as the main component (highest mass fraction), not a
    blend / first-listed / last-listed."""
    n = 5
    M = 5.972e24
    components, _phases = compute_layer_phase_columns(
        pressure=np.linspace(3.0e11, 1.0e5, n),
        temperature=np.linspace(6000.0, 1000.0, n),
        mass_enclosed=np.linspace(0.0, M, n),
        cmb_mass=0.30 * M,
        core_mantle_mass=M,
        layer_eos_config={
            'core': 'Seager2007:iron',
            'mantle': 'Seager2007:MgSiO3:0.6+Seager2007:H2O:0.4',
            'ice_layer': '',
        },
    )
    # Mantle shells: dominant is MgSiO3 (0.6 > 0.4).
    mantle_components = [c for c, m in zip(components, np.linspace(0.0, M, n)) if m >= 0.30 * M]
    assert all(c == 'MgSiO3' for c in mantle_components)


def test_compute_layer_phase_columns_returns_consistent_lengths():
    """Output arrays must always match the radial-grid length, even
    when layer EOS is missing for a region."""
    n = 7
    M = 5.972e24
    components, phases = compute_layer_phase_columns(
        pressure=np.linspace(3.0e11, 1.0e5, n),
        temperature=np.linspace(6000.0, 1000.0, n),
        mass_enclosed=np.linspace(0.0, M, n),
        cmb_mass=0.30 * M,
        core_mantle_mass=0.80 * M,
        layer_eos_config={'core': 'Seager2007:iron'},  # mantle / ice missing
    )
    assert len(components) == n
    assert len(phases) == n
    # Shells above the CMB must fall back to 'unknown' since neither
    # mantle nor ice EOS were provided.
    assert components[-1] == PHASE_UNKNOWN
    assert phases[-1] == PHASE_UNKNOWN


def test_compute_layer_phase_columns_chabrier_envelope_supercritical():
    """A 3-layer config with a Chabrier:H envelope as the ice layer
    produces 'supercritical' for shells above core_mantle_mass."""
    n = 8
    M = 5.972e24
    pressure = np.linspace(3.0e11, 1.0e7, n)  # all > critical P
    temperature = np.linspace(6000.0, 200.0, n)  # all > critical T
    mass = np.linspace(0.0, M, n)
    components, phases = compute_layer_phase_columns(
        pressure=pressure,
        temperature=temperature,
        mass_enclosed=mass,
        cmb_mass=0.30 * M,
        core_mantle_mass=0.60 * M,
        layer_eos_config={
            'core': 'Seager2007:iron',
            'mantle': 'Seager2007:MgSiO3',
            'ice_layer': 'Chabrier:H',
        },
    )
    envelope_phases = [p for p, m in zip(phases, mass) if m >= 0.60 * M]
    envelope_components = [c for c, m in zip(components, mass) if m >= 0.60 * M]
    assert envelope_components and all(c == 'H2' for c in envelope_components)
    assert all(p == 'supercritical' for p in envelope_phases)
