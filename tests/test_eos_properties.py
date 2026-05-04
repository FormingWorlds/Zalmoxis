"""Coverage tests for the lazy ``EOS_REGISTRY`` and legacy material-dict
``__getattr__`` accessor in ``zalmoxis.eos_properties``.

The existing test suite exercises ``EOS_REGISTRY[key]`` and ``key in
EOS_REGISTRY`` indirectly via its consumers. The remaining paths are
the dict-like wrapper's iteration / size / ``get`` / ``keys`` /
``values`` / ``items`` / ``__repr__`` methods, plus the module-level
``__getattr__`` for legacy material dicts.

Anti-happy-path: an unknown legacy attribute name must raise
``AttributeError``, an unknown registry key via ``get`` must return
the default, and ``__contains__`` must return False for an unregistered
key.
"""

from __future__ import annotations

import pytest

import zalmoxis.eos_properties as eos_properties
from zalmoxis.eos_properties import EOS_REGISTRY

pytestmark = pytest.mark.unit


class TestLazyRegistryWrapper:
    """Exercise every dict-like method on ``_LazyRegistry``."""

    def test_iter_yields_all_registered_keys(self):
        """Iterating must yield every registered EOS identifier and at
        minimum cover the four families in production use."""
        keys = set(iter(EOS_REGISTRY))
        # These four are load-bearing for the PROTEUS solver path and
        # must always be present.
        for must_have in (
            'Seager2007:iron',
            'WolfBower2018:MgSiO3',
            'PALEOS:iron',
            'PALEOS-API:iron',
        ):
            assert must_have in keys
        # Boundary: the registry is non-trivial (>= 10 entries currently).
        assert len(keys) >= 10

    def test_len_matches_iter_length(self):
        """``len`` must equal the count of iterated keys (consistency)."""
        assert len(EOS_REGISTRY) == sum(1 for _ in EOS_REGISTRY)

    def test_get_returns_value_for_known_key(self):
        """``get`` returns the registry entry for a known key."""
        entry = EOS_REGISTRY.get('PALEOS:iron')
        assert entry is not None
        assert entry.get('format') == 'paleos_unified'
        assert 'eos_file' in entry

    def test_get_returns_default_for_unknown_key(self):
        """``get`` returns the supplied default for an unregistered key."""
        sentinel = object()
        assert EOS_REGISTRY.get('NotARealEOS:nope', sentinel) is sentinel
        # Default-default is None
        assert EOS_REGISTRY.get('AnotherFake:eos') is None

    def test_contains_false_for_unknown_key(self):
        """``__contains__`` returns False for unregistered keys."""
        assert 'DefinitelyNotAnEOS:fake' not in EOS_REGISTRY

    def test_keys_values_items_consistent(self):
        """``keys``, ``values``, ``items`` must agree pairwise."""
        keys = list(EOS_REGISTRY.keys())
        values = list(EOS_REGISTRY.values())
        items = list(EOS_REGISTRY.items())
        assert len(keys) == len(values) == len(items)
        # Items pair (key, value) must reconstruct keys + values.
        item_keys = [k for k, _ in items]
        item_values = [v for _, v in items]
        assert item_keys == keys
        assert item_values == values

    def test_repr_contains_a_known_key(self):
        """``__repr__`` should render the underlying dict — at minimum
        a known registry key appears in its string form."""
        text = repr(EOS_REGISTRY)
        assert 'PALEOS:iron' in text


class TestLegacyMaterialDictAccess:
    """Module-level ``__getattr__`` returns the five legacy material
    dictionaries on demand and raises ``AttributeError`` for anything
    else."""

    @pytest.mark.parametrize(
        'name',
        [
            'material_properties_iron_silicate_planets',
            'material_properties_iron_Tdep_silicate_planets',
            'material_properties_water_planets',
            'material_properties_iron_RTPress100TPa_silicate_planets',
            'material_properties_iron_PALEOS_silicate_planets',
        ],
    )
    def test_known_legacy_name_returns_dict(self, name):
        """Each known legacy alias must return a dict with at least a
        'core' or 'mantle' sub-key (i.e. a real material spec)."""
        mat = getattr(eos_properties, name)
        assert isinstance(mat, dict)
        # Every legacy spec contains 'core' (Seager iron).
        assert 'core' in mat
        # Iron core spec must point at a Seager2007 iron file.
        assert 'eos_file' in mat['core']
        assert 'eos_seager07_iron.txt' in mat['core']['eos_file']

    def test_unknown_attribute_raises(self):
        """Any other attribute name must raise ``AttributeError`` so
        normal Python attribute lookup semantics are preserved."""
        with pytest.raises(AttributeError, match='no attribute'):
            _ = eos_properties.this_attribute_does_not_exist  # noqa: F841

    def test_unknown_attribute_does_not_match_legacy_pattern(self):
        """Names that look like legacy aliases but aren't in the set
        must still raise. Defensive: the set check is exact, not a
        prefix match."""
        with pytest.raises(AttributeError):
            _ = eos_properties.material_properties_unobtainium  # noqa: F841


class TestRegistryIsBuiltOnce:
    """The lazy build hook must be idempotent: subsequent accesses
    return the same dict instance, never rebuild."""

    def test_repeated_access_returns_identical_entries(self):
        """Two reads of the same key must return the same dict object
        (identity, not just equality)."""
        first = EOS_REGISTRY['Seager2007:iron']
        second = EOS_REGISTRY['Seager2007:iron']
        assert first is second
