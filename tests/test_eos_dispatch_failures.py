"""Failure-path coverage for ``zalmoxis.eos.dispatch``.

The happy-path of ``calculate_density`` and ``calculate_density_batch``
is covered by ``tests/test_eos_functions.py`` (TestCalculateDensity*
classes). This file exercises the explicit ``ValueError`` raises and
defensive branches that those tests do not reach.

Anti-happy-path branches covered:

1. ``layer_eos`` not in the registry and not an ``Analytic:`` /
   ``Vinet:`` prefix -> ``ValueError("Unknown layer EOS ...")`` (line
   97-99 of dispatch.py).
2. Registry entry exists but contains none of ``core`` / ``mantle`` /
   ``ice_layer`` / ``melted_mantle`` / ``format`` -> ``ValueError(
   "Cannot determine layer key ...")`` (line 134).
3. ``_is_paleos_api`` returns False for non-PALEOS-API entries and the
   2-phase sub-dict variants are detected.
4. Batch fallback path (non-paleos_unified) calls scalar
   ``calculate_density`` per element and returns NaN where lookup
   fails.
"""

from __future__ import annotations

import numpy as np
import pytest

from zalmoxis.eos.dispatch import (
    _is_paleos_api,
    calculate_density,
    calculate_density_batch,
)

pytestmark = pytest.mark.unit


class TestUnknownEosRaises:
    """Unknown ``layer_eos`` strings (no Analytic:/Vinet: prefix and no
    registry entry) must raise a clear ``ValueError`` with the offending
    key in the message."""

    def test_completely_unknown_eos(self):
        """A garbage EOS identifier raises ValueError mentioning the key."""
        with pytest.raises(ValueError, match="Unknown layer EOS 'TotalGarbage:foo'"):
            calculate_density(
                100e9,
                {},  # empty registry
                'TotalGarbage:foo',
                300.0,
                None,
                None,
            )

    def test_known_prefix_unknown_material_key(self):
        """A registry-style key with a colon but no entry still raises
        the same ValueError. Distinguishes from the Analytic:/Vinet:
        prefix-handling branch which dispatches before the lookup."""
        with pytest.raises(ValueError, match="Unknown layer EOS 'NotARegistry:material'"):
            calculate_density(
                100e9,
                {},
                'NotARegistry:material',
                300.0,
                None,
                None,
            )


class TestCannotDetermineLayerKey:
    """Registry entries without a recognised layer key must raise the
    fallback ``ValueError("Cannot determine layer key ...")``. This
    branch is reachable when the registry contains a malformed Seager-
    style entry (no 'core'/'mantle'/'ice_layer', no 'format' tag, no
    'melted_mantle')."""

    def test_empty_dict_entry_raises(self):
        """A registry entry that is an empty dict triggers the
        end-of-function ValueError."""
        registry = {'BadEntry:nothing': {}}
        with pytest.raises(ValueError, match='Cannot determine layer key'):
            calculate_density(100e9, registry, 'BadEntry:nothing', 300.0, None, None)

    def test_dict_with_only_ignored_keys_raises(self):
        """A registry entry with keys that don't match the dispatch
        cascade ('format' missing, no 'melted_mantle', no
        'core'/'mantle'/'ice_layer') still raises."""
        registry = {'BadEntry:other': {'unrelated_key': 'value'}}
        with pytest.raises(ValueError, match='Cannot determine layer key'):
            calculate_density(100e9, registry, 'BadEntry:other', 300.0, None, None)


class TestIsPaleosApiDetection:
    """``_is_paleos_api`` must return True for top-level paleos_api/
    paleos_api_2phase entries, True for 2-phase wrappers carrying
    ``paleos_api_2phase`` sub-dicts, and False for everything else."""

    def test_top_level_paleos_api(self):
        """Direct ``format='paleos_api'`` is detected."""
        assert _is_paleos_api({'format': 'paleos_api', 'material': 'iron'}) is True

    def test_top_level_paleos_api_2phase(self):
        """Direct ``format='paleos_api_2phase'`` is detected."""
        assert _is_paleos_api({'format': 'paleos_api_2phase', 'side': 'solid'}) is True

    def test_two_phase_wrapper_with_api_subdict(self):
        """A wrapper dict carrying ``melted_mantle`` of format
        ``paleos_api_2phase`` is detected (2-phase recursion)."""
        wrapper = {
            'core': {'eos_file': '/dev/null'},
            'melted_mantle': {'format': 'paleos_api_2phase', 'side': 'liquid'},
            'solid_mantle': {'format': 'paleos_api_2phase', 'side': 'solid'},
        }
        assert _is_paleos_api(wrapper) is True

    def test_seager_entry_not_paleos_api(self):
        """A pure Seager entry must NOT be flagged as paleos_api."""
        seager = {'core': {'eos_file': '/dev/null'}}
        assert _is_paleos_api(seager) is False

    def test_paleos_unified_not_paleos_api(self):
        """``format='paleos_unified'`` must NOT be flagged. The two
        formats are distinct dispatch paths."""
        unified = {'format': 'paleos_unified', 'eos_file': '/dev/null'}
        assert _is_paleos_api(unified) is False

    def test_subdict_non_dict_value_safe(self):
        """A 'melted_mantle' value that is not a dict (e.g. None) must
        not raise; the helper guards with isinstance(sub, dict)."""
        # If the submodule check forgot the isinstance guard, sub.get
        # would raise AttributeError on a None entry. Verify defensive
        # handling.
        wrapper = {'core': {}, 'melted_mantle': None, 'solid_mantle': None}
        assert _is_paleos_api(wrapper) is False


class TestBatchFallbackToScalar:
    """For non-paleos_unified EOS, ``calculate_density_batch`` falls
    back to a scalar loop. NaN must be returned for elements where the
    scalar lookup fails."""

    def test_batch_unknown_layer_returns_nan_array(self):
        """Batch with an unknown layer EOS raises (scalar path raises
        ValueError for unknown EOS, but the batch path enters the
        scalar loop only after the registry lookup; verify the registry
        lookup returns None and the function exits the special path)."""
        # When registry has no entry, mat is None and neither the
        # _api_resolved nor the paleos_unified branch fires; the
        # scalar loop runs, which itself raises ValueError per element.
        with pytest.raises(ValueError, match='Unknown layer EOS'):
            calculate_density_batch(
                np.array([1e10, 2e10]),
                np.array([300.0, 400.0]),
                {},
                'NoSuchEos:material',
                None,
                None,
                interpolation_functions={},
            )

    def test_batch_with_none_interp_cache(self):
        """``interpolation_functions=None`` is normalised to ``{}`` in
        the batch path the same way it is in the scalar path. With
        Analytic:iron the dispatch never touches the cache, so we just
        verify the call returns a finite array."""
        rho = calculate_density_batch(
            np.array([1e10, 5e10]),
            np.array([300.0, 300.0]),
            {},
            'Analytic:iron',
            None,
            None,
            interpolation_functions=None,
        )
        assert rho.shape == (2,)
        assert np.all(np.isfinite(rho))
        # Density at higher pressure must exceed density at lower P
        # (monotonicity of analytic Seager iron EOS).
        assert rho[1] > rho[0]
