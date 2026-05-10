"""Tests for the lazy PALEOS-API resolution path in ``eos.dispatch``.

Covers:
- ``_is_paleos_api`` returning True on the three trigger formats:
  ``paleos_api``, ``paleos_api_2phase``, and a 2-phase parent whose
  ``solid_mantle`` or ``melted_mantle`` sub-dict is ``paleos_api_2phase``.
- ``calculate_density`` calling ``resolve_registry_entry`` once and then
  setting ``_api_resolved`` so subsequent calls short-circuit.
- ``calculate_density_batch`` taking the same lazy-resolve branch.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from zalmoxis.eos.dispatch import (
    _is_paleos_api,
    calculate_density,
    calculate_density_batch,
)

pytestmark = pytest.mark.unit


def test_is_paleos_api_flat_format_returns_true():
    assert _is_paleos_api({'format': 'paleos_api'}) is True


def test_is_paleos_api_2phase_format_returns_true():
    assert _is_paleos_api({'format': 'paleos_api_2phase'}) is True


def test_is_paleos_api_nested_2phase_subdict_returns_true():
    """A normal 2-phase parent whose sub-table is paleos_api_2phase fires too."""
    mat = {
        'format': 'paleos_2phase',
        'solid_mantle': {'format': 'paleos_api_2phase'},
        'melted_mantle': {'format': 'paleos'},
    }
    assert _is_paleos_api(mat) is True


def test_is_paleos_api_unrelated_dict_returns_false():
    assert _is_paleos_api({'format': 'paleos_unified'}) is False
    assert _is_paleos_api({'format': 'seager_table'}) is False


def test_calculate_density_resolves_paleos_api_once_and_caches(monkeypatch):
    """First call invokes resolve_registry_entry; second call skips it."""
    fake_unified = {
        'format': 'paleos_unified',
        'eos_file': '/tmp/fake_resolved.dat',
    }

    call_count = {'n': 0}

    def fake_resolver(mat):
        call_count['n'] += 1
        # Mutate in place to mimic the real resolver
        mat.clear()
        mat.update(fake_unified)
        return mat

    fake_density = 5500.0

    def fake_unified_density(pressure, temperature, mat, mushy_zone_factor, cache):
        return fake_density

    mat = {'format': 'paleos_api', 'eos_file': 'whatever.txt'}
    material_dicts = {'PALEOS-API:iron': mat}

    with (
        patch(
            'zalmoxis.eos.paleos_api_cache.resolve_registry_entry',
            side_effect=fake_resolver,
        ),
        patch(
            'zalmoxis.eos.dispatch.get_paleos_unified_density',
            side_effect=fake_unified_density,
        ),
    ):
        rho1 = calculate_density(
            pressure=1.0e10,
            material_dictionaries=material_dicts,
            layer_eos='PALEOS-API:iron',
            temperature=3000.0,
            solidus_func=None,
            liquidus_func=None,
            interpolation_functions={},
        )
        rho2 = calculate_density(
            pressure=1.0e10,
            material_dictionaries=material_dicts,
            layer_eos='PALEOS-API:iron',
            temperature=3000.0,
            solidus_func=None,
            liquidus_func=None,
            interpolation_functions={},
        )

    assert rho1 == pytest.approx(fake_density)
    assert rho2 == pytest.approx(fake_density)
    # Resolver should run exactly once (first call sets _api_resolved=True)
    assert call_count['n'] == 1
    assert mat['_api_resolved'] is True


def test_calculate_density_batch_lazy_resolves_paleos_api():
    """The batch path takes the same lazy-resolve branch on first call."""
    call_count = {'n': 0}

    def fake_resolver(mat):
        call_count['n'] += 1
        mat.clear()
        mat.update({'format': 'paleos_unified', 'eos_file': '/tmp/fake.dat'})
        return mat

    n = 5
    pressures = np.full(n, 1.0e10)
    temperatures = np.full(n, 3000.0)
    fake_densities = np.full(n, 5500.0)

    def fake_batch_density(pressures, temperatures, mat, mushy_zone_factor, cache):
        return fake_densities

    mat = {'format': 'paleos_api'}
    material_dicts = {'PALEOS-API:iron': mat}

    with (
        patch(
            'zalmoxis.eos.paleos_api_cache.resolve_registry_entry',
            side_effect=fake_resolver,
        ),
        patch(
            'zalmoxis.eos.dispatch.get_paleos_unified_density_batch',
            side_effect=fake_batch_density,
        ),
    ):
        out = calculate_density_batch(
            pressures=pressures,
            temperatures=temperatures,
            material_dictionaries=material_dicts,
            layer_eos='PALEOS-API:iron',
            solidus_func=None,
            liquidus_func=None,
            interpolation_functions={},
        )

    assert call_count['n'] == 1
    assert mat['_api_resolved'] is True
    assert np.allclose(out, fake_densities)
