"""Shared test fixtures for the Zalmoxis test suite.

Provides session-scoped fixtures for common resources:
- ZALMOXIS_ROOT validation and access (lazy, skips for pure-math unit tests)
- Cached planetary structure solver to avoid redundant runs

References:
    - docs/test_infrastructure.md
    - docs/test_categorization.md
"""

from __future__ import annotations

import pytest

from tests import _paleos_mock

# Install the in-process ``paleos`` mock at conftest collection time, before
# ``tests/test_paleos_api.py`` (or anything else) imports
# ``zalmoxis.eos.paleos_api``. The mock is a no-op when the real PALEOS
# package is importable; on CI runners where it is not, the mock injects
# the minimum surface the producer exercises so the 60+ tests in
# ``test_paleos_api.py`` run on every push instead of being skipped.
_paleos_mock.install_if_missing()


@pytest.fixture(scope='session')
def zalmoxis_root():
    """Root directory of the Zalmoxis repository.

    Resolved lazily via get_zalmoxis_root(). Skips the test session
    if the root cannot be determined (ZALMOXIS_ROOT not set and
    auto-detection fails).
    """
    from zalmoxis import get_zalmoxis_root

    try:
        return get_zalmoxis_root()
    except RuntimeError:
        pytest.skip('ZALMOXIS_ROOT not set and auto-detection failed')


@pytest.fixture(scope='session')
def cached_solver():
    """Session-scoped cached solver runner.

    Wraps run_zalmoxis_rocky_water so that identical parameter combinations
    reuse the same output files instead of re-running the full solver.

    Returns
    -------
    callable
        Same signature as run_zalmoxis_rocky_water, with transparent caching.
    """
    from tools.setup.setup_tests import run_zalmoxis_rocky_water

    _cache = {}

    def _run(mass, config_type, cmf, immf, layer_eos_override=None):
        eos_key = tuple(sorted(layer_eos_override.items())) if layer_eos_override else None
        key = (mass, config_type, cmf, immf, eos_key)
        if key not in _cache:
            _cache[key] = run_zalmoxis_rocky_water(
                mass, config_type, cmf, immf, layer_eos_override
            )
        return _cache[key]

    return _run
