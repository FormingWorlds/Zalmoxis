"""Shared test fixtures for the Zalmoxis test suite.

Provides session-scoped fixtures for common resources:
- ZALMOXIS_ROOT validation and access
- Cached planetary structure solver to avoid redundant runs

References:
    - docs/test_infrastructure.md
    - docs/test_categorization.md
"""

from __future__ import annotations

import os

import pytest

from tools.setup_tests import run_zalmoxis_rocky_water


@pytest.fixture(scope='session', autouse=True)
def _validate_environment():
    """Validate required environment variables at session start."""
    if not os.getenv('ZALMOXIS_ROOT'):
        pytest.exit('ZALMOXIS_ROOT environment variable not set', returncode=1)


@pytest.fixture(scope='session')
def zalmoxis_root():
    """Root directory of the Zalmoxis repository."""
    return os.environ['ZALMOXIS_ROOT']


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
