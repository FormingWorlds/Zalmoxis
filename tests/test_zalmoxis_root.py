"""Coverage tests for ``zalmoxis.get_zalmoxis_root`` and the
``ImportError`` fallback in ``zalmoxis/__init__.py``.

The lazy-resolution function has three branches:
1. Env var ``ZALMOXIS_ROOT`` is set and points at a real directory.
2. Env var unset; auto-detected via ``__file__.parent.parent.parent`` containing
   ``pyproject.toml`` (the editable / source-tree case).
3. Env var unset; ``pyproject.toml`` not found; fallback to two levels up
   (the installed-into-site-packages case where ``src/`` is the repo root).
4. After all that, if the resolved path does not exist, ``RuntimeError``.

Anti-happy-path: the failure path (4) is exercised explicitly. The
``ImportError`` fallback in the module header is exercised by reimport
under a monkeypatched module table.
"""

from __future__ import annotations

import importlib
import sys

import pytest

import zalmoxis

pytestmark = pytest.mark.unit


@pytest.fixture
def reset_root_cache(monkeypatch):
    """Reset the lazy ``_zalmoxis_root`` cache between tests so each test
    sees a clean resolution path."""
    monkeypatch.setattr(zalmoxis, '_zalmoxis_root', None)
    yield


class TestGetZalmoxisRoot:
    """Cover the three resolution branches plus the failure raise."""

    def test_env_var_set_returns_value(self, tmp_path, monkeypatch, reset_root_cache):
        """If ZALMOXIS_ROOT points at an existing directory, return it
        verbatim and cache it."""
        monkeypatch.setenv('ZALMOXIS_ROOT', str(tmp_path))
        first = zalmoxis.get_zalmoxis_root()
        assert first == str(tmp_path)
        # Cache hit on second call: should return the cached value
        # without re-reading the env var.
        monkeypatch.delenv('ZALMOXIS_ROOT', raising=False)
        second = zalmoxis.get_zalmoxis_root()
        assert second == first

    def test_unset_env_auto_detects_repo_root_when_pyproject_present(
        self, monkeypatch, reset_root_cache
    ):
        """With env unset and pyproject.toml at __file__.parent.parent.parent,
        the function returns that directory."""
        monkeypatch.delenv('ZALMOXIS_ROOT', raising=False)
        root = zalmoxis.get_zalmoxis_root()
        # The actual install in this repo has pyproject.toml three levels
        # above __init__.py, so we should land on the repo root.
        import pathlib

        expected = pathlib.Path(zalmoxis.__file__).parent.parent.parent.resolve()
        assert root == str(expected)

    def test_unset_env_fallback_when_no_pyproject(
        self, tmp_path, monkeypatch, reset_root_cache
    ):
        """When auto-detect's first candidate has no pyproject.toml, the
        fallback two-levels-up branch executes."""
        # Plant a fake __file__ deep in tmp_path so the .parent.parent.parent
        # candidate has no pyproject.toml. The fallback then collapses to
        # tmp_path/fake_src which we make sure exists.
        fake_src = tmp_path / 'fake_src'
        fake_pkg = fake_src / 'pkg'
        fake_pkg.mkdir(parents=True)
        fake_init = fake_pkg / '__init__.py'
        fake_init.touch()

        monkeypatch.delenv('ZALMOXIS_ROOT', raising=False)
        # Force the resolution to use our fake __file__
        monkeypatch.setattr(zalmoxis, '__file__', str(fake_init))
        root = zalmoxis.get_zalmoxis_root()
        # Fallback returns __file__.parent.parent which is fake_src
        assert root == str(fake_src.resolve())

    def test_raises_when_root_does_not_exist(self, monkeypatch, reset_root_cache):
        """If env var points at a non-existent directory and auto-detect
        also fails, RuntimeError must be raised."""
        # Clear env first
        monkeypatch.delenv('ZALMOXIS_ROOT', raising=False)
        # Plant a fake __file__ such that BOTH candidates (parent**3 and
        # parent**2) resolve to non-existent paths.
        monkeypatch.setattr(
            zalmoxis,
            '__file__',
            '/nonexistent/path/that/cannot/exist/pkg/__init__.py',
        )
        with pytest.raises(RuntimeError, match='ZALMOXIS_ROOT'):
            zalmoxis.get_zalmoxis_root()


class TestVersionImportFallback:
    """Cover the ``ImportError`` fallback when ``_version.py`` is missing.

    The branch executes during the ``zalmoxis`` import; we simulate this
    by injecting a fake import failure through ``sys.modules`` and
    re-importing.
    """

    def test_fallback_version_when_version_module_missing(self, monkeypatch):
        """Reimport ``zalmoxis`` with ``_version`` made unimportable and
        verify the fallback constants are used."""
        # Make `from ._version import ...` fail by replacing the cached
        # _version submodule entry with one that raises on attribute
        # access AND removing zalmoxis from sys.modules so reimport runs.
        monkeypatch.setitem(sys.modules, 'zalmoxis._version', _VersionRaiser())
        monkeypatch.delitem(sys.modules, 'zalmoxis', raising=False)

        reloaded = importlib.import_module('zalmoxis')
        try:
            assert reloaded.__version__ == '0.0.0.dev0'
            assert reloaded.__version_tuple__ == (0, 0, 0, 'dev0')
        finally:
            # Force a clean reimport of the real zalmoxis after this test
            # so other tests see normal state.
            sys.modules.pop('zalmoxis', None)
            sys.modules.pop('zalmoxis._version', None)
            importlib.import_module('zalmoxis')


class _VersionRaiser:
    """Module stand-in that raises ``ImportError`` on any attribute lookup,
    forcing the ``except ImportError`` branch in ``zalmoxis/__init__.py``."""

    def __getattr__(self, name):
        raise ImportError(f'simulated missing _version.{name}')
