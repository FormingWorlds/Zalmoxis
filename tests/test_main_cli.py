"""Coverage test for the ``python -m zalmoxis`` CLI entry point.

The body of ``zalmoxis/__main__.py`` only executes under ``__name__
== '__main__'``, so the standard import path leaves it uncovered.
``runpy.run_module`` re-imports the module with the run-name set to
``__main__``, exercising the lines that wire up logging and call
``load_zalmoxis_config()`` / ``post_processing()``.

Anti-happy-path: rather than running the actual solver (which would be
several seconds and pull in EOS data), both downstream calls are
mocked. Tests verify the CLI plumbing only:

* ``get_zalmoxis_root()`` is consulted for the log path
* logging.basicConfig is configured with the expected file path
* ``load_zalmoxis_config`` is invoked with no arguments
* ``post_processing`` is invoked with the loaded config
"""

from __future__ import annotations

import logging
import runpy
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def reset_logging():
    """Snapshot and restore the root logger's handlers + level so the
    runpy invocation doesn't leak handler state into the rest of the
    test session."""
    logger = logging.getLogger()
    saved_handlers = list(logger.handlers)
    saved_level = logger.level
    yield
    for h in list(logger.handlers):
        logger.removeHandler(h)
    for h in saved_handlers:
        logger.addHandler(h)
    logger.setLevel(saved_level)


class TestZalmoxisCli:
    """Exercise ``python -m zalmoxis`` plumbing with the heavy
    submodules mocked out."""

    def test_main_invokes_load_then_post_process(self, tmp_path, reset_logging):
        """The CLI must call ``load_zalmoxis_config`` once and feed its
        return value into ``post_processing``."""
        sentinel_config = {'sentinel': 'cli-test-config'}

        # Fake root so basicConfig writes to a tmp file rather than the
        # repo's output/ directory.
        with patch('zalmoxis.get_zalmoxis_root', return_value=str(tmp_path)):
            with patch(
                'zalmoxis.config.load_zalmoxis_config', return_value=sentinel_config
            ) as mock_load:
                with patch('zalmoxis.output.post_processing') as mock_post:
                    # Ensure output/ exists for basicConfig's filename
                    (tmp_path / 'output').mkdir(exist_ok=True)
                    runpy.run_module('zalmoxis', run_name='__main__')

                    mock_load.assert_called_once_with()
                    mock_post.assert_called_once_with(sentinel_config)

    def test_main_configures_log_file_under_zalmoxis_root(self, tmp_path, reset_logging):
        """The log file path must be ``{ZALMOXIS_ROOT}/output/zalmoxis.log``,
        constructed via ``os.path.join`` with the resolved root.

        ``logging.basicConfig`` is a no-op once the root logger already
        has handlers (which it does in a long pytest session), so the
        cleanest assertion is that the CLI passes the expected
        ``filename`` keyword to ``basicConfig`` rather than checking the
        on-disk file."""
        import os

        with patch('zalmoxis.get_zalmoxis_root', return_value=str(tmp_path)):
            with patch('zalmoxis.config.load_zalmoxis_config', return_value={}):
                with patch('zalmoxis.output.post_processing'):
                    with patch('logging.basicConfig') as mock_basicconfig:
                        runpy.run_module('zalmoxis', run_name='__main__')

        mock_basicconfig.assert_called_once()
        kwargs = mock_basicconfig.call_args.kwargs
        expected_filename = os.path.join(str(tmp_path), 'output', 'zalmoxis.log')
        assert kwargs.get('filename') == expected_filename
        assert kwargs.get('level') == logging.INFO
        assert kwargs.get('filemode') == 'w'

    def test_main_does_not_swallow_load_config_errors(self, tmp_path, reset_logging):
        """If ``load_zalmoxis_config`` raises, the exception must
        propagate (no silent catch). Important for CI where a malformed
        config should fail loudly."""
        with patch('zalmoxis.get_zalmoxis_root', return_value=str(tmp_path)):
            with patch(
                'zalmoxis.config.load_zalmoxis_config',
                side_effect=RuntimeError('synthetic config failure'),
            ):
                with patch('zalmoxis.output.post_processing') as mock_post:
                    (tmp_path / 'output').mkdir(exist_ok=True)
                    with pytest.raises(RuntimeError, match='synthetic config failure'):
                        runpy.run_module('zalmoxis', run_name='__main__')
                    # post_processing must not have been invoked when
                    # config loading failed.
                    mock_post.assert_not_called()
