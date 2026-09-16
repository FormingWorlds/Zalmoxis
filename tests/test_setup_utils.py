"""Provenance tests for the data-folder downloader in ``tools.setup.setup_utils``.

``download`` reuses a data folder that is already on disk. These tests pin the
rule that decides when that reuse is safe: a folder may be kept only when the
marker written beside it names the Zenodo record currently pinned, so that
moving a pin to a new release replaces the folder instead of being ignored.
The network is never touched; ``download_zenodo_folder`` and the OSF fallback
are replaced by fakes that record whether they were called.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


def _marker(folder_dir: Path) -> Path:
    from tools.setup.setup_utils import _SOURCE_MARKER

    return folder_dir / _SOURCE_MARKER


def _fake_fetch(folder_dir: Path, payload: str):
    """Build a stand-in for download_zenodo_folder that rewrites the folder."""
    calls = []

    def _fetch(zenodo_id, folder_dir=folder_dir, keep_files=None):
        calls.append(zenodo_id)
        folder_dir.mkdir(parents=True, exist_ok=True)
        (folder_dir / 'table.dat').write_text(payload)

    return _fetch, calls


class TestDownloadProvenance:
    """Reuse-or-refresh decisions made by ``download`` on an existing folder."""

    def test_marker_records_the_record_that_supplied_the_folder(self, tmp_path):
        """A fresh download leaves the pinned record id beside the data."""
        from tools.setup.setup_utils import download

        folder_dir = tmp_path / 'EOS_PALEOS_iron'
        fetch, calls = _fake_fetch(folder_dir, 'v1.3.0')
        with patch('tools.setup.setup_utils.download_zenodo_folder', side_effect=fetch):
            download(folder='EOS_PALEOS_iron', data_dir=tmp_path, zenodo_id=22776069)

        assert calls == [22776069]
        # The id is stored as text so a later run can compare it to its own pin.
        assert _marker(folder_dir).read_text().strip() == '22776069'

    def test_folder_from_the_pinned_record_is_reused(self, tmp_path):
        """A folder whose marker matches the pin costs no download."""
        from tools.setup.setup_utils import download

        folder_dir = tmp_path / 'EOS_PALEOS_iron'
        folder_dir.mkdir(parents=True)
        (folder_dir / 'table.dat').write_text('v1.3.0')
        _marker(folder_dir).write_text('22776069\n')

        fetch, calls = _fake_fetch(folder_dir, 'refetched')
        with patch('tools.setup.setup_utils.download_zenodo_folder', side_effect=fetch):
            download(folder='EOS_PALEOS_iron', data_dir=tmp_path, zenodo_id=22776069)

        assert calls == []
        assert (folder_dir / 'table.dat').read_text() == 'v1.3.0'

    def test_folder_from_a_superseded_record_is_refreshed(self, tmp_path):
        """Moving the pin to a new release replaces an older folder.

        This is the case the marker exists for: before it, an installation
        holding an earlier release kept that release forever, because the
        folder merely existing was enough to skip the download.
        """
        from tools.setup.setup_utils import download

        folder_dir = tmp_path / 'EOS_PALEOS_iron'
        folder_dir.mkdir(parents=True)
        (folder_dir / 'table.dat').write_text('v1.0.0')
        _marker(folder_dir).write_text('19000316\n')

        fetch, calls = _fake_fetch(folder_dir, 'v1.3.0')
        with patch('tools.setup.setup_utils.download_zenodo_folder', side_effect=fetch):
            download(folder='EOS_PALEOS_iron', data_dir=tmp_path, zenodo_id=22776069)

        assert calls == [22776069]
        assert (folder_dir / 'table.dat').read_text() == 'v1.3.0'
        assert _marker(folder_dir).read_text().strip() == '22776069'

    def test_folder_predating_the_marker_is_refreshed_once(self, tmp_path):
        """A folder with no marker has unknown provenance and is refreshed.

        Every folder downloaded before this bookkeeping existed lands here, so
        the first run after the change migrates it and the marker it leaves
        behind keeps later runs from downloading again.
        """
        from tools.setup.setup_utils import download

        folder_dir = tmp_path / 'EOS_PALEOS_H2O'
        folder_dir.mkdir(parents=True)
        (folder_dir / 'table.dat').write_text('unknown vintage')
        assert not _marker(folder_dir).exists()

        fetch, calls = _fake_fetch(folder_dir, 'v1.3.0')
        with patch('tools.setup.setup_utils.download_zenodo_folder', side_effect=fetch):
            download(folder='EOS_PALEOS_H2O', data_dir=tmp_path, zenodo_id=22776069)
            assert calls == [22776069]
            # The second run sees the marker this one wrote and stands down.
            download(folder='EOS_PALEOS_H2O', data_dir=tmp_path, zenodo_id=22776069)
        assert calls == [22776069]

    def test_osf_only_folder_is_left_alone(self, tmp_path):
        """With no Zenodo pin there is nothing to compare, so the folder stands."""
        from tools.setup.setup_utils import download

        folder_dir = tmp_path / 'Population'
        folder_dir.mkdir(parents=True)
        (folder_dir / 'table.dat').write_text('from OSF')

        fetch, calls = _fake_fetch(folder_dir, 'refetched')
        with patch('tools.setup.setup_utils.download_zenodo_folder', side_effect=fetch):
            download(folder='Population', data_dir=tmp_path, zenodo_id=None, osf_id='dpkjb')

        assert calls == []
        assert (folder_dir / 'table.dat').read_text() == 'from OSF'

    def test_marker_written_after_the_osf_fallback(self, tmp_path):
        """A folder delivered by OSF still records the record it stands in for."""
        from tools.setup.setup_utils import download

        folder_dir = tmp_path / 'EOS_Seager2007'

        def _osf(storage, folders, data_dir):
            folder_dir.mkdir(parents=True, exist_ok=True)
            (folder_dir / 'table.dat').write_text('from OSF')

        with patch(
            'tools.setup.setup_utils.download_zenodo_folder',
            side_effect=RuntimeError('zenodo_get failed with exit code 1'),
        ):
            with patch('tools.setup.setup_utils.get_osf', return_value=object()):
                with patch('tools.setup.setup_utils.download_OSF_folder', side_effect=_osf):
                    download(
                        folder='EOS_Seager2007',
                        data_dir=tmp_path,
                        zenodo_id=15727998,
                        osf_id='dpkjb',
                    )

        assert _marker(folder_dir).read_text().strip() == '15727998'

    def test_no_marker_written_when_both_sources_fail(self, tmp_path):
        """A failed download must not leave provenance implying success."""
        from tools.setup.setup_utils import download

        folder_dir = tmp_path / 'EOS_PALEOS_iron'
        with patch(
            'tools.setup.setup_utils.download_zenodo_folder',
            side_effect=RuntimeError('zenodo_get failed with exit code 1'),
        ):
            with patch('tools.setup.setup_utils.get_osf', return_value=object()):
                with patch(
                    'tools.setup.setup_utils.download_OSF_folder',
                    side_effect=RuntimeError('osf unreachable'),
                ):
                    with pytest.raises(RuntimeError, match='from both Zenodo and OSF'):
                        download(
                            folder='EOS_PALEOS_iron',
                            data_dir=tmp_path,
                            zenodo_id=22776069,
                            osf_id='dpkjb',
                        )

        assert not _marker(folder_dir).exists()
