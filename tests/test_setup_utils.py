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

    def test_zenodo_incomplete_delivery_falls_through_to_osf(self, tmp_path):
        """Zenodo reporting success while short a kept file must still try OSF.

        This is the case that reached production: ``zenodo_get`` exits 0 but
        the record does not carry every file ``keep_files`` names. The OSF
        fallback is configured for exactly this folder, so it must run
        instead of the download being accepted or failing outright.
        """
        from tools.setup.setup_utils import download

        folder_dir = tmp_path / 'mass_radius_curves'
        zenodo_calls = []
        osf_calls = []

        def _zenodo(zenodo_id, folder_dir=folder_dir, keep_files=None):
            zenodo_calls.append(zenodo_id)
            folder_dir.mkdir(parents=True, exist_ok=True)
            (folder_dir / 'a.txt').write_text('present')
            # 'b.txt' never arrives: Zenodo under-delivers without raising.

        def _osf(storage, folders, data_dir):
            osf_calls.append(folders)
            folder_dir.mkdir(parents=True, exist_ok=True)
            (folder_dir / 'a.txt').write_text('from OSF')
            (folder_dir / 'b.txt').write_text('from OSF')

        with patch('tools.setup.setup_utils.download_zenodo_folder', side_effect=_zenodo):
            with patch('tools.setup.setup_utils.get_osf', return_value=object()):
                with patch('tools.setup.setup_utils.download_OSF_folder', side_effect=_osf):
                    download(
                        folder='mass_radius_curves',
                        data_dir=tmp_path,
                        zenodo_id=15727899,
                        osf_id='dpkjb',
                        keep_files=['a.txt', 'b.txt'],
                    )

        assert zenodo_calls == [15727899]
        assert osf_calls == [['mass_radius_curves']]
        assert (folder_dir / 'a.txt').read_text() == 'from OSF'
        assert (folder_dir / 'b.txt').read_text() == 'from OSF'
        assert _marker(folder_dir).read_text().strip() == '15727899'

    def test_osf_fallback_does_not_keep_leftover_zenodo_files(self, tmp_path):
        """A Zenodo partial delivery must not survive into the OSF result.

        Zenodo writes 'a.txt' then under-delivers; the OSF fallback only ever
        writes 'b.txt'. If the leftover 'a.txt' were kept, the folder would be
        a hybrid of two sources recorded under a single marker.
        """
        from tools.setup.setup_utils import download

        folder_dir = tmp_path / 'mass_radius_curves'

        def _zenodo(zenodo_id, folder_dir=folder_dir, keep_files=None):
            folder_dir.mkdir(parents=True, exist_ok=True)
            (folder_dir / 'a.txt').write_text('zenodo leftover')
            # 'b.txt' never arrives: Zenodo under-delivers without raising.

        def _osf(storage, folders, data_dir):
            folder_dir.mkdir(parents=True, exist_ok=True)
            (folder_dir / 'b.txt').write_text('from OSF')

        with patch('tools.setup.setup_utils.download_zenodo_folder', side_effect=_zenodo):
            with patch('tools.setup.setup_utils.get_osf', return_value=object()):
                with patch('tools.setup.setup_utils.download_OSF_folder', side_effect=_osf):
                    download(
                        folder='mass_radius_curves',
                        data_dir=tmp_path,
                        zenodo_id=15727899,
                        osf_id='dpkjb',
                        keep_files=['b.txt'],
                    )

        assert not (folder_dir / 'a.txt').exists()
        assert (folder_dir / 'b.txt').read_text() == 'from OSF'

    def test_osf_fallback_incomplete_raises(self, tmp_path):
        """An OSF fallback that also under-delivers a kept file is not accepted."""
        from tools.setup.setup_utils import download

        folder_dir = tmp_path / 'mass_radius_curves'

        def _osf(storage, folders, data_dir):
            folder_dir.mkdir(parents=True, exist_ok=True)
            (folder_dir / 'a.txt').write_text('from OSF')
            # 'b.txt' never arrives.

        with patch(
            'tools.setup.setup_utils.download_zenodo_folder',
            side_effect=RuntimeError('zenodo_get failed with exit code 1'),
        ):
            with patch('tools.setup.setup_utils.get_osf', return_value=object()):
                with patch('tools.setup.setup_utils.download_OSF_folder', side_effect=_osf):
                    with pytest.raises(RuntimeError, match='from both Zenodo and OSF'):
                        download(
                            folder='mass_radius_curves',
                            data_dir=tmp_path,
                            zenodo_id=15727899,
                            osf_id='dpkjb',
                            keep_files=['a.txt', 'b.txt'],
                        )

        assert not _marker(folder_dir).exists()

    def test_folder_with_every_kept_file_and_matching_marker_is_reused(self, tmp_path):
        """A folder that matches the pin and holds every kept file costs no download.

        Pins the counterpart of the missing-file case: a future edit that
        inverts the ``if not missing`` check would pass every other test here
        while forcing a full re-download of a complete folder on every run.
        """
        from tools.setup.setup_utils import download

        folder_dir = tmp_path / 'mass_radius_curves'
        folder_dir.mkdir(parents=True)
        (folder_dir / 'a.txt').write_text('present')
        (folder_dir / 'b.txt').write_text('present')
        _marker(folder_dir).write_text('15727899\n')

        fetch, calls = _fake_fetch(folder_dir, 'refetched')
        with patch('tools.setup.setup_utils.download_zenodo_folder', side_effect=fetch):
            download(
                folder='mass_radius_curves',
                data_dir=tmp_path,
                zenodo_id=15727899,
                keep_files=['a.txt', 'b.txt'],
            )

        assert calls == []
        assert (folder_dir / 'a.txt').read_text() == 'present'
        assert (folder_dir / 'b.txt').read_text() == 'present'

    def test_folder_missing_a_kept_file_is_refreshed_despite_matching_marker(self, tmp_path):
        """A folder that matches the pin but is short a kept file is not reused.

        This is the state an incomplete download left cached before this fix:
        the marker matches the pin, so the old reuse check stood down forever
        without ever looking at what actually landed on disk.
        """
        from tools.setup.setup_utils import download

        folder_dir = tmp_path / 'mass_radius_curves'
        folder_dir.mkdir(parents=True)
        (folder_dir / 'a.txt').write_text('stale')
        _marker(folder_dir).write_text('15727899\n')

        calls = []

        def _fetch(zenodo_id, folder_dir=folder_dir, keep_files=None):
            calls.append(zenodo_id)
            folder_dir.mkdir(parents=True, exist_ok=True)
            for fname in keep_files:
                (folder_dir / fname).write_text('refetched')

        with patch('tools.setup.setup_utils.download_zenodo_folder', side_effect=_fetch):
            download(
                folder='mass_radius_curves',
                data_dir=tmp_path,
                zenodo_id=15727899,
                keep_files=['a.txt', 'b.txt'],
            )

        assert calls == [15727899]
        assert (folder_dir / 'b.txt').read_text() == 'refetched'


class TestMissingKeptFiles:
    """Unit tests for the shared completeness check used at every call site."""

    def test_no_keep_files_means_nothing_is_checked(self, tmp_path):
        from tools.setup.setup_utils import missing_kept_files

        assert missing_kept_files(tmp_path, None) == []

    def test_an_absent_file_is_missing(self, tmp_path):
        from tools.setup.setup_utils import missing_kept_files

        assert missing_kept_files(tmp_path, ['a.txt']) == ['a.txt']

    def test_a_zero_byte_file_is_missing(self, tmp_path):
        """A truncated mid-write download must not pass an existence-only check."""
        from tools.setup.setup_utils import missing_kept_files

        (tmp_path / 'a.txt').write_text('present')
        (tmp_path / 'b.txt').write_text('')
        assert missing_kept_files(tmp_path, ['a.txt', 'b.txt']) == ['b.txt']

    def test_a_directory_matching_a_kept_filename_is_missing(self, tmp_path):
        """A directory is not the kept file, even though it exists at that path."""
        from tools.setup.setup_utils import missing_kept_files

        (tmp_path / 'a.txt').mkdir()
        assert missing_kept_files(tmp_path, ['a.txt']) == ['a.txt']
