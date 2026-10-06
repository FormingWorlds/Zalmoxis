"""Tests for the data setup in ``tools.setup.setup_utils``.

The setup fetches each dataset through fwl-io into FWL_DATA and links it into
``<ZALMOXIS_ROOT>/data/<Folder>``. The network is never touched: the fetch is
replaced by a fake that builds the version directory.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.setup import setup_utils
from zalmoxis.datasets import manifest_path

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[1]


def test_every_folder_maps_to_a_declared_dataset():
    """Each data folder Zalmoxis reads names a dataset an installed manifest declares."""
    datasets = setup_utils._datasets()
    assert len(setup_utils.DATASETS) == 11
    missing = [key for key, _ in setup_utils.DATASETS.values() if key not in datasets]
    assert missing == []


def test_zalmoxis_manifest_declares_the_radial_profiles():
    """The Zalmoxis manifest pins the radial profiles to Zenodo and DataverseNL with their registry."""
    from fwl_io import load_manifest

    (ds,) = load_manifest(manifest_path())
    assert (ds.key, ds.zenodo) == ('interior.radial_profiles', '10.5281/zenodo.16837954')
    assert ds.dataverse == '10.34894/N6NVEU'
    assert ds.required_by == ('zalmoxis',)
    registry = ds.registry()
    assert len(registry) == 9
    assert registry['radiusdensitySeagerEarth.txt'] == 'md5:a33bb8fc796c641cb16dad4c481c9f9d'


def test_fetch_dataset_passes_the_manifest_entry(monkeypatch, tmp_path):
    """The fetcher gets the entry's location, pins, registry and archive type."""
    seen = {}

    class FakeFetcher:
        target_dir = tmp_path / 'r1'

        def fetch_all(self):
            seen['fetched'] = True

    def fake_create_fetcher(**kwargs):
        seen.update(kwargs)
        return FakeFetcher()

    monkeypatch.setattr('fwl_io.create_fetcher', fake_create_fetcher)
    ds = SimpleNamespace(
        subdir='a/b', zenodo='z', dataverse='d', registry=lambda: {'f': 'md5:0'}, extract='tar'
    )
    assert setup_utils.fetch_dataset('a.b', {'a.b': ds}) == tmp_path / 'r1'
    assert seen == {
        'subdir': 'a/b',
        'zenodo': 'z',
        'dataverse': 'd',
        'registry': {'f': 'md5:0'},
        'extract': 'tar',
        'fetched': True,
    }
    with pytest.raises(KeyError):
        setup_utils.fetch_dataset('x.y', {'a.b': ds})


def test_fetch_dataset_needs_fwl_data(monkeypatch):
    """Without FWL_DATA the fetch stops with fwl-io's own error before any download."""
    from fwl_io import MissingDataRootError

    monkeypatch.delenv('FWL_DATA', raising=False)
    datasets = setup_utils._datasets()
    with pytest.raises(MissingDataRootError, match='FWL_DATA'):
        setup_utils.fetch_dataset('interior.radial_profiles', datasets)


def test_link_folder_creates_keeps_and_replaces_links(tmp_path):
    """A missing link is made, a correct one kept, and one to another place replaced."""
    first, second = tmp_path / 'r1', tmp_path / 'r2'
    first.mkdir()
    second.mkdir()
    link = tmp_path / 'data' / 'EOS'
    setup_utils.link_folder(link, first)
    assert link.is_symlink() and link.resolve() == first.resolve()
    setup_utils.link_folder(link, first)
    assert link.resolve() == first.resolve()
    setup_utils.link_folder(link, second)
    assert link.resolve() == second.resolve()


def test_link_folder_keeps_a_real_folder_and_warns(tmp_path, caplog):
    """A real folder from an earlier setup stays, and the warning names how to remove it."""
    target = tmp_path / 'r1'
    target.mkdir()
    old = tmp_path / 'data' / 'EOS'
    old.mkdir(parents=True)
    (old / 'table.dat').write_text('old')
    with caplog.at_level(logging.WARNING):
        setup_utils.link_folder(old, target)
    assert not old.is_symlink()
    assert (old / 'table.dat').read_text() == 'old'
    assert f"rm -r '{old}'" in caplog.text


def _fake_fetch(root: Path):
    """Build a stand-in for fetch_dataset that makes each version directory."""

    def fetch(key, datasets):
        target = root / key.replace('.', '/') / 'r1'
        (target / 'EOS_Chabrier2021_HHe').mkdir(parents=True, exist_ok=True)
        return target

    return fetch


def test_download_data_links_every_folder(monkeypatch, tmp_path):
    """Every folder becomes a link to its dataset; Chabrier to the folder inside the archive."""
    monkeypatch.setattr(setup_utils, '_datasets', dict)
    monkeypatch.setattr(setup_utils, 'fetch_dataset', _fake_fetch(tmp_path / 'fwl'))
    monkeypatch.setattr(setup_utils, 'get_zalmoxis_root', lambda: str(tmp_path / 'zal'))
    setup_utils.download_data()
    data = tmp_path / 'zal' / 'data'
    assert sorted(p.name for p in data.iterdir()) == sorted(setup_utils.DATASETS)
    seager = tmp_path / 'fwl' / 'interior' / 'eos' / 'seager_2007' / 'r1'
    assert (data / 'EOS_Seager2007').resolve() == seager.resolve()
    chabrier = tmp_path / 'fwl' / 'interior' / 'eos' / 'chabrier_2021_hhe' / 'r1'
    assert (data / 'EOS_Chabrier2021_HHe').resolve() == (
        chabrier / 'EOS_Chabrier2021_HHe'
    ).resolve()


def test_download_data_stops_when_the_archive_folder_is_missing(monkeypatch, tmp_path):
    """An archive without the expected inner folder stops the setup with its name."""

    def fetch(key, datasets):
        target = tmp_path / key.replace('.', '/')
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(setup_utils, '_datasets', dict)
    monkeypatch.setattr(setup_utils, 'fetch_dataset', fetch)
    monkeypatch.setattr(setup_utils, 'get_zalmoxis_root', lambda: str(tmp_path / 'zal'))
    with pytest.raises(FileNotFoundError, match="no folder 'EOS_Chabrier2021_HHe'"):
        setup_utils.download_data()


def test_get_zalmoxis_sh_needs_fwl_data(tmp_path):
    """The setup script stops before any download when FWL_DATA is not set."""
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path)}
    result = subprocess.run(
        ['bash', str(REPO / 'tools' / 'setup' / 'get_zalmoxis.sh')],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert 'FWL_DATA is not set' in result.stderr
    assert 'Starting Zalmoxis data setup' not in result.stdout


def test_create_output_makes_the_folder_once(monkeypatch, tmp_path):
    """The output folder is created when missing and left alone when present."""
    monkeypatch.setattr(setup_utils, 'get_zalmoxis_root', lambda: str(tmp_path))
    setup_utils.create_output()
    (tmp_path / 'output' / 'keep.txt').write_text('x')
    setup_utils.create_output()
    assert (tmp_path / 'output' / 'keep.txt').read_text() == 'x'
