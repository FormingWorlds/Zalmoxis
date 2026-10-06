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

# Files Zalmoxis reads from each folder (eos_properties, melting_curves, tests, tools).
READ_FILES = {
    'EOS_Seager2007': [f'eos_seager07_{m}.txt' for m in ('iron', 'silicate', 'water')],
    'EOS_WolfBower2018_1TPa': [
        'density_melt.dat',
        'density_solid.dat',
        'adiabat_temp_grad_melt.dat',
    ],
    'radial_profiles': [
        'radiusdensityWagner.txt',
        'radiusdensitySeagerEarthbymass.txt',
        'radiusdensitySeagerwaterbymass.txt',
    ],
    'mass_radius_curves': ['massradiusEarthlikeRocky.txt', 'massradiusFe.txt'],
    'EOS_RTPress_melt_100TPa': ['density_melt.dat', 'adiabat_temp_grad_melt.dat'],
    'melting_curves_Monteux-600': ['liquidus.dat', 'solidus.dat'],
    'EOS_PALEOS_MgSiO3': [
        f'paleos_mgsio3_tables_pt_proteus_{p}{r}.dat'
        for p in ('solid', 'liquid')
        for r in ('', '_highres')
    ],
    'EOS_PALEOS_iron': ['paleos_iron_eos_table_pt.dat'],
    'EOS_PALEOS_MgSiO3_unified': ['paleos_mgsio3_eos_table_pt.dat'],
    'EOS_PALEOS_H2O': ['paleos_water_eos_table_pt.dat'],
}


@pytest.mark.parametrize('folder', sorted(READ_FILES))
def test_each_folder_maps_to_the_dataset_holding_its_files(folder):
    """The dataset a folder links to declares the files Zalmoxis reads from that folder."""
    key, inner = setup_utils.FOLDERS[folder]
    assert inner == ''
    assert set(READ_FILES[folder]) <= set(setup_utils._datasets()[key].registry())


def test_the_chabrier_folder_is_the_one_inside_its_archive():
    """Chabrier ships as an archive whose files sit in its own top-level folder."""
    key, inner = setup_utils.FOLDERS['EOS_Chabrier2021_HHe']
    ds = setup_utils._datasets()[key]
    assert (ds.extract, inner) == ('tar', 'EOS_Chabrier2021_HHe')
    assert set(setup_utils.FOLDERS) == set(READ_FILES) | {'EOS_Chabrier2021_HHe'}


def test_zalmoxis_manifest_declares_the_radial_profiles():
    """The Zalmoxis manifest pins the radial profiles to Zenodo and DataverseNL."""
    from fwl_io import load_manifest

    (ds,) = load_manifest(manifest_path())
    assert (ds.key, ds.zenodo) == ('interior.radial_profiles', '10.5281/zenodo.16837954')
    assert ds.dataverse == '10.34894/N6NVEU'
    assert ds.required_by == ('zalmoxis',)


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
        'progress': True,
        'fetched': True,
    }


def test_fetch_dataset_names_an_undeclared_key():
    """A key no installed manifest declares asks for a newer fwl-io."""
    with pytest.raises(
        KeyError, match='x.y is in no installed fwl-io manifest; upgrade fwl-io'
    ):
        setup_utils.fetch_dataset('x.y', {})


def test_fetch_dataset_needs_fwl_data(monkeypatch):
    """Without FWL_DATA the fetch stops with fwl-io's own error before any download."""
    from fwl_io import MissingDataRootError

    monkeypatch.delenv('FWL_DATA', raising=False)
    datasets = setup_utils._datasets()
    with pytest.raises(MissingDataRootError, match='FWL_DATA'):
        setup_utils.fetch_dataset('interior.radial_profiles', datasets)


@pytest.fixture
def tree(tmp_path):
    """Return a data root with two version directories and a data/ folder path."""
    root = tmp_path / 'fwl'
    for name in ('r1', 'r2'):
        (root / name).mkdir(parents=True)
    return SimpleNamespace(
        root=root, r1=root / 'r1', r2=root / 'r2', link=tmp_path / 'data' / 'EOS'
    )


def test_link_folder_makes_and_moves_its_own_links(tree):
    """A missing link is made, and a link into the data root moves to the new target."""
    assert setup_utils.link_folder(tree.link, tree.r1, tree.root)
    assert tree.link.resolve() == tree.r1.resolve()
    assert setup_utils.link_folder(tree.link, tree.r2, tree.root)
    assert tree.link.resolve() == tree.r2.resolve()


def test_link_folder_replaces_a_dangling_link_and_an_empty_folder(tree):
    """A dangling link and an empty folder hold no data, so both become links."""
    tree.link.parent.mkdir()
    tree.link.symlink_to(tree.root / 'gone')
    assert setup_utils.link_folder(tree.link, tree.r1, tree.root)
    assert tree.link.resolve() == tree.r1.resolve()
    tree.link.unlink()
    tree.link.mkdir()
    assert setup_utils.link_folder(tree.link, tree.r1, tree.root)
    assert tree.link.is_symlink()


def test_link_folder_keeps_the_users_own_data(tree, tmp_path):
    """A folder with files, a link outside the data root and a plain file all stay."""
    tree.link.mkdir(parents=True)
    (tree.link / 'table.dat').write_text('old')
    assert not setup_utils.link_folder(tree.link, tree.r1, tree.root)
    assert (tree.link / 'table.dat').read_text() == 'old'

    mine = tmp_path / 'mine'
    mine.mkdir()
    other = tree.link.parent / 'OTHER'
    other.symlink_to(mine)
    assert not setup_utils.link_folder(other, tree.r1, tree.root)
    assert other.resolve() == mine.resolve()

    plain = tree.link.parent / 'PLAIN'
    plain.write_text('x')
    assert not setup_utils.link_folder(plain, tree.r1, tree.root)
    assert plain.read_text() == 'x'


def _fake_fetch(root: Path):
    """Build a stand-in for fetch_dataset that makes each version directory."""

    def fetch(key, datasets):
        target = root / key.replace('.', '/') / 'r1'
        (target / 'EOS_Chabrier2021_HHe').mkdir(parents=True, exist_ok=True)
        return target

    return fetch


@pytest.fixture
def fake_setup(monkeypatch, tmp_path):
    """Route download_data to a fake fetch below tmp_path/fwl and a Zalmoxis root."""
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'fwl'))
    monkeypatch.setattr(setup_utils, 'fetch_dataset', _fake_fetch(tmp_path / 'fwl'))
    monkeypatch.setattr(setup_utils, 'get_zalmoxis_root', lambda: str(tmp_path / 'zal'))
    return tmp_path


def test_download_data_links_every_folder_to_its_dataset(fake_setup):
    """Every folder links to the version directory of its key; Chabrier to its inner folder."""
    setup_utils.download_data()
    data = fake_setup / 'zal' / 'data'
    assert sorted(p.name for p in data.iterdir()) == sorted(setup_utils.FOLDERS)
    for folder, (key, inner) in setup_utils.FOLDERS.items():
        target = fake_setup / 'fwl' / key.replace('.', '/') / 'r1' / inner
        assert (data / folder).resolve() == target.resolve()


def test_download_data_lists_the_folders_it_keeps(fake_setup, caplog):
    """A run that keeps a folder with data ends with one warning naming how to remove it."""
    old = fake_setup / 'zal' / 'data' / 'EOS_Seager2007'
    old.mkdir(parents=True)
    (old / 'eos_seager07_iron.txt').write_text('old')
    with caplog.at_level(logging.WARNING):
        setup_utils.download_data()
    assert not old.is_symlink()
    assert (old / 'eos_seager07_iron.txt').read_text() == 'old'
    assert (fake_setup / 'zal' / 'data' / 'EOS_PALEOS_iron').is_symlink()
    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert f"rm -r '{old}'" in record.getMessage()


def test_download_data_stops_when_the_archive_folder_is_missing(monkeypatch, tmp_path):
    """An archive without the expected inner folder stops the setup with its name."""

    def fetch(key, datasets):
        target = tmp_path / key.replace('.', '/')
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr(setup_utils, 'fetch_dataset', fetch)
    monkeypatch.setattr(setup_utils, 'get_zalmoxis_root', lambda: str(tmp_path / 'zal'))
    monkeypatch.setenv('FWL_DATA', str(tmp_path))
    with pytest.raises(FileNotFoundError, match="no folder 'EOS_Chabrier2021_HHe'"):
        setup_utils.download_data()


@pytest.mark.parametrize('fwl_data', [None, ''])
def test_get_zalmoxis_sh_needs_fwl_data(tmp_path, fwl_data):
    """The setup script stops before any download when FWL_DATA is unset or empty."""
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path)}
    if fwl_data is not None:
        env['FWL_DATA'] = fwl_data
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
