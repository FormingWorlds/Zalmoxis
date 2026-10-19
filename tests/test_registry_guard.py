"""Tests for the check that a PALEOS table on disk is the release the fwl-io registry pins."""

from __future__ import annotations

import hashlib
import os
import pickle

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from zalmoxis.eos import interpolation, registry_guard
from zalmoxis.eos.registry_guard import StaleTableError

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """No test reaches a mirror: a fetch that is not served from a local cache fails at once."""
    monkeypatch.setenv('FWL_IO_OFFLINE', '1')


NAME = 'paleos_iron_eos_table_pt.dat'
KEY = 'interior.eos.paleos_iron'


def _table_text():
    """A 4 x 4 unified-format table (P, T, rho, u, s, cp, cv, alpha, nabla_ad, phase)."""
    rows = ['# a small unified table']
    for p in np.logspace(9, 10, 4):
        for t in np.logspace(3, 3.5, 4):
            rows.append(
                f'{p:.8e} {t:.8e} 8.0e+03 1.0e+06 2.0e+03 5.0e+02 4.5e+02 2.0e-05 3.0e-01 liquid'
            )
    return '\n'.join(rows) + '\n'


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


@pytest.fixture
def pinned(monkeypatch):
    """Pin NAME to the md5 of the small table and return that text."""
    text = _table_text()
    monkeypatch.setattr(registry_guard, '_pins', lambda: {NAME: {KEY: _md5(text.encode())}})
    registry_guard._HASHES.clear()
    return text


def test_installed_registry_pins_every_paleos_table():
    pins = registry_guard._pins()
    for name in (
        'paleos_iron_eos_table_pt.dat',
        'paleos_mgsio3_eos_table_pt.dat',
        'paleos_water_eos_table_pt.dat',
        'paleos_mgsio3_tables_pt_proteus_solid.dat',
        'paleos_mgsio3_tables_pt_proteus_liquid.dat',
    ):
        assert name in pins, name
        assert all(len(h) == 32 for h in pins[name].values())


def test_current_table_is_read(tmp_path, pinned):
    path = tmp_path / NAME
    path.write_text(pinned)
    cached = interpolation.load_paleos_unified_table(str(path))
    assert cached['p_min'] == pytest.approx(1e9)


def test_stale_table_raises_before_it_is_read(tmp_path, pinned, monkeypatch):
    path = tmp_path / NAME
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    with pytest.raises(StaleTableError) as err:
        interpolation.load_paleos_unified_table(str(path))
    text = str(err.value)
    assert str(path) in text and KEY in text
    assert _md5(path.read_bytes()) in text and _md5(pinned.encode()) in text
    assert 'get_zalmoxis.sh' in text


def test_stale_error_is_not_caught_as_a_solver_failure():
    assert not issubclass(
        StaleTableError, (RuntimeError, ValueError, OSError, FloatingPointError)
    )


def test_unpinned_name_is_read_without_a_hash(tmp_path, pinned, monkeypatch):
    path = tmp_path / 'my_table.dat'
    path.write_text(pinned.replace('8.0e+03', '8.1e+03'))
    monkeypatch.setattr(
        registry_guard, '_md5', lambda p: pytest.fail('hashed an unpinned table')
    )
    interpolation.require_eos_file(str(path))


def test_hash_is_reused_until_the_file_changes(tmp_path, pinned, monkeypatch):
    path = tmp_path / NAME
    path.write_text(pinned)
    calls = []
    real = registry_guard._md5
    monkeypatch.setattr(registry_guard, '_md5', lambda p: calls.append(p) or real(p))
    registry_guard.verify(str(path))
    registry_guard.verify(str(path))
    assert len(calls) == 1
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    os.utime(path, ns=(1, 1))
    with pytest.raises(StaleTableError):
        registry_guard.verify(str(path))
    assert len(calls) == 2


def test_binary_cache_of_a_stale_table_is_not_used(tmp_path, pinned):
    path = tmp_path / NAME
    path.write_text(pinned)
    cache = {}
    interpolation._ensure_unified_cache(str(path), cache)
    assert os.path.isfile(interpolation.cache_path_for(str(tmp_path / NAME)))
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    os.utime(path, ns=(1, 1))
    with pytest.raises(StaleTableError):
        interpolation._ensure_unified_cache(str(path), {})


def test_binary_cache_without_its_table_needs_a_pinned_source(tmp_path, pinned):
    path = tmp_path / NAME
    path.write_text(pinned)
    interpolation._ensure_unified_cache(str(path), {})
    path.unlink()
    assert interpolation._ensure_unified_cache(str(path), {})['p_min'] == pytest.approx(1e9)
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    with pytest.raises(StaleTableError):
        interpolation._ensure_unified_cache(str(path), {})
    os.utime(path, ns=(1, 1))
    registry_guard._HASHES.clear()
    pkl = interpolation.cache_path_for(str(tmp_path / NAME))
    with open(pkl, 'rb') as f:
        stored = pickle.load(f)
    stored['source_md5'] = _md5(path.read_bytes())
    with open(pkl, 'wb') as f:
        pickle.dump(stored, f)
    path.unlink()
    with pytest.raises(StaleTableError):
        interpolation._ensure_unified_cache(str(path), {})


def test_binary_cache_in_the_old_layout_is_rebuilt(tmp_path, pinned):
    path = tmp_path / NAME
    path.write_text(pinned)
    entry = interpolation.load_paleos_unified_table(str(path))
    entry['p_min'] = -1.0
    with open(interpolation.cache_path_for(str(tmp_path / NAME)), 'wb') as f:
        pickle.dump(entry, f)
    assert interpolation._ensure_unified_cache(str(path), {})['p_min'] == pytest.approx(1e9)


def test_binary_cache_of_another_table_is_rebuilt(tmp_path, pinned):
    path = tmp_path / 'my_table.dat'
    path.write_text(pinned)
    pkl = interpolation.cache_path_for(str(path))
    other = interpolation.load_paleos_unified_table(str(path))
    other['p_min'] = -1.0
    with open(pkl, 'wb') as f:
        pickle.dump({'source_md5': '0' * 32, 'entry': other}, f)
    os.utime(path, ns=(1, 1))
    cached = interpolation._ensure_unified_cache(str(path), {})
    assert cached['p_min'] == pytest.approx(1e9)


def test_stale_table_in_its_version_directory_is_fetched_again(tmp_path, pinned, monkeypatch):
    rid = '22776069'
    root, cache_root = tmp_path / 'data', tmp_path / 'cache'
    rel = f'interior/eos/paleos_iron/r{rid}'
    (root / rel).mkdir(parents=True)
    (cache_root / rel).mkdir(parents=True)
    (cache_root / rel / NAME).write_text(pinned)
    path = root / rel / NAME
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    monkeypatch.setenv('FWL_DATA_CACHE', str(cache_root))
    monkeypatch.setenv('FWL_DATA', str(root))
    monkeypatch.setattr(
        registry_guard, '_manifest', lambda: (_Dataset(KEY, {}, f'10.5281/zenodo.{rid}'),)
    )
    registry_guard.verify(str(path))
    assert path.read_text() == pinned


def test_refetch_that_fails_offline_raises_stale(tmp_path, pinned, monkeypatch):
    rid = '22776069'
    rel = f'interior/eos/paleos_iron/r{rid}'
    (tmp_path / rel).mkdir(parents=True)
    path = tmp_path / rel / NAME
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    monkeypatch.delenv('FWL_DATA_CACHE', raising=False)
    monkeypatch.setenv('FWL_DATA', str(tmp_path))
    monkeypatch.setattr(
        registry_guard, '_manifest', lambda: (_Dataset(KEY, {}, f'10.5281/zenodo.{rid}'),)
    )
    with pytest.raises(StaleTableError, match='offline'):
        registry_guard.verify(str(path))


@settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    pos=st.integers(min_value=0, max_value=10_000), byte=st.integers(min_value=1, max_value=255)
)
def test_any_changed_byte_raises(tmp_path, pinned, pos, byte):
    data = bytearray(pinned.encode())
    i = pos % len(data)
    data[i] ^= byte
    path = tmp_path / NAME
    path.write_bytes(bytes(data))
    stamp = 2 + pos * 256 + byte
    os.utime(path, ns=(stamp, stamp))
    with pytest.raises(StaleTableError):
        registry_guard.verify(str(path))


class _Dataset:
    """A manifest dataset with a fixed registry."""

    def __init__(self, key, registry, zenodo=None):
        self.key, self._registry, self.zenodo = key, registry, zenodo
        self.subdir, self.dataverse, self.extract = key.replace('.', '/'), None, None

    def registry(self):
        return self._registry


def test_only_md5_entries_are_pinned_with_their_digest(monkeypatch):
    """A sha256 entry is not pinned; an md5 entry keeps its digest unchanged."""
    ds = _Dataset(
        'interior.eos.paleos_iron', {NAME: 'md5:' + 'a' * 32, 'x.dat': 'sha256:' + 'b' * 64}
    )
    monkeypatch.setattr(registry_guard, '_manifest', lambda: (ds,))
    assert registry_guard._pins.__wrapped__() == {NAME: {'interior.eos.paleos_iron': 'a' * 32}}


def test_the_installed_manifests_list_only_paleos_datasets():
    """_manifest keeps PALEOS datasets only, so tables of other EOS are not checked here."""
    keys = [ds.key for ds in registry_guard._manifest()]
    assert keys and all('paleos' in k for k in keys)
    assert 'density_melt.dat' not in registry_guard._pins()


def test_the_installed_registry_pins_the_v1_3_0_iron_table():
    """The pin is the digest itself: PALEOS v1.3.0 iron (Zenodo 22776069) has md5 259a2267..."""
    assert (
        registry_guard._pins()[NAME]['interior.eos.paleos_iron']
        == '259a2267dcf58af3b2c4dd389105b597'
    )


def test_a_file_of_another_size_with_the_same_mtime_is_hashed_again(tmp_path, pinned):
    """The hash cache key holds the size: a change of length with the mtime kept is seen."""
    path = tmp_path / NAME
    path.write_text(pinned)
    os.utime(path, ns=(5, 5))
    registry_guard.verify(str(path))
    path.write_text(pinned + '# one more line\n')
    os.utime(path, ns=(5, 5))
    with pytest.raises(StaleTableError):
        registry_guard.verify(str(path))


def test_the_binary_cache_has_its_own_file_name(tmp_path):
    """The cache is <table>.cache.pkl, so older code that reads <table>.pkl never reads it."""
    assert interpolation.cache_path_for(str(tmp_path / NAME)) == str(
        tmp_path / 'paleos_iron_eos_table_pt.cache.pkl'
    )
    assert not interpolation.cache_path_for('a/b.dat').endswith('b.pkl')


TWO_PHASE = 'paleos_mgsio3_tables_pt_proteus_solid.dat'
TWO_PHASE_KEY = 'interior.eos.paleos_mgsio3'


@pytest.fixture
def stale_tables(tmp_path, monkeypatch):
    """A stale unified iron and a stale 2-phase solid table under their pinned names."""
    text = _table_text()
    pins = {n: {k: _md5(text.encode())} for n, k in ((NAME, KEY), (TWO_PHASE, TWO_PHASE_KEY))}
    monkeypatch.setattr(registry_guard, '_pins', lambda: pins)
    registry_guard._HASHES.clear()
    stale = text.replace('8.0e+03', '8.1e+03', 1)
    for n in (NAME, TWO_PHASE):
        (tmp_path / n).write_text(stale)
    return tmp_path


@pytest.mark.parametrize(
    'reader',
    ['load_paleos_table', 'load_paleos_unified_table', 'load_paleos_all_properties'],
)
def test_every_reader_of_a_stale_table_raises(stale_tables, reader):
    """The 2-phase loader, the unified loader and the export reader all refuse a stale table."""
    from zalmoxis import eos_export

    name = TWO_PHASE if reader == 'load_paleos_table' else NAME
    fn = getattr(
        eos_export if reader == 'load_paleos_all_properties' else interpolation, reader
    )
    with pytest.raises(StaleTableError):
        fn(str(stale_tables / name))


def test_the_scalar_density_lookups_pass_the_stale_error_on(stale_tables):
    """get_paleos_unified_density and get_tabulated_eos do not turn a stale table into None."""
    from zalmoxis.eos.paleos import get_paleos_unified_density
    from zalmoxis.eos.seager import get_tabulated_eos

    with pytest.raises(StaleTableError):
        get_paleos_unified_density(5e9, 2000.0, {'eos_file': str(stale_tables / NAME)}, 1.0, {})
    md = {'solid_mantle': {'eos_file': str(stale_tables / TWO_PHASE), 'format': 'paleos'}}
    with pytest.raises(StaleTableError):
        get_tabulated_eos(5e9, md, 'solid_mantle', 2000.0, {})


def test_a_same_size_rewrite_with_the_mtime_kept_is_seen(tmp_path, pinned):
    """The hash cache key holds the change time, which utime cannot set back."""
    path = tmp_path / NAME
    path.write_text(pinned)
    os.utime(path, ns=(5, 5))
    registry_guard.verify(str(path))
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    os.utime(path, ns=(5, 5))
    with pytest.raises(StaleTableError):
        registry_guard.verify(str(path))


def test_a_link_to_a_stale_table_under_another_name_raises(tmp_path, pinned):
    """The name the file has on disk is checked too, so a link of another name is no way round."""
    real = tmp_path / NAME
    real.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    alias = tmp_path / 'my_iron.dat'
    alias.symlink_to(real)
    with pytest.raises(StaleTableError):
        registry_guard.verify(str(alias))


def test_a_stale_table_in_another_data_root_is_not_replaced(tmp_path, pinned, monkeypatch):
    """Only a table under the current FWL_DATA is fetched again; elsewhere it stays as it is."""
    rid = '22776069'
    old_root = tmp_path / 'old'
    rel = f'interior/eos/paleos_iron/r{rid}'
    (old_root / rel).mkdir(parents=True)
    path = old_root / rel / NAME
    stale = pinned.replace('8.0e+03', '8.1e+03', 1)
    path.write_text(stale)
    (tmp_path / 'cache' / rel).mkdir(parents=True)
    (tmp_path / 'cache' / rel / NAME).write_text(pinned)
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'current'))
    monkeypatch.setenv('FWL_DATA_CACHE', str(tmp_path / 'cache'))
    with pytest.raises(StaleTableError):
        registry_guard.verify(str(path))
    assert path.read_text() == stale


def test_the_refetch_uses_the_location_the_manifest_gives(tmp_path, pinned, monkeypatch):
    """With the real manifest, a stale iron table in its version directory is fetched again."""
    ds = next(d for d in registry_guard._manifest() if d.key == KEY)
    from fwl_io.doi import zenodo_record_id

    rel = f'{ds.subdir}/r{zenodo_record_id(ds.zenodo)}'
    root = tmp_path / 'data'
    (root / rel).mkdir(parents=True)
    (tmp_path / 'cache' / rel).mkdir(parents=True)
    (tmp_path / 'cache' / rel / NAME).write_text(pinned)
    path = root / rel / NAME
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    monkeypatch.setenv('FWL_DATA', str(root))
    monkeypatch.setenv('FWL_DATA_CACHE', str(tmp_path / 'cache'))
    registry_guard.verify(str(path))
    assert path.read_text() == pinned


def test_a_fetcher_that_cannot_be_built_gives_the_stale_error(tmp_path, pinned, monkeypatch):
    """An fwl-io error while the fetcher is built is a StaleTableError, not a ValueError."""
    import fwl_io

    rid = '22776069'
    rel = f'interior/eos/paleos_iron/r{rid}'
    (tmp_path / rel).mkdir(parents=True)
    path = tmp_path / rel / NAME
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    monkeypatch.setenv('FWL_DATA', str(tmp_path))

    def broken(**kwargs):
        raise ValueError('bad manifest entry')

    monkeypatch.setattr(fwl_io, 'create_fetcher', broken)
    with pytest.raises(StaleTableError, match='bad manifest entry'):
        registry_guard.verify(str(path))


@pytest.mark.parametrize(
    'stored',
    [{'entry': {'p_min': -1.0}}, {'format': 0, 'entry': {'p_min': -1.0}}],
    ids=['no source md5', 'another format'],
)
def test_a_cache_without_its_md5_or_of_another_format_is_rebuilt(tmp_path, pinned, stored):
    """A cache that lacks the md5 or carries another format number is rebuilt, not used."""
    path = tmp_path / NAME
    path.write_text(pinned)
    full = {'source_md5': _md5(pinned.encode()), **stored}
    if 'format' not in stored:
        full.pop('source_md5')
    with open(interpolation.cache_path_for(str(path)), 'wb') as f:
        pickle.dump(full, f)
    assert interpolation._ensure_unified_cache(str(path), {})['p_min'] == pytest.approx(1e9)


def test_without_fwl_data_a_stale_table_is_not_fetched(tmp_path, pinned, monkeypatch):
    """With no FWL_DATA there is no tree to fetch into: the stale table raises."""
    rel = 'interior/eos/paleos_iron/r22776069'
    (tmp_path / rel).mkdir(parents=True)
    path = tmp_path / rel / NAME
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    monkeypatch.delenv('FWL_DATA', raising=False)
    monkeypatch.setattr(
        registry_guard, '_manifest', lambda: (_Dataset(KEY, {}, '10.5281/zenodo.22776069'),)
    )
    with pytest.raises(StaleTableError, match='get_zalmoxis.sh'):
        registry_guard.verify(str(path))


def test_a_dataset_without_a_zenodo_pin_is_not_fetched(tmp_path, pinned, monkeypatch):
    """A dataset with no Zenodo record has no version directory: the stale table raises."""
    rel = 'interior/eos/paleos_iron/r22776069'
    (tmp_path / rel).mkdir(parents=True)
    path = tmp_path / rel / NAME
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    monkeypatch.setenv('FWL_DATA', str(tmp_path))
    monkeypatch.setattr(registry_guard, '_manifest', lambda: (_Dataset(KEY, {}),))
    with pytest.raises(StaleTableError, match='get_zalmoxis.sh'):
        registry_guard.verify(str(path))
