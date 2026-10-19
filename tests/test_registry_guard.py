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
    opened = []
    real_open = open

    def spy(f, mode='r', *a, **k):
        opened.append((str(f), mode))
        return real_open(f, mode, *a, **k)

    monkeypatch.setattr('builtins.open', spy)
    with pytest.raises(StaleTableError) as err:
        interpolation.load_paleos_unified_table(str(path))
    text = str(err.value)
    assert str(path) in text and KEY in text
    assert _md5(path.read_bytes()) in text and _md5(pinned.encode()) in text
    assert 'get_zalmoxis.sh' in text
    assert (str(path), 'r') not in opened


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
    monkeypatch.setenv('FWL_IO_OFFLINE', '1')
    monkeypatch.setattr(
        registry_guard,
        '_datasets',
        lambda: {KEY: (KEY.replace('.', '/'), f'10.5281/zenodo.{rid}', None, None)},
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
    monkeypatch.setenv('FWL_IO_OFFLINE', '1')
    monkeypatch.setattr(
        registry_guard,
        '_datasets',
        lambda: {KEY: (KEY.replace('.', '/'), f'10.5281/zenodo.{rid}', None, None)},
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

    def __init__(self, key, registry):
        self.key, self._registry = key, registry
        self.subdir, self.zenodo, self.dataverse, self.extract = (
            key.replace('.', '/'),
            None,
            None,
            None,
        )

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
