"""Tests for the check that a PALEOS table on disk is the release the fwl-io registry pins."""

from __future__ import annotations

import hashlib
import os
import time

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
    monkeypatch.setattr(
        interpolation, 'open', lambda *a, **k: pytest.fail('table opened'), raising=False
    )
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
    stored = interpolation.read_cache(pkl)
    stored['source_md5'] = _md5(path.read_bytes())
    interpolation.write_cache(pkl, stored)
    path.unlink()
    with pytest.raises(StaleTableError):
        interpolation._ensure_unified_cache(str(path), {})


def test_binary_cache_in_the_old_layout_is_rebuilt(tmp_path, pinned):
    path = tmp_path / NAME
    path.write_text(pinned)
    entry = interpolation.load_paleos_unified_table(str(path))
    entry['p_min'] = -1.0
    interpolation.write_cache(interpolation.cache_path_for(str(tmp_path / NAME)), entry)
    assert interpolation._ensure_unified_cache(str(path), {})['p_min'] == pytest.approx(1e9)


def test_binary_cache_of_another_table_is_rebuilt(tmp_path, pinned):
    path = tmp_path / 'my_table.dat'
    path.write_text(pinned)
    pkl = interpolation.cache_path_for(str(path))
    other = interpolation.load_paleos_unified_table(str(path))
    other['p_min'] = -1.0
    interpolation.write_cache(pkl, _valid('0' * 32, other))
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


def test_pins_keep_their_algorithm_and_digest(monkeypatch):
    """An md5 pin keeps its bare digest; another algorithm keeps its prefix, and a bare
    registry hash is sha256."""
    ds = _Dataset(
        'interior.eos.paleos_iron',
        {NAME: 'MD5:' + 'A' * 32, 'x.dat': 'sha256:' + 'b' * 64, 'y.dat': 'c' * 64},
    )
    monkeypatch.setattr(registry_guard, '_manifest', lambda: (ds,))
    key = 'interior.eos.paleos_iron'
    assert registry_guard._pins.__wrapped__() == {
        NAME: {key: 'a' * 32},
        'x.dat': {key: 'sha256:' + 'b' * 64},
        'y.dat': {key: 'sha256:' + 'c' * 64},
    }


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


# Keys of the entry load_paleos_unified_table returns, for CACHE_FORMAT 1.
_ENTRY_KEYS_FORMAT_1 = set(
    [
        'density_grid',
        'density_interp',
        'density_nn',
        'dlog_p',
        'dlog_t',
        'liquidus_log_p',
        'liquidus_log_t',
        'logp_max',
        'logp_min',
        'logt_max',
        'logt_min',
        'logt_valid_max',
        'logt_valid_min',
        'n_p',
        'n_t',
        'nabla_ad_grid',
        'nabla_ad_interp',
        'nabla_ad_nn',
        'p_max',
        'p_min',
        'phase_grid',
        't_max',
        't_min',
        'type',
        'unique_log_p',
        'unique_log_t',
    ]
)


def _valid(source_md5, entry):
    """A cache payload of this code for a table of md5 ``source_md5``."""
    return {
        'format': interpolation.CACHE_FORMAT,
        'builder': interpolation.cache_builder(),
        'source_md5': source_md5,
        'entry': entry,
    }


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
    time.sleep(0.05)  # past a coarse ctime tick
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


@pytest.mark.parametrize(
    'rel',
    ['interior/eos/paleos_iron/r22776069', 'elsewhere/r22776069'],
    ids=['same layout', 'other folder of the same version'],
)
def test_a_stale_table_in_another_data_root_is_not_replaced(tmp_path, pinned, monkeypatch, rel):
    """Only a table in its version directory under the current FWL_DATA is fetched again; a table
    under another root stays as it is, no fetch starts and the current root stays empty."""
    import fwl_io

    rid = '22776069'
    old_root = tmp_path / 'old'
    (old_root / rel).mkdir(parents=True)
    monkeypatch.setattr(fwl_io, 'create_fetcher', lambda **k: pytest.fail('fetch started'))
    monkeypatch.setattr(
        registry_guard,
        '_manifest',
        lambda: (_Dataset(KEY, {NAME: 'md5:x'}, zenodo=f'10.5281/zenodo.{rid}'),),
    )
    path = old_root / rel / NAME
    stale = pinned.replace('8.0e+03', '8.1e+03', 1)
    path.write_text(stale)
    (tmp_path / 'current').mkdir()
    monkeypatch.setenv('FWL_DATA', str(tmp_path / 'current'))
    with pytest.raises(StaleTableError):
        registry_guard.verify(str(path))
    assert path.read_text() == stale
    assert list((tmp_path / 'current').iterdir()) == []


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
    [
        {'format': 0, 'source_md5': None, 'entry': {'p_min': -1.0}},
        {'format': 1, 'source_md5': '0' * 32, 'entry': {'p_min': -1.0}},
        {'format': 1, 'entry': {'p_min': -1.0}},
        {'format': 1, 'source_md5': None},
    ],
    ids=['another format', 'another md5', 'no md5', 'no entry'],
)
def test_a_cache_of_another_format_md5_or_without_an_entry_is_rebuilt(tmp_path, pinned, stored):
    """With the table present, a cache whose format or md5 differs, or that lacks either or its
    entry, is rebuilt from the text table, not used."""
    path = tmp_path / NAME
    path.write_text(pinned)
    full = {**stored}
    if full.get('source_md5') is None and 'source_md5' in full:
        full['source_md5'] = _md5(pinned.encode())
    interpolation.write_cache(
        interpolation.cache_path_for(str(path)),
        {'builder': interpolation.cache_builder(), **full},
    )
    assert interpolation._ensure_unified_cache(str(path), {})['p_min'] == pytest.approx(1e9)


def test_a_cache_without_its_table_and_without_an_md5_is_refused(tmp_path, pinned):
    """With no table on disk, a cache of a pinned name that holds no md5 cannot be trusted."""
    path = tmp_path / NAME
    interpolation.write_cache(
        interpolation.cache_path_for(str(path)),
        {
            'format': interpolation.CACHE_FORMAT,
            'builder': interpolation.cache_builder(),
            'entry': {'p_min': -1.0},
        },
    )
    with pytest.raises(StaleTableError, match='cache.pkl'):
        interpolation._ensure_unified_cache(str(path), {})


@pytest.mark.parametrize(
    'link, target',
    [
        (NAME, 'paleos_water_eos_table_pt.dat'),
        (
            'paleos_mgsio3_tables_pt_proteus_solid.dat',
            'paleos_mgsio3_tables_pt_proteus_liquid.dat',
        ),
    ],
    ids=['iron name to water', 'solid name to liquid'],
)
def test_a_link_named_like_one_table_to_another_table_raises(
    tmp_path, monkeypatch, link, target
):
    """A link must hold a release pinned for its own name as well as for the target's name."""
    texts = {link: _table_text(), target: _table_text().replace('8.0e+03', '9.0e+03')}
    pins = {
        link: {'interior.eos.a': _md5(texts[link].encode())},
        target: {'interior.eos.a': _md5(texts[target].encode())},
    }
    monkeypatch.setattr(registry_guard, '_pins', lambda: pins)
    registry_guard._HASHES.clear()
    (tmp_path / target).write_text(texts[target])
    (tmp_path / 'links').mkdir()
    (tmp_path / 'links' / link).symlink_to(tmp_path / target)
    with pytest.raises(StaleTableError):
        registry_guard.verify(str(tmp_path / 'links' / link))
    registry_guard.verify(str(tmp_path / target))


def test_a_name_in_other_letter_case_is_checked(tmp_path, pinned):
    """A pinned name written in other case (one file on a case-insensitive disk) is checked."""
    path = tmp_path / NAME.upper()
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    with pytest.raises(StaleTableError):
        registry_guard.verify(str(path))


def test_a_fetch_that_leaves_the_file_stale_raises(tmp_path, pinned, monkeypatch):
    """After a fetch the file is checked again: a fetch that did not replace it is no pass."""
    import fwl_io

    rel = 'interior/eos/paleos_iron/r22776069'
    (tmp_path / rel).mkdir(parents=True)
    path = tmp_path / rel / NAME
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    monkeypatch.setenv('FWL_DATA', str(tmp_path))
    monkeypatch.setattr(
        registry_guard, '_manifest', lambda: (_Dataset(KEY, {}, '10.5281/zenodo.22776069'),)
    )

    class _NoOp:
        def fetch(self, name):
            return None

    monkeypatch.setattr(fwl_io, 'create_fetcher', lambda **k: _NoOp())
    with pytest.raises(StaleTableError):
        registry_guard.verify(str(path))


def test_a_stale_table_below_a_linked_dataset_folder_names_the_fwl_io_refusal(
    tmp_path, pinned, monkeypatch
):
    """A dataset folder that links out of FWL_DATA is its version directory, but fwl-io does
    not write outside its data root: the stale table raises with that reason and stays."""
    root, elsewhere = tmp_path / 'data', tmp_path / 'scratch' / 'paleos_iron'
    (root / 'interior/eos').mkdir(parents=True)
    (elsewhere / 'r22776069').mkdir(parents=True)
    (root / 'interior/eos/paleos_iron').symlink_to(elsewhere)
    rel = 'interior/eos/paleos_iron/r22776069'
    (tmp_path / 'cache' / rel).mkdir(parents=True)
    (tmp_path / 'cache' / rel / NAME).write_text(pinned)
    path = root / rel / NAME
    stale = pinned.replace('8.0e+03', '8.1e+03', 1)
    path.write_text(stale)
    monkeypatch.setenv('FWL_DATA', str(root))
    monkeypatch.setenv('FWL_DATA_CACHE', str(tmp_path / 'cache'))
    monkeypatch.setattr(
        registry_guard, '_manifest', lambda: (_Dataset(KEY, {}, '10.5281/zenodo.22776069'),)
    )
    with pytest.raises(StaleTableError, match='escapes the data root'):
        registry_guard.verify(str(path))
    assert path.read_text() == stale


def test_a_pinned_name_of_no_manifest_dataset_is_not_fetched(tmp_path, pinned, monkeypatch):
    """A pin whose dataset key the manifest does not list gives the stale error."""
    rel = 'interior/eos/paleos_iron/r22776069'
    (tmp_path / rel).mkdir(parents=True)
    path = tmp_path / rel / NAME
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    monkeypatch.setenv('FWL_DATA', str(tmp_path))
    monkeypatch.setattr(registry_guard, '_manifest', lambda: ())
    with pytest.raises(StaleTableError):
        registry_guard.verify(str(path))


def test_the_export_reader_says_a_folder_is_not_a_file(tmp_path, pinned):
    """A folder under a pinned name is not a file: FileNotFoundError, not an md5 of a folder."""
    from zalmoxis import eos_export

    (tmp_path / NAME).mkdir()
    with pytest.raises(FileNotFoundError, match='is not a file'):
        eos_export.load_paleos_all_properties(str(tmp_path / NAME))


def test_the_cache_entry_layout_is_that_of_cache_format(tmp_path, pinned):
    """A change of the cached entry keys must come with a new CACHE_FORMAT."""
    path = tmp_path / NAME
    path.write_text(pinned)
    keys = set(interpolation.load_paleos_unified_table(str(path)))
    assert interpolation.CACHE_FORMAT == 1, 'update the key set below with the new format'
    assert keys == _ENTRY_KEYS_FORMAT_1, 'the entry keys changed: raise CACHE_FORMAT'


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


def _digest(text, alg):
    return hashlib.new(alg, text.encode()).hexdigest()


@pytest.mark.parametrize(
    'pin',
    ['sha256:{sha256}', '{sha256}', 'MD5:{md5}', 'SHA256:{sha256}'],
    ids=['sha256', 'bare (sha256)', 'upper MD5', 'upper SHA256'],
)
def test_a_pin_in_another_hash_form_stops_and_names_it(tmp_path, monkeypatch, pin):
    """A sha256 pin or a bare hash (sha256, as fwl-io reads it) stops with the pin named; an
    upper-case MD5 prefix is an md5 pin and checks."""
    text = _table_text()
    value = pin.format(sha256=_digest(text, 'sha256'), md5=_digest(text, 'md5'))
    monkeypatch.setattr(registry_guard, '_manifest', lambda: (_Dataset(KEY, {NAME: value}),))
    monkeypatch.setattr(registry_guard, '_pins', registry_guard._pins.__wrapped__)
    registry_guard._HASHES.clear()
    good, bad = tmp_path / 'good', tmp_path / 'bad'
    good.mkdir(), bad.mkdir()
    (good / NAME).write_text(text)
    (bad / NAME).write_text(text.replace('8.0e+03', '8.1e+03', 1))
    if pin.startswith('MD5'):
        registry_guard.verify(str(good / NAME))
    else:
        with pytest.raises(StaleTableError, match=f'pinned as sha256:{value[-64:]}.*only md5'):
            registry_guard.verify(str(good / NAME))
    with pytest.raises(StaleTableError):
        registry_guard.verify(str(bad / NAME))


def test_a_pin_of_an_unknown_algorithm_stops(tmp_path, monkeypatch):
    """A PALEOS pin that cannot be checked is not skipped: the table is refused."""
    monkeypatch.setattr(
        registry_guard, '_manifest', lambda: (_Dataset(KEY, {NAME: 'crc32:abcd'}),)
    )
    monkeypatch.setattr(registry_guard, '_pins', registry_guard._pins.__wrapped__)
    (tmp_path / NAME).write_text(_table_text())
    with pytest.raises(StaleTableError, match='pinned as crc32:abcd by .*only md5'):
        registry_guard.verify(str(tmp_path / NAME))


def test_two_datasets_that_pin_one_name_differently_stop(tmp_path, monkeypatch):
    """When two datasets pin one name to different md5 values, no table of that name is read."""
    manifest = (
        _Dataset(KEY, {NAME: 'md5:' + 'a' * 32}),
        _Dataset('interior.eos.paleos_iron_old', {NAME: 'md5:' + 'b' * 32}),
    )
    monkeypatch.setattr(registry_guard, '_manifest', lambda: manifest)
    monkeypatch.setattr(registry_guard, '_pins', registry_guard._pins.__wrapped__)
    (tmp_path / NAME).write_text(_table_text())
    with pytest.raises(StaleTableError, match='disagree'):
        registry_guard.verify(str(tmp_path / NAME))


def test_any_other_failure_of_the_check_is_a_stale_error(tmp_path, pinned, monkeypatch):
    """A broken registry or an unreadable table stops the read with StaleTableError, never None."""
    path = tmp_path / NAME
    path.write_text(pinned)

    def broken():
        raise ValueError('registry does not match its files')

    monkeypatch.setattr(registry_guard, '_pins', broken)
    with pytest.raises(StaleTableError, match='registry does not match'):
        registry_guard.verify(str(path))


@pytest.mark.skipif(os.geteuid() == 0, reason='root reads a file of mode 000')
def test_an_unreadable_table_is_a_stale_error_and_its_cache_is_still_used(tmp_path, pinned):
    """A table of mode 000 cannot be checked (StaleTableError); a cache of its pinned md5 beside it
    is used, as on a shared FWL_DATA where fwl-io writes tables with mode 600."""
    path = tmp_path / NAME
    path.write_text(pinned)
    interpolation._ensure_unified_cache(str(path), {})
    path.chmod(0)
    try:
        with pytest.raises(StaleTableError):
            registry_guard.verify(str(path))
        assert interpolation._ensure_unified_cache(str(path), {})['p_min'] == pytest.approx(1e9)
    finally:
        path.chmod(0o644)


def test_a_damaged_cache_is_rebuilt_without_unpickling(tmp_path, pinned, monkeypatch):
    """Zeroed bytes in a cache fail its digest, so pickle never reads them."""
    path = tmp_path / NAME
    path.write_text(pinned)
    interpolation._ensure_unified_cache(str(path), {})
    pkl = interpolation.cache_path_for(str(path))
    raw = bytearray(open(pkl, 'rb').read())
    raw[200:208] = bytes(8)
    open(pkl, 'wb').write(bytes(raw))
    monkeypatch.setattr(
        interpolation.pickle, 'loads', lambda b: pytest.fail('damaged bytes read')
    )
    assert interpolation._ensure_unified_cache(str(path), {})['p_min'] == pytest.approx(1e9)


def test_the_cache_is_written_by_a_rename(tmp_path, pinned, monkeypatch):
    """The cache appears in one step: written to a temporary file and renamed into place."""
    path = tmp_path / NAME
    path.write_text(pinned)
    renames = []
    real = os.replace
    monkeypatch.setattr(
        interpolation.os, 'replace', lambda a, b: renames.append((a, b)) or real(a, b)
    )
    interpolation._ensure_unified_cache(str(path), {})
    assert [b for _, b in renames] == [interpolation.cache_path_for(str(path))]
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [NAME, NAME.replace('.dat', '.cache.pkl')]
    )


def test_a_cache_of_other_builder_code_is_rebuilt(tmp_path, pinned, monkeypatch):
    """A cache built by other loader code or other numpy or scipy versions is rebuilt."""
    path = tmp_path / NAME
    path.write_text(pinned)
    other = interpolation.load_paleos_unified_table(str(path))
    other['p_min'] = -1.0
    stored = {**_valid(_md5(pinned.encode()), other), 'builder': 'x' * 64}
    interpolation.write_cache(interpolation.cache_path_for(str(path)), stored)
    assert interpolation._ensure_unified_cache(str(path), {})['p_min'] == pytest.approx(1e9)
    builder = interpolation.cache_builder()
    monkeypatch.setattr(interpolation.np, '__version__', '0.0.1')
    interpolation.cache_builder.cache_clear()
    try:
        assert interpolation.cache_builder() != builder
    finally:
        interpolation.cache_builder.cache_clear()


def test_the_builder_reads_the_loader_code_when_its_name_is_patched(monkeypatch):
    """A first builder call while load_paleos_unified_table is patched gives the same identity."""
    interpolation.cache_builder.cache_clear()
    builder = interpolation.cache_builder()
    interpolation.cache_builder.cache_clear()
    monkeypatch.setattr(interpolation, 'load_paleos_unified_table', lambda eos_file: {})
    try:
        assert interpolation.cache_builder() == builder
    finally:
        interpolation.cache_builder.cache_clear()


def test_the_refetch_is_logged_and_a_failed_one_leaves_the_file(
    tmp_path, pinned, monkeypatch, caplog
):
    """A refetch names the file and its dataset in a warning; offline, the stale file stays."""
    rid = '22776069'
    rel = f'interior/eos/paleos_iron/r{rid}'
    root = tmp_path / 'data'
    (root / rel).mkdir(parents=True)
    path = root / rel / NAME
    stale = pinned.replace('8.0e+03', '8.1e+03', 1)
    path.write_text(stale)
    monkeypatch.setenv('FWL_DATA', str(root))
    monkeypatch.setattr(
        registry_guard, '_manifest', lambda: (_Dataset(KEY, {}, f'10.5281/zenodo.{rid}'),)
    )
    with pytest.raises(StaleTableError):
        registry_guard.verify(str(path))
    assert path.read_text() == stale
    (tmp_path / 'cache' / rel).mkdir(parents=True)
    (tmp_path / 'cache' / rel / NAME).write_text(pinned)
    monkeypatch.setenv('FWL_DATA_CACHE', str(tmp_path / 'cache'))
    with caplog.at_level('WARNING', logger='zalmoxis.eos.registry_guard'):
        registry_guard.verify(str(path))
    assert 'fetched it again' in caplog.text and KEY in caplog.text


def test_a_link_whose_names_share_no_md5_creates_no_fetcher(tmp_path, monkeypatch):
    """No fetch is tried for a link whose names have no common pinned md5."""
    import fwl_io

    target = 'paleos_water_eos_table_pt.dat'
    rel = 'interior/eos/paleos_iron/r22776069'
    (tmp_path / rel).mkdir(parents=True)
    texts = {NAME: _table_text(), target: _table_text().replace('8.0e+03', '9.0e+03')}
    pins = {n: {KEY: _md5(t.encode())} for n, t in texts.items()}
    monkeypatch.setattr(registry_guard, '_pins', lambda: pins)
    monkeypatch.setattr(
        registry_guard, '_manifest', lambda: (_Dataset(KEY, {}, '10.5281/zenodo.22776069'),)
    )
    monkeypatch.setenv('FWL_DATA', str(tmp_path))
    monkeypatch.setattr(fwl_io, 'create_fetcher', lambda **k: pytest.fail('fetcher built'))
    (tmp_path / rel / target).write_text(texts[NAME])
    (tmp_path / NAME).symlink_to(tmp_path / rel / target)
    with pytest.raises(StaleTableError):
        registry_guard.verify(str(tmp_path / NAME))


def test_a_data_link_into_the_version_directory_is_fetched_into(tmp_path, pinned, monkeypatch):
    """data/<folder> is a link to the version directory, as get_zalmoxis.sh makes it."""
    rid = '22776069'
    rel = f'interior/eos/paleos_iron/r{rid}'
    root = tmp_path / 'fwl'
    (root / rel).mkdir(parents=True)
    (tmp_path / 'cache' / rel).mkdir(parents=True)
    (tmp_path / 'cache' / rel / NAME).write_text(pinned)
    (root / rel / NAME).write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    (tmp_path / 'data').mkdir()
    (tmp_path / 'data' / 'EOS_PALEOS_iron').symlink_to(root / rel)
    monkeypatch.setenv('FWL_DATA', str(root))
    monkeypatch.setenv('FWL_DATA_CACHE', str(tmp_path / 'cache'))
    monkeypatch.setattr(
        registry_guard, '_manifest', lambda: (_Dataset(KEY, {}, f'10.5281/zenodo.{rid}'),)
    )
    registry_guard.verify(str(tmp_path / 'data' / 'EOS_PALEOS_iron' / NAME))
    assert (root / rel / NAME).read_text() == pinned


def test_the_export_reader_keys_its_parse_by_the_fetched_file(tmp_path, pinned, monkeypatch):
    """After a refetch, the export reader parses under the identity of the new file."""
    from zalmoxis import eos_export

    rid = '22776069'
    rel = f'interior/eos/paleos_iron/r{rid}'
    root = tmp_path / 'fwl'
    (root / rel).mkdir(parents=True)
    (tmp_path / 'cache' / rel).mkdir(parents=True)
    (tmp_path / 'cache' / rel / NAME).write_text(pinned)
    path = root / rel / NAME
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    monkeypatch.setenv('FWL_DATA', str(root))
    monkeypatch.setenv('FWL_DATA_CACHE', str(tmp_path / 'cache'))
    monkeypatch.setattr(
        registry_guard, '_manifest', lambda: (_Dataset(KEY, {}, f'10.5281/zenodo.{rid}'),)
    )
    seen = []
    real = eos_export._parse_paleos_table
    monkeypatch.setattr(
        eos_export, '_parse_paleos_table', lambda p, fid: seen.append(fid) or real(p, fid)
    )
    eos_export.load_paleos_all_properties(str(path))
    st = os.stat(path)
    assert seen == [(st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)]


def test_the_hash_key_holds_the_size(tmp_path, pinned, monkeypatch):
    """With every other stat field held, a change of size alone hashes again."""
    path = tmp_path / NAME
    path.write_text(pinned)
    calls = []
    real_md5 = registry_guard._md5
    monkeypatch.setattr(registry_guard, '_md5', lambda p: calls.append(p) or real_md5(p))
    sizes = iter([100, 100, 101])  # then 101 for any later stat, a traceback's included

    class _Stat:
        st_dev = st_ino = st_mtime_ns = st_ctime_ns = 1

        @property
        def st_size(self):
            return self._size

    def fake_stat(p):
        st = _Stat()
        st._size = next(sizes, 101)
        return st

    class _Os:  # only the guard sees the fake stat; pytest keeps the real one
        stat = staticmethod(fake_stat)

        def __getattr__(self, name):
            return getattr(os, name)

    monkeypatch.setattr(registry_guard, 'os', _Os())
    for _ in range(3):
        registry_guard.file_md5(str(path))
    assert len(calls) == 2


def test_a_failed_cache_write_leaves_no_temporary_file(tmp_path, pinned, monkeypatch):
    """When the rename fails, the temporary file is removed and the table is still returned."""
    path = tmp_path / NAME
    path.write_text(pinned)

    def fail(a, b):
        raise OSError('disk full')

    monkeypatch.setattr(interpolation.os, 'replace', fail)
    assert interpolation._ensure_unified_cache(str(path), {})['p_min'] == pytest.approx(1e9)
    assert sorted(p.name for p in tmp_path.iterdir()) == [NAME]


def test_the_builder_identity_is_taken_at_import(monkeypatch):
    """Source read after import (an edited checkout under old code in memory) is not used."""
    interpolation.cache_builder.cache_clear()
    builder = interpolation.cache_builder()
    interpolation.cache_builder.cache_clear()
    monkeypatch.setattr(interpolation.inspect, 'getsource', lambda fn: 'def edited(): pass\n')
    try:
        assert interpolation.cache_builder() == builder
    finally:
        interpolation.cache_builder.cache_clear()


_SRC = "def f(x):\n    s = {'a', 'b', 'c'}\n    return [y * 2 for y in x if y in s]\n"


def _load(path, text, name):
    import importlib.util

    path.write_text(text)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.f


def test_the_same_loader_source_at_two_paths_has_one_identity(tmp_path):
    """The identity holds no file name or line number: the same source at 2 paths agrees."""
    identity = getattr(interpolation, '_code_identity', None)
    assert identity is not None
    (tmp_path / 'a').mkdir(), (tmp_path / 'b').mkdir()
    fa = _load(tmp_path / 'a' / 'm.py', _SRC, 'zal_m_a')
    fb = _load(tmp_path / 'b' / 'm.py', _SRC, 'zal_m_b')
    assert identity(fa) == identity(fb)


def test_without_source_the_identity_is_taken_from_the_code(tmp_path):
    """Without source the identity comes from co_code, co_consts and co_names: the same code at
    other paths and lines agrees, other code differs, and the value is the same in a process with
    another hash seed."""
    import subprocess
    import sys

    identity = getattr(interpolation, '_code_identity', None)
    assert identity is not None

    def compiled(text, filename):
        scope = {}
        exec(compile(text, filename, 'exec'), scope)
        return scope['f']

    one = identity(compiled(_SRC, '/nowhere/a.py'))
    assert one == identity(compiled('\n\n' + _SRC, '/elsewhere/b.py'))
    assert one != identity(compiled(_SRC.replace('* 2', '* 3'), '/nowhere/a.py'))
    probe = (
        'from zalmoxis.eos import interpolation as i\n'
        f'scope = {{}}; exec(compile({_SRC!r}, "/nowhere/a.py", "exec"), scope)\n'
        "print(i._code_identity(scope['f']))\n"
    )
    env = dict(os.environ, PYTHONHASHSEED='1')
    first = subprocess.run(
        [sys.executable, '-c', probe], capture_output=True, text=True, env=env
    )
    env['PYTHONHASHSEED'] = '2'
    second = subprocess.run(
        [sys.executable, '-c', probe], capture_output=True, text=True, env=env
    )
    assert first.stdout.strip() == second.stdout.strip() == one


def test_a_table_that_changes_during_the_parse_gets_no_cache(tmp_path, caplog):
    """When the table changes between the md5 and the end of the parse, no cache is written."""
    import logging

    path = tmp_path / 'own_table.dat'
    path.write_text(_table_text())
    real = interpolation.load_paleos_unified_table

    def parse_then_replace(eos_file):
        entry = real(eos_file)
        with open(eos_file, 'a') as f:
            f.write('# replaced during the parse\n')
        return entry

    with caplog.at_level(logging.WARNING, logger='zalmoxis.eos.interpolation'):
        interpolation.load_paleos_unified_table = parse_then_replace
        try:
            interpolation._ensure_unified_cache(str(path), {})
        finally:
            interpolation.load_paleos_unified_table = real
    assert not os.path.exists(interpolation.cache_path_for(str(path)))
    assert 'changed while it was read' in caplog.text


def test_a_stale_melting_curve_error_is_passed_on(monkeypatch):
    """load_melting_curve turns a read failure into None, but never a StaleTableError."""
    from zalmoxis import melting_curves
    from zalmoxis.eos import tdep

    def stale(path):
        raise StaleTableError('stale curve')

    monkeypatch.setattr(melting_curves, '_load_tabulated_curve', stale)
    with pytest.raises(StaleTableError, match='stale curve'):
        tdep.load_melting_curve('any.dat')


def test_read_table_columns_refuses_a_stale_table(tmp_path, pinned):
    """The public column reader checks a pinned table too."""
    path = tmp_path / NAME
    path.write_text(pinned.replace('8.0e+03', '8.1e+03', 1))
    with pytest.raises(StaleTableError):
        interpolation.read_table_columns(str(path), range(9))


def test_a_name_pinned_md5_and_sha256_names_the_sha256_pin(tmp_path, monkeypatch):
    """Two datasets that pin one file as md5 and as sha256 stop on the sha256 pin, not 'disagree'."""
    text = _table_text()
    manifest = (
        _Dataset(KEY, {NAME: 'md5:' + _digest(text, 'md5')}),
        _Dataset('interior.eos.paleos_iron_sha', {NAME: 'sha256:' + _digest(text, 'sha256')}),
    )
    monkeypatch.setattr(registry_guard, '_manifest', lambda: manifest)
    monkeypatch.setattr(registry_guard, '_pins', registry_guard._pins.__wrapped__)
    (tmp_path / NAME).write_text(text)
    with pytest.raises(StaleTableError, match='pinned as sha256:.*only md5'):
        registry_guard.verify(str(tmp_path / NAME))
