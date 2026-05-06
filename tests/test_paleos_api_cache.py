"""Tests for ``zalmoxis.eos.paleos_api_cache``: PALEOS-API table cache resolver.

Covers the cache-key + on-disk routing layer between live PALEOS tabulation
and downstream EOS readers. The actual generators (``generate_paleos_api_*``)
and the SHA probe (``paleos_installed_sha``) are mocked at module-import scope
so no PALEOS / multiprocessing work runs in the unit suite.

Anti-happy-path coverage: every class includes (a) a stale-SHA / mismatch-grid
edge case and (b) a physically unreasonable / error input (missing sub-dict,
mismatched grids, ``None`` registry entry, missing cache root).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from zalmoxis.eos import paleos_api_cache as pac
from zalmoxis.eos.paleos_api import GridSpec

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Discriminating values: asymmetric P / T axes, prime-ish node counts so any
# axis-swap or off-by-one in cache-key construction would change the hash.
_GRID_A = GridSpec(p_lo=1.3e5, p_hi=4.7e13, n_p=37, t_lo=311.0, t_hi=1.13e4, n_t=29)
_GRID_B = GridSpec(p_lo=1.3e5, p_hi=4.7e13, n_p=37, t_lo=311.0, t_hi=1.13e4, n_t=31)
_SHA_FAKE = 'a1b2c3d4e5f60718293a4b5c6d7e8f9012345678'  # 40-char synthetic
_SHA_OTHER = '0fedcba98765432100112233445566778899aabb'


def _fake_paleos_unified_content(sha: str, material: str = 'iron') -> str:
    """Return synthetic .dat content with a PALEOS_SHA header line.

    Ten-column rows are accepted unchanged by downstream readers. Values are
    physically plausible (positive T, P, rho) but distinct enough to not
    collide with default outputs.
    """
    return (
        f'# Synthetic test cache file ({material})\n'
        f'# PALEOS_SHA: {sha}\n'
        f'# grid_hash: deadbeefca\n'
        '1.300000e+05 3.110000e+02 4173.5 5.83e3 1142.7 '
        '1183.0 1093.0 2.71e-5 0.293 solid\n'
    )


@pytest.fixture
def patched_cache_root(monkeypatch, tmp_path):
    """Redirect ``_cache_root`` to a fresh per-test scratch directory."""
    cache_dir = tmp_path / 'paleos_api_cache_root'
    cache_dir.mkdir()
    monkeypatch.setattr('zalmoxis.eos.paleos_api_cache._cache_root', lambda: cache_dir)
    return cache_dir


@pytest.fixture
def patched_sha(monkeypatch):
    """Replace ``paleos_installed_sha`` with a deterministic SHA."""
    monkeypatch.setattr(
        'zalmoxis.eos.paleos_api_cache.paleos_installed_sha',
        lambda: _SHA_FAKE,
    )
    return _SHA_FAKE


@pytest.fixture
def patched_generators(monkeypatch):
    """Replace the unified + 2-phase generators with file-writing fakes.

    Returns a dict-of-counters so tests can assert the number of times each
    generator was invoked. The fakes write the exact PALEOS_SHA header line
    that ``_header_sha_matches`` expects, so a second resolve_* call on the
    same path returns a cache hit.
    """
    counters = {'unified': 0, 'twophase': 0, 'last_unified_kwargs': None}

    def fake_unified(material, out_path, grid, h2o_table_path=None, n_workers=-1):
        counters['unified'] += 1
        counters['last_unified_kwargs'] = {
            'material': material,
            'out_path': Path(out_path),
            'grid': grid,
            'h2o_table_path': h2o_table_path,
            'n_workers': n_workers,
        }
        Path(out_path).write_text(_fake_paleos_unified_content(_SHA_FAKE, material))

    def fake_twophase(out_solid, out_liquid, grid, n_workers=-1):
        counters['twophase'] += 1
        Path(out_solid).write_text(_fake_paleos_unified_content(_SHA_FAKE, 'mgsio3-solid'))
        Path(out_liquid).write_text(_fake_paleos_unified_content(_SHA_FAKE, 'mgsio3-liquid'))

    monkeypatch.setattr(
        'zalmoxis.eos.paleos_api_cache.generate_paleos_api_unified_table',
        fake_unified,
    )
    monkeypatch.setattr(
        'zalmoxis.eos.paleos_api_cache.generate_paleos_api_2phase_mgsio3_tables',
        fake_twophase,
    )
    return counters


# ---------------------------------------------------------------------------
# _cache_root
# ---------------------------------------------------------------------------


class TestCacheRoot:
    """The default cache root resolves under ``$ZALMOXIS_ROOT/data/EOS_PALEOS_API/``."""

    def test_default_layout_under_zalmoxis_root(self, monkeypatch, tmp_path):
        """``_cache_root()`` returns ``<root>/data/EOS_PALEOS_API``."""
        fake_root = tmp_path / 'fake_zalmoxis_repo'
        fake_root.mkdir()
        monkeypatch.setattr('zalmoxis.get_zalmoxis_root', lambda: str(fake_root))
        result = pac._cache_root()
        assert result == fake_root / 'data' / 'EOS_PALEOS_API'

    def test_returns_path_instance_not_str(self, monkeypatch, tmp_path):
        """Edge case: caller relies on ``Path`` API (``/``, ``.exists()``)."""
        monkeypatch.setattr('zalmoxis.get_zalmoxis_root', lambda: str(tmp_path))
        result = pac._cache_root()
        assert isinstance(result, Path)

    def test_nonexistent_root_does_not_raise(self, monkeypatch, tmp_path):
        """Physically unreasonable input: missing repo root does not error here.

        Resolution is path-construction only; existence checks happen later in
        the flow. This locks in that contract.
        """
        bogus = tmp_path / 'does_not_exist'
        monkeypatch.setattr('zalmoxis.get_zalmoxis_root', lambda: str(bogus))
        result = pac._cache_root()
        assert result.parts[-2:] == ('data', 'EOS_PALEOS_API')


# ---------------------------------------------------------------------------
# _sha_short
# ---------------------------------------------------------------------------


class TestShaShort:
    """``_sha_short`` truncates to 10 chars; passes through shorter strings."""

    def test_truncates_full_sha_to_ten_chars(self):
        """40-char SHA-1 truncates to its 10-char prefix."""
        result = pac._sha_short(_SHA_FAKE)
        assert result == 'a1b2c3d4e5'
        assert len(result) == 10

    def test_passthrough_when_shorter_than_ten(self):
        """Short identifiers (e.g. ``'unknown'``) are returned unchanged."""
        assert pac._sha_short('unknown') == 'unknown'
        assert pac._sha_short('paleos-not-installed') == 'paleos-not'

    def test_empty_string_edge_case(self):
        """Edge case: empty SHA returns empty string (no crash)."""
        assert pac._sha_short('') == ''

    def test_exactly_ten_chars_passthrough(self):
        """Boundary case: a 10-char input should be returned unchanged."""
        result = pac._sha_short('0123456789')
        assert result == '0123456789'
        assert len(result) == 10


# ---------------------------------------------------------------------------
# _header_sha_matches
# ---------------------------------------------------------------------------


class TestHeaderShaMatches:
    """Header-SHA validation gates whether a cached file is reused."""

    def test_missing_file_returns_false(self, tmp_path):
        """Edge case: file does not exist."""
        missing = tmp_path / 'no_such_file.dat'
        assert pac._header_sha_matches(missing, _SHA_FAKE) is False

    def test_matching_sha_returns_true(self, tmp_path):
        """Header carries the expected SHA → return True."""
        f = tmp_path / 'cached.dat'
        f.write_text(_fake_paleos_unified_content(_SHA_FAKE))
        assert pac._header_sha_matches(f, _SHA_FAKE) is True

    def test_mismatched_sha_returns_false(self, tmp_path):
        """Header carries a different SHA → False (triggers regeneration)."""
        f = tmp_path / 'stale.dat'
        f.write_text(_fake_paleos_unified_content(_SHA_OTHER))
        assert pac._header_sha_matches(f, _SHA_FAKE) is False

    def test_no_paleos_sha_line_returns_false(self, tmp_path):
        """Legacy / hand-placed file with no ``PALEOS_SHA:`` line → False."""
        f = tmp_path / 'legacy.dat'
        f.write_text(
            '# Hand-placed test file\n'
            '# grid_hash: cafebabe01\n'
            '1e6 1000 4000 1e9 1000 1200 1100 1e-5 0.3 solid\n'
        )
        assert pac._header_sha_matches(f, _SHA_FAKE) is False

    def test_no_comment_lines_returns_false(self, tmp_path):
        """Physically unreasonable input: data file with no header at all."""
        f = tmp_path / 'no_header.dat'
        f.write_text('1e6 1000 4000 1e9 1000 1200 1100 1e-5 0.3 solid\n')
        assert pac._header_sha_matches(f, _SHA_FAKE) is False

    def test_directory_path_returns_false(self, tmp_path):
        """OSError branch: passing a directory triggers IsADirectoryError → False.

        ``open()`` on a directory raises ``IsADirectoryError`` which is a
        subclass of ``OSError``; the function catches and returns False.
        """
        d = tmp_path / 'a_directory.dat'
        d.mkdir()
        # ``Path.exists`` is True for directories; the OSError lives in open().
        assert pac._header_sha_matches(d, _SHA_FAKE) is False

    def test_empty_file_returns_false(self, tmp_path):
        """Edge case: empty file (zero iterations of the header loop) → False."""
        f = tmp_path / 'empty.dat'
        f.write_text('')
        assert pac._header_sha_matches(f, _SHA_FAKE) is False


# ---------------------------------------------------------------------------
# resolve_paleos_api_unified
# ---------------------------------------------------------------------------


class TestResolveUnified:
    """``resolve_paleos_api_unified`` enforces (SHA, grid_hash) cache keying."""

    def test_cache_miss_invokes_generator(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """Cache miss path: generator runs once and target file lands on disk."""
        path = pac.resolve_paleos_api_unified('iron', _GRID_A)
        assert path.exists()
        assert patched_generators['unified'] == 1
        # File name carries sha10 + grid_hash; check the structure.
        sha10 = _SHA_FAKE[:10]
        ghash = _GRID_A.hash_short()
        assert path.name == f'unified_iron_{sha10}_{ghash}.dat'
        assert path.parent == patched_cache_root

    def test_cache_hit_skips_generator(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """Second resolve on a fresh-header file does NOT call the generator."""
        first = pac.resolve_paleos_api_unified('iron', _GRID_A)
        assert patched_generators['unified'] == 1
        second = pac.resolve_paleos_api_unified('iron', _GRID_A)
        assert second == first
        assert patched_generators['unified'] == 1  # still 1, no extra call

    def test_stale_sha_triggers_regeneration(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """Edge case: existing file with wrong SHA header → regenerate."""
        # Pre-place a file at the expected path with a *different* SHA.
        sha10 = _SHA_FAKE[:10]
        ghash = _GRID_A.hash_short()
        stale = patched_cache_root / f'unified_mgsio3_{sha10}_{ghash}.dat'
        stale.write_text(_fake_paleos_unified_content(_SHA_OTHER))
        path = pac.resolve_paleos_api_unified('mgsio3', _GRID_A)
        assert path == stale
        assert patched_generators['unified'] == 1

    def test_force_regenerates_even_on_fresh_hit(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """``force=True`` runs the generator even though the file is current."""
        pac.resolve_paleos_api_unified('iron', _GRID_A)
        assert patched_generators['unified'] == 1
        pac.resolve_paleos_api_unified('iron', _GRID_A, force=True)
        assert patched_generators['unified'] == 2

    def test_h2o_table_path_forwarded_to_generator(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """h2o_table_path kwarg must reach the generator (not silently dropped)."""
        aqua = '/some/synthetic/aqua/path.dat'
        pac.resolve_paleos_api_unified('h2o', _GRID_A, h2o_table_path=aqua)
        assert patched_generators['last_unified_kwargs']['h2o_table_path'] == aqua

    def test_different_grids_get_different_filenames(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """Discriminating: distinct ``GridSpec`` → distinct cache filename."""
        p1 = pac.resolve_paleos_api_unified('iron', _GRID_A)
        p2 = pac.resolve_paleos_api_unified('iron', _GRID_B)
        assert p1 != p2
        assert patched_generators['unified'] == 2

    def test_n_workers_forwarded(self, patched_cache_root, patched_sha, patched_generators):
        """Physically unreasonable input: n_workers=0 must still be forwarded.

        The cache layer does not validate ``n_workers``; that's the
        generator's job. We lock in that the value passes through unchanged.
        """
        pac.resolve_paleos_api_unified('iron', _GRID_A, n_workers=0)
        assert patched_generators['last_unified_kwargs']['n_workers'] == 0


# ---------------------------------------------------------------------------
# resolve_paleos_api_2phase_mgsio3
# ---------------------------------------------------------------------------


class TestResolveTwoPhase:
    """2-phase resolver always emits a (solid, liquid) pair."""

    def test_pair_returned_on_cold_cache(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """Both files exist after one call; generator invoked once."""
        solid, liquid = pac.resolve_paleos_api_2phase_mgsio3(_GRID_A)
        assert solid.exists() and liquid.exists()
        assert solid != liquid
        assert solid.name.startswith('2phase_solid_')
        assert liquid.name.startswith('2phase_liquid_')
        assert patched_generators['twophase'] == 1

    def test_warm_cache_skips_generator(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """Cache hit: matching headers on both → no regeneration."""
        pac.resolve_paleos_api_2phase_mgsio3(_GRID_A)
        pac.resolve_paleos_api_2phase_mgsio3(_GRID_A)
        assert patched_generators['twophase'] == 1

    def test_one_missing_triggers_full_pair_regeneration(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """Atomic-pair semantics: deleting only the liquid still rebuilds both."""
        solid, liquid = pac.resolve_paleos_api_2phase_mgsio3(_GRID_A)
        assert patched_generators['twophase'] == 1
        os.remove(liquid)
        solid2, liquid2 = pac.resolve_paleos_api_2phase_mgsio3(_GRID_A)
        assert solid2 == solid and liquid2 == liquid
        assert patched_generators['twophase'] == 2

    def test_stale_solid_sha_triggers_pair_regeneration(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """Edge case: solid header SHA wrong → both regenerated."""
        sha10 = _SHA_FAKE[:10]
        ghash = _GRID_A.hash_short()
        solid = patched_cache_root / f'2phase_solid_{sha10}_{ghash}.dat'
        liquid = patched_cache_root / f'2phase_liquid_{sha10}_{ghash}.dat'
        # Place a stale solid + a fresh liquid: still regenerates both.
        solid.write_text(_fake_paleos_unified_content(_SHA_OTHER))
        liquid.write_text(_fake_paleos_unified_content(_SHA_FAKE))
        pac.resolve_paleos_api_2phase_mgsio3(_GRID_A)
        assert patched_generators['twophase'] == 1

    def test_force_regenerates_even_on_fresh_hit(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """``force=True`` bypasses the SHA check."""
        pac.resolve_paleos_api_2phase_mgsio3(_GRID_A)
        pac.resolve_paleos_api_2phase_mgsio3(_GRID_A, force=True)
        assert patched_generators['twophase'] == 2


# ---------------------------------------------------------------------------
# resolve_registry_entry + helpers
# ---------------------------------------------------------------------------


class TestResolveRegistryEntry:
    """Top-level registry entry resolver dispatches by ``format`` flag."""

    def test_unified_entry_mutated_in_place(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """``format='paleos_api'`` → eos_file populated, format flipped."""
        entry = {
            'format': 'paleos_api',
            'material': 'iron',
            'grid_spec': _GRID_A,
        }
        result = pac.resolve_registry_entry(entry)
        assert result is entry  # in-place mutation contract
        assert entry['format'] == 'paleos_unified'
        assert 'eos_file' in entry
        assert Path(entry['eos_file']).exists()
        assert patched_generators['unified'] == 1

    def test_two_phase_entry_mutates_sub_dicts(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """Sub-dict ``format='paleos_api_2phase'`` → both sub-dicts updated."""
        outer = {
            'solid_mantle': {'format': 'paleos_api_2phase', 'grid_spec': _GRID_A},
            'melted_mantle': {'format': 'paleos_api_2phase', 'grid_spec': _GRID_A},
        }
        pac.resolve_registry_entry(outer)
        assert outer['solid_mantle']['format'] == 'paleos'
        assert outer['melted_mantle']['format'] == 'paleos'
        assert Path(outer['solid_mantle']['eos_file']).exists()
        assert Path(outer['melted_mantle']['eos_file']).exists()
        assert patched_generators['twophase'] == 1

    def test_none_input_returns_none(self):
        """Physically unreasonable input: ``None`` is passed through."""
        assert pac.resolve_registry_entry(None) is None

    def test_unknown_format_passes_through_unchanged(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """Edge case: dict with unrelated format must not be touched."""
        entry = {'format': 'paleos', 'eos_file': '/already/resolved.dat'}
        original = dict(entry)
        result = pac.resolve_registry_entry(entry)
        assert result is entry
        assert entry == original
        assert patched_generators['unified'] == 0
        assert patched_generators['twophase'] == 0

    def test_dict_with_no_format_at_all(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """Edge case: dict has neither top-level format nor 2-phase sub-dicts."""
        entry = {'material': 'iron', 'note': 'pre-resolved'}
        result = pac.resolve_registry_entry(entry)
        assert result is entry
        assert entry == {'material': 'iron', 'note': 'pre-resolved'}
        assert patched_generators['unified'] == 0


# ---------------------------------------------------------------------------
# _resolve_2phase_in_place
# ---------------------------------------------------------------------------


class TestResolveTwoPhaseInPlace:
    """2-phase helper enforces both halves present and grid-spec consistency."""

    def test_missing_solid_raises(self, patched_cache_root, patched_sha, patched_generators):
        """Physically unreasonable: only one phase present → ValueError."""
        outer = {'melted_mantle': {'format': 'paleos_api_2phase', 'grid_spec': _GRID_A}}
        with pytest.raises(ValueError, match='solid_mantle and melted_mantle'):
            pac._resolve_2phase_in_place(outer)
        assert patched_generators['twophase'] == 0

    def test_missing_liquid_raises(self, patched_cache_root, patched_sha, patched_generators):
        """Physically unreasonable: solid only → ValueError."""
        outer = {'solid_mantle': {'format': 'paleos_api_2phase', 'grid_spec': _GRID_A}}
        with pytest.raises(ValueError, match='solid_mantle and melted_mantle'):
            pac._resolve_2phase_in_place(outer)

    def test_mismatched_grid_specs_raise(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """Edge case: solid + liquid carry different grids → ValueError.

        Locks in that the cache resolver refuses to silently mix grids
        between the two phases of the mushy-zone mixing pair.
        """
        outer = {
            'solid_mantle': {'format': 'paleos_api_2phase', 'grid_spec': _GRID_A},
            'melted_mantle': {'format': 'paleos_api_2phase', 'grid_spec': _GRID_B},
        }
        with pytest.raises(ValueError, match='share a GridSpec'):
            pac._resolve_2phase_in_place(outer)
        assert patched_generators['twophase'] == 0

    def test_matching_grids_populate_both_sides(
        self, patched_cache_root, patched_sha, patched_generators
    ):
        """Happy path with discriminating values: both eos_file fields land."""
        outer = {
            'solid_mantle': {'format': 'paleos_api_2phase', 'grid_spec': _GRID_A},
            'melted_mantle': {'format': 'paleos_api_2phase', 'grid_spec': _GRID_A},
        }
        pac._resolve_2phase_in_place(outer)
        assert outer['solid_mantle']['eos_file'].endswith('.dat')
        assert outer['melted_mantle']['eos_file'].endswith('.dat')
        assert outer['solid_mantle']['eos_file'] != outer['melted_mantle']['eos_file']
        assert outer['solid_mantle']['format'] == 'paleos'
        assert outer['melted_mantle']['format'] == 'paleos'


# ---------------------------------------------------------------------------
# invalidate_cache
# ---------------------------------------------------------------------------


class TestInvalidateCache:
    """``invalidate_cache`` removes ``.dat`` files filtered by material."""

    def test_missing_root_returns_zero(self, monkeypatch, tmp_path):
        """Edge case: cache root does not exist → 0 removed (no crash)."""
        bogus = tmp_path / 'nope'
        monkeypatch.setattr('zalmoxis.eos.paleos_api_cache._cache_root', lambda: bogus)
        assert pac.invalidate_cache() == 0
        assert pac.invalidate_cache('iron') == 0

    def test_remove_all_when_material_is_none(self, patched_cache_root):
        """Default material=None removes every ``.dat`` in the root."""
        files = [
            'unified_iron_aaaaaaaaaa_bbbbbbbbbb.dat',
            'unified_mgsio3_aaaaaaaaaa_bbbbbbbbbb.dat',
            '2phase_solid_aaaaaaaaaa_bbbbbbbbbb.dat',
            '2phase_liquid_aaaaaaaaaa_bbbbbbbbbb.dat',
        ]
        for n in files:
            (patched_cache_root / n).write_text('# stub\n')
        # Add a non-.dat file: must be ignored.
        (patched_cache_root / 'README.txt').write_text('not a cache file\n')
        # And a sub-directory: must also be ignored (not a file).
        (patched_cache_root / 'subdir').mkdir()
        n_removed = pac.invalidate_cache()
        assert n_removed == 4
        assert (patched_cache_root / 'README.txt').exists()
        assert (patched_cache_root / 'subdir').exists()
        for n in files:
            assert not (patched_cache_root / n).exists()

    def test_filter_by_iron_only_removes_iron(self, patched_cache_root):
        """Discriminating: ``material='iron'`` leaves mgsio3 / 2phase alone."""
        kept = [
            'unified_mgsio3_aaaaaaaaaa_bbbbbbbbbb.dat',
            '2phase_solid_aaaaaaaaaa_bbbbbbbbbb.dat',
            '2phase_liquid_aaaaaaaaaa_bbbbbbbbbb.dat',
        ]
        removed = ['unified_iron_aaaaaaaaaa_bbbbbbbbbb.dat']
        for n in kept + removed:
            (patched_cache_root / n).write_text('# stub\n')
        n = pac.invalidate_cache('iron')
        assert n == 1
        for k in kept:
            assert (patched_cache_root / k).exists()
        for r in removed:
            assert not (patched_cache_root / r).exists()

    def test_filter_by_mgsio3_includes_2phase_files(self, patched_cache_root):
        """``material='mgsio3'`` matches both ``unified_mgsio3_*`` and ``2phase_*``.

        This locks in the special-case branch in
        ``invalidate_cache``: the 2-phase files have no ``mgsio3`` token in
        their filename but they belong to the mgsio3 cache logically.
        """
        files_removed = [
            'unified_mgsio3_aaaaaaaaaa_bbbbbbbbbb.dat',
            '2phase_solid_aaaaaaaaaa_bbbbbbbbbb.dat',
            '2phase_liquid_aaaaaaaaaa_bbbbbbbbbb.dat',
        ]
        files_kept = [
            'unified_iron_aaaaaaaaaa_bbbbbbbbbb.dat',
            'unified_h2o_aaaaaaaaaa_bbbbbbbbbb.dat',
        ]
        for n in files_removed + files_kept:
            (patched_cache_root / n).write_text('# stub\n')
        n = pac.invalidate_cache('mgsio3')
        assert n == 3
        for f in files_kept:
            assert (patched_cache_root / f).exists()
        for f in files_removed:
            assert not (patched_cache_root / f).exists()

    def test_unknown_material_removes_nothing(self, patched_cache_root):
        """Physically unreasonable input: an unknown material is a no-op."""
        names = [
            'unified_iron_aaaaaaaaaa_bbbbbbbbbb.dat',
            'unified_mgsio3_aaaaaaaaaa_bbbbbbbbbb.dat',
            '2phase_solid_aaaaaaaaaa_bbbbbbbbbb.dat',
        ]
        for n in names:
            (patched_cache_root / n).write_text('# stub\n')
        assert pac.invalidate_cache('plutonium') == 0
        for n in names:
            assert (patched_cache_root / n).exists()

    def test_oserror_on_remove_does_not_propagate(self, patched_cache_root, monkeypatch):
        """OSError branch: ``os.remove`` failures are logged + counted out.

        Simulate by patching ``os.remove`` inside the module to raise on the
        second call. The function must keep going and report only the
        successful removal in its returned count.
        """
        names = [
            'unified_iron_aaaaaaaaaa_bbbbbbbbbb.dat',
            'unified_iron_cccccccccc_dddddddddd.dat',
        ]
        for n in names:
            (patched_cache_root / n).write_text('# stub\n')

        real_remove = os.remove
        call_count = {'n': 0}

        def flaky_remove(path):
            call_count['n'] += 1
            if call_count['n'] == 2:
                raise PermissionError('synthetic OSError for test')
            real_remove(path)

        monkeypatch.setattr('zalmoxis.eos.paleos_api_cache.os.remove', flaky_remove)
        n_removed = pac.invalidate_cache('iron')
        # One succeeded, one raised → count is 1.
        assert n_removed == 1
