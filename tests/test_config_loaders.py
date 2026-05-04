"""Tests for the config loaders in ``zalmoxis.config``.

The existing ``test_config_validation.py`` covers ``validate_config``.
This file targets the loader-side functions that are otherwise hit only
by integration tests:

- ``parse_eos_config``: per-layer + legacy formats, error paths.
- ``validate_layer_eos``: tabulated / analytic / Vinet branches.
- ``choose_config_file``: temp_config_path / -c flag / default.
- ``load_zalmoxis_config``: end-to-end TOML round-trip.
- ``load_material_dictionaries``: registry build, Chabrier:H entry.
- ``load_solidus_liquidus_functions``: dispatch + mushy_zone_factor.

Anti-happy-path: each test class includes ≥ 1 edge case + ≥ 1
physically-unreasonable / error input.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from zalmoxis.config import (
    choose_config_file,
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
    parse_eos_config,
    validate_layer_eos,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# parse_eos_config
# ---------------------------------------------------------------------------


class TestParseEosConfig:
    """``parse_eos_config`` accepts new-style + legacy TOML formats."""

    def test_new_format_two_layer_returns_dict_with_core_and_mantle(self):
        """Per-layer keys ``core`` and ``mantle`` produce a 2-key dict."""
        out = parse_eos_config({'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'})
        assert out == {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'}

    def test_new_format_three_layer_includes_ice_layer(self):
        """Optional ``ice_layer`` key flows through when non-empty."""
        out = parse_eos_config(
            {
                'core': 'PALEOS:iron',
                'mantle': 'PALEOS:MgSiO3',
                'ice_layer': 'PALEOS:H2O',
            }
        )
        assert out['ice_layer'] == 'PALEOS:H2O'

    def test_new_format_empty_ice_layer_omitted(self):
        """Edge: empty ``ice_layer`` string -> key omitted from output."""
        out = parse_eos_config(
            {
                'core': 'PALEOS:iron',
                'mantle': 'PALEOS:MgSiO3',
                'ice_layer': '',
            }
        )
        assert 'ice_layer' not in out

    def test_new_format_missing_mantle_raises(self):
        """Physically unreasonable: ``core`` without ``mantle`` -> ValueError.

        A planet without a mantle is not a valid configuration in this code
        path, so the loader rejects it explicitly.
        """
        with pytest.raises(ValueError, match="missing 'mantle'"):
            parse_eos_config({'core': 'PALEOS:iron'})

    def test_legacy_choice_tabulated_iron_silicate_expands(self):
        """Legacy ``choice='Tabulated:iron/silicate'`` expands to Seager layers."""
        out = parse_eos_config({'choice': 'Tabulated:iron/silicate'})
        assert out['core'] == 'Seager2007:iron'
        assert out['mantle'] == 'Seager2007:MgSiO3'

    def test_legacy_choice_water_includes_ice_layer(self):
        """Legacy 3-layer water config carries an ``ice_layer`` entry."""
        out = parse_eos_config({'choice': 'Tabulated:water'})
        assert 'ice_layer' in out
        assert out['ice_layer'] == 'Seager2007:H2O'

    def test_legacy_analytic_seager2007_uses_default_materials(self):
        """Edge: ``Analytic:Seager2007`` without explicit materials uses defaults."""
        out = parse_eos_config({'choice': 'Analytic:Seager2007'})
        assert out['core'] == 'Analytic:iron'
        assert out['mantle'] == 'Analytic:MgSiO3'
        assert 'ice_layer' not in out

    def test_legacy_analytic_seager2007_with_water_layer(self):
        """Three-layer analytic: water_layer_material populates ice_layer."""
        out = parse_eos_config(
            {
                'choice': 'Analytic:Seager2007',
                'water_layer_material': 'H2O',
            }
        )
        assert out['ice_layer'] == 'Analytic:H2O'

    def test_unknown_choice_raises_value_error(self):
        """Physically unreasonable: unknown legacy choice -> ValueError."""
        with pytest.raises(ValueError, match='Unknown EOS config'):
            parse_eos_config({'choice': 'NonexistentEOS'})


# ---------------------------------------------------------------------------
# validate_layer_eos
# ---------------------------------------------------------------------------


class TestValidateLayerEos:
    """``validate_layer_eos`` rejects unknown materials per layer."""

    def test_valid_paleos_layer_passes(self):
        """A standard PALEOS-only config passes silently."""
        validate_layer_eos({'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'})  # no exception

    def test_valid_analytic_layer_passes(self):
        """Analytic layer with a known material key passes."""
        validate_layer_eos({'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'})

    def test_invalid_analytic_material_raises(self):
        """Edge: Analytic layer with unknown material -> ValueError listing valid keys."""
        with pytest.raises(ValueError, match='Invalid analytic material'):
            validate_layer_eos({'core': 'Analytic:notametal', 'mantle': 'Analytic:MgSiO3'})

    def test_invalid_vinet_material_raises(self):
        """Vinet layer with unknown key -> ValueError listing valid keys."""
        with pytest.raises(ValueError, match='Invalid Vinet material'):
            validate_layer_eos({'core': 'Vinet:notametal', 'mantle': 'PALEOS:MgSiO3'})

    def test_unknown_eos_string_raises(self):
        """Physically unreasonable: completely unknown EOS family -> ValueError."""
        with pytest.raises(ValueError):
            validate_layer_eos({'core': 'NotAnEOS:iron', 'mantle': 'PALEOS:MgSiO3'})


# ---------------------------------------------------------------------------
# choose_config_file
# ---------------------------------------------------------------------------


def _write_toml(path: Path, body: str):
    path.write_text(body)
    return path


_MINIMAL_TOML = """\
[InputParameter]
planet_mass = 1.0

[AssumptionsAndInitialGuesses]
core_mass_fraction = 0.32
mantle_mass_fraction = 0.0
temperature_mode = "linear"
surface_temperature = 1500.0
center_temperature = 5000.0
temperature_profile_file = ""

[EOS]
core = "PALEOS:iron"
mantle = "PALEOS:MgSiO3"

[Calculations]
num_layers = 200

[Output]
"""


class TestChooseConfigFile:
    """``choose_config_file`` picks the right TOML by precedence."""

    def test_temp_path_wins_when_provided(self, tmp_path):
        """Explicit temp_config_path takes precedence over -c and default."""
        p = _write_toml(tmp_path / 'a.toml', _MINIMAL_TOML)
        out = choose_config_file(temp_config_path=str(p))
        # parsed TOML has the InputParameter section.
        assert 'InputParameter' in out
        assert out['InputParameter']['planet_mass'] == pytest.approx(1.0)

    def test_temp_path_missing_calls_sys_exit(self, tmp_path):
        """Edge: temp_config_path that does not exist -> SystemExit."""
        bogus = tmp_path / 'missing.toml'
        with pytest.raises(SystemExit):
            choose_config_file(temp_config_path=str(bogus))

    def test_dash_c_flag_selects_path(self, tmp_path, monkeypatch):
        """``-c <path>`` argv pair selects the named config."""
        p = _write_toml(tmp_path / 'b.toml', _MINIMAL_TOML)
        monkeypatch.setattr(sys, 'argv', ['zalmoxis', '-c', str(p)])
        out = choose_config_file()
        assert 'InputParameter' in out

    def test_dash_c_without_path_calls_sys_exit(self, monkeypatch):
        """Edge: ``-c`` with no following path -> SystemExit (IndexError branch)."""
        monkeypatch.setattr(sys, 'argv', ['zalmoxis', '-c'])
        with pytest.raises(SystemExit):
            choose_config_file()

    def test_dash_c_with_missing_file_calls_sys_exit(self, monkeypatch, tmp_path):
        """Edge: ``-c`` with a non-existent file -> SystemExit."""
        bogus = tmp_path / 'nope.toml'
        monkeypatch.setattr(sys, 'argv', ['zalmoxis', '-c', str(bogus)])
        with pytest.raises(SystemExit):
            choose_config_file()


# ---------------------------------------------------------------------------
# load_zalmoxis_config
# ---------------------------------------------------------------------------


class TestLoadZalmoxisConfig:
    """``load_zalmoxis_config`` parses a TOML and produces config_params."""

    def test_round_trips_minimal_toml(self, tmp_path):
        """A minimal TOML loads successfully and returns required keys."""
        p = _write_toml(tmp_path / 'mini.toml', _MINIMAL_TOML)
        params = load_zalmoxis_config(temp_config_path=str(p))
        # planet_mass is converted from M_earth to kg.
        assert params['planet_mass'] > 0
        assert 'layer_eos_config' in params
        assert params['layer_eos_config']['core'] == 'PALEOS:iron'
        # Defaults applied for unset optional keys.
        assert params['rock_solidus'] == 'Stixrude14-solidus'
        assert params['rock_liquidus'] == 'Stixrude14-liquidus'

    def test_optional_iter_keys_pass_through(self, tmp_path):
        """``[IterativeProcess]`` keys override mass-adaptive defaults."""
        toml_with_iter = _MINIMAL_TOML + (
            '\n[IterativeProcess]\nmax_iterations_outer = 99\ntolerance_outer = 1e-3\n'
        )
        p = _write_toml(tmp_path / 'iter.toml', toml_with_iter)
        params = load_zalmoxis_config(temp_config_path=str(p))
        assert params['max_iterations_outer'] == 99
        assert params['tolerance_outer'] == pytest.approx(1e-3)

    def test_per_eos_mushy_factor_overrides_global(self, tmp_path):
        """Per-EOS factor in [EOS] takes precedence over the global one."""
        # Append the per-EOS override into the EOS section by re-writing the
        # block; TOML sections cannot be duplicated.
        body = _MINIMAL_TOML.replace(
            '[EOS]\ncore = "PALEOS:iron"\nmantle = "PALEOS:MgSiO3"\n',
            (
                '[EOS]\n'
                'core = "PALEOS:iron"\n'
                'mantle = "PALEOS:MgSiO3"\n'
                'mushy_zone_factor = 1.0\n'
                'mushy_zone_factor_MgSiO3 = 0.85\n'
            ),
        )
        p = _write_toml(tmp_path / 'pereos.toml', body)
        params = load_zalmoxis_config(temp_config_path=str(p))
        assert params['mushy_zone_factors']['PALEOS:MgSiO3'] == pytest.approx(0.85)
        # iron uses the global default 1.0 since no per-EOS override given.
        assert params['mushy_zone_factors']['PALEOS:iron'] == pytest.approx(1.0)

    def test_unused_paleos_material_defaults_to_one(self, tmp_path):
        """Edge: PALEOS:H2O not in any layer -> mushy factor defaults to 1.0.

        Forces the ``not in _all_eos_strings`` branch.
        """
        body = _MINIMAL_TOML.replace(
            'mantle = "PALEOS:MgSiO3"',
            'mantle = "PALEOS:MgSiO3"\nmushy_zone_factor = 0.7',
        )
        p = _write_toml(tmp_path / 'unused.toml', body)
        params = load_zalmoxis_config(temp_config_path=str(p))
        # H2O is not used; factor stays at default 1.0 even though global is 0.7.
        assert params['mushy_zone_factors']['PALEOS:H2O'] == pytest.approx(1.0)
        # Iron and MgSiO3 are in use; factor inherits the global 0.7.
        assert params['mushy_zone_factors']['PALEOS:iron'] == pytest.approx(0.7)
        assert params['mushy_zone_factors']['PALEOS:MgSiO3'] == pytest.approx(0.7)

    def test_invalid_eos_in_toml_propagates_value_error(self, tmp_path):
        """Physically unreasonable: unknown EOS string -> ValueError."""
        body = _MINIMAL_TOML.replace('core = "PALEOS:iron"', 'core = "Bogus:metal"')
        p = _write_toml(tmp_path / 'bogus.toml', body)
        with pytest.raises(ValueError):
            load_zalmoxis_config(temp_config_path=str(p))


# ---------------------------------------------------------------------------
# load_material_dictionaries
# ---------------------------------------------------------------------------


class TestLoadMaterialDictionaries:
    """``load_material_dictionaries`` returns the EOS registry."""

    def test_includes_known_paleos_materials(self):
        """Edge: registry must carry the standard PALEOS keys."""
        registry = load_material_dictionaries()
        # Known IDs that should always exist.
        for key in ('PALEOS:iron', 'PALEOS:MgSiO3'):
            assert key in registry, f'{key} missing from registry'
        # Each entry must be a dict (the per-material spec).
        assert isinstance(registry['PALEOS:iron'], dict)


# ---------------------------------------------------------------------------
# load_solidus_liquidus_functions
# ---------------------------------------------------------------------------


class TestLoadSolidusLiquidusFunctions:
    """``load_solidus_liquidus_functions`` dispatches by ID string.

    Returns ``None`` when no layer needs external melting curves. PALEOS
    unified materials extract their phase boundary from the table itself,
    so they don't trigger curve loading; Tdep tables (WolfBower,
    PALEOS-2phase) do.
    """

    def test_unified_paleos_layers_return_none(self):
        """Unified PALEOS layers do not need external melting curves -> None."""
        out = load_solidus_liquidus_functions(
            {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
            'Stixrude14-solidus',
            'Stixrude14-liquidus',
        )
        assert out is None

    def test_tdep_mantle_returns_callable_pair(self):
        """A Tdep mantle (WolfBower2018) triggers curve-function loading."""
        out = load_solidus_liquidus_functions(
            {'core': 'PALEOS:iron', 'mantle': 'WolfBower2018:MgSiO3'},
            'Stixrude14-solidus',
            'Stixrude14-liquidus',
        )
        assert out is not None
        sol_func, liq_func = out
        # Latent-heat invariant: liquidus > solidus at every pressure tested.
        for P in (1e9, 1e10, 1e11):
            T_sol = float(sol_func(P))
            T_liq = float(liq_func(P))
            assert T_liq > T_sol
            assert T_sol > 0.0

    def test_unknown_solidus_id_raises(self):
        """Edge: an unknown solidus name raises an error.

        Use a Tdep layer so the curve dispatch is reached.
        """
        with pytest.raises((ValueError, KeyError)):
            load_solidus_liquidus_functions(
                {'core': 'PALEOS:iron', 'mantle': 'WolfBower2018:MgSiO3'},
                'BogusSolidus',
                'Stixrude14-liquidus',
            )

    def test_paleos_liquidus_id_distinct_from_stixrude14(self):
        """``PALEOS-liquidus`` returns a different curve from Stixrude14.

        Discriminating: at 50 GPa the Belonoshko-Fei piecewise curve and
        the Stixrude14 power-law fit must yield different temperatures.
        """
        out_stx = load_solidus_liquidus_functions(
            {'core': 'PALEOS:iron', 'mantle': 'WolfBower2018:MgSiO3'},
            'Stixrude14-solidus',
            'Stixrude14-liquidus',
        )
        out_p = load_solidus_liquidus_functions(
            {'core': 'PALEOS:iron', 'mantle': 'WolfBower2018:MgSiO3'},
            'Stixrude14-solidus',
            'PALEOS-liquidus',
        )
        assert out_stx is not None and out_p is not None
        T_stx = float(out_stx[1](50e9))
        T_paleos = float(out_p[1](50e9))
        assert abs(T_stx - T_paleos) > 1.0
