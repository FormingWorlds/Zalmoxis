"""Unit tests for the grid-runner output contract and the plot tools.

These tests do not run the Zalmoxis solver. They exercise:

- The pure helper functions in the three plot tools
  (``_resolve_grid_dir``, ``_load_summary``, ``_detect_sweep_params``,
  ``_try_float``, ``_choose_colour_param``, ``_read_str``,
  ``_mantle_uses_external_curves``, ``_layer_fractions``).
- The full main functions (``plot_grid_profiles``, ``plot_grid_pt``,
  ``plot_grid_composition``) end-to-end, by building a minimal
  synthetic grid directory (``grid_summary.csv`` + per-cell
  ``<label>.csv`` profile files) in ``tmp_path`` and asserting each
  tool writes its output image.
- The ``run_grid.run_single`` serialisation contract by monkeypatching
  the solver call with a synthetic ``model_results`` payload.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# plot_grid_composition._layer_fractions
# ---------------------------------------------------------------------------
def _synthetic_profile(M_total, R_total, f_core, f_cm, num_layers=50):
    """Build a monotone mass_enclosed profile for a simple test case.

    ``mass_enclosed`` is linear in the shell index; ``radii`` span 0 to
    ``R_total``. ``cmb_mass`` and ``core_mantle_mass`` are set to
    ``f_core * M_total`` and ``f_cm * M_total`` so the plot helper can
    locate the CMB and mantle outer boundary by linear interpolation.
    """
    radii = np.linspace(0.0, R_total, num_layers)
    mass = np.linspace(0.0, M_total, num_layers)
    return {
        'radii': radii,
        'mass_enclosed': mass,
        'cmb_mass': np.array(f_core * M_total),
        'core_mantle_mass': np.array(f_cm * M_total),
    }


def test_layer_fractions_two_layer_default():
    """When cmb_mass == core_mantle_mass (Zalmoxis 2-layer convention),
    the mantle must fill to the surface and the ice segment must be 0."""
    from tools.plots.plot_grid_composition import _layer_fractions

    M_total = 1.0e24
    R_total = 6.0e6
    # f_core == f_cm triggers the 2-layer branch
    data = _synthetic_profile(M_total, R_total, f_core=0.325, f_cm=0.325)
    fracs = _layer_fractions(data)

    core_m, mantle_m, ice_m = fracs['mass']
    core_r, mantle_r, ice_r = fracs['radius']

    assert ice_m == pytest.approx(0.0, abs=1e-12)
    assert ice_r == pytest.approx(0.0, abs=1e-12)
    assert core_m == pytest.approx(0.325, abs=1e-6)
    assert mantle_m == pytest.approx(0.675, abs=1e-6)
    # Sum to one in both panels.
    assert core_m + mantle_m + ice_m == pytest.approx(1.0, abs=1e-9)
    assert core_r + mantle_r + ice_r == pytest.approx(1.0, abs=1e-9)


def test_layer_fractions_three_layer():
    """3-layer config (mmf > 0): mantle and ice both occupy non-zero
    mass and radius."""
    from tools.plots.plot_grid_composition import _layer_fractions

    M_total = 1.0e24
    R_total = 6.0e6
    data = _synthetic_profile(M_total, R_total, f_core=0.30, f_cm=0.80)
    # Need explicit non-empty ice_layer_eos for the metadata branch.
    data['ice_layer_eos'] = np.str_('PALEOS:H2O')
    fracs = _layer_fractions(data)

    core_m, mantle_m, ice_m = fracs['mass']
    assert core_m == pytest.approx(0.30, abs=1e-6)
    assert mantle_m == pytest.approx(0.50, abs=1e-6)
    assert ice_m == pytest.approx(0.20, abs=1e-6)
    assert core_m + mantle_m + ice_m == pytest.approx(1.0, abs=1e-9)


def test_layer_fractions_thin_mantle_three_layer_uses_metadata():
    """A genuinely 3-layer run with a thin mantle (delta < 1e-3 of M)
    must NOT be misclassified as 2-layer when ice_layer_eos is set."""
    from tools.plots.plot_grid_composition import _layer_fractions

    M_total = 1.0e24
    R_total = 6.0e6
    # Mantle so thin the numeric heuristic would call this 2-layer
    # (delta = 1e-4 of M). The metadata flag keeps it 3-layer.
    data = _synthetic_profile(M_total, R_total, f_core=0.32, f_cm=0.3201)
    data['ice_layer_eos'] = np.str_('PALEOS:H2O')
    fracs = _layer_fractions(data)
    _, _, ice_m = fracs['mass']
    # Ice fraction should be ~ 1 - 0.3201 = 0.6799 (NOT zero).
    assert ice_m == pytest.approx(1.0 - 0.3201, abs=1e-6)


def test_layer_fractions_two_layer_metadata_overrides_tolerance():
    """Empty ice_layer_eos forces 2-layer interpretation even when
    cmb_mass and core_mantle_mass differ slightly."""
    from tools.plots.plot_grid_composition import _layer_fractions

    M_total = 1.0e24
    R_total = 6.0e6
    data = _synthetic_profile(M_total, R_total, f_core=0.325, f_cm=0.330)
    data['ice_layer_eos'] = np.str_('')  # explicitly empty
    fracs = _layer_fractions(data)
    _, _, ice_m = fracs['mass']
    assert ice_m == pytest.approx(0.0, abs=1e-12)


def test_layer_fractions_rejects_zero_mass():
    """Zero total mass returns None (caller skips with a 'zero mass' note)."""
    from tools.plots.plot_grid_composition import _layer_fractions

    n = 20
    data = {
        'radii': np.linspace(0.0, 6.0e6, n),
        'mass_enclosed': np.zeros(n),
        'cmb_mass': np.array(0.0),
        'core_mantle_mass': np.array(0.0),
    }
    assert _layer_fractions(data) is None


@pytest.mark.parametrize(
    'mantle_eos, expected',
    [
        ('PALEOS:MgSiO3', False),
        ('WolfBower2018:MgSiO3', True),
        ('RTPress100TPa:MgSiO3', True),
        ('PALEOS-2phase:MgSiO3', True),
        ('PALEOS:MgSiO3:0.9+Chabrier:H:0.1', False),
        ('WolfBower2018:MgSiO3:0.5+Chabrier:H:0.5', True),
        ('', False),
    ],
)
def test_mantle_uses_external_curves(mantle_eos, expected):
    from tools.plots.plot_grid_pt import _mantle_uses_external_curves

    assert _mantle_uses_external_curves(mantle_eos) is expected


def test_mantle_external_curves_is_solver_source():
    """The plot tool imports the solver's set directly so the two cannot
    drift out of sync. Asserting object identity ensures a future
    re-introduction of a local copy fails this test."""
    from tools.plots.plot_grid_pt import _EOS_USES_EXTERNAL_CURVES
    from zalmoxis.config import _NEEDS_MELTING_CURVES

    assert _EOS_USES_EXTERNAL_CURVES is _NEEDS_MELTING_CURVES


# ---------------------------------------------------------------------------
# run_grid.run_single: serialisation contract
# ---------------------------------------------------------------------------
# Six radial-profile arrays that must appear as columns in the per-cell
# CSV body, plus eight metadata fields that must appear in the comment
# header. ``label`` is added to the metadata header by the writer.
_EXPECTED_PROFILE_KEYS = {
    'radii',
    'density',
    'gravity',
    'pressure',
    'temperature',
    'mass_enclosed',
}
_EXPECTED_METADATA_KEYS = {
    'cmb_mass',
    'core_mantle_mass',
    'converged',
    'core_eos',
    'mantle_eos',
    'ice_layer_eos',
    'rock_solidus_id',
    'rock_liquidus_id',
    'label',
}


def _fake_model_results(num_layers=20):
    """A minimal synthetic ``model_results`` dict with physically
    monotone arrays, matching the keys ``run_single`` reads."""
    radii = np.linspace(0.0, 6.0e6, num_layers)
    # Linear mass profile; density is derived to be monotone but is not
    # required to be self-consistent for this contract test.
    mass = np.linspace(0.0, 5.972e24, num_layers)
    density = np.linspace(12000.0, 3000.0, num_layers)
    pressure = np.linspace(3.0e11, 1.0e5, num_layers)
    temperature = np.linspace(6000.0, 3000.0, num_layers)
    gravity = np.linspace(0.0, 9.81, num_layers)
    return {
        'layer_eos_config': {
            'core': 'PALEOS:iron',
            'mantle': 'PALEOS:MgSiO3',
            'ice_layer': '',
        },
        'radii': radii,
        'density': density,
        'gravity': gravity,
        'pressure': pressure,
        'temperature': temperature,
        'mass_enclosed': mass,
        'cmb_mass': 0.325 * mass[-1],
        'core_mantle_mass': 0.325 * mass[-1],  # 2-layer convention
        'total_time': 12.3,
        'converged': True,
        'converged_pressure': True,
        'converged_density': True,
        'converged_mass': True,
    }


def _install_fake_solver(monkeypatch):
    """Redirect the solver and loaders inside ``run_grid`` to synthetic
    outputs so the test never touches the real EOS data or runs Zalmoxis."""
    import tools.grids.run_grid as rg

    fake_config = {
        'layer_eos_config': {
            'core': 'PALEOS:iron',
            'mantle': 'PALEOS:MgSiO3',
            'ice_layer': '',
        },
        'rock_solidus': 'Monteux16-solidus',
        'rock_liquidus': 'Monteux16-liquidus-A-chondritic',
        # Extra keys the runner may read.
        'data_output_enabled': True,
        'plotting_enabled': True,
    }

    monkeypatch.setattr(rg, 'load_zalmoxis_config', lambda _path: fake_config)
    monkeypatch.setattr(rg, 'load_material_dictionaries', lambda: {})
    monkeypatch.setattr(
        rg,
        'load_solidus_liquidus_functions',
        lambda *_args, **_kwargs: (None, None),
    )
    monkeypatch.setattr(rg, 'main', lambda *_args, **_kwargs: _fake_model_results())


def test_run_single_writes_csv_with_expected_keys(tmp_path, monkeypatch):
    """save_profiles=True: the .csv must contain the six profile columns
    in its body and the eight metadata fields in its comment header."""
    from tools.grids.run_grid import run_single
    from tools.plots._grid_io import load_profile

    _install_fake_solver(monkeypatch)

    # run_single opens and unlinks the config path, so give it a real
    # (but arbitrary) temp file.
    cfg = tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False, dir=tmp_path)
    cfg.write('# fake config, contents are ignored by the mocked loader\n')
    cfg.close()

    label = 'planet_mass=1.0'
    out_dir = str(tmp_path / 'grid_out')
    result = run_single((label, cfg.name, out_dir, True))

    assert result['error'] is None
    assert result['converged'] is True

    # Summary files.
    assert os.path.isfile(os.path.join(out_dir, f'{label}.json'))
    with open(os.path.join(out_dir, f'{label}.json')) as fh:
        payload = json.load(fh)
    assert payload['label'] == label
    assert payload['converged'] is True

    # The profile CSV itself.
    csv_path = os.path.join(out_dir, f'{label}.csv')
    assert os.path.isfile(csv_path)

    # Comment header should be human-readable: opening the file in any
    # editor must show the metadata as `# key: value` lines.
    with open(csv_path) as fh:
        head = fh.read(2048)
    for key in _EXPECTED_METADATA_KEYS:
        assert f'# {key}:' in head, f'metadata key {key!r} missing from CSV header'

    # Body parses through the shared loader; preserves legacy in-memory
    # key shape so plot tools are untouched.
    data = load_profile(out_dir, label)
    assert data is not None
    for key in _EXPECTED_PROFILE_KEYS:
        assert key in data, f'profile column {key!r} missing'
        assert data[key].shape == (20,)
        assert data[key].dtype.kind == 'f'

    assert data['converged'] is True
    assert data['core_eos'] == 'PALEOS:iron'
    assert data['mantle_eos'] == 'PALEOS:MgSiO3'
    assert data['ice_layer_eos'] == ''
    assert data['rock_solidus_id'] == 'Monteux16-solidus'
    assert data['rock_liquidus_id'] == 'Monteux16-liquidus-A-chondritic'

    # Round-trip precision: 17g formatting preserves the synthetic
    # arrays bit-for-bit (within float64 round-trip).
    assert data['radii'][0] == pytest.approx(0.0, abs=0.0)
    assert data['radii'][-1] == pytest.approx(6.0e6, rel=0.0, abs=0.0)


def test_run_single_no_profiles_when_disabled(tmp_path, monkeypatch):
    """save_profiles=False: no .csv is written (only the per-run JSON)."""
    from tools.grids.run_grid import run_single

    _install_fake_solver(monkeypatch)

    cfg = tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False, dir=tmp_path)
    cfg.write('# fake config\n')
    cfg.close()

    label = 'planet_mass=1.0'
    out_dir = str(tmp_path / 'grid_out')
    result = run_single((label, cfg.name, out_dir, False))

    assert result['error'] is None
    assert os.path.isfile(os.path.join(out_dir, f'{label}.json'))
    assert not os.path.exists(os.path.join(out_dir, f'{label}.csv'))


def test_load_grid_config_rejects_non_bool_save_profiles(tmp_path, monkeypatch):
    """Guard against the quoted 'false' footgun: save_profiles must be
    a real TOML bool, otherwise load_grid_config raises TypeError."""
    import tools.grids.run_grid as rg

    monkeypatch.setattr(rg, 'get_zalmoxis_root', lambda: str(tmp_path))
    base = tmp_path / 'input'
    base.mkdir()
    (base / 'default.toml').write_text('# base config')
    grid = tmp_path / 'bad.toml'
    grid.write_text(
        '[base]\nconfig = "input/default.toml"\n'
        '[sweep]\nplanet_mass = [1.0]\n'
        '[output]\ndir = "output/bad"\nsave_profiles = "false"\n'
    )
    with pytest.raises(TypeError, match='save_profiles must be a bool'):
        rg.load_grid_config(str(grid))


def test_load_grid_config_happy_path(tmp_path, monkeypatch):
    """Well-formed grid TOML with save_profiles=true returns the 4-tuple
    (base_config_path, sweeps, output_dir, save_profiles) correctly."""
    import tools.grids.run_grid as rg

    monkeypatch.setattr(rg, 'get_zalmoxis_root', lambda: str(tmp_path))
    base = tmp_path / 'input'
    base.mkdir()
    (base / 'default.toml').write_text('# base config')
    grid = tmp_path / 'ok.toml'
    grid.write_text(
        '[base]\nconfig = "input/default.toml"\n'
        '[sweep]\nplanet_mass = [0.5, 1.0]\n'
        '[output]\ndir = "output/ok"\nsave_profiles = true\n'
    )
    base_cfg, sweeps, out_dir, save = rg.load_grid_config(str(grid))
    assert base_cfg.endswith('default.toml')
    assert sweeps == {'planet_mass': [0.5, 1.0]}
    assert out_dir.endswith('output/ok')
    assert save is True


def test_load_grid_config_default_save_profiles_false(tmp_path, monkeypatch):
    """If [output].save_profiles is omitted, the default is False (no
    per-grid-point .csv write)."""
    import tools.grids.run_grid as rg

    monkeypatch.setattr(rg, 'get_zalmoxis_root', lambda: str(tmp_path))
    base = tmp_path / 'input'
    base.mkdir()
    (base / 'default.toml').write_text('# base config')
    grid = tmp_path / 'no_save.toml'
    grid.write_text(
        '[base]\nconfig = "input/default.toml"\n'
        '[sweep]\nplanet_mass = [1.0]\n'
        '[output]\ndir = "output/x"\n'
    )
    _, _, _, save = rg.load_grid_config(str(grid))
    assert save is False


def test_load_grid_config_unknown_sweep_param(tmp_path, monkeypatch):
    """An unknown sweep parameter name is rejected at load time."""
    import tools.grids.run_grid as rg

    monkeypatch.setattr(rg, 'get_zalmoxis_root', lambda: str(tmp_path))
    base = tmp_path / 'input'
    base.mkdir()
    (base / 'default.toml').write_text('# base config')
    grid = tmp_path / 'bad_param.toml'
    grid.write_text(
        '[base]\nconfig = "input/default.toml"\n'
        '[sweep]\nwizardry = [1, 2]\n'
        '[output]\ndir = "out"\n'
    )
    with pytest.raises(ValueError, match='Unknown sweep parameter'):
        rg.load_grid_config(str(grid))


def test_param_map_target_surface_pressure_registered():
    """target_surface_pressure must map to (PressureAdjustment,
    target_surface_pressure). Pins the registry entry so a future
    refactor of _PARAM_MAP cannot silently drop it (the entry was
    missing before 2026-05-12 and broke standalone grids)."""
    import tools.grids.run_grid as rg

    assert 'target_surface_pressure' in rg._PARAM_MAP
    section, key = rg._PARAM_MAP['target_surface_pressure']
    assert section == 'PressureAdjustment'
    assert key == 'target_surface_pressure'


def test_load_grid_config_accepts_target_surface_pressure(tmp_path, monkeypatch):
    """End-to-end loader regression: a sweep over target_surface_pressure
    parses cleanly and round-trips the values. Guards against the
    pre-2026-05-12 ValueError ("Unknown sweep parameter
    'target_surface_pressure'") that blocked standalone P_surf grids."""
    import tools.grids.run_grid as rg

    monkeypatch.setattr(rg, 'get_zalmoxis_root', lambda: str(tmp_path))
    base = tmp_path / 'input'
    base.mkdir()
    (base / 'default.toml').write_text('# base config')
    grid = tmp_path / 'psurf.toml'
    grid.write_text(
        '[base]\nconfig = "input/default.toml"\n'
        '[sweep]\ntarget_surface_pressure = [1.013e5, 1.0e7, 1.0e9]\n'
        '[output]\ndir = "output/psurf"\n'
    )
    base_cfg, sweeps, out_dir, save = rg.load_grid_config(str(grid))
    assert base_cfg.endswith('default.toml')
    assert sweeps == {'target_surface_pressure': [1.013e5, 1.0e7, 1.0e9]}
    assert out_dir.endswith('output/psurf')
    assert save is False


def test_run_single_profile_write_failure_reports_error(tmp_path, monkeypatch):
    """When the profile-CSV writer raises OSError, run_single logs a
    warning and populates result['error'] so the failure surfaces in
    grid_summary.csv (rather than silently disagreeing with the JSON)."""
    import tools.grids.run_grid as rg
    from tools.grids.run_grid import run_single

    _install_fake_solver(monkeypatch)

    # Make the profile-CSV writer blow up inside run_single.
    def _raise(*_args, **_kwargs):
        raise OSError('simulated disk full')

    monkeypatch.setattr(rg, '_write_profile_csv', _raise)

    cfg = tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False, dir=tmp_path)
    cfg.write('# fake config\n')
    cfg.close()

    label = 'planet_mass=1.0'
    out_dir = str(tmp_path / 'grid_out')
    result = run_single((label, cfg.name, out_dir, True))

    assert result['error'] is not None
    assert 'profile write failed' in result['error']
    assert 'simulated disk full' in result['error']


# ---------------------------------------------------------------------------
# End-to-end tests for the three plot tools
# ---------------------------------------------------------------------------
def _write_synthetic_grid(gdir, masses=(0.5, 1.0, 2.0), mantle_eos='PALEOS:MgSiO3'):
    """Create a tiny ``grid_summary.csv`` + per-cell profile CSV set in
    ``gdir``. Lets the plot tools run end-to-end with no solver and no
    real EOS data.
    """
    import csv

    from tools.grids.run_grid import _write_profile_csv

    gdir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'label',
        'planet_mass',
        'R_earth',
        'M_earth',
        'converged',
        'converged_pressure',
        'converged_density',
        'converged_mass',
        'time_s',
        'error',
    ]
    with open(gdir / 'grid_summary.csv', 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for m in masses:
            writer.writerow(
                {
                    'label': f'planet_mass={m}',
                    'planet_mass': m,
                    'R_earth': m**0.27,
                    'M_earth': m,
                    'converged': 'True',
                    'converged_pressure': 'True',
                    'converged_density': 'True',
                    'converged_mass': 'True',
                    'time_s': 10.0,
                    'error': '',
                }
            )

    n = 20
    M_earth = 5.972e24
    R_earth = 6.371e6
    layer_eos = {
        'core': 'PALEOS:iron',
        'mantle': mantle_eos,
        'ice_layer': '',
    }
    for m in masses:
        label = f'planet_mass={m}'
        model_results = {
            'radii': np.linspace(0.0, R_earth * m**0.27, n),
            'density': np.linspace(12000.0, 3000.0, n),
            'pressure': np.linspace(3e11 * m, 1e5, n),
            'temperature': np.linspace(6000.0 * m**0.2, 3000.0, n),
            'gravity': np.linspace(0.0, 10.0 * m**0.5, n),
            'mass_enclosed': np.linspace(0.0, M_earth * m, n),
            'cmb_mass': 0.325 * M_earth * m,
            'core_mantle_mass': 0.325 * M_earth * m,  # 2-layer
            'converged': True,
        }
        _write_profile_csv(
            gdir / f'{label}.csv',
            label=label,
            model_results=model_results,
            layer_eos=layer_eos,
            rock_solidus_id='Monteux16-solidus',
            rock_liquidus_id='Monteux16-liquidus-A-chondritic',
        )
    return gdir


@pytest.fixture
def fake_grid_dir(tmp_path):
    return _write_synthetic_grid(tmp_path / 'grid')


@pytest.fixture
def fake_grid_dir_external_curves(tmp_path):
    """Synthetic grid whose mantle EOS triggers the external-curves overlay."""
    return _write_synthetic_grid(tmp_path / 'grid_wb', mantle_eos='WolfBower2018:MgSiO3')


def test_plot_grid_profiles_end_to_end(fake_grid_dir, tmp_path):
    from tools.plots.plot_grid_profiles import plot_grid_profiles

    out = tmp_path / 'profiles.png'
    result = plot_grid_profiles(str(fake_grid_dir), out=str(out))
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 1000  # non-empty image


def test_plot_grid_profiles_log_pressure_masks_zero(fake_grid_dir, tmp_path):
    """log_pressure=True exercises the P>0 mask branch."""
    from tools.grids.run_grid import _write_profile_csv
    from tools.plots._grid_io import load_profile
    from tools.plots.plot_grid_profiles import plot_grid_profiles

    # Rewrite one of the per-cell CSVs so its surface shell has P == 0.
    label = 'planet_mass=1.0'
    data = load_profile(str(fake_grid_dir), label)
    assert data is not None
    data['pressure'][-1] = 0.0
    _write_profile_csv(
        fake_grid_dir / f'{label}.csv',
        label=label,
        model_results=data,
        layer_eos={
            'core': data.get('core_eos', ''),
            'mantle': data.get('mantle_eos', ''),
            'ice_layer': data.get('ice_layer_eos', ''),
        },
        rock_solidus_id=data.get('rock_solidus_id', ''),
        rock_liquidus_id=data.get('rock_liquidus_id', ''),
    )

    out = tmp_path / 'profiles_log.png'
    result = plot_grid_profiles(str(fake_grid_dir), out=str(out), log_pressure=True)
    assert os.path.isfile(result)


def test_plot_grid_pt_end_to_end_no_overlay(fake_grid_dir, tmp_path):
    """PALEOS:MgSiO3 mantle: overlay is auto-suppressed with a note."""
    from tools.plots.plot_grid_pt import plot_grid_pt

    out = tmp_path / 'pt.png'
    result = plot_grid_pt(str(fake_grid_dir), out=str(out))
    assert os.path.isfile(result)


def test_plot_grid_pt_no_melting_curves_flag(fake_grid_dir, tmp_path):
    """show_melting_curves=False short-circuits the overlay branch."""
    from tools.plots.plot_grid_pt import plot_grid_pt

    out = tmp_path / 'pt_bare.png'
    result = plot_grid_pt(str(fake_grid_dir), out=str(out), show_melting_curves=False)
    assert os.path.isfile(result)


def test_plot_grid_pt_overlay_from_metadata(fake_grid_dir_external_curves, tmp_path):
    """WolfBower2018:MgSiO3 mantle: overlay is loaded from the stored
    rock_solidus_id / rock_liquidus_id and drawn."""
    from tools.plots.plot_grid_pt import plot_grid_pt

    out = tmp_path / 'pt_wb.png'
    result = plot_grid_pt(str(fake_grid_dir_external_curves), out=str(out))
    assert os.path.isfile(result)


def test_plot_grid_pt_forced_overlay(fake_grid_dir, tmp_path):
    """Explicit --solidus / --liquidus override the auto-suppression."""
    from tools.plots.plot_grid_pt import plot_grid_pt

    out = tmp_path / 'pt_forced.png'
    result = plot_grid_pt(
        str(fake_grid_dir),
        out=str(out),
        solidus='Stixrude14-solidus',
        liquidus='Stixrude14-liquidus',
    )
    assert os.path.isfile(result)


def test_plot_grid_pt_linear_pressure(fake_grid_dir, tmp_path):
    from tools.plots.plot_grid_pt import plot_grid_pt

    out = tmp_path / 'pt_linear.png'
    result = plot_grid_pt(str(fake_grid_dir), out=str(out), linear_pressure=True)
    assert os.path.isfile(result)


def test_plot_grid_composition_end_to_end(fake_grid_dir, tmp_path):
    from tools.plots.plot_grid_composition import plot_grid_composition

    out = tmp_path / 'comp.png'
    result = plot_grid_composition(str(fake_grid_dir), out=str(out))
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 1000


def test_plot_tools_skip_non_converged(fake_grid_dir, tmp_path):
    """A row marked converged=False must be skipped by all three tools."""
    csv_path = fake_grid_dir / 'grid_summary.csv'
    lines = csv_path.read_text().splitlines()
    header = lines[0]
    new_lines = [header]
    for line in lines[1:]:
        parts = line.split(',')
        if parts[0] == 'planet_mass=1.0':
            parts[4] = 'False'  # 'converged' column
        new_lines.append(','.join(parts))
    csv_path.write_text('\n'.join(new_lines) + '\n')

    from tools.plots.plot_grid_composition import plot_grid_composition
    from tools.plots.plot_grid_profiles import plot_grid_profiles
    from tools.plots.plot_grid_pt import plot_grid_pt

    assert os.path.isfile(plot_grid_profiles(str(fake_grid_dir), out=str(tmp_path / 'p.png')))
    assert os.path.isfile(plot_grid_pt(str(fake_grid_dir), out=str(tmp_path / 'pt.png')))
    assert os.path.isfile(
        plot_grid_composition(str(fake_grid_dir), out=str(tmp_path / 'c.png'))
    )


def test_plot_grid_profiles_default_output_path(fake_grid_dir):
    """When out is None the tool writes <grid_dir>/profiles_vs_radius.pdf."""
    from tools.plots.plot_grid_profiles import plot_grid_profiles

    result = plot_grid_profiles(str(fake_grid_dir))
    assert os.path.basename(result) == 'profiles_vs_radius.pdf'
    assert os.path.isfile(result)


# ---------------------------------------------------------------------------
# Helper-function tests
# ---------------------------------------------------------------------------
def test_resolve_grid_dir_accepts_directory(fake_grid_dir):
    from tools.plots.plot_grid_profiles import _resolve_grid_dir

    assert _resolve_grid_dir(str(fake_grid_dir)) == str(fake_grid_dir)


def test_resolve_grid_dir_accepts_csv_path(fake_grid_dir):
    from tools.plots.plot_grid_profiles import _resolve_grid_dir

    csv_path = str(fake_grid_dir / 'grid_summary.csv')
    resolved = _resolve_grid_dir(csv_path)
    assert os.path.basename(resolved) == os.path.basename(str(fake_grid_dir))


def test_resolve_grid_dir_rejects_other_files(tmp_path):
    from tools.plots.plot_grid_profiles import _resolve_grid_dir

    bogus = tmp_path / 'note.txt'
    bogus.write_text('hi')
    with pytest.raises(ValueError):
        _resolve_grid_dir(str(bogus))


def test_load_summary_missing_csv(tmp_path):
    from tools.plots.plot_grid_profiles import _load_summary

    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        _load_summary(str(empty))


def test_try_float_behaviour():
    from tools.plots.plot_grid_profiles import _try_float

    assert _try_float('1.5') == 1.5
    assert _try_float('0') == 0.0
    assert _try_float('') is None
    assert _try_float('abc') is None
    assert _try_float(None) is None


def test_detect_sweep_params_strips_fixed_columns():
    from tools.plots.plot_grid_profiles import _detect_sweep_params

    rows = [
        {
            'label': 'x',
            'planet_mass': '1.0',
            'mantle': 'PALEOS:MgSiO3',
            'R_earth': '1.0',
            'M_earth': '1.0',
            'converged': 'True',
            'converged_pressure': 'True',
            'converged_density': 'True',
            'converged_mass': 'True',
            'time_s': '1',
            'error': '',
        }
    ]
    assert _detect_sweep_params(rows) == ['mantle', 'planet_mass']


def test_choose_colour_param_rejects_invalid_override():
    from tools.plots.plot_grid_profiles import _choose_colour_param

    with pytest.raises(ValueError):
        _choose_colour_param(['planet_mass'], [{'planet_mass': '1.0'}], 'nope')


def test_choose_colour_param_prefers_numeric():
    from tools.plots.plot_grid_profiles import _choose_colour_param

    rows = [{'mantle': 'PALEOS:MgSiO3', 'planet_mass': '1.0'}]
    assert _choose_colour_param(['mantle', 'planet_mass'], rows) == 'planet_mass'


def test_read_str_handles_missing_and_numpy_scalars():
    from tools.plots.plot_grid_pt import _read_str

    data = {'a': np.str_('hello'), 'b': 'raw'}
    assert _read_str(data, 'a') == 'hello'
    assert _read_str(data, 'b') == 'raw'
    assert _read_str(data, 'missing') == ''


# ---------------------------------------------------------------------------
# CLI parser coverage for the three plot tools
# ---------------------------------------------------------------------------
def test_plot_grid_profiles_parser_defaults_and_flags():
    from tools.plots.plot_grid_profiles import _build_parser

    parser = _build_parser()
    args = parser.parse_args(['/some/grid'])
    assert args.path == '/some/grid'
    assert args.output is None
    assert args.colour_by is None
    assert args.log_pressure is False
    assert args.dpi == 200

    args = parser.parse_args(
        [
            '/g',
            '-o',
            'out.pdf',
            '--colour-by',
            'surface_temperature',
            '--log-pressure',
            '--dpi',
            '120',
        ]
    )
    assert args.output == 'out.pdf'
    assert args.colour_by == 'surface_temperature'
    assert args.log_pressure is True
    assert args.dpi == 120


def test_plot_grid_pt_parser_defaults_and_flags():
    from tools.plots.plot_grid_pt import _build_parser

    parser = _build_parser()
    args = parser.parse_args(['/g'])
    assert args.path == '/g'
    assert args.solidus is None
    assert args.liquidus is None
    assert args.show_melting_curves is True
    assert args.linear_pressure is False

    args = parser.parse_args(
        [
            '/g',
            '--solidus',
            'Stixrude14-solidus',
            '--liquidus',
            'Stixrude14-liquidus',
            '--no-melting-curves',
            '--linear-pressure',
            '--color-by',
            'planet_mass',
        ]
    )
    assert args.solidus == 'Stixrude14-solidus'
    assert args.liquidus == 'Stixrude14-liquidus'
    assert args.show_melting_curves is False
    assert args.linear_pressure is True
    assert args.colour_by == 'planet_mass'


def test_plot_grid_composition_parser_defaults_and_flags():
    from tools.plots.plot_grid_composition import _build_parser

    parser = _build_parser()
    args = parser.parse_args(['/g'])
    assert args.path == '/g'
    assert args.output is None
    assert args.label_by is None
    assert args.dpi == 200

    args = parser.parse_args(['/g', '-o', 'c.png', '--label-by', 'mantle', '--dpi', '300'])
    assert args.output == 'c.png'
    assert args.label_by == 'mantle'
    assert args.dpi == 300


# ---------------------------------------------------------------------------
# Additional plot_grid_pt overlay-suppression branches
# ---------------------------------------------------------------------------
def _rewrite_profile_csv(path, **overrides):
    """Helper: load a profile CSV, apply metadata overrides, write back."""
    from tools.grids.run_grid import _write_profile_csv
    from tools.plots._grid_io import load_profile

    grid_dir = path.parent
    label = path.stem
    data = load_profile(str(grid_dir), label)
    assert data is not None, f'expected an existing CSV at {path}'

    layer_eos = {
        'core': data.get('core_eos', ''),
        'mantle': data.get('mantle_eos', ''),
        'ice_layer': data.get('ice_layer_eos', ''),
    }
    rock_solidus_id = data.get('rock_solidus_id', '')
    rock_liquidus_id = data.get('rock_liquidus_id', '')

    for key, value in overrides.items():
        if key == 'mantle_eos':
            layer_eos['mantle'] = value
        elif key == 'core_eos':
            layer_eos['core'] = value
        elif key == 'ice_layer_eos':
            layer_eos['ice_layer'] = value
        elif key == 'rock_solidus_id':
            rock_solidus_id = value
        elif key == 'rock_liquidus_id':
            rock_liquidus_id = value
        else:
            data[key] = value

    _write_profile_csv(
        path,
        label=label,
        model_results=data,
        layer_eos=layer_eos,
        rock_solidus_id=rock_solidus_id,
        rock_liquidus_id=rock_liquidus_id,
    )


def test_plot_grid_pt_mantle_varies_across_grid_suppresses_overlay(tmp_path):
    """A grid whose mantle_eos differs across points must suppress the
    overlay with the 'differs across grid points' note."""
    from tools.plots.plot_grid_pt import plot_grid_pt

    gdir = _write_synthetic_grid(
        tmp_path / 'mixed1', masses=(1.0, 2.0), mantle_eos='PALEOS:MgSiO3'
    )
    # Rewrite one CSV so it carries a different mantle_eos string but
    # keep grid_summary.csv consistent.
    _rewrite_profile_csv(
        gdir / 'planet_mass=1.0.csv',
        mantle_eos='WolfBower2018:MgSiO3',
    )

    out = tmp_path / 'pt_mixed.png'
    result = plot_grid_pt(str(gdir), out=str(out))
    assert os.path.isfile(result)


def test_plot_grid_pt_missing_curve_metadata_suppresses_overlay(tmp_path):
    """An external-curves mantle with empty rock_solidus_id must suppress
    the overlay (fallback branch when metadata is absent)."""
    from tools.plots.plot_grid_pt import plot_grid_pt

    gdir = _write_synthetic_grid(
        tmp_path / 'nometa', masses=(1.0, 2.0), mantle_eos='WolfBower2018:MgSiO3'
    )
    for csv_path in gdir.glob('planet_mass=*.csv'):
        _rewrite_profile_csv(csv_path, rock_solidus_id='', rock_liquidus_id='')

    out = tmp_path / 'pt_nometa.png'
    result = plot_grid_pt(str(gdir), out=str(out))
    assert os.path.isfile(result)


def test_plot_grid_pt_forced_invalid_curve_id_caught(tmp_path, fake_grid_dir):
    """An unknown curve id in an explicit override is caught and the plot
    still renders with an empty overlay (the exception handler runs)."""
    from tools.plots.plot_grid_pt import plot_grid_pt

    out = tmp_path / 'pt_badid.png'
    result = plot_grid_pt(
        str(fake_grid_dir),
        out=str(out),
        solidus='NotARealCurveId',
        liquidus='AlsoBogus',
    )
    assert os.path.isfile(result)


def test_plot_grid_profiles_skips_missing_csv(tmp_path):
    """A summary row whose profile CSV does not exist is skipped."""
    from tools.plots.plot_grid_profiles import plot_grid_profiles

    gdir = _write_synthetic_grid(tmp_path / 'g', masses=(0.5, 1.0))
    (gdir / 'planet_mass=1.0.csv').unlink()  # remove one profile

    out = tmp_path / 'profiles_missing.png'
    result = plot_grid_profiles(str(gdir), out=str(out))
    assert os.path.isfile(result)


def test_plot_grid_profiles_custom_colour_by(fake_grid_dir, tmp_path):
    """Non-default --colour-by exercises the explicit-override branch."""
    from tools.plots.plot_grid_profiles import plot_grid_profiles

    out = tmp_path / 'profiles_by_pm.png'
    result = plot_grid_profiles(str(fake_grid_dir), out=str(out), colour_by='planet_mass')
    assert os.path.isfile(result)
