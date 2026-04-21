"""Unit tests for the grid-runner output contract and the plot tools.

These tests do not run the Zalmoxis solver. They exercise:

- The pure helper functions in the three plot tools
  (``_resolve_grid_dir``, ``_load_summary``, ``_detect_sweep_params``,
  ``_try_float``, ``_choose_colour_param``, ``_read_str``,
  ``_mantle_uses_external_curves``, ``_layer_fractions``).
- The full main functions (``plot_grid_profiles``, ``plot_grid_pt``,
  ``plot_grid_composition``) end-to-end, by building a minimal
  synthetic grid directory (CSV + ``.npz`` files) in ``tmp_path`` and
  asserting each tool writes its output image.
- The ``run_grid.run_single`` serialisation contract by monkeypatching
  the solver call with a synthetic ``model_results`` payload.
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest


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


@pytest.mark.unit
def test_layer_fractions_two_layer_default():
    """When cmb_mass == core_mantle_mass (Zalmoxis 2-layer convention),
    the mantle must fill to the surface and the ice segment must be 0."""
    from src.tools.plot_grid_composition import _layer_fractions

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


@pytest.mark.unit
def test_layer_fractions_three_layer():
    """When core_mantle_mass < M_total, the ice segment fills the rest."""
    from src.tools.plot_grid_composition import _layer_fractions

    M_total = 1.0e24
    R_total = 6.0e6
    # f_core = 0.30, f_cm = 0.70 -> mantle 0.40, ice 0.30
    data = _synthetic_profile(M_total, R_total, f_core=0.30, f_cm=0.70)
    fracs = _layer_fractions(data)

    core_m, mantle_m, ice_m = fracs['mass']
    assert core_m == pytest.approx(0.30, abs=1e-6)
    assert mantle_m == pytest.approx(0.40, abs=1e-6)
    assert ice_m == pytest.approx(0.30, abs=1e-6)
    assert core_m + mantle_m + ice_m == pytest.approx(1.0, abs=1e-9)

    # Linear mass -> radius mapping in the synthetic profile, so radius
    # fractions equal mass fractions to within interpolation resolution.
    core_r, mantle_r, ice_r = fracs['radius']
    assert core_r == pytest.approx(0.30, abs=0.05)
    assert mantle_r == pytest.approx(0.40, abs=0.05)
    assert ice_r == pytest.approx(0.30, abs=0.05)


@pytest.mark.unit
def test_layer_fractions_thin_mantle_three_layer_uses_metadata():
    """A valid 3-layer run with a tiny but nonzero mantle fraction (below
    the old 1e-3 numerical tolerance) must be classified as 3-layer when
    ice_layer_eos metadata is present, so the ice segment renders
    correctly. Regression guard for bot review comment #2."""
    from src.tools.plot_grid_composition import _layer_fractions

    M_total = 1.0e24
    R_total = 6.0e6
    # f_core = 0.325, f_cm = 0.3255 -> mantle = 5e-4, ice = 0.6745
    # This is well inside the old tolerance |cm - core| < 1e-3 * M_total.
    data = _synthetic_profile(M_total, R_total, f_core=0.325, f_cm=0.3255)
    # Explicit metadata: non-empty ice_layer_eos flags a 3-layer run.
    data['ice_layer_eos'] = np.array('Seager2007:H2O')

    fracs = _layer_fractions(data)
    core_m, mantle_m, ice_m = fracs['mass']

    assert core_m == pytest.approx(0.325, abs=1e-6)
    assert mantle_m == pytest.approx(5e-4, abs=1e-6)
    assert ice_m == pytest.approx(0.6745, abs=1e-6)
    assert ice_m > 0.0  # THE point of this test: ice segment is not suppressed


@pytest.mark.unit
def test_layer_fractions_two_layer_metadata_overrides_tolerance():
    """With empty ``ice_layer_eos`` metadata, the run is a 2-layer planet
    regardless of the numerical relationship between cmb_mass and
    core_mantle_mass."""
    from src.tools.plot_grid_composition import _layer_fractions

    M_total = 1.0e24
    R_total = 6.0e6
    data = _synthetic_profile(M_total, R_total, f_core=0.325, f_cm=0.325)
    data['ice_layer_eos'] = np.array('')

    fracs = _layer_fractions(data)
    core_m, mantle_m, ice_m = fracs['mass']

    assert ice_m == pytest.approx(0.0, abs=1e-12)
    assert mantle_m == pytest.approx(0.675, abs=1e-6)


@pytest.mark.unit
def test_layer_fractions_rejects_zero_mass():
    """A grid point with zero total mass or radius returns None."""
    from src.tools.plot_grid_composition import _layer_fractions

    bad = {
        'radii': np.zeros(10),
        'mass_enclosed': np.zeros(10),
        'cmb_mass': np.array(0.0),
        'core_mantle_mass': np.array(0.0),
    }
    assert _layer_fractions(bad) is None


# ---------------------------------------------------------------------------
# plot_grid_pt._mantle_uses_external_curves
# ---------------------------------------------------------------------------
@pytest.mark.unit
@pytest.mark.parametrize(
    ('mantle_eos', 'expected'),
    [
        ('WolfBower2018:MgSiO3', True),
        ('RTPress100TPa:MgSiO3', True),
        ('PALEOS-2phase:MgSiO3', True),
        ('PALEOS:MgSiO3', False),
        ('PALEOS:iron', False),
        ('Seager2007:MgSiO3', False),
        ('Analytic:MgSiO3', False),
        ('', False),
        # Mixture with a needs-curves component triggers the overlay.
        ('WolfBower2018:MgSiO3:0.9+Chabrier:H:0.1', True),
        # Mixture where no component needs external curves.
        ('PALEOS:MgSiO3:0.9+PALEOS:H2O:0.1', False),
    ],
)
def test_mantle_uses_external_curves(mantle_eos, expected):
    """The overlay policy must stay in lockstep with the solver's
    ``_NEEDS_MELTING_CURVES`` set; this catches drift if either side is
    updated without the other."""
    from src.tools.plot_grid_pt import _mantle_uses_external_curves

    assert _mantle_uses_external_curves(mantle_eos) is expected


@pytest.mark.unit
def test_mantle_external_curves_is_solver_source():
    """The plot tool now imports the solver's set directly (same
    ``src.zalmoxis.zalmoxis`` path used by ``run_grid.py``); assert the
    object identity so a future re-introduction of a local copy fails
    this test."""
    from src.tools.plot_grid_pt import _EOS_USES_EXTERNAL_CURVES
    from src.zalmoxis.zalmoxis import _NEEDS_MELTING_CURVES

    assert _EOS_USES_EXTERNAL_CURVES is _NEEDS_MELTING_CURVES


# ---------------------------------------------------------------------------
# run_grid.run_single: serialisation contract
# ---------------------------------------------------------------------------
_EXPECTED_NPZ_KEYS = {
    'radii',
    'density',
    'gravity',
    'pressure',
    'temperature',
    'mass_enclosed',
    'cmb_mass',
    'core_mantle_mass',
    'converged',
    'core_eos',
    'mantle_eos',
    'ice_layer_eos',
    'rock_solidus_id',
    'rock_liquidus_id',
}


def _fake_model_results(num_layers=20):
    """A minimal synthetic ``model_results`` dict with physically monotone
    arrays, matching the keys ``run_single`` reads."""
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
    import src.tools.run_grid as rg

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


@pytest.mark.unit
def test_run_single_writes_npz_with_expected_keys(tmp_path, monkeypatch):
    """save_profiles=True: the .npz must contain exactly the 14 keys the
    plot tools depend on, in the expected dtypes."""
    from src.tools.run_grid import run_single

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

    # The archive itself.
    npz_path = os.path.join(out_dir, f'{label}.npz')
    assert os.path.isfile(npz_path)
    with np.load(npz_path) as data:
        assert set(data.files) == _EXPECTED_NPZ_KEYS
        # Profile arrays: 1D, length 20, float dtype.
        assert data['radii'].shape == (20,)
        assert data['density'].dtype.kind == 'f'
        # Converged stored as a true boolean.
        assert data['converged'].dtype == np.bool_
        assert bool(data['converged']) is True
        # Metadata strings.
        assert str(data['core_eos'].item()) == 'PALEOS:iron'
        assert str(data['mantle_eos'].item()) == 'PALEOS:MgSiO3'
        assert str(data['ice_layer_eos'].item()) == ''
        assert str(data['rock_solidus_id'].item()) == 'Monteux16-solidus'
        assert str(data['rock_liquidus_id'].item()) == 'Monteux16-liquidus-A-chondritic'


@pytest.mark.unit
def test_run_single_no_profiles_when_disabled(tmp_path, monkeypatch):
    """save_profiles=False: no .npz is written (backward-compat path)."""
    from src.tools.run_grid import run_single

    _install_fake_solver(monkeypatch)

    cfg = tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False, dir=tmp_path)
    cfg.write('# fake config\n')
    cfg.close()

    label = 'planet_mass=1.0'
    out_dir = str(tmp_path / 'grid_out')
    result = run_single((label, cfg.name, out_dir, False))

    assert result['error'] is None
    assert os.path.isfile(os.path.join(out_dir, f'{label}.json'))
    assert not os.path.exists(os.path.join(out_dir, f'{label}.npz'))


@pytest.mark.unit
def test_load_grid_config_rejects_non_bool_save_profiles(tmp_path, monkeypatch):
    """Guard against the quoted 'false' footgun: save_profiles must be
    a real TOML bool, otherwise load_grid_config raises TypeError."""
    import src.tools.run_grid as rg

    monkeypatch.setattr(rg, 'ZALMOXIS_ROOT', str(tmp_path))
    base = tmp_path / 'input'
    base.mkdir()
    (base / 'default.toml').write_text('# base config')
    grid = tmp_path / 'bad.toml'
    grid.write_text(
        '[base]\nconfig = "input/default.toml"\n'
        '[sweep]\nplanet_mass = [1.0]\n'
        '[output]\ndir = "output_files/bad"\nsave_profiles = "false"\n'
    )
    with pytest.raises(TypeError, match='save_profiles must be a bool'):
        rg.load_grid_config(str(grid))


@pytest.mark.unit
def test_load_grid_config_happy_path(tmp_path, monkeypatch):
    """Well-formed grid TOML with save_profiles=true returns the 4-tuple
    (base_config_path, sweeps, output_dir, save_profiles) correctly."""
    import src.tools.run_grid as rg

    monkeypatch.setattr(rg, 'ZALMOXIS_ROOT', str(tmp_path))
    base = tmp_path / 'input'
    base.mkdir()
    (base / 'default.toml').write_text('# base config')
    grid = tmp_path / 'ok.toml'
    grid.write_text(
        '[base]\nconfig = "input/default.toml"\n'
        '[sweep]\nplanet_mass = [0.5, 1.0]\n'
        '[output]\ndir = "output_files/ok"\nsave_profiles = true\n'
    )
    base_cfg, sweeps, out_dir, save = rg.load_grid_config(str(grid))
    assert base_cfg.endswith('default.toml')
    assert sweeps == {'planet_mass': [0.5, 1.0]}
    assert out_dir.endswith('output_files/ok')
    assert save is True


@pytest.mark.unit
def test_load_grid_config_unknown_sweep_param(tmp_path, monkeypatch):
    """An unknown sweep parameter name is rejected at load time."""
    import src.tools.run_grid as rg

    monkeypatch.setattr(rg, 'ZALMOXIS_ROOT', str(tmp_path))
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


@pytest.mark.unit
def test_run_single_profile_write_failure_reports_error(tmp_path, monkeypatch):
    """When np.savez_compressed raises OSError, run_single logs a warning
    and populates result['error'] so the failure surfaces in
    grid_summary.csv (fix for bot review #3)."""
    import src.tools.run_grid as rg
    from src.tools.run_grid import run_single

    _install_fake_solver(monkeypatch)

    # Make np.savez_compressed blow up inside run_single.
    def _raise(*_args, **_kwargs):
        raise OSError('simulated disk full')

    monkeypatch.setattr(rg.np, 'savez_compressed', _raise)

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
    """Create a tiny grid_summary.csv + <label>.npz set in gdir.

    Lets the plot tools run end-to-end with no solver and no real EOS data.
    """
    import csv

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
    for m in masses:
        np.savez_compressed(
            gdir / f'planet_mass={m}.npz',
            radii=np.linspace(0.0, R_earth * m**0.27, n),
            density=np.linspace(12000.0, 3000.0, n),
            pressure=np.linspace(3e11 * m, 1e5, n),
            temperature=np.linspace(6000.0 * m**0.2, 3000.0, n),
            gravity=np.linspace(0.0, 10.0 * m**0.5, n),
            mass_enclosed=np.linspace(0.0, M_earth * m, n),
            cmb_mass=np.array(0.325 * M_earth * m),
            core_mantle_mass=np.array(0.325 * M_earth * m),  # 2-layer
            converged=np.bool_(True),
            core_eos=np.str_('PALEOS:iron'),
            mantle_eos=np.str_(mantle_eos),
            ice_layer_eos=np.str_(''),
            rock_solidus_id=np.str_('Monteux16-solidus'),
            rock_liquidus_id=np.str_('Monteux16-liquidus-A-chondritic'),
        )
    return gdir


@pytest.fixture
def fake_grid_dir(tmp_path):
    return _write_synthetic_grid(tmp_path / 'grid')


@pytest.fixture
def fake_grid_dir_external_curves(tmp_path):
    """Synthetic grid whose mantle EOS triggers the external-curves overlay."""
    return _write_synthetic_grid(tmp_path / 'grid_wb', mantle_eos='WolfBower2018:MgSiO3')


@pytest.mark.unit
def test_plot_grid_profiles_end_to_end(fake_grid_dir, tmp_path):
    from src.tools.plot_grid_profiles import plot_grid_profiles

    out = tmp_path / 'profiles.png'
    result = plot_grid_profiles(str(fake_grid_dir), out=str(out))
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 1000  # non-empty image


@pytest.mark.unit
def test_plot_grid_profiles_log_pressure_masks_zero(fake_grid_dir, tmp_path):
    """log_pressure=True exercises the P>0 mask branch (fix for bot #5)."""
    from src.tools.plot_grid_profiles import plot_grid_profiles

    # Overwrite one of the .npz files so its surface shell has P == 0.
    label = 'planet_mass=1.0'
    old = dict(np.load(fake_grid_dir / f'{label}.npz'))
    old['pressure'][-1] = 0.0
    np.savez_compressed(fake_grid_dir / f'{label}.npz', **old)

    out = tmp_path / 'profiles_log.png'
    result = plot_grid_profiles(str(fake_grid_dir), out=str(out), log_pressure=True)
    assert os.path.isfile(result)


@pytest.mark.unit
def test_plot_grid_pt_end_to_end_no_overlay(fake_grid_dir, tmp_path):
    """PALEOS:MgSiO3 mantle: overlay is auto-suppressed with a note."""
    from src.tools.plot_grid_pt import plot_grid_pt

    out = tmp_path / 'pt.png'
    result = plot_grid_pt(str(fake_grid_dir), out=str(out))
    assert os.path.isfile(result)


@pytest.mark.unit
def test_plot_grid_pt_no_melting_curves_flag(fake_grid_dir, tmp_path):
    """show_melting_curves=False short-circuits the overlay branch."""
    from src.tools.plot_grid_pt import plot_grid_pt

    out = tmp_path / 'pt_bare.png'
    result = plot_grid_pt(str(fake_grid_dir), out=str(out), show_melting_curves=False)
    assert os.path.isfile(result)


@pytest.mark.unit
def test_plot_grid_pt_overlay_from_metadata(fake_grid_dir_external_curves, tmp_path):
    """WolfBower2018:MgSiO3 mantle: overlay is loaded from the stored
    rock_solidus_id / rock_liquidus_id and drawn."""
    from src.tools.plot_grid_pt import plot_grid_pt

    out = tmp_path / 'pt_wb.png'
    result = plot_grid_pt(str(fake_grid_dir_external_curves), out=str(out))
    assert os.path.isfile(result)


@pytest.mark.unit
def test_plot_grid_pt_forced_overlay(fake_grid_dir, tmp_path):
    """Explicit --solidus / --liquidus override the auto-suppression."""
    from src.tools.plot_grid_pt import plot_grid_pt

    out = tmp_path / 'pt_forced.png'
    result = plot_grid_pt(
        str(fake_grid_dir),
        out=str(out),
        solidus='Stixrude14-solidus',
        liquidus='Stixrude14-liquidus',
    )
    assert os.path.isfile(result)


@pytest.mark.unit
def test_plot_grid_pt_linear_pressure(fake_grid_dir, tmp_path):
    from src.tools.plot_grid_pt import plot_grid_pt

    out = tmp_path / 'pt_linear.png'
    result = plot_grid_pt(str(fake_grid_dir), out=str(out), linear_pressure=True)
    assert os.path.isfile(result)


@pytest.mark.unit
def test_plot_grid_composition_end_to_end(fake_grid_dir, tmp_path):
    from src.tools.plot_grid_composition import plot_grid_composition

    out = tmp_path / 'comp.png'
    result = plot_grid_composition(str(fake_grid_dir), out=str(out))
    assert os.path.isfile(result)
    assert os.path.getsize(result) > 1000


@pytest.mark.unit
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

    from src.tools.plot_grid_composition import plot_grid_composition
    from src.tools.plot_grid_profiles import plot_grid_profiles
    from src.tools.plot_grid_pt import plot_grid_pt

    assert os.path.isfile(plot_grid_profiles(str(fake_grid_dir), out=str(tmp_path / 'p.png')))
    assert os.path.isfile(plot_grid_pt(str(fake_grid_dir), out=str(tmp_path / 'pt.png')))
    assert os.path.isfile(
        plot_grid_composition(str(fake_grid_dir), out=str(tmp_path / 'c.png'))
    )


@pytest.mark.unit
def test_plot_grid_profiles_default_output_path(fake_grid_dir):
    """When out is None the tool writes <grid_dir>/profiles_vs_radius.pdf."""
    from src.tools.plot_grid_profiles import plot_grid_profiles

    result = plot_grid_profiles(str(fake_grid_dir))
    assert os.path.basename(result) == 'profiles_vs_radius.pdf'
    assert os.path.isfile(result)


# ---------------------------------------------------------------------------
# Helper-function tests
# ---------------------------------------------------------------------------
@pytest.mark.unit
def test_resolve_grid_dir_accepts_directory(fake_grid_dir):
    from src.tools.plot_grid_profiles import _resolve_grid_dir

    assert _resolve_grid_dir(str(fake_grid_dir)) == str(fake_grid_dir)


@pytest.mark.unit
def test_resolve_grid_dir_accepts_csv_path(fake_grid_dir):
    from src.tools.plot_grid_profiles import _resolve_grid_dir

    csv_path = str(fake_grid_dir / 'grid_summary.csv')
    resolved = _resolve_grid_dir(csv_path)
    assert os.path.basename(resolved) == os.path.basename(str(fake_grid_dir))


@pytest.mark.unit
def test_resolve_grid_dir_rejects_other_files(tmp_path):
    from src.tools.plot_grid_profiles import _resolve_grid_dir

    bogus = tmp_path / 'note.txt'
    bogus.write_text('hi')
    with pytest.raises(ValueError):
        _resolve_grid_dir(str(bogus))


@pytest.mark.unit
def test_load_summary_missing_csv(tmp_path):
    from src.tools.plot_grid_profiles import _load_summary

    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        _load_summary(str(empty))


@pytest.mark.unit
def test_try_float_behaviour():
    from src.tools.plot_grid_profiles import _try_float

    assert _try_float('1.5') == 1.5
    assert _try_float('0') == 0.0
    assert _try_float('') is None
    assert _try_float('abc') is None
    assert _try_float(None) is None


@pytest.mark.unit
def test_detect_sweep_params_strips_fixed_columns():
    from src.tools.plot_grid_profiles import _detect_sweep_params

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


@pytest.mark.unit
def test_choose_colour_param_rejects_invalid_override():
    from src.tools.plot_grid_profiles import _choose_colour_param

    with pytest.raises(ValueError):
        _choose_colour_param(['planet_mass'], [{'planet_mass': '1.0'}], 'nope')


@pytest.mark.unit
def test_choose_colour_param_prefers_numeric():
    from src.tools.plot_grid_profiles import _choose_colour_param

    rows = [{'mantle': 'PALEOS:MgSiO3', 'planet_mass': '1.0'}]
    assert _choose_colour_param(['mantle', 'planet_mass'], rows) == 'planet_mass'


@pytest.mark.unit
def test_read_str_handles_missing_and_numpy_scalars():
    from src.tools.plot_grid_pt import _read_str

    data = {'a': np.str_('hello'), 'b': 'raw'}
    assert _read_str(data, 'a') == 'hello'
    assert _read_str(data, 'b') == 'raw'
    assert _read_str(data, 'missing') == ''
