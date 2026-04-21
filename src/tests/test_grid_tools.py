"""Unit tests for the grid-runner output contract and plot-helper logic.

These tests do not run the Zalmoxis solver. They exercise the pure helper
functions in ``src/tools/plot_grid_composition.py`` and
``src/tools/plot_grid_pt.py``, and the .npz serialisation contract in
``src/tools/run_grid.py`` by monkeypatching the solver call with a
synthetic ``model_results`` payload.

Covers:

- ``plot_grid_composition._layer_fractions``: 2-layer vs. 3-layer logic,
  including the Zalmoxis convention where ``core_mantle_mass == cmb_mass``
  for runs with ``mantle_mass_fraction = 0``.
- ``plot_grid_pt._mantle_uses_external_curves``: decision policy must
  stay in sync with ``_NEEDS_MELTING_CURVES`` in ``src/zalmoxis/zalmoxis.py``.
- ``run_grid.run_single``: when ``save_profiles`` is true the archive is
  written with exactly the expected 14 keys and the metadata the solver
  used (no external curves invented).
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
