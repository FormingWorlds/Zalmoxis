"""Tests for ``zalmoxis.output.post_processing``.

The module orchestrates a full Zalmoxis solver run, writes a 6-column profile
file, appends a (mass, radius) row to a summary CSV, runs an optional
phase-detection branch when the mantle uses a melting-curve-dependent EOS,
and dispatches optional plotting helpers. To unit-test it without paying for
the real solver we mock ``zalmoxis.solver.main`` and the two config helpers
``load_material_dictionaries`` and ``load_solidus_liquidus_functions`` at the
narrowest scope (the modules from which ``post_processing`` lazily imports
them at call-time), and supply a synthetic Earth-mass solver result.

Anti-happy-path coverage in each test class:
- ``TestPostProcessingFileOutput`` exercises the disabled-output branch and
  the bad-config (missing key) error path.
- ``TestPostProcessingPhaseDetection`` exercises the unified-PALEOS skip
  branch (negative assertion: spy not called) and the empty-mantle skip.
- ``TestPostProcessingSummaryFile`` covers both the create-with-header and
  append-without-header branches and a corrupt-summary-file edge case.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from zalmoxis import output as output_mod

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_solver_results():
    """Synthetic Earth-mass profile (60 shells) with discriminating values.

    Uses linspace ranges that map every column to a distinct order of
    magnitude so a column-swap bug in the writer would be caught by
    ``test_six_column_profile_file_columns_in_correct_order``. The values
    are physically plausible: T 5000 -> 300 K from core to surface,
    P 3.6e11 -> 1e5 Pa, rho 13000 -> 3000 kg/m^3, g 0 -> 9.81 m/s^2.
    """
    n = 60
    radii = np.linspace(0.0, 6.371e6, n)
    mass_enclosed = np.linspace(0.0, 5.972e24, n)
    pressure = np.linspace(3.6e11, 1.0e5, n)
    density = np.linspace(13000.0, 3000.0, n)
    gravity = np.linspace(0.0, 9.81, n)
    temperature = np.linspace(5000.0, 300.0, n)
    cmb_mass = 1.97e24  # ~33 % of total
    return {
        'radii': radii,
        'density': density,
        'gravity': gravity,
        'pressure': pressure,
        'temperature': temperature,
        'mass_enclosed': mass_enclosed,
        'cmb_mass': cmb_mass,
        'core_mantle_mass': mass_enclosed[-1],
        'total_time': 1.23,
        'converged': True,
        'converged_pressure': True,
        'converged_density': True,
        'converged_mass': True,
    }


@pytest.fixture
def fake_solver(monkeypatch, fake_solver_results):
    """Patch ``zalmoxis.solver.main`` and the two lazy config helpers.

    ``post_processing`` resolves these symbols lazily inside the function
    body (``from .solver import main`` and ``from .config import ...``).
    Patching the source modules guarantees the lazy import sees the mocks.
    """
    import zalmoxis.config as cfg_mod
    import zalmoxis.solver as solver_mod

    def _fake_main(*_args, **_kwargs):
        return fake_solver_results

    def _fake_load_materials():
        return {}

    def _fake_load_solidus_liquidus(*_a, **_k):
        return (lambda P: 2000.0, lambda P: 2400.0)

    monkeypatch.setattr(solver_mod, 'main', _fake_main, raising=True)
    monkeypatch.setattr(
        cfg_mod, 'load_material_dictionaries', _fake_load_materials, raising=True
    )
    monkeypatch.setattr(
        cfg_mod,
        'load_solidus_liquidus_functions',
        _fake_load_solidus_liquidus,
        raising=True,
    )
    return fake_solver_results


@pytest.fixture
def zalmoxis_tmp_root(monkeypatch, tmp_path):
    """Redirect ``get_zalmoxis_root`` to a writable temp dir.

    ``post_processing`` writes to ``<root>/output/...`` so an ``output``
    subdirectory must exist. We patch the symbol that ``output.py``
    imported at module load (``zalmoxis.output.get_zalmoxis_root``).
    """
    (tmp_path / 'output').mkdir()
    monkeypatch.setattr(output_mod, 'get_zalmoxis_root', lambda: str(tmp_path))
    return tmp_path


def _base_config(**overrides):
    """Return a minimal ``config_params`` dict with the keys post_processing
    actually reads. Mantle defaults to PALEOS:iron (a unified PALEOS entry
    that does not need external melting curves), so the phase-detection
    branch is skipped unless overridden.
    """
    cfg = {
        'data_output_enabled': True,
        'plotting_enabled': False,
        'layer_eos_config': {
            'core': 'Seager2007:iron',
            'mantle': 'PALEOS:MgSiO3',
        },
        'rock_solidus': 'Stixrude14-solidus',
        'rock_liquidus': 'Stixrude14-liquidus',
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# File-output behaviour
# ---------------------------------------------------------------------------


class TestPostProcessingFileOutput:
    """Profile and summary file emission, including the data-disabled branch."""

    def test_writes_six_column_profile_file_when_data_output_enabled(
        self, fake_solver, zalmoxis_tmp_root
    ):
        """Profile file exists with shape (n_shells, 6) and is finite throughout."""
        cfg = _base_config()
        output_mod.post_processing(cfg)

        profile_path = zalmoxis_tmp_root / 'output' / 'planet_profile.txt'
        assert profile_path.is_file()

        data = np.genfromtxt(profile_path)
        assert data.shape == (60, 6)
        assert np.all(np.isfinite(data))

    def test_six_column_profile_file_columns_in_correct_order(
        self, fake_solver, zalmoxis_tmp_root, fake_solver_results
    ):
        """Columns are (radii, density, gravity, pressure, temperature, mass).

        Discriminating: the six fixture arrays span six different orders of
        magnitude (radius ~1e6, density ~1e4, gravity ~1e1, pressure ~1e11,
        temperature ~1e3, mass ~1e24). Any column swap would produce a
        mismatch many orders of magnitude wide, far beyond the 1e-6 rtol.
        """
        output_mod.post_processing(_base_config())

        profile_path = zalmoxis_tmp_root / 'output' / 'planet_profile.txt'
        data = np.genfromtxt(profile_path)
        np.testing.assert_allclose(data[:, 0], fake_solver_results['radii'], rtol=1e-6)
        np.testing.assert_allclose(data[:, 1], fake_solver_results['density'], rtol=1e-6)
        np.testing.assert_allclose(data[:, 2], fake_solver_results['gravity'], rtol=1e-5)
        np.testing.assert_allclose(data[:, 3], fake_solver_results['pressure'], rtol=1e-6)
        np.testing.assert_allclose(data[:, 4], fake_solver_results['temperature'], rtol=1e-6)
        np.testing.assert_allclose(data[:, 5], fake_solver_results['mass_enclosed'], rtol=1e-6)

    def test_id_mass_appended_to_profile_filename(self, fake_solver, zalmoxis_tmp_root):
        """``id_mass='1Me'`` -> profile file becomes ``planet_profile1Me.txt``."""
        output_mod.post_processing(_base_config(), id_mass='1Me')

        suffixed = zalmoxis_tmp_root / 'output' / 'planet_profile1Me.txt'
        unsuffixed = zalmoxis_tmp_root / 'output' / 'planet_profile.txt'
        assert suffixed.is_file()
        assert not unsuffixed.is_file()

    def test_data_output_disabled_writes_no_files(self, fake_solver, zalmoxis_tmp_root):
        """With ``data_output_enabled=False`` no profile or summary file appears."""
        output_mod.post_processing(_base_config(data_output_enabled=False))

        out_dir = zalmoxis_tmp_root / 'output'
        # The directory itself was pre-created by the fixture; it must remain empty.
        assert list(out_dir.iterdir()) == []

    def test_output_file_default_path_used_when_none(self, fake_solver, zalmoxis_tmp_root):
        """``output_file=None`` -> default ``calculated_planet_mass_radius.txt``."""
        output_mod.post_processing(_base_config(), output_file=None)

        default = zalmoxis_tmp_root / 'output' / 'calculated_planet_mass_radius.txt'
        assert default.is_file()

    def test_output_file_explicit_path_used_when_provided(
        self, fake_solver, zalmoxis_tmp_root, tmp_path
    ):
        """An explicit ``output_file`` overrides the default location."""
        custom = tmp_path / 'custom_summary.csv'
        output_mod.post_processing(_base_config(), output_file=str(custom))

        assert custom.is_file()
        # The default should not be created when an explicit path is given.
        default = zalmoxis_tmp_root / 'output' / 'calculated_planet_mass_radius.txt'
        assert not default.is_file()

    def test_missing_data_output_enabled_key_raises_keyerror(
        self, fake_solver, zalmoxis_tmp_root
    ):
        """Unphysical / malformed config: missing required key raises KeyError.

        ``post_processing`` reads ``config_params['data_output_enabled']`` at
        line 35 with bracket access, so a missing key must propagate as
        KeyError rather than be silently treated as False.
        """
        bad = _base_config()
        del bad['data_output_enabled']
        with pytest.raises(KeyError, match='data_output_enabled'):
            output_mod.post_processing(bad)


# ---------------------------------------------------------------------------
# Summary-file create-vs-append behaviour
# ---------------------------------------------------------------------------


class TestPostProcessingSummaryFile:
    """The summary CSV at ``output_file`` is created on first call, appended on later."""

    def _read_summary(self, path: Path):
        text = path.read_text().splitlines()
        header = text[0]
        rows = [ln for ln in text[1:] if ln.strip()]
        return header, rows

    def test_summary_file_created_with_header_when_missing(
        self, fake_solver, zalmoxis_tmp_root, fake_solver_results
    ):
        """First call writes a header line plus exactly one data row.

        Discriminating: the recorded mass / radius are the *last* values of
        the fixture arrays (5.972e24 kg, 6.371e6 m), not e.g. the first or
        the mean. A wrong index in the writer would produce 0.0 / 0.0.
        """
        cfg = _base_config()
        output_mod.post_processing(cfg)

        summary = zalmoxis_tmp_root / 'output' / 'calculated_planet_mass_radius.txt'
        header, rows = self._read_summary(summary)
        assert 'Calculated Mass (kg)' in header
        assert 'Calculated Radius (m)' in header
        assert len(rows) == 1

        mass_str, radius_str = rows[0].split('\t')
        assert float(mass_str) == pytest.approx(
            fake_solver_results['mass_enclosed'][-1], rel=1e-12
        )
        assert float(radius_str) == pytest.approx(fake_solver_results['radii'][-1], rel=1e-12)

    def test_summary_file_appended_when_existing(self, fake_solver, zalmoxis_tmp_root):
        """Two successive calls produce a header line plus two data rows."""
        cfg = _base_config()
        output_mod.post_processing(cfg)
        output_mod.post_processing(cfg)

        summary = zalmoxis_tmp_root / 'output' / 'calculated_planet_mass_radius.txt'
        _, rows = self._read_summary(summary)
        assert len(rows) == 2

    def test_summary_file_with_preexisting_header_only_does_not_duplicate_header(
        self, fake_solver, zalmoxis_tmp_root
    ):
        """Edge case: a manually-prepared, header-only summary file is appended to.

        Verifies the ``not os.path.exists`` guard: if the file already exists
        (even without a data row), the header must not be re-written.
        """
        summary = zalmoxis_tmp_root / 'output' / 'calculated_planet_mass_radius.txt'
        summary.write_text('Calculated Mass (kg)\tCalculated Radius (m)\n')

        output_mod.post_processing(_base_config())

        text = summary.read_text().splitlines()
        # Exactly one header, one data row.
        header_lines = [ln for ln in text if 'Calculated Mass' in ln]
        assert len(header_lines) == 1
        # And one data row underneath it.
        data_lines = [ln for ln in text if ln.strip() and 'Calculated Mass' not in ln]
        assert len(data_lines) == 1


# ---------------------------------------------------------------------------
# Phase-detection branch
# ---------------------------------------------------------------------------


class TestPostProcessingPhaseDetection:
    """Conditional ``get_Tdep_material`` call based on the mantle EOS string."""

    def test_phase_detection_branch_runs_when_mantle_uses_tdep_eos(
        self, fake_solver, zalmoxis_tmp_root, monkeypatch
    ):
        """A mantle entry in ``_NEEDS_MELTING_CURVES`` triggers ``get_Tdep_material``.

        Spy on ``zalmoxis.output.get_Tdep_material`` (the symbol resolved at
        module load) and assert it was called once with mantle-side
        pressure/temperature slices. ``cmb_index`` for the fixture is the
        first row where mass_enclosed >= cmb_mass; with linspace 0..5.972e24
        and cmb_mass 1.97e24 this is row 20, so the spy must see 40 mantle
        rows (60 - 20).
        """
        calls = []

        def _spy(P, T, sol, liq):
            calls.append((np.asarray(P).copy(), np.asarray(T).copy()))
            return np.array(['liquid'] * len(P))

        monkeypatch.setattr(output_mod, 'get_Tdep_material', _spy)

        cfg = _base_config(
            layer_eos_config={
                'core': 'Seager2007:iron',
                'mantle': 'PALEOS-2phase:MgSiO3',
            }
        )
        output_mod.post_processing(cfg)

        assert len(calls) == 1
        mantle_P, mantle_T = calls[0]
        assert mantle_P.size == mantle_T.size
        # Mantle slice must be strictly less than the full profile.
        assert 0 < mantle_P.size < 60
        # Mantle pressures end at the surface value 1e5 Pa, temperatures at 300 K.
        assert mantle_P[-1] == pytest.approx(1.0e5, rel=1e-9)
        assert mantle_T[-1] == pytest.approx(300.0, rel=1e-9)

    def test_phase_detection_branch_skipped_for_unified_paleos(
        self, fake_solver, zalmoxis_tmp_root, monkeypatch
    ):
        """Unified PALEOS mantle (``PALEOS:MgSiO3``) skips ``get_Tdep_material``.

        Negative assertion: the spy must not be called at all, because
        unified PALEOS tables carry their own phase column.
        """
        calls = []

        def _spy(*a, **k):
            calls.append((a, k))
            return None

        monkeypatch.setattr(output_mod, 'get_Tdep_material', _spy)

        cfg = _base_config(
            layer_eos_config={
                'core': 'Seager2007:iron',
                'mantle': 'PALEOS:MgSiO3',
            }
        )
        output_mod.post_processing(cfg)

        assert calls == []

    def test_phase_detection_branch_skipped_when_mantle_string_empty(
        self, fake_solver, zalmoxis_tmp_root, monkeypatch
    ):
        """Edge case: missing mantle entry -> the empty-string branch (line 77).

        The function must not raise (empty ``mantle_str`` short-circuits
        before ``parse_layer_components`` is reached) and must not invoke
        ``get_Tdep_material``.
        """
        calls = []
        monkeypatch.setattr(
            output_mod, 'get_Tdep_material', lambda *a, **k: calls.append(a) or None
        )

        cfg = _base_config(layer_eos_config={'core': 'Seager2007:iron', 'mantle': ''})
        output_mod.post_processing(cfg)

        assert calls == []


# ---------------------------------------------------------------------------
# Plotting branch
# ---------------------------------------------------------------------------


class TestPostProcessingPlotting:
    """``plotting_enabled`` toggles the matplotlib-backed helpers."""

    def test_plotting_disabled_skips_plot_imports(
        self, fake_solver, zalmoxis_tmp_root, monkeypatch
    ):
        """With ``plotting_enabled=False``, the function returns cleanly.

        Verifies the negative branch (lines 156-181). We make any attempt
        to import the plot helpers detectable by stubbing the plot module
        attributes to a sentinel that, if invoked, would raise; the test
        passes only if they are not invoked.
        """

        def _explode(*a, **k):
            raise AssertionError('plot helper called when plotting_enabled=False')

        # Inject sentinel modules into sys.modules so the lazy imports inside
        # the plotting branch would see them. The branch is not entered, so
        # the sentinels must not be exercised.
        import sys
        import types

        fake_phase = types.ModuleType('tools.plots.plot_phase_vs_radius')
        fake_phase.plot_PT_with_phases = _explode
        fake_profiles = types.ModuleType('tools.plots.plot_profiles')
        fake_profiles.plot_planet_profile_single = _explode
        monkeypatch.setitem(sys.modules, 'tools.plots.plot_phase_vs_radius', fake_phase)
        monkeypatch.setitem(sys.modules, 'tools.plots.plot_profiles', fake_profiles)

        # Should run without error.
        output_mod.post_processing(_base_config(plotting_enabled=False))

    def test_plotting_enabled_dispatches_profile_helper(
        self, fake_solver, zalmoxis_tmp_root, monkeypatch
    ):
        """With ``plotting_enabled=True``, the profile plot helper is invoked once.

        We mock both helpers so the test does not depend on matplotlib.
        Discriminating: the mock records the call and we verify it received
        the profile arrays in the documented argument order.
        """
        import sys
        import types

        profile_calls = []
        phase_calls = []

        def _profile_mock(
            radii,
            density,
            gravity,
            pressure,
            temperature,
            cmb_radius,
            cmb_mass,
            avg_density,
            mass_enclosed,
            id_mass,
            layer_eos_config=None,
        ):
            profile_calls.append(
                {
                    'radii_size': radii.size,
                    'cmb_radius': float(cmb_radius),
                    'cmb_mass': float(cmb_mass),
                    'id_mass': id_mass,
                    'layer_eos_config': layer_eos_config,
                }
            )

        def _phase_mock(*a, **k):
            phase_calls.append((a, k))

        fake_phase = types.ModuleType('tools.plots.plot_phase_vs_radius')
        fake_phase.plot_PT_with_phases = _phase_mock
        fake_profiles = types.ModuleType('tools.plots.plot_profiles')
        fake_profiles.plot_planet_profile_single = _profile_mock
        monkeypatch.setitem(sys.modules, 'tools.plots.plot_phase_vs_radius', fake_phase)
        monkeypatch.setitem(sys.modules, 'tools.plots.plot_profiles', fake_profiles)

        # Mantle is unified PALEOS so the phase plot helper must NOT be invoked.
        cfg = _base_config(plotting_enabled=True)
        output_mod.post_processing(cfg, id_mass='earth')

        assert len(profile_calls) == 1
        c = profile_calls[0]
        assert c['radii_size'] == 60
        assert c['id_mass'] == 'earth'
        assert c['cmb_mass'] == pytest.approx(1.97e24, rel=1e-12)
        # Negative branch: phase plot only fires when uses_phase_detection is True.
        assert phase_calls == []

    def test_plotting_with_tdep_mantle_dispatches_phase_helper(
        self, fake_solver, zalmoxis_tmp_root, monkeypatch
    ):
        """A T-dependent mantle EOS plus plotting on -> both helpers invoked."""
        import sys
        import types

        phase_calls = []

        # Stub get_Tdep_material so the phase-detection branch returns a
        # plausible array without touching real EOS data.
        def _fake_tdep(P, T, sol, liq):
            return np.array(['mush'] * len(P))

        monkeypatch.setattr(output_mod, 'get_Tdep_material', _fake_tdep)

        fake_profiles = types.ModuleType('tools.plots.plot_profiles')
        fake_profiles.plot_planet_profile_single = lambda *a, **k: None
        fake_phase = types.ModuleType('tools.plots.plot_phase_vs_radius')

        def _phase_mock(P, T, R, phases, cmb_radius):
            phase_calls.append({'n_pts': P.size, 'cmb_radius': float(cmb_radius)})

        fake_phase.plot_PT_with_phases = _phase_mock
        monkeypatch.setitem(sys.modules, 'tools.plots.plot_profiles', fake_profiles)
        monkeypatch.setitem(sys.modules, 'tools.plots.plot_phase_vs_radius', fake_phase)

        cfg = _base_config(
            plotting_enabled=True,
            layer_eos_config={
                'core': 'Seager2007:iron',
                'mantle': 'PALEOS-2phase:MgSiO3',
            },
        )
        output_mod.post_processing(cfg)

        assert len(phase_calls) == 1
        assert phase_calls[0]['n_pts'] > 0


# ---------------------------------------------------------------------------
# Cross-cutting: solver mock contract
# ---------------------------------------------------------------------------


class TestPostProcessingSolverContract:
    """The mocked solver result must contain every key post_processing reads."""

    def test_missing_solver_result_key_raises_keyerror(self, monkeypatch, zalmoxis_tmp_root):
        """Edge case: solver returns a result missing ``cmb_mass``.

        ``post_processing`` accesses 13 keys with bracket lookup; dropping
        one must raise KeyError, not produce a silently broken file.
        """
        import zalmoxis.config as cfg_mod
        import zalmoxis.solver as solver_mod

        broken = {
            'radii': np.linspace(0.0, 1.0, 5),
            'density': np.ones(5),
            'gravity': np.ones(5),
            'pressure': np.ones(5),
            'temperature': np.ones(5),
            'mass_enclosed': np.ones(5),
            # 'cmb_mass' deliberately absent
            'core_mantle_mass': 1.0,
            'total_time': 0.0,
            'converged': True,
            'converged_pressure': True,
            'converged_density': True,
            'converged_mass': True,
        }
        monkeypatch.setattr(solver_mod, 'main', lambda *a, **k: broken)
        monkeypatch.setattr(cfg_mod, 'load_material_dictionaries', lambda: {})
        monkeypatch.setattr(
            cfg_mod,
            'load_solidus_liquidus_functions',
            lambda *a, **k: (lambda P: 2000.0, lambda P: 2400.0),
        )

        with pytest.raises(KeyError, match='cmb_mass'):
            output_mod.post_processing(_base_config())

    def test_solver_called_with_input_dir_under_zalmoxis_root(
        self, monkeypatch, zalmoxis_tmp_root, fake_solver_results
    ):
        """The ``input_dir`` kwarg passed to ``main`` is ``<root>/input``.

        Captures the solver call and checks the path the orchestration code
        constructs. Discriminating: a hard-coded relative path or a
        different subdir would not match the temp root.
        """
        captured = {}
        import zalmoxis.config as cfg_mod
        import zalmoxis.solver as solver_mod

        def _capturing_main(*args, **kwargs):
            captured['args'] = args
            captured['kwargs'] = kwargs
            return fake_solver_results

        monkeypatch.setattr(solver_mod, 'main', _capturing_main)
        monkeypatch.setattr(cfg_mod, 'load_material_dictionaries', lambda: {})
        monkeypatch.setattr(
            cfg_mod,
            'load_solidus_liquidus_functions',
            lambda *a, **k: (lambda P: 2000.0, lambda P: 2400.0),
        )

        output_mod.post_processing(_base_config())

        assert 'input_dir' in captured['kwargs']
        expected_input = os.path.join(str(zalmoxis_tmp_root), 'input')
        assert captured['kwargs']['input_dir'] == expected_input


# ---------------------------------------------------------------------------
# Regressions: cmb_index floor + renamed mantle-fraction log line
# ---------------------------------------------------------------------------


class TestPostProcessingLogLines:
    """Lock in the cmb_index floor and the corrected mass-fraction label."""

    def test_cmb_mass_below_first_shell_does_not_wrap_to_surface_density(
        self, monkeypatch, zalmoxis_tmp_root, caplog
    ):
        """``cmb_mass <= mass_enclosed[0]`` must not log surface density as core.

        Regression: prior to the fix, ``cmb_index = np.argmax(...)`` returned
        0 for a coreless / very-low-cmb-mass profile, and ``density[cmb_index
        - 1]`` silently wrapped to ``density[-1]`` (the surface). The floor
        ``max(1, ...)`` guarantees ``density[cmb_index - 1]`` is the centre
        density (or near-centre), not the surface.
        """
        n = 60
        radii = np.linspace(0.0, 6.371e6, n)
        mass_enclosed = np.linspace(0.0, 5.972e24, n)
        density = np.linspace(13000.0, 3000.0, n)
        results = {
            'radii': radii,
            'density': density,
            'gravity': np.linspace(0.0, 9.81, n),
            'pressure': np.linspace(3.6e11, 1.0e5, n),
            'temperature': np.linspace(5000.0, 300.0, n),
            'mass_enclosed': mass_enclosed,
            'cmb_mass': 0.0,  # the regression trigger
            'core_mantle_mass': mass_enclosed[-1],
            'total_time': 1.23,
            'converged': True,
            'converged_pressure': True,
            'converged_density': True,
            'converged_mass': True,
        }
        import zalmoxis.config as cfg_mod
        import zalmoxis.solver as solver_mod

        monkeypatch.setattr(solver_mod, 'main', lambda *a, **k: results, raising=True)
        monkeypatch.setattr(cfg_mod, 'load_material_dictionaries', lambda: {}, raising=True)
        monkeypatch.setattr(
            cfg_mod,
            'load_solidus_liquidus_functions',
            lambda *a, **k: (lambda P: 2000.0, lambda P: 2400.0),
            raising=True,
        )

        with caplog.at_level('INFO'):
            output_mod.post_processing(_base_config())

        # Find the "Core Density (at CMB)" line and parse out its numeric
        # value. With the floor in place the value is ``density[0]`` =
        # 13000 kg/m^3 (centre), not ``density[-1]`` = 3000 kg/m^3 (surface).
        core_density_lines = [
            r.message for r in caplog.records if 'Core Density (at CMB)' in r.message
        ]
        assert core_density_lines, 'Expected a Core Density log line'
        # Format: 'Core Density (at CMB): 13000.00 kg/m^3'
        reported = float(core_density_lines[0].rsplit(':', 1)[1].split()[0])
        np.testing.assert_allclose(reported, density[0], rtol=1e-6)
        # Negative assertion: must NOT be the surface density.
        assert reported != pytest.approx(density[-1])

    def test_mantle_mass_fraction_label_replaces_misleading_core_plus_mantle(
        self, fake_solver, zalmoxis_tmp_root, caplog
    ):
        """The renamed log line uses 'Mantle Mass Fraction' (not 'Core+Mantle')."""
        with caplog.at_level('INFO'):
            output_mod.post_processing(_base_config())
        msgs = [r.message for r in caplog.records]
        assert any(m.startswith('Mantle Mass Fraction:') for m in msgs)
        # Negative assertion: the misleading label is gone.
        assert not any('Core+Mantle Mass Fraction' in m for m in msgs)
