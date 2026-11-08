"""The optional ``[Output] output_dir``: the one folder of the standalone output files.

``post_processing``, the two plot writers and the CLI log write into ``output_dir`` when it
is given; without it a library call keeps ``<ZALMOXIS_ROOT>/output`` and the CLI sets that
folder itself.
"""

from __future__ import annotations

import logging
import os
import runpy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tests.test_config_validators_extra import _make_minimal_config
from tests.test_output import _base_config, fake_solver, fake_solver_results  # noqa: F401
from zalmoxis import output as output_mod
from zalmoxis.config import validate_config
from zalmoxis.plots import plot_phase_vs_radius, plot_profiles

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


@pytest.fixture
def fake_root(monkeypatch, tmp_path):
    """A ZALMOXIS_ROOT with an empty output/ for the writers that fall back to it."""
    root = tmp_path / 'root'
    (root / 'output').mkdir(parents=True)
    for mod in (output_mod, plot_profiles, plot_phase_vs_radius):
        monkeypatch.setattr(mod, 'get_zalmoxis_root', lambda: str(root))
    return root


def _plotting_config(**overrides):
    """A config that writes data, plots and the mantle P-T plot (a melting-curve mantle)."""
    cfg = _base_config(plotting_enabled=True)
    cfg['layer_eos_config'] = {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'}
    cfg.update(overrides)
    return cfg


@pytest.mark.usefixtures('fake_solver')
def test_post_processing_writes_every_file_into_output_dir(fake_root, tmp_path, monkeypatch):
    """Profile, summary, profile plot and P-T plot go to output_dir; root/output and cwd stay empty."""
    run, cwd = tmp_path / 'run', tmp_path / 'cwd'
    run.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    output_mod.post_processing(_plotting_config(output_dir=str(run)), id_mass='1.0')
    assert sorted(p.name for p in run.iterdir()) == [
        'calculated_planet_mass_radius.txt',
        'mantle_PT_profile1.0.pdf',
        'planet_profile1.0.png',
        'planet_profile1.0.txt',
    ]
    assert list((fake_root / 'output').iterdir()) == []
    assert list(cwd.iterdir()) == []


@pytest.mark.usefixtures('fake_solver')
def test_post_processing_without_output_dir_keeps_the_root_output(fake_root):
    """A library call without the key writes the same files into <root>/output."""
    output_mod.post_processing(_plotting_config())
    assert sorted(p.name for p in (fake_root / 'output').iterdir()) == [
        'calculated_planet_mass_radius.txt',
        'mantle_PT_profile.pdf',
        'planet_profile.png',
        'planet_profile.txt',
    ]
    assert 'output_dir' not in _plotting_config()


@pytest.mark.usefixtures('fake_solver')
def test_summary_appends_in_output_dir(fake_root, tmp_path):
    """Two calls append 2 rows under one header in <output_dir>/calculated_planet_mass_radius.txt."""
    run = tmp_path / 'run'
    run.mkdir()
    for _ in range(2):
        output_mod.post_processing(_base_config(output_dir=run))
    lines = (run / 'calculated_planet_mass_radius.txt').read_text().splitlines()
    assert lines[0] == 'Calculated Mass (kg)\tCalculated Radius (m)'
    assert len(lines) == 3
    assert not (fake_root / 'output' / 'calculated_planet_mass_radius.txt').exists()


@pytest.mark.parametrize(
    'id_mass, name', [(None, 'mantle_PT_profile.pdf'), ('7', 'mantle_PT_profile7.pdf')]
)
def test_mantle_pt_plot_takes_the_id_mass_and_the_folder(fake_root, tmp_path, id_mass, name):
    """The P-T plot is named like the profiles (id_mass suffix) and written into the given folder."""
    r = np.linspace(3.5e6, 6.4e6, 20)
    phases = ['solid_mantle'] * 10 + ['melted_mantle'] * 10
    plot_phase_vs_radius.plot_PT_with_phases(
        np.linspace(1.3e11, 1e5, 20),
        np.linspace(4000, 1500, 20),
        r,
        phases,
        3.5e6,
        id_mass=id_mass,
        output_dir=tmp_path,
    )
    assert (tmp_path / name).is_file()
    assert list((fake_root / 'output').iterdir()) == []


class TestOutputDirValidation:
    """The optional output_dir: a folder that exists, or absent."""

    @pytest.mark.parametrize(
        'value', [123, True, ['x'], b'/tmp'], ids=['int', 'bool', 'list', 'bytes']
    )
    def test_wrong_type_is_refused_by_name(self, value):
        """A value that is not a path string raises ValueError naming the key and the value."""
        with pytest.raises(ValueError, match='output_dir') as err:
            validate_config(_make_minimal_config(output_dir=value))
        assert repr(value) in str(err.value)

    def test_empty_values_are_refused_by_name(self):
        """'' and Path('') (which is '.') are refused as empty; a blank name is no folder."""
        for value in ('', Path('')):
            with pytest.raises(ValueError, match='output_dir must not be empty'):
                validate_config(_make_minimal_config(output_dir=value))
        with pytest.raises(ValueError, match='is not an existing folder'):
            validate_config(_make_minimal_config(output_dir=' '))

    def test_missing_folder_is_refused_and_not_created(self, tmp_path):
        """A path that is not an existing folder raises ValueError naming it; nothing is created."""
        missing = tmp_path / 'not_there'
        with pytest.raises(ValueError, match='output_dir.*not_there'):
            validate_config(_make_minimal_config(output_dir=str(missing)))
        assert not missing.exists()

    def test_existing_folder_and_absent_key_pass(self, tmp_path):
        """An existing folder (str or Path), '.' and a config without the key validate."""
        for value in (str(tmp_path), tmp_path, '.'):
            assert validate_config(_make_minimal_config(output_dir=value)) is None
        assert validate_config(_make_minimal_config()) is None


@settings(max_examples=60, deadline=None)
@given(
    st.one_of(
        st.none(),
        st.booleans(),
        st.integers(),
        st.floats(allow_nan=True, allow_infinity=True),
        st.lists(st.text(max_size=3), max_size=2),
        st.dictionaries(st.text(max_size=2), st.integers(), max_size=1),
        st.binary(max_size=4),
    )
)
def test_any_non_path_output_dir_is_refused(value):
    """None, bool, int, float (0, negative, NaN, inf) and containers raise ValueError."""
    with pytest.raises(ValueError, match='output_dir must be a folder path'):
        validate_config(_make_minimal_config(output_dir=value))


@settings(
    max_examples=60, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(st.text(alphabet='abcXYZ019_- .', min_size=1, max_size=12))
def test_any_output_dir_that_is_not_a_folder_is_refused(tmp_path, name):
    """A name under a missing folder, and a regular file, raise ValueError; nothing is created."""
    with pytest.raises(ValueError, match='is not an existing folder'):
        validate_config(_make_minimal_config(output_dir=os.path.join(tmp_path, 'absent', name)))
    afile = tmp_path / 'a_file.txt'
    afile.write_text('x')
    with pytest.raises(ValueError, match='is not an existing folder'):
        validate_config(_make_minimal_config(output_dir=str(afile)))
    assert sorted(p.name for p in tmp_path.iterdir()) == ['a_file.txt']


def test_output_dir_set_in_the_toml_reaches_the_config(tmp_path):
    """A folder named in [Output] is in the returned dict; without the line the key is absent."""
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import load_zalmoxis_config

    text = open(os.path.join(get_zalmoxis_root(), 'input', 'default.toml')).read()
    assert text.count('[Output]\n') == 1
    run = tmp_path / 'run'
    run.mkdir()
    for line, name in ((f"output_dir = '{run}'", 'with.toml'), ('', 'without.toml')):
        (tmp_path / name).write_text(text.replace('[Output]\n', f'[Output]\n{line}\n'))
    assert load_zalmoxis_config(str(tmp_path / 'with.toml'))['output_dir'] == str(run)
    assert 'output_dir' not in load_zalmoxis_config(str(tmp_path / 'without.toml'))


@pytest.fixture
def reset_logging():
    """Restore the root logger handlers and level after a runpy CLI call."""
    logger = logging.getLogger()
    saved, level = list(logger.handlers), logger.level
    yield
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()
    for h in saved:
        logger.addHandler(h)
    logger.setLevel(level)


def _run_cli(root, config):
    """Run ``python -m zalmoxis`` with a loader that logs one line and returns ``config``."""

    def load():
        logging.getLogger('zalmoxis.config').info('loading the test config')
        return config

    with patch('zalmoxis.get_zalmoxis_root', return_value=str(root)):
        with patch('zalmoxis.config.load_zalmoxis_config', side_effect=load):
            with patch('zalmoxis.output.post_processing') as post:
                runpy.run_module('zalmoxis', run_name='__main__')
    return post


@pytest.mark.parametrize('given_dir', [True, False], ids=['output_dir-given', 'root-default'])
def test_cli_log_and_post_processing_follow_the_output_folder(
    tmp_path, reset_logging, given_dir
):
    """The CLI log, with the lines of the config load, goes to output_dir or to <root>/output."""
    root, run = tmp_path / 'root', tmp_path / 'run'
    (root / 'output').mkdir(parents=True)
    run.mkdir()
    config = {'output_dir': str(run)} if given_dir else {}
    post = _run_cli(root, config)
    folder = run if given_dir else root / 'output'
    assert post.call_args.args[0]['output_dir'] == str(folder)
    assert 'loading the test config' in (folder / 'zalmoxis.log').read_text()
    other = root / 'output' if given_dir else run
    assert list(other.iterdir()) == []
