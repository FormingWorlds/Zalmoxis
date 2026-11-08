"""The optional ``[Output] output_dir``: the one folder of the standalone output files.

``post_processing``, the two plot writers and the CLI log write into ``output_dir`` when it
is given; without it a library call keeps ``<ZALMOXIS_ROOT>/output`` and the CLI sets that
folder itself.
"""

from __future__ import annotations

import logging
import os
import re
import runpy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.test_config_validators_extra import _make_minimal_config
from tests.test_output import _base_config, fake_solver, fake_solver_results  # noqa: F401
from zalmoxis import output as output_mod
from zalmoxis.config import validate_config
from zalmoxis.plots import plot_phase_vs_radius, plot_profiles

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]
_ROOT_USER = getattr(os, 'geteuid', lambda: 1)() == 0


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


@pytest.mark.parametrize(
    'name, ok',
    [
        ('a\0b', False),
        ('trailing ', False),
        ('~', False),
        ('n' * 300, False),
        ('link', True),
        ('rel', True),
    ],
    ids=['NUL', 'trailing-space', 'tilde', 'long-name', 'symlink-to-folder', 'relative'],
)
def test_output_dir_names_at_the_edges(tmp_path, monkeypatch, name, ok):
    """NUL, a trailing space, '~' (not expanded) and a long name are no folder; a symlink and a relative path are."""
    (tmp_path / 'rel').mkdir()
    (tmp_path / 'link').symlink_to(tmp_path / 'rel')
    monkeypatch.chdir(tmp_path)
    cfg = _make_minimal_config(output_dir=name)
    if ok:
        assert validate_config(cfg) is None
    else:
        with pytest.raises(ValueError, match='is not an existing folder'):
            validate_config(cfg)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['link', 'rel']


@pytest.mark.skipif(_ROOT_USER, reason='root writes into a read-only folder')
@pytest.mark.parametrize('mode', [0o500, 0o600, 0o200], ids=['r-x', 'rw-', '-w-'])
def test_output_dir_that_is_not_writable_is_refused(tmp_path, mode):
    """A folder without write or search permission (no file can be made in it) raises ValueError."""
    ro = tmp_path / 'ro'
    ro.mkdir()
    ro.chmod(mode)
    try:
        with pytest.raises(ValueError, match="output_dir .*ro' is not writable"):
            validate_config(_make_minimal_config(output_dir=str(ro)))
    finally:
        ro.chmod(0o700)
    assert validate_config(_make_minimal_config(output_dir=str(ro))) is None


@pytest.mark.usefixtures('fake_solver')
@pytest.mark.parametrize(
    'value, match', [('', 'must not be empty'), ('missing/run', 'is not an existing folder')]
)
def test_post_processing_checks_a_given_output_dir(
    fake_root, tmp_path, monkeypatch, value, match
):
    """A library dict that skipped validate_config: '' and a missing folder raise ValueError, nothing is made."""
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=match):
        output_mod.post_processing(_base_config(output_dir=value))
    assert not (tmp_path / 'missing').exists()
    assert list((fake_root / 'output').iterdir()) == []


@pytest.mark.usefixtures('fake_solver')
def test_plots_alone_create_the_root_output_of_a_fresh_checkout(tmp_path, monkeypatch):
    """data_enabled false, plots on, no <root>/output: the folder is made and holds the PNG."""
    root = tmp_path / 'root'
    root.mkdir()
    for mod in (output_mod, plot_profiles, plot_phase_vs_radius):
        monkeypatch.setattr(mod, 'get_zalmoxis_root', lambda: str(root))
    output_mod.post_processing(_base_config(data_output_enabled=False, plotting_enabled=True))
    assert sorted(p.name for p in (root / 'output').iterdir()) == ['planet_profile.png']


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
        if h not in saved:
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


def test_cli_keeps_the_log_lines_of_a_refused_config(tmp_path, reset_logging):
    """A config load that logs and then raises: the error propagates, the lines reach <root>/output."""
    root = tmp_path / 'root'
    (root / 'output').mkdir(parents=True)

    def load():
        logging.getLogger('zalmoxis.config').info('reading the refused config')
        raise ValueError('output_dir must not be empty')

    with patch('zalmoxis.get_zalmoxis_root', return_value=str(root)):
        with patch('zalmoxis.config.load_zalmoxis_config', side_effect=load):
            with patch('zalmoxis.output.post_processing') as post:
                with pytest.raises(ValueError, match='must not be empty'):
                    runpy.run_module('zalmoxis', run_name='__main__')
    post.assert_not_called()
    log = root / 'output' / 'zalmoxis.log'
    assert log.is_file()
    assert 'reading the refused config' in log.read_text()


def test_cli_creates_the_root_output_of_a_fresh_checkout(tmp_path, reset_logging):
    """A root without output/: the CLI creates it for the log, and a refused config still raises."""
    root = tmp_path / 'root'
    root.mkdir()
    with patch('zalmoxis.get_zalmoxis_root', return_value=str(root)):
        with patch('zalmoxis.config.load_zalmoxis_config', side_effect=ValueError('refused')):
            with pytest.raises(ValueError, match='refused'):
                runpy.run_module('zalmoxis', run_name='__main__')
    assert (root / 'output' / 'zalmoxis.log').is_file()
    assert sorted(p.name for p in root.iterdir()) == ['output']


@pytest.mark.skipif(_ROOT_USER, reason='root writes into a read-only folder')
def test_cli_keeps_the_config_error_when_the_log_cannot_be_opened(tmp_path, reset_logging):
    """A refused config with a read-only <root>/output: the config error escapes, not the log error."""
    root = tmp_path / 'root'
    (root / 'output').mkdir(parents=True)
    (root / 'output').chmod(0o500)
    try:
        with patch('zalmoxis.get_zalmoxis_root', return_value=str(root)):
            with patch(
                'zalmoxis.config.load_zalmoxis_config', side_effect=ValueError('refused')
            ):
                with pytest.raises(Exception) as err:
                    runpy.run_module('zalmoxis', run_name='__main__')
    finally:
        (root / 'output').chmod(0o700)
    assert type(err.value) is ValueError
    assert list((root / 'output').iterdir()) == []


def test_cli_log_names_the_error_of_a_refused_config(tmp_path, reset_logging):
    """The log of a refused config holds the error that stopped the run."""
    root = tmp_path / 'root'
    (root / 'output').mkdir(parents=True)
    with patch('zalmoxis.get_zalmoxis_root', return_value=str(root)):
        with patch('zalmoxis.config.load_zalmoxis_config', side_effect=KeyError('planet_mass')):
            with pytest.raises(KeyError):
                runpy.run_module('zalmoxis', run_name='__main__')
    text = (root / 'output' / 'zalmoxis.log').read_text()
    assert ' - ERROR - ' in text
    assert 'planet_mass' in text


def test_cli_log_is_overwritten_formatted_and_at_info(tmp_path, reset_logging):
    """An old log is replaced; each line reads '<time> - <LEVEL> - <message>'; DEBUG lines are left out."""
    root, run = tmp_path / 'root', tmp_path / 'run'
    (root / 'output').mkdir(parents=True)
    run.mkdir()
    (run / 'zalmoxis.log').write_text('OLD LINE\n')

    def load():
        logging.getLogger('fwl.zalmoxis.config').debug('a debug line')
        logging.getLogger('fwl.zalmoxis.config').info('an info line')
        return {'output_dir': str(run)}

    with patch('zalmoxis.get_zalmoxis_root', return_value=str(root)):
        with patch('zalmoxis.config.load_zalmoxis_config', side_effect=load):
            with patch('zalmoxis.output.post_processing'):
                runpy.run_module('zalmoxis', run_name='__main__')
    lines = (run / 'zalmoxis.log').read_text().splitlines()
    assert lines == [lines[0]] and lines[0].endswith(' - INFO - an info line')
    assert re.fullmatch(r'\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} - INFO - an info line', lines[0])


def test_cli_with_the_real_loader_logs_into_the_folder_of_the_toml(tmp_path, reset_logging):
    """The real load_zalmoxis_config of a default.toml with output_dir: the log and the call go there."""
    from zalmoxis import get_zalmoxis_root

    root, run = tmp_path / 'root', tmp_path / 'run'
    (root / 'input').mkdir(parents=True)
    run.mkdir()
    text = Path(get_zalmoxis_root(), 'input', 'default.toml').read_text()
    (root / 'input' / 'default.toml').write_text(
        text.replace('[Output]\n', f"[Output]\noutput_dir = '{run}'\n")
    )
    with patch('zalmoxis.get_zalmoxis_root', return_value=str(root)):
        with patch('zalmoxis.config.get_zalmoxis_root', return_value=str(root)):
            with patch('sys.argv', ['zalmoxis']):
                with patch('zalmoxis.output.post_processing') as post:
                    runpy.run_module('zalmoxis', run_name='__main__')
    assert post.call_args.args[0]['output_dir'] == str(run)
    assert 'Reading default config file' in (run / 'zalmoxis.log').read_text()
    assert not (root / 'output').exists()


@pytest.mark.usefixtures('fake_solver')
@pytest.mark.skipif(_ROOT_USER, reason='root writes into a read-only folder')
def test_post_processing_that_writes_nothing_needs_no_folder(tmp_path, monkeypatch):
    """Data and plots off, a read-only root: no folder is made and nothing raises."""
    root = tmp_path / 'root'
    root.mkdir()
    root.chmod(0o500)
    for mod in (output_mod, plot_profiles, plot_phase_vs_radius):
        monkeypatch.setattr(mod, 'get_zalmoxis_root', lambda: str(root))
    try:
        output_mod.post_processing(_base_config(data_output_enabled=False))
    finally:
        root.chmod(0o700)
    assert list(root.iterdir()) == []


def test_cli_warns_when_the_log_file_cannot_be_opened(tmp_path, reset_logging, capsys):
    """A folder named zalmoxis.log in output_dir: the log goes to stderr with a warning that names the file."""
    root, run = tmp_path / 'root', tmp_path / 'run'
    (root / 'output').mkdir(parents=True)
    (run / 'zalmoxis.log').mkdir(parents=True)
    with patch('zalmoxis.get_zalmoxis_root', return_value=str(root)):
        with patch(
            'zalmoxis.config.load_zalmoxis_config', return_value={'output_dir': str(run)}
        ):
            with patch('zalmoxis.output.post_processing') as post:
                runpy.run_module('zalmoxis', run_name='__main__')
    post.assert_called_once()
    err = capsys.readouterr().err
    assert ' - WARNING - ' in err
    assert 'zalmoxis.log' in err and 'is written to stderr' in err
