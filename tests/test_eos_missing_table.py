"""A missing EOS table names its cause and the setup script that restores it."""

from __future__ import annotations

import logging
import os

import pytest

from zalmoxis.eos.interpolation import (
    load_paleos_table,
    load_paleos_unified_table,
    require_eos_file,
)
from zalmoxis.eos.seager import get_tabulated_eos
from zalmoxis.eos.tdep import load_melting_curve
from zalmoxis.melting_curves import _load_tabulated_curve

pytestmark = pytest.mark.unit


@pytest.fixture
def dangling(tmp_path):
    """A data/ folder linked to an FWL_DATA version directory that no longer exists."""
    data = tmp_path / 'data'
    data.mkdir()
    gone = tmp_path / 'fwl_data' / 'interior' / 'eos' / 'seager_2007' / 'r15727998'
    os.symlink(gone, data / 'EOS_Seager2007')
    return data / 'EOS_Seager2007' / 'eos_seager07_iron.txt', data / 'EOS_Seager2007', gone


def test_a_table_behind_a_dangling_link_names_the_link_and_the_setup_script(dangling):
    table, link, gone = dangling
    with pytest.raises(FileNotFoundError) as raised:
        require_eos_file(str(table))
    message = str(raised.value)
    assert f'{link} links to {gone}' in message
    assert 'bash tools/setup/get_zalmoxis.sh to relink it' in message


def test_a_plain_missing_table_asks_for_the_fetch(tmp_path):
    with pytest.raises(
        FileNotFoundError, match='remove any data/ folder it reports as kept'
    ) as raised:
        require_eos_file(str(tmp_path / 'absent.txt'))
    assert 'links to' not in str(raised.value)


def test_a_present_table_passes(tmp_path):
    table = tmp_path / 'table.txt'
    table.write_text('1 2\n')
    assert require_eos_file(str(table)) is None


@pytest.mark.parametrize('loader', [load_paleos_table, load_paleos_unified_table])
def test_the_paleos_loaders_report_the_dangling_link(dangling, loader):
    table, link, _ = dangling
    with pytest.raises(FileNotFoundError, match='relink it'):
        loader(str(table))


def test_the_tabulated_eos_logs_the_cause_and_returns_none(dangling, caplog):
    table, link, _ = dangling
    materials = {'core': {'eos_file': str(table)}}
    with caplog.at_level(logging.ERROR, logger='zalmoxis.eos.seager'):
        assert get_tabulated_eos(1e10, materials, 'core', temperature=300.0) is None
    assert f'{link} links to' in caplog.text
    assert 'get_zalmoxis.sh' in caplog.text


def test_the_melting_curve_loaders_report_the_dangling_link(dangling, capsys):
    table, link, _ = dangling
    with pytest.raises(FileNotFoundError, match='relink it'):
        _load_tabulated_curve(str(table))
    assert load_melting_curve(str(table)) is None
    assert f'{link} links to' in capsys.readouterr().out
