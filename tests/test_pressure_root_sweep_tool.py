"""Tests for ``tools.validation.pressure_root_sweep`` with written setup records, no solves."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from tools.validation import pressure_root_sweep as tool

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _write(outdir, error_at=None, all_padded=False):
    """Even setups reach the surface with a miss of (i + 1) / 100 P_t, odd ones are padded."""
    for i, (name, _) in enumerate(tool.SETUPS):
        rec = {'index': i, 'name': name, 'target': 101325.0, 'error': None}
        if i == error_at:
            rec.update(converged=False, error='StructureSolveError: no root')
        elif i % 2 or all_padded:
            rec.update(converged=True, P_R=0.0, padded=1)
        else:
            rec.update(converged=True, P_R=101325.0 * (1.0 + (i + 1) / 100), padded=0)
        (outdir / f'setup_{i}.json').write_text(json.dumps(rec))


def _capture_save(monkeypatch):
    import proteus_mpl.validation as pv

    seen = {}

    def save(figure, stem, assets, command):
        seen.update(stem=stem, command=command, fig=figure())
        return []

    monkeypatch.setattr(pv, 'save', save)
    return seen


@pytest.mark.parametrize('fault', ['error', 'not_converged'])
def test_the_plot_marks_and_names_a_setup_without_an_accepted_profile(
    tmp_path, monkeypatch, capsys, fault
):
    """A setup that raised or did not converge has no P(R): the plot draws a cross above the
    cap at its index and prints its index, name and error."""
    seen = _capture_save(monkeypatch)
    _write(tmp_path, error_at=3)
    if fault == 'not_converged':
        path = tmp_path / 'setup_3.json'
        rec = dict(json.loads(path.read_text()), error=None, P_R=1.2e5, padded=0)
        path.write_text(json.dumps(rec))
    tool.plot(str(tmp_path), str(tmp_path / 'assets'))
    lines = {line.get_label(): line for line in seen['fig'].axes[0].lines}
    assert sorted(lines) == ['Cap', 'Failed', 'Padded', 'Reached']
    failed = lines['Failed']
    assert (
        list(failed.get_xdata()) == [3] and failed.get_ydata()[0] > lines['Cap'].get_ydata()[0]
    )
    assert failed.get_marker() == 'x'
    assert (len(lines['Reached'].get_xdata()), len(lines['Padded'].get_xdata())) == (13, 12)
    want = 'StructureSolveError' if fault == 'error' else 'not converged'
    assert f'3 Analytic 1, 2000 layers: {want}' in capsys.readouterr().out


def test_the_plot_stops_on_a_setup_without_a_record(tmp_path):
    """A setup whose process died writes no record: the plot stops and names it."""
    _write(tmp_path)
    (tmp_path / 'setup_5.json').unlink()
    with pytest.raises(SystemExit, match='5 WB isothermal 3000 K: no record'):
        tool.plot(str(tmp_path), str(tmp_path / 'assets'))


def test_the_plot_writes_its_files_with_the_run_and_plot_commands(tmp_path, monkeypatch):
    """Every setup accepted: the plot is saved with the run and plot commands as provenance."""
    seen = _capture_save(monkeypatch)
    _write(tmp_path)
    tool.plot(str(tmp_path), str(tmp_path / 'assets'))
    assert seen['stem'] == 'pressure_root_sweep'
    assert seen['command'].count('python -m tools.validation.pressure_root_sweep') == 2
    lines = {line.get_label(): line for line in seen['fig'].axes[0].lines}
    assert sorted(lines) == ['Cap', 'Padded', 'Reached']
    assert (len(lines['Reached'].get_xdata()), len(lines['Padded'].get_xdata())) == (13, 13)


@pytest.mark.parametrize(('error_at', 'counts'), [(None, (13, 13, 0)), (3, (13, 12, 1))])
def test_the_plot_prints_the_counts_and_the_reached_miss(
    tmp_path, monkeypatch, capsys, error_at, counts
):
    """The plot prints the reached, padded and failed counts and the median and largest miss of
    the reached setups, the numbers the docs quote."""
    _capture_save(monkeypatch)
    _write(tmp_path, error_at=error_at)
    tool.plot(str(tmp_path), str(tmp_path / 'assets'))
    want = 'reached {}, padded {}, failed {}; reached miss median 0.130, max 0.250 P_t'
    assert want.format(*counts) in capsys.readouterr().out


def test_the_plot_of_a_sweep_without_a_reached_setup_prints_its_counts(
    tmp_path, monkeypatch, capsys
):
    """Every setup padded: no reached miss to summarise, and the plot is still written."""
    seen = _capture_save(monkeypatch)
    _write(tmp_path, all_padded=True)
    tool.plot(str(tmp_path), str(tmp_path / 'assets'))
    assert 'reached 0, padded 26, failed 0\n' in capsys.readouterr().out
    assert seen['stem'] == 'pressure_root_sweep'


def test_a_run_removes_the_records_of_an_earlier_run(tmp_path, monkeypatch):
    """A setup whose process dies writes no record, so a run first removes the records of
    its own setups that an earlier run left; other files stay."""
    monkeypatch.setattr(tool, 'SETUPS', tool.SETUPS[:2])
    monkeypatch.setattr(
        tool.subprocess, 'run', lambda cmd, check: SimpleNamespace(returncode=1)
    )
    (tmp_path / 'setup_1.json').write_text('{}')
    (tmp_path / 'setup_99.json').write_text('{}')
    (tmp_path / 'notes.txt').write_text('kept')
    tool.run_all(str(tmp_path), 1)
    assert sorted(p.name for p in tmp_path.iterdir()) == ['notes.txt', 'setup_99.json']
