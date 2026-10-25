"""Tests for ``tools.validation.pressure_root_sweep`` with written setup records, no solves."""

from __future__ import annotations

import json

import pytest

from tools.validation import pressure_root_sweep as tool

pytestmark = pytest.mark.unit


def _write(outdir, error_at=None):
    for i, (name, _) in enumerate(tool.SETUPS):
        rec = {'index': i, 'name': name, 'target': 101325.0, 'error': None}
        if i == error_at:
            rec.update(converged=False, error='StructureSolveError: no root')
        else:
            rec.update(converged=True, P_R=1.2e5, padded=int(i % 2))
        (outdir / f'setup_{i}.json').write_text(json.dumps(rec))


def test_the_plot_names_a_failed_setup_instead_of_a_key_error(tmp_path):
    """A setup that raised has no P(R): the plot stops with its index, name and error."""
    _write(tmp_path, error_at=3)
    with pytest.raises(SystemExit, match='3 Analytic 1, 2000 layers: StructureSolveError'):
        tool.plot(str(tmp_path), str(tmp_path / 'assets'))


def test_the_plot_writes_its_files_with_a_runnable_command(tmp_path, monkeypatch):
    """Every setup accepted: the plot is saved with the run and plot commands as provenance."""
    import proteus_mpl.validation as pv

    seen = {}

    def save(figure, stem, assets, command):
        seen.update(stem=stem, command=command, fig=figure())
        return []

    monkeypatch.setattr(pv, 'save', save)
    _write(tmp_path)
    tool.plot(str(tmp_path), str(tmp_path / 'assets'))
    assert seen['stem'] == 'pressure_root_sweep'
    assert seen['command'].count('python -m tools.validation.pressure_root_sweep') == 2
    assert len(seen['fig'].axes[0].lines) == 3  # reached, padded, cap
