"""Tests for ``tools.validation.proteus_ic_solvers`` with written records, no solves."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tools.validation import proteus_ic_solvers as tool

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]


def _rec(dT=0.0, P_cmb=105.7e9, n=39):
    """A record whose mantle T is a ramp shifted by ``dT`` at its middle point."""
    T_q = np.linspace(6100.0, 3950.0, n)
    T_q[n // 2] += dT
    x = np.linspace(0.0, 1.0, n + 2)[1:-1]
    return {
        'case': 'all_options',
        'x_q': list(x),
        'm_q': list((0.325 + 0.675 * x) * 5.972e24),
        'T_q': list(T_q),
        'P_cmb': P_cmb,
    }


@pytest.mark.parametrize(('dT', 'dP'), [(0.0, 0.0), (4.9, 0.09e9), (-16.0, -0.25e9)])
def test_the_comparison_gives_the_largest_mantle_t_and_the_cmb_pressure_difference(dT, dP):
    """dT is the largest |T_Newton - T_Picard| over the mantle points, dP the CMB pressure
    difference, both as magnitudes."""
    c = tool.compare(_rec(), _rec(dT=dT, P_cmb=105.7e9 + dP))
    assert c == {'dT': pytest.approx(abs(dT), abs=1e-9), 'dP': pytest.approx(abs(dP), abs=1e-3)}


@pytest.mark.parametrize('field', ['T_q', 'P_cmb'])
def test_a_nan_in_one_record_gives_a_nan_difference(field):
    """A NaN mantle T or CMB pressure makes its difference NaN, which no limit accepts."""
    b = _rec()
    b[field] = [math.nan] * 39 if field == 'T_q' else math.nan
    c = tool.compare(_rec(), b)
    assert math.isnan(c['dT' if field == 'T_q' else 'dP'])


def test_the_comparison_refuses_records_on_different_mass_points():
    """The mantle T of the 2 solves is compared at the same absolute masses only."""
    a, b = _rec(), _rec()
    b['m_q'] = list(np.asarray(b['m_q']) * 0.99)
    with pytest.raises(ValueError, match='mass points'):
        tool.compare(a, b)


def _write(outdir):
    for case, dT in {'all_options': 0.81, 'dummy': 0.02}.items():
        recs = {'picard': dict(_rec(), case=case), 'newton': dict(_rec(dT=dT), case=case)}
        (outdir / f'case_{case}.json').write_text(json.dumps(recs))


def _capture_save(monkeypatch):
    import proteus_mpl.validation as pv

    seen = {}

    def save(figure, stem, assets, command):
        seen.update(stem=stem, command=command, fig=figure())
        return []

    monkeypatch.setattr(pv, 'save', save)
    return seen


def test_the_plot_draws_the_difference_of_each_setup_and_the_5_K_limit(tmp_path, monkeypatch):
    """One series per setup: |T_Newton - T_Picard| at the mantle points, floored at 1e-3 K,
    and one line at the 5 K limit."""
    seen = _capture_save(monkeypatch)
    _write(tmp_path)
    tool.plot(str(tmp_path), str(tmp_path / 'assets'))
    assert seen['stem'] == 'proteus_ic_solvers'
    assert seen['command'].count('python -m tools.validation.proteus_ic_solvers') == 2
    (ax,) = seen['fig'].axes
    lines = {ln.get_label(): ln for ln in ax.lines}
    assert len(lines) == len(ax.lines) == 3
    assert list(lines['5 K limit'].get_ydata()) == [5.0, 5.0]
    assert max(lines['all_options'].get_ydata()) == pytest.approx(0.81, rel=1e-9)
    assert min(lines['dummy'].get_ydata()) == 1e-3
    assert lines['all_options'].get_color() != lines['dummy'].get_color()


def test_the_legend_covers_no_data_point(tmp_path, monkeypatch):
    """The legend sits outside every line vertex in display coordinates."""
    seen = _capture_save(monkeypatch)
    _write(tmp_path)
    tool.plot(str(tmp_path), str(tmp_path / 'assets'))
    fig = seen['fig']
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    boxes = [lg.get_window_extent(r) for lg in fig.legends]
    assert fig.legends
    for ax in fig.axes:
        for ln in ax.lines:
            xy = ln.get_transform().transform(np.column_stack([ln.get_xdata(), ln.get_ydata()]))
            for b in boxes:
                assert b.count_contains(xy) == 0, (ln.get_label(), b)


def test_the_plot_stops_on_a_case_without_a_record(tmp_path):
    """A case whose solve died writes no record: the plot stops and names it."""
    _write(tmp_path)
    (tmp_path / 'case_dummy.json').unlink()
    with pytest.raises(SystemExit, match='dummy: no record'):
        tool.plot(str(tmp_path), str(tmp_path / 'assets'))


def test_a_record_interpolates_the_mantle_t_and_the_cmb_pressure_on_the_mass():
    """A 12-node result with the CMB at half the planet mass: 39 mantle points from the CMB
    to the surface mass, T and P interpolated on the mass, and the accept state kept."""
    cfg = {**tool.PROTEUS_IC['common'], **tool.PROTEUS_IC['all_options']}
    M = cfg['planet_mass']
    m = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.49, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) * M
    T = np.r_[[3936.0] * 6, np.linspace(6121.0, 3000.0, 6)]
    P = np.linspace(360e9, 5e6, 12)
    res = {'mass_enclosed': m, 'pressure': P, 'temperature': T, 'cmb_mass': 0.5 * M}
    res.update(converged=True, adiabat_computed=True, adiabat_blend=1.0, accepted_on='strict')
    res.update(adiabat_table_blend=1.0, adiabat_self_consistency=0.2)
    rec = tool.record(res, cfg)
    assert len(rec['T_q']) == 39 and rec['x_q'][0] == pytest.approx(0.025)
    assert rec['m_q'][0] == pytest.approx(0.5125 * M, rel=1e-12)
    assert rec['T_q'][0] == pytest.approx(6121.0 + 0.025 * (3000.0 - 6121.0), rel=1e-9)
    assert rec['P_cmb'] == pytest.approx(P[6], rel=1e-12)
    assert (rec['blend'], rec['table_blend'], rec['self_consistency']) == (1.0, 1.0, 0.2)
    assert rec['converged'] is True and rec['accepted_on'] == 'strict'


def test_run_writes_one_record_per_case_and_plot_reads_them(tmp_path, monkeypatch, capsys):
    """``run`` makes the output directory, stores the records of each named case and prints
    its comparison; ``plot`` reads the directory."""
    monkeypatch.setattr(
        tool, 'run_case', lambda case: {'picard': _rec(), 'newton': _rec(dT=1.0)}
    )
    out = tmp_path / 'new'
    tool.main(['run', str(out), 'dummy'])
    assert [p.name for p in out.iterdir()] == ['case_dummy.json']
    assert '"dT": 1.0' in capsys.readouterr().out
    seen = {}
    monkeypatch.setattr(
        tool, 'plot', lambda outdir, assets: seen.update(outdir=outdir, assets=assets)
    )
    tool.main(['plot', str(tmp_path)])
    assert seen == {'outdir': str(tmp_path), 'assets': 'docs/assets/validation'}
