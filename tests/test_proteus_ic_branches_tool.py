"""Tests for ``tools.validation.proteus_ic_branches`` with written records, no solves."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tools.validation import proteus_ic_branches as tool

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

T_SURF, T_CMB = 3936.17, 6121.26


def _rec(branch, dT=0.0, P_cmb=105.7e9, n=39):
    """A record whose last core node sits on ``branch`` and whose mantle T is a ramp shifted
    by ``dT`` at its middle point."""
    frac = {'T_surf': 4e-4, 'raised': 0.35}[branch]
    T_q = np.linspace(6100.0, 3950.0, n)
    T_q[n // 2] += dT
    return {
        'case': 'all_options',
        'branch': branch,
        'frac': frac,
        'T_last': T_SURF + frac * (T_CMB - T_SURF),
        'x_q': list(np.linspace(0.0, 1.0, n + 2)[1:-1]),
        'T_q': list(T_q),
        'P_cmb': P_cmb,
    }


@pytest.mark.parametrize(
    ('T_last', 'want'),
    [(3937.0, 'T_surf'), (T_SURF, 'T_surf'), (4705.34, 'raised'), (4678.88 + 1934.43, None)],
)
def test_the_last_core_node_is_named_by_its_branch(T_last, want):
    """3937 K (0.04 % of the way from T_surf to T_cmb) is the T_surf branch and 4705 K (35 %)
    the raised branch; a node near T_cmb is neither and raises."""
    if want is None:
        with pytest.raises(ValueError, match='neither branch'):
            tool.last_core_branch(T_last, T_SURF, T_CMB)
        return
    name, frac = tool.last_core_branch(T_last, T_SURF, T_CMB)
    assert name == want
    assert frac == pytest.approx((T_last - T_SURF) / (T_CMB - T_SURF), rel=1e-12)


@pytest.mark.parametrize('frac', [0.05, 0.1, 0.19, 0.51, 1.0, -0.06])
def test_a_node_between_the_branch_windows_raises(frac):
    """The gap between the windows (0.05 to 0.2) and the values outside them name no branch."""
    with pytest.raises(ValueError, match='neither branch'):
        tool.last_core_branch(T_SURF + frac * (T_CMB - T_SURF), T_SURF, T_CMB)


@pytest.mark.parametrize(
    ('T_last', 'T_surf', 'T_cmb'),
    [
        (math.nan, T_SURF, T_CMB),
        (math.inf, T_SURF, T_CMB),
        (4000.0, T_CMB, T_CMB),
        (4000.0, T_CMB, T_SURF),
        (4000.0, math.nan, T_CMB),
    ],
)
def test_the_branch_needs_finite_temperatures_and_a_hotter_cmb(T_last, T_surf, T_cmb):
    """A NaN or infinite temperature, or a CMB anchor not above T_surf, has no branch."""
    with pytest.raises(ValueError):
        tool.last_core_branch(T_last, T_surf, T_cmb)


@settings(max_examples=200, deadline=None)
@given(st.floats(allow_nan=True, allow_infinity=True, width=64))
def test_every_temperature_gets_a_branch_inside_its_window_or_raises(T_last):
    """Over all floats: a returned branch has its fraction inside that branch's window."""
    try:
        name, frac = tool.last_core_branch(T_last, T_SURF, T_CMB)
    except ValueError:
        return
    lo, hi = tool.BRANCH_WINDOWS[name]
    assert lo <= frac < hi
    assert name in ('T_surf', 'raised')


@pytest.mark.parametrize(
    ('bp', 'bn', 'dT', 'dP', 'fails'),
    [
        ('raised', 'raised', 4.9, 0.09e9, []),
        ('raised', 'raised', 6.0, 0.0, ['mantle T']),
        ('T_surf', 'T_surf', 0.0, 0.15e9, ['P_cmb']),
        ('raised', 'T_surf', 10.84, 0.05e9, []),
        ('T_surf', 'raised', 9.84, 0.149e9, []),
        ('raised', 'T_surf', 16.0, 0.05e9, ['mantle T']),
        ('raised', 'T_surf', 0.0, 0.25e9, ['P_cmb']),
        ('raised', 'raised', 16.0, 0.25e9, ['mantle T', 'P_cmb']),
    ],
)
def test_a_same_branch_pair_meets_5_K_and_a_cross_branch_pair_15_K(bp, bn, dT, dP, fails):
    """Same branch: 5 K and 0.1 GPa; different branches: the measured branch step, at most 15 K
    and 0.2 GPa. Each failure names the branch of both solves."""
    c = tool.compare(_rec(bp), _rec(bn, dT=dT, P_cmb=105.7e9 + dP))
    assert c['same'] is (bp == bn)
    assert (c['limit_T'], c['limit_P']) == ((5.0, 0.1e9) if bp == bn else (15.0, 0.2e9))
    assert c['dT'] == pytest.approx(dT, abs=1e-9)
    got = tool.failures(c)
    assert [f.split(':')[0] for f in got] == fails
    for f in got:
        assert f'Picard {bp}' in f and f'Newton {bn}' in f


def test_the_comparison_refuses_records_on_different_mass_points():
    """The mantle T of the 2 solves is compared at the same mass points only."""
    a, b = _rec('raised'), _rec('raised')
    b['x_q'] = list(np.asarray(b['x_q']) * 0.5)
    with pytest.raises(ValueError, match='mass points'):
        tool.compare(a, b)


def _write(outdir):
    for case, (bp, bn, dT) in {
        'all_options': ('raised', 'T_surf', 10.84),
        'dummy': ('raised', 'raised', 0.02),
    }.items():
        recs = {'picard': dict(_rec(bp), case=case), 'newton': dict(_rec(bn, dT=dT), case=case)}
        (outdir / f'case_{case}.json').write_text(json.dumps(recs))


def _capture_save(monkeypatch):
    import proteus_mpl.validation as pv

    seen = {}

    def save(figure, stem, assets, command):
        seen.update(stem=stem, command=command, fig=figure())
        return []

    monkeypatch.setattr(pv, 'save', save)
    return seen


def test_the_plot_draws_the_difference_and_the_limit_of_each_branch_pair(tmp_path, monkeypatch):
    """One series per setup: |T_Newton - T_Picard| at the mantle points, floored at 1e-3 K,
    with the limit of its branch pair; the labels name both branches."""
    seen = _capture_save(monkeypatch)
    _write(tmp_path)
    tool.plot(str(tmp_path), str(tmp_path / 'assets'))
    assert seen['stem'] == 'proteus_ic_branches'
    assert seen['command'].count('python -m tools.validation.proteus_ic_branches') == 2
    (ax,) = seen['fig'].axes
    lines = {ln.get_label(): ln for ln in ax.lines}
    assert (
        lines['15 K limit'].get_ydata()[0],
        lines['5 K limit'].get_ydata()[0],
    ) == (15.0, 5.0)
    a, d = (
        lines['all_options: Picard raised, Newton T_surf'],
        lines['dummy: Picard raised, Newton raised'],
    )
    assert max(a.get_ydata()) == pytest.approx(10.84, rel=1e-9)
    assert min(d.get_ydata()) == 1e-3
    assert lines['5 K limit'].get_color() == d.get_color() != a.get_color()


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
                inside = (
                    (xy[:, 0] > b.x0)
                    & (xy[:, 0] < b.x1)
                    & (xy[:, 1] > b.y0)
                    & (xy[:, 1] < b.y1)
                )
                assert not inside.any(), (ln.get_label(), b)


def test_the_plot_stops_on_a_case_without_a_record(tmp_path):
    """A case whose solve died writes no record: the plot stops and names it."""
    _write(tmp_path)
    (tmp_path / 'case_dummy.json').unlink()
    with pytest.raises(SystemExit, match='dummy: no record'):
        tool.plot(str(tmp_path), str(tmp_path / 'assets'))


def _result(T_last):
    """A 12-node result: 5 core nodes at T_surf, the last core node at ``T_last``, then a mantle
    ramp from T_cmb down; mass in units of the planet mass, CMB at 0.5."""
    cfg = {**tool.PROTEUS_IC['common'], **tool.PROTEUS_IC['all_options']}
    M = cfg['planet_mass']
    m = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.49, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]) * M
    T = np.r_[[T_SURF] * 5, T_last, np.linspace(T_CMB, 3000.0, 6)]
    P = np.linspace(360e9, 5e6, 12)
    res = {'mass_enclosed': m, 'pressure': P, 'temperature': T, 'cmb_mass': 0.5 * M}
    res.update(converged=True, adiabat_computed=True, adiabat_blend=1.0, accepted_on='strict')
    return res, cfg


@pytest.mark.parametrize(('T_last', 'branch'), [(3937.0, 'T_surf'), (4705.3, 'raised')])
def test_a_record_reads_the_last_node_below_the_cmb_mass(T_last, branch):
    """The record takes its branch from the last node below the CMB mass, not from the first
    mantle node at T_cmb, and interpolates the mantle T and the CMB pressure on the mass."""
    res, cfg = _result(T_last)
    rec = tool.record(res, cfg)
    assert (rec['branch'], rec['T_last']) == (branch, T_last)
    assert len(rec['T_q']) == 39 and rec['x_q'][0] == pytest.approx(0.025)
    assert rec['T_q'][0] == pytest.approx(T_CMB + 0.025 * (3000.0 - T_CMB), rel=1e-9)
    assert rec['P_cmb'] == pytest.approx(np.linspace(360e9, 5e6, 12)[6], rel=1e-12)
    assert rec['converged'] is True and rec['blend'] == 1.0


def test_run_writes_one_record_per_case_and_plot_reads_them(tmp_path, monkeypatch, capsys):
    """``run`` stores the records of each named case and prints its comparison; ``plot`` reads
    the directory."""
    monkeypatch.setattr(
        tool,
        'run_case',
        lambda case: {'picard': _rec('raised'), 'newton': _rec('raised', dT=1.0)},
    )
    tool.main(['run', str(tmp_path), 'dummy'])
    assert [p.name for p in tmp_path.iterdir()] == ['case_dummy.json']
    assert '"dT": 1.0' in capsys.readouterr().out
    seen = {}
    monkeypatch.setattr(
        tool, 'plot', lambda outdir, assets: seen.update(outdir=outdir, assets=assets)
    )
    tool.main(['plot', str(tmp_path)])
    assert seen == {'outdir': str(tmp_path), 'assets': 'docs/assets/validation'}
