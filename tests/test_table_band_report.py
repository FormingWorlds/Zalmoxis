"""Band report of the converged mantle profile against the valid T band of regular tables.

Module under test: ``zalmoxis.solver._report_table_bands`` and its call from
``zalmoxis.solver.main``. Contract: each cached regular table that a mantle node reads
gets the count of nodes outside its band and the largest weighted excess in K; one
warning per call names the tables; an excess above ``max_band_excess`` returns
``converged=False`` and ``converged_bands=False``. The unit tests use the synthetic fill
table of ``tests/test_seager_branches.py``; the integration tests solve a 1 M_earth Wolf
& Bower mantle. See docs/How-to/build_tests.md for the test tiers.
"""

from __future__ import annotations

import ast
import logging
import os
import re

import numpy as np
import pytest

from tests.test_seager_branches import _fill_table, _write_table
from zalmoxis import get_zalmoxis_root, mixing, solver
from zalmoxis.config import band_limit, load_zalmoxis_config, validate_config
from zalmoxis.eos import seager, tdep
from zalmoxis.eos.seager import get_tabulated_eos
from zalmoxis.eos.temperature import compute_adiabatic_temperature
from zalmoxis.eos_properties import EOS_REGISTRY
from zalmoxis.mixing import parse_all_layer_mixtures
from zalmoxis.solver import _report_table_bands


@pytest.fixture
def mats(tmp_path, monkeypatch):
    """Registry entry with the fill table as melt, solid and gradient table, all cached."""
    monkeypatch.setattr(solver, '_interpolation_cache', {})
    melt, grad, solid = (
        _write_table(tmp_path / name, _fill_table())['melted_mantle']['eos_file']
        for name in ('density_melt.dat', 'grad.dat', 'density_solid.dat')
    )
    for path in (melt, grad, solid):
        get_tabulated_eos(
            1e9,
            {'melted_mantle': {'eos_file': path}},
            'melted_mantle',
            2000.0,
            solver._interpolation_cache,
        )
    entry = {
        'melted_mantle': {'eos_file': melt, 'adiabat_grad_file': grad},
        'solid_mantle': {'eos_file': solid},
    }
    return {'X:MgSiO3': entry}, melt, grad, solid


def _result(T, P=(0.0, 1e9, 2e9, 3e9), m=(1.0, 2.0, 3.0, 4.0), ice='', adiabat=False):
    """A converged result whose nodes 1 to 3 (m >= 2) are mantle."""
    return {
        'layer_eos_config': {'core': 'Seager2007:iron', 'mantle': 'X:MgSiO3', 'ice_layer': ice},
        'mass_enclosed': np.array(m),
        'pressure': np.array(P),
        'temperature': np.array(T, dtype=float),
        'cmb_mass': 2.0,
        'core_mantle_mass': 3.5,
        'converged': True,
        'adiabat_computed': adiabat,
    }


def _curves(t_sol, t_liq):
    """Constant solidus and liquidus functions."""
    return (lambda p: t_sol, lambda p: t_liq)


def _check(res, md, curves, mixtures=None, **cfg):
    """Run the band check with the layers of ``res`` in the config, as main() passes them."""
    cfg = {'layer_eos_config': res['layer_eos_config'], **cfg}
    return _report_table_bands(res, cfg, md, curves, mixtures)


HOT = [0, 3300.0, 2000.0, 3600.0]  # node 1: 300 K above row 1, node 3: 100 K above row 3


@pytest.mark.unit
class TestReportTableBands:
    """Counts, excess, table use per node, warning and the converged flags."""

    def test_melt_nodes_report_count_and_largest_excess(self, mats, caplog):
        """Molten nodes read the melt table; 3300 K at 1 GPa is 300 K beyond its band."""
        md, melt, grad, solid = mats
        with caplog.at_level(logging.WARNING, logger='fwl.zalmoxis.solver'):
            res = _check(_result(HOT), md, _curves(500.0, 600.0))
        assert res['table_band_excess'] == {melt: (2, pytest.approx(300.0)), solid: (0, 0.0)}
        assert res['converged'] is False and res['converged_bands'] is False
        msgs = [r.getMessage() for r in caplog.records]
        assert len(msgs) == 2
        assert 'density_melt.dat 2, 300' in msgs[0] and 'value there is taken' in msgs[0]
        assert 'max_band_excess = 200 K' in msgs[1]

    def test_limit_parameter_keeps_converged(self, mats, caplog):
        """With max_band_excess = 300 K (a TOML string works too) the profile stays converged."""
        md, _, _, _ = mats
        for limit in (300.0, '300'):
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger='fwl.zalmoxis.solver'):
                res = _check(_result(HOT), md, _curves(500.0, 600.0), max_band_excess=limit)
            assert res['converged'] is True and res['converged_bands'] is True
            assert len(caplog.records) == 1

    def test_inside_every_band_is_silent(self, mats, caplog):
        """A profile inside every band reports zeros and logs nothing."""
        md, melt, grad, solid = mats
        res = _result([9999.0, 2000.0, 2000.0, 2000.0], adiabat=True)
        with caplog.at_level(logging.WARNING, logger='fwl.zalmoxis.solver'):
            _check(res, md, _curves(1800.0, 2200.0))
        assert res['table_band_excess'] == {melt: (0, 0.0), solid: (0, 0.0), grad: (0, 0.0)}
        assert res['converged'] is True and not caplog.records

    @pytest.mark.parametrize(
        ('t_sol', 't_liq'),
        [
            (5000.0, 6000.0),
            (100.0, 200.0),
            (100.0, 6000.0),
            (np.nan, np.nan),
            (6000.0, 100.0),
            (3600.0, 3700.0),
            (3500.0, 3600.0),
        ],
        ids=[
            'solid',
            'molten',
            'mushy',
            'undefined',
            'degenerate',
            'at_solidus',
            'at_liquidus',
        ],
    )
    def test_tables_match_the_density_lookup(self, mats, monkeypatch, t_sol, t_liq):
        """A node counts against exactly the tables get_Tdep_density reads at its P and T."""
        md, melt, _, solid = mats
        read = []
        monkeypatch.setattr(
            tdep, 'get_tabulated_eos', lambda p, mat, phase, *a: read.append(phase) or 3000.0
        )
        tdep.get_Tdep_density(1e9, 3600.0, md['X:MgSiO3'], *_curves(t_sol, t_liq))
        res = _check(
            _result([0, 3600.0, 2000.0, 2000.0]), md, _curves(t_sol, t_liq), max_band_excess=1e9
        )
        counted = {
            'melted_mantle': res['table_band_excess'][melt][0],
            'solid_mantle': res['table_band_excess'][solid][0],
        }
        assert counted == {k: int(k in read) for k in counted}

    @pytest.mark.parametrize(
        ('mode', 'expected'),
        [('adiabatic', (1, 100.0)), ('adiabatic_from_cmb', (1, 300.0))],
        ids=['surface_anchor', 'cmb_anchor'],
    )
    def test_gradient_counts_where_the_adiabat_reads(self, mats, mode, expected):
        """The surface-anchored adiabat never reads the innermost mantle node, the
        CMB-anchored one never the outermost; no gradient entry without an adiabat."""
        md, _, grad, _ = mats
        res = _check(
            _result(HOT, adiabat=True),
            md,
            _curves(500.0, 600.0),
            temperature_mode=mode,
            max_band_excess=1e9,
        )
        assert res['table_band_excess'][grad] == (expected[0], pytest.approx(expected[1]))
        assert grad not in _check(_result(HOT), md, _curves(500.0, 600.0))['table_band_excess']

    @pytest.mark.parametrize(
        ('pressures', 'ice', 'expected'),
        [
            ((1e9, 1e9, 2e9, 3e9), 'Seager2007:H2O', (1, 300.0)),  # node 3 is ice
            ((1e9, 1e9, 2e9, 0.0), '', (2, 6500.0)),  # node 3 is the surface: 9000 - 2500 K
            ((1e9, 1e9, 2e9, 3e9), '', (2, 5500.0)),  # node 3 is mantle: 9000 - 3500 K
        ],
        ids=['ice_layer', 'surface_node', 'mantle_node'],
    )
    def test_core_and_ice_nodes_are_left_out(self, mats, pressures, ice, expected):
        """Core nodes and ice nodes above core_mantle_mass are not mantle; the surface is."""
        md, melt, _, _ = mats
        res = _check(
            _result([9000.0, 3300.0, 2000.0, 9000.0], P=pressures, ice=ice),
            md,
            _curves(100.0, 200.0),
            max_band_excess=1e9,
        )
        assert res['table_band_excess'][melt] == (expected[0], pytest.approx(expected[1]))

    def test_excess_is_weighted_between_rows(self, mats):
        """Halfway between rows 1 and 2, 1200 K is 300 K below row 2 only: 150 K."""
        md, melt, _, _ = mats
        res = _check(
            _result([0, 1200.0, 2000.0, 2000.0], P=(0.0, 1.5e9, 2e9, 3e9)),
            md,
            _curves(100.0, 200.0),
        )
        assert res['table_band_excess'][melt] == (1, pytest.approx(150.0))

    @pytest.mark.parametrize('pressure', [1e9, np.nan], ids=['nan_T', 'nan_P_and_T'])
    def test_non_finite_node_fails_the_check(self, mats, pressure):
        """A mantle node with a NaN temperature or pressure counts as infinitely far outside."""
        md, melt, _, solid = mats
        res = _check(
            _result([0, np.nan, 2000.0, 2000.0], P=(0.0, pressure, 2e9, 3e9)),
            md,
            _curves(100.0, 200.0),
        )
        assert res['table_band_excess'][melt] == (1, np.inf)
        assert res['table_band_excess'][solid] == (1, np.inf)
        assert res['converged'] is False

    def test_nan_pressure_alone_fails_the_check(self, mats):
        """A NaN pressure at a node with a finite temperature also counts as infinite."""
        md, melt, _, _ = mats
        res = _check(
            _result([0, 2000.0, 2000.0, 2000.0], P=(0.0, np.nan, 2e9, 3e9)),
            md,
            _curves(100.0, 200.0),
        )
        assert res['table_band_excess'][melt] == (1, np.inf)

    def test_uncached_table_is_not_checked(self, mats):
        """A table no lookup of the solve has read is left out of the report."""
        md, melt, grad, solid = mats
        solver._interpolation_cache.pop(solid)
        res = _check(_result(HOT), md, _curves(500.0, 600.0))
        assert list(res['table_band_excess']) == [melt]
        assert solid not in solver._interpolation_cache

    def test_component_at_zero_fraction_is_not_checked(self, mats):
        """A W&B-like table cached by an earlier solve is not counted for a mantle that holds
        it at fraction 0, while the earlier all-W&B mantle did count it."""
        md, melt, _, _ = mats
        first = _check(_result(HOT), md, _curves(500.0, 600.0))
        assert first['table_band_excess'][melt][0] == 2 and first['converged'] is False
        res = _result(HOT)
        res['layer_eos_config']['mantle'] = 'Seager2007:MgSiO3:1.0+X:MgSiO3:0.0'
        second = _check(res, md, _curves(500.0, 600.0))
        assert second['table_band_excess'] == {} and second['converged'] is True

    def test_layer_mixtures_take_precedence_over_the_config(self, mats):
        """The mixtures the solve used choose the tables, not the config string."""
        md, melt, _, _ = mats
        res = _result(HOT)
        res['layer_eos_config']['mantle'] = 'Seager2007:MgSiO3'
        assert _check(res, md, _curves(500.0, 600.0))['table_band_excess'] == {}
        mixtures = parse_all_layer_mixtures({'core': 'Seager2007:iron', 'mantle': 'X:MgSiO3'})
        assert melt in _check(res, md, _curves(500.0, 600.0), mixtures)['table_band_excess']

    def test_mantle_without_regular_tables_reports_nothing(self, mats):
        """A mantle of a non-tabulated EOS, or no mantle, gives an empty report."""
        md, _, _, _ = mats
        res = _result([0, 9000.0, 9000.0, 9000.0], adiabat=True)
        res['layer_eos_config']['mantle'] = 'Seager2007:MgSiO3'
        assert _check(res, md, None)['table_band_excess'] == {}
        res['layer_eos_config']['mantle'] = ''
        assert _check(res, md, None)['converged_bands'] is None  # not checked

    @pytest.mark.parametrize('bad', [-1.0, float('nan'), float('inf'), True, 'abc', None])
    def test_validator_rejects_negative_nan_and_text(self, bad):
        """max_band_excess must be a finite number >= 0, not a bool; the message names it."""
        cfg = load_zalmoxis_config(os.path.join(get_zalmoxis_root(), 'input', 'default.toml'))
        assert cfg['max_band_excess'] == 200.0 and band_limit({'max_band_excess': '7'}) == 7.0
        cfg['max_band_excess'] = bad
        with pytest.raises(ValueError, match=f'finite number >= 0, got {bad!r}'):
            validate_config(cfg)
        with pytest.raises(ValueError, match='max_band_excess must be a finite number >= 0'):
            _report_table_bands({}, cfg, {}, None, {})

    def test_node_within_the_layer_tolerance_is_mantle(self, mats):
        """A node 1e-13 below cmb_mass is mantle, as in structure_model.get_layer_mixture."""
        md, melt, _, _ = mats
        res = _check(
            _result(HOT, m=(1.0, 2.0 * (1 - 1e-13), 3.0, 4.0)), md, _curves(100.0, 200.0)
        )
        assert res['table_band_excess'][melt][0] == 2

    def test_non_finite_gradient_node_above_the_mantle_fails(self, mats):
        """The surface-anchored adiabat reads the first ice node; a NaN there fails the check."""
        md, _, grad, _ = mats
        res = _result(
            [0, 2000.0, 2000.0, np.nan],
            m=(1.0, 2.0, 3.0, 4.0),
            ice='Seager2007:H2O',
            adiabat=True,
        )
        res['core_mantle_mass'] = 3.5
        _check(res, md, _curves(100.0, 200.0))
        assert res['table_band_excess'][grad] == (1, np.inf) and res['converged'] is False

    def test_main_rejects_a_bad_limit_before_the_solve(self, monkeypatch):
        """A non-numeric max_band_excess stops main() before any solve runs."""
        monkeypatch.setattr(solver, '_solve', lambda *a, **k: pytest.fail('solve ran'))
        cfg = {'layer_eos_config': {'mantle': 'Seager2007:MgSiO3'}, 'max_band_excess': 'x'}
        with pytest.raises(ValueError, match='max_band_excess must be a finite number >= 0'):
            solver.main(cfg, {}, None, '/tmp')

    def test_empty_mantle_and_missing_profile(self, mats):
        """No mantle node gives a checked, empty count; a result without a profile is not checked."""
        md, melt, _, _ = mats
        res = _check(_result(HOT, m=(1.0, 1.0, 1.0, 1.0)), md, _curves(100.0, 200.0))
        assert res['table_band_excess'][melt] == (0, 0.0) and res['converged_bands'] is None
        res = _check({'layer_eos_config': _result(HOT)['layer_eos_config']}, md, None)
        assert res['table_band_excess'] == {} and res['converged_bands'] is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ('mode', 'ice', 'surface_p'),
    [
        ('adiabatic', '', 0.0),
        ('adiabatic', 'Seager2007:H2O', 0.0),
        ('adiabatic', '', 1e5),
        ('adiabatic_from_cmb', '', 0.0),
        ('adiabatic_from_cmb', '', 1e5),
        ('adiabatic_from_cmb', 'Seager2007:H2O', 0.0),
    ],
    ids=['surface', 'surface_ice', 'surface_1bar', 'cmb', 'cmb_1bar', 'cmb_ice'],
)
def test_gradient_nodes_are_the_adiabat_reads(mats, monkeypatch, mode, ice, surface_p):
    """The gradient entry covers exactly the nodes compute_adiabatic_temperature reads."""
    md, _, grad, _ = mats
    md = {'WolfBower2018:MgSiO3': md['X:MgSiO3']}
    layers = {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3', 'ice_layer': ice}
    mixtures = parse_all_layer_mixtures(layers)
    m = np.arange(1.0, 11.0)
    P = np.append(np.linspace(9e9, 1e9, 9), surface_p)
    adiabat_reads, check_reads = [], []
    monkeypatch.setattr(
        mixing, 'get_mixed_nabla_ad', lambda p, *a: adiabat_reads.append(p) or 0.1
    )
    compute_adiabatic_temperature(
        m,
        P,
        m,
        3000.0,
        3.5,
        8.5,
        mixtures,
        md,
        {},
        *_curves(100.0, 200.0),
        anchor='cmb' if mode == 'adiabatic_from_cmb' else 'surface',
        cmb_temperature=5000.0,
    )

    def spy(cached, p, temperature):
        if cached is solver._interpolation_cache[grad]:
            check_reads.append(p)
        return 1.0

    monkeypatch.setattr(seager, 'band_excess', spy)
    res = {
        **_result(np.full(10, 9000.0), P=P, m=m, ice=ice, adiabat=True),
        'layer_eos_config': layers,
    }
    res.update(cmb_mass=3.5, core_mantle_mass=8.5)
    _report_table_bands(
        res,
        {'temperature_mode': mode, 'layer_eos_config': layers},
        md,
        _curves(100.0, 200.0),
        mixtures,
    )
    assert sorted(check_reads) == sorted(adiabat_reads) and adiabat_reads


@pytest.mark.unit
@pytest.mark.parametrize('outer', ['picard', 'newton'])
def test_main_checks_the_profile_with_the_mixtures_it_got(mats, monkeypatch, caplog, outer):
    """main() runs the check on both outer solvers with the mixtures passed in; a band failure
    leaves the numerical convergence flags and accepted_on as they are. A Newton solve that kept
    the linear profile in an adiabatic mode logs a warning."""
    md, melt, grad, _ = mats
    res = {**_result(HOT, adiabat=True), 'converged_mass': True, 'converged_density': True}
    res['accepted_on'] = 'strict'
    newton_res = {**res, 'adiabat_computed': False}
    monkeypatch.setattr(solver, '_solve', lambda *a, **k: dict(res))
    monkeypatch.setattr(solver, '_solve_newton_outer', lambda *a, **k: dict(newton_res))
    cfg = {
        'layer_eos_config': {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'},
        'temperature_mode': 'adiabatic',
        'outer_solver': outer,
    }
    mixtures = parse_all_layer_mixtures({'core': 'Seager2007:iron', 'mantle': 'X:MgSiO3'})
    with caplog.at_level(logging.WARNING, logger='fwl.zalmoxis.solver'):
        out = solver.main(cfg, md, _curves(500.0, 600.0), '/tmp', layer_mixtures=mixtures)
    tables = {melt, md['X:MgSiO3']['solid_mantle']['eos_file']}
    assert set(out['table_band_excess']) == (tables if outer == 'newton' else tables | {grad})
    blend = [r for r in caplog.records if 'accepted a temperature profile' in r.getMessage()]
    assert not blend  # the band check took the accept back, so no adiabat warning
    assert out['converged'] is False and out['converged_bands'] is False
    assert out['converged_mass'] is True and out['converged_density'] is True
    assert out['accepted_on'] == 'strict'


WB = EOS_REGISTRY['WolfBower2018:MgSiO3']
CUT_LOGS = ('Wall-clock timeout', 'Retrying with tighter')  # a clock-dependent solve end


@pytest.mark.unit
def test_cut_logs_are_solver_messages():
    """Each text the Wolf & Bower solves must not log is in a logger call of the solver."""
    with open(solver.__file__) as f:
        calls = [n for n in ast.walk(ast.parse(f.read())) if isinstance(n, ast.Call)]
    logged = [
        a.value
        for c in calls
        if isinstance(c.func, ast.Attribute) and getattr(c.func.value, 'id', '') == 'logger'
        for a in c.args
        if isinstance(a, ast.Constant) and isinstance(a.value, str)
    ]
    assert [sum(text in s for s in logged) for text in CUT_LOGS] == [
        2,
        1,
    ]  # outer, inner; retry


def _need_wb_tables():
    """Skip without the Wolf & Bower tables; fail if ZALMOXIS_NEED_WOLFBOWER is non-empty."""
    for phase in ('melted_mantle', 'solid_mantle'):
        if not os.path.isfile(WB[phase]['eos_file']):
            stop = pytest.fail if os.environ.get('ZALMOXIS_NEED_WOLFBOWER') else pytest.skip
            stop(f'Wolf & Bower table missing: {WB[phase]["eos_file"]}')


@pytest.mark.unit
@pytest.mark.parametrize('phase', ['melted_mantle', 'solid_mantle'])
@pytest.mark.parametrize(
    'env, outcome',
    [
        (None, pytest.skip.Exception),
        ('', pytest.skip.Exception),
        ('1', pytest.fail.Exception),
        ('0', pytest.fail.Exception),
    ],
)
def test_missing_tables_skip_or_fail(monkeypatch, phase, env, outcome):
    """Without a table the solves skip, or fail where ZALMOXIS_NEED_WOLFBOWER is non-empty."""
    monkeypatch.setitem(WB[phase], 'eos_file', '/nonexistent/table.dat')
    monkeypatch.delenv('ZALMOXIS_NEED_WOLFBOWER', raising=False)
    if env is not None:
        monkeypatch.setenv('ZALMOXIS_NEED_WOLFBOWER', env)
    with pytest.raises((pytest.skip.Exception, pytest.fail.Exception)) as stop:
        _need_wb_tables()
    assert stop.type is outcome


def _wb_solve(tmp_dir, surface_temperature, **params):
    """Solve 1 M_earth with a Wolf & Bower mantle; return the result and the band warnings.

    The wall-clock limit is far above the solve time, so the iteration limits end the
    solve on any runner.
    """
    from zalmoxis.config import load_material_dictionaries, load_solidus_liquidus_functions

    _need_wb_tables()
    root = get_zalmoxis_root()
    text = open(os.path.join(root, 'input', 'default.toml')).read()
    for key, value in (
        ('mantle', '"WolfBower2018:MgSiO3"'),
        ('data_enabled', 'false'),
        ('surface_temperature', str(surface_temperature)),
    ):
        text = re.sub(rf'(?m)^{key}\s*=.*$', f'{key} = {value}', text)
    path = tmp_dir / 'wb.toml'
    path.write_text(text)
    cfg = load_zalmoxis_config(str(path))
    cfg.update({'wall_timeout': 3600.0, **params})
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        cfg['rock_solidus'],
        cfg['rock_liquidus'],
        cfg['mushy_zone_factor'],
    )
    records = []
    handler = logging.Handler()
    handler.emit = records.append
    logging.getLogger('fwl.zalmoxis.solver').addHandler(handler)
    try:
        res = solver.main(
            cfg,
            material_dictionaries=load_material_dictionaries(),
            melting_curves_functions=melt,
            input_dir=os.path.join(root, 'input'),
        )
    finally:
        logging.getLogger('fwl.zalmoxis.solver').removeHandler(handler)
    msgs = [r.getMessage() for r in records]
    cut = [m for m in msgs if any(text in m for text in CUT_LOGS)]
    assert not cut, cut
    return res, [m for m in msgs if 'valid T band' in m]


@pytest.fixture(scope='module')
def wb_solves(tmp_path_factory):
    """Solve at 3871 K, then at 4500 K, in one process."""
    return {ts: _wb_solve(tmp_path_factory.mktemp('wb'), float(ts)) for ts in ('3871', '4500')}


@pytest.mark.integration
@pytest.mark.wolfbower
class TestWolfBowerSolves:
    """Real solves: inside the limit the clamp stays, beyond it the solve fails."""

    def test_inside_the_limit_converges_and_reports(self, wb_solves):
        """At 3871 K the molten mantle lies up to about 58 K beyond the melt band."""
        res, msgs = wb_solves['3871']
        n, excess = res['table_band_excess'][WB['melted_mantle']['eos_file']]
        assert res['converged'] is True and res['converged_bands'] is True
        assert n > 0 and excess == pytest.approx(3871.0 - 3815.3, abs=1.0)  # the 1-bar node
        assert res['adiabat_computed'] is True
        assert res['table_band_excess'][WB['melted_mantle']['adiabat_grad_file']][0] > 0
        assert len(msgs) == 1

    def test_beyond_the_limit_returns_not_converged(self, wb_solves):
        """At 4500 K the excess passes 200 K and the solve returns converged=False."""
        res, msgs = wb_solves['4500']
        melt = WB['melted_mantle']['eos_file']
        n, excess = res['table_band_excess'][melt]
        P, T = res['pressure'][-1], res['temperature'][-1]
        surface = seager.band_excess(solver._interpolation_cache[melt], P, T)
        assert surface == pytest.approx(4500.0 - 3815.3, abs=1.0)  # the 1-bar node
        # an inner node of the hot adiabat can lie further outside than the 1-bar node
        assert n > 0 and excess >= surface > band_limit({})
        assert res['converged'] is False and res['converged_bands'] is False
        assert res['converged_pressure'] and res['converged_density']  # numerically converged

    def test_second_solve_in_the_process_reports_too(self, wb_solves):
        """The 4500 K solve after the 3871 K one logs its own band warning, with its counts."""
        res, msgs = wb_solves['4500']
        n, excess = res['table_band_excess'][WB['melted_mantle']['eos_file']]
        assert len(msgs) == 1 and f'density_melt.dat {n}, {excess:.0f}' in msgs[0]
