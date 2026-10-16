"""Band report of the converged mantle profile against the valid T band of regular tables.

Module under test: ``zalmoxis.solver._report_table_bands`` and its call from
``zalmoxis.solver.main``. Contract: each table that a mantle node reads gets the count
of nodes outside its band and the largest excess in K; one warning per solve names the
tables; an excess above ``max_band_excess`` returns ``converged=False``. The unit tests
use the synthetic fill table of ``tests/test_seager_branches.py``; the integration tests
solve a 1 M_earth Wolf & Bower mantle. See docs/How-to/build_tests.md for the test tiers.
"""

from __future__ import annotations

import logging
import os
import re

import numpy as np
import pytest

from tests.test_seager_branches import _fill_table, _write_table
from zalmoxis import solver
from zalmoxis.config import validate_config
from zalmoxis.eos_properties import EOS_REGISTRY
from zalmoxis.solver import _report_table_bands


@pytest.fixture
def mats(tmp_path, monkeypatch):
    """Registry entry with the fill table as melt, solid and gradient table."""
    monkeypatch.setattr(solver, '_interpolation_cache', {})
    melt, grad, solid = (
        _write_table(tmp_path / name, _fill_table())['melted_mantle']['eos_file']
        for name in ('density_melt.dat', 'grad.dat', 'density_solid.dat')
    )
    return (
        {
            'X:MgSiO3': {
                'melted_mantle': {'eos_file': melt, 'adiabat_grad_file': grad},
                'solid_mantle': {'eos_file': solid},
            }
        },
        melt,
        grad,
        solid,
    )


def _result(T, P=(0.0, 1e9, 2e9, 3e9), m=(1.0, 2.0, 3.0, 4.0), ice=''):
    """A converged result whose nodes 1 to 3 (m >= 2) are mantle."""
    return {
        'layer_eos_config': {'core': 'Seager2007:iron', 'mantle': 'X:MgSiO3', 'ice_layer': ice},
        'mass_enclosed': np.array(m),
        'pressure': np.array(P),
        'temperature': np.array(T, dtype=float),
        'cmb_mass': 2.0,
        'core_mantle_mass': 3.5,
        'converged': True,
    }


def _curves(t_sol, t_liq):
    """Constant solidus and liquidus functions."""
    return (lambda p: t_sol, lambda p: t_liq)


def _check(res, cfg, md, curves, adiabat):
    """Run the band check with the layers of ``res`` in the config, as main() passes them."""
    return _report_table_bands(
        res, {'layer_eos_config': res['layer_eos_config'], **cfg}, md, curves, adiabat
    )


@pytest.mark.unit
class TestReportTableBands:
    """Counts, excess, table use per node, warning and the converged flag."""

    def test_melt_nodes_report_count_and_largest_excess(self, mats, caplog):
        """Molten nodes read melt and gradient tables; 3300 K at 1 GPa is 300 K beyond."""
        md, melt, grad, solid = mats
        with caplog.at_level(logging.WARNING, logger='zalmoxis.solver'):
            res = _check(
                _result([0, 3300.0, 2000.0, 3600.0]), {}, md, _curves(500.0, 600.0), True
            )
        assert res['table_band_excess'][melt] == (2, pytest.approx(300.0))
        assert res['table_band_excess'][grad] == (2, pytest.approx(300.0))
        assert res['table_band_excess'][solid] == (0, 0.0)
        assert res['converged'] is False  # 300 K > 200 K
        msgs = [r.getMessage() for r in caplog.records]
        assert len(msgs) == 2
        assert 'density_melt.dat 2, 300' in msgs[0] and 'value there is taken' in msgs[0]
        assert 'max_band_excess = 200 K' in msgs[1]

    def test_limit_parameter_keeps_converged(self, mats, caplog):
        """With max_band_excess = 300 K the same profile stays converged and warns once."""
        md, melt, _, _ = mats
        with caplog.at_level(logging.WARNING, logger='zalmoxis.solver'):
            res = _check(
                _result([0, 3300.0, 2000.0, 3600.0]),
                {'max_band_excess': 300.0},
                md,
                _curves(500.0, 600.0),
                False,
            )
        assert res['converged'] is True
        assert len(caplog.records) == 1
        assert all('grad' not in os.path.basename(k) for k in res['table_band_excess'])

    def test_inside_every_band_is_silent(self, mats, caplog):
        """A profile inside every band reports zeros and logs nothing."""
        md, melt, grad, solid = mats
        with caplog.at_level(logging.WARNING, logger='zalmoxis.solver'):
            res = _check(
                _result([9999.0, 2000.0, 2000.0, 2000.0]), {}, md, _curves(1800.0, 2200.0), True
            )
        assert res['table_band_excess'] == {melt: (0, 0.0), solid: (0, 0.0), grad: (0, 0.0)}
        assert res['converged'] is True and not caplog.records

    @pytest.mark.parametrize(
        ('t_sol', 't_liq', 'melt_n', 'solid_n'),
        [
            (5000.0, 6000.0, 0, 2),  # solid: only the solid table
            (100.0, 200.0, 2, 0),  # molten: only the melt table
            (100.0, 6000.0, 2, 2),  # mushy: both
            (np.nan, np.nan, 0, 2),  # curves undefined: solid
            (6000.0, 100.0, 0, 2),  # degenerate curves below the solidus: solid
        ],
    )
    def test_phase_selects_the_tables(self, mats, t_sol, t_liq, melt_n, solid_n):
        """The melt table counts above the solidus, the solid table where it is read."""
        md, melt, _, solid = mats
        res = _check(
            _result([0, 3300.0, 2000.0, 3600.0]),
            {'max_band_excess': 1e9},
            md,
            _curves(t_sol, t_liq),
            False,
        )
        assert res['table_band_excess'][melt][0] == melt_n
        assert res['table_band_excess'][solid][0] == solid_n

    @pytest.mark.parametrize(
        ('pressures', 'ice', 'expected'),
        [
            ((1e9, 1e9, 2e9, 3e9), 'Seager2007:H2O', (1, 300.0)),  # node 3 is ice
            ((1e9, 1e9, 2e9, 0.0), '', (1, 300.0)),  # node 3 is the P = 0 padding
            ((1e9, 1e9, 2e9, 3e9), '', (2, 5500.0)),  # node 3 is mantle: 9000 - 3500 K
        ],
    )
    def test_core_ice_and_padding_nodes_are_left_out(self, mats, pressures, ice, expected):
        """Core nodes, ice nodes above core_mantle_mass and P = 0 nodes are not mantle."""
        md, melt, _, _ = mats
        res = _check(
            _result([9000.0, 3300.0, 2000.0, 9000.0], P=pressures, ice=ice),
            {'max_band_excess': 1e9},
            md,
            _curves(100.0, 200.0),
            False,
        )
        assert res['table_band_excess'][melt] == (expected[0], pytest.approx(expected[1]))

    def test_table_not_in_cache_is_loaded(self, mats):
        """A table no lookup has read yet is loaded for the check."""
        md, melt, _, _ = mats
        assert melt not in solver._interpolation_cache
        _check(_result([0, 2000.0, 2000.0, 2000.0]), {}, md, None, False)
        assert 't_lo' in solver._interpolation_cache[melt]

    def test_mantle_without_regular_tables_reports_nothing(self, mats):
        """A mantle of a non-tabulated EOS, or no mantle, gives an empty report."""
        md, _, _, _ = mats
        res = _result([0, 9000.0, 9000.0, 9000.0])
        res['layer_eos_config']['mantle'] = 'Seager2007:MgSiO3'
        assert _check(res, {}, md, None, True)['table_band_excess'] == {}
        res['layer_eos_config']['mantle'] = ''
        assert _check(res, {}, md, None, True)['converged'] is True

    @pytest.mark.parametrize('bad', [-1.0, float('nan')])
    def test_validator_rejects_negative_and_nan(self, bad):
        """max_band_excess must be a number >= 0."""
        from zalmoxis.config import load_zalmoxis_config

        cfg = load_zalmoxis_config()
        assert cfg['max_band_excess'] == 200.0
        cfg['max_band_excess'] = bad
        with pytest.raises(ValueError, match='max_band_excess must be >= 0'):
            validate_config(cfg)


WB = EOS_REGISTRY['WolfBower2018:MgSiO3']


@pytest.fixture(scope='module')
def wb_solves(tmp_path_factory):
    """Solve 1 M_earth with a Wolf & Bower mantle at 3871 K, then 4500 K, in one process."""
    from zalmoxis import get_zalmoxis_root
    from zalmoxis.config import (
        load_material_dictionaries,
        load_solidus_liquidus_functions,
        load_zalmoxis_config,
    )

    for phase in ('melted_mantle', 'solid_mantle'):
        if not os.path.isfile(WB[phase]['eos_file']):
            pytest.skip(f'Wolf & Bower table missing: {WB[phase]["eos_file"]}')
    root = get_zalmoxis_root()
    out = {}
    for ts in (3871.0, 4500.0):
        text = open(os.path.join(root, 'input', 'default.toml')).read()
        for key, value in (
            ('mantle', '"WolfBower2018:MgSiO3"'),
            ('data_enabled', 'false'),
            ('surface_temperature', str(ts)),
        ):
            text = re.sub(rf'(?m)^{key}\s*=.*$', f'{key} = {value}', text)
        path = tmp_path_factory.mktemp('wb') / 'wb.toml'
        path.write_text(text)
        cfg = load_zalmoxis_config(str(path))
        melt = load_solidus_liquidus_functions(
            cfg['layer_eos_config'],
            cfg['rock_solidus'],
            cfg['rock_liquidus'],
            cfg['mushy_zone_factor'],
        )
        records = []
        handler = logging.Handler()
        handler.emit = records.append
        logging.getLogger('zalmoxis.solver').addHandler(handler)
        try:
            res = solver.main(
                cfg,
                material_dictionaries=load_material_dictionaries(),
                melting_curves_functions=melt,
                input_dir=os.path.join(root, 'input'),
            )
        finally:
            logging.getLogger('zalmoxis.solver').removeHandler(handler)
        out[ts] = (res, [r.getMessage() for r in records if 'valid T band' in r.getMessage()])
    return out


@pytest.mark.integration
@pytest.mark.timeout(900)
class TestWolfBowerSolves:
    """Real solves: inside the limit the clamp stays, beyond it the solve fails."""

    def test_inside_the_limit_converges_and_reports(self, wb_solves):
        """At 3871 K the molten mantle lies up to about 58 K beyond the melt band."""
        res, msgs = wb_solves[3871.0]
        n, excess = res['table_band_excess'][WB['melted_mantle']['eos_file']]
        assert res['converged'] is True
        assert n > 0 and 0.0 < excess <= 200.0
        assert len(msgs) == 1

    def test_beyond_the_limit_returns_not_converged(self, wb_solves):
        """At 4500 K the excess passes 200 K and the solve returns converged=False."""
        res, msgs = wb_solves[4500.0]
        _, excess = res['table_band_excess'][WB['melted_mantle']['eos_file']]
        assert excess > 200.0
        assert res['converged'] is False

    def test_second_solve_in_the_process_reports_too(self, wb_solves):
        """The 4500 K solve after the 3871 K one logs its own band warning."""
        assert len(wb_solves[4500.0][1]) == 1
        assert wb_solves[4500.0][1] != wb_solves[3871.0][1]
