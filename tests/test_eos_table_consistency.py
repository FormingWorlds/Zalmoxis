"""Thermodynamic consistency of regular EOS tables.

Module under test: ``zalmoxis.eos.consistency``. The analytic tables of
``tests/_analytic_eos.py`` hold every identity exactly, so their residuals are the truncation of
the central differences alone. See docs/How-to/build_tests.md for the test tiers.
"""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests import _analytic_eos as a
from zalmoxis.eos import consistency as c

LP = np.arange(5.0, 11.0 + 1e-9, 1 / 150)  # the PALEOS grid: 150 points per decade
LT = np.arange(np.log10(500.0), np.log10(5000.0) + 1e-9, 1 / 150)


@pytest.fixture(scope='module')
def murnaghan():
    return a.table(a.murnaghan, LP, LT)


@pytest.mark.unit
@pytest.mark.parametrize('name', ['cp', 'maxwell', 'alpha', 'dudt', 'cp_cv'])
def test_the_residual_of_an_exact_table_is_the_h2_truncation(murnaghan, name):
    """R_2h / R_h is 4, and the Richardson value R_x = (4 R_h - R_2h) / 3 keeps an h^4 rest."""
    r1, r2, r4 = (c.residuals(murnaghan, k)[name] for k in (1, 2, 4))
    assert np.nanmedian(r2 / r1) == pytest.approx(4.0, abs=1e-2)
    x1, x2 = (4 * r1 - r2) / 3, (4 * r2 - r4) / 3
    assert np.nanmedian(x2 / x1) == pytest.approx(16.0, abs=0.1)


@pytest.mark.unit
def test_an_algebraic_identity_holds_to_round_off(murnaghan):
    assert np.nanmax(np.abs(c.residuals(murnaghan)['nabla_ad'])) < 1e-14


@pytest.mark.unit
def test_a_stencil_never_crosses_a_phase_boundary(murnaghan):
    t = {**murnaghan, 'phase': murnaghan['phase'].copy()}
    t['phase'][:, 80:] = 'liquid'
    r = c.residuals(t, 2)['cp']
    assert (
        np.isnan(r[:, 78:82]).all()
        and np.isfinite(r[:, 76]).all()
        and np.isfinite(r[:, 84]).all()
    )


@pytest.mark.unit
def test_a_cell_without_a_value_ends_the_stencils_around_it(murnaghan):
    t = {**murnaghan, 's': murnaghan['s'].copy()}
    t['s'][400, 75] = np.nan
    r = c.residuals(t, 1)['cp']
    assert np.isnan(r[400, 74:77]).all() and np.isfinite(r[400, 73]) and np.isfinite(r[399, 75])


@pytest.mark.unit
@pytest.mark.parametrize('size', [1e-6, 1e-4, 1e-2])
def test_a_planted_entropy_error_shows_in_the_richardson_residual(murnaghan, size):
    t = {**murnaghan, 's': murnaghan['s'].copy()}
    t['s'][400, 75] *= 1 + size
    r1, r2 = c.residuals(t, 1)['cp'], c.residuals(t, 2)['cp']
    rx = np.abs(4 * r1 - r2) / 3
    exact = c.residuals(murnaghan, 1)['cp'], c.residuals(murnaghan, 2)['cp']
    assert np.nanmax(rx[400, 72:79]) > 100 * np.nanmax(np.abs(4 * exact[0] - exact[1]) / 3)


@pytest.mark.unit
def test_a_grid_shorter_than_the_stencil_gives_no_derivative():
    seg = np.ones((4, 3), int)
    assert np.isnan(c.derivative(np.ones((4, 3)), seg, 1, 2, 0.1)).all()


EPS9 = 5e-9  # 9 significant digits, the precision of the shipped tables


@pytest.fixture(scope='module')
def stored():
    return {
        n: a.table(f, LP, LT, digits=9)
        for n, f in (('murnaghan', a.murnaghan), ('ideal', a.ideal_gas))
    }


@pytest.mark.unit
@pytest.mark.parametrize('name', ['murnaghan', 'ideal'])
def test_an_exact_table_at_9_digits_has_no_failing_cell(stored, name):
    res = c.check(stored[name], EPS9)
    assert {n: int(v['fail'].sum()) for n, v in res.items()} == dict.fromkeys(c.IDENTITIES, 0)
    assert all(v['tested'].sum() > 100_000 for v in res.values())


@pytest.mark.unit
@pytest.mark.parametrize(
    'col, name, axis, size',
    [
        ('s', 'cp', 1, 1e-4),
        ('rho', 'cp_cv', 0, 1e-4),
        ('s', 'cp', 1, 1e-6),
        ('rho', 'cp_cv', 0, 1e-6),
        ('s', 'maxwell', 0, 1e-6),
        ('rho', 'alpha', 1, 1e-6),
        ('u', 'dudt', 1, 1e-6),
    ],
)
def test_a_planted_value_fails_one_and_two_steps_away(stored, col, name, axis, size):
    t = {**stored['murnaghan'], col: stored['murnaghan'][col].copy()}
    t[col][400, 75] *= 1 + size
    fail = c.check(t, EPS9)[name]['fail']
    at = {o: bool(fail[(400 + o, 75) if axis == 0 else (400, 75 + o)]) for o in (-2, -1, 1, 2)}
    assert at == dict.fromkeys((-2, -1, 1, 2), True)
    assert int(fail.sum()) <= 6  # only the cells whose stencil holds the planted value


@pytest.mark.unit
def test_the_resolution_one_step_away_is_2e_8(stored):
    """A planted s of 2e-8 is found one step away and one of 1e-8 is not (eps 5e-9)."""
    for size, found in ((2e-8, True), (1e-8, False)):
        t = {**stored['murnaghan'], 's': stored['murnaghan']['s'].copy()}
        t['s'][400, 75] *= 1 + size
        assert bool(c.check(t, EPS9)['cp']['fail'][400, 76]) is found, size


@pytest.mark.unit
def test_nabla_ad_is_tested_against_its_floor(stored):
    t = {**stored['murnaghan'], 'nabla_ad': stored['murnaghan']['nabla_ad'].copy()}
    t['nabla_ad'][300, 50] *= 1 + 1e-6
    fail = c.check(t, EPS9)['nabla_ad']['fail']
    assert fail[300, 50] and int(fail.sum()) == 1


@pytest.mark.unit
def test_a_coarse_precision_hides_a_fault_the_9_digit_floor_finds(stored):
    t = {**stored['murnaghan'], 's': stored['murnaghan']['s'].copy()}
    t['s'][400, 75] *= 1 + 1e-6
    assert c.check(t, 0.0)['cp']['fail'][400, 76]
    assert not c.check(t, 1e-4)['cp']['fail'][400, 76]


@pytest.mark.unit
def test_a_table_off_a_uniform_log_grid_is_rejected(stored):
    t = {
        k: np.delete(v, 10, axis=0)
        for k, v in stored['murnaghan'].items()
        if k != 'unique_log_t'
    }
    t['unique_log_t'] = stored['murnaghan']['unique_log_t']
    with pytest.raises(ValueError, match='uniform'):
        c.check(t, EPS9)


@pytest.mark.unit
def test_a_smooth_drift_of_the_grid_step_is_rejected(stored):
    m = stored['murnaghan']
    lp = m['unique_log_p'] + 1e-5 * np.sin(np.arange(m['unique_log_p'].size) / 50)
    with pytest.raises(ValueError, match='uniform'):
        c.check({**m, 'unique_log_p': lp}, EPS9)


@pytest.mark.unit
def test_a_drift_of_2e_6_of_the_step_is_rejected(stored):
    """A smooth drift with a step spread of 2e-6 gives false maxwell failures on an exact
    table checked with the mean step; it departs from a linear fit by 8e-7 in log10 P."""
    m, i = stored['murnaghan'], np.arange(stored['murnaghan']['unique_log_p'].size)
    lp = m['unique_log_p'] + 1e-4 * np.sin(i / 100) / 150
    with pytest.raises(ValueError, match='uniform'):
        c.check({**m, 'unique_log_p': lp}, EPS9)


@pytest.mark.unit
@pytest.mark.parametrize('ppd', [600, 1000, 2000])
def test_9_digit_coordinates_pass_at_every_density(ppd):
    lp, lt = np.arange(8.0, 9.0 + 1e-9, 1 / ppd), np.arange(3.0, 3.3 + 1e-9, 1 / ppd)
    t = a.table(a.murnaghan, lp, lt, digits=9)
    r9 = np.vectorize(lambda x: float(f'{x:.8e}'))
    t['unique_log_p'], t['unique_log_t'] = np.log10(r9(10.0**lp)), np.log10(r9(10.0**lt))
    assert not any(v['fail'].any() for v in c.check(t, EPS9).values())


@pytest.mark.unit
def test_coordinates_stored_at_9_digits_pass(stored):
    m = stored['murnaghan']
    r9 = np.vectorize(lambda x: float(f'{x:.8e}'))
    t = {
        **m,
        'unique_log_p': np.log10(r9(10.0 ** m['unique_log_p'])),
        'unique_log_t': np.log10(r9(10.0 ** m['unique_log_t'])),
    }
    assert not any(v['fail'].any() for v in c.check(t, EPS9).values())


@pytest.mark.unit
@pytest.mark.parametrize('change', ['reverse', 'nan', 'repeat'])
def test_an_axis_that_is_not_finite_and_increasing_is_rejected(stored, change):
    m = stored['murnaghan']
    lp = m['unique_log_p'][::-1] if change == 'reverse' else m['unique_log_p'].copy()
    if change == 'nan':
        lp[10] = np.nan
    if change == 'repeat':
        lp[11] = lp[10]
    with pytest.raises(ValueError, match='finite and strictly increasing'):
        c.check({**m, 'unique_log_p': lp}, EPS9)


@pytest.mark.unit
@pytest.mark.parametrize('key', ['u', 'phase'])
def test_a_column_off_the_shape_of_the_grid_is_rejected(stored, key):
    m = stored['murnaghan']
    with pytest.raises(ValueError, match='does not have the shape of the'):
        c.check({**m, key: m[key][:-1]}, EPS9)


@pytest.mark.unit
def test_a_check_that_tests_no_cell_raises(stored):
    res = c.check(
        a.table(a.murnaghan, LP[:8], LT[:8], digits=9), EPS9
    )  # too short to differentiate
    assert {n for n, v in res.items() if v['skipped']} == set(c.IDENTITIES) - {'nabla_ad'}
    t = {**stored['murnaghan'], 'rho': np.full_like(stored['murnaghan']['rho'], np.nan)}
    with pytest.raises(ValueError, match='no cell'):
        c.check(t, EPS9)


@pytest.mark.unit
def test_an_identity_that_cannot_be_tested_is_skipped_and_the_others_run(stored):
    t = {**stored['murnaghan'], 'alpha': np.zeros_like(stored['murnaghan']['alpha'])}
    try:
        res = c.check(t, EPS9)
    except ValueError as err:
        pytest.fail(f'a table with testable identities raised: {err}')
    skipped = {n for n, v in res.items() if v['skipped']}
    assert skipped == {'maxwell', 'alpha', 'nabla_ad', 'cp_cv'}
    assert not any(res[n]['fail'].any() for n in skipped)
    assert not res['cp']['skipped'] and res['cp']['tested'].sum() > 100_000


@pytest.mark.unit
def test_a_one_point_axis_is_rejected_by_name(stored):
    m = stored['murnaghan']
    t = {k: v[:1] if k != 'unique_log_t' else v for k, v in m.items()}
    with pytest.raises(ValueError, match='unique_log_p needs at least 2 points'):
        c.check(t, EPS9)


@pytest.mark.unit
def test_signs_rejects_a_reversed_pressure_axis(stored):
    m = stored['murnaghan']
    t = {k: v[::-1] if k != 'unique_log_t' else v for k, v in m.items()}
    with pytest.raises(ValueError, match='strictly increasing'):
        c.signs(t, EPS9)


@pytest.mark.unit
def test_signs_rejects_a_column_off_the_shape_of_the_grid(stored):
    m = stored['murnaghan']
    with pytest.raises(ValueError, match='shape'):
        c.signs({**m, 'cv': m['cv'][:1]}, EPS9)


@pytest.mark.unit
@pytest.mark.parametrize('name', ['cp', 'maxwell', 'alpha', 'dudt', 'cp_cv'])
def test_the_bound_is_the_floor_plus_twice_the_h4_term(stored, name):
    m = stored['murnaghan']
    r = {k: c.residuals(m, k)[name] for k in (1, 2, 4)}
    f1, f2 = (c.floors(m, EPS9, k)[name] for k in (1, 2))
    rx = (4 * r[1] - r[2]) / 3
    est = np.abs((4 * r[2] - r[4]) / 3 - rx) / 15
    np.testing.assert_allclose(
        c.check(m, 0.0)[name]['bound'], (4 * f1 + f2) / 3 + 2 * est, rtol=1e-12
    )
    assert c.EPS_FLOOR == 5e-9


@pytest.mark.unit
def test_a_factor_of_1_misses_a_fault_two_cells_away_that_2_finds(stored, monkeypatch):
    t = {**stored['murnaghan'], 's': stored['murnaghan']['s'].copy()}
    t['s'][400, 75] *= 1 + 1e-6
    assert c.check(t, EPS9)['cp']['fail'][400, 77]
    monkeypatch.setattr(c, 'FACTOR', 1.0)
    assert c.check(t, EPS9)['cp']['fail'][400, 77]  # a smaller factor still finds it
    monkeypatch.setattr(c, 'FACTOR', 3.0)
    assert not c.check(t, EPS9)['cp']['fail'][400, 77]


@pytest.mark.unit
def test_an_exact_table_on_a_coarse_grid_passes_by_its_h4_term():
    lp, lt = (
        np.arange(5.0, 11.0 + 1e-9, 1 / 20),
        np.arange(np.log10(500.0), np.log10(5000.0) + 1e-9, 1 / 20),
    )
    res = c.check(a.table(a.murnaghan, lp, lt, digits=9), EPS9)
    assert not any(v['fail'].any() for v in res.values())


@pytest.mark.unit
def test_a_missing_cell_leaves_four_untested_cells_on_each_side(stored):
    t = {**stored['murnaghan'], 's': stored['murnaghan']['s'].copy()}
    t['s'][400, 75] = np.nan
    tested = c.check(t, EPS9)['cp']['tested']
    assert not tested[400, 71:80].any() and tested[400, [70, 80]].all()


@pytest.mark.unit
def test_the_floor_grows_with_the_stored_precision(stored):
    f9, f12 = (c.floors(stored['murnaghan'], e)['cp'] for e in (EPS9, 5e-12))
    assert np.nanmedian(f9 / f12) == pytest.approx(1000.0)


@pytest.mark.unit
def test_an_exact_table_breaks_no_sign_condition(stored):
    res = c.signs(stored['murnaghan'], EPS9)
    assert all(not v['fail'].any() and v['tested'].sum() > 100_000 for v in res.values())


@pytest.mark.unit
@pytest.mark.parametrize(
    'col, value, name',
    [
        ('rho', -1.0, 'rho'),
        ('rho', 0.0, 'rho'),
        ('cp', 0.0, 'cp'),
        ('cv', -5.0, 'cv'),
        ('cv', 2.0e4, 'cp_ge_cv'),
    ],
)
def test_a_cell_that_breaks_a_sign_condition_is_flagged(stored, col, value, name):
    t = {**stored['murnaghan'], col: stored['murnaghan'][col].copy()}
    t[col][300, 50] = value
    assert c.signs(t, EPS9)[name]['fail'][300, 50]


@pytest.mark.unit
def test_a_density_that_falls_with_pressure_breaks_kappa_t(stored):
    t = {**stored['murnaghan'], 'rho': stored['murnaghan']['rho'].copy()}
    t['rho'][301, 50] = t['rho'][299, 50] * (1 - 1e-6)
    flags = c.signs(t, EPS9)['kappa_T']['fail']
    assert flags[300, 50] and int(flags.sum()) == 1


@pytest.mark.unit
@pytest.mark.parametrize('eos', [a.murnaghan, a.ideal_gas])
def test_a_float64_table_is_checked_at_the_9_digit_floor(eos):
    """eps below EPS_FLOOR is raised to it, so an exact float64 table has no failing cell."""
    res = c.check(a.table(eos, LP, LT), np.finfo(float).eps)
    assert sum(int(v['fail'].sum()) for v in res.values()) == 0


_PT_NAMES = {'density': 'rho', 'heat_capacity': 'cp', 'thermal_exp': 'alpha', 'entropy': 's'}


@pytest.fixture(scope='module')
def pt_files(tmp_path_factory):
    """Exact tables written as the Aragog P-T files of eos_export ('%.10e', P slowest)."""
    P, T = np.linspace(1e5, 3.5e11, 200), np.linspace(300.0, 6000.0, 200)  # the PROTEUS grid
    PP, TT = np.meshgrid(P, T, indexing='ij')
    out = {}
    for name, eos in (('murnaghan', a.murnaghan), ('ideal', a.ideal_gas)):
        d, cols, tabs = tmp_path_factory.mktemp(name), eos(PP, TT), {}
        for q, key in _PT_NAMES.items():
            path = d / f'{q}_solid.dat'
            with open(path, 'w') as f:
                f.write(f'#pressure\ttemperature\t{q}\n')
                rows = np.column_stack([PP.ravel(), TT.ravel(), cols[key].ravel()])
                np.savetxt(f, rows, fmt='%.10e', delimiter='\t')
            r = np.loadtxt(path)
            Pu, Tu = np.unique(r[:, 0]), np.unique(r[:, 1])
            tabs[q] = {'P': Pu, 'T': Tu, 'values': r[:, 2].reshape(len(Pu), len(Tu))}
        out[name] = tabs
    return out


@pytest.mark.unit
@pytest.mark.parametrize('name', ['murnaghan', 'ideal'])
def test_an_exact_aragog_pt_table_has_no_failing_cell(pt_files, name):
    res = c.check_pt(pt_files[name], 5e-11)
    assert {n: int(v['fail'].sum()) for n, v in res.items()} == dict.fromkeys(
        c.PT_IDENTITIES, 0
    )
    assert all(v['tested'].sum() > 30_000 for v in res.values())


@pytest.mark.unit
@pytest.mark.parametrize('name', ['murnaghan', 'ideal'])
@pytest.mark.parametrize(
    'quantity, identity, axis',
    [('entropy', 'cp', 1), ('density', 'alpha', 1), ('entropy', 'maxwell', 0)],
)
@pytest.mark.parametrize('size', [1e-4, 1e-6])
def test_a_planted_value_in_aragog_pt_files_fails_one_and_two_steps_away(
    pt_files, name, quantity, identity, axis, size
):
    t = pt_files[name]
    t = {**t, quantity: {**t[quantity], 'values': t[quantity]['values'].copy()}}
    t[quantity]['values'][100, 100] *= 1 + size
    fail = c.check_pt(t, 5e-11)[identity]['fail']
    cells = [(100 + o, 100) if axis == 0 else (100, 100 + o) for o in (-2, -1, 1, 2)]
    assert [bool(fail[x]) for x in cells] == [True] * 4


@pytest.mark.unit
def test_the_fill_value_counts_as_a_missing_cell(pt_files):
    t = pt_files['murnaghan']
    t = {**t, 'entropy': {**t['entropy'], 'values': t['entropy']['values'].copy()}}
    t['entropy']['values'][30, 30] = c.PT_FILL
    res = c.check_pt(t, 5e-11)['cp']
    assert not res['tested'][30, 29:32].any() and not res['fail'].any()


@pytest.mark.unit
def test_aragog_pt_files_on_a_reversed_axis_are_rejected(pt_files):
    t = {
        q: {**v, 'P': v['P'][::-1], 'values': v['values'][::-1]}
        for q, v in pt_files['murnaghan'].items()
    }
    with pytest.raises(ValueError, match='strictly increasing'):
        c.check_pt(t, EPS9)


@pytest.mark.unit
def test_aragog_pt_values_off_the_grid_shape_are_rejected(pt_files):
    t = pt_files['murnaghan']
    t = {**t, 'entropy': {**t['entropy'], 'values': t['entropy']['values'][:, :-1]}}
    with pytest.raises(ValueError, match='entropy values do not have the shape'):
        c.check_pt(t, EPS9)


@pytest.mark.unit
def test_aragog_pt_files_on_different_grids_are_rejected(pt_files):
    t = pt_files['murnaghan']
    t = {**t, 'entropy': {**t['entropy'], 'T': t['entropy']['T'] * 1.01}}
    with pytest.raises(ValueError, match='grid'):
        c.check_pt(t, EPS9)


@pytest.mark.unit
def test_a_coordinate_grid_shorter_than_the_stencil_gives_no_derivative():
    assert np.isnan(c._coord_stencil(np.ones((3, 4)), np.arange(3.0), 0, 2, -1.0)).all()


@pytest.mark.unit
@pytest.mark.parametrize('col', ['rho', 'cp', 'cv', 'u', 's', 'alpha', 'nabla_ad'])
def test_a_missing_value_breaks_no_sign_condition(stored, col):
    t = {**stored['murnaghan'], col: stored['murnaghan'][col].copy()}
    t[col][300, 50] = np.nan
    assert not any(v['fail'][300, 50] for v in c.signs(t, EPS9).values())


@pytest.mark.unit
@pytest.mark.parametrize(
    'col, value, name', [('rho', -np.inf, 'rho'), ('cv', np.inf, 'cp_ge_cv')]
)
def test_an_infinite_value_breaks_its_sign_condition(stored, col, value, name):
    t = {**stored['murnaghan'], col: stored['murnaghan'][col].copy()}
    t[col][300, 50] = value
    res = c.signs(t, EPS9)[name]
    assert res['tested'][300, 50] and res['fail'][300, 50]


@pytest.mark.unit
def test_kappa_t_needs_only_the_density_of_its_stencil(stored):
    t = {
        k: stored['murnaghan'][k].copy() if k in ('rho', 'u') else v
        for k, v in stored['murnaghan'].items()
    }
    t['u'][300, 50] = np.nan
    t['rho'][301, 50] = t['rho'][299, 50] * (1 - 1e-6)
    assert c.signs(t, EPS9)['kappa_T']['fail'][300, 50]


@pytest.mark.unit
def test_kappa_t_is_tested_only_where_its_stencil_exists(stored):
    res = c.signs(stored['murnaghan'], EPS9)
    assert not res['kappa_T']['tested'][[0, -1]].any() and res['kappa_T']['tested'][1:-1].all()
    assert res['rho']['tested'].all()


@pytest.mark.unit
@pytest.mark.parametrize('excess, flagged', [(5e-9, False), (3e-8, True)])
def test_the_sign_conditions_use_the_9_digit_floor(stored, excess, flagged):
    t = {**stored['murnaghan'], 'cv': stored['murnaghan']['cv'].copy()}
    t['cv'][300, 50] = t['cp'][300, 50] * (1 + excess)
    assert bool(c.signs(t, 1e-12)['cp_ge_cv']['fail'][300, 50]) is flagged


@pytest.mark.unit
@pytest.mark.parametrize('planted', [0.0, 1e-4])
def test_the_csv_report_lists_exactly_the_failing_cells(tmp_path, planted):
    from tools.validation.eos_table_consistency_report import report

    lp, lt = np.arange(8.0, 9.0 + 1e-9, 1 / 150), np.arange(3.0, 3.3 + 1e-9, 1 / 150)
    PP, TT = np.meshgrid(10.0**lp, 10.0**lt, indexing='ij')
    cols = a.murnaghan(PP, TT)
    cols['s'][75, 20] *= 1 + planted
    path = tmp_path / 'table.dat'
    with open(path, 'w') as f:
        f.write('# P T rho u s cp cv alpha nabla_ad phase\n')
        for i in range(PP.shape[0]):
            for j in range(PP.shape[1]):
                vals = [PP[i, j], TT[i, j]] + [cols[k][i, j] for k in a.COLUMNS]
                f.write(' '.join(f'{v:.8e}' for v in vals) + ' solid\n')
    counts = report(str(path), str(tmp_path / 'out.csv'))
    rows = (tmp_path / 'out.csv').read_text().strip().splitlines()
    assert rows[0] == 'condition,P_Pa,T_K,phase,residual,bound' and counts['cp'][0] > 1000
    assert len(rows) - 1 == sum(failing for _, failing in counts.values())
    cp = [r.split(',') for r in rows[1:] if r.startswith('cp,')]
    assert len(cp) == (4 if planted else 0)
    for _, P, T, _, rx, bound in cp:
        assert float(P) == pytest.approx(PP[75, 20], rel=1e-8) and abs(float(rx)) > float(bound)
        assert min(abs(float(T) / TT[75, j] - 1) for j in (18, 19, 21, 22)) < 1e-8
    assert counts['sign_kappa_T'][0] < counts['sign_rho'][0] == PP.size
    assert report(str(path), str(tmp_path / 'out8.csv'), np.uint8(9)) == counts


_BAD_EPS = st.one_of(
    st.sampled_from([np.nan, np.inf, -np.inf, True, False, None, '5e-9', 1.0]),
    st.floats(max_value=0.0, exclude_max=True),
    st.floats(min_value=1.0),
    st.integers(min_value=1),
    st.integers(max_value=-1),
    st.sampled_from([10**400, -(10**400), np.int64(2), np.False_, b'1e-3']),
    st.sampled_from([bytearray(b'1e-3'), memoryview(b'1e-3'), np.array(1e-3), Decimal('1e-3')]),
)


@pytest.fixture(scope='module')
def small():
    return a.table(a.murnaghan, LP[:40], LT[:40], digits=9)


@pytest.mark.unit
@settings(max_examples=60, deadline=None)
@given(eps=_BAD_EPS)
def test_a_precision_that_is_not_a_finite_fraction_is_rejected(small, pt_files, eps):
    """A NaN or infinite eps would make every bound NaN or infinite and hide every cell."""
    for call in (
        lambda: c.check(small, eps),
        lambda: c.signs(small, eps),
        lambda: c.check_pt(pt_files['murnaghan'], eps),
    ):
        with pytest.raises(ValueError, match='eps'):
            call()


@pytest.mark.unit
@settings(max_examples=15, deadline=None)
@given(eps=st.one_of(st.just(0), st.floats(min_value=0.0, max_value=0.99)))
def test_an_exact_table_passes_at_every_valid_precision(small, eps):
    res = c.check(small, eps)
    assert all(v['tested'].sum() > 0 and not v['fail'].any() for v in res.values())


@pytest.mark.unit
@pytest.mark.parametrize('digits', [0, -1, 18, True, 9.0, np.int64(0)])
def test_the_csv_report_rejects_a_digit_count_outside_1_to_17(tmp_path, digits):
    from tools.validation.eos_table_consistency_report import report

    with pytest.raises(ValueError, match='digits'):
        report('unused.dat', str(tmp_path / 'out.csv'), digits)


@pytest.mark.unit
def test_the_csv_report_takes_a_numpy_digit_count(tmp_path):
    from tools.validation.eos_table_consistency_report import report

    with pytest.raises(FileNotFoundError):
        report(str(tmp_path / 'missing.dat'), str(tmp_path / 'out.csv'), np.int64(9))


@pytest.mark.unit
@pytest.mark.parametrize('eps', [np.timedelta64(0), np.timedelta64(3, 's')])
def test_a_time_interval_is_not_a_precision(stored, eps):
    """np.timedelta64 is a NumPy integer type but no number."""
    for call in (c.check, c.signs):
        with pytest.raises(ValueError, match='eps'):
            call(stored['murnaghan'], eps)


# Per identity: (stored column, offsets of its derivative ends along the axis), and the
# columns of its algebraic factor at the cell; a log column carries eps as an absolute error.
_FLOOR_PARTS = {
    'cp': ([('s', 1)], ['cp']),
    'maxwell': ([('s', 0)], ['alpha', 'rho']),
    'alpha': ([('rho', 1)], ['alpha']),
    'dudt': ([('u', 1)], ['cp', 'alpha', 'rho']),
    'cp_cv': ([('rho', 0)], ['alpha', 'rho', 'cp', 'cv']),
}


@pytest.mark.unit
@pytest.mark.parametrize('name', list(_FLOOR_PARTS) + ['nabla_ad'])
@pytest.mark.parametrize('cell', [(300, 50), (600, 100)])
@pytest.mark.parametrize('k', [1, 2])
def test_the_floor_is_the_first_order_round_off_of_each_stored_value(stored, name, cell, k):
    """Moving each stored value that enters the identity by eps alone, the sum of the changes
    of R at the cell is its floor (first order): no term of the floor is lost or doubled. On the
    ideal gas with s and u shifted to 0 at the cell (no identity changes), the stencil term is
    small enough that the algebraic-factor term weighs in."""
    m, eps, (i, j) = dict(stored['ideal']), 1e-9, cell
    for col in ('s', 'u'):
        m[col] = m[col] - m[col][i, j]
    base = c.residuals(m, k)[name][i, j]
    if name == 'nabla_ad':
        moves = [(col, (i, j)) for col in ('nabla_ad', 'rho', 'cp', 'alpha')]
    else:
        ends, factors = _FLOOR_PARTS[name]
        moves = [
            (col, (i + o, j) if ax == 0 else (i, j + o)) for col, ax in ends for o in (-k, k)
        ]
        moves += [(col, (i, j)) for col in factors]
    total = 0.0
    for col, at in moves:
        t = {**m, col: m[col].copy()}
        t[col][at] *= 1 + eps
        total += abs(c.residuals(t, k)[name][i, j] - base)
    assert total == pytest.approx(c.floors(m, eps, k)[name][i, j], rel=1e-4)


def _set(t, col, at, value):
    """``t`` with ``col`` set to ``value`` at ``at``."""
    v = np.array(t[col], float)
    v[at] = value
    return {**t, col: v}


@pytest.mark.unit
@pytest.mark.parametrize('col', ['rho', 'cp', 'cv', 'u', 's', 'alpha', 'nabla_ad'])
@pytest.mark.parametrize('value', [np.inf, -np.inf])
def test_an_isolated_infinite_value_fails_every_identity_at_its_cell(stored, col, value):
    """An inf among numbers is a table fault: every identity that reads the cell fails there
    (nabla_ad reads only rho, cp, alpha and nabla_ad), and none is skipped."""
    res = c.check(_set(stored['murnaghan'], col, (300, 50), value), EPS9)
    reads = {'rho', 'cp', 'alpha', 'nabla_ad'}
    expected = set(c.IDENTITIES) - (set() if col in reads else {'nabla_ad'})
    assert {n for n, v in res.items() if v['fail'][300, 50]} == expected
    assert not any(v['skipped'] for v in res.values())


@pytest.mark.unit
def test_an_isolated_zero_alpha_fails_the_identities_it_divides(stored):
    """alpha = 0 at one cell makes maxwell, alpha, nabla_ad and cp_cv infinite there: they fail."""
    res = c.check(_set(stored['murnaghan'], 'alpha', (300, 50), 0.0), EPS9)
    failing = {n for n, v in res.items() if v['fail'][300, 50]}
    assert {'maxwell', 'alpha', 'nabla_ad', 'cp_cv'} <= failing
    assert not any(v['skipped'] for v in res.values())


@pytest.mark.unit
@pytest.mark.parametrize('col, still', [('rho', set()), ('s', {'nabla_ad'}), ('alpha', set())])
def test_a_missing_value_is_not_a_failure(stored, col, still):
    """NaN marks a missing value: nothing fails, and only nabla_ad, which has no s, still tests
    the cell of a missing entropy."""
    res = c.check(_set(stored['murnaghan'], col, (300, 50), np.nan), EPS9)
    assert not any(v['fail'].any() for v in res.values())
    assert {n for n, v in res.items() if v['tested'][300, 50]} == still


@pytest.mark.unit
def test_an_infinite_density_fails_the_aragog_pt_check(pt_files):
    """In the Aragog P-T files an inf among numbers fails, the fill value stays missing."""
    t = pt_files['murnaghan']
    t = {**t, 'density': {**t['density'], 'values': np.array(t['density']['values'], float)}}
    t['density']['values'][30, 30] = np.inf
    res = c.check_pt(t, 5e-11)
    assert {n for n, v in res.items() if v['fail'][30, 30]} >= {'maxwell'}
    assert not any(v['skipped'] for v in res.values())
