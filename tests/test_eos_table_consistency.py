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
def test_a_wobble_of_6e_9_off_a_uniform_grid_is_rejected(stored):
    """GRID_FIT is 4e-9: the shipped PALEOS axes lie at most about 2.1e-9 off a linear fit."""
    m, i = stored['murnaghan'], np.arange(stored['murnaghan']['unique_log_p'].size)
    with pytest.raises(ValueError, match='uniform'):
        c.check({**m, 'unique_log_p': m['unique_log_p'] + 6e-9 * np.sin(i / 7)}, EPS9)


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
    m = stored['murnaghan']
    t = {**m, **{col: np.full_like(m[col], np.nan) for col in c.COLUMNS}}
    with pytest.raises(ValueError, match='no cell'):
        c.check(t, EPS9)


@pytest.mark.unit
def test_a_column_of_zero_alpha_fails_the_identities_it_divides_and_skips_none(stored):
    """A stored 0 is a value: maxwell, alpha, nabla_ad and cp_cv fail in every cell where
    alpha is 0, no identity is skipped, and cp, which does not read alpha, is unchanged."""
    m = stored['murnaghan']
    res = c.check({**m, 'alpha': np.zeros_like(m['alpha'])}, EPS9)
    clean = c.check(m, EPS9)['cp']
    assert not any(v['skipped'] for v in res.values())
    for n in ('maxwell', 'alpha', 'nabla_ad', 'cp_cv'):
        assert res[n]['fail'].sum() > 100_000
    assert np.array_equal(res['cp']['tested'], clean['tested']) and not res['cp']['fail'].any()


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
def test_factors_1_and_2_find_a_fault_two_cells_away_that_3_misses(stored, monkeypatch):
    t = {**stored['murnaghan'], 's': stored['murnaghan']['s'].copy()}
    t['s'][400, 75] *= 1 + 1e-6
    assert c.check(t, EPS9)['cp']['fail'][400, 77]
    monkeypatch.setattr(c, 'FACTOR', 1.0)
    assert c.check(t, EPS9)['cp']['fail'][400, 77]  # a smaller factor still finds it
    monkeypatch.setattr(c, 'FACTOR', 3.0)
    assert not c.check(t, EPS9)['cp']['fail'][400, 77]


@pytest.mark.unit
def test_an_exact_table_at_10_points_per_decade_fails_one_cp_cell():
    """The coarse-grid example of the check docstring: cp and cv are positive everywhere."""
    lp, lt = np.arange(5.0, 11.5 + 1e-9, 0.1), np.arange(np.log10(300.0), np.log10(8000.0), 0.1)
    t = a.table(a.murnaghan, lp, lt, digits=9)
    assert np.nanmin(t['cp']) > 0 and np.nanmin(t['cv']) > 0
    res = c.check(t, EPS9)
    assert {n: int(v['fail'].sum()) for n, v in res.items() if v['fail'].any()} == {'cp': 1}


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
    assert (
        rows[0] == 'condition,P_Pa,T_K,phase,residual,bound' and counts['cp']['tested'] > 1000
    )
    assert len(rows) - 1 == sum(v['failing'] for v in counts.values())
    cp = [r.split(',') for r in rows[1:] if r.startswith('cp,')]
    assert len(cp) == (4 if planted else 0)
    for _, P, T, _, rx, bound in cp:
        assert float(P) == pytest.approx(PP[75, 20], rel=1e-8) and abs(float(rx)) > float(bound)
        assert min(abs(float(T) / TT[75, j] - 1) for j in (18, 19, 21, 22)) < 1e-8
    assert counts['sign_kappa_T']['tested'] < counts['sign_rho']['tested'] == PP.size
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
    """Moving each stored value that enters the identity, and the two end coordinates of the
    axis of its derivative, by eps alone, the sum of the changes of R at the cell is its floor
    (first order): no term of the floor is lost or doubled. On the ideal gas with s and u
    shifted to 0 at the cell (no identity changes), the stencil term is small enough that the
    algebraic-factor term weighs in."""
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
        axis = ('unique_log_p', 'unique_log_t')[ends[0][1]]
        moves += [(axis, 0), (axis, -1)]
    total = 0.0
    for col, at in moves:
        t = {**m, col: np.array(m[col], float)}
        if col.startswith('unique'):
            t[col][at] += np.log10(1 + eps)
        else:
            t[col][at] *= 1 + eps
        total += abs(c.residuals(t, k)[name][i, j] - base)
    assert total == pytest.approx(c.floors(m, eps, k)[name][i, j], rel=1e-4)


def _narrow_ideal_gas():
    """9 x 9 ideal gas around P = 1e5 Pa and T = 1500 K at a step of 0.001 dex, s = 0 at the
    centre, values at the exact points, every column and both axes stored to 9 digits."""
    lp, lt = 5.0 + 0.001 * np.arange(-4, 5), np.log10(1500.0) + 0.001 * np.arange(-4, 5)
    t = a.table(lambda P, T: a.ideal_gas(P, T, cp=1000.0, T0=1500.0, s0=0.0), lp, lt, digits=9)
    for key, x in (('unique_log_p', lp), ('unique_log_t', lt)):
        t[key] = np.log10([float(f'{v:.8e}') for v in 10.0**x])
    return t


@pytest.mark.unit
def test_a_narrow_axis_stored_to_9_digits_passes_and_still_finds_a_fault():
    """On axes spanning 0.008 dex, 9-digit coordinates move the mean step by up to 2 eps / span
    (5e-7 here); the floor covers that, and a 1e-5 fault in s next to the centre is still found."""
    t = _narrow_ideal_gas()
    res = c.check(t, EPS9)
    assert res['cp']['tested'].sum() == 9
    assert sum(int(v['fail'].sum()) for v in res.values()) == 0
    planted = _set(t, 's', (4, 5), t['s'][4, 5] * (1 + 1e-5))
    assert c.check(planted, EPS9)['cp']['fail'][4, 4]


@pytest.mark.unit
@pytest.mark.parametrize('name', ['cp', 'maxwell'])
def test_the_floor_on_a_narrow_axis_is_the_first_order_round_off(name):
    """On exact 9-point axes at 0.001 dex, where the mean-step term 2 eps / span is most of the
    floor, the sum of the changes of R from moving each stored value and each end coordinate
    of the derivative axis by eps is the floor."""
    lp, lt = 5.0 + 0.001 * np.arange(-4, 5), np.log10(1500.0) + 0.001 * np.arange(-4, 5)
    m = a.table(lambda P, T: a.ideal_gas(P, T, cp=1000.0, T0=1500.0, s0=0.0), lp, lt)
    eps, (ends, factors) = 1e-9, _FLOOR_PARTS[name]
    ax = ends[0][1]
    moves = [(col, (4 + o, 4) if ax == 0 else (4, 4 + o)) for col, _ in ends for o in (-1, 1)]
    moves += [(col, (4, 4)) for col in factors]
    moves += [(('unique_log_p', 'unique_log_t')[ax], at) for at in (0, -1)]
    base = c.residuals(m, 1)[name][4, 4]
    total = 0.0
    for col, at in moves:
        t = {**m, col: np.array(m[col], float)}
        if col.startswith('unique'):
            t[col][at] += np.log10(1 + eps)
        else:
            t[col][at] *= 1 + eps
        total += abs(c.residuals(t, 1)[name][4, 4] - base)
    floor = c.floors(m, eps, 1)[name][4, 4]
    assert total == pytest.approx(floor, rel=1e-4)
    assert floor > 20 * eps


def _set(t, col, at, value):
    """``t`` with ``col`` set to ``value`` at ``at``."""
    v = np.array(t[col], float)
    v[at] = value
    return {**t, col: v}


@pytest.mark.unit
@pytest.mark.parametrize('col', ['rho', 'cp', 'cv', 'u', 's', 'alpha', 'nabla_ad'])
@pytest.mark.parametrize('value', [np.inf, -np.inf])
def test_an_isolated_infinite_value_fails_the_identities_that_read_it_near_its_cell(
    stored, col, value
):
    """An inf among numbers is a table fault: every identity that reads the column fails at
    the cell or the cells whose stencil holds it, the others fail nowhere, none is skipped."""
    res = c.check(_set(stored['murnaghan'], col, (300, 50), value), EPS9)
    near = {n for n, v in res.items() if v['fail'][296:305, 46:55].any()}
    assert near == {n for n in c.IDENTITIES if col in READS[n]}
    assert not any(v['fail'].any() for n, v in res.items() if n not in near)
    assert not any(v['skipped'] for v in res.values())


@pytest.mark.unit
def test_an_isolated_zero_alpha_fails_the_identities_it_divides(stored):
    """alpha = 0 at one cell makes maxwell, alpha, nabla_ad and cp_cv infinite there: they fail."""
    res = c.check(_set(stored['murnaghan'], 'alpha', (300, 50), 0.0), EPS9)
    failing = {n for n, v in res.items() if v['fail'][300, 50]}
    assert {'maxwell', 'alpha', 'nabla_ad', 'cp_cv'} <= failing
    assert not any(v['skipped'] for v in res.values())


@pytest.mark.unit
@pytest.mark.parametrize(
    'col, still',
    [('rho', {'cp'}), ('s', {'alpha', 'dudt', 'nabla_ad', 'cp_cv'}), ('alpha', {'cp'})],
)
def test_a_missing_value_is_not_a_failure(stored, col, still):
    """NaN marks a missing value: nothing fails, and the identities that do not read the column
    still test the cell."""
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


READS = {
    'cp': {'s', 'cp'},
    'maxwell': {'s', 'rho', 'alpha'},
    'alpha': {'rho', 'alpha'},
    'dudt': {'u', 'cp', 'rho', 'alpha'},
    'nabla_ad': {'nabla_ad', 'rho', 'cp', 'alpha'},
    'cp_cv': {'rho', 'alpha', 'cp', 'cv'},
}


@pytest.fixture(scope='module')
def clean(stored):
    return c.check(stored['murnaghan'], EPS9)


@pytest.mark.unit
@pytest.mark.parametrize('col', c.COLUMNS)
@pytest.mark.parametrize('value', [np.inf, 0.0])
def test_a_column_of_inf_or_zero_fails_the_identities_that_read_it_and_no_other(
    stored, clean, col, value
):
    """A stored inf or 0 is a value: no identity is skipped, those that read the column fail,
    and the others test and pass exactly the cells they test on the clean table."""
    m = stored['murnaghan']
    res = c.check({**m, col: np.full_like(m[col], value)}, EPS9)
    assert not any(v['skipped'] for v in res.values())
    for n, v in res.items():
        if col in READS[n]:
            assert v['fail'].sum() >= 0.8 * clean[n]['tested'].sum(), n  # finite rests can pass
        else:
            assert np.array_equal(v['tested'], clean[n]['tested']) and not v['fail'].any(), n


@pytest.mark.unit
@pytest.mark.parametrize('col', c.COLUMNS)
def test_a_column_of_nan_skips_the_identities_that_read_it_and_no_other(stored, clean, col):
    """NaN is a missing value: an identity is skipped when none of its cells has its inputs."""
    m = stored['murnaghan']
    res = c.check({**m, col: np.full_like(m[col], np.nan)}, EPS9)
    assert {n for n, v in res.items() if v['skipped']} == {n for n in READS if col in READS[n]}
    assert not any(v['fail'].any() for v in res.values())
    for n in READS:
        if col not in READS[n]:
            assert np.array_equal(res[n]['tested'], clean[n]['tested']), n


@pytest.mark.unit
def test_a_missing_energy_in_one_temperature_column_leaves_cp_unchanged(stored, clean):
    m = stored['murnaghan']
    u = np.array(m['u'], float)
    u[:, 75] = np.nan
    res = c.check({**m, 'u': u}, EPS9)
    assert np.array_equal(res['cp']['tested'], clean['cp']['tested'])
    assert res['dudt']['tested'].sum() < clean['dudt']['tested'].sum()


@pytest.mark.unit
def test_the_report_names_a_skipped_identity_and_a_column_of_inf(stored):
    from tools.validation.eos_table_consistency_report import notes

    m = stored['murnaghan']
    t = {**m, 'u': np.full_like(m['u'], np.inf), 'cv': np.full_like(m['cv'], np.nan)}
    out = notes(t, c.check(t, EPS9))
    assert out['cp_cv'] == 'skipped: cv holds no value'
    assert out['dudt'] == 'u is 0 or inf in every cell that holds a value'
    assert out['cp'] == ''


@pytest.mark.unit
def test_the_report_names_a_stencil_too_short_as_the_reason_for_a_skip():
    from tools.validation.eos_table_consistency_report import notes

    t = a.table(a.murnaghan, LP[:8], LT[:8], digits=9)
    out = notes(t, c.check(t, EPS9))
    stencil = ' present where the derivative has 4 cells on each side in one phase'
    assert out == {
        **{n: f'skipped: no cell with {", ".join(c.INPUTS[n])}{stencil}' for n in c.IDENTITIES},
        'nabla_ad': '',
    }


@pytest.mark.unit
def test_the_report_names_two_empty_columns_and_a_nabla_ad_skip_without_a_stencil(stored):
    from tools.validation.eos_table_consistency_report import notes

    m = stored['murnaghan']
    t = {**m, 'cp': np.full_like(m['cp'], np.nan), 'cv': np.full_like(m['cv'], np.nan)}
    assert notes(t, c.check(t, EPS9))['cp_cv'] == 'skipped: cp, cv hold no value'
    rho, alpha = np.array(m['rho'], float), np.array(m['alpha'], float)
    rho[:, ::2], alpha[:, 1::2] = np.nan, np.nan  # no cell holds both
    t = {**m, 'rho': rho, 'alpha': alpha}
    assert notes(t, c.check(t, EPS9))['nabla_ad'] == (
        'skipped: no cell with nabla_ad, rho, cp, alpha present'
    )


@pytest.mark.unit
def test_the_report_leaves_other_check_errors_unchanged(stored, tmp_path, monkeypatch):
    import tools.validation.eos_table_consistency_report as tool

    m = stored['murnaghan']
    t = {**m, 'unique_log_p': m['unique_log_p'] + 1e-3 * np.sin(np.arange(LP.size))}
    with pytest.raises(ValueError) as expected:
        c.check(t, EPS9)
    monkeypatch.setattr(tool, 'load_paleos_all_properties', lambda _: t)
    with pytest.raises(ValueError) as err:
        tool.report('in.dat', str(tmp_path / 'out.csv'))
    assert str(err.value) == str(expected.value)


@pytest.mark.unit
def test_the_report_names_an_empty_column_and_what_each_identity_needs(tmp_path, monkeypatch):
    import tools.validation.eos_table_consistency_report as tool

    t = a.table(a.murnaghan, LP[:8], LT[:8], digits=9)
    t = {**t, 'nabla_ad': np.full_like(t['nabla_ad'], np.nan)}
    monkeypatch.setattr(tool, 'load_paleos_all_properties', lambda _: t)
    with pytest.raises(
        ValueError, match='columns without a value: nabla_ad; each identity needs'
    ):
        tool.report('in.dat', str(tmp_path / 'out.csv'))


@pytest.mark.unit
def test_the_report_gives_what_each_identity_needs_when_no_column_is_empty(
    tmp_path, monkeypatch
):
    import tools.validation.eos_table_consistency_report as tool

    t = a.table(a.murnaghan, LP[:8], LT[:8], digits=9)
    rho, alpha = np.array(t['rho'], float), np.array(t['alpha'], float)
    rho[:, ::2], alpha[:, 1::2] = np.nan, np.nan  # no cell holds both
    t = {**t, 'rho': rho, 'alpha': alpha}
    monkeypatch.setattr(tool, 'load_paleos_all_properties', lambda _: t)
    with pytest.raises(ValueError, match='no cell tested') as err:
        tool.report('in.dat', str(tmp_path / 'out.csv'))
    assert 'columns without a value' not in str(err.value)
    assert str(err.value).endswith(tool.NEEDS)


@pytest.mark.unit
def test_the_report_main_prints_the_reason_instead_of_zero_counts(monkeypatch, capsys):
    import tools.validation.eos_table_consistency_report as tool

    counts = {
        'cp': {'present': 12, 'tested': 10, 'blind': 2, 'failing': 3, 'note': ''},
        'dudt': {
            'present': 0,
            'tested': 0,
            'blind': 0,
            'failing': 0,
            'note': 'skipped: u holds no value',
        },
        'alpha': {'present': 5, 'tested': 5, 'blind': 0, 'failing': 5, 'note': 'x'},
        'sign_rho': {'tested': 7, 'failing': 1, 'note': ''},
    }
    monkeypatch.setattr(tool, 'report', lambda *_: counts)
    monkeypatch.setattr('sys.argv', ['tool', 'in.dat', 'out.csv'])
    tool.main()
    assert capsys.readouterr().out.splitlines() == [
        'cp: present 12, tested 10 (2 with a bound of 1 or more), failing 3',
        'dudt: skipped: u holds no value',
        'alpha: present 5, tested 5 (0 with a bound of 1 or more), failing 5 (x)',
        'sign_rho: tested 7, failing 1',
    ]


@pytest.mark.unit
def test_the_report_names_the_empty_columns_when_every_identity_is_skipped(
    stored, tmp_path, monkeypatch
):
    import tools.validation.eos_table_consistency_report as tool

    m = stored['murnaghan']
    t = {**m, **{col: np.full_like(m[col], np.nan) for col in ('rho', 's', 'u')}}
    monkeypatch.setattr(tool, 'load_paleos_all_properties', lambda _: t)
    with pytest.raises(ValueError, match='columns without a value: rho, u, s'):
        tool.report('in.dat', str(tmp_path / 'out.csv'))


def _rounded(x, digits):
    fmt = f'{{:.{digits - 1}e}}'
    return np.array([float(fmt.format(v)) for v in np.ravel(x)]).reshape(np.shape(x))


def _pt_tables(eos, digits, P=None, T=None):
    """Exact Aragog P-T tables with values and axes rounded to ``digits`` digits."""
    P = np.linspace(1e5, 3.5e11, 200) if P is None else P
    T = np.linspace(300.0, 6000.0, 200) if T is None else T
    PP, TT = np.meshgrid(P, T, indexing='ij')
    cols = eos(PP, TT)
    Pr, Tr = _rounded(P, digits), _rounded(T, digits)
    return {
        q: {'P': Pr, 'T': Tr, 'values': _rounded(cols[key], digits)}
        for q, key in _PT_NAMES.items()
    }


@pytest.mark.unit
@pytest.mark.parametrize('name', ['murnaghan', 'ideal'])
def test_11_digit_aragog_pt_files_pass_at_their_own_precision(pt_files, name, monkeypatch):
    """With the floor lowered to the stored 11 digits, the axis term alone covers the rounding
    of the stored P and T coordinates."""
    monkeypatch.setattr(c, 'EPS_FLOOR', 5e-11)
    res = c.check_pt(pt_files[name], 5e-11)
    assert {n: int(v['fail'].sum()) for n, v in res.items()} == dict.fromkeys(
        c.PT_IDENTITIES, 0
    )


@pytest.mark.unit
@pytest.mark.parametrize('eos', [a.murnaghan, a.ideal_gas], ids=['murnaghan', 'ideal'])
def test_9_digit_aragog_pt_tables_with_9_digit_axes_pass(eos):
    res = c.check_pt(_pt_tables(eos, 9), 5e-9)
    assert {n: int(v['fail'].sum()) for n, v in res.items()} == dict.fromkeys(
        c.PT_IDENTITIES, 0
    )


_AXIS_CASES = [
    (axis, change, message)
    for axis in ('P', 'T')
    for change, message in (
        ('inf end', 'must be finite and strictly increasing'),
        ('nan', 'must be finite and strictly increasing'),
        ('one point', 'needs at least 2 points'),
        ('2-D', 'must be one-dimensional'),
        ('jitter', 'is neither uniform nor geometric'),
    )
] + [('T', 'zero start', 'must be positive'), ('P', 'negative start', 'must not be negative')]


@pytest.mark.unit
@pytest.mark.parametrize('axis, change, message', _AXIS_CASES)
def test_aragog_pt_axes_that_are_not_finite_long_and_regular_are_rejected(
    pt_files, axis, change, message
):
    t = pt_files['murnaghan']
    x = np.array(t['density'][axis], float)
    if change == 'inf end':
        x[-1] = np.inf
    elif change == 'nan':
        x[50] = np.nan
    elif change == 'zero start':
        x[0] = 0.0
    elif change == 'negative start':
        x = x - x[1]  # uniform from minus one step
    elif change == 'one point':
        x = x[:1]
    elif change == 'jitter':
        rng = np.random.default_rng(1)
        x = x + 0.2 * np.diff(x).mean() * rng.uniform(-1, 1, x.size)
    n = x.size
    new = {}
    for q, v in t.items():
        vals = v['values'][:n] if axis == 'P' else v['values'][:, :n]
        new[q] = {**v, axis: x.reshape(1, -1) if change == '2-D' else x, 'values': vals}
    try:
        c.check_pt(new, 5e-11)
    except Exception as err:  # any other exception type fails the assertion below
        assert type(err) is ValueError and f'{axis} {message}' in str(err), repr(err)
    else:
        pytest.fail('the axis was accepted')


@pytest.mark.unit
def test_an_exact_aragog_pt_table_on_a_geometric_pressure_axis_passes():
    res = c.check_pt(_pt_tables(a.murnaghan, 11, P=np.geomspace(1e5, 3.5e11, 200)), 5e-11)
    assert all(v['tested'].sum() > 30_000 and not v['fail'].any() for v in res.values())


@pytest.mark.unit
def test_the_aragog_pt_check_counts_the_fill_cells_of_each_file(pt_files):
    t = pt_files['murnaghan']
    t = {q: {**v, 'values': np.array(v['values'], float)} for q, v in t.items()}
    t['entropy']['values'][30, 30:33] = c.PT_FILL
    t['density']['values'][50, 50] = c.PT_FILL
    res = c.check_pt(t, 5e-11)
    expected = {'density': 1, 'heat_capacity': 0, 'thermal_exp': 0, 'entropy': 3}
    assert all(v['fill'] == expected for v in res.values())


def _plant_pt(t, quantity, at, size):
    v = np.array(t[quantity]['values'], float)
    v[at] *= 1 + size
    return {**t, quantity: {**t[quantity], 'values': v}}


def _found_pt(res, identity, axis, at=(100, 100)):
    i, j = at
    cells = [(i + o, j) if axis == 0 else (i, j + o) for o in (-2, -1, 1, 2)]
    return [bool(res[identity]['fail'][x]) for x in cells]


@pytest.mark.unit
@pytest.mark.parametrize(
    'quantity, identity, axis',
    [('entropy', 'cp', 1), ('density', 'alpha', 1), ('entropy', 'maxwell', 0)],
)
def test_a_1e_6_fault_is_found_in_9_digit_aragog_pt_tables(quantity, identity, axis):
    t = _plant_pt(_pt_tables(a.murnaghan, 9), quantity, (100, 100), 1e-6)
    assert _found_pt(c.check_pt(t, 5e-9), identity, axis) == [True] * 4


@pytest.mark.unit
def test_an_axis_off_its_fit_by_half_the_limit_raises_the_pt_floor_and_keeps_detection():
    """A jitter of 0.5 PT_GRID_FIT steps on T still counts as uniform: the floor grows, the
    exact table passes, and a 1e-4 entropy fault is found one and two steps away."""
    T = np.linspace(300.0, 6000.0, 200)
    rng = np.random.default_rng(5)
    T2 = T + 0.5 * c.PT_GRID_FIT * np.diff(T).mean() * rng.uniform(-1, 1, T.size)
    clean, jit = _pt_tables(a.murnaghan, 11), _pt_tables(a.murnaghan, 11, T=T2)
    a0, a1 = (np.nanmedian(c._pt_parts(t, 1, 5e-11)['cp'][4]) for t in (clean, jit))
    assert a1 > 10 * a0
    assert not any(v['fail'].any() for v in c.check_pt(jit, 5e-11).values())
    planted = c.check_pt(_plant_pt(jit, 'entropy', (100, 100), 1e-4), 5e-11)
    assert _found_pt(planted, 'cp', 1) == [True] * 4


@pytest.mark.unit
def test_the_axis_error_is_none_for_an_axis_that_is_neither_uniform_nor_geometric():
    irregular = np.cumsum(np.r_[1.0, np.tile([1.0, 3.0], 20)])
    assert c._axis_error(irregular, 5e-9) is None
    x = np.linspace(1.0, 2.0, 11)
    assert (c._axis_error(x, 5e-9) < 1e-14).all()


@pytest.mark.unit
@pytest.mark.parametrize(
    'axes', [('unique_log_p',), ('unique_log_t',), ('unique_log_p', 'unique_log_t')]
)
def test_an_exact_table_with_stored_axes_3_9e_9_off_their_fit_passes(axes):
    """GRID_FIT accepts this wobble: the nabla_ad floor covers the rounding of P, and in the
    derivative identities the coordinate rounding stays below their floor."""
    lp, lt = np.linspace(5.0, 14.0, 1351), np.linspace(np.log10(300.0), 5.0, 380)
    m = a.table(a.murnaghan, lp, lt, digits=9)
    t = {**m}
    for key in axes:
        x = m[key]
        t[key] = x + 3.9e-9 * np.where(np.arange(x.size) % 4 < 2, 1.0, -1.0)
    res = c.check(t, EPS9)
    assert {n: int(v['fail'].sum()) for n, v in res.items()} == dict.fromkeys(c.IDENTITIES, 0)
    assert all(v['tested'].sum() > 400_000 for v in res.values())


@pytest.mark.unit
@pytest.mark.parametrize('fraction, accepted', [(0.9, True), (2.0, False)])
def test_the_pt_grid_fit_edge(fraction, accepted):
    T = np.linspace(300.0, 6000.0, 200)
    T2 = T + fraction * c.PT_GRID_FIT * np.diff(T).mean() * np.where(
        np.arange(T.size) % 2, 1, -1
    )
    t = _pt_tables(a.murnaghan, 17, T=T2)
    if accepted:
        assert not any(v['fail'].any() for v in c.check_pt(t, 5e-11).values())
    else:
        with pytest.raises(ValueError, match='neither uniform nor geometric'):
            c.check_pt(t, 5e-11)


@pytest.mark.unit
def test_the_axis_term_is_at_most_the_rounding_of_the_stored_coordinates(pt_files):
    """A <= (e[i+k] + e[i-k]) / dx with e at most eps |x|: bounded by eps max|T| / min dT."""
    t = pt_files['murnaghan']
    T = t['density']['T']
    A = c._pt_parts(t, 1, 5e-11)['cp'][4]
    assert 0 < np.nanmax(A) <= 1.05 * 5e-11 * T.max() / np.diff(T).min()


@pytest.mark.unit
def test_a_1e_5_fault_is_found_on_a_geometric_pressure_axis():
    t = _pt_tables(a.murnaghan, 11, P=np.geomspace(1e5, 3.5e11, 200))
    found = _found_pt(
        c.check_pt(_plant_pt(t, 'entropy', (100, 100), 1e-5), 5e-11), 'maxwell', 0
    )
    assert found == [True] * 4


@pytest.mark.unit
def test_an_exact_aragog_pt_table_on_a_pressure_axis_from_zero_passes():
    t = _pt_tables(a.murnaghan, 11, P=np.linspace(0.0, 3.5e11, 200))
    try:
        res = c.check_pt(t, 5e-11)
    except ValueError as err:
        pytest.fail(f'a P axis from 0 was rejected: {err}')
    assert all(v['tested'].sum() > 30_000 and not v['fail'].any() for v in res.values())
    assert res['maxwell']['tested'][4, 100]  # the first row with a full P stencil


def _power_law(m):
    """Murnaghan columns with rho ~ T^-m and alpha = m / T, so that T alpha = m."""

    def eos(P, T):
        cols = a.murnaghan(P, T)
        cols['rho'] = 3000.0 * (T / 1000.0) ** (-m)
        cols['alpha'] = m / T
        return cols

    return eos


@pytest.mark.unit
def test_a_table_with_t_alpha_200_and_t_axis_2_1e_9_off_its_fit_passes():
    """The floor of alpha covers the rounding of the stored T, which a large T alpha amplifies."""
    lp, lt = np.linspace(5.0, 14.0, 1351), np.linspace(np.log10(300.0), np.log10(3000.0), 151)
    t = a.table(_power_law(200.0), lp, lt, digits=9)
    t['unique_log_t'] = t['unique_log_t'] + 2.1e-9 * np.where(np.arange(lt.size) % 4 < 2, 1, -1)
    res = c.check(t, EPS9)['alpha']
    assert res['tested'].sum() > 100_000 and not res['fail'].any()


@pytest.mark.unit
@pytest.mark.parametrize(
    'key, amplitude, grown',
    [
        ('unique_log_p', 3.9e-9, ('maxwell', 'dudt', 'cp_cv')),
        ('unique_log_t', 3.0e-9, ('alpha', 'dudt', 'cp_cv')),
    ],
)
def test_the_floor_grows_with_the_departure_of_the_stored_axes(stored, key, amplitude, grown):
    """Each identity whose algebraic factor holds P or T gains the rounding of that axis."""
    m = stored['murnaghan']
    w = {**m, key: m[key] + amplitude * np.where(np.arange(m[key].size) % 4 < 2, 1.0, -1.0)}
    f0, f1 = c.floors(m, EPS9), c.floors(w, EPS9)
    for n in grown:
        assert np.nanmedian(f1[n] / f0[n]) - 1 > 5e-7, n


@pytest.mark.unit
@pytest.mark.parametrize('kind', ['linear', 'geometric'])
@pytest.mark.parametrize('fraction, regular', [(0.9, True), (1.1, False)])
def test_with_eps_0_a_pt_axis_is_regular_within_1e_6_of_its_step(kind, fraction, regular):
    sign = np.where(np.arange(200) % 2, 1.0, -1.0)
    if kind == 'linear':
        x = np.linspace(300.0, 6000.0, 200)
        x = x + fraction * 1e-6 * np.diff(x).mean() * sign
    else:
        ly = np.log(np.geomspace(1e5, 3.5e11, 200))
        x = np.exp(ly + fraction * 1e-6 * np.diff(ly).mean() * sign)
    assert (c._axis_error(x, 0.0) is not None) is regular


@pytest.mark.unit
def test_a_geometric_temperature_axis_at_9_digits_passes_the_aragog_pt_check():
    """The departure of each stored T from the geometric fit covers its 9-digit rounding."""
    res = c.check_pt(_pt_tables(a.murnaghan, 9, T=np.geomspace(300.0, 6000.0, 200)), 5e-9)
    assert all(v['tested'].sum() > 30_000 and not v['fail'].any() for v in res.values())


@pytest.mark.unit
def test_a_nan_in_the_entropy_file_is_missing_in_the_aragog_pt_check(pt_files):
    t = pt_files['murnaghan']
    t = {**t, 'entropy': {**t['entropy'], 'values': np.array(t['entropy']['values'], float)}}
    t['entropy']['values'][100, 100] = np.nan
    res = c.check_pt(t, 5e-11)
    assert not any(v['fail'].any() for v in res.values())
    assert (
        not res['cp']['tested'][100, 96:105].any()
        and not res['maxwell']['tested'][96:105, 100].any()
    )


_LP20 = np.arange(5.0, 11.0 + 1e-9, 1 / 20)
_LT20 = np.arange(np.log10(500.0), np.log10(5000.0) + 1e-9, 1 / 20)


@pytest.mark.unit
@pytest.mark.parametrize('s_value, blind', [(1e10, True), (2200.0, False)])
def test_maxwell_cells_with_a_bound_of_1_or_more_are_counted_apart(s_value, blind):
    """Murnaghan at 20 points per decade (log P 5 to 11, log T log10 500 to log10 5000, 9
    digits) with a constant s: there R_x = -1, so a cell detects the missing derivative only
    where its maxwell bound is below 1; s = 1e10 makes every bound 1 or more."""
    m = a.table(a.murnaghan, _LP20, _LT20, digits=9)
    v = c.check({**m, 's': np.full_like(m['s'], s_value)}, EPS9)['maxwell']
    assert v['tested'].sum() == 2373
    assert v['blind'].sum() == (2373 if blind else 0)
    assert v['fail'].sum() == (0 if blind else 2373)


@pytest.mark.unit
def test_the_report_counts_add_up_when_present_cells_are_not_finite(
    stored, tmp_path, monkeypatch
):
    """rho = 0 leaves alpha with no tested cell; its failing cells are counted among the
    present cells, so failing never exceeds present."""
    import tools.validation.eos_table_consistency_report as tool

    m = stored['murnaghan']
    monkeypatch.setattr(
        tool, 'load_paleos_all_properties', lambda _: {**m, 'rho': np.zeros_like(m['rho'])}
    )
    counts = tool.report('in.dat', str(tmp_path / 'out.csv'))
    alpha = counts['alpha']
    assert alpha['tested'] == 0 and 0 < alpha['failing'] <= alpha['present']


@pytest.mark.unit
def test_only_tested_cells_count_as_blind(stored):
    """An inf entropy makes the maxwell bound infinite at untested cells; they are not blind."""
    res = c.check(_set(stored['murnaghan'], 's', (300, 50), np.inf), EPS9)['maxwell']
    assert (res['bound'][~res['tested']] >= 1).any()
    assert not (res['blind'] & ~res['tested']).any()


@pytest.mark.unit
def test_the_report_counts_the_blind_maxwell_cells(tmp_path, monkeypatch):
    import tools.validation.eos_table_consistency_report as tool

    m = a.table(a.murnaghan, _LP20, _LT20, digits=9)
    t = {**m, 's': np.full_like(m['s'], 1e10)}
    monkeypatch.setattr(tool, 'load_paleos_all_properties', lambda _: t)
    maxwell = tool.report('in.dat', str(tmp_path / 'out.csv'))['maxwell']
    assert maxwell['blind'] == maxwell['tested'] == 2373
