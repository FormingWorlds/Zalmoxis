"""Thermodynamic consistency of regular EOS tables.

Module under test: ``zalmoxis.eos.consistency``. The analytic tables of
``tests/_analytic_eos.py`` hold every identity exactly, so their residuals are the truncation of
the central differences alone. See docs/How-to/build_tests.md for the test tiers.
"""

from __future__ import annotations

import numpy as np
import pytest

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
def test_the_floor_grows_with_the_stored_precision(stored):
    f9, f12 = (c.floors(stored['murnaghan'], e)['cp'] for e in (EPS9, 5e-12))
    assert np.nanmedian(f9 / f12) == pytest.approx(1000.0)


@pytest.mark.unit
def test_an_exact_table_breaks_no_sign_condition(stored):
    assert all(not v.any() for v in c.signs(stored['murnaghan'], EPS9).values())


@pytest.mark.unit
@pytest.mark.parametrize(
    'col, value, name',
    [('rho', -1.0, 'rho'), ('cp', 0.0, 'cp'), ('cv', np.nan, 'cv'), ('cv', 2.0e4, 'cp_ge_cv')],
)
def test_a_cell_that_breaks_a_sign_condition_is_flagged(stored, col, value, name):
    t = {**stored['murnaghan'], col: stored['murnaghan'][col].copy()}
    t[col][300, 50] = value
    assert c.signs(t, EPS9)[name][300, 50]


@pytest.mark.unit
def test_a_density_that_falls_with_pressure_breaks_kappa_t(stored):
    t = {**stored['murnaghan'], 'rho': stored['murnaghan']['rho'].copy()}
    t['rho'][301, 50] = t['rho'][299, 50] * (1 - 1e-6)
    flags = c.signs(t, EPS9)['kappa_T']
    assert flags[300, 50] and int(flags.sum()) == 1


@pytest.mark.unit
@pytest.mark.parametrize('eos', [a.murnaghan, a.ideal_gas])
def test_a_float64_table_is_checked_at_the_9_digit_floor(eos):
    """eps below EPS_FLOOR is raised to it, so an exact float64 table has no failing cell."""
    res = c.check(a.table(eos, LP, LT), np.finfo(float).eps)
    assert sum(int(v['fail'].sum()) for v in res.values()) == 0
