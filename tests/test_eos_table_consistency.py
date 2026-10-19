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
