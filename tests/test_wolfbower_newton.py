"""Newton outer solve of a Wolf & Bower mantle against the band report.

Module under test: ``zalmoxis.solver.main`` with ``outer_solver='newton'``. Contract: a
Newton solve keeps the linear T profile, so the adiabat is not computed and the gradient
table is left out of the band report (a known limitation of the Newton outer solver, not
a contract of the physics). Its own file lets the nightly run it beside the Picard solves
of ``tests/test_table_band_report.py``. See docs/How-to/build_tests.md for the test tiers.
"""

from __future__ import annotations

import pytest

from tests.test_table_band_report import WB, _wb_solve

pytestmark = [pytest.mark.integration, pytest.mark.timeout(1500)]


def test_newton_solve_has_no_gradient_entry(tmp_path):
    """At 500 K a Newton solve converges in band without reading the gradient table."""
    res, msgs = _wb_solve(
        tmp_path,
        500.0,
        outer_solver='newton',
        relative_tolerance=1e-9,
        absolute_tolerance=1e-10,
        newton_tol=1e-2,
    )
    assert res['adiabat_computed'] is False
    assert WB['melted_mantle']['adiabat_grad_file'] not in res['table_band_excess']
    assert res['converged_bands'] is True and msgs == []
