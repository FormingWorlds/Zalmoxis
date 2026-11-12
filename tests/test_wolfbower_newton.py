"""Newton outer solve of a Wolf & Bower mantle against the band report.

Module under test: ``zalmoxis.solver.main`` with ``outer_solver='newton'``. Contract: in an
adiabatic temperature mode the Newton root seeds the Picard blend ramp, so the result carries
the adiabat and the band report reads the gradient table like a Picard solve. Its own file lets
the nightly run it beside the Picard solves of ``tests/test_table_band_report.py``. See
docs/How-to/build_tests.md for the test tiers.
"""

from __future__ import annotations

import os

import pytest

from tests.test_table_band_report import WB, _wb_solve
from zalmoxis.constants import MAX_BAND_EXCESS_DEFAULT

pytestmark = [pytest.mark.integration, pytest.mark.wolfbower]


@pytest.mark.timeout(5300)  # the other solves, the 3600 s clock and the 600 s retry
def test_newton_solve_at_500_k_leaves_the_gradient_table_band(tmp_path):
    """At a 500 K surface the adiabat of the Newton solve leaves the band of the Wolf & Bower
    gradient table by more than max_band_excess, so the structure is returned unconverged. The
    linear first guess stayed in band only because it never read the gradient table."""
    res, msgs = _wb_solve(
        tmp_path,
        500.0,
        outer_solver='newton',
        relative_tolerance=1e-9,
        absolute_tolerance=1e-10,
        newton_tol=1e-2,  # 1 % in mass is enough for a band check and ends in minutes
    )
    grad = WB['melted_mantle']['adiabat_grad_file']
    assert res['adiabat_computed'] is True and res['adiabat_blend'] == 1.0
    nodes, excess = res['table_band_excess'][grad]
    assert nodes > 0 and excess > MAX_BAND_EXCESS_DEFAULT
    assert res['converged'] is False and res['converged_bands'] is False
    assert any(os.path.basename(grad) in m for m in msgs)
