"""One retry at tighter integrator tolerances for an accept far from the target surface pressure.

The surface pressure of a converged solve carries the integration error of the whole profile,
of order rtol P_c; at 10 M_earth on the analytic EOS a change of the step sequence moves P(R)
by about 8e8 Pa. ``_noise_retry`` solves such an accept once more at 1/100 of the integrator
tolerances when |P(R) - P_target| exceeds ``_RETRY_RESIDUAL_TARGETS`` target pressures, and
keeps the retry only when it converges closer to the target.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import zalmoxis.solver as zs
from zalmoxis import get_zalmoxis_root
from zalmoxis.config import (
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
)

P_TARGET = 101325.0
PARAMS = {'relative_tolerance': 1e-5, 'absolute_tolerance': 1e-6, 'tolerance_outer': 3e-3}


def _result(p_surface, converged=True):
    return {'converged': converged, 'pressure': np.array([3e11, 1e9, p_surface])}


@pytest.mark.unit
@pytest.mark.timeout(30)
def test_only_an_accept_far_from_the_target_is_solved_again():
    """A converged accept more than 10 target pressures off gets one solve at 1/100 of rtol and
    atol, with every other parameter kept; one within the bound, or not converged, gets none."""
    seen = []

    def resolve(params):
        seen.append(params)
        return _result(1.2 * P_TARGET)

    far = _result(2e8)
    out, params = zs._noise_retry(far, PARAMS, P_TARGET, resolve)
    assert len(seen) == 1
    assert seen[0]['relative_tolerance'] == pytest.approx(1e-7, rel=1e-12)
    assert seen[0]['absolute_tolerance'] == pytest.approx(1e-8, rel=1e-12)
    assert seen[0]['tolerance_outer'] == PARAMS['tolerance_outer']
    assert out['pressure'][-1] == 1.2 * P_TARGET and params is seen[0]
    assert zs._RETRY_RESIDUAL_TARGETS == 10.0
    for first in (_result(11.0 * P_TARGET - 1.0), _result(0.0), _result(2e8, converged=False)):
        seen.clear()
        out, params = zs._noise_retry(first, PARAMS, P_TARGET, resolve)
        assert seen == [] and out is first and params is PARAMS


@pytest.mark.unit
@pytest.mark.timeout(30)
def test_a_retry_that_does_not_get_closer_is_dropped():
    """The first accept stays when the retry misses the target by more, or does not converge."""
    first = _result(2e8)
    for retry in (_result(3e8), _result(1.2 * P_TARGET, converged=False)):
        out, params = zs._noise_retry(first, PARAMS, P_TARGET, lambda p, r=retry: r)
        assert out is first and params is PARAMS


@pytest.mark.smoke
@pytest.mark.physics_invariant
@pytest.mark.timeout(300)
def test_an_accept_on_the_integration_noise_step_is_solved_again():
    """At 10 M_earth on the analytic EOS the accept at rtol 1e-5 reaches R with P(R) about
    1928 P_target; the retry at rtol 1e-7 brings it within 10 P_target."""
    root = get_zalmoxis_root()
    cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    cfg['layer_eos_config'] = dict(
        cfg['layer_eos_config'], core='Analytic:iron', mantle='Analytic:MgSiO3'
    )
    cfg.update(
        planet_mass=10 * cfg['planet_mass'],
        num_layers=150,
        target_surface_pressure=P_TARGET,
        data_output_enabled=False,
        plotting_enabled=False,
    )
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        cfg['rock_solidus'],
        cfg['rock_liquidus'],
        cfg['mushy_zone_factor'],
    )
    result = zs.main(cfg, load_material_dictionaries(), melt, os.path.join(root, 'input'))
    p_surface = float(np.asarray(result['pressure'])[-1])
    assert result['converged']
    assert abs(p_surface - P_TARGET) <= 10.0 * P_TARGET, f'P(R) {p_surface:.6e} Pa'
