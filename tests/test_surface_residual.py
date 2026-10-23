"""The surface pressure residual of the central pressure solve near the surface stop.

``_pressure_residual`` of ``zalmoxis.solver._solve`` is f(P_c) = P(R) - P_target. A solve
whose pressure reaches zero before R is padded; its residual continues below -P_target, so
f is continuous across the central pressure where the stop reaches R and the root has
P(R) > 0. The analytic Seager et al. (2007) EOS gives a smooth f, so continuity and the
accepted surface residual are tested on it.
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


@pytest.fixture(scope='module')
def analytic_solve():
    """main() at 1 M_earth on the analytic EOS; the result, the last P_c residual and its root."""
    calls = []
    real = zs.brentq

    def spy(f, a, b, *args, **kwargs):
        out = real(f, a, b, *args, **kwargs)
        if a > 1e8:  # the central pressure solve, not the outer radius search
            calls.append((f, float(out[0])))
        return out

    root = get_zalmoxis_root()
    cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    cfg.update(
        layer_eos_config=dict(
            cfg['layer_eos_config'], core='Analytic:iron', mantle='Analytic:MgSiO3'
        ),
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
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(zs, 'brentq', spy)
        result = zs.main(cfg, load_material_dictionaries(), melt, os.path.join(root, 'input'))
    assert calls, 'no central pressure solve was seen'
    f, p_root = calls[-1]
    return result, f, p_root


def _threshold_scan(f, p_root, n=201):
    """f on n central pressures across the step where the stop reaches R."""
    s = np.linspace(-2e-3, 2e-3, 81)
    y = np.array([f(p_root * (1 + k)) for k in s])
    i = int(np.flatnonzero((y[:-1] <= -P_TARGET) & (y[1:] > -P_TARGET))[0])
    x = p_root * (1 + np.linspace(s[i] - (s[1] - s[0]), s[i + 1] + (s[1] - s[0]), n))
    return x, np.array([f(p) for p in x])


@pytest.mark.smoke
@pytest.mark.physics_invariant
def test_the_residual_rises_through_the_surface_stop(analytic_solve):
    """f rises at every step across the stop at R, with no jump; a flat padded branch fails."""
    _, f, p_root = analytic_solve
    x, y = _threshold_scan(f, p_root)
    assert y.min() <= -P_TARGET < y.max(), 'the scan does not cross the stop at R'
    d = np.diff(y)
    assert np.all(d > 0), f'{int(np.sum(d <= 0))} of {d.size} steps do not rise'
    assert d.max() < 3 * np.median(d), f'jump {d.max():.3e} Pa, median step {np.median(d):.3e}'


@pytest.mark.smoke
@pytest.mark.physics_invariant
def test_the_accepted_surface_is_not_padded_and_holds_the_target(analytic_solve):
    """The accepted profile reaches R with P(R) within 0.15 P_target of the target."""
    result, _, _ = analytic_solve
    p = np.asarray(result['pressure'], float)
    assert result['converged']
    assert p[-1] > 0, 'the accepted outer node is padded'
    assert abs(p[-1] - P_TARGET) <= 0.15 * P_TARGET, f'P(R) {p[-1]:.6e} Pa'
