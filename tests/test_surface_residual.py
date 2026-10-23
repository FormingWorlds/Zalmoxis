"""The surface pressure residual of the central pressure solve near the surface stop.

``_pressure_residual`` of ``zalmoxis.solver._solve`` is f(P_c) = P(R) - P_target. A solve
that stops before R at the pressure P_stop is padded; its residual continues as P_stop -
P_target - k (R - r_stop) with k = rho_ref G M_target / R^2 fixed for the pressure solve, so f
is continuous across the central pressure where the stop reaches R and the root has P(R) > 0.
The analytic Seager et al. (2007) EOS gives a smooth f, so continuity and the accepted surface
residual are tested on it.
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
from zalmoxis.constants import G

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]

P_TARGET = 101325.0
NOISE_ANALYTIC = 4e3  # integrator noise of f on the analytic EOS [Pa]


@pytest.fixture(scope='module')
def analytic_solve():
    """main() at 1 M_earth on the analytic EOS; the result, the last P_c residual and its root."""
    calls, stops = [], []
    real, real_structure = zs.brentq, zs.solve_structure

    def spy(f, a, b, *args, **kwargs):
        out = real(f, a, b, *args, **kwargs)
        if a > 1e8:  # the central pressure solve, not the outer radius search
            calls.append((f, float(out[0])))
        return out

    def structure_spy(*args, **kwargs):
        out = real_structure(*args, **kwargs)
        if isinstance(kwargs.get('stop'), dict):
            stops.append(
                (dict(kwargs['stop']), np.asarray(args[3], float), *map(np.array, out))
            )
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
        mp.setattr(zs, 'solve_structure', structure_spy)
        result = zs.main(cfg, load_material_dictionaries(), melt, os.path.join(root, 'input'))
    assert calls, 'no central pressure solve was seen'
    f, p_root = calls[-1]
    return result, f, p_root, stops, cfg


def _threshold_scan(f, p_root, n=201):
    """f on n central pressures across the step where the stop reaches R."""
    s = np.linspace(-2e-3, 2e-3, 81)
    y = np.array([f(p_root * (1 + k)) for k in s])
    i = int(np.flatnonzero((y[:-1] <= -P_TARGET) & (y[1:] > -P_TARGET))[0])
    x = p_root * (1 + np.linspace(s[i] - (s[1] - s[0]), s[i + 1] + (s[1] - s[0]), n))
    return x, np.array([f(p) for p in x])


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_residual_rises_through_the_surface_stop(analytic_solve, validation_plot):
    """f rises at every step across the stop at R, with no jump; a flat padded branch fails.

    Reference: the closed form of the stopped branch, f = -P_target - rho_ref g_ref (R -
    r_stop) for a zero-pressure stop, which equals -P_target at r_stop = R, on the Seager et al.
    (2007) analytic EOS. Each step moves f by more than 10 times the integrator noise of f on
    this EOS (3e3 to 4e3 Pa), so a jump at the stop cannot hide in the noise.
    """
    _, f, p_root, _, _ = analytic_solve
    x, y = _threshold_scan(f, p_root)
    assert y.min() <= -P_TARGET < y.max(), 'the scan does not cross the stop at R'
    d = np.diff(y)
    assert np.all(d > 0), f'{int(np.sum(d <= 0))} of {d.size} steps do not rise'
    assert np.median(d) > 10 * NOISE_ANALYTIC, f'median step {np.median(d):.3e} Pa'
    assert d.max() < 3 * np.median(d), f'jump {d.max():.3e} Pa, median step {np.median(d):.3e}'

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot((x - p_root) / 1e6, y / P_TARGET, 'o', ms=3, label='surface residual')
        ax.axhline(-1, color=plt.rcParams['text.color'], ls='--', label='stop at R')
        ax.set(
            xlabel='Central pressure minus the root [MPa]', ylabel='Residual / target pressure'
        )
        ax.legend()
        fig.tight_layout()
        return fig

    validation_plot(figure, 'surface_residual_continuity')


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_accepted_surface_is_not_padded_and_holds_the_target(analytic_solve):
    """The accepted profile reaches R with P(R) within 0.15 P_target of the target.

    Reference: the target surface pressure on the Seager et al. (2007) analytic EOS; a padded
    accept gives P(R) = 0, a deviation of P_target.
    """
    result, *_ = analytic_solve
    p = np.asarray(result['pressure'], float)
    assert result['converged']
    assert p[-1] > 0, 'the accepted outer node is padded'
    assert abs(p[-1] - P_TARGET) <= 0.15 * P_TARGET, f'P(R) {p[-1]:.6e} Pa'


def test_a_stopped_solve_reports_where_it_stopped(analytic_solve):
    """The pressure solve passes a stop dict; a padded solve fills it with the stop radius in its
    first padded shell and the pressure there, zero at the terminal event; a solve that reaches
    R leaves it empty."""
    _, _, _, stops, _ = analytic_solve
    assert stops, 'no structure solve was given a stop dict'
    padded = [s for s in stops if s[4][-1] == 0.0]
    assert padded and len(padded) < len(stops), 'the scan has no padded or no unpadded solve'
    for stop, radii, mass, gravity, pressure in stops:
        if pressure[-1] > 0.0:
            assert stop == {}
            continue
        n = int(np.argmax(pressure == 0.0))
        assert set(stop) == {'r', 'p'}
        assert radii[n - 1] < stop['r'] <= radii[n]
        assert stop['p'] == 0.0


def test_the_stopped_branch_slope_is_fixed_for_the_pressure_solve(analytic_solve, monkeypatch):
    """Across the scan, every stopped evaluation uses one k = rho_ref G M_target / R^2: the
    gravity of the target mass at the target radius, not the gravity at each stop, which goes
    to zero with r_stop and turns the stopped branch over where r_stop < R / 2."""
    _, f, p_root, _, cfg = analytic_solve
    seen, real = [], zs._surface_residual

    def spy(pressure, stop, p_target, radius, k):
        seen.append((dict(stop), radius, k))
        return real(pressure, stop, p_target, radius, k)

    monkeypatch.setattr(zs, '_surface_residual', spy)
    _threshold_scan(f, p_root)
    stopped = [(stop, radius, k) for stop, radius, k in seen if stop]
    assert stopped and len(stopped) < len(seen), 'the scan has no stopped or no reached solve'
    ks = {k for _, _, k in seen}
    assert len(ks) == 1, f'{len(ks)} slopes in one scan'
    radius = seen[0][1]
    rho_ref = ks.pop() / (G * cfg['planet_mass'] / radius**2)
    # rho_ref is the density of the outermost node with P > 0: MgSiO3 near zero pressure.
    assert 3000.0 < rho_ref < 5000.0, f'rho_ref {rho_ref:.1f} kg/m3'
