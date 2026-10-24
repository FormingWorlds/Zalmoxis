"""Real solves of the central-pressure root (``zalmoxis.solver._refine_pressure_root``).

Sweep setups from ``input/default.toml`` whose first brentq root can be padded: the accepted
profile must reach the surface. Padding depends on round-off, so each case runs at several
planet-mass shifts; on the code before the refinement at least one shift pads.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import zalmoxis
import zalmoxis.solver as zs
from zalmoxis.config import load_material_dictionaries, load_zalmoxis_config

pytestmark = [pytest.mark.smoke, pytest.mark.accept_solves, pytest.mark.timeout(2400)]

P_T = 101325.0  # target surface pressure [Pa]


def _case(root, **kw):
    """A sweep setup from input/default.toml: 150 layers, the Newton outer solver at rtol 1e-9."""
    cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    cfg.update(
        num_layers=150,
        outer_solver='newton',
        relative_tolerance=1e-9,
        absolute_tolerance=1e-10,
        wall_timeout=1800.0,
        data_output_enabled=False,
        plotting_enabled=False,
    )
    cfg.update(kw)
    return cfg


ANALYTIC = {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'}
# Planet mass shifts [1]: padding depends on round-off, so the case runs at several.
SHIFTS = (0.0, 1e-9, -1e-9)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('shift', SHIFTS, ids=['m0', 'm_plus_1e-9', 'm_minus_1e-9'])
def test_the_smooth_kink_planet_reaches_its_surface(shift):
    """Analytic 1 M_earth with the Newton outer solver at rtol 1e-9: its residual has a smooth
    kink at the stop (slope 0.47), where the first root can be padded. The accepted profile
    reaches P_t within 0.5 P_t and is not padded."""
    root = zalmoxis.get_zalmoxis_root()
    cfg = _case(root, layer_eos_config=dict(ANALYTIC), planet_mass=5.972e24 * (1.0 + shift))
    r = zs.main(cfg, load_material_dictionaries(), None, os.path.join(root, 'input'))
    P = np.asarray(r['pressure'], dtype=float)
    assert r['converged'] and r['accepted_on'] == 'newton_tol'
    assert int(np.sum(P == 0.0)) == 0 and r['padded'] is False
    assert abs(P[-1] - P_T) < 0.5 * P_T


PALEOS = {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'}


@pytest.mark.physics_invariant
def test_the_paleos_newton_planet_reaches_its_surface_within_half_p_t():
    """PALEOS adiabatic 1 M_earth with the Newton outer solver at rtol 1e-9: the brentq root
    of xtol 1e6 Pa can miss P_t by up to 0.87 P_t; the refined root misses by under 0.5 P_t."""
    root = zalmoxis.get_zalmoxis_root()
    cfg = _case(root, layer_eos_config=dict(PALEOS), planet_mass=5.972e24)
    r = zs.main(cfg, load_material_dictionaries(), None, os.path.join(root, 'input'))
    P = np.asarray(r['pressure'], dtype=float)
    assert r['converged'] and r['accepted_on'] == 'newton_tol'
    assert r['padded'] is False and P[-1] > 0.0
    assert abs(P[-1] - P_T) < 0.5 * P_T


@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_the_accepted_root_meets_the_surface_boundary_condition(monkeypatch, validation_plot):
    """Boundary condition P(R) = P_t: on the smooth-kink planet (Analytic 1 M_earth, Newton
    rtol 1e-9) a pressure solve whose brentq root is padded, P(R) = 0, ends at a refined
    root within 0.1 P_t of P_t. The shifts run until one solve has a padded brentq root.
    The plot shows P(R) of every evaluated central pressure of that solve with both roots.
    See ``docs/Validation/hydrostatic_structure.md``."""
    calls, real = [], zs._refine_pressure_root

    def spy(root, info, evals, residual, p_target, maxiter):
        out = real(root, info, evals, residual, p_target, maxiter)
        calls.append((root, out[0], dict(evals)))
        return out

    monkeypatch.setattr(zs, '_refine_pressure_root', spy)
    zroot = zalmoxis.get_zalmoxis_root()
    padded = []
    for shift in SHIFTS:
        calls.clear()
        cfg = _case(zroot, layer_eos_config=dict(ANALYTIC), planet_mass=5.972e24 * (1 + shift))
        r = zs.main(cfg, load_material_dictionaries(), None, os.path.join(zroot, 'input'))
        assert r['padded'] is False
        assert abs(np.asarray(r['pressure'])[-1] - P_T) < 0.5 * P_T
        padded = [c for c in calls if c[2][c[0]] == -P_T]
        if padded:
            break
    assert padded, 'no pressure solve had a padded brentq root at any shift'
    first, refined, evals = padded[-1]
    assert abs(evals[refined]) <= 0.1 * P_T

    def figure():
        import matplotlib.pyplot as plt

        x = np.array(sorted(evals))
        pr = np.array([evals[k] for k in x]) + P_T
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(x / 1e9, pr / 1e5, 'o', ms=4, label='Evaluations')
        ax.axhline(P_T / 1e5, ls='--', label=r'$P_t$')
        ax.plot([first / 1e9], [0.0], 's', ms=8, label='brentq root')
        ax.plot([refined / 1e9], [(evals[refined] + P_T) / 1e5], '*', ms=12, label='Refined')
        ax.set_xlim(first / 1e9 - 0.02, first / 1e9 + 0.02)
        ax.set_ylim(-0.2, 5.0)
        ax.set_xlabel('Central pressure [GPa]')
        ax.set_ylabel('Surface pressure [bar]')
        ax.legend()
        fig.tight_layout()
        return fig

    validation_plot(figure, 'pressure_root_boundary_condition')
