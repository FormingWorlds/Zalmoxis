"""Real solves of the central-pressure root (``zalmoxis.solver._refine_pressure_root``).

Sweep setups from ``input/default.toml`` whose first brentq root can be padded: the accepted
profile must reach the surface. Padding depends on round-off, so each case runs at several
planet-mass shifts.
"""

from __future__ import annotations

import logging
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
CAP = zs._PADDED_REPLACE_MAX_MISS_PT  # largest miss of an accepted profile [P_t]


@pytest.mark.physics_invariant
@pytest.mark.parametrize('shift', SHIFTS, ids=['m0', 'm_plus_1e-9', 'm_minus_1e-9'])
def test_the_smooth_kink_planet_reaches_its_surface(shift):
    """Analytic 1 M_earth with the Newton outer solver at rtol 1e-9: its residual has a smooth
    kink at the stop (slope 0.47), where the first root can be padded. The accepted profile
    is not padded and misses P_t by at most the cap of a replaced padded root, 1 P_t."""
    root = zalmoxis.get_zalmoxis_root()
    cfg = _case(root, layer_eos_config=dict(ANALYTIC), planet_mass=5.972e24 * (1.0 + shift))
    r = zs.main(cfg, load_material_dictionaries(), None, os.path.join(root, 'input'))
    P = np.asarray(r['pressure'], dtype=float)
    assert r['converged'] and r['accepted_on'] == 'newton_tol'
    assert int(np.sum(P == 0.0)) == 0 and r['padded'] is False
    assert abs(P[-1] - P_T) <= CAP * P_T


PALEOS = {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'}


@pytest.mark.physics_invariant
def test_the_paleos_newton_planet_reaches_its_surface_within_the_cap(monkeypatch):
    """PALEOS adiabatic 1 M_earth with the Newton outer solver at rtol 1e-9: the accepted
    profile is not padded and misses P_t by at most 1 P_t, the cap of a padded root replaced
    by its reached end: at planet_mass (1 + 1e-13) M_earth that replacement ends at 0.536 P_t.
    Second solves move roots closer to P_t, and none ends further from it."""
    moves, real = [], zs._refine_pressure_root

    def spy(root, info, evals, residual, p_target, maxiter):
        out = real(root, info, evals, residual, p_target, maxiter)
        moves.append(abs(evals[out[0]]) - abs(evals[root]))
        return out

    monkeypatch.setattr(zs, '_refine_pressure_root', spy)
    root = zalmoxis.get_zalmoxis_root()
    cfg = _case(root, layer_eos_config=dict(PALEOS), planet_mass=5.972e24)
    r = zs.main(cfg, load_material_dictionaries(), None, os.path.join(root, 'input'))
    P = np.asarray(r['pressure'], dtype=float)
    assert r['converged'] and r['accepted_on'] == 'newton_tol'
    assert r['padded'] is False and P[-1] > 0.0
    assert abs(P[-1] - P_T) <= CAP * P_T
    assert sum(m < 0.0 for m in moves) > len(moves) // 2 and max(moves) <= 0.0


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
        assert abs(np.asarray(r['pressure'])[-1] - P_T) <= CAP * P_T
        padded = [c for c in calls if c[2][c[0]] == -P_T]
        if padded:
            break
    assert padded, 'no pressure solve had a padded brentq root at any shift'
    first, refined, evals = padded[-1]
    assert abs(evals[refined]) <= 0.1 * P_T

    def figure():
        import matplotlib.pyplot as plt

        x = np.array(sorted(evals))
        near = np.abs(x - first) < 1e7
        dx = (x[near] - first) / 1e6
        pr = (np.array([evals[k] for k in x[near]]) + P_T) / 1e5
        top = pr > 3.0
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(dx[~top], pr[~top], 'o', ms=4, label='Evaluated')
        if top.any():
            ax.plot(dx[top], np.full(top.sum(), 2.9), '^', ms=6, label='Over 3 bar')
        ax.axhline(P_T / 1e5, ls='--', label=r'$P_t$')
        ax.plot([0.0], [0.0], 's', ms=8, label='brentq root')
        ax.plot(
            [(refined - first) / 1e6],
            [(evals[refined] + P_T) / 1e5],
            '*',
            ms=12,
            label='Refined',
        )
        ax.set_ylim(-0.2, 3.0)
        ax.set_yticks([0, 1, 2, 3])
        ax.set_xlabel(r'$P_c$ - brentq root [MPa]')
        ax.set_ylabel('Surface pressure [bar]')
        ax.legend()
        fig.tight_layout()
        return fig

    validation_plot(figure, 'pressure_root_boundary_condition')


# The 6 Newton setups of the sweep, the PROTEUS default outer solver: name, EOS, mass [M_earth], JAX.
NEWTON_SETUPS = (
    ('Analytic 1', ANALYTIC, 1.0, False),
    ('PALEOS 1', PALEOS, 1.0, False),
    ('PALEOS 1 JAX', PALEOS, 1.0, True),
    ('PALEOS 3 JAX', PALEOS, 3.0, True),
    ('PALEOS 5', PALEOS, 5.0, False),
    ('PALEOS 10', PALEOS, 10.0, False),
)
# P(R) near the stop scatters on the JAX path; a padded accept stays possible (1 of 16 round-off starts).
MAY_PAD = {'PALEOS 1 JAX'}


@pytest.mark.accept_solves_long
@pytest.mark.timeout(4800)
@pytest.mark.reference_pinned
@pytest.mark.physics_invariant
def test_every_newton_setup_meets_the_surface_boundary_condition(validation_plot):
    """Boundary condition P(R) = P_t on the 6 Newton setups (rtol 1e-9, 150 layers): each
    accepted profile outside ``MAY_PAD`` reaches the surface, the padded flag holds exactly
    when P(R) = 0, and a reached profile misses P_t by at most 1 P_t. A JAX setup takes T(r)
    from a numpy Picard solve at rtol 1e-5, as PROTEUS hands it the interior profile.
    See ``docs/Validation/hydrostatic_structure.md``."""
    zroot = zalmoxis.get_zalmoxis_root()
    md, misses, padded = load_material_dictionaries(), [], []
    for name, eos, mass, jax in NEWTON_SETUPS:
        cfg = _case(zroot, layer_eos_config=dict(eos), planet_mass=5.972e24 * mass)
        tarr = None
        if jax:
            pre = dict(
                cfg, outer_solver='picard', relative_tolerance=1e-5, absolute_tolerance=1e-6
            )
            r0 = zs.main(pre, md, None, os.path.join(zroot, 'input'))
            tarr = (np.asarray(r0['radii'], float), np.asarray(r0['temperature'], float))
            cfg['use_jax'] = True
        r = zs.main(cfg, md, None, os.path.join(zroot, 'input'), temperature_arrays=tarr)
        P = np.asarray(r['pressure'], dtype=float)
        assert r['converged'] and P[-1] >= 0.0 and r['padded'] is bool(P[-1] == 0.0)
        assert name in MAY_PAD or not r['padded'], name
        misses.append(abs(P[-1] - P_T) / P_T)
        padded.append(r['padded'])
    assert max(m for m, p in zip(misses, padded) if not p) <= CAP, misses

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        x = range(len(misses))
        for mark, want, label in (('o', False, 'Reached'), ('s', True, 'Padded, $P(R) = 0$')):
            pts = [(i, m) for i, m, p in zip(x, misses, padded) if p is want]
            if pts:
                ax.plot(*zip(*pts), mark, ms=7, label=label)
        ax.axhline(CAP, ls='--', label='Tolerance')
        ax.set_xticks(x, [s[0] for s in NEWTON_SETUPS], rotation=30, ha='right')
        ax.set_ylim(0.0, 1.1 * CAP)
        ax.set_ylabel(r'$|P(R) - P_t| / P_t$')
        ax.legend()
        fig.tight_layout()
        return fig

    validation_plot(figure, 'pressure_root_newton_setups')


def test_a_jump_at_the_stop_returns_a_flagged_padded_profile(caplog):
    """Analytic 10 M_earth with the Picard outer solver at rtol 1e-5: P(R) jumps from 0 at the
    stop to about 2e8 Pa, so no reached evaluation lies within 1 P_t and the padded root
    stays. The result is flagged padded, P(R) = 0, and main warns once."""
    root = zalmoxis.get_zalmoxis_root()
    cfg = _case(
        root,
        layer_eos_config=dict(ANALYTIC),
        planet_mass=10 * 5.972e24,
        outer_solver='picard',
        relative_tolerance=1e-5,
        absolute_tolerance=1e-6,
    )
    with caplog.at_level(logging.WARNING, logger='zalmoxis.solver'):
        r = zs.main(cfg, load_material_dictionaries(), None, os.path.join(root, 'input'))
    P = np.asarray(r['pressure'], dtype=float)
    assert r['converged'] and P[-1] == 0.0
    assert r['padded'] is True
    assert sum('surface stop' in rec.getMessage() for rec in caplog.records) == 1
