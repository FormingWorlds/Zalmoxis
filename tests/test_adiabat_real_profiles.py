"""The adiabat integration on stored structure profiles of real solves, with the PALEOS tables.

The profiles (``tests/data/adiabat_profiles.npz``, written by
``python -m tools.validation.adiabat_profiles write``) are 4 round-off starts of
``input/default.toml`` and the 2 PROTEUS initial structure setups. See
``docs/Validation/thermal_state.md``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from tests._plot_checks import assert_text_clear, legend_below
from tools.validation import adiabat_profiles as ap
from zalmoxis.eos.temperature import ADIABAT_P_FLOOR as P_FLOOR  # Pa

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(600)]

DATA = Path(__file__).parent / 'data' / 'adiabat_profiles.npz'


def _need_paleos(*extra):
    from zalmoxis import get_zalmoxis_root

    root = Path(get_zalmoxis_root()) / 'data'
    for folder in ('EOS_PALEOS_iron', 'EOS_PALEOS_MgSiO3_unified', *extra):
        if not (root / folder).is_dir():
            pytest.skip(f'PALEOS table folder {folder} not found')


@pytest.fixture(scope='module')
def profiles():
    """{case: (record, call kwargs)} of every stored profile."""
    _need_paleos()
    recs = ap.load(DATA)
    return {c: (recs[c], ap.call_kwargs(recs[c])) for c in ap.CASES}


def _cmb_index(kw):
    n = len(kw['pressure'])
    return max(1, min(int(np.searchsorted(kw['mass_enclosed'], kw['cmb_mass'])), n - 1))


def _on_adiabat(kw):
    """Mask of the nodes the adiabat integrates: all, or the CMB node outward."""
    keep = np.ones(len(kw['pressure']), bool)
    if kw['anchor'] == 'cmb':
        keep[: _cmb_index(kw)] = False
    return keep


def _eos(kw):
    """The EOS arguments of ``get_mixed_nabla_ad`` from call kwargs, in their order."""
    return tuple(kw[k] for k in ap.EOS_ARGS)


def _reference(kw, method='DOP853', p_ice=None, cut_ice=True):
    """scipy ``solve_ivp`` (``method``, rtol 1e-13, atol 1e-15) of d ln T / d ln P = nabla over
    each node segment, with the rules of the routine: a segment that holds the CMB is cut
    there, with ln P linear in the enclosed mass between its nodes; a segment that holds the
    boundary of an ice layer is cut there at P linear in m, or at ``p_ice(lo, hi, b)`` [Pa]
    (``lo``, ``hi`` the (m, P) of its inner and outer node), or not at all without
    ``cut_ice``. Each piece takes its own mixture, a piece that ends at its node the mixture of
    that node unless the node lies on a boundary; T held below 1e5 Pa (a padded top, P = 0,
    counts as 1e5 Pa) and for a T-independent layer, and the node caps. The stored profiles
    hold no inner core; the unit tests of tests/test_core_two_phase.py check the ICB cut.
    Returns (T, count of nabla reads that were None or not positive)."""
    from scipy.integrate import solve_ivp

    from zalmoxis.mixing import get_mixed_nabla_ad
    from zalmoxis.structure_model import get_layer_mixture

    assert not kw.get('icb_mass'), 'the reference has no ICB cut'
    P, m, T, order, inward = ap.node_walk(kw)
    eos, bad, lo, cmb = _eos(kw), [0], (0.0 if inward else 100.0), kw['cmb_mass']

    def mixture(mass):
        return get_layer_mixture(mass, cmb, kw['core_mantle_mass'], kw['layer_mixtures'])

    def rhs(x, y, mx):
        v = get_mixed_nabla_ad(math.exp(x), math.exp(y[0]), mx, *eos)
        if v is None or not v > 0:
            bad[0] += 1
            return [0.0]
        return [v]

    ice = kw['core_mantle_mass'] if cut_ice else None
    bounds = sorted(
        {b for b in (cmb, ice) if b and mixture(b * (1 - 1e-9)) is not mixture(b * (1 + 1e-9))}
    )
    for i, j in order:
        ends = [(m[j], P[j]), (m[i], P[i])]
        (m_lo, p_lo), (m_hi, p_hi) = sorted(ends)
        if p_lo > 0 and p_hi > 0:
            cuts = [
                (b, math.exp(np.interp(b, (m_lo, m_hi), np.log((p_lo, p_hi)))))
                if b == cmb
                else (
                    b,
                    p_ice((m_lo, p_lo), (m_hi, p_hi), b)
                    if p_ice
                    else np.interp(b, (m_lo, m_hi), (p_lo, p_hi)),
                )
                for b in bounds
                if m_lo < b < m_hi
            ]
            ends[1:1] = cuts[::-1] if ends[0][0] > m[i] else cuts
        on_bound = any(abs(m[i] - b) <= 1e-12 * b for b in bounds)
        t = T[j]
        for (m_a, p_a), (m_b, p_b) in zip(ends, ends[1:]):
            mx = mixture(m[i] if m_b == m[i] and not on_bound else 0.5 * (m_a + m_b))
            x0, x1 = math.log(max(p_a, P_FLOOR)), math.log(max(p_b, P_FLOOR))
            if not mx.has_tdep() or (x1 - x0) * (1 if inward else -1) <= 0:
                continue
            y = solve_ivp(
                rhs, (x0, x1), [math.log(t)], method=method, rtol=1e-13, atol=1e-15, args=(mx,)
            )
            t = min(max(math.exp(y.y[0, -1]), lo), 1e5)
        T[i] = t
    return T, bad[0]


# H2O layer positions of the ice cases: (profile, top pressure [Pa] or None as stored, the
# node segment from the top whose middle holds the boundary)
ICE_POSITIONS = [
    ('default_k0', 1.2e5, 1),
    ('ic_dummy', None, 1),
    ('default_k0', None, 2),
    ('default_k0', None, 5),
    ('default_k0', None, 20),
    ('default_k0', None, 40),
]
ICE_IDS = ['top_surface', 'top_cmb', 'seg2', 'seg5', 'seg20', 'seg40']


def _ice_case(profiles, case, p_top, k, eos):
    """Call kwargs of a stored profile with an ``eos`` layer from the middle of the k-th node
    segment from the top outward (for the temperature only, not a solved 3-layer planet)."""
    from zalmoxis.mixing import LayerMixture

    if eos.startswith('PALEOS'):
        _need_paleos('EOS_PALEOS_H2O')
    kw = dict(profiles[case][1])
    P, m = np.array(kw['pressure'], float), np.asarray(kw['mass_enclosed'], float)
    if p_top is not None:
        P[-1] = p_top
    n = P.size
    layers = dict(kw['layer_mixtures'], ice_layer=LayerMixture([eos], [1.0]))
    cmm = 0.5 * (m[n - 1 - k] + m[n - k])
    return dict(kw, pressure=P, layer_mixtures=layers, core_mantle_mass=cmm)


def _hydrostatic(kw):
    """``p_ice`` of ``_reference``: P [Pa] at the boundary mass b from dP/dm = -G m / (4 pi r^4)
    integrated inward from the outer node of the segment, r^3 linear in m between its nodes."""
    G = 6.674e-11
    m, r = np.asarray(kw['mass_enclosed'], float), np.asarray(kw['radii'], float)

    def p_at(lo, hi, b):
        k = int(np.searchsorted(m, lo[0]))
        mm = np.linspace(hi[0], b, 4001)
        r3 = np.interp(mm, (m[k], m[k + 1]), (r[k] ** 3, r[k + 1] ** 3))
        return float(hi[1] - np.trapezoid(G * mm / (4 * math.pi * r3 ** (4 / 3)), mm))

    return p_at


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_midpoint_adiabat_meets_the_reference_on_real_profiles(profiles, validation_plot):
    """On the 6 stored profiles the default call stays within 1 K of a DOP853 integration of the
    same ODE (rtol 1e-13), which agrees with RK45 at rtol 1e-13 to 0.01 K, at every node on the
    adiabat, with no nabla read that is missing or not positive. The forward Euler step in P per
    node, from ``forward_euler`` of the profile tool, equals the stored T_euler to a relative
    1e-12 and is at least 5 times farther and more than 2 K from the reference on every profile.
    A padded top gives the profile of a top at 1 Pa within 1e-6 K."""
    from zalmoxis.eos.temperature import compute_adiabatic_temperature

    kw0 = profiles['default_k0'][1]
    p_cmb = float(np.interp(kw0['cmb_mass'], kw0['mass_enclosed'], kw0['pressure']))
    for case, (rec, kw) in profiles.items():
        T = compute_adiabatic_temperature(**kw)
        ref, bad = _reference(kw)
        keep = _on_adiabat(kw)
        assert float(np.max(np.abs(ref - _reference(kw, 'RK45')[0])[keep])) <= 0.01, case
        err = float(np.max(np.abs(T - ref)[keep]))
        euler = ap.forward_euler(kw)
        # the stored comparison scheme, to the round-off of the CPU
        np.testing.assert_allclose(euler, rec['T_euler'], rtol=1e-12, atol=0)
        native = float(np.max(np.abs(euler - ref)[keep]))
        assert bad == 0, case
        assert err <= 1.0, (case, err)
        assert native > max(5 * err, 2.0), (case, native, err)
        if rec['padded']:  # a padded top integrates like a top at 1 Pa
            P1 = np.asarray(kw['pressure'], float).copy()
            P1[-1] = 1.0
            T1 = compute_adiabatic_temperature(**{**kw, 'pressure': P1})
            assert kw['pressure'][-1] == 0.0 and float(np.max(np.abs(T - T1))) <= 1e-6
        if case == 'default_k0':
            k0 = (kw['pressure'][keep], T[keep], ref[keep], euler[keep])
        print(f'{case}: midpoint {err:.4f} K, forward Euler {native:.2f} K from the reference')

    def profile_figure():
        import matplotlib.pyplot as plt

        from zalmoxis.melting_curves import iron_melting_anzellini13

        fig, ax = plt.subplots()
        P, T, ref, native = k0
        top = P > 0
        gpa = P[top] / 1e9
        core = P >= p_cmb
        ax.plot(gpa, native[top], ls='--', color='C2', label='forward Euler step')
        ax.plot(gpa, T[top], color='C1', label='midpoint, default')
        ax.plot(
            gpa, ref[top], ls=':', color=plt.rcParams['text.color'], label='DOP853 reference'
        )
        ax.plot(
            P[core] / 1e9,
            iron_melting_anzellini13(P[core]),
            alpha=0.5,
            label='iron melting (core)',
        )
        ax.set_xlabel('Pressure [GPa]')
        ax.set_ylabel('Temperature [K]')
        ax.minorticks_off()
        legend_below(fig, ncol=2)
        return assert_text_clear(fig)

    validation_plot(profile_figure, 'adiabat_real_profile')


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize('eos', ['PALEOS:H2O', 'Seager2007:H2O'])
@pytest.mark.parametrize(('case', 'p_top', 'k'), ICE_POSITIONS, ids=ICE_IDS)
def test_the_adiabat_meets_the_reference_across_an_ice_boundary(
    profiles, case, p_top, k, eos, request
):
    """Case B: an H2O layer from the middle of the k-th node segment from the top (the top
    cell with the surface and the CMB anchor, the 2nd, 5th, 20th and 40th segment), T-dependent
    PALEOS or T-independent Seager2007. The default call stays within 1 K of the DOP853
    reference that cuts the boundary at the same pressure, at every node on the adiabat, with no
    nabla read that is missing or not positive; the reference without the cut lies more than
    1 K off. The reference shares the boundary pressure of the code (case B2 checks it)."""
    from zalmoxis.eos.temperature import compute_adiabatic_temperature

    kw = _ice_case(profiles, case, p_top, k, eos)
    T = compute_adiabatic_temperature(**kw)
    ref, bad = _reference(kw)
    uncut = _reference(kw, cut_ice=False)[0]
    keep = _on_adiabat(kw)
    err, off = (float(np.max(np.abs(x - ref)[keep])) for x in (T, uncut))
    print(
        f'{case} {p_top} seg {k} {eos}: cut {err:.3g} K, uncut {off:.3g} K from the reference'
    )
    assert bad == 0 and err <= 1.0, (bad, err)
    assert off > 1.0, off
    if (k, eos) != (5, 'PALEOS:H2O') or request.config.getoption('validation_plots') is None:
        return

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        P = np.asarray(kw['pressure'], float)
        on = keep & (P > 0)
        gpa = P[on] / 1e9
        floor = 1e-3  # K: shows a node that meets the reference to round-off
        ax.plot(
            gpa,
            np.maximum(np.abs(uncut - ref)[on], floor),
            '--',
            label='one mixture per segment',
        )
        ax.plot(gpa, np.maximum(np.abs(T - ref)[on], floor), label='cut at the boundary')
        p_b = np.interp(kw['core_mantle_mass'], kw['mass_enclosed'], P) / 1e9
        ax.axvline(p_b, alpha=0.4, lw=1.0, label='mantle and ice boundary')
        ax.set(
            xscale='log', yscale='log', xlabel='Pressure [GPa]', ylabel='|T - reference| [K]'
        )
        ax.minorticks_off()
        legend_below(fig, ncol=3)
        return assert_text_clear(fig)

    request.getfixturevalue('validation_plot')(figure, 'adiabat_ice_boundary')


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize('eos', ['PALEOS:H2O', 'Seager2007:H2O'])
@pytest.mark.parametrize(('case', 'p_top', 'k'), ICE_POSITIONS, ids=ICE_IDS)
def test_the_ice_boundary_pressure_meets_hydrostatic_equilibrium(profiles, case, p_top, k, eos):
    """Case B2: the positions of case B against a DOP853 reference whose boundary pressure
    comes from dP/dm = -G m / (4 pi r^4) integrated inside the segment (r^3 linear in m). The
    default call, with P linear in m, stays within 10 K of it in the top cell and 2.5 K deeper
    at every node on the adiabat."""
    from zalmoxis.eos.temperature import compute_adiabatic_temperature

    kw = _ice_case(profiles, case, p_top, k, eos)
    T = compute_adiabatic_temperature(**kw)
    ref, bad = _reference(kw, p_ice=_hydrostatic(kw))
    err = float(np.max(np.abs(T - ref)[_on_adiabat(kw)]))
    print(f'{case} {p_top} seg {k} {eos}: {err:.3g} K from the hydrostatic reference')
    assert bad == 0 and err <= (10.0 if k == 1 else 2.5), (bad, err)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('eos', ['PALEOS:H2O', 'Seager2007:H2O'])
@pytest.mark.parametrize(('case', 'p_top', 'k'), ICE_POSITIONS[2:], ids=ICE_IDS[2:])
def test_the_lookup_follows_the_cut_inside_the_boundary_segment(profiles, case, p_top, k, eos):
    """The T(P) table of the density solve gains the ice boundary of a segment below the top
    cell as one more point, with T stepped from the outer node, so the lookup does not
    interpolate across the change of slope: inside the segment it stays within 1 K of a DOP853
    march cut at the same pressure (10 K for PALEOS H2O in the 2nd segment, where the ice
    piece is steep and curved: 8.75 K measured there; a lookup without the point misses by 2.4
    to 79 K at the boundary), and within 0.01 K at the inner node."""
    from scipy.integrate import solve_ivp

    from zalmoxis.eos.temperature import compute_adiabatic_temperature as adiabat
    from zalmoxis.mixing import get_mixed_nabla_ad
    from zalmoxis.solver import _adiabat_table, _shell_adiabat, _table_lookup
    from zalmoxis.structure_model import get_layer_mixture

    kw = _ice_case(profiles, case, p_top, k, eos)
    P, m = np.asarray(kw['pressure'], float), np.asarray(kw['mass_enclosed'], float)
    T, cmm = adiabat(**kw), kw['core_mantle_mass']

    def shell(radii, pressure, mass, top):
        call = dict(kw, radii=radii, pressure=pressure, mass_enclosed=mass, anchor='surface')
        return adiabat(**dict(call, surface_temperature=top))

    lp, t = _adiabat_table(
        P, T, m, lambda q, t0, mq: _shell_adiabat(shell, q, t0, mq), False, cmm
    )
    look = _table_lookup(lp, t)
    lo, hi = P.size - 1 - k, P.size - k
    p_b = float(np.interp(cmm, (m[lo], m[hi]), (P[lo], P[hi])))
    x = np.linspace(math.log(P[hi]), math.log(P[lo]), 400)
    march, y0 = np.empty(x.size), math.log(T[hi])
    for x_a, x_b, mass in ((x[0], math.log(p_b), m[hi]), (math.log(p_b), x[-1], m[lo])):
        on = (x >= x_a) & (x <= x_b)
        mx = get_layer_mixture(mass, kw['cmb_mass'], cmm, kw['layer_mixtures'])
        if not mx.has_tdep():
            march[on] = math.exp(y0)
            continue

        def nab(x_, y, mx=mx):
            return [get_mixed_nabla_ad(math.exp(x_), math.exp(y[0]), mx, *_eos(kw))]

        sol = solve_ivp(
            nab,
            (x_a, x_b),
            [y0],
            'DOP853',
            t_eval=x[on],
            rtol=1e-12,
            atol=1e-14,
            dense_output=True,
        )
        march[on], y0 = np.exp(sol.y[0]), float(sol.sol(x_b)[0])
    miss = float(np.max(np.abs([look(math.exp(v)) - w for v, w in zip(x, march)])))
    print(f'{case} seg {k} {eos}: lookup {miss:.3g} K from the march')
    assert miss < (10.0 if (k, eos) == (2, 'PALEOS:H2O') else 1.0), miss
    assert abs(look(P[lo]) - march[-1]) < 0.01


@pytest.mark.physics_invariant
@pytest.mark.parametrize(
    ('case', 'nodes'),
    [('ic_dummy', (148, 147, 146, 145)), ('default_k0', (147, 146, 145, 144))],
)
def test_pressure_is_near_linear_in_mass_over_a_top_segment(profiles, case, nodes):
    """Why the ice boundary takes P linear in m: a node of a top segment, dropped and predicted
    from its 2 neighbours at its own mass, comes back within 1 % with P linear in m; with ln P
    linear in m (the CMB rule) the top cell of ``ic_dummy`` misses by more than 50 %."""
    kw = profiles[case][1]
    P, m = np.asarray(kw['pressure'], float), np.asarray(kw['mass_enclosed'], float)
    for k in nodes:
        lin = np.interp(m[k], (m[k - 1], m[k + 1]), (P[k - 1], P[k + 1]))
        assert abs(lin / P[k] - 1) < 0.01, (k, lin / P[k] - 1)
    if case == 'ic_dummy':
        k = nodes[0]
        log = math.exp(np.interp(m[k], (m[k - 1], m[k + 1]), np.log((P[k - 1], P[k + 1]))))
        assert log / P[k] - 1 < -0.5, log / P[k] - 1


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize('anchor', ['cmb', 'surface'])
def test_temperature_is_continuous_across_one_bar(profiles, anchor, request):
    """On the PROTEUS ``dummy`` structure (top near 1 atm) the top pressure is set to 41 values
    from 0.9e5 to 1.1e5 Pa: no node temperature changes by more than 0.01 K between neighbour
    values, both anchors. The forward Euler step, which skips a step once an end is
    below 1e5 Pa, jumps by more than 10 K at some node under both anchors."""
    from zalmoxis.eos.temperature import compute_adiabatic_temperature

    rec, kw = profiles['ic_dummy']
    kw = {**kw, 'anchor': anchor}
    P0 = np.asarray(kw['pressure'], float)
    assert P0[-1] > 0 and P0[-2] > 1.1e5
    tops = np.linspace(0.9e5, 1.1e5, 41)
    T, native = [], []
    for p in tops:
        P = P0.copy()
        P[-1] = p
        T.append(compute_adiabatic_temperature(**{**kw, 'pressure': P}))
        native.append(ap.forward_euler({**kw, 'pressure': P}))
    T = np.array(T)
    step = float(np.max(np.abs(np.diff(T, axis=0))))
    assert step <= 0.01, step
    native = np.array(native)
    assert float(np.max(np.abs(np.diff(native, axis=0)))) > 10.0
    if anchor == 'cmb':

        def figure():
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot(
                tops / 1e5,
                native[:, -1] - native[-1, -1],
                ls='--',
                label='step skipped below 1 bar',
            )
            ax.plot(tops / 1e5, T[:, -1] - T[-1, -1], label='midpoint with the 1 bar floor')
            ax.set_xlabel('Top pressure [bar]')
            ax.set_ylabel('Top node T minus its value at 1.1 bar [K]')
            ax.minorticks_off()
            legend_below(fig, ncol=2)
            return assert_text_clear(fig)

        request.getfixturevalue('validation_plot')(figure, 'adiabat_low_pressure')


class TopCell(NamedTuple):
    x: np.ndarray  # ln P grid over the cell
    lookup: np.ndarray  # T [K] of the table
    march: np.ndarray  # T [K] of the DOP853 march
    end: float  # shell at the inner node minus its T [K]
    mass_used: np.ndarray  # masses [kg] the table gave the shell points
    mass_rule: np.ndarray  # masses [kg] of the rule at the same points
    layers_differ: bool  # the 2 nodes of the cell lie in different layers


def _top_cell(kw, p_top=None, layer_eos=None):
    """The T(P) lookup that the density solve builds on a stored profile, with its top node at
    p_top (None: as stored; 0: padded), over the top cell from max(P_top, 1e5 Pa) to the next
    node, and a DOP853 march (rtol 1e-12, atol 1e-14 on ln T) of d ln T / d ln P = nabla over
    the cell from the top T. Over an unpadded cell the rule gives the shell points the mass of
    P linear in m between the 2 nodes, and the march changes mixture at the boundary pressure
    of that rule; over a padded cell both take the mass of the node that the adiabat gives the
    cell (inner node inward, the node outside it outward from the CMB). A layer without a
    T-dependent EOS holds T, as the adiabat does. With ``layer_eos`` (an EOS name) a layer of
    it starts in the middle of the top cell. Returns a ``TopCell``."""
    from scipy.integrate import solve_ivp

    from zalmoxis.eos.temperature import compute_adiabatic_temperature as adiabat
    from zalmoxis.mixing import LayerMixture, get_mixed_nabla_ad
    from zalmoxis.solver import _TOP_SHELL_POINTS, _adiabat_table, _shell_adiabat
    from zalmoxis.structure_model import get_layer_mixture

    P, m = np.array(kw['pressure'], float), np.asarray(kw['mass_enclosed'], float)
    if p_top is not None:
        P[-1] = p_top
    kw = dict(kw, pressure=P)
    if layer_eos:
        layers = dict(kw['layer_mixtures'], ice_layer=LayerMixture([layer_eos], [1.0]))
        kw.update(layer_mixtures=layers, core_mantle_mass=0.5 * (m[-1] + m[-2]))
    T = adiabat(**kw)

    def shell(radii, pressure, mass, top):
        call = dict(kw, radii=radii, pressure=pressure, mass_enclosed=mass, anchor='surface')
        return adiabat(**dict(call, surface_temperature=top))

    used = []

    def top_shell(q, t_top, mass):
        used.append(mass)
        return _shell_adiabat(shell, q, t_top, mass)

    def mixture(mass):
        return get_layer_mixture(
            mass, kw['cmb_mass'], kw['core_mantle_mass'], kw['layer_mixtures']
        )

    boundary = kw['core_mantle_mass'] if layer_eos else None
    lp, t = _adiabat_table(P, T, m, top_shell, kw['anchor'] == 'cmb', boundary)
    pad = P[-1] <= 0
    inner = int(np.flatnonzero(P > 0)[-1 if pad else -2])  # deeper node of the cell
    node = inner + 1 if kw['anchor'] == 'cmb' else inner
    q = np.geomspace(max(P[-1], P_FLOOR), P[inner], _TOP_SHELL_POINTS + 1)[:-1]
    if boundary is not None and not pad:  # the boundary pressure is one more shell point
        q = np.union1d(q, np.interp(boundary, (m[inner], m[-1]), (P[inner], P[-1])))
    q = np.append(q, P[inner])
    rule = (
        np.full(q.size, m[node]) if pad else np.interp(q, (P[-1], P[inner]), (m[-1], m[inner]))
    )
    end = _shell_adiabat(shell, q, T[-1], rule)[-1] - T[inner]
    upper, lower = mixture(m[-1]), mixture(m[inner])
    differ = upper.components != lower.components
    cell = (end, np.broadcast_to(used[0], (q.size - 1,)), rule[:-1], differ)
    x = np.linspace(math.log(max(P[-1], P_FLOOR)), math.log(P[inner]), 400)
    look = np.interp(x / math.log(10), lp, t)
    if pad or not differ:
        pieces = [(x[0], x[-1], mixture(m[node]))]
    else:  # the boundary at P linear in m between the 2 nodes, held at the floor
        p_b = float(np.interp(kw['core_mantle_mass'], (m[inner], m[-1]), (P[inner], P[-1])))
        x_b = min(max(math.log(p_b), x[0]), x[-1])
        pieces = [(x[0], x_b, upper), (x_b, x[-1], lower)]
    march, y0 = np.empty(x.size), math.log(T[-1])
    for x_a, x_b, mx in pieces:
        on = (x >= x_a) & (x <= x_b)
        if not mx.has_tdep() or x_b <= x_a:
            march[on] = math.exp(y0)
            continue

        def nab(x_, y, mx=mx):
            return [get_mixed_nabla_ad(math.exp(x_), math.exp(y[0]), mx, *_eos(kw))]

        sol = solve_ivp(
            nab,
            (x_a, x_b),
            [y0],
            'DOP853',
            t_eval=x[on],
            rtol=1e-12,
            atol=1e-14,
            dense_output=True,
        )
        march[on], y0 = np.exp(sol.y[0]), float(sol.sol(x_b)[0])
    return TopCell(x, look, march, *cell)


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
@pytest.mark.parametrize(
    ('case', 'p_top'),
    [
        ('ic_dummy', None),
        ('ic_all_options', None),
        ('default_k0', 1.2e5),
        ('default_k0', 9.09e4),
    ],
)
def test_the_lookup_follows_the_adiabat_over_an_unpadded_top_cell(
    profiles, case, p_top, request
):
    """An unpadded top: the T(P) lookup over the top cell, from max(P_top, 1e5 Pa) to the next
    node (2 to 4 decades in P), lies within 0.3 K of an independent DOP853 march of that layer
    from the top T, with a top node above and below 1e5 Pa, on both anchors; a straight line in
    log10 P over the cell lies 9 to 28 K off on these profiles. The shell, carried to the next
    node, meets the T of that node within 1e-5 K (at most 2e-7 K measured on these cases)."""
    x, look, ref, end, got, want, _ = _top_cell(profiles[case][1], p_top)
    assert np.array_equal(got, want)  # the shell points take the masses of the rule
    assert np.max(np.abs(look - ref)) < 0.3, np.max(np.abs(look - ref))
    assert abs(end) < 1e-5, end
    if case != 'ic_dummy' or request.config.getoption('validation_plots') is None:
        return

    def figure():
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        c = plt.rcParams['axes.prop_cycle'].by_key()['color']
        k = np.r_[0 : x.size : 8, x.size - 1]  # every 8th point keeps the file small
        p = np.exp(x[k])
        ax.plot(p[[0, -1]], ref[[0, -1]], '--', color=c[1], lw=1.2, label='straight line')
        ax.plot(p, look[k], color=c[0], lw=2.4, label='lookup')
        ax.plot(p, ref[k], ':', color=c[2], lw=1.6, zorder=3, label='DOP853 integration')
        ax.set(xscale='log', xlabel='Pressure [Pa]', ylabel='Temperature [K]')
        ax.xaxis.set_major_locator(plt.LogLocator(numticks=3))  # 3 labels keep the file small
        ax.grid(False)
        legend_below(fig, ncol=3)
        return assert_text_clear(fig)

    # one plot pair, written only with --validation-plots
    request.getfixturevalue('validation_plot')(figure, 'adiabat_top_cell_lookup')


@pytest.mark.physics_invariant
@pytest.mark.parametrize('eos', ['Seager2007:H2O', 'PALEOS:H2O'])
@pytest.mark.parametrize(
    ('case', 'p_top'),
    [('default_k0', 1.2e5), ('ic_dummy', 0.0), ('ic_dummy', None)],
    ids=['surface_unpadded', 'cmb_padded', 'cmb_unpadded'],
)
def test_the_top_cell_shell_meets_the_next_node_across_a_layer_boundary(
    profiles, case, p_top, eos
):
    """An H2O layer (T-independent Seager2007 or T-dependent PALEOS) from the middle of the top
    cell outward, on both anchors. Over an unpadded cell the shell points take the masses of P
    linear in m between the 2 nodes, so the shell cuts at the boundary pressure of the adiabat;
    over a padded cell they take the mass of the node whose mixture the adiabat gives the cell.
    The boundary pressure is one more shell point. The shell integrated on to the inner node
    meets its T within 1e-5 K; for PALEOS H2O within 1e-4 K with the CMB anchor (the midpoint
    step out and back) and 0.01 K with the surface anchor, where T rises by about 3500 K over
    the cell and the shell and the node segment split their substeps differently (2.7e-3 K
    measured, 5.5e-5 K at a 5 times smaller step). The lookup lies within 0.5 K of a DOP853
    march that changes mixture at the same pressure."""
    if eos.startswith('PALEOS'):
        _need_paleos('EOS_PALEOS_H2O')
    kw = profiles[case][1]
    assert kw['anchor'] == ('cmb' if case == 'ic_dummy' else 'surface')
    cell = _top_cell(kw, p_top, eos)
    assert cell.layers_differ and np.array_equal(cell.mass_used, cell.mass_rule)
    cmb_anchor = kw['anchor'] == 'cmb'
    miss = float(np.max(np.abs(cell.lookup - cell.march)))
    print(
        f'{case} top {p_top} {eos}: shell end {cell.end:.3g} K, lookup minus march {miss:.3g} K'
    )
    paleos = eos.startswith('PALEOS')
    assert abs(cell.end) < ((1e-4 if cmb_anchor else 0.01) if paleos else 1e-5), cell.end
    assert miss < 0.5, miss


@pytest.mark.physics_invariant
@pytest.mark.parametrize('case', ['default_k0', 'ic_dummy', 'ic_all_options'])
def test_the_top_cell_lookup_does_not_flip_with_the_padding(profiles, case):
    """One structure (default start 0, surface anchor; the 2 PROTEUS profiles, CMB anchor) with
    its top node at P = 0 (padded), 1 Pa, 1e5 Pa and 1 atm: above 1e5 Pa the lookup is the
    padded one to round-off for a top at or below 1e5 Pa (the adiabat holds T below 1 bar, so
    the shell is the same), and within 0.01 K for 1 atm (at most 1.6e-3 K measured); a solve
    that pads or not by round-off reads the same temperatures."""
    kw = profiles[case][1]
    x, padded, *_ = _top_cell(kw, 0.0)
    for p_top, bound in ((1.0, 1e-9), (P_FLOOR, 1e-9), (101325.0, 0.01)):
        y, look, *_ = _top_cell(kw, p_top)
        on = y >= math.log(P_FLOOR)
        diff = np.max(np.abs(look[on] - np.interp(y[on], x, padded)))
        assert diff <= bound, (p_top, diff)


@pytest.mark.physics_invariant
@pytest.mark.parametrize('case', ['default_k0', 'default_k1', 'default_k2', 'default_k3'])
def test_the_cmb_point_of_the_icb_readers_is_on_the_core_adiabat(profiles, case):
    """Case V2 of the CMB point. On each stored default profile the CMB point of the ICB
    readers, with the core gradient stepped from the last core node (``cmb_point_temperature``),
    meets the adiabat with a node inserted at the core mass to 1e-6 K (self-consistency of the
    step); T linear in ln P across the segment lies more than 3 K above it."""
    from zalmoxis.core_profile import _core_nodes
    from zalmoxis.eos.temperature import cmb_point_temperature, compute_adiabatic_temperature
    from zalmoxis.mixing import get_mixed_nabla_ad

    kw = profiles[case][1]
    m, p, r = (np.asarray(kw[k], float) for k in ('mass_enclosed', 'pressure', 'radii'))
    cmb, T = kw['cmb_mass'], compute_adiabatic_temperature(**kw)
    n = int(np.count_nonzero(m < cmb * (1 - 1e-12)))
    core = kw['layer_mixtures']['core']
    rule = cmb_point_temperature(lambda P, t: get_mixed_nabla_ad(P, t, core, *_eos(kw)))
    _, pc, tc = (a[-1] for a in _core_nodes(m, p, T, cmb, cmb_point_rule=rule))
    inserted = dict(
        kw,
        radii=np.insert(r, n, 0.5 * (r[n - 1] + r[n])),
        pressure=np.insert(p, n, pc),
        mass_enclosed=np.insert(m, n, cmb),
    )
    exact = compute_adiabatic_temperature(**inserted)[n]
    assert abs(tc - exact) <= 1e-6, (tc, exact)
    assert _core_nodes(m, p, T, cmb)[2][-1] - exact > 3.0
