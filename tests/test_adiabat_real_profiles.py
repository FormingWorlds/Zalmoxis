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

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]

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


def _reference(kw, method='DOP853'):
    """scipy ``solve_ivp`` (``method``, rtol 1e-13, atol 1e-15) of d ln T / d ln P = nabla over
    each node segment, with the rules of the routine: the mixture of the node the segment fills,
    T held below 1e5 Pa (a padded top, P = 0, counts as 1e5 Pa) and for a T-independent layer,
    and the node caps. Returns (T, count of nabla reads that were None or not positive)."""
    from scipy.integrate import solve_ivp

    from zalmoxis.mixing import get_mixed_nabla_ad
    from zalmoxis.structure_model import get_layer_mixture

    P, m, T, order, inward = ap.node_walk(kw)
    eos, bad, lo = _eos(kw), [0], (0.0 if inward else 100.0)
    for i, j in order:
        mx = get_layer_mixture(
            m[i], kw['cmb_mass'], kw['core_mantle_mass'], kw['layer_mixtures']
        )
        x0, x1 = math.log(max(P[j], P_FLOOR)), math.log(max(P[i], P_FLOOR))
        if not mx.has_tdep() or (x1 - x0) * (1 if inward else -1) <= 0:
            T[i] = T[j]
            continue

        def rhs(x, y, mx=mx):
            v = get_mixed_nabla_ad(math.exp(x), math.exp(y[0]), mx, *eos)
            if v is None or not v > 0:
                bad[0] += 1
                return [0.0]
            return [v]

        y = solve_ivp(rhs, (x0, x1), [math.log(T[j])], method=method, rtol=1e-13, atol=1e-15)
        T[i] = min(max(math.exp(y.y[0, -1]), lo), 1e5)
    return T, bad[0]


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_midpoint_adiabat_meets_the_reference_on_real_profiles(profiles, validation_plot):
    """On the 6 stored profiles the default call stays within 1 K of a DOP853 integration of the
    same ODE (rtol 1e-13), which agrees with RK45 at rtol 1e-13 to 0.01 K, at every node on the
    adiabat, with no nabla read that is missing or not positive. The forward Euler step in P per
    node, from ``forward_euler`` of the profile tool, equals the stored T_euler to a relative 1e-12
    and is at least 5 times farther and more than 2 K from the reference on every profile. A
    padded top gives the profile of a top at 1 Pa within 1e-6 K."""
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
    end_other: float  # the same with the mass of the other node of the cell
    mass_used: float  # mass [kg] the table gave the shell
    mass_rule: float  # mass [kg] of the node by the anchor rule
    layers_differ: bool  # the 2 nodes of the cell lie in different layers


def _top_cell(kw, p_top=None, layer_eos=None):
    """The T(P) lookup that the density solve builds on a stored profile, with its top node at
    p_top (None: as stored; 0: padded), over the top cell from max(P_top, 1e5 Pa) to the next
    node, and a DOP853 march (rtol 1e-12, atol 1e-14 on ln T) of d ln T / d ln P = nabla over
    the cell from the top T. The march takes the mixture of the node that the adiabat gives
    the cell (inner node inward, the node outside it outward from the CMB); in a layer without
    a T-dependent EOS it holds T, as the adiabat does. With ``layer_eos`` (an EOS name) a layer
    of it starts in the middle of the top cell. Returns a ``TopCell``."""
    from scipy.integrate import solve_ivp

    from zalmoxis.eos.temperature import compute_adiabatic_temperature as adiabat
    from zalmoxis.mixing import LayerMixture, get_mixed_nabla_ad
    from zalmoxis.solver import _TOP_SHELL_POINTS, _adiabat_table, _shell_adiabat
    from zalmoxis.structure_model import get_layer_mixture

    P, m = np.array(kw['pressure'], float), kw['mass_enclosed']
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

    lp, t = _adiabat_table(P, T, m, top_shell, kw['anchor'] == 'cmb')
    inner = int(np.flatnonzero(P > 0)[-1 if P[-1] <= 0 else -2])  # deeper node of the cell
    node, other = (inner + 1, inner) if kw['anchor'] == 'cmb' else (inner, inner + 1)
    q = np.geomspace(max(P[-1], P_FLOOR), P[inner], _TOP_SHELL_POINTS + 1)
    ends = [_shell_adiabat(shell, q, T[-1], m[k])[-1] - T[inner] for k in (node, other)]
    mx, mo = (
        get_layer_mixture(m[k], kw['cmb_mass'], kw['core_mantle_mass'], kw['layer_mixtures'])
        for k in (node, other)
    )
    cell = (*ends, used[0], m[node], mx.components != mo.components)
    x = np.linspace(math.log(max(P[-1], P_FLOOR)), math.log(P[inner]), 400)
    look = np.interp(x / math.log(10), lp, t)
    if not mx.has_tdep():
        return TopCell(x, look, np.full(x.size, T[-1]), *cell)

    def nab(x_, y):
        return [get_mixed_nabla_ad(math.exp(x_), math.exp(y[0]), mx, *_eos(kw))]

    ref = solve_ivp(
        nab, x[[0, -1]], [math.log(T[-1])], 'DOP853', t_eval=x, rtol=1e-12, atol=1e-14
    )
    return TopCell(x, look, np.exp(ref.y[0]), *cell)


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
    node, meets the T of that node within 1e-5 K (at most 2e-7 K measured)."""
    x, look, ref, end, _, got, want, _ = _top_cell(profiles[case][1], p_top)
    assert got == want  # the same array element: the shell takes the node of the rule
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
    cell outward, on both anchors: the table gives the shell the mass of the node whose mixture
    the adiabat takes over the cell (inner node inward, the node outside it outward), so the
    shell integrated on to the inner node meets its T within 1e-5 K, and within 1e-4 K for
    PALEOS H2O with the CMB anchor (1.25e-5 K measured: the midpoint step out and back over a
    cell where T changes by about 3200 K); with the mass of the other node it misses by more
    than 1 K. The lookup lies within 0.5 K of the march (0.36 K measured for PALEOS H2O with
    the CMB anchor, where the 64 shell points resolve the steep cell less well)."""
    if eos.startswith('PALEOS'):
        _need_paleos('EOS_PALEOS_H2O')
    kw = profiles[case][1]
    assert kw['anchor'] == ('cmb' if case == 'ic_dummy' else 'surface')
    cell = _top_cell(kw, p_top, eos)
    assert cell.layers_differ and cell.mass_used == cell.mass_rule
    steep = eos.startswith('PALEOS') and kw['anchor'] == 'cmb'
    assert abs(cell.end) < (1e-4 if steep else 1e-5), cell.end
    assert abs(cell.end_other) > 1.0, cell.end_other
    assert np.max(np.abs(cell.lookup - cell.march)) < 0.5


@pytest.mark.physics_invariant
def test_the_top_cell_lookup_does_not_flip_with_the_padding(profiles):
    """One structure (default start 0) with its top node at P = 0 (padded), 1 Pa, 1e5 Pa and
    1 atm: above 1e5 Pa the lookup is the padded one to round-off for a top at or below 1e5 Pa
    (the adiabat holds T below 1 bar, so the shell is the same), and within 0.01 K for 1 atm
    (1.6e-3 K measured); a solve that pads or not by round-off reads the same temperatures."""
    kw = profiles['default_k0'][1]
    x, padded, *_ = _top_cell(kw, 0.0)
    for p_top, bound in ((1.0, 1e-9), (P_FLOOR, 1e-9), (101325.0, 0.01)):
        y, look, *_ = _top_cell(kw, p_top)
        on = y >= math.log(P_FLOOR)
        diff = np.max(np.abs(look[on] - np.interp(y[on], x, padded)))
        assert diff <= bound, (p_top, diff)
