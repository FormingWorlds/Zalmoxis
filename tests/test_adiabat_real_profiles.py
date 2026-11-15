"""The adiabat integration on stored structure profiles of real solves, with the PALEOS tables.

The profiles (``tests/data/adiabat_profiles.npz``, written by
``python -m tools.validation.adiabat_profiles write``) are 4 round-off starts of
``input/default.toml`` and the 2 PROTEUS initial structure setups. See
``docs/Validation/thermal_state.md``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tests._plot_checks import assert_text_clear, legend_below
from tools.validation import adiabat_profiles as ap

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(60)]

DATA = Path(__file__).parent / 'data' / 'adiabat_profiles.npz'
P_FLOOR = 1e5  # Pa


def _need_paleos():
    from zalmoxis import get_zalmoxis_root

    root = Path(get_zalmoxis_root()) / 'data'
    for folder in ('EOS_PALEOS_iron', 'EOS_PALEOS_MgSiO3_unified'):
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
    keys = ('material_dictionaries', 'interpolation_functions', 'solidus_func', 'liquidus_func')
    keys += (
        'mushy_zone_factors',
        'condensed_rho_min',
        'condensed_rho_scale',
        'binodal_T_scale',
    )
    return tuple(kw[k] for k in keys)


def _forward_euler(kw):
    """The comparison scheme, one forward Euler step in P per node segment: nabla at the known
    node, no step where an end is below 1e5 Pa, the node caps of 1e5 K and 100 K."""
    from zalmoxis.mixing import get_mixed_nabla_ad
    from zalmoxis.structure_model import get_layer_mixture

    P, m = np.asarray(kw['pressure'], float), np.asarray(kw['mass_enclosed'], float)
    n, T, inward = len(P), np.empty(len(P)), kw['anchor'] != 'cmb'
    if inward:
        T[-1], order = kw['surface_temperature'], [(i, i + 1) for i in range(n - 2, -1, -1)]
    else:
        ic = _cmb_index(kw)
        T[:ic], T[ic] = kw['surface_temperature'], kw['cmb_temperature']
        order = [(i, i - 1) for i in range(ic + 1, n)]
    for i, j in order:
        mx = get_layer_mixture(
            m[i], kw['cmb_mass'], kw['core_mantle_mass'], kw['layer_mixtures']
        )
        T[i] = T[j]
        if not mx.has_tdep() or P[j] < P_FLOOR or P[i] < P_FLOOR:
            continue
        nb = get_mixed_nabla_ad(P[j], T[j], mx, *_eos(kw))
        if nb is not None and nb > 0 and (P[i] > P[j]) == inward:
            t = T[j] + nb * T[j] / P[j] * (P[i] - P[j])
            T[i] = min(t, 1e5) if inward else max(min(t, 1e5), 100.0)
    return T


def _reference(kw):
    """scipy ``solve_ivp`` (RK45, rtol 1e-10, atol 1e-12) of d ln T / d ln P = nabla over each
    node segment, with the rules of the routine: the mixture of the node the segment fills, T
    held below 1e5 Pa, at a padded top (P = 0) and for a T-independent layer, and the node caps.
    Returns (T, count of nabla reads that were None or not positive)."""
    from scipy.integrate import solve_ivp

    from zalmoxis.mixing import get_mixed_nabla_ad
    from zalmoxis.structure_model import get_layer_mixture

    P, m = np.asarray(kw['pressure'], float), np.asarray(kw['mass_enclosed'], float)
    eos = _eos(kw)
    n, T, bad = len(P), np.empty(len(P)), [0]
    inward = kw['anchor'] != 'cmb'
    if inward:
        T[-1], order, lo = (
            kw['surface_temperature'],
            [(i, i + 1) for i in range(n - 2, -1, -1)],
            0.0,
        )
    else:
        ic = _cmb_index(kw)
        T[:ic], T[ic] = kw['surface_temperature'], kw['cmb_temperature']
        order, lo = [(i, i - 1) for i in range(ic + 1, n)], 100.0
    for i, j in order:
        mx = get_layer_mixture(
            m[i], kw['cmb_mass'], kw['core_mantle_mass'], kw['layer_mixtures']
        )
        x0, x1 = math.log(max(P[j], P_FLOOR)), math.log(max(P[i], P_FLOOR))
        outer = P[j] if inward else P[i]
        if not mx.has_tdep() or outer <= 0 or (x1 - x0) * (1 if inward else -1) <= 0:
            T[i] = T[j]
            continue

        def rhs(x, y, mx=mx):
            v = get_mixed_nabla_ad(math.exp(x), math.exp(y[0]), mx, *eos)
            if v is None or not v > 0:
                bad[0] += 1
                return [0.0]
            return [v]

        y = solve_ivp(rhs, (x0, x1), [math.log(T[j])], method='RK45', rtol=1e-10, atol=1e-12)
        T[i] = min(max(math.exp(y.y[0, -1]), lo), 1e5)
    return T, bad[0]


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_midpoint_adiabat_meets_the_reference_on_real_profiles(profiles, validation_plot):
    """On the 6 stored profiles the default call stays within 1 K of an RK45 integration of the
    same ODE (rtol 1e-10) at every node on the adiabat, with no nabla read that is missing or
    not positive. The forward Euler step in P per node, computed in the test from the stored
    structure, equals the stored temperature of each solve and is at least 5 times farther and
    more than 2 K from the reference on every profile. A padded top gives the profile of a top
    at 1 Pa within 1e-6 K."""
    from zalmoxis.eos.temperature import compute_adiabatic_temperature

    kw0 = profiles['default_k0'][1]
    p_cmb = float(np.interp(kw0['cmb_mass'], kw0['mass_enclosed'], kw0['pressure']))
    for case, (rec, kw) in profiles.items():
        T = compute_adiabatic_temperature(**kw)
        ref, bad = _reference(kw)
        keep = _on_adiabat(kw)
        err = float(np.max(np.abs(T - ref)[keep]))
        euler = _forward_euler(kw)
        np.testing.assert_array_equal(euler, rec['T'])  # the stored call of the Euler scheme
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
        print(
            f'{case}: midpoint {err:.4f} K, forward Euler step {native:.2f} K from the reference'
        )

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
        ax.plot(gpa, ref[top], ls=':', color=plt.rcParams['text.color'], label='RK45 reference')
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
        native.append(_forward_euler({**kw, 'pressure': P}))
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
