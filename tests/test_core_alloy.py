"""Tests for src/zalmoxis/eos/core_alloy.py and the core-alloy branch of the mixing rules.

Invariants and contract clauses: the light-element fractions of a core mixture
(one iron base, known elements, non-negative and finite, sum below 1, total O
never negative); the linear law, calibrated on the outer cores of Hirose et al. (2021) for S,
Si, O and H-linear and from Li et al. (2019) for C and N; the Luo et al. (2024) Eq. 14
members against an independent bisection; the Fe-H-O carrier with its O counted
against the core O, continuous where the O remainder changes sign; the bounds
that fail a composition without positive density; the limits of the core
mixture on the PALEOS iron table (no light element gives the table, the Fe-H-O
carrier at its own O gives the harmonic mean); the core adiabat staying that of
iron; and the config checks of a light-element core.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest

from zalmoxis.eos import core_alloy as ca
from zalmoxis.mixing import LayerMixture, parse_layer_components

hypothesis = pytest.importorskip('hypothesis')
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

GPA = 1e9
# 136 GPa, 8000 K: inside the Luo fit box, near the Earth core-mantle boundary pressure
P_CMB, T_LUO = 136 * GPA, 8000.0
# Hirose et al. 2021, PDF p. 3: single-element outer cores [wt%] at T_CMB 4137 K, T_ICB 5400 K
HIROSE_RANGE = {'S': (13.3, 14.4), 'Si': (9.3, 9.8), 'O': (8.4, 8.9), 'H-linear': (0.96, 0.99)}
HIROSE_MID = {el: 0.5 * (lo + hi) for el, (lo, hi) in HIROSE_RANGE.items()}
HIROSE_STATES = {'CMB': (136 * GPA, 4137.0), 'ICB': (330 * GPA, 5400.0)}

# Luo, Dorn & Deng 2024, Nat. Astron., Methods Eq. 14 (PDF p6), retyped from the paper
_EQ14 = {
    'dry': (49.249, 4.976, 1043.912, -15.957, 20.946, -3.811, 64 * 55.845),
    'wet': (57.162, 4.535, 1181.389, -8.478, 12.049, -1.258, 64 * 55.845 + 8 * 18.015),
}


def _eq14_density(p_gpa, t, member):
    """Density in kg/m^3 by bisection on x = V0/V of Eq. 14 (independent of the module)."""
    k, kp, v0, a, b, c, mass_amu = _EQ14[member]

    def pressure(x):
        cold = (
            1.5 * k * (x ** (7 / 3) - x ** (5 / 3)) * (1 + 0.75 * (kp - 4) * (x ** (2 / 3) - 1))
        )
        return cold + (t - 8000.0) / 1000.0 * (a + b * x + c * x * x)

    lo, hi = 0.7, 6.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if pressure(mid) < p_gpa else (lo, mid)
    volume_m3 = v0 / (0.5 * (lo + hi)) * 1e-30
    return mass_amu * 1.66053906660e-27 / volume_m3


def _mix(**w):
    """Core LayerMixture with the given light-element fractions on PALEOS:iron."""
    names = ['PALEOS:iron'] + [f'CoreAlloy:{k.replace("_", "-")}' for k in w]
    fracs = [1.0 - sum(w.values())] + list(w.values())
    return LayerMixture(names, fracs)


class TestSplitAlloy:
    """Composition checks of a light-element core."""

    def test_valid_mixture_returns_base_and_fractions(self):
        """The base and each element fraction come back unchanged, H-linear by its key."""
        base, w = ca.split_alloy(_mix(H_linear=0.004, Si=0.03, S=0.02))
        assert base == 'PALEOS:iron'
        assert w == {
            'H-linear': pytest.approx(0.004),
            'Si': pytest.approx(0.03),
            'S': pytest.approx(0.02),
        }
        assert ca.is_alloy_mixture(_mix(O=0.01))
        assert not ca.is_alloy_mixture(
            parse_layer_components('PALEOS:MgSiO3:0.9+PALEOS:H2O:0.1')
        )

    @pytest.mark.parametrize(
        'mixture, message',
        [
            (
                LayerMixture(['PALEOS:iron', 'CoreAlloy:Xe'], [0.99, 0.01]),
                'unknown core light element',
            ),
            (
                LayerMixture(['PALEOS:iron', 'CoreAlloy:H', 'CoreAlloy:H'], [0.98, 0.01, 0.01]),
                'given twice',
            ),
            (LayerMixture(['CoreAlloy:H', 'CoreAlloy:O'], [0.5, 0.5]), 'exactly one iron base'),
            (
                LayerMixture(
                    ['PALEOS:iron', 'Seager2007:iron', 'CoreAlloy:S'], [0.5, 0.49, 0.01]
                ),
                'exactly one iron base',
            ),
            (
                LayerMixture(
                    ['PALEOS:iron', 'CoreAlloy:H', 'CoreAlloy:H-linear'], [0.98, 0.01, 0.01]
                ),
                'exclude each other',
            ),
            (LayerMixture(['PALEOS:iron', 'CoreAlloy:O'], [1.01, -0.01]), 'mass fraction'),
        ],
        ids=[
            'unknown_element',
            'element_twice',
            'no_base',
            'two_bases',
            'both_h_forms',
            'negative_oxygen',
        ],
    )
    def test_invalid_composition_raises(self, mixture, message):
        """Each malformed core composition raises CoreAlloyError, a ValueError."""
        with pytest.raises(ca.CoreAlloyError) as exc:
            ca.split_alloy(mixture)
        assert message in str(exc.value)
        assert isinstance(exc.value, ValueError)

    def test_light_elements_that_leave_no_iron_raise(self):
        """A light-element sum of 1 leaves no iron; the bound is strict."""
        mixture = LayerMixture(['PALEOS:iron', 'CoreAlloy:S'], [0.5, 0.5])
        mixture.fractions = [0.0, 1.0]
        with pytest.raises(ca.CoreAlloyError, match='sum to'):
            ca.split_alloy(mixture)
        assert ca.split_alloy(_mix(N=0.999))[1]['N'] == pytest.approx(0.999)

    @pytest.mark.parametrize('bad', [math.nan, math.inf])
    def test_non_finite_fraction_raises(self, bad):
        """A NaN or infinite fraction (a TOML 'nan' or 'inf') is refused."""
        mixture = LayerMixture(['PALEOS:iron', 'CoreAlloy:C'], [0.5, 0.5])
        mixture.fractions = [0.99, bad]
        with pytest.raises(ca.CoreAlloyError, match='mass fraction'):
            ca.split_alloy(mixture)
        assert not math.isfinite(mixture.fractions[1])

    @settings(derandomize=True, max_examples=200, deadline=None)
    @given(
        st.dictionaries(
            st.sampled_from(sorted(ca.ALLOY_ELEMENTS - {'H-linear'})),
            st.floats(0.0, 0.03, allow_nan=False),
            min_size=1,
        )
    )
    def test_any_valid_fraction_set_round_trips(self, w):
        """Over the valid domain every fraction comes back and the iron base keeps the rest."""
        # the normalised O rest of the carrier stays above -0.2 up to w_H 0.0205
        w['H'] = min(w.get('H', 0.0), 0.0205)
        base, out = ca.split_alloy(_mix(**w))
        assert base == 'PALEOS:iron'
        assert set(out) == set(w)
        for el, frac in w.items():
            assert out[el] == pytest.approx(frac, abs=1e-15)


class TestLinearLaw:
    """Linear law: Hirose et al. (2021) calibration, Li et al. (2019) for C and N."""

    @pytest.mark.physics_invariant
    def test_no_light_element_returns_the_iron_density_exactly(self):
        """With every fraction 0 the core density is the iron density, bit for bit."""
        rho_fe = 11319.997247075
        assert ca.alloy_density(rho_fe, {}, P_CMB, 4000.0) == rho_fe
        assert ca.alloy_density(rho_fe, {'O': 0.0, 'Si': 0.0}, P_CMB, 4000.0) == rho_fe
        assert ca.alloy_density(math.nan, {'O': 0.01}, P_CMB, 4000.0) is None

    @pytest.mark.physics_invariant
    @pytest.mark.reference_pinned
    @pytest.mark.parametrize(
        'element, beta, wrong_beta',
        [('C', 1.3, 1.4)]
        + [(el, 7.0 / mid, 7.0 / mid * 1.01) for el, mid in HIROSE_MID.items()],
    )
    def test_one_wt_percent_lowers_the_density_by_beta_percent(self, element, beta, wrong_beta):
        """1 wt% lowers the density by beta %: Li et al. 2019 for C, else Hirose et al. 2021."""
        rho_fe = 1.0e4
        rho = ca.alloy_density(rho_fe, {element: 0.01}, P_CMB, 4000.0)
        assert rho == pytest.approx(rho_fe * (1 - beta / 100), rel=1e-12)
        # 1.4 for C or a 1 % slip in a calibrated beta lands outside
        assert abs(rho - rho_fe * (1 - wrong_beta / 100)) > 1e-12 * rho_fe
        assert rho < rho_fe

    def test_nitrogen_leaves_the_density_unchanged(self):
        """Table 3.3 has no N row, so N is carried at zero density effect."""
        rho = ca.alloy_density(1.0e4, {'N': 0.02, 'S': 0.01}, P_CMB, 4000.0)
        assert rho == pytest.approx(1.0e4 * (1 - 0.07 / 13.85), rel=1e-12)
        assert rho == pytest.approx(
            ca.alloy_density(1.0e4, {'S': 0.01}, P_CMB, 4000.0), rel=1e-15
        )

    def test_composition_checks_run_in_split_alloy(self):
        """The linear-law sign and the PALEOS:iron base fail the mixture itself, at any (P, T)."""
        with pytest.raises(ca.CoreAlloyError, match='linear-law core density'):
            ca.split_alloy(_mix(H_linear=0.2))
        bad_base = LayerMixture(['Seager2007:iron', 'CoreAlloy:S'], [0.99, 0.01])
        with pytest.raises(ca.CoreAlloyError, match='PALEOS:iron'):
            ca.split_alloy(bad_base)
        assert ca.split_alloy(_mix(H_linear=0.02))[0] == 'PALEOS:iron'

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize(
        'el, refused, admitted',
        [('O', 0.248, 0.246), ('C', 0.154, 0.153), ('S', 0.396, 0.395)]
        + [('Si', 0.273, 0.272), ('H-linear', 0.0279, 0.0278)],
    )
    def test_linear_factor_below_0p8_is_refused(self, el, refused, admitted):
        """1 - lin below 0.8 raises with the value and the bound; just above it passes."""
        key = el.replace('-', '_')
        with pytest.raises(ca.CoreAlloyError) as exc:
            ca.alloy_density(1.0e4, ca.split_alloy(_mix(**{key: refused}))[1], P_CMB, 4000.0)
        assert f'{1 - ca.LINEAR_BETA[el] * refused:.4f}' in str(exc.value)
        assert 'bound 0.8' in str(exc.value)
        rho = ca.alloy_density(1.0e4, ca.split_alloy(_mix(**{key: admitted}))[1], P_CMB, 4000.0)
        assert rho == pytest.approx(1.0e4 * (1 - ca.LINEAR_BETA[el] * admitted), rel=1e-12)

    @settings(derandomize=True, max_examples=200, deadline=None)
    @given(
        st.dictionaries(st.sampled_from(['C', 'O', 'S', 'Si']), st.floats(0.0, 0.3), min_size=1)
    )
    def test_linear_bound_over_the_composition_domain(self, w):
        """Without H, a core with iron passes exactly when sum(beta_i w_i) is at most 0.2."""
        lin = sum(ca.LINEAR_BETA[el] * f for el, f in w.items())
        if sum(w.values()) >= 1:  # no iron left, or a negative iron fraction
            with pytest.raises(ca.CoreAlloyError, match='sum to|mass fraction'):
                ca.split_alloy(_mix(**w))
        elif lin <= 0.2:
            assert ca.split_alloy(_mix(**w))[1] == pytest.approx(w)
        else:
            with pytest.raises(ca.CoreAlloyError, match='bound 0.8'):
                ca.split_alloy(_mix(**w))

    def test_mostly_oxygen_core_is_refused_at_validation(self):
        """PALEOS:iron:0.1+CoreAlloy:O:0.9 (rho/rho_Fe 0.01 by the linear law) is refused."""
        from zalmoxis.config import validate_layer_eos

        with pytest.raises(ca.CoreAlloyError, match='bound 0.8'):
            validate_layer_eos({'core': 'PALEOS:iron:0.1+CoreAlloy:O:0.9'})


class TestLuoMembers:
    """Luo, Dorn & Deng (2024), Methods Eq. 14: liquid Fe and Fe with water."""

    @pytest.mark.physics_invariant
    @pytest.mark.reference_pinned
    def test_eq14_density_matches_an_independent_bisection(self, validation_plot):
        """Eq. 14 (PDF p6) on 50-1300 GPa, 8000-14000 K: module root solve vs a bisection, rel 1e-6."""
        pressures = np.geomspace(50, 1300, 15)
        temps = (8000.0, 11000.0, 14000.0)
        curves = {}
        worst = 0.0
        for member, table in (('dry', ca.LUO_DRY), ('wet', ca.LUO_WET)):
            for t in temps:
                got = np.array([ca.luo_density(p * GPA, t, table) for p in pressures])
                ref = np.array([_eq14_density(p, t, member) for p in pressures])
                worst = max(worst, float(np.max(np.abs(got / ref - 1))))
                curves[(member, t)] = got
        assert worst < 1e-6
        # water lowers the density at every point; heating lowers it at fixed P
        assert np.all(curves[('wet', 8000.0)] < curves[('dry', 8000.0)])
        assert np.all(curves[('dry', 14000.0)] < curves[('dry', 8000.0)])
        # scale: liquid iron near 10.4 g/cm^3 at 136 GPa, 8000 K, not 1.04 or 104
        assert 1.03e4 < ca.luo_density(P_CMB, T_LUO, ca.LUO_DRY) < 1.05e4

        def figure():
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            for t, ls in zip(temps, ('-', '--', ':')):
                ax.plot(pressures, curves[('dry', t)] / 1e3, ls, color='C0')
                ax.plot(pressures, curves[('wet', t)] / 1e3, ls, color='C1')
                ax.plot([], [], ls, color='0.55', label=f'{t:.0f} K')
            ax.plot([], [], color='C0', label='Fe')
            ax.plot([], [], color='C1', label='Fe + water')
            ax.set(xscale='log', xlabel='Pressure [GPa]', ylabel='Density [g cm$^{-3}$]')
            ax.minorticks_off()
            ax.legend(fontsize='small')
            fig.tight_layout()
            return fig

        validation_plot(figure, 'core_alloy_luo2024_members')

    @pytest.mark.physics_invariant
    def test_reference_volume_has_zero_pressure_at_the_reference_temperature(self):
        """At V = V0 and T = T0 both members have P = 0; heating at V0 adds the thermal term."""
        for table in (ca.LUO_DRY, ca.LUO_WET):
            assert ca.luo_pressure(table['V0'], ca.LUO_T0, table) == pytest.approx(
                0.0, abs=1e-6
            )
            hot = ca.luo_pressure(table['V0'], ca.LUO_T0 + 1000.0, table)
            assert hot == pytest.approx((table['a'] + table['b'] + table['c']) * GPA, rel=1e-12)

    def test_no_root_in_the_bracket_returns_none(self, monkeypatch):
        """A pressure above the bracket has no volume; a member without one gives no density."""
        assert ca.luo_cell_volume(1e15, T_LUO, ca.LUO_DRY) is None
        assert ca.luo_density(1e15, T_LUO, ca.LUO_WET) is None
        monkeypatch.setattr(ca, 'luo_cell_volume', lambda p, t, member: None)
        assert ca.carrier_excess_volume(P_CMB, T_LUO) is None
        assert ca.alloy_density(1.0e4, {'H': 0.005}, P_CMB, T_LUO) is None

    def test_every_point_of_the_fit_box_has_both_volumes(self):
        """The bracket holds a root for both members on a grid over the whole Luo box."""
        for p in np.geomspace(*ca.LUO_P_RANGE, 9):
            for t in np.linspace(*ca.LUO_T_RANGE, 7):
                assert ca.luo_cell_volume(p, t, ca.LUO_DRY) is not None
                assert ca.luo_cell_volume(p, t, ca.LUO_WET) is not None


class TestCarrier:
    """The Fe-H-O carrier with its O counted against the core O."""

    @pytest.mark.physics_invariant
    def test_remaining_alloy_fractions_are_normalised(self):
        """With the carrier, the linear law acts on w_i / (1 - w_c) of the remaining alloy.

        1 wt% H, 3 wt% Si, no core O at 136 GPa, 8000 K: the linear sum without the
        1 / (1 - w_c) factor gives a density 0.45 % off, far outside the 1e-12 pin.
        """
        rho_fe, w_h, w_si = 1.0e4, 0.01, 0.03
        w_c = w_h * ca.CARRIER_PER_H
        o_rest = 0.0 - (w_c - w_h)
        dv_c = ca.carrier_excess_volume(P_CMB, T_LUO)

        def density(lin):
            return 1.0 / ((1 - w_c) / (rho_fe * (1 - lin)) + w_c * dv_c)

        right = density((7.0 / 9.55 * w_si + 7.0 / 8.65 * o_rest) / (1 - w_c))
        got = ca.alloy_density(rho_fe, {'H': w_h, 'Si': w_si}, P_CMB, T_LUO)
        assert got == pytest.approx(right, rel=1e-12)
        assert abs(density(7.0 / 9.55 * w_si + 7.0 / 8.65 * o_rest) / right - 1) > 1e-3

    def test_carrier_that_leaves_no_alloy_mass_is_refused(self):
        """A carrier mass fraction 8.94 w_H of 1 or more is no core: CoreAlloyError, also at 1."""
        w_full = 1.0 / ca.CARRIER_PER_H
        for w_h in (w_full, math.nextafter(w_full, 1.0), 0.15):
            try:
                ca.alloy_density(1.1e4, {'H': w_h}, P_CMB, T_LUO)
                raised = None
            except Exception as err:
                raised = err
            assert isinstance(raised, ca.CoreAlloyError), (w_h, raised)
            assert 'carrier' in str(raised)
            with pytest.raises(ca.CoreAlloyError, match='carrier'):
                ca.split_alloy(_mix(H=w_h))
        # 0.43 wt% H, the carrier fraction of the Luo wet cell, is accepted
        assert ca.split_alloy(_mix(H=0.0043))[1] == {'H': pytest.approx(0.0043)}

    @pytest.mark.parametrize(
        ('w_h', 'w_o', 'admitted'),
        [(0.02, 0.0, True), (0.0205, 0.0, True), (0.0207, 0.0, False), (0.05, 0.0252, False)],
    )
    def test_normalised_o_rest_bound(self, w_h, w_o, admitted):
        """The first-order carrier split holds to an O rest of -0.2 of the remaining alloy."""
        w_c = w_h * ca.CARRIER_PER_H
        rest = (w_o - (w_c - w_h)) / (1 - w_c)
        assert (rest >= -0.2) is admitted
        try:
            ca.split_alloy(_mix(H=w_h, O=w_o))
            raised = None
        except Exception as err:
            raised = err
        if admitted:
            assert raised is None, raised
        else:
            assert isinstance(raised, ca.CoreAlloyError), raised
            assert '-0.2' in str(raised) and f'{rest:.3f}' in str(raised)

    def test_carrier_above_the_luo_cell_fraction_warns_once(self, caplog, monkeypatch):
        """A carrier fraction above that of the Luo wet cell (0.0388) warns once per process."""
        monkeypatch.setattr(ca, '_CELL_WARNED', [False])
        assert ca.LUO_CARRIER_FRACTION == pytest.approx(0.0388, abs=5e-5)
        w_cell = ca.LUO_CARRIER_FRACTION / ca.CARRIER_PER_H
        with caplog.at_level('WARNING', logger='fwl.zalmoxis.eos.core_alloy'):
            ca.split_alloy(_mix(H=0.999 * w_cell))
            assert caplog.messages == []
            ca.split_alloy(_mix(H=1.001 * w_cell))
            assert len(caplog.messages) == 1
            ca.split_alloy(_mix(H=0.01))
        assert len([m for m in caplog.messages if 'Luo' in m and 'cell' in m]) == 1

    def test_member_cell_mass_travels_with_the_member(self):
        """A copy of a Luo member gives the same density as the member itself."""
        for member in (ca.LUO_DRY, ca.LUO_WET):
            same = ca.luo_density(P_CMB, T_LUO, dict(member))
            assert same == ca.luo_density(P_CMB, T_LUO, member)
        # the wet cell carries 8 H2O more than the dry cell: about 4 % more mass per cell
        wet, dry = ca.LUO_WET['cell_mass'], ca.LUO_DRY['cell_mass']
        assert wet / dry - 1 == pytest.approx(8 * 18.015 / (64 * 55.845), rel=1e-3)

    @pytest.mark.physics_invariant
    def test_carrier_mixture_without_positive_density_raises(self, monkeypatch):
        """A carrier excess volume that drives 1/rho to or below zero fails the composition.

        The Luo members give a positive excess volume; the stand-in values reach the guard.
        """
        w_c = 0.01 * ca.CARRIER_PER_H
        # core O equal to the carrier O: no O remainder, so the linear base is rho_fe
        w, rho_fe = {'H': 0.01, 'O': w_c - 0.01}, 1.0e4
        dv_zero = -(1.0 - w_c) / (rho_fe * w_c)  # 1/rho = 0 at this excess volume
        for dv in (10 * dv_zero, dv_zero * (1 + 1e-9)):
            monkeypatch.setattr(ca, 'carrier_excess_volume', lambda p, t, dv=dv: dv)
            try:
                ca.alloy_density(rho_fe, w, P_CMB, 4000.0)
                raised = None
            except Exception as err:
                raised = err
            assert isinstance(raised, ca.CoreAlloyError), raised
            assert 'core alloy density' in str(raised)
        monkeypatch.setattr(ca, 'carrier_excess_volume', lambda p, t: 0.5 * dv_zero)
        rho = ca.alloy_density(rho_fe, w, P_CMB, 4000.0)
        assert rho == pytest.approx(2 * rho_fe / (1 - w_c), rel=1e-12)

    @pytest.mark.physics_invariant
    @pytest.mark.reference_pinned
    def test_effective_beta_h_at_136_gpa_8000_k(self):
        """With no core O the carrier lowers the density by about 15.1 % per wt% H."""
        rho_fe = ca.luo_density(P_CMB, T_LUO, ca.LUO_DRY)
        h = 1e-6
        beta_h = (1 - ca.alloy_density(rho_fe, {'H': h}, P_CMB, T_LUO) / rho_fe) / h
        assert beta_h == pytest.approx(15.14, abs=0.01)
        # the linear value 7.18 and the carrier without the O correction (21.6) lie outside
        assert abs(beta_h - ca.LINEAR_BETA['H-linear']) > 1.0
        k = ca.CARRIER_PER_H
        no_o_correction = k * ca.carrier_excess_volume(P_CMB, T_LUO) * rho_fe - k
        assert abs(beta_h - no_o_correction) > 1.0

    @pytest.mark.physics_invariant
    @pytest.mark.reference_pinned
    def test_carrier_at_its_own_oxygen_is_the_harmonic_mean(self):
        """O = 7.94 H: Fe and the carrier mix by volume; a mass-weighted mean differs."""
        rho_fe = 1.12e4
        w_h = 0.005
        w_c = w_h * ca.CARRIER_PER_H
        rho_c = 1.0 / ca.carrier_excess_volume(P_CMB, T_LUO)
        rho = ca.alloy_density(rho_fe, {'H': w_h, 'O': w_c - w_h}, P_CMB, T_LUO)
        assert rho == pytest.approx(1.0 / ((1 - w_c) / rho_fe + w_c / rho_c), rel=1e-12)
        assert abs(rho - ((1 - w_c) * rho_fe + w_c * rho_c)) > 0.01 * rho
        assert rho_c < rho < rho_fe

    @pytest.mark.physics_invariant
    def test_density_is_continuous_where_the_oxygen_remainder_changes_sign(self):
        """Across O = 7.94 H the density changes by the linear O slope, with no step."""
        rho_fe, w_h = 1.12e4, 0.004
        w_oc = w_h * (ca.CARRIER_PER_H - 1)
        eps = 1e-7
        below = ca.alloy_density(rho_fe, {'H': w_h, 'O': w_oc - eps}, P_CMB, T_LUO)
        at = ca.alloy_density(rho_fe, {'H': w_h, 'O': w_oc}, P_CMB, T_LUO)
        above = ca.alloy_density(rho_fe, {'H': w_h, 'O': w_oc + eps}, P_CMB, T_LUO)
        slope_below, slope_above = (at - below) / eps, (above - at) / eps
        assert slope_below == pytest.approx(slope_above, rel=1e-3)
        # the step the alternative rule (linear H below the carrier O) would make
        linear_h = ca.alloy_density(rho_fe, {'H-linear': w_h, 'O': w_oc - eps}, P_CMB, T_LUO)
        assert abs(linear_h - at) > 1e4 * abs(below - at)

    def test_out_of_range_point_is_clamped_counted_and_warned_once(self, caplog, monkeypatch):
        """Outside 8000-14000 K, 50-1300 GPa the carrier volume is the one at the nearest box point."""
        monkeypatch.setitem(ca.LUO_OUT_OF_RANGE, 'count', 0)
        dv = ca.carrier_excess_volume
        with caplog.at_level('WARNING', logger='fwl.zalmoxis.eos.core_alloy'):
            inside = dv(P_CMB, T_LUO)
            assert dv(P_CMB, 4000.0) == inside
            assert dv(5 * GPA, 3000.0) == dv(50 * GPA, 8000.0)
            assert dv(2000 * GPA, 20000.0) == dv(1300 * GPA, 14000.0)
        assert ca.LUO_OUT_OF_RANGE['count'] == 3
        assert sum('outside the Luo' in m for m in caplog.messages) == 1
        # the extrapolated Eq. 14 value at 4000 K differs: the clamp is not a no-op
        v = {
            m: ca.luo_cell_volume(P_CMB, 4000.0, ca.LUO_WET if m else ca.LUO_DRY)
            for m in (0, 1)
        }
        extrapolated = (v[1] - v[0]) * 1e-30 / (8 * 18.015 * 1.66053906660e-27)
        assert abs(extrapolated / inside - 1) > 0.05

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize(
        'edge, step',
        [
            ((50 * GPA, 9000.0), (1e-6 * GPA, 0.0)),
            ((1300 * GPA, 9000.0), (1e-6 * GPA, 0.0)),
            ((P_CMB, 8000.0), (0.0, 1e-6)),
            ((P_CMB, 14000.0), (0.0, 1e-6)),
        ],
        ids=['p_low', 'p_high', 't_low', 't_high'],
    )
    def test_carrier_volume_is_continuous_across_each_box_edge(self, edge, step, monkeypatch):
        """Across each face of the Luo box the clamped carrier volume has no step."""
        monkeypatch.setitem(ca.LUO_OUT_OF_RANGE, 'count', 0)
        (p, t), (dp, dt) = edge, step
        below = ca.carrier_excess_volume(p - dp, t - dt)
        above = ca.carrier_excess_volume(p + dp, t + dt)
        assert below == pytest.approx(above, rel=1e-8)
        assert 1e-4 < below < 6e-4

    def test_low_pressure_cool_core_point_has_a_density(self, monkeypatch):
        """At 5 GPa, 3000 K (no Eq. 14 root for dry Fe) the 1 wt% H core still has a density."""
        monkeypatch.setitem(ca.LUO_OUT_OF_RANGE, 'count', 0)
        assert ca.luo_cell_volume(5 * GPA, 3000.0, ca.LUO_DRY) is None
        rho = ca.alloy_density(7.0e3, {'H': 0.01}, 5 * GPA, 3000.0)
        assert 0.8 * 7.0e3 < rho < 7.0e3

    @settings(derandomize=True, max_examples=150, deadline=None)
    @given(
        st.floats(0.0, 0.01),
        st.floats(0.0, 0.08),
        st.sampled_from(['C', 'Si', 'S', 'O', 'H']),
    )
    def test_more_light_element_never_raises_the_density(self, w_h, w_o, extra):
        """At 136 GPa, 8000 K adding any light element lowers or keeps the density."""
        rho_fe = 1.12e4
        w = {'H': w_h, 'O': w_o}
        base = ca.alloy_density(rho_fe, w, P_CMB, T_LUO)
        more = dict(w)
        more[extra] = more.get(extra, 0.0) + 0.002
        assert 0 < ca.alloy_density(rho_fe, more, P_CMB, T_LUO) < base * (1 + 1e-12)
        assert base <= rho_fe * (1 + 1e-12)


def _paleos_iron():
    from zalmoxis.config import load_material_dictionaries

    md = load_material_dictionaries()
    if not os.path.isfile(md['PALEOS:iron'].get('eos_file', '')):
        pytest.skip('PALEOS iron table not linked under data/')
    return md


class TestHiroseCalibration:
    """The linear law against the single-element outer cores of Hirose et al. (2021)."""

    LI_2019 = {'S': 0.8, 'Si': 0.7, 'O': 1.1, 'H-linear': 8.7}
    # Luo-law deficit [%] of 0.96 and 0.99 wt% H with no core O, on 2 iron bases (docs table)
    LUO_H_DEFICIT = {
        ('CMB', 'paleos'): (14.904, 15.308),
        ('ICB', 'paleos'): (11.602, 11.931),
        ('CMB', 'luo_liquid_8000K'): (12.986, 13.347),
        ('ICB', 'luo_liquid_8000K'): (10.822, 11.133),
    }

    @staticmethod
    def _deficit(md, cache, w, p, t):
        """Percent density deficit of the core mixture against the PALEOS iron table."""
        from zalmoxis.mixing import calculate_mixed_density as rho

        iron = rho(p, t, _mix(), md, None, None, cache)
        return 100 * (1 - rho(p, t, _mix(**w), md, None, None, cache) / iron)

    @pytest.mark.physics_invariant
    @pytest.mark.reference_pinned
    def test_the_calibration_gives_7_percent_on_any_iron_base(self):
        """Calibration check: beta = 7 / midpoint gives 7 % at each Hirose midpoint (1e-12
        percentage points) on any iron density, 6-8 % at the range ends; the Li et al. (2019)
        values miss the band for S, O and H-linear."""
        for el, (lo, hi) in HIROSE_RANGE.items():
            for rho_fe in (1.0e4, 1.2345e4):
                rho = ca.alloy_density(rho_fe, {el: HIROSE_MID[el] / 100}, P_CMB, 4000.0)
                assert abs(100 * (1 - rho / rho_fe) - 7.0) < 1e-12
            ends = [
                100 * (1 - ca.alloy_density(1.0e4, {el: x / 100}, P_CMB, 4000.0) / 1.0e4)
                for x in (lo, hi)
            ]
            assert 6.0 < ends[0] < 7.0 < ends[1] < 8.0
            if el != 'Si':
                assert self.LI_2019[el] * lo > 8.0

    @pytest.mark.physics_invariant
    @pytest.mark.reference_pinned
    def test_each_single_element_outer_core_is_7_percent_lighter_than_iron(
        self, validation_plot, monkeypatch
    ):
        """Hirose et al. 2021, p. 3: on the PALEOS iron table, solid at both states, each
        midpoint core is 7 % lighter (1e-12 percentage points) and the range ends 6-8 %; the
        default H law gives more than 8 % for their H core (regression pins, table version
        dependent).
        """
        from zalmoxis.eos import calculate_density
        from zalmoxis.phase_columns import phase_for_component

        monkeypatch.setitem(ca.LUO_OUT_OF_RANGE, 'count', 0)
        monkeypatch.setattr(ca, '_CELL_WARNED', [False])
        md, cache = _paleos_iron(), {}
        for (p, t), ratio in zip(HIROSE_STATES.values(), (1.0227, 1.0124)):
            assert phase_for_component('PALEOS:iron', p, t).startswith('solid')
            solid = calculate_density(p, md, 'PALEOS:iron', t, None, None, cache, 1.0)
            assert solid / ca.luo_density(p, t, ca.LUO_DRY) == pytest.approx(ratio, abs=1e-4)
        bars = {}
        for el, (lo, hi) in HIROSE_RANGE.items():
            for state, (p, t) in HIROSE_STATES.items():
                mid = self._deficit(md, cache, {el: HIROSE_MID[el] / 100}, p, t)
                assert abs(mid - 7.0) < 1e-12
                ends = [self._deficit(md, cache, {el: x / 100}, p, t) for x in (lo, hi)]
                assert 6.0 < ends[0] < 7.0 < ends[1] < 8.0
            bars[el] = (ends, [self.LI_2019[el] * x for x in (lo, hi)])
        assert bars['S'][1][0] > 8.0 and bars['O'][1][0] > 8.0
        for state, (p, t) in HIROSE_STATES.items():
            luo = [
                self._deficit(md, cache, {'H': x / 100}, p, t) for x in HIROSE_RANGE['H-linear']
            ]
            np.testing.assert_allclose(luo, self.LUO_H_DEFICIT[(state, 'paleos')], atol=1e-3)
            assert min(luo) > 8.0
            bars[f'H\n{state}'] = (luo, None)

        def figure():
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(6.4, 3.4))
            ax.axhspan(6.0, 8.0, alpha=0.15, lw=0, label='Hirose 2021')
            colour = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
            for i, (name, (new, old)) in enumerate(bars.items()):
                ax.plot(
                    [i + 0.1] * 2,
                    new,
                    '-o',
                    color=colour,
                    ms=4,
                    label='calibrated' if i == 0 else None,
                )
                if old is not None:
                    ax.plot(
                        [i - 0.1] * 2,
                        old,
                        '--o',
                        color=colour,
                        mfc='none',
                        ms=4,
                        alpha=0.6,
                        label='Li 2019' if i == 0 else None,
                    )
            ax.set_xticks(range(len(bars)), [n.replace('H-linear', 'H\nlinear') for n in bars])
            ax.set(ylabel='Density deficit [%]', xlim=(-0.5, len(bars) - 0.5))
            ax.grid(False)
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
            fig.tight_layout()
            fig.canvas.draw()
            box = ax.get_legend().get_window_extent()
            assert box.x0 >= ax.get_window_extent().x1
            ticks = [lab.get_window_extent() for lab in ax.get_xticklabels()]
            assert all(a.x1 + 5 < b.x0 for a, b in zip(ticks, ticks[1:])), [
                (b.x0, b.x1) for b in ticks
            ]
            return fig

        validation_plot(figure, 'core_alloy_hirose2021_deficit')

    @pytest.mark.physics_invariant
    @pytest.mark.reference_pinned
    def test_luo_h_law_deficit_on_a_liquid_iron_base(self, monkeypatch):
        """The default H law on the Luo liquid iron at 8000 K, where the carrier volume is taken
        (regression pins)."""
        monkeypatch.setattr(ca, '_CELL_WARNED', [False])
        for state, (p, _) in HIROSE_STATES.items():
            rho_fe = ca.luo_density(p, T_LUO, ca.LUO_DRY)
            got = [
                100 * (1 - ca.alloy_density(rho_fe, {'H': x / 100}, p, T_LUO) / rho_fe)
                for x in HIROSE_RANGE['H-linear']
            ]
            np.testing.assert_allclose(
                got, self.LUO_H_DEFICIT[(state, 'luo_liquid_8000K')], atol=1e-3
            )
            assert min(got) > 8.0


class TestCompoundBounds:
    """A composition past the end compound of its binary warns once and is not refused."""

    @pytest.mark.parametrize(
        ('w', 'p_gpa', 'warned'),
        [({'S': 0.1606}, 136, False), ({'S': 0.1607}, 136, True), ({'O': 0.2226}, 136, False)]
        + [({'O': 0.2228}, 136, True), ({'C': 0.0668}, 100, False), ({'C': 0.0670}, 100, True)]
        + [({'C': 0.0843}, 300, False), ({'C': 0.0845}, 300, True), ({'Si': 0.25}, 136, False)]
        + [
            ({'C': 0.075}, 254.9, True),
            ({'C': 0.075}, 255.0, False),
            ({'S': 0.10, 'C': 0.05}, 136, True),
        ]
        + [({'S': 0.08, 'O': 0.10}, 136, False), ({'S': 0.10, 'O': 0.10}, 136, True)]
        + [({'S': 0.155}, 136, False), ({'S': 0.155, 'Si': 0.05}, 136, True)],
    )
    def test_a_core_past_the_end_compound_of_its_binary_warns(
        self, w, p_gpa, warned, caplog, monkeypatch
    ):
        """The S, O and C units of the alloy liquidus (Fe3S, FeO, Fe3C below 255 GPa and Fe7C3
        at or above) together need more Fe than the core holds: 16.06 wt% S, 22.27 wt% O, 6.69
        or 8.44 wt% C alone; Si takes no Fe. The pressure is a NumPy scalar, as in a structure
        solve."""
        monkeypatch.setattr(ca, '_COMPOUND_WARNED', set())
        with caplog.at_level('WARNING', logger='fwl.zalmoxis.eos.core_alloy'):
            for _ in range(2):
                assert ca.alloy_density(1.0e4, w, np.float64(p_gpa * GPA), 4000.0) > 0.0
        assert sum('more Fe than the core holds' in m for m in caplog.messages) == int(warned)

    def test_the_compound_warning_comes_once_per_process(self, caplog, monkeypatch):
        """A second composition past the bound, also with another element, warns no more."""
        monkeypatch.setattr(ca, '_COMPOUND_WARNED', set())
        with caplog.at_level('WARNING', logger='fwl.zalmoxis.eos.core_alloy'):
            ca.alloy_density(1.0e4, {'S': 0.17}, np.float64(136 * GPA), 4000.0)
            ca.alloy_density(1.0e4, {'O': 0.23}, np.float64(136 * GPA), 4000.0)
        assert sum('more Fe than the core holds' in m for m in caplog.messages) == 1


class TestPaleosCore:
    """The core-alloy branch of the mixing rules on the PALEOS iron table."""

    @pytest.mark.physics_invariant
    @pytest.mark.parametrize(
        ('p_gpa', 'temps'),
        [(1e-4, (3000.0, 8000.0, 16000.0)), (10.0, (3000.0, 8000.0, 16000.0))]
        + [
            (p, (3000.0, 8000.0, 14000.0, 16000.0, 35000.0, 59000.0))
            for p in (20.0, 136.0, 360.0)
        ]
        + [(p, (3000.0, 8000.0, 14000.0, 59000.0)) for p in (1300.0, 3000.0)],
    )
    def test_the_density_falls_with_h_where_the_bound_admits_the_core(self, p_gpa, temps):
        """Up to 16000 K at every pressure and up to 59000 K above 20 GPa, the core density
        falls with w_H for every admitted core, down to the linear-factor bound 0.8 set by
        Si."""
        from zalmoxis.eos import calculate_density

        md, cache = _paleos_iron(), {}
        w_h = np.linspace(0.0, 0.02, 41)
        for t in temps:
            rho_fe = calculate_density(
                p_gpa * GPA, md, 'PALEOS:iron', t, None, None, cache, 1.0
            )
            for w_si in (0.0, 0.1, 0.2, (0.2 - 1e-4) / ca.LINEAR_BETA['Si']):
                rho = []
                for h in w_h:
                    try:
                        rho.append(
                            ca.alloy_density(rho_fe, {'H': h, 'Si': w_si}, p_gpa * GPA, t)
                        )
                    except ca.CoreAlloyError:
                        break
                assert len(rho) > 5, (p_gpa, t, w_si)
                assert np.all(np.diff(rho) < 0.0), (p_gpa, t, w_si)

    @pytest.mark.physics_invariant
    def test_without_the_bound_the_density_can_rise_with_h(self, monkeypatch):
        """The scan above can fail: with the linear-factor bound removed, a core with Si far
        below the bound (factor 0.56, below r* = 0.595 at 1300 GPa and 8000 K) gets denser with
        H."""
        from zalmoxis.eos import calculate_density

        monkeypatch.setattr(ca, 'LIN_FACTOR_MIN', 0.0)
        md, cache = _paleos_iron(), {}
        rho_fe = calculate_density(
            1300 * GPA, md, 'PALEOS:iron', 8000.0, None, None, cache, 1.0
        )
        rho = [
            ca.alloy_density(rho_fe, {'H': h, 'Si': 0.6}, 1300 * GPA, 8000.0)
            for h in (0.0, 0.002)
        ]
        assert rho[1] > rho[0]

    @pytest.mark.physics_invariant
    @pytest.mark.reference_pinned
    def test_core_mixture_limits_on_paleos_iron(self, validation_plot):
        """No light element gives the PALEOS iron density; the carrier mixes harmonically."""
        from zalmoxis.mixing import calculate_mixed_density, calculate_mixed_density_batch

        md, cache = _paleos_iron(), {}
        pressures = np.geomspace(140, 360, 12) * GPA
        temps = np.full_like(pressures, T_LUO)
        iron = calculate_mixed_density_batch(
            pressures, temps, parse_layer_components('PALEOS:iron'), md, None, None, cache
        )
        zero = _mix(H=0.0, O=0.0)
        same = calculate_mixed_density_batch(pressures, temps, zero, md, None, None, cache)
        np.testing.assert_array_equal(same, iron)
        curves = {0.0: iron}
        for w_h in (0.005, 0.01):
            w_c = w_h * ca.CARRIER_PER_H
            mix = _mix(H=w_h, O=w_c - w_h)
            got = calculate_mixed_density_batch(pressures, temps, mix, md, None, None, cache)
            rho_c = np.array([1.0 / ca.carrier_excess_volume(p, T_LUO) for p in pressures])
            harmonic = 1.0 / ((1 - w_c) / iron + w_c / rho_c)
            np.testing.assert_allclose(got, harmonic, rtol=1e-12)
            assert np.all(np.abs(got - ((1 - w_c) * iron + w_c * rho_c)) > 0.01 * got)
            # the scalar branch (structure RHS) and the batch branch (node update) agree
            scalar = []
            for p in pressures:
                try:
                    scalar.append(calculate_mixed_density(p, T_LUO, mix, md, None, None, cache))
                except ValueError as err:
                    pytest.fail(f'scalar core-alloy density raised at {p:.3e} Pa: {err}')
            np.testing.assert_allclose(scalar, got, rtol=1e-9)
            curves[w_h] = got

        def figure():
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            for (w_h, rho), ls in zip(curves.items(), ('-', '--', ':')):
                ax.plot(pressures / GPA, rho / 1e3, ls, label=f'{100 * w_h:.1f} wt% H')
            ax.set(xlabel='Pressure [GPa]', ylabel='Core density [g cm$^{-3}$]')
            ax.legend(title='8000 K, O of the carrier')
            fig.tight_layout()
            return fig

        validation_plot(figure, 'core_alloy_paleos_iron_limits')

    def test_alloy_branch_uses_the_mushy_zone_factor_of_the_base(self):
        """At 136 GPa, 4000 K a mushy-zone factor 0.8 on the base moves the alloy with the iron."""
        from zalmoxis.eos import calculate_density
        from zalmoxis.mixing import calculate_mixed_density, calculate_mixed_density_batch

        md, cache, mix = _paleos_iron(), {}, _mix(H=0.005, O=0.005 * ca.CARRIER_PER_H - 0.005)
        got = {}
        for mzf in (0.8, 1.0):
            iron = calculate_density(P_CMB, md, 'PALEOS:iron', 4000.0, None, None, cache, mzf)
            ref = ca.alloy_density(iron, ca.split_alloy(mix)[1], P_CMB, 4000.0)
            factors = {'PALEOS:iron': mzf}
            scalar = calculate_mixed_density(P_CMB, 4000.0, mix, md, None, None, cache, factors)
            batch = calculate_mixed_density_batch(
                np.array([P_CMB]), np.array([4000.0]), mix, md, None, None, cache, factors
            )
            assert scalar == pytest.approx(ref, rel=1e-9)
            assert batch[0] == pytest.approx(ref, rel=1e-9)
            got[mzf] = scalar
        assert abs(got[0.8] / got[1.0] - 1) > 0.01

    def test_batch_gives_nan_where_a_point_has_no_density(self, monkeypatch):
        """alloy_density_batch: NaN at a non-finite iron density and where dv_c is None."""
        rho_fe = np.array([1.1e4, np.nan, 1.1e4, 1.1e4])
        p = np.full(4, P_CMB)
        t = np.array([T_LUO, T_LUO, 9000.0, T_LUO])
        real = ca.carrier_excess_volume
        monkeypatch.setattr(
            ca, 'carrier_excess_volume', lambda pp, tt: None if tt == 9000.0 else real(pp, tt)
        )
        out = ca.alloy_density_batch(rho_fe, {'H': 0.005}, p, t)
        assert np.isnan(out[1]) and np.isnan(out[2])
        assert out[0] == pytest.approx(ca.alloy_density(1.1e4, {'H': 0.005}, P_CMB, T_LUO))
        assert out[3] == out[0]

    @pytest.mark.physics_invariant
    @pytest.mark.reference_pinned
    def test_luo_dry_member_equals_paleos_liquid_iron(self):
        """Cross-code: Eq. 14 dry Fe here equals the PALEOS liquid iron table to 1e-5.

        At the PALEOS solid points of the grid the two differ by more than 1e-3, and the wet
        member misses the liquid points by more than 3 %.
        """
        from zalmoxis.eos import calculate_density
        from zalmoxis.phase_columns import phase_for_component

        md, cache, worst, solid = _paleos_iron(), {}, 0.0, []
        for p in (60, 136, 360, 700, 1300):
            for t in (8000.0, 11000.0, 14000.0):
                pal = calculate_density(p * GPA, md, 'PALEOS:iron', t, None, None, cache, 1.0)
                rel = abs(pal / ca.luo_density(p * GPA, t, ca.LUO_DRY) - 1)
                if phase_for_component('PALEOS:iron', p * GPA, t) == 'liquid':
                    worst = max(worst, rel)
                    wet = ca.luo_density(p * GPA, t, ca.LUO_WET)
                    assert abs(pal / wet - 1) > 0.03
                else:
                    solid.append(rel)
        assert worst < 1e-5
        assert solid and min(solid) > 1e-3

    @pytest.mark.physics_invariant
    def test_core_adiabat_is_that_of_iron(self):
        """The light elements leave nabla_ad at the iron value; a non-alloy mixture is unchanged."""
        from zalmoxis.mixing import get_mixed_nabla_ad

        md, cache = _paleos_iron(), {}
        iron = get_mixed_nabla_ad(
            P_CMB, 4000.0, parse_layer_components('PALEOS:iron'), md, cache, None, None
        )
        alloy = get_mixed_nabla_ad(P_CMB, 4000.0, _mix(H=0.005, Si=0.03), md, cache, None, None)
        assert alloy == iron
        assert 0.0 < iron < 1.0


class TestConfig:
    """Config checks of a light-element core."""

    def test_validate_config_gives_no_eos_warning_for_an_alloy_core(self, caplog):
        """CoreAlloy components are neither T-independent nor non-iron for the config warnings."""
        from zalmoxis import get_zalmoxis_root
        from zalmoxis.config import load_zalmoxis_config, validate_config

        root = get_zalmoxis_root()
        cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
        cfg['layer_eos_config'] = dict(
            cfg['layer_eos_config'],
            core='PALEOS:iron:0.96+CoreAlloy:H:0.005+CoreAlloy:Si:0.035',
        )
        with caplog.at_level('WARNING', logger='fwl.zalmoxis.config'):
            validate_config(cfg)
        eos_warnings = [m for m in caplog.messages if 'T-independent' in m or 'iron EOS' in m]
        assert eos_warnings == []
        # a real non-iron core component still warns
        caplog.clear()
        cfg['layer_eos_config'] = dict(cfg['layer_eos_config'], core='PALEOS:MgSiO3')
        with caplog.at_level('WARNING', logger='fwl.zalmoxis.config'):
            validate_config(cfg)
        assert any('iron EOS' in m for m in caplog.messages)

    def test_light_element_core_passes_validation(self):
        """A PALEOS iron core with CoreAlloy components is a valid core."""
        from zalmoxis.config import validate_layer_eos

        cfg = {
            'core': 'PALEOS:iron:0.96+CoreAlloy:H:0.005+CoreAlloy:Si:0.035',
            'mantle': 'PALEOS:MgSiO3',
        }
        assert validate_layer_eos(cfg) is None
        assert ca.split_alloy(parse_layer_components(cfg['core']))[1]['Si'] == pytest.approx(
            0.035
        )

    @pytest.mark.parametrize(
        'cfg, message',
        [
            (
                {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3:0.99+CoreAlloy:H:0.01'},
                'belong in the core',
            ),
            ({'core': 'Seager2007:iron:0.99+CoreAlloy:S:0.01'}, 'PALEOS:iron base'),
            ({'core': 'PALEOS:iron:0.99+CoreAlloy:nan:0.01'}, 'unknown core light element'),
            ({'core': 'PALEOS:iron:0.99+CoreAlloy:C:nan'}, 'mass fraction'),
            ({'core': 'PALEOS:iron:0.9+CoreAlloy:H:0.005'}, 'sum to 1'),
            ({'core': 'PALEOS:iron:0.99+CoreAlloy:H'}, 'explicit fraction'),
            ({'core': 'PALEOS:iron+CoreAlloy:H:0.005'}, 'explicit fraction'),
        ],
        ids=[
            'alloy_in_mantle',
            'seager_base',
            'unknown_element',
            'nan_fraction',
            'fractions_sum_0p905',
            'alloy_without_fraction',
            'base_without_fraction',
        ],
    )
    def test_invalid_light_element_core_raises(self, cfg, message, caplog):
        """Each misplaced or malformed light-element core is refused with its reason."""
        from zalmoxis.config import validate_layer_eos

        with caplog.at_level('WARNING'), pytest.raises(ValueError) as exc:
            validate_layer_eos(cfg)
        assert message in str(exc.value)
        assert 'Invalid EOS component' not in str(exc.value)
        # the refusal comes before the parser can report a renormalisation
        assert not [m for m in caplog.messages if 'normalizing' in m]

    @pytest.mark.parametrize('excess, ok', [(1e-8, False), (1e-10, True)])
    def test_fraction_sum_tolerance_is_1e_9(self, excess, ok):
        """Fractions that miss 1 by 1e-8 are refused, by 1e-10 accepted."""
        from zalmoxis.config import validate_layer_eos

        cfg = {'core': f'PALEOS:iron:{0.99 + excess!r}+CoreAlloy:H:0.01'}
        if ok:
            assert validate_layer_eos(cfg) is None
        else:
            with pytest.raises(ValueError, match='sum to 1'):
                validate_layer_eos(cfg)


class TestZeroCarrier:
    """A core without H returns the linear law without evaluating the carrier volume."""

    def test_zero_light_elements_return_the_iron_density(self):
        """With H and O at 0 the density is rho_Fe at 200 random states, without the carrier."""
        from unittest import mock

        rng = np.random.default_rng(0)
        states = zip(
            rng.uniform(1.0, 3e4, 200), rng.uniform(1e9, 3e12, 200), rng.uniform(500, 3e4, 200)
        )
        with mock.patch.object(ca, 'carrier_excess_volume', side_effect=AssertionError):
            for rho_fe, p, t in states:
                assert ca.alloy_density(rho_fe, {'H': 0.0, 'O': 0.0}, p, t) == rho_fe


class TestDispatchWithoutData:
    """The alloy branches of the mixing functions, with the iron EOS patched."""

    MIX = 'PALEOS:iron:0.98+CoreAlloy:H:0.005+CoreAlloy:O:0.015'

    def test_scalar_density_applies_the_alloy_law_to_the_iron_base(self):
        """calculate_mixed_density gives alloy_density of the base iron density and its mzf."""
        from unittest import mock

        from zalmoxis.mixing import calculate_mixed_density

        mix = parse_layer_components(self.MIX)
        with mock.patch('zalmoxis.eos.calculate_density', return_value=11000.0) as iron:
            got = calculate_mixed_density(
                P_CMB, T_LUO, mix, {}, None, None, {}, mushy_zone_factors={'PALEOS:iron': 0.8}
            )
        assert iron.call_args.args[2] == 'PALEOS:iron'
        assert iron.call_args.args[-1] == 0.8
        assert got == ca.alloy_density(11000.0, ca.split_alloy(mix)[1], P_CMB, T_LUO)
        assert got < 0.95 * 11000.0

    def test_batch_density_applies_the_alloy_law_to_the_iron_base(self):
        """calculate_mixed_density_batch gives alloy_density_batch of the base densities."""
        from unittest import mock

        from zalmoxis.mixing import calculate_mixed_density_batch

        mix = parse_layer_components(self.MIX)
        p, t = np.full(3, P_CMB), np.array([5000.0, 8000.0, 9000.0])
        rho_fe = np.array([11000.0, 10400.0, 10300.0])
        with mock.patch('zalmoxis.eos.calculate_density_batch', return_value=rho_fe) as iron:
            got = calculate_mixed_density_batch(
                p, t, mix, {}, None, None, {}, mushy_zone_factors={'PALEOS:iron': 0.8}
            )
        assert iron.call_args.args[3] == 'PALEOS:iron'
        assert iron.call_args.args[-1] == 0.8
        want = ca.alloy_density_batch(rho_fe, ca.split_alloy(mix)[1], p, t)
        assert np.array_equal(got, want)
        assert np.all(got < 0.95 * rho_fe)

    def test_nabla_ad_is_that_of_the_iron_base(self):
        """get_mixed_nabla_ad of an alloy core is the adiabatic gradient of its iron base."""
        from unittest import mock

        from zalmoxis.mixing import get_mixed_nabla_ad

        mix = parse_layer_components(self.MIX)
        values = {'PALEOS:iron': 0.3}
        with mock.patch(
            'zalmoxis.mixing._nabla_ad_for_component',
            side_effect=lambda comp, *a, **k: values.get(comp, 0.9),
        ):
            assert get_mixed_nabla_ad(P_CMB, T_LUO, mix, {}, {}) == 0.3
