"""Tests for src/zalmoxis/eos/core_alloy.py and the core-alloy branch of the mixing rules.

Invariants and contract clauses: the light-element fractions of a core mixture
(one iron base, known elements, non-negative and finite, sum below 1, total O
never negative); the linear law of Li et al. (2019); the Luo et al. (2024) Eq. 14
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
        assert ca.split_alloy(_mix(S=0.999))[1]['S'] == pytest.approx(0.999)

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
            st.floats(0.0, 0.15, allow_nan=False),
            min_size=1,
        )
    )
    def test_any_valid_fraction_set_round_trips(self, w):
        """Over the valid domain every fraction comes back and the iron base keeps the rest."""
        base, out = ca.split_alloy(_mix(**w))
        assert base == 'PALEOS:iron'
        assert set(out) == set(w)
        for el, frac in w.items():
            assert out[el] == pytest.approx(frac, abs=1e-15)


class TestLinearLaw:
    """Li et al. (2019) Table 3.3 linear density decrease."""

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
        [
            ('C', 1.3, 1.4),
            ('O', 1.1, 1.2),
            ('Si', 0.7, 0.8),
            ('S', 0.8, 0.8 * 0.9),
            ('H-linear', 8.7, 8.7 / 2),
        ],
    )
    def test_one_wt_percent_matches_li_2019_table_3_3(self, element, beta, wrong_beta):
        """Li et al. 2019 Table 3.3: 1 wt% lowers the density by beta percent (liquid CMB; H solid)."""
        rho_fe = 1.0e4
        rho = ca.alloy_density(rho_fe, {element: 0.01}, P_CMB, 4000.0)
        assert rho == pytest.approx(rho_fe * (1 - beta / 100), rel=1e-12)
        # the solid value (C, O, Si), a 10 % slip (S) or a factor 2 (H) lands outside
        assert abs(rho - rho_fe * (1 - wrong_beta / 100)) > 1e-12 * rho_fe
        assert rho < rho_fe

    def test_nitrogen_leaves_the_density_unchanged(self):
        """Table 3.3 has no N row, so N is carried at zero density effect."""
        rho = ca.alloy_density(1.0e4, {'N': 0.02, 'S': 0.01}, P_CMB, 4000.0)
        assert rho == pytest.approx(1.0e4 * (1 - 0.008), rel=1e-12)
        assert rho == pytest.approx(
            ca.alloy_density(1.0e4, {'S': 0.01}, P_CMB, 4000.0), rel=1e-15
        )

    @pytest.mark.physics_invariant
    def test_linear_alloy_without_positive_density_raises(self):
        """A linear law that drives the density to or below zero fails the composition."""
        with pytest.raises(ca.CoreAlloyError) as exc:
            ca.alloy_density(1.0e4, {'H-linear': 0.2}, P_CMB, 4000.0)
        assert 'linear-law core density' in str(exc.value)
        rho = ca.alloy_density(1.0e4, {'H-linear': 0.1}, P_CMB, 4000.0)
        assert rho == pytest.approx(1.0e4 * (1 - 0.87), rel=1e-12)


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
    @pytest.mark.reference_pinned
    def test_effective_beta_h_at_136_gpa_8000_k(self):
        """With no core O the carrier lowers the density by about 12.8 % per wt% H (docs value)."""
        rho_fe = ca.luo_density(P_CMB, T_LUO, ca.LUO_DRY)
        h = 1e-6
        beta_h = (1 - ca.alloy_density(rho_fe, {'H': h}, P_CMB, T_LUO) / rho_fe) / h
        assert beta_h == pytest.approx(12.83, abs=0.01)
        # the solid linear value 8.7 and the carrier without the O correction (21.6) lie outside
        assert abs(beta_h - 8.7) > 1.0
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
        with caplog.at_level('WARNING', logger='zalmoxis.eos.core_alloy'):
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
    def test_carrier_volume_is_continuous_across_each_box_edge(self, edge, step):
        """Across each face of the Luo box the clamped carrier volume has no step."""
        (p, t), (dp, dt) = edge, step
        below = ca.carrier_excess_volume(p - dp, t - dt)
        above = ca.carrier_excess_volume(p + dp, t + dt)
        assert below == pytest.approx(above, rel=1e-8)
        assert 1e-4 < below < 6e-4

    def test_low_pressure_cool_core_point_has_a_density(self):
        """At 5 GPa, 3000 K (no Eq. 14 root for dry Fe) the 1 wt% H core still has a density."""
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


class TestPaleosCore:
    """The core-alloy branch of the mixing rules on the PALEOS iron table."""

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
        ],
        ids=['alloy_in_mantle', 'seager_base', 'unknown_element', 'nan_fraction'],
    )
    def test_invalid_light_element_core_raises(self, cfg, message):
        """Each misplaced or malformed light-element core is refused with its reason."""
        from zalmoxis.config import validate_layer_eos

        with pytest.raises(ValueError) as exc:
            validate_layer_eos(cfg)
        assert message in str(exc.value)
        assert 'Invalid EOS component' not in str(exc.value)
