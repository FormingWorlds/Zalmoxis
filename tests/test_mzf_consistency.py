"""Tests for mushy_zone_factor consistency across the Zalmoxis density paths.

``mushy_zone_factor`` (mzf) sets ``T_sol = mzf * T_liq``. Zalmoxis has two
ways to turn it into a density:

- unified PALEOS tables blend inside ``get_paleos_unified_density``, which
  takes mzf directly;
- 2-phase PALEOS tables blend inside ``get_Tdep_density``, which only sees a
  solidus and a liquidus function, so mzf reaches it through
  ``load_solidus_liquidus_functions``.

This file exercises:

- the 2-phase density responds to mzf (discrimination, edge continuity,
  monotonicity across the mushy zone, mzf = 1.0 collapse);
- the solidus derived for every mzf is the same fraction of the liquidus;
- the unified and 2-phase densities coincide outside the mushy zone and
  differ by a bounded amount inside it;
- the unified-table liquidus and the analytic PALEOS liquidus stay within
  a measured tolerance.

Anti-happy-path: each class holds an edge case (mzf = 1.0, mushy-zone edges)
and an error path (missing melting curves).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from zalmoxis.config import load_material_dictionaries, load_solidus_liquidus_functions
from zalmoxis.eos.tdep import get_Tdep_density
from zalmoxis.melting_curves import (
    derive_solidus_from_liquidus,
    get_solidus_liquidus_functions,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(30)]

_TWO_PHASE = {'core': 'Seager2007:iron', 'mantle': 'PALEOS-2phase:MgSiO3'}
_RHO_SOLID = 4500.0
_RHO_LIQUID = 3500.0


def _stub_table(pressure, material, phase, temperature, interpolation_functions):
    """Return plausible constant solid and liquid mantle densities (kg/m^3)."""
    return _RHO_SOLID if phase == 'solid_mantle' else _RHO_LIQUID


@pytest.fixture
def stub_tables(monkeypatch):
    """Replace the tabulated lookup used by ``get_Tdep_density`` with constants."""
    monkeypatch.setattr('zalmoxis.eos.tdep.get_tabulated_eos', _stub_table)


def _curves(mzf):
    """Return the (solidus, liquidus) pair the 2-phase loader builds for ``mzf``."""
    out = load_solidus_liquidus_functions(
        _TWO_PHASE, 'Stixrude14-solidus', 'PALEOS-liquidus', mzf
    )
    assert out is not None
    return out


def _density(mzf, frac_of_liquidus, pressure=50e9):
    """2-phase density at ``T = frac_of_liquidus * T_liq`` with stub end members."""
    sol, liq = _curves(mzf)
    temperature = frac_of_liquidus * float(liq(pressure))
    return get_Tdep_density(pressure, temperature, {}, sol, liq, {})


class TestTwoPhaseDensityHonorsMzf:
    """The 2-phase density changes with mzf only inside the mushy zone."""

    def test_mushy_zone_density_depends_on_mzf(self, stub_tables):
        """At T = 0.9 T_liq the mzf = 0.8 mushy midpoint gives the volume-additive mix.

        x = (0.9 - 0.8) / (1 - 0.8) = 0.5, smoothstep(0.5) = 0.5, so the
        specific volume is the mean of the end members. At mzf = 1.0 the same
        point is still below the liquidus and stays solid. A solver that
        ignores mzf returns the solid value for both.
        """
        rho_mushy = _density(0.8, 0.9)
        rho_solid_side = _density(1.0, 0.9)
        expected = 1.0 / (0.5 / _RHO_LIQUID + 0.5 / _RHO_SOLID)
        assert rho_mushy == pytest.approx(expected, rel=1e-9)
        assert rho_solid_side == pytest.approx(_RHO_SOLID, rel=1e-12)
        assert rho_mushy < rho_solid_side

    @pytest.mark.parametrize(
        ('frac', 'expected'),
        [(0.6, _RHO_SOLID), (1.3, _RHO_LIQUID)],
        ids=['far-below-solidus', 'far-above-liquidus'],
    )
    def test_density_outside_mushy_zone_ignores_mzf(self, stub_tables, frac, expected):
        """Deep solid and deep liquid points return the end member for any mzf."""
        for mzf in (0.7, 0.8, 0.9, 1.0):
            assert _density(mzf, frac) == pytest.approx(expected, rel=1e-12)

    def test_density_is_continuous_at_mushy_zone_edges(self, stub_tables):
        """No jump at T_sol or T_liq: the smoothstep ramp meets both end members.

        Steps of 1e-9 in T/T_liq keep the ramp within 1e-6 (relative) of the
        end member, well below the 22% solid-to-liquid gap of the stub tables.
        """
        sol, liq = _curves(0.8)
        p = 50e9
        t_sol, t_liq = float(sol(p)), float(liq(p))
        just_above_sol = get_Tdep_density(p, t_sol * (1.0 + 1e-9), {}, sol, liq, {})
        just_below_liq = get_Tdep_density(p, t_liq * (1.0 - 1e-9), {}, sol, liq, {})
        assert just_above_sol == pytest.approx(_RHO_SOLID, rel=1e-6)
        assert just_below_liq == pytest.approx(_RHO_LIQUID, rel=1e-6)

    def test_density_decreases_monotonically_across_mushy_zone(self, stub_tables):
        """Melting lowers the density at every step through the mushy zone."""
        fracs = np.linspace(0.8, 1.0, 41)
        rho = np.array([_density(0.8, f) for f in fracs])
        assert np.all(np.diff(rho) <= 0.0)
        assert rho[0] > rho[-1]
        assert np.all(rho > 0.0)

    def test_mzf_one_collapses_the_mushy_zone(self, stub_tables):
        """At mzf = 1.0 the density switches solid to liquid across T_liq, and stays finite.

        The point T = T_liq itself takes the solid branch (``T <= T_sol`` with
        T_sol = T_liq), so the test brackets the liquidus instead.
        """
        below = _density(1.0, 1.0 - 1e-6)
        above = _density(1.0, 1.0 + 1e-6)
        assert below == pytest.approx(_RHO_SOLID, rel=1e-12)
        assert above == pytest.approx(_RHO_LIQUID, rel=1e-12)
        assert np.isfinite(below) and np.isfinite(above)

    def test_missing_melting_curves_raise(self, stub_tables):
        """A 2-phase density without melting curves is an error, not a silent solid."""
        with pytest.raises(ValueError, match='solidus_func and liquidus_func'):
            get_Tdep_density(50e9, 4000.0, {}, None, None, {})


class TestDerivedSolidus:
    """The solidus is the same fixed fraction of the liquidus at every pressure."""

    @pytest.mark.parametrize('mzf', [0.7, 0.8, 0.9, 1.0])
    def test_solidus_over_liquidus_equals_mzf(self, mzf):
        """T_sol / T_liq equals mzf across 1 GPa to 1 TPa for the 2-phase loader."""
        sol, liq = _curves(mzf)
        pressures = np.logspace(9, 12, 25)
        ratio = np.array([float(sol(p)) / float(liq(p)) for p in pressures])
        np.testing.assert_allclose(ratio, mzf, rtol=1e-12)

    def test_loader_matches_direct_derivation(self):
        """The loader output equals ``derive_solidus_from_liquidus`` on PALEOS-liquidus."""
        _, base_liq = get_solidus_liquidus_functions(liquidus_id='PALEOS-liquidus')
        direct = derive_solidus_from_liquidus(base_liq, 0.85)
        sol, liq = _curves(0.85)
        for p in (1e9, 1e10, 1e11, 1e12):
            assert float(sol(p)) == pytest.approx(float(direct(p)), rel=1e-12)
            assert float(liq(p)) == pytest.approx(float(base_liq(p)), rel=1e-12)

    def test_solidus_scales_the_liquidus_exactly(self):
        """The wrapper multiplies, not offsets: doubling P-independent mzf halves nothing."""
        liq = lambda p: 2000.0 + 0.0 * p  # noqa: E731  (constant curve isolates the factor)
        sol = derive_solidus_from_liquidus(liq, 0.75)
        assert sol(1e9) == pytest.approx(1500.0, rel=1e-12)
        assert sol(1e11) == pytest.approx(1500.0, rel=1e-12)


def _unified_table_file():
    """Path of the unified PALEOS MgSiO3 table, or skip when it is not on disk."""
    registry = load_material_dictionaries()
    candidates = [registry['PALEOS:MgSiO3']['eos_file']]
    fwl = os.environ.get('FWL_DATA')
    if fwl:
        candidates.append(
            os.path.join(
                fwl,
                'zalmoxis_eos',
                'EOS_PALEOS_MgSiO3_unified',
                'paleos_mgsio3_eos_table_pt.dat',
            )
        )
    for path in candidates:
        if os.path.isfile(path):
            return path
    pytest.skip('unified PALEOS MgSiO3 table not available')


def _two_phase_tables_available():
    """Skip when the 2-phase PALEOS MgSiO3 tables are not on disk."""
    mat = load_material_dictionaries()['PALEOS-2phase:MgSiO3']
    for key in ('solid_mantle', 'melted_mantle'):
        if not os.path.isfile(mat[key]['eos_file']):
            pytest.skip('2-phase PALEOS MgSiO3 tables not available')
    return mat


@pytest.fixture(scope='module')
def two_phase_cache():
    """Interpolation cache shared by the 2-phase lookups of one worker.

    Each 2-phase table is parsed from text once per worker instead of once per
    density call.
    """
    return {}


@pytest.fixture(scope='module')
def unified_cache():
    """Interpolation cache shared by the unified-table lookups of one worker."""
    return {}


# Real tables are parsed on first use, which is slow under coverage tracing.
@pytest.mark.timeout(1800)
class TestUnifiedVersusTwoPhase:
    """The unified blend and the 2-phase blend share one phase-boundary basis."""

    def test_table_liquidus_tracks_analytic_liquidus(self):
        """The unified-table melt curve stays within 2% of PALEOS-liquidus, 1 GPa to 1 TPa.

        The 2-phase and coupled paths build their liquidus from the analytic
        curve, the unified blend reads the table's own phase boundary. The
        measured maximum difference is 1.3%.
        """
        from zalmoxis.eos.interpolation import _ensure_unified_cache

        cached = _ensure_unified_cache(_unified_table_file(), {})
        log_p, log_t = cached['liquidus_log_p'], cached['liquidus_log_t']
        _, analytic = get_solidus_liquidus_functions(liquidus_id='PALEOS-liquidus')
        pressures = np.logspace(9, 12, 13)
        table_t = 10.0 ** np.interp(np.log10(pressures), log_p, log_t)
        analytic_t = np.array([float(analytic(p)) for p in pressures])
        np.testing.assert_allclose(table_t, analytic_t, rtol=0.02)
        assert np.max(np.abs(table_t / analytic_t - 1.0)) > 1e-4  # curves are not identical

    def test_unified_density_responds_to_mzf(self):
        """Inside the mushy zone the unified density moves with mzf.

        At P = 50 GPa and T = 0.9 T_liq the mzf = 0.8 point is partly melted
        (about 8% lower density than the mzf = 1.0 solid-side value). Ignoring
        mzf in the unified blend would leave the two equal.
        """
        from zalmoxis.eos.paleos import get_paleos_unified_density

        mat = dict(load_material_dictionaries()['PALEOS:MgSiO3'])
        mat['eos_file'] = _unified_table_file()
        _, liq = _curves(1.0)
        p = 50e9
        t = 0.9 * float(liq(p))
        cache = {}
        rho_mushy = get_paleos_unified_density(p, t, mat, 0.8, cache)
        rho_solid = get_paleos_unified_density(p, t, mat, 1.0, cache)
        assert rho_mushy < rho_solid
        assert (rho_solid - rho_mushy) / rho_solid > 0.05

    @pytest.mark.parametrize('pressure', [5e9, 50e9, 200e9], ids=['5GPa', '50GPa', '200GPa'])
    @pytest.mark.parametrize('frac', [0.6, 1.3], ids=['solid', 'liquid'])
    def test_unified_and_two_phase_agree_outside_mushy_zone(
        self, pressure, frac, two_phase_cache, unified_cache
    ):
        """Both blends reduce to the same table value away from the mushy zone."""
        from zalmoxis.eos.paleos import get_paleos_unified_density

        two_phase = _two_phase_tables_available()
        uni = dict(load_material_dictionaries()['PALEOS:MgSiO3'])
        uni['eos_file'] = _unified_table_file()
        sol, liq = _curves(0.8)
        t = frac * float(liq(pressure))
        rho_two = get_Tdep_density(pressure, t, two_phase, sol, liq, two_phase_cache)
        rho_uni = get_paleos_unified_density(pressure, t, uni, 0.8, unified_cache)
        assert rho_two == pytest.approx(rho_uni, rel=1e-4)

    @pytest.mark.parametrize('pressure', [5e9, 50e9, 200e9], ids=['5GPa', '50GPa', '200GPa'])
    def test_mushy_zone_disagreement_is_bounded(self, pressure, two_phase_cache, unified_cache):
        """Inside the mushy zone the two blends differ by less than 5% (measured max 3.4%)."""
        from zalmoxis.eos.paleos import get_paleos_unified_density

        two_phase = _two_phase_tables_available()
        uni = dict(load_material_dictionaries()['PALEOS:MgSiO3'])
        uni['eos_file'] = _unified_table_file()
        sol, liq = _curves(0.8)
        t_liq = float(liq(pressure))
        for frac in (0.85, 0.9, 0.95, 1.0):
            rho_two = get_Tdep_density(
                pressure, frac * t_liq, two_phase, sol, liq, two_phase_cache
            )
            rho_uni = get_paleos_unified_density(
                pressure, frac * t_liq, uni, 0.8, unified_cache
            )
            assert rho_two == pytest.approx(rho_uni, rel=0.05)


_NABLA_SOLID = 0.20
_NABLA_LIQUID = 0.30


@pytest.fixture
def stub_nabla(monkeypatch):
    """Return distinct solid and liquid nabla_ad values from the table lookup."""

    def _nabla(pressure, temperature, material, phase, interpolation_functions):
        return _NABLA_SOLID if phase == 'solid_mantle' else _NABLA_LIQUID

    monkeypatch.setattr('zalmoxis.eos.temperature._get_paleos_nabla_ad', _nabla)


class TestDtdpPhaseRouting:
    """``_compute_paleos_dtdp`` picks the phase table from the solidus and liquidus."""

    _P = 50e9

    def _dtdp(self, t_sol, t_liq, temperature):
        from zalmoxis.eos.temperature import _compute_paleos_dtdp

        return _compute_paleos_dtdp(
            self._P, temperature, {}, lambda p: t_sol, lambda p: t_liq, {}
        )

    def test_inverted_curves_fall_back_to_solid_table(self, stub_nabla):
        """With T_liq < T_sol the solid table is used even at T above both curves."""
        temperature = 3500.0
        dtdp = self._dtdp(t_sol=3000.0, t_liq=2000.0, temperature=temperature)
        assert dtdp == pytest.approx(_NABLA_SOLID * temperature / self._P)

    def test_zero_width_mushy_zone_uses_liquid_above_curve(self, stub_nabla):
        """With T_sol == T_liq (mzf = 1.0) a temperature above the curve is liquid."""
        temperature = 3500.0
        dtdp = self._dtdp(t_sol=3000.0, t_liq=3000.0, temperature=temperature)
        assert dtdp == pytest.approx(_NABLA_LIQUID * temperature / self._P)

    def test_zero_width_mushy_zone_uses_solid_below_curve(self, stub_nabla):
        """With T_sol == T_liq a temperature below the curve is solid."""
        temperature = 2500.0
        dtdp = self._dtdp(t_sol=3000.0, t_liq=3000.0, temperature=temperature)
        assert dtdp == pytest.approx(_NABLA_SOLID * temperature / self._P)

    def test_mushy_zone_blends_with_melt_fraction(self, stub_nabla):
        """Halfway through the mushy zone nabla_ad is the mean of the two tables."""
        temperature = 2500.0
        dtdp = self._dtdp(t_sol=2000.0, t_liq=3000.0, temperature=temperature)
        expected = 0.5 * (_NABLA_SOLID + _NABLA_LIQUID) * temperature / self._P
        assert dtdp == pytest.approx(expected)


class TestCollapsedBoundaryConsumersAgree:
    """Every numpy consumer routes T == T_sol == T_liq to the solid branch.

    At mzf = 1.0 the mushy band has zero width, so T_sol == T_liq. The three
    numpy density/phase consumers must all pick solid at that exact point:

    - ``get_Tdep_density`` (density);
    - ``get_Tdep_material`` -> ``evaluate_phase`` (phase label);
    - ``_compute_paleos_dtdp`` (nabla_ad routing).

    A consumer that used an inclusive ``>= T_liq`` there would flip to liquid
    and disagree with the other two. The JAX density path is pinned to the
    same convention in ``test_jax_tdep_parity``.
    """

    _P = 50e9
    _T_STAR = 3000.0

    def test_density_selects_solid_at_equality(self, stub_tables):
        """get_Tdep_density returns the solid end member at T == T_sol == T_liq."""
        const = lambda p: self._T_STAR  # noqa: E731
        rho = get_Tdep_density(self._P, self._T_STAR, {}, const, const, {})
        assert rho == pytest.approx(_RHO_SOLID, rel=1e-12)

    def test_phase_label_is_solid_at_equality(self):
        """get_Tdep_material labels T == T_sol == T_liq as solid, not melted."""
        from zalmoxis.eos.tdep import get_Tdep_material

        const = lambda p: self._T_STAR  # noqa: E731
        phase = get_Tdep_material(self._P, self._T_STAR, const, const)
        assert phase == 'solid_mantle'

    def test_nabla_ad_routing_is_solid_at_equality(self, stub_nabla):
        """_compute_paleos_dtdp uses the solid table at T == T_sol == T_liq."""
        from zalmoxis.eos.temperature import _compute_paleos_dtdp

        dtdp = _compute_paleos_dtdp(
            self._P,
            self._T_STAR,
            {},
            lambda p: self._T_STAR,
            lambda p: self._T_STAR,
            {},
        )
        assert dtdp == pytest.approx(_NABLA_SOLID * self._T_STAR / self._P)
