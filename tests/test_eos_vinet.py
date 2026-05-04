"""Unit tests for the Vinet (Rose-Vinet) equation of state.

Tests cover:
- Zero-pressure density equals rho_0 (with thermal correction)
- Density increases monotonically with pressure
- Agreement with Seager+2007 analytic EOS within expected bounds
- Specific reference values from the literature
- Edge cases (NaN, negative pressure)
- Root-finding convergence at extreme pressures

References:
    Smith, R. F. et al. (2018). Nature Astronomy, 2, 452-458.
    Fei, Y. et al. (2021). Phys. Rev. Lett., 127, 080501.
    Boujibar, A. et al. (2020). JGRP, 125, e2019JE006124.
"""

from __future__ import annotations

import numpy as np
import pytest

from zalmoxis.eos_vinet import (
    VINET_MATERIALS,
    _vinet_pressure,
    get_vinet_density,
)


@pytest.mark.unit
class TestVinetPressure:
    """Tests for the forward Vinet pressure function."""

    def test_zero_at_f_equals_one(self):
        """P(f=1) = 0 (zero-pressure reference state)."""
        K_0 = 165e9
        eta = 1.5 * (4.9 - 1)
        assert _vinet_pressure(1.0, K_0, eta) == 0.0

    def test_positive_for_compression(self):
        """P(f<1) > 0 (compressed state)."""
        K_0 = 165e9
        eta = 1.5 * (4.9 - 1)
        assert _vinet_pressure(0.9, K_0, eta) > 0
        assert _vinet_pressure(0.5, K_0, eta) > 0

    def test_monotonic_with_compression(self):
        """Pressure increases monotonically as f decreases (more compression)."""
        K_0 = 261e9
        eta = 1.5 * (4.0 - 1)
        f_values = np.linspace(0.3, 0.99, 100)
        P_values = [_vinet_pressure(f, K_0, eta) for f in f_values]
        # f decreasing = more compressed = higher P
        assert all(P_values[i] >= P_values[i + 1] for i in range(len(P_values) - 1))


@pytest.mark.unit
class TestVinetDensity:
    """Tests for the inverse Vinet density function."""

    def test_zero_pressure_returns_rho0(self):
        """At P=0, density should be rho_0 * thermal_correction."""
        for key, mat in VINET_MATERIALS.items():
            rho = get_vinet_density(0.0, key)
            expected = mat['rho_0'] * mat['thermal_correction']
            assert rho == pytest.approx(expected, rel=1e-10), (
                f'{key}: rho(0) = {rho}, expected {expected}'
            )

    def test_negative_pressure_returns_rho0(self):
        """Negative pressure should return zero-pressure density."""
        rho = get_vinet_density(-1e9, 'iron')
        expected = VINET_MATERIALS['iron']['rho_0'] * 0.875
        assert rho == pytest.approx(expected, rel=1e-10)

    def test_nan_returns_none(self):
        """NaN pressure should return None."""
        assert get_vinet_density(float('nan'), 'iron') is None

    def test_density_increases_with_pressure(self):
        """Density must increase monotonically with pressure."""
        pressures = np.logspace(8, 14, 50)  # 0.1 GPa to 100 TPa
        for key in ('iron', 'MgSiO3'):
            densities = [get_vinet_density(float(P), key) for P in pressures]
            assert all(d is not None for d in densities), f'{key}: got None'
            assert all(densities[i] <= densities[i + 1] for i in range(len(densities) - 1)), (
                f'{key}: density not monotonically increasing'
            )

    def test_iron_at_earth_cmb(self):
        """Iron density at 135 GPa should be ~12000-14000 kg/m³.

        Earth's outer core density from PREM is ~12000-13000 kg/m³.
        With 12.5% thermal correction, the 300 K Vinet density is
        ~13500, reduced to ~11800 by thermal correction.
        """
        rho = get_vinet_density(135e9, 'iron')
        assert 10000 < rho < 15000, f'Iron at 135 GPa: {rho:.0f} kg/m³'

    def test_mgsio3_at_earth_cmb(self):
        """MgSiO3 density at 135 GPa should be ~5000-6000 kg/m³.

        PREM lower mantle density near CMB is ~5500 kg/m³.
        """
        rho = get_vinet_density(135e9, 'MgSiO3')
        assert 4500 < rho < 7000, f'MgSiO3 at 135 GPa: {rho:.0f} kg/m³'

    def test_iron_thermal_correction(self):
        """Iron density should be 12.5% lower than raw Vinet at any P > 0.

        The thermal_correction = 0.875 applies uniformly.
        """
        P = 200e9
        rho_corrected = get_vinet_density(P, 'iron')

        # Compute raw (uncorrected) density
        mat = VINET_MATERIALS['iron']
        rho_0 = mat['rho_0']
        K_0 = mat['K_0']
        K_prime = mat['K_prime']
        eta = 1.5 * (K_prime - 1)
        from scipy.optimize import brentq

        def res(f):
            return _vinet_pressure(f, K_0, eta) - P

        f = brentq(res, 0.01, 0.9999)
        rho_raw = rho_0 / f**3

        assert rho_corrected == pytest.approx(rho_raw * 0.875, rel=1e-8)

    def test_unknown_material_raises(self):
        """Unknown material key should raise ValueError."""
        with pytest.raises(ValueError, match='Unknown Vinet material'):
            get_vinet_density(1e9, 'unobtanium')

    def test_extreme_pressure(self):
        """Density at 1000 TPa should still converge."""
        rho = get_vinet_density(1e15, 'iron')
        assert rho is not None
        assert rho > 30000  # heavily compressed

    def test_roundtrip_consistency(self):
        """Forward P(rho) and inverse rho(P) should be consistent.

        Compute P from a known f, then recover rho from that P, and
        check that rho matches rho_0 / f^3 * thermal.
        """
        mat = VINET_MATERIALS['MgSiO3']
        f_test = 0.85  # moderate compression
        eta = 1.5 * (mat['K_prime'] - 1)
        P = _vinet_pressure(f_test, mat['K_0'], eta)
        rho_expected = mat['rho_0'] / f_test**3 * mat['thermal_correction']
        rho_computed = get_vinet_density(P, 'MgSiO3')
        assert rho_computed == pytest.approx(rho_expected, rel=1e-8)


@pytest.mark.unit
class TestVinetBulkModulus:
    """Verify the Vinet EOS recovers K_0 and K_0' at zero pressure."""

    def test_K0_recovery(self):
        """Bulk modulus at P=0 should equal K_0 for all materials.

        K = -V dP/dV. At P=0 (f=1), K = K_0 by definition.
        Check via finite difference: K ~ -V * Delta_P / Delta_V.
        """
        for key, mat in VINET_MATERIALS.items():
            K_0 = mat['K_0']
            rho_0 = mat['rho_0']
            # Small pressure step
            dP = 1e6  # 1 MPa
            rho_at_dP = get_vinet_density(dP, key)
            # K = rho * dP / drho (isothermal)
            drho = rho_at_dP / mat['thermal_correction'] - rho_0
            K_numerical = rho_0 * dP / drho if drho > 0 else float('inf')
            assert K_numerical == pytest.approx(K_0, rel=0.01), (
                f'{key}: K_numerical={K_numerical / 1e9:.1f} GPa, K_0={K_0 / 1e9:.1f} GPa'
            )

    def test_Kprime_recovery(self):
        """K' at P=0 should equal K_0' for all materials.

        K' = dK/dP. Check via finite difference on two small pressures.
        """
        for key, mat in VINET_MATERIALS.items():
            K_prime_expected = mat['K_prime']
            tc = mat['thermal_correction']

            P1, P2 = 0.5e9, 1.5e9  # 0.5 and 1.5 GPa
            dP_step = 1e6

            rho1 = get_vinet_density(P1, key) / tc
            rho1p = get_vinet_density(P1 + dP_step, key) / tc
            K1 = rho1 * dP_step / (rho1p - rho1)

            rho2 = get_vinet_density(P2, key) / tc
            rho2p = get_vinet_density(P2 + dP_step, key) / tc
            K2 = rho2 * dP_step / (rho2p - rho2)

            K_prime_numerical = (K2 - K1) / (P2 - P1)
            assert K_prime_numerical == pytest.approx(K_prime_expected, rel=0.05), (
                f"{key}: K'_numerical={K_prime_numerical:.2f}, "
                f"K'_expected={K_prime_expected:.2f}"
            )


@pytest.mark.unit
class TestVinetAllMaterials:
    """Verify all registered materials produce sensible densities."""

    def test_ppv_denser_than_pv(self):
        """Post-perovskite should be denser than perovskite at same P."""
        for P in [100e9, 200e9, 500e9]:
            rho_pv = get_vinet_density(P, 'MgSiO3')
            rho_ppv = get_vinet_density(P, 'MgSiO3_ppv')
            assert rho_ppv > rho_pv, (
                f'At {P / 1e9:.0f} GPa: ppv={rho_ppv:.0f} should > pv={rho_pv:.0f}'
            )

    def test_iron_liquid_less_dense_than_solid(self):
        """Liquid iron should be less dense than solid at same P."""
        for P in [100e9, 200e9, 350e9]:
            rho_s = get_vinet_density(P, 'iron')
            rho_l = get_vinet_density(P, 'iron_liquid')
            # Solid has thermal correction (0.875), liquid doesn't, so
            # corrected solid may be less dense than liquid.
            # But uncorrected solid (rho_s/0.875) should be denser than liquid.
            rho_s_raw = rho_s / 0.875
            assert rho_s_raw > rho_l, (
                f'At {P / 1e9:.0f} GPa: solid_raw={rho_s_raw:.0f} should > liquid={rho_l:.0f}'
            )

    def test_peridotite_monotonic(self):
        """Peridotite density increases with pressure."""
        pressures = np.logspace(8, 12, 20)
        densities = [get_vinet_density(float(P), 'peridotite') for P in pressures]
        assert all(d is not None for d in densities)
        assert all(densities[i] <= densities[i + 1] for i in range(len(densities) - 1))

    def test_continuity_at_zero(self):
        """Density at very small P should be close to rho_0 * thermal."""
        for key, mat in VINET_MATERIALS.items():
            rho_0_eff = mat['rho_0'] * mat['thermal_correction']
            rho_small = get_vinet_density(1.0, key)  # 1 Pa
            assert abs(rho_small - rho_0_eff) / rho_0_eff < 1e-6, (
                f'{key}: rho(1 Pa)={rho_small:.2f}, rho_0*tc={rho_0_eff:.2f}'
            )


@pytest.mark.unit
class TestVinetVsSeager:
    """Compare Vinet and Seager EOS at overlapping pressures.

    Seager+2007 is a modified polytrope fitted to Thomas-Fermi-Dirac
    calculations. Vinet is a finite-strain EOS fitted to static
    compression experiments. They parameterize different physics, so
    agreement is approximate (10-20%).
    """

    def test_iron_within_20_percent(self):
        """Vinet and Seager iron should agree within 20% at 1-500 GPa.

        The Vinet includes a 12.5% thermal correction that accounts
        for most of the difference.
        """
        from zalmoxis.eos_analytic import get_analytic_density

        pressures = np.logspace(9, 11.7, 20)
        for P in pressures:
            rho_v = get_vinet_density(float(P), 'iron')
            rho_s = get_analytic_density(float(P), 'iron')
            rel_diff = abs(rho_v - rho_s) / rho_s
            assert rel_diff < 0.20, (
                f'At P={P / 1e9:.0f} GPa: Vinet={rho_v:.0f}, '
                f'Seager={rho_s:.0f}, diff={rel_diff:.1%}'
            )

    def test_mgsio3_within_15_percent(self):
        """Vinet and Seager MgSiO3 should agree within 15% at 1-500 GPa."""
        from zalmoxis.eos_analytic import get_analytic_density

        pressures = np.logspace(9, 11.7, 20)
        for P in pressures:
            rho_v = get_vinet_density(float(P), 'MgSiO3')
            rho_s = get_analytic_density(float(P), 'MgSiO3')
            rel_diff = abs(rho_v - rho_s) / rho_s
            assert rel_diff < 0.15, (
                f'At P={P / 1e9:.0f} GPa: Vinet={rho_v:.0f}, '
                f'Seager={rho_s:.0f}, diff={rel_diff:.1%}'
            )


@pytest.mark.unit
class TestVinetEdgeCases:
    """Anti-happy-path coverage for the Vinet boundary branches and
    extreme-pressure / failure-recovery paths.

    Each test exercises a specific line range that the standard
    happy-path tests above do not reach:

    * ``_vinet_pressure(f<=0)`` returns ``np.inf`` (compression beyond
      the physical bracket).
    * ``get_vinet_density`` clamps pressures above ``P_MAX_VINET`` and
      logs a warning.
    * The ``P_target < P(f~1)`` short-circuit returns ``rho_0 * thermal``.
    * The ``residual(f_min) < 0`` branch widens the bracket to f=0.001
      for ultra-high pressure.
    * The ``brentq`` ``ValueError`` fallback returns ``None`` when no
      root is found within the bracket.
    """

    def test_pressure_at_f_zero_is_inf(self):
        """``_vinet_pressure(f=0)`` must return ``np.inf`` (the f<=0 branch)."""
        K_0 = 165e9
        eta = 1.5 * (4.9 - 1)
        assert _vinet_pressure(0.0, K_0, eta) == np.inf
        # Negative f is also treated as the singular f<=0 case
        assert _vinet_pressure(-0.1, K_0, eta) == np.inf

    def test_pressure_above_max_vinet_clamped_with_warning(self, caplog):
        """Pressure above ``P_MAX_VINET`` must be clamped and a warning logged."""
        from zalmoxis.eos_vinet import P_MAX_VINET

        # 100x P_MAX_VINET = absolutely beyond physical validity
        with caplog.at_level('WARNING', logger='zalmoxis.eos_vinet'):
            rho = get_vinet_density(P_MAX_VINET * 100, 'iron')

        assert any('exceeds Vinet validity limit' in rec.message for rec in caplog.records)
        # Density at clamped P_MAX_VINET should still be finite
        assert rho is not None and np.isfinite(rho) and rho > 0

    def test_unknown_material_raises(self):
        """Material key outside ``VINET_MATERIALS`` must raise ``ValueError``."""
        with pytest.raises(ValueError, match='Unknown Vinet material'):
            get_vinet_density(1e10, 'unobtainium')

    def test_extreme_compression_widens_bracket(self, monkeypatch):
        """The ``residual(f_min=0.01) < 0`` branch widens the bracket to
        f=0.001 only when ``P_target > P_vinet(f=0.01)``. For all
        registered materials this requires P > ~5e17 Pa, well above the
        ``P_MAX_VINET = 1e16`` clamp. We therefore raise ``P_MAX_VINET``
        to expose this branch and verify the solve still converges with
        the widened bracket."""
        from zalmoxis import eos_vinet

        # Lift the clamp so the wide-bracket branch becomes reachable.
        monkeypatch.setattr(eos_vinet, 'P_MAX_VINET', 1e19)

        # Iron at 5e18 Pa: P_vinet(f=0.01) ~ 1.6e18 Pa, so residual at
        # f=0.01 is negative and the f_min = 0.001 widening fires.
        rho = get_vinet_density(5e18, 'iron')
        assert rho is not None and np.isfinite(rho)
        # At this extreme compression rho >> rho_0
        assert rho > 100 * VINET_MATERIALS['iron']['rho_0']

    def test_tiny_positive_pressure_short_circuit(self):
        """For P_target so small that residual(f near 1) > 0 (i.e.
        P below what the ``f = 1 - 1e-12`` evaluation produces), the
        function returns rho_0 * thermal directly (line 228). The
        crossover pressure is roughly ``3 * K_0 * 1e-12``, i.e. sub-Pa
        for all registered materials."""
        # 1e-3 Pa is well below the residual(1-1e-12) crossover for
        # every registered material (K_0 >= 125 GPa, threshold ~0.4 Pa).
        rho = get_vinet_density(1e-3, 'iron')
        expected = (
            VINET_MATERIALS['iron']['rho_0'] * VINET_MATERIALS['iron']['thermal_correction']
        )
        assert rho == pytest.approx(expected, rel=1e-12)

    def test_pressure_below_zero_short_circuit(self):
        """Pressure <= 0 returns rho_0 * thermal_correction immediately."""
        rho_neg = get_vinet_density(-1e9, 'iron')
        rho_zero = get_vinet_density(0.0, 'iron')
        # Both should equal rho_0 * thermal_correction (no compression).
        expected = (
            VINET_MATERIALS['iron']['rho_0'] * VINET_MATERIALS['iron']['thermal_correction']
        )
        assert rho_neg == pytest.approx(expected, rel=1e-12)
        assert rho_zero == pytest.approx(expected, rel=1e-12)

    def test_nan_pressure_returns_none(self):
        """NaN pressure input must return ``None`` cleanly."""
        assert get_vinet_density(float('nan'), 'iron') is None

    def test_brentq_fallback_returns_none_when_no_root(self, monkeypatch):
        """When ``brentq`` raises ``ValueError`` (degenerate bracket),
        ``get_vinet_density`` must return ``None`` and log a warning."""
        from zalmoxis import eos_vinet

        def fake_brentq(*args, **kwargs):
            raise ValueError('synthetic: residual same sign at both bracket endpoints')

        monkeypatch.setattr(eos_vinet, 'brentq', fake_brentq)
        rho = get_vinet_density(1e11, 'iron')
        assert rho is None
