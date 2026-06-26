"""Tests for ``solver.solve_strong_partition``.

The unit tier mocks ``main()`` and exercises the outer phi_avg
self-consistency loop in isolation. The smoke tier runs the full
solver on a 1 M_earth sub-Neptune-relevant config with PALEOS data.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest

from zalmoxis.mixing import (
    LayerMixture,
    build_partition_profile,
    compute_melt_fraction,
)
from zalmoxis.solver import _mantle_mass_weighted_phi, solve_strong_partition


def _paleos_data_available() -> bool:
    root = os.environ.get('ZALMOXIS_ROOT', '')
    return os.path.isfile(
        os.path.join(root, 'data', 'EOS_PALEOS_iron', 'paleos_iron_eos_table_pt.dat')
    )


def _synthetic_main_result(
    *,
    radii: np.ndarray,
    density: np.ndarray,
    pressure: np.ndarray,
    temperature: np.ndarray,
    mass_enclosed: np.ndarray,
    cmb_mass: float,
    converged: bool = True,
) -> dict:
    """Return a minimal dict with the keys ``solve_strong_partition`` reads."""
    return {
        'converged': converged,
        'radii': radii.copy(),
        'density': density.copy(),
        'pressure': pressure.copy(),
        'temperature': temperature.copy(),
        'mass_enclosed': mass_enclosed.copy(),
        'cmb_mass': cmb_mass,
    }


def _earth_like_shell_grid(n: int = 50):
    """A monotone mantle-only grid above a synthetic CMB."""
    radii = np.linspace(2.0e6, 6.4e6, n)
    density = np.linspace(5500.0, 3000.0, n)
    pressure = np.linspace(1.4e11, 1.0e5, n)
    temperature = np.linspace(4500.0, 3000.0, n)
    mass_enclosed = (
        4.0
        * np.pi
        * np.array(
            [
                np.trapezoid((radii[: i + 1] ** 2) * density[: i + 1], radii[: i + 1])
                for i in range(n)
            ]
        )
    )
    # cmb_mass = 0 puts every shell above the CMB cut-off, so the
    # mantle integral covers the whole synthetic profile.
    cmb_mass = 0.0
    return radii, density, pressure, temperature, mass_enclosed, cmb_mass


@pytest.mark.unit
class TestSolveStrongPartitionOuterLoop:
    """Outer phi_avg loop logic with a mocked ``main()``."""

    def test_pure_silicate_mantle_runs_main_once(self):
        """No volatiles in the mantle: single main() call, sentinel keys."""
        radii, density, pressure, temperature, mass_enclosed, cmb_mass = (
            _earth_like_shell_grid()
        )
        result_template = _synthetic_main_result(
            radii=radii,
            density=density,
            pressure=pressure,
            temperature=temperature,
            mass_enclosed=mass_enclosed,
            cmb_mass=cmb_mass,
        )
        layer_mixtures = {
            'core': LayerMixture(['PALEOS:iron'], [1.0]),
            'mantle': LayerMixture(['PALEOS:MgSiO3'], [1.0]),
        }

        with patch('zalmoxis.solver.main', return_value=result_template) as mocked_main:
            result = solve_strong_partition(
                config_params={'layer_eos_config': {}},
                material_dictionaries={},
                melting_curves_functions=None,
                input_dir='.',
                layer_mixtures=layer_mixtures,
            )

        assert mocked_main.call_count == 1
        assert result['strong_partition_converged'] is True
        assert result['strong_partition_iterations'] == 0
        assert result['phi_avg_converged'] is None
        assert result['X_bulk_mantle'] == {}

    def test_converges_when_main_yields_constant_phi(self):
        """If the structure solve always implies the same phi_avg as the
        guess (here approximated by a synthetic profile where ``phi=1`` at
        every mantle shell), the loop converges on iteration 2: iter 0
        starts at 0.5, iter 1 sees phi_avg=1, iter 2 confirms stability."""
        radii, density, pressure, temperature, mass_enclosed, cmb_mass = (
            _earth_like_shell_grid()
        )
        result_template = _synthetic_main_result(
            radii=radii,
            density=density,
            pressure=pressure,
            temperature=temperature,
            mass_enclosed=mass_enclosed,
            cmb_mass=cmb_mass,
        )
        layer_mixtures = {
            'core': LayerMixture(['PALEOS:iron'], [1.0]),
            'mantle': LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.9, 0.1]),
        }
        # T = 3000 K everywhere, T_sol = 1000, T_liq = 1000 (degenerate) so
        # compute_melt_fraction returns 1.0 wherever T > T_sol.
        sol = liq = lambda _p: 1000.0  # noqa: E731  # intentional terse fixture

        with patch('zalmoxis.solver.main', return_value=result_template) as mocked_main:
            result = solve_strong_partition(
                config_params={'layer_eos_config': {}},
                material_dictionaries={},
                melting_curves_functions=(sol, liq),
                input_dir='.',
                layer_mixtures=layer_mixtures,
                max_iterations=5,
                phi_tolerance=1e-3,
                initial_phi_avg=0.5,
            )

        assert mocked_main.call_count >= 2
        assert result['strong_partition_converged'] is True
        assert result['phi_avg_converged'] == pytest.approx(1.0)
        assert result['X_bulk_mantle'] == {'PALEOS:H2O': pytest.approx(0.1)}

    def test_max_iterations_exhausted_flags_non_convergence(self):
        """Cap ``max_iterations=1`` with an initial guess far from the
        integrated phi_avg: the loop exits unconverged
        (``strong_partition_converged=False``), and ``phi_avg_converged``
        reports the value that built the returned structure (the initial
        guess here), not the dangling look-ahead update."""
        radii, density, pressure, temperature, mass_enclosed, cmb_mass = (
            _earth_like_shell_grid()
        )
        result_template = _synthetic_main_result(
            radii=radii,
            density=density,
            pressure=pressure,
            temperature=temperature,
            mass_enclosed=mass_enclosed,
            cmb_mass=cmb_mass,
        )
        layer_mixtures = {
            'core': LayerMixture(['PALEOS:iron'], [1.0]),
            'mantle': LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.9, 0.1]),
        }

        # T varies 4500 -> 3000 across the mantle; T_sol = 1000,
        # T_liq = 5000 gives a non-trivial phi profile and a phi_avg far
        # from the 0.05 initial guess, so a single iteration cannot meet
        # the very tight tolerance.
        def solidus(_p):
            return 1000.0

        def liquidus(_p):
            return 5000.0

        with patch('zalmoxis.solver.main', return_value=result_template):
            result = solve_strong_partition(
                config_params={'layer_eos_config': {}},
                material_dictionaries={},
                melting_curves_functions=(solidus, liquidus),
                input_dir='.',
                layer_mixtures=layer_mixtures,
                max_iterations=1,
                phi_tolerance=1e-9,
                initial_phi_avg=0.05,
            )

        # Initial guess (0.05) is far from the integrated phi_avg of the
        # synthetic profile, and with only one iteration allowed the loop
        # cannot iterate to a smaller residual.
        assert result['strong_partition_iterations'] == 1
        assert result['strong_partition_converged'] is False
        # The returned structure was built with the initial guess (one
        # iteration), so phi_avg_converged must equal it and not the
        # look-ahead value that would seed a second iteration.
        assert result['phi_avg_converged'] == pytest.approx(0.05)

    def test_main_non_convergence_breaks_loop(self):
        """If main() fails to converge, the outer loop exits gracefully
        without further iteration and the result carries
        ``strong_partition_converged=False``."""
        radii, density, pressure, temperature, mass_enclosed, cmb_mass = (
            _earth_like_shell_grid()
        )
        result_template = _synthetic_main_result(
            radii=radii,
            density=density,
            pressure=pressure,
            temperature=temperature,
            mass_enclosed=mass_enclosed,
            cmb_mass=cmb_mass,
            converged=False,
        )
        layer_mixtures = {
            'core': LayerMixture(['PALEOS:iron'], [1.0]),
            'mantle': LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.9, 0.1]),
        }

        def sol(_p):
            return 1000.0

        def liq(_p):
            return 5000.0

        with patch('zalmoxis.solver.main', return_value=result_template) as mocked_main:
            result = solve_strong_partition(
                config_params={'layer_eos_config': {}},
                material_dictionaries={},
                melting_curves_functions=(sol, liq),
                input_dir='.',
                layer_mixtures=layer_mixtures,
                max_iterations=5,
            )

        assert mocked_main.call_count == 1
        assert result['strong_partition_converged'] is False
        assert 'X_bulk_mantle' in result

    def test_missing_mantle_raises(self):
        """A layer_mixtures dict without a mantle key is a config bug."""
        with pytest.raises(ValueError, match='requires a mantle layer'):
            solve_strong_partition(
                config_params={'layer_eos_config': {}},
                material_dictionaries={},
                melting_curves_functions=None,
                input_dir='.',
                layer_mixtures={'core': LayerMixture(['PALEOS:iron'], [1.0])},
            )


@pytest.mark.unit
class TestSolveStrongPartitionMassAndLayers:
    """Mass conservation over a varying phi(r), ice-layer exclusion, the
    floor fallback flag, and input guards. All on a mocked ``main()``."""

    @staticmethod
    def _integrate_mantle_w(result, profile, species, sol, liq, cmb_mass, mantle_top=None):
        """Mantle-mass-weighted mean of the per-shell volatile fraction,
        mirroring the midpoint scheme of ``_mantle_mass_weighted_phi``."""
        radii = result['radii']
        density = result['density']
        pressure = result['pressure']
        temperature = result['temperature']
        mass_enclosed = result['mass_enclosed']
        num = 0.0
        mantle_mass = 0.0
        for i in range(len(radii) - 1):
            m_mid = 0.5 * (mass_enclosed[i] + mass_enclosed[i + 1])
            if m_mid < cmb_mass:
                continue
            if mantle_top is not None and m_mid >= mantle_top:
                continue
            r_mid = 0.5 * (radii[i] + radii[i + 1])
            dr = radii[i + 1] - radii[i]
            rho_mid = 0.5 * (density[i] + density[i + 1])
            P_mid = 0.5 * (pressure[i] + pressure[i + 1])
            T_mid = 0.5 * (temperature[i] + temperature[i + 1])
            if rho_mid <= 0 or P_mid <= 0 or dr <= 0:
                continue
            phi_i = compute_melt_fraction(P_mid, T_mid, sol, liq)
            w = profile.blend(phi_i)[species]
            shell_mass = rho_mid * 4.0 * np.pi * r_mid**2 * dr
            num += w * shell_mass
            mantle_mass += shell_mass
        return num / mantle_mass

    def test_mass_conserved_over_varying_phi(self):
        """At the converged phi_avg the mantle-integrated water fraction
        recovers X_bulk even though phi(r) varies shell to shell. The
        pre-existing test only checked the constant-phi identity."""
        radii, density, pressure, temperature, mass_enclosed, cmb_mass = (
            _earth_like_shell_grid()
        )
        result_template = _synthetic_main_result(
            radii=radii,
            density=density,
            pressure=pressure,
            temperature=temperature,
            mass_enclosed=mass_enclosed,
            cmb_mass=cmb_mass,
        )
        layer_mixtures = {
            'core': LayerMixture(['PALEOS:iron'], [1.0]),
            'mantle': LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.9, 0.1]),
        }
        # sol=0, liq=9000 => phi = T/9000 varies across mantle shells.
        sol = lambda _p: 0.0  # noqa: E731
        liq = lambda _p: 9000.0  # noqa: E731

        with patch('zalmoxis.solver.main', return_value=result_template):
            result = solve_strong_partition(
                config_params={'layer_eos_config': {}},
                material_dictionaries={},
                melting_curves_functions=(sol, liq),
                input_dir='.',
                layer_mixtures=layer_mixtures,
                max_iterations=50,
                phi_tolerance=1e-12,
            )

        assert result['strong_partition_converged'] is True
        assert result['strong_partition_fallback_to_uniform'] is False
        # phi(r) genuinely varies, so this is not the constant-phi identity.
        phis = [compute_melt_fraction(p, t, sol, liq) for p, t in zip(pressure, temperature)]
        assert max(phis) - min(phis) > 0.05
        profile, _, _ = build_partition_profile(
            layer_mixtures['mantle'], 'strong', result['phi_avg_converged']
        )
        recovered = self._integrate_mantle_w(result, profile, 'PALEOS:H2O', sol, liq, cmb_mass)
        assert recovered == pytest.approx(0.1, rel=1e-6)

    def test_ice_shells_excluded_from_phi_avg(self):
        """3-layer core+mantle+ice: ice shells must not enter phi_avg while
        w_liquid=X/phi_avg is applied to the mantle alone. Pre-fix,
        _mantle_mass_weighted_phi integrated every shell above the CMB."""
        radii, density, pressure, temperature, mass_enclosed, cmb_mass = (
            _earth_like_shell_grid()
        )
        core_mantle_mass = 0.6 * mass_enclosed[-1]
        result_template = _synthetic_main_result(
            radii=radii,
            density=density,
            pressure=pressure,
            temperature=temperature,
            mass_enclosed=mass_enclosed,
            cmb_mass=cmb_mass,
        )
        result_template['core_mantle_mass'] = core_mantle_mass
        layer_mixtures = {
            'core': LayerMixture(['PALEOS:iron'], [1.0]),
            'mantle': LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.9, 0.1]),
            'ice_layer': LayerMixture(['PALEOS:H2O'], [1.0]),
        }
        sol = lambda _p: 0.0  # noqa: E731
        liq = lambda _p: 9000.0  # noqa: E731

        mantle_only, _ = _mantle_mass_weighted_phi(
            result_template, cmb_mass, sol, liq, core_mantle_mass
        )
        with_ice, _ = _mantle_mass_weighted_phi(result_template, cmb_mass, sol, liq, None)
        # The mantle bound must actually change the answer, else the test
        # proves nothing (this is the value the pre-fix code returned).
        assert mantle_only != pytest.approx(with_ice, rel=1e-3)

        with patch('zalmoxis.solver.main', return_value=result_template):
            result = solve_strong_partition(
                config_params={'layer_eos_config': {}},
                material_dictionaries={},
                melting_curves_functions=(sol, liq),
                input_dir='.',
                layer_mixtures=layer_mixtures,
                max_iterations=50,
                phi_tolerance=1e-12,
            )

        assert result['phi_avg_converged'] == pytest.approx(mantle_only)
        assert result['phi_avg_converged'] != pytest.approx(with_ice, rel=1e-3)

    def test_below_floor_reports_fallback_not_strong(self, caplog):
        """When the converged phi_avg sits below the floor the run is
        effectively uniform: strong_partition_converged is False, the
        fallback flag is set, and a warning is logged."""
        import logging

        radii, density, pressure, temperature, mass_enclosed, cmb_mass = (
            _earth_like_shell_grid()
        )
        result_template = _synthetic_main_result(
            radii=radii,
            density=density,
            pressure=pressure,
            temperature=temperature,
            mass_enclosed=mass_enclosed,
            cmb_mass=cmb_mass,
        )
        layer_mixtures = {
            'core': LayerMixture(['PALEOS:iron'], [1.0]),
            # 50% water: floor = 1.01*0.5 = 0.505, above the ~0.4 phi mean.
            'mantle': LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.5, 0.5]),
        }
        sol = lambda _p: 0.0  # noqa: E731
        liq = lambda _p: 9000.0  # noqa: E731

        with patch('zalmoxis.solver.main', return_value=result_template):
            with caplog.at_level(logging.WARNING, logger='zalmoxis.solver'):
                result = solve_strong_partition(
                    config_params={'layer_eos_config': {}},
                    material_dictionaries={},
                    melting_curves_functions=(sol, liq),
                    input_dir='.',
                    layer_mixtures=layer_mixtures,
                    max_iterations=50,
                    phi_tolerance=1e-12,
                )

        assert result['strong_partition_fallback_to_uniform'] is True
        assert result['strong_partition_converged'] is False
        assert 'fell back to the uniform spread' in caplog.text

    def test_max_iterations_zero_raises(self):
        """max_iterations < 1 is a caller error, not a silent None result."""
        layer_mixtures = {
            'mantle': LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.9, 0.1]),
        }
        with pytest.raises(ValueError, match='max_iterations >= 1'):
            solve_strong_partition(
                config_params={'layer_eos_config': {}},
                material_dictionaries={},
                melting_curves_functions=None,
                input_dir='.',
                layer_mixtures=layer_mixtures,
                max_iterations=0,
            )

    def test_invalid_relaxation_raises(self):
        """relaxation must be in (0, 1]."""
        layer_mixtures = {
            'mantle': LayerMixture(['PALEOS:MgSiO3', 'PALEOS:H2O'], [0.9, 0.1]),
        }
        with pytest.raises(ValueError, match='relaxation'):
            solve_strong_partition(
                config_params={'layer_eos_config': {}},
                material_dictionaries={},
                melting_curves_functions=None,
                input_dir='.',
                layer_mixtures=layer_mixtures,
                relaxation=1.5,
            )


@pytest.mark.smoke
@pytest.mark.skipif(
    not _paleos_data_available(),
    reason='PALEOS data not bootstrapped; run tools/setup/get_zalmoxis.sh.',
)
class TestSolveStrongPartitionSmoke:
    """1 M_earth sub-Neptune-relevant run with PALEOS-2phase mantle + H2O."""

    def test_one_earth_mass_strong_partition_converges(self, zalmoxis_root):
        """A 1 M_earth planet with a 10% H2O mantle on partition_rule='strong'
        runs end-to-end, the outer phi_avg loop converges, and the bulk H2O
        inventory is preserved in the returned ``X_bulk_mantle``."""
        from zalmoxis.config import (
            load_material_dictionaries,
            load_solidus_liquidus_functions,
        )
        from zalmoxis.constants import earth_mass

        layer_eos_config = {
            'core': 'PALEOS:iron',
            'mantle': 'PALEOS:MgSiO3:0.9+PALEOS:H2O:0.1',
        }
        config_params = {
            'planet_mass': 1.0 * earth_mass,
            'core_mass_fraction': 0.325,
            'mantle_mass_fraction': 0,
            'temperature_mode': 'adiabatic',
            'surface_temperature': 3000.0,
            'center_temperature': 6000.0,
            'temp_profile_file': '',
            'layer_eos_config': layer_eos_config,
            'rock_solidus': 'Monteux16-solidus',
            'rock_liquidus': 'Monteux16-liquidus-A-chondritic',
            'mushy_zone_factor': 1.0,
            'mushy_zone_factors': {
                'PALEOS:iron': 1.0,
                'PALEOS:MgSiO3': 1.0,
                'PALEOS:H2O': 1.0,
            },
            'condensed_rho_min': 322.0,
            'condensed_rho_scale': 50.0,
            'binodal_T_scale': 50.0,
            'partition_rule': 'strong',
            'num_layers': 150,
            'target_surface_pressure': 101325,
            'data_output_enabled': False,
            'plotting_enabled': False,
        }
        melting_curves_functions = load_solidus_liquidus_functions(
            layer_eos_config,
            config_params['rock_solidus'],
            config_params['rock_liquidus'],
        )
        result = solve_strong_partition(
            config_params,
            material_dictionaries=load_material_dictionaries(),
            melting_curves_functions=melting_curves_functions,
            input_dir=os.path.join(zalmoxis_root, 'input'),
            max_iterations=8,
            phi_tolerance=5e-3,
        )

        assert result['converged'], 'inner Zalmoxis solve did not converge'
        assert result['strong_partition_converged'], 'phi_avg outer loop did not converge'
        assert result['X_bulk_mantle'] == {'PALEOS:H2O': pytest.approx(0.1)}
        assert 0.0 <= result['phi_avg_converged'] <= 1.0
        # Sanity: radius is in a plausible range for a 1 M_earth body
        # with a wet mantle (just above Earth's 6371 km).
        assert 6.0e6 < result['radii'][-1] < 7.5e6
