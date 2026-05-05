"""Solver-level tests that the configured ``temperature_mode`` reaches the
density solve and the output T(r) array.

Background. A bug in ``zalmoxis.solver._solve`` silently overwrote
``temperature_mode = 'isothermal'`` (and ``'prescribed'``) with
``'linear'`` inside the ``uses_Tdep`` dispatch. Separately, the
T-independent EOS branch hardcoded ``T = 300 K`` regardless of the
configured ``surface_temperature``. Both bugs went uncaught because the
existing ``temperature_mode == 'isothermal'`` tests only exercised
config parsing, not the solver. These tests pin the expected behaviour
end-to-end so the regression cannot return.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _zalmoxis_root() -> str:
    import zalmoxis as _zal

    return os.path.normpath(os.path.join(os.path.dirname(_zal.__file__), '..', '..'))


def _has_paleos_data() -> bool:
    root = _zalmoxis_root()
    return all(
        os.path.isfile(os.path.join(root, 'data', sub, name))
        for sub, name in (
            ('EOS_PALEOS_iron', 'paleos_iron_eos_table_pt.dat'),
            ('EOS_PALEOS_MgSiO3_unified', 'paleos_mgsio3_eos_table_pt.dat'),
        )
    )


def _has_seager_data() -> bool:
    root = _zalmoxis_root()
    return all(
        os.path.isfile(os.path.join(root, 'data', 'EOS_Seager2007', name))
        for name in ('eos_seager07_iron.txt', 'eos_seager07_silicate.txt')
    )


def _input_dir() -> str:
    return os.path.join(_zalmoxis_root(), 'input')


def _base_config(*, T_surf: float, T_center: float, mode: str, eos_pair: tuple[str, str]):
    """Build a minimal but complete ``config_params`` dict for ``main()``."""
    from zalmoxis.constants import earth_mass

    return {
        'planet_mass': 1.0 * earth_mass,
        'core_mass_fraction': 0.325,
        'mantle_mass_fraction': 0,
        'temperature_mode': mode,
        'surface_temperature': T_surf,
        'center_temperature': T_center,
        'temp_profile_file': '',
        'layer_eos_config': {'core': eos_pair[0], 'mantle': eos_pair[1]},
        'mushy_zone_factor': 1.0,
        'num_layers': 50,
        'target_surface_pressure': 101325,
        'data_output_enabled': False,
        'plotting_enabled': False,
    }


@pytest.mark.unit
@pytest.mark.slow
class TestIsothermalTdepEOS:
    """Isothermal mode with PALEOS (T-dependent) must produce a flat T(r).

    Tagged ``slow``: each test runs a full isothermal Picard solve over
    the unified PALEOS table at 1 M_earth (~12-16s); the four-test
    class is ~55s of CI wall. Default ``pytest -m "unit and not slow"``
    excludes; full sweep available via ``pytest -m slow``.
    """

    @pytest.fixture(scope='class')
    def materials(self):
        if not _has_paleos_data():
            pytest.skip('PALEOS unified EOS data not available')
        from zalmoxis.config import load_material_dictionaries

        return load_material_dictionaries()

    def test_isothermal_yields_constant_T_equal_to_surface(self, materials):
        """T(r) is constant and equals ``surface_temperature``.

        Discriminating choice: ``surface_temperature = 3000 K`` and
        ``center_temperature = 6000 K``. The old bug produced a linear
        profile from 6000 K at the centre to 3000 K at the surface, so
        ``T[0]`` was 6000 K. Asserting ``T[0] = 3000 K`` to within
        floating-point noise rules out the linear profile.
        """
        from zalmoxis.solver import main

        T_iso = 3000.0
        cfg = _base_config(
            T_surf=T_iso,
            T_center=6000.0,
            mode='isothermal',
            eos_pair=('PALEOS:iron', 'PALEOS:MgSiO3'),
        )
        result = main(
            cfg,
            material_dictionaries=materials,
            melting_curves_functions=None,
            input_dir=_input_dir(),
        )

        assert result['converged'] is True, 'isothermal solve must converge'
        T = np.asarray(result['temperature'], dtype=float)

        assert np.all(T > 0), 'temperatures must be positive everywhere'
        np.testing.assert_allclose(T, T_iso, rtol=0, atol=1e-9)
        # Discriminator against the historical linear-profile bug.
        assert abs(T[0] - 6000.0) > 100, (
            f'T[0]={T[0]:.1f} K close to center_temperature; the linear-profile '
            'bug would set T[0]=6000 K. Isothermal dispatch is wrong.'
        )

    @pytest.mark.parametrize('T_iso', [500.0, 4500.0])
    def test_isothermal_at_validator_extremes(self, materials, T_iso):
        """Cold (500 K) and hot (4500 K) bounds of the configured T range
        also yield a flat profile. These are near the validator floor
        (300 K) and ceiling (5000 K) and exercise the EOS table edges.
        """
        from zalmoxis.solver import main

        cfg = _base_config(
            T_surf=T_iso,
            T_center=6000.0,
            mode='isothermal',
            eos_pair=('PALEOS:iron', 'PALEOS:MgSiO3'),
        )
        result = main(
            cfg,
            material_dictionaries=materials,
            melting_curves_functions=None,
            input_dir=_input_dir(),
        )
        assert result['converged'] is True
        T = np.asarray(result['temperature'], dtype=float)
        assert np.all(T > 0)
        np.testing.assert_allclose(T, T_iso, rtol=0, atol=1e-9)

    def test_linear_mode_reaches_configured_endpoints(self, materials):
        """Regression check for the patch: linear mode must still produce
        a strictly decreasing linear T(r) from ``center_temperature`` at
        ``r=0`` to ``surface_temperature`` at ``r=R``. This pins the
        path that was correct before the fix and stayed correct after.
        """
        from zalmoxis.solver import main

        cfg = _base_config(
            T_surf=3000.0,
            T_center=6000.0,
            mode='linear',
            eos_pair=('PALEOS:iron', 'PALEOS:MgSiO3'),
        )
        result = main(
            cfg,
            material_dictionaries=materials,
            melting_curves_functions=None,
            input_dir=_input_dir(),
        )
        assert result['converged'] is True
        T = np.asarray(result['temperature'], dtype=float)
        assert np.all(T > 0)
        np.testing.assert_allclose(T[0], 6000.0, rtol=1e-12)
        np.testing.assert_allclose(T[-1], 3000.0, rtol=1e-12)
        # Strictly monotonically decreasing centre to surface.
        assert np.all(np.diff(T) < 0), 'linear T(r) must decrease monotonically'


@pytest.mark.unit
@pytest.mark.slow
class TestIsothermalTindepEOS:
    """Seager2007 (T-independent) must honour ``surface_temperature``.

    Tagged ``slow`` for the same reason as TestIsothermalTdepEOS: each
    test runs a full Picard solve at 1 M_earth, contributing ~13s
    each. Default CI ``pytest -m "unit and not slow"`` excludes.
    """

    @pytest.fixture(scope='class')
    def materials(self):
        if not _has_seager_data():
            pytest.skip('Seager2007 EOS data not available')
        from zalmoxis.config import load_material_dictionaries

        return load_material_dictionaries()

    def test_T_column_equals_surface_not_hardcoded_300(self, materials):
        """Output T(r) reflects ``surface_temperature``, not the legacy 300 K.

        The Seager2007 EOS is T-independent so the structure (rho, P, R)
        is unchanged by this fix; only the T column was wrong. We pin
        ``T[r] = surface_temperature`` and use a discriminator that the
        old hardcoded ``np.ones * 300`` would fail.
        """
        from zalmoxis.solver import main

        T_iso = 3000.0
        cfg = _base_config(
            T_surf=T_iso,
            T_center=6000.0,
            mode='isothermal',
            eos_pair=('Seager2007:iron', 'Seager2007:MgSiO3'),
        )
        result = main(
            cfg,
            material_dictionaries=materials,
            melting_curves_functions=None,
            input_dir=_input_dir(),
        )

        assert result['converged'] is True
        T = np.asarray(result['temperature'], dtype=float)
        assert np.all(T > 0)
        np.testing.assert_allclose(T, T_iso, rtol=0, atol=1e-9)
        # Discriminator: the legacy bug wrote 300 K regardless of input.
        assert abs(T[0] - 300.0) > 100, (
            f'T[0]={T[0]:.1f} K close to legacy hardcoded 300 K; the '
            'T-indep else branch is still ignoring surface_temperature.'
        )
