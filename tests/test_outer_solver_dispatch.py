"""Tests for the outer-solver dispatch knob.

Covers the plumbing only; the Newton iteration body itself is tested
in ``test_outer_solver_newton.py``.

Three angles:

(1) Default behavior: when ``outer_solver`` is absent from
    ``config_params``, ``main()`` dispatches to ``_solve()`` (damped
    Picard) and behavior is unchanged. Verified by mocking ``_solve``
    and confirming it is called.

(2) Validation: ``main()`` raises ``ValueError`` on any value other
    than ``'picard'`` or ``'newton'``. Tests the boundary against
    plausible typos (``'Newton'``, ``''``, ``None`` does NOT raise
    because it falls back to the default).

(3) Newton dispatch: when ``outer_solver='newton'``, ``main()`` calls
    ``_solve_newton_outer()`` (currently raises NotImplementedError;
    the test asserts the raise happens, not what's inside).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from zalmoxis.solver import _VALID_OUTER_SOLVERS, _default_solver_params, main

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# Minimal config fixture: enough for main() to reach the dispatch line
# without actually solving anything.
# ----------------------------------------------------------------------


@pytest.fixture
def minimal_config():
    """A minimal config_params dict that gets through main()'s dispatch.

    The actual values do not need to produce a solvable structure:
    main()'s dispatch happens before any solver work, so we can mock
    _solve() / _solve_newton_outer() entirely.
    """
    cp = {
        'planet_mass': 5.972e24,
        'core_mass_fraction': 0.325,
        'mantle_mass_fraction': 0.675,
        'temperature_mode': 'isothermal',
        'surface_temperature': 1500.0,
        'cmb_temperature': 4000.0,
        'center_temperature': 5000.0,
        'layer_eos_config': {'core': 'PALEOS:iron', 'mantle': 'PALEOS-2phase:MgSiO3'},
        'num_layers': 100,
        'use_jax': False,
        'use_anderson': False,
    }
    return cp


# ----------------------------------------------------------------------
# (1) Default behavior: 'outer_solver' absent -> picard path
# ----------------------------------------------------------------------


def test_default_dispatches_to_picard(minimal_config):
    """When 'outer_solver' is absent, _solve() (Picard) is called."""
    sentinel = {'converged': True, 'mass_enclosed': [1.0e24], 'radii': [6.4e6]}
    with (
        patch('zalmoxis.solver._solve', return_value=sentinel) as mock_solve,
        patch('zalmoxis.solver._solve_newton_outer') as mock_newton,
    ):
        result = main(
            minimal_config,
            material_dictionaries={},
            melting_curves_functions=None,
            input_dir='/tmp',
        )
    assert mock_solve.called, '_solve() must be called when outer_solver absent'
    assert not mock_newton.called, '_solve_newton_outer() must NOT be called'
    # main() also runs retry-skip logic which mutates the sentinel; the
    # important thing is _solve was called and the result flowed through.
    assert result is sentinel or result.get('converged') is True


def test_explicit_picard_dispatches_to_picard(minimal_config):
    """outer_solver='picard' explicitly is equivalent to absent."""
    cp = dict(minimal_config)
    cp['outer_solver'] = 'picard'
    sentinel = {'converged': True, 'mass_enclosed': [1.0e24], 'radii': [6.4e6]}
    with (
        patch('zalmoxis.solver._solve', return_value=sentinel) as mock_solve,
        patch('zalmoxis.solver._solve_newton_outer') as mock_newton,
    ):
        main(cp, material_dictionaries={}, melting_curves_functions=None, input_dir='/tmp')
    assert mock_solve.called
    assert not mock_newton.called


# ----------------------------------------------------------------------
# (2) Validation: invalid values raise ValueError
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    'bad_value',
    [
        'Newton',  # case-sensitivity: must be lowercase
        'PICARD',
        '',
        'broyden',
        'levenberg',
        'newton ',  # trailing space
        ' newton',
        42,  # wrong type
    ],
)
def test_invalid_outer_solver_raises(minimal_config, bad_value):
    """ValueError on any value not in _VALID_OUTER_SOLVERS."""
    cp = dict(minimal_config)
    cp['outer_solver'] = bad_value
    with pytest.raises(ValueError, match='outer_solver'):
        main(cp, material_dictionaries={}, melting_curves_functions=None, input_dir='/tmp')


def test_valid_outer_solvers_set_is_picard_and_newton():
    """Discriminating sanity: the validation set is exactly these two.

    Catches accidental additions/removals during refactoring.
    """
    assert set(_VALID_OUTER_SOLVERS) == {'picard', 'newton'}


# ----------------------------------------------------------------------
# (3) Newton dispatch: routes to _solve_newton_outer
# ----------------------------------------------------------------------


def test_newton_dispatches_to_solve_newton_outer(minimal_config):
    """outer_solver='newton' calls _solve_newton_outer (not _solve)."""
    cp = dict(minimal_config)
    cp['outer_solver'] = 'newton'
    sentinel = {'converged': True, 'mass_enclosed': [1.0e24], 'radii': [6.4e6]}
    with (
        patch('zalmoxis.solver._solve_newton_outer', return_value=sentinel) as mock_newton,
        patch('zalmoxis.solver._solve') as mock_solve,
    ):
        result = main(
            cp,
            material_dictionaries={},
            melting_curves_functions=None,
            input_dir='/tmp',
        )
    assert mock_newton.called, '_solve_newton_outer() must be called for newton path'
    assert not mock_solve.called, '_solve() must NOT be called for newton path'
    assert result is sentinel


def test_newton_rejects_loose_integrator_tolerance(minimal_config):
    """Newton requires relative_tolerance <= 1e-7.

    Zalmoxis' default integrator tolerances (1e-5 / 1e-6) leave M(R)
    noise of ~1e-3 which swamps Newton's central-difference dM/dR.
    The function must fail loudly at entry rather than silently
    producing garbage.

    Discriminating: ValueError (not generic Exception); message
    mentions both 'relative_tolerance' and 'newton' so a reader can
    diagnose without running the iteration.
    """
    cp = dict(minimal_config)
    cp['outer_solver'] = 'newton'
    cp['relative_tolerance'] = 1.0e-5  # default, too loose
    with pytest.raises(ValueError, match='relative_tolerance'):
        main(
            cp,
            material_dictionaries={},
            melting_curves_functions=None,
            input_dir='/tmp',
        )


# ----------------------------------------------------------------------
# (4) Default param defaults set 'outer_solver' to 'picard'
# ----------------------------------------------------------------------


def test_default_solver_params_outer_solver_is_picard():
    """_default_solver_params must include 'outer_solver': 'picard'.

    This guarantees that any caller using these defaults (incl. the
    retry path, _tighten_solver_params, etc) keeps Picard as the
    fall-back: a Newton-broken state should NOT silently convert
    to Picard via tightened defaults.
    """
    defaults = _default_solver_params(planet_mass=5.972e24)
    assert defaults['outer_solver'] == 'picard'


def test_default_solver_params_outer_solver_consistent_across_masses():
    """The default outer_solver must NOT depend on planet_mass.

    A user setting outer_solver='newton' for a 1 M_E planet should
    get the same outer_solver back if they mass-sweep to 10 M_E:
    the choice is algorithmic, not physical.
    """
    for m_kg in (5.972e23, 5.972e24, 5.972e25):
        d = _default_solver_params(planet_mass=m_kg)
        assert d['outer_solver'] == 'picard', (
            f'Default outer_solver must be picard at all masses, '
            f'got {d["outer_solver"]!r} at {m_kg:.2e} kg'
        )


# ----------------------------------------------------------------------
# (5) Advisory: implicit-picard hot/massive-profile INFO message
# ----------------------------------------------------------------------
#
# When outer_solver is absent from config_params (so the default picard
# dispatch fires) AND the profile is hot fully-molten or super-Earth
# scale, main() should log a one-line INFO suggesting Newton. The
# advisory must NOT fire when outer_solver is set explicitly (either
# value).
#
# Threshold rationale:
# - planet_mass > 2 M_E: the basin attractor recurs across 1, 3, 5,
#   10 M_E for hot ICs; 2 M_E is a conservative boundary above which
#   Picard is most likely to need rescue.
# - surface_temperature > 3000 K: hot fully-molten regime where the
#   failure mode is observed in coupled runs.

# Earth mass in kg, matches zalmoxis.constants.earth_mass to 4 sig figs;
# tests don't import the constant to keep the test file self-contained.
_EARTH_MASS_KG = 5.972e24


def _patch_solvers_to_sentinel():
    """Helper context: patch both solver entry points to a converged
    sentinel so main() reaches and exits the dispatch logic without
    doing real work.
    """
    sentinel = {'converged': True, 'mass_enclosed': [1.0e24], 'radii': [6.4e6]}
    return sentinel


def test_advisory_fires_on_super_earth(minimal_config, caplog):
    """planet_mass > 2 M_E with implicit picard -> advisory INFO fires."""
    cp = dict(minimal_config)
    cp['planet_mass'] = 3.0 * _EARTH_MASS_KG  # 3 M_E
    # surface_temperature stays at fixture default 1500 K (cool)
    sentinel = _patch_solvers_to_sentinel()
    with (
        caplog.at_level('INFO', logger='zalmoxis.solver'),
        patch('zalmoxis.solver._solve', return_value=sentinel),
    ):
        main(cp, material_dictionaries={}, melting_curves_functions=None, input_dir='/tmp')
    msgs = [r.getMessage() for r in caplog.records if r.levelname == 'INFO']
    advisory = [m for m in msgs if "outer_solver='newton'" in m]
    assert len(advisory) == 1, f'expected exactly 1 advisory, got {len(advisory)}: {advisory}'
    # Discriminating: the message must name the *triggering* condition,
    # not just emit a generic suggestion.
    assert 'planet_mass' in advisory[0]
    assert '3.00 M_E' in advisory[0]
    # Cool surface should NOT be cited.
    assert 'surface_temperature' not in advisory[0]


def test_advisory_fires_on_hot_surface(minimal_config, caplog):
    """surface_temperature > 3000 K with implicit picard -> advisory fires."""
    cp = dict(minimal_config)
    # planet_mass stays at fixture default 1 M_E (light)
    cp['surface_temperature'] = 3500.0  # hot fully-molten
    sentinel = _patch_solvers_to_sentinel()
    with (
        caplog.at_level('INFO', logger='zalmoxis.solver'),
        patch('zalmoxis.solver._solve', return_value=sentinel),
    ):
        main(cp, material_dictionaries={}, melting_curves_functions=None, input_dir='/tmp')
    msgs = [r.getMessage() for r in caplog.records if r.levelname == 'INFO']
    advisory = [m for m in msgs if "outer_solver='newton'" in m]
    assert len(advisory) == 1
    assert 'surface_temperature=3500' in advisory[0]
    # Light planet should NOT be cited.
    assert 'planet_mass' not in advisory[0]


def test_advisory_fires_on_both_triggers_and_lists_both(minimal_config, caplog):
    """Both triggers should be cited explicitly in the same INFO line."""
    cp = dict(minimal_config)
    cp['planet_mass'] = 5.0 * _EARTH_MASS_KG
    cp['surface_temperature'] = 4000.0
    sentinel = _patch_solvers_to_sentinel()
    with (
        caplog.at_level('INFO', logger='zalmoxis.solver'),
        patch('zalmoxis.solver._solve', return_value=sentinel),
    ):
        main(cp, material_dictionaries={}, melting_curves_functions=None, input_dir='/tmp')
    msgs = [r.getMessage() for r in caplog.records if r.levelname == 'INFO']
    advisory = [m for m in msgs if "outer_solver='newton'" in m]
    assert len(advisory) == 1
    # Both triggers must be in the same line.
    assert 'planet_mass' in advisory[0]
    assert '5.00 M_E' in advisory[0]
    assert 'surface_temperature=4000' in advisory[0]


def test_advisory_silent_on_earth_like(minimal_config, caplog):
    """1 M_E + 1500 K (fixture defaults) -> no advisory."""
    sentinel = _patch_solvers_to_sentinel()
    with (
        caplog.at_level('INFO', logger='zalmoxis.solver'),
        patch('zalmoxis.solver._solve', return_value=sentinel),
    ):
        main(
            minimal_config,
            material_dictionaries={},
            melting_curves_functions=None,
            input_dir='/tmp',
        )
    msgs = [r.getMessage() for r in caplog.records if r.levelname == 'INFO']
    advisory = [m for m in msgs if "outer_solver='newton'" in m]
    assert advisory == [], f'no advisory expected for 1 M_E + 1500 K profile, got {advisory}'


@pytest.mark.parametrize('outer_solver_value', ['picard', 'newton'])
def test_advisory_silent_when_outer_solver_explicit(minimal_config, caplog, outer_solver_value):
    """Explicit outer_solver (either value) suppresses the advisory.

    Discriminating: explicit 'picard' must also suppress, even though the
    behavior is identical to implicit-default picard. The advisory is for
    users who haven't thought about the choice; explicit is a deliberate
    decision that should not be re-suggested.
    """
    cp = dict(minimal_config)
    cp['planet_mass'] = 5.0 * _EARTH_MASS_KG  # would trigger advisory if implicit
    cp['surface_temperature'] = 4000.0  # would also trigger
    cp['outer_solver'] = outer_solver_value
    # Newton requires tightened tols.
    if outer_solver_value == 'newton':
        cp['relative_tolerance'] = 1.0e-9
        cp['absolute_tolerance'] = 1.0e-10
    sentinel = _patch_solvers_to_sentinel()
    with (
        caplog.at_level('INFO', logger='zalmoxis.solver'),
        patch('zalmoxis.solver._solve', return_value=sentinel),
        patch('zalmoxis.solver._solve_newton_outer', return_value=sentinel),
    ):
        main(cp, material_dictionaries={}, melting_curves_functions=None, input_dir='/tmp')
    msgs = [r.getMessage() for r in caplog.records if r.levelname == 'INFO']
    advisory = [m for m in msgs if "outer_solver='newton'" in m]
    assert advisory == [], (
        f'explicit outer_solver={outer_solver_value!r} must suppress advisory, got {advisory}'
    )


def test_advisory_boundary_2_M_E_and_3000_K_silent(minimal_config, caplog):
    """Strict-greater-than thresholds: exactly 2 M_E and exactly 3000 K
    must NOT fire.

    Discriminating: this catches an off-by-one in the threshold (>= vs >).
    Picard works fine at exactly 2 M_E; the basin attractor is a
    structural feature beyond the boundary.
    """
    cp = dict(minimal_config)
    cp['planet_mass'] = 2.0 * _EARTH_MASS_KG  # exactly at threshold
    cp['surface_temperature'] = 3000.0  # exactly at threshold
    sentinel = _patch_solvers_to_sentinel()
    with (
        caplog.at_level('INFO', logger='zalmoxis.solver'),
        patch('zalmoxis.solver._solve', return_value=sentinel),
    ):
        main(cp, material_dictionaries={}, melting_curves_functions=None, input_dir='/tmp')
    msgs = [r.getMessage() for r in caplog.records if r.levelname == 'INFO']
    advisory = [m for m in msgs if "outer_solver='newton'" in m]
    assert advisory == []


def test_advisory_handles_missing_keys_gracefully(minimal_config, caplog):
    """Defensive: absent planet_mass/surface_temperature must not crash.

    The advisory uses ``config_params.get(...) or 0.0`` so missing keys
    cannot raise KeyError. With both keys absent, neither threshold can
    fire, so no advisory.
    """
    cp = dict(minimal_config)
    cp.pop('planet_mass', None)
    cp.pop('surface_temperature', None)
    sentinel = _patch_solvers_to_sentinel()
    with (
        caplog.at_level('INFO', logger='zalmoxis.solver'),
        patch('zalmoxis.solver._solve', return_value=sentinel),
    ):
        # Note: missing planet_mass would crash deeper in main(); we
        # patch _solve to short-circuit. The dispatch / advisory block
        # itself must not raise.
        main(cp, material_dictionaries={}, melting_curves_functions=None, input_dir='/tmp')
    msgs = [r.getMessage() for r in caplog.records if r.levelname == 'INFO']
    advisory = [m for m in msgs if "outer_solver='newton'" in m]
    assert advisory == []
