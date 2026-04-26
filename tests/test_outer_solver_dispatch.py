"""Tests for the outer-solver dispatch knob (T2.1b).

Covers the plumbing only; the Newton iteration body itself (T2.1c) is
tested in ``test_outer_solver_newton.py`` once landed.

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
    with patch(
        'zalmoxis.solver._solve', return_value=sentinel
    ) as mock_solve, patch(
        'zalmoxis.solver._solve_newton_outer'
    ) as mock_newton:
        result = main(
            minimal_config, material_dictionaries={}, melting_curves_functions=None,
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
    with patch(
        'zalmoxis.solver._solve', return_value=sentinel
    ) as mock_solve, patch(
        'zalmoxis.solver._solve_newton_outer'
    ) as mock_newton:
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
    with patch(
        'zalmoxis.solver._solve_newton_outer', return_value=sentinel
    ) as mock_newton, patch(
        'zalmoxis.solver._solve'
    ) as mock_solve:
        result = main(
            cp, material_dictionaries={}, melting_curves_functions=None,
            input_dir='/tmp',
        )
    assert mock_newton.called, '_solve_newton_outer() must be called for newton path'
    assert not mock_solve.called, '_solve() must NOT be called for newton path'
    assert result is sentinel


def test_newton_stub_raises_not_implemented(minimal_config):
    """Until T2.1c lands, _solve_newton_outer() must fail loudly.

    Discriminating: NotImplementedError, not generic Exception. The
    error message must mention T2.1c so a reader can find the next
    step.
    """
    cp = dict(minimal_config)
    cp['outer_solver'] = 'newton'
    # Don't mock _solve_newton_outer here: we want the real stub to fire.
    with pytest.raises(NotImplementedError, match='T2.1c'):
        main(
            cp, material_dictionaries={}, melting_curves_functions=None,
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
