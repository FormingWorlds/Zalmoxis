"""The core temperature table ``core_temperature_arrays``: its form, its interpolation, the
refusals of ``main`` before a solve and the checks of an accepted result."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests._core_table_helpers import _table
from tests.test_core_two_phase import _params
from tests.test_jax_temperature_arrays import _INPUT_DIR, _chili_like_profile, _load_fixtures
from zalmoxis import solver as zs
from zalmoxis.config import _check_icb_rule, load_material_dictionaries
from zalmoxis.invariants import StructureInvariantError
from zalmoxis.structure_model import check_core_temperature_arrays, core_temperature

pytestmark = [pytest.mark.unit, pytest.mark.timeout(60)]


def test_core_temperature_is_linear_in_p_and_extends_with_the_end_slopes():
    """Inside the table np.interp; below and above it the slope of the end interval, the same
    numbers on numpy and JAX."""
    import jax.numpy as jnp

    table = (np.array([1.0, 2.0, 4.0]), np.array([10.0, 20.0, 30.0]))
    p = np.array([0.0, 1.0, 1.5, 3.0, 4.0, 6.0])
    want = np.array([0.0, 10.0, 15.0, 25.0, 30.0, 40.0])
    np.testing.assert_allclose(core_temperature(p, table), want, rtol=1e-14)
    got = core_temperature(jnp.asarray(p), tuple(map(jnp.asarray, table)), xp=jnp)
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-14)


_BAD = [
    True,
    1.0,
    'table',
    (np.array([1.0, 2.0]),),
    (np.array([1.0, 2.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0])),
    (np.array([[1.0, 2.0]]), np.array([[1.0, 2.0]])),
    (np.array([]), np.array([])),
    (np.array([1.0]), np.array([1.0])),
    (np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0])),
    (np.array([True, False]), np.array([1.0, 2.0])),
    (np.array([1.0, 2.0]), np.array([True, True])),
    (np.array(['1', '2']), np.array([1.0, 2.0])),
    (np.array([1.0, 2.0]), np.array([2.0, 1.0])),  # T falls with P
]


@pytest.mark.parametrize('bad', _BAD, ids=range(len(_BAD)))
def test_a_table_of_the_wrong_form_is_refused(bad):
    with pytest.raises(ValueError, match='core_temperature_arrays'):
        check_core_temperature_arrays(bad)


_FLOAT = st.floats(1.0, 1e13, allow_nan=False)


@settings(max_examples=200, deadline=None)
@given(
    p=st.lists(_FLOAT, min_size=2, max_size=8, unique=True).map(sorted),
    t=st.data(),
    where=st.sampled_from(['P', 'T']),
    bad=st.sampled_from([np.nan, np.inf, -np.inf, 0.0, -1.0, 'flip']),
)
def test_a_table_with_a_bad_value_is_refused_and_a_good_one_passes(p, t, where, bad):
    """NaN, inf, P or T <= 0, P not strictly increasing and T falling with P are refused; the
    clean table (T not falling) passes."""
    temps = sorted(t.draw(st.lists(st.floats(1.0, 1e5), min_size=len(p), max_size=len(p))))
    good = (np.array(p), np.array(temps))
    got = check_core_temperature_arrays(good)
    assert all(np.array_equal(a, b) and a.dtype == float for a, b in zip(got, good))
    arr = [good[0].copy(), good[1].copy()]
    i = 0 if where == 'P' else 1
    if bad == 'flip':
        arr[i] = arr[0][::-1] if i == 0 else np.linspace(2.0, 1.0, len(p))  # P or T falls
    else:
        arr[i][len(p) // 2] = bad
    with pytest.raises(ValueError, match='core_temperature_arrays'):
        check_core_temperature_arrays(tuple(arr))


def _no_solve(monkeypatch):
    def fail(*a, **k):
        pytest.fail('main started a solve on a refused input')

    monkeypatch.setattr(zs, '_solve', fail)


def test_main_refuses_a_bad_table_and_a_table_without_a_mantle_source(monkeypatch):
    """The refusals come before any solve; on numpy the mantle arrays alone set no T."""
    _no_solve(monkeypatch)
    cp, mat, mf = _load_fixtures()
    r, t, _, t_cmb = _chili_like_profile()
    with pytest.raises(ValueError, match='core_temperature_arrays'):
        zs.main(
            cp,
            mat,
            mf,
            _INPUT_DIR,
            temperature_arrays=(r, t),
            core_temperature_arrays=(np.array([2.0, 1.0]), np.array([1.0, 2.0])),
        )
    for kw in ({}, {'temperature_arrays': (r, t)}):
        with pytest.raises(ValueError, match='mantle temperature'):
            zs.main(cp, mat, mf, _INPUT_DIR, core_temperature_arrays=_table(t_cmb), **kw)


def test_the_liquidus_rule_with_the_cmb_anchored_adiabat_needs_the_core_table(monkeypatch):
    """Without a core table the CMB-anchored adiabat would give the core the surface T, so the
    liquidus rule is refused; with the table the core T is the table and the rule stands."""
    cfg = _params(temperature_mode='adiabatic_from_cmb', cmb_temperature=4000.0)
    with pytest.raises(ValueError, match='core_liquidus needs a core temperature'):
        _check_icb_rule(cfg)
    _check_icb_rule(cfg, core_table=True)
    _no_solve(monkeypatch)
    with pytest.raises(ValueError, match='core_liquidus needs a core temperature'):
        zs.main(
            cfg,
            load_material_dictionaries(),
            None,
            _INPUT_DIR,
            temperature_function=lambda r, p: 4000.0,
        )


_CASES = ['near', 'far', 'core', 'mantle', 'cold', 'not accepted', 'unused']


@pytest.mark.physics_invariant
@pytest.mark.parametrize('case', _CASES)
def test_the_checks_of_an_accepted_result_with_a_core_table(case, caplog):
    """On an accepted result core nodes carry table(P) and mantle nodes the mantle source; 1e-6
    off either fails, and a core node at T <= 0 fails with the table, the node pressure and the
    side of the table named. Nodes beyond the table are counted with their largest distance as
    a fraction of the table range; above 0.1 of it the result is no longer accepted. An
    unaccepted result is not checked; an unused table counts no node and logs no warning."""
    table = (np.array([2.0, 4.0]), np.array([100.0, 200.0]))
    m, r = np.arange(6.0), np.arange(6.0) * 10.0
    p = np.array([5.0 if case == 'far' else 4.1, 3.0, 1.9, 1.0, 0.5, 0.0])
    t = np.where(m < 3.0, core_temperature(p, table), 7.0 * r)
    res = {
        'mass_enclosed': m,
        'pressure': p,
        'radii': r,
        'cmb_mass': 3.0,
        'temperature': t,
        'converged': True,
        'accepted_on': None if case == 'not accepted' else 'strict',
    }
    used = case != 'unused'
    if case in ('core', 'mantle', 'cold'):
        t[1 if case == 'core' else 4] *= 1.0 + 1e-6
        if case == 'cold':
            t[2] = -1.0
        match = 'below the table range' if case == 'cold' else f'a {case} node'
        with pytest.raises(StructureInvariantError, match=match):
            zs._core_table_keys(res, table, used, lambda r_, p_: 7.0 * r_)
        return
    if case in ('not accepted', 'unused'):
        t[1] *= 2.0  # not checked
    out = zs._core_table_keys(res, table, used, lambda r_, p_: 7.0 * r_)
    assert out['core_T_used'] is used
    if not used:
        assert out['core_T_nodes_outside'] == 0 and out['core_T_max_excess'] == 0.0
        assert not caplog.records
        return
    assert out['core_T_nodes_outside'] == 2
    excess = (p[0] - 4.0) / (4.0 - 2.0)
    assert out['core_T_max_excess'] == pytest.approx(excess, rel=1e-12)
    far = case == 'far'
    assert out['converged'] is not far
    assert out['accepted_on'] == (None if far or case == 'not accepted' else 'strict')
    assert any('Not accepted' in x.getMessage() for x in caplog.records) is far
