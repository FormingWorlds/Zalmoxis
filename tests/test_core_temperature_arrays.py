"""The core temperature table ``core_temperature_arrays``: T(P) on the core nodes, the mantle
source (``temperature_arrays`` or ``temperature_function``) elsewhere."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tests.test_core_two_phase import _params
from tests.test_jax_temperature_arrays import _INPUT_DIR, _chili_like_profile, _load_fixtures
from zalmoxis import solver as zs
from zalmoxis.config import _check_icb_rule, load_material_dictionaries
from zalmoxis.structure_model import check_core_temperature_arrays, core_temperature

STEP = 150.0  # K: the table at the CMB lies this far above the mantle T there
RTOL_LAYER = 1e-12  # the layer rule of structure_model.get_layer_mixture


def _table(t_cmb, p_lo=1.0e11, p_hi=4.5e11):
    """A 3-point core table [Pa, K], kinked so the end slopes differ, at T_cmb + STEP at p_lo."""
    p = np.array([p_lo, 0.5 * (p_lo + p_hi), p_hi])
    return p, t_cmb + STEP + np.array([0.0, 700.0, 1000.0])


def _solve(jax=True, table=None, **over):
    """1 M_E PALEOS iron core and PALEOS-2phase mantle with the r-indexed CHILI-like mantle T."""
    cp, mat, mf = _load_fixtures()
    cp.update(use_jax=jax, **over)
    r, t, _, t_cmb = _chili_like_profile()
    kw = {} if table is None else {'core_temperature_arrays': table(t_cmb)}
    res = zs.main(
        cp,
        mat,
        mf,
        _INPUT_DIR,
        temperature_function=lambda r_, p_: float(np.interp(r_, r, t)),
        temperature_arrays=(r, t),
        **kw,
    )
    assert res['converged'], res.get('accepted_on')
    return res, (r, t)


def _key(res):
    """R, P_c, P_cmb and rho_c of a result."""
    p_cmb = np.interp(res['cmb_mass'], res['mass_enclosed'], res['pressure'])
    return res['radii'][-1], res['pressure'][0], p_cmb, res['density'][0]


def _layers(res):
    m = np.asarray(res['mass_enclosed'])
    core = m < res['cmb_mass'] * (1.0 - RTOL_LAYER)
    return core, np.asarray(res['pressure'], float), np.asarray(res['temperature'], float)


@pytest.mark.unit
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
]


@pytest.mark.unit
@pytest.mark.parametrize('bad', _BAD, ids=range(len(_BAD)))
def test_a_table_of_the_wrong_form_is_refused(bad):
    with pytest.raises(ValueError, match='core_temperature_arrays'):
        check_core_temperature_arrays(bad)


_FLOAT = st.floats(1.0, 1e13, allow_nan=False)


@pytest.mark.unit
@settings(max_examples=200, deadline=None)
@given(
    p=st.lists(_FLOAT, min_size=2, max_size=8, unique=True).map(sorted),
    t=st.data(),
    where=st.sampled_from(['P', 'T']),
    bad=st.sampled_from([np.nan, np.inf, -np.inf, 0.0, -1.0, 'flip']),
)
def test_a_table_with_a_bad_value_is_refused_and_a_good_one_passes(p, t, where, bad):
    """NaN, inf, P or T <= 0 and P not strictly increasing are refused; the clean table passes."""
    temps = t.draw(st.lists(st.floats(1.0, 1e5), min_size=len(p), max_size=len(p)))
    good = (np.array(p), np.array(temps))
    got = check_core_temperature_arrays(good)
    assert all(np.array_equal(a, b) and a.dtype == float for a, b in zip(got, good))
    arr = [good[0].copy(), good[1].copy()]
    i = 0 if where == 'P' else 1
    if bad == 'flip':
        arr[0] = arr[0][::-1]
    else:
        arr[i][len(p) // 2] = bad
    with pytest.raises(ValueError, match='core_temperature_arrays'):
        check_core_temperature_arrays(tuple(arr))


def _no_solve(monkeypatch):
    def fail(*a, **k):
        pytest.fail('main started a solve on a refused input')

    monkeypatch.setattr(zs, '_solve', fail)


@pytest.mark.unit
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


@pytest.mark.unit
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


@pytest.mark.unit
@pytest.mark.parametrize('off', [None, 'core', 'mantle', 'not accepted'])
def test_an_accepted_node_off_its_temperature_source_fails_the_solve(off):
    """The invariant on an accepted result: core nodes carry table(P), mantle nodes the mantle
    source; 1e-6 off either fails, an unaccepted result is not checked. The nodes beyond the
    table are counted with their largest distance as a fraction of the table range."""
    from zalmoxis.invariants import StructureInvariantError

    table = (np.array([2.0, 4.0]), np.array([100.0, 200.0]))
    m, p, r = np.arange(6.0), np.array([5.0, 3.0, 1.5, 1.0, 0.5, 0.0]), np.arange(6.0) * 10.0
    t = np.where(m < 3.0, core_temperature(p, table), 7.0 * r)
    res = {
        'mass_enclosed': m,
        'pressure': p,
        'radii': r,
        'cmb_mass': 3.0,
        'temperature': t,
        'accepted_on': None if off == 'not accepted' else 'strict',
    }
    if off in ('core', 'mantle'):
        t[1 if off == 'core' else 4] *= 1.0 + 1e-6
        with pytest.raises(StructureInvariantError, match=f'a {off} node'):
            zs._core_table_keys(res, table, True, lambda r_, p_: 7.0 * r_)
        return
    out = zs._core_table_keys(res, table, True, lambda r_, p_: 7.0 * r_)
    assert out['core_T_used'] is True and out['core_T_nodes_outside'] == 2
    assert out['core_T_max_excess'] == pytest.approx((5.0 - 4.0) / (4.0 - 2.0), rel=1e-15)


@pytest.mark.integration
def test_no_table_gives_the_solve_of_today():
    """T0: core_temperature_arrays=None is the solve without the keyword, bit for bit."""
    a, _ = _solve()
    b, _ = _solve(table=lambda t_cmb: None)
    for k in ('radii', 'pressure', 'mass_enclosed', 'density', 'temperature', 'gravity'):
        assert np.array_equal(a[k], b[k]), k
    assert 'core_T_used' not in a and 'core_T_used' not in b


@pytest.mark.integration
@pytest.mark.parametrize('jax', [True, False], ids=['jax', 'numpy'])
def test_core_nodes_take_the_table_and_mantle_nodes_the_mantle_source(jax):
    """T1: a core table STEP K above the mantle T at the CMB; the returned core nodes carry
    table(P), the mantle nodes the mantle arrays, so the step shows at the CMB."""
    res, (r_arr, t_arr) = _solve(jax=jax, table=_table)
    core, p, t = _layers(res)
    table = _table(_chili_like_profile()[3])
    np.testing.assert_allclose(t[core], core_temperature(p[core], table), rtol=1e-9)
    r = np.asarray(res['radii'])
    np.testing.assert_allclose(t[~core], np.interp(r[~core], r_arr, t_arr), rtol=1e-9)
    top, first = np.flatnonzero(core)[-1], np.flatnonzero(~core)[0]
    assert t[top] - t[first] > 0.5 * STEP
    assert res['core_T_used'] is True
    assert res['core_T_nodes_outside'] == 0 and res['core_T_max_excess'] == 0.0


@pytest.mark.integration
def test_jax_and_numpy_agree_with_the_table():
    """T3: the same inputs on 2 integrators and 2 T codes give R, P_c, P_cmb and rho_c within
    1e-4, at an ODE tolerance (1e-8) below the integration difference of the 2 paths."""
    ode = {'relative_tolerance': 1e-8, 'absolute_tolerance': 1e-9}
    a, _ = _solve(jax=True, table=_table, **ode)
    b, _ = _solve(jax=False, table=_table, **ode)
    assert _key(a) == pytest.approx(_key(b), rel=1e-4)


@pytest.mark.integration
def test_core_nodes_outside_the_table_take_the_end_slope_and_are_counted():
    """T5: a table over the upper half of the core pressures only; the nodes below it are
    extended with the end slope, and the result counts them."""
    full, _ = _solve()
    core, p, _ = _layers(full)
    lo, hi = 0.5 * (p[core].min() + p[core].max()), p[core].max() * 1.01
    res, _ = _solve(table=lambda t_cmb: _table(t_cmb, lo, hi))
    core, p, t = _layers(res)
    table = _table(_chili_like_profile()[3], lo, hi)
    np.testing.assert_allclose(t[core], core_temperature(p[core], table), rtol=1e-9)
    out = core & ((p < lo) | (p > hi))
    excess = np.maximum(lo - p[out], p[out] - hi).max() / (hi - lo)
    assert out.sum() > 0 and res['core_T_nodes_outside'] == out.sum()
    assert res['core_T_max_excess'] == pytest.approx(excess, rel=1e-12)


@pytest.mark.integration
def test_a_core_without_a_t_dependent_eos_ignores_the_table():
    """The no-op: an Analytic iron core reads no T, so the table leaves the solve unchanged."""
    over = {'layer_eos_config': {'core': 'Analytic:iron', 'mantle': 'PALEOS-2phase:MgSiO3'}}
    a, _ = _solve(jax=False, **over)
    b, _ = _solve(jax=False, table=_table, **over)
    for k in ('radii', 'pressure', 'mass_enclosed', 'density', 'temperature'):
        assert np.array_equal(a[k], b[k]), k
    assert b['core_T_used'] is False


@pytest.mark.integration
def test_a_table_of_the_core_t_of_an_r_indexed_run_gives_its_structure():
    """T4: r-indexed arrays that cover the core as well, continuous at the CMB of the arrays;
    a table of the core (P, T) of that run adds no T change at the nodes, so the run with it
    has the same R, P_c, P_cmb and rho_c within 1e-4, at an ODE tolerance (1e-8) below the
    integration difference of the 2 runs."""
    r_m, t_m, r_cmb, t_cmb = _chili_like_profile()
    r_c = np.linspace(0.0, r_cmb, 40, endpoint=False)
    arrays = (
        np.concatenate([r_c, r_m]),
        np.concatenate([t_cmb + 900.0 * (1.0 - (r_c / r_cmb) ** 2), t_m]),
    )
    cp, mat, mf = _load_fixtures()
    cp.update(use_jax=True, relative_tolerance=1e-8, absolute_tolerance=1e-9)
    a = zs.main(cp, mat, mf, _INPUT_DIR, temperature_arrays=arrays)
    core, p, t = _layers(a)
    table = (p[core][::-1], t[core][::-1])
    b = zs.main(
        cp, mat, mf, _INPUT_DIR, temperature_arrays=arrays, core_temperature_arrays=table
    )
    assert a['converged'] and b['converged'] and b['core_T_used'] is True
    assert _key(b) == pytest.approx(_key(a), rel=1e-4)


@pytest.mark.integration
def test_the_numpy_fallback_of_a_jax_solve_keeps_the_table(monkeypatch):
    """T6: the first JAX structure solve raises, so main goes on with numpy, which keeps the
    core table: the T1 node rule holds on the result."""
    import zalmoxis.jax_eos.wrapper as jw

    real, raised = jw.solve_structure_via_jax, []

    def once(*args, **kwargs):
        if not raised:
            raised.append(1)
            raise ValueError('forced JAX failure')
        return real(*args, **kwargs)

    monkeypatch.setattr(jw, 'solve_structure_via_jax', once)
    res, (r_arr, t_arr) = _solve(jax=True, table=_table)
    assert raised and res['core_T_used'] is True
    core, p, t = _layers(res)
    np.testing.assert_allclose(
        t[core], core_temperature(p[core], _table(_chili_like_profile()[3])), rtol=1e-9
    )
    r = np.asarray(res['radii'])
    np.testing.assert_allclose(t[~core], np.interp(r[~core], r_arr, t_arr), rtol=1e-9)
