"""The outer node of a structure whose integration stops just below the outer radius.

Whether a converged solve stops inside the last shell depends on round-off, so these solves
take each root of the central-pressure search, or the root lowered by the smallest of 1e-5 to
1e-2 that makes the outward integration stop below R (``_stop_below_r``); the outer node is
then padded as the surface on every platform. They also lift the 300 s wall clock, whose
accept returns a best pass that need not be padded.
"""

from __future__ import annotations

import numpy as np
import pytest

import zalmoxis.solver as zs
from tests.test_failed_structure_solve import _cfg, _run
from zalmoxis.config import load_material_dictionaries
from zalmoxis.constants import G
from zalmoxis.mixing import calculate_mixed_density, parse_all_layer_mixtures

pytestmark = pytest.mark.smoke


def _stop_below_r(monkeypatch, target=101325.0):
    """Take each root of the central-pressure search (the brentq call with xtol 1e6), or lower
    it until the residual is that of a padded solve; return the roots that no shift up to 1e-2
    pads."""
    real, misses = zs.brentq, []

    def brentq(f, a, b, *args, **kwargs):
        out = real(f, a, b, *args, **kwargs)
        if kwargs.get('xtol') != 1e6:
            return out
        root, info = out
        for k in (0.0, 1e-5, 1e-4, 1e-3, 1e-2):
            if f(root * (1.0 - k)) == -target:
                return root * (1.0 - k), info
        misses.append(root)
        return root, info

    monkeypatch.setattr(zs, 'brentq', brentq)
    return misses


def _assert_padded(P, misses):
    """Exactly the outer node is padded, by a forced stop."""
    assert not misses and np.count_nonzero(P == 0.0) == 1 and P[-1] == 0.0 and P[-2] > 0.0


@pytest.fixture(scope='module')
def padded():
    """An accepted solve whose outer node is padded."""
    with pytest.MonkeyPatch.context() as mp:
        misses = _stop_below_r(mp)
        r = _run(_cfg(wall_timeout=1e6))
    assert r['converged']
    _assert_padded(np.asarray(r['pressure']), misses)
    return r


@pytest.mark.timeout(2400)
def test_a_padded_surface_node_has_the_eos_density_at_the_target_pressure(padded):
    cfg = _cfg()
    mix = parse_all_layer_mixtures(cfg['layer_eos_config'])['mantle']
    ref = calculate_mixed_density(
        cfg['target_surface_pressure'],
        padded['temperature'][-1],
        mix,
        load_material_dictionaries(),
        None,
        None,
        {},
        None,
    )
    assert ref > 0
    assert padded['density'][-1] == pytest.approx(ref, rel=1e-8)


@pytest.mark.timeout(2400)
def test_a_padded_surface_node_has_the_gravity_of_the_planet_mass_at_its_radius(padded):
    m, r, g = (np.asarray(padded[k]) for k in ('mass_enclosed', 'radii', 'gravity'))
    assert g[-1] == pytest.approx(G * m[-1] / r[-1] ** 2, rel=1e-12)


@pytest.mark.timeout(2400)
def test_a_padded_node_of_a_cmb_anchored_adiabat_takes_the_top_of_the_adiabat(monkeypatch):
    """With the adiabat integrated from the CMB, the configured surface_temperature is not the
    temperature at the surface: the padded node takes the adiabat continued from the node
    below to the target pressure (checked against a 4000-step integration), and its EOS
    density there."""
    import os

    from tests.test_spider_coupling_convergence import _has_paleos_data
    from zalmoxis.config import load_solidus_liquidus_functions, load_zalmoxis_config

    if not _has_paleos_data():
        pytest.skip('PALEOS tables not on disk')
    root = os.path.normpath(os.path.join(os.path.dirname(zs.__file__), '..', '..'))
    cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    cfg.update(
        temperature_mode='adiabatic_from_cmb',
        cmb_temperature=4000.0,
        num_layers=50,
        wall_timeout=1e6,
        data_output_enabled=False,
        plotting_enabled=False,
    )
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        cfg['rock_solidus'],
        cfg['rock_liquidus'],
        cfg['mushy_zone_factor'],
    )
    md, spies = load_material_dictionaries(), {}
    for name in ('_surface_density', '_adiabat_to_surface'):
        real = getattr(zs, name)

        def spy(*args, _real=real, _name=name):
            spies[_name] = args
            return _real(*args)

        monkeypatch.setattr(zs, name, spy)
    misses = _stop_below_r(monkeypatch)
    r = zs.main(cfg, md, melt, os.path.join(root, 'input'))
    P, T, rho = (np.asarray(r[k], float) for k in ('pressure', 'temperature', 'density'))
    assert r['converged'] and r['adiabat_blend'] == 1.0
    _assert_padded(P, misses)
    # both lookups get the arguments of calculate_mixed_density_batch after the mixture
    eos = (md, *(melt or (None, None)), zs._interpolation_cache, *zs._eos_params(cfg))
    for name, i in (('_surface_density', 6), ('_adiabat_to_surface', 4)):
        assert len(spies[name][i]) == len(eos)
        assert all(x is y or x == y for x, y in zip(spies[name][i], eos)), name
    from zalmoxis.mixing import get_mixed_nabla_ad

    mixes = parse_all_layer_mixtures(cfg['layer_eos_config'])
    mzf, rho_min, rho_scale, bts = zs._eos_params(cfg)
    md, sol_liq, cache = load_material_dictionaries(), (melt or (None, None)), {}
    t_ref, lnp = T[-2], np.linspace(np.log(P[-2]), np.log(cfg['target_surface_pressure']), 4001)
    for lo, hi in zip(lnp[:-1], lnp[1:]):  # explicit Euler in ln P
        nab = get_mixed_nabla_ad(
            np.exp(lo),
            t_ref,
            mixes['mantle'],
            md,
            cache,
            *sol_liq,
            mzf,
            rho_min,
            rho_scale,
            bts,
        )
        t_ref *= np.exp(nab * (hi - lo))
    assert t_ref < T[-2] - 50.0
    assert T[-1] == pytest.approx(t_ref, abs=1.0)
    mix = parse_all_layer_mixtures(cfg['layer_eos_config'])['mantle']
    ref = calculate_mixed_density(
        cfg['target_surface_pressure'],
        T[-1],
        mix,
        load_material_dictionaries(),
        *(melt or (None, None)),
        {},
        cfg['mushy_zone_factor'],
    )
    assert rho[-1] == pytest.approx(ref, rel=1e-8)


@pytest.mark.timeout(2400)
# blend 0.75 (max_iterations_outer 4) returns a best pass that is not padded
@pytest.mark.parametrize('outer, blend', [(2, 0.25), (3, 0.5)])
def test_a_padded_node_of_a_cmb_run_accepted_mid_ramp_keeps_the_linear_profile_at_r(
    monkeypatch, outer, blend
):
    """An accept before the adiabat blend reaches 1 is not a CMB-anchored adiabat: the padded
    node keeps the linear profile at R (the configured surface_temperature), not a continuation
    of the blended temperature below it."""
    import os

    from tests.test_spider_coupling_convergence import _has_paleos_data
    from zalmoxis.config import load_solidus_liquidus_functions, load_zalmoxis_config

    if not _has_paleos_data():
        pytest.skip('PALEOS tables not on disk')
    root = os.path.normpath(os.path.join(os.path.dirname(zs.__file__), '..', '..'))
    cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    cfg.update(
        temperature_mode='adiabatic_from_cmb',
        cmb_temperature=4000.0,
        num_layers=50,
        max_iterations_outer=outer,
        wall_timeout=1e6,
        data_output_enabled=False,
        plotting_enabled=False,
    )
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        cfg['rock_solidus'],
        cfg['rock_liquidus'],
        cfg['mushy_zone_factor'],
    )
    misses = _stop_below_r(monkeypatch)
    r = zs.main(cfg, load_material_dictionaries(), melt, os.path.join(root, 'input'))
    P, T = (np.asarray(r[k], float) for k in ('pressure', 'temperature'))
    assert r['converged'] and r['adiabat_blend'] == blend and r['adiabat_computed']
    _assert_padded(P, misses)
    assert T[-1] == cfg['surface_temperature'] != T[-2]


@pytest.mark.timeout(2400)
def test_both_padded_lookups_get_the_eos_arguments_of_the_solve_in_order(monkeypatch):
    """On a PALEOS 2-phase mantle with its melting curves and condensed_rho_scale apart from
    binodal_T_scale, every entry of the argument tuple differs, so a swap of two of them in
    _solve breaks the comparison."""
    import os

    from tests.test_spider_coupling_convergence import _has_paleos_data
    from zalmoxis.config import load_solidus_liquidus_functions, load_zalmoxis_config

    if not _has_paleos_data():
        pytest.skip('PALEOS tables not on disk')
    root = os.path.normpath(os.path.join(os.path.dirname(zs.__file__), '..', '..'))
    cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    cfg['layer_eos_config'] = {**cfg['layer_eos_config'], 'mantle': 'PALEOS-2phase:MgSiO3'}
    cfg.update(
        temperature_mode='adiabatic_from_cmb',
        cmb_temperature=4000.0,
        num_layers=50,
        condensed_rho_scale=60.0,
        wall_timeout=1e6,
        data_output_enabled=False,
        plotting_enabled=False,
    )
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        cfg['rock_solidus'],
        cfg['rock_liquidus'],
        cfg['mushy_zone_factor'],
    )
    md, spies = load_material_dictionaries(), {}
    for name in ('_surface_density', '_adiabat_to_surface'):
        real = getattr(zs, name)

        def spy(*args, _real=real, _name=name):
            spies[_name] = args
            return _real(*args)

        monkeypatch.setattr(zs, name, spy)
    misses = _stop_below_r(monkeypatch)
    r = zs.main(cfg, md, melt, os.path.join(root, 'input'))
    eos = (md, *melt, zs._interpolation_cache, *zs._eos_params(cfg))
    assert r['converged'] and r['adiabat_blend'] == 1.0
    _assert_padded(np.asarray(r['pressure'], float), misses)
    assert melt[0] is not melt[1] and eos[-2] != eos[-1]
    for name, i in (('_surface_density', 6), ('_adiabat_to_surface', 4)):
        got = spies[name][i]
        assert len(got) == len(eos) and all(x is y for x, y in zip(got[:4], eos[:4])), name
        assert list(got[4:]) == list(eos[4:]), name


@pytest.mark.timeout(2400)
def test_a_padded_node_of_a_surface_anchored_adiabat_keeps_the_surface_temperature(
    monkeypatch,
):
    """The adiabat of input/default.toml is anchored at the surface, so its padded node keeps
    surface_temperature; only the CMB-anchored adiabat is continued to the target pressure."""
    import os

    from tests.test_spider_coupling_convergence import _has_paleos_data
    from zalmoxis.config import load_solidus_liquidus_functions, load_zalmoxis_config

    if not _has_paleos_data():
        pytest.skip('PALEOS tables not on disk')
    root = os.path.normpath(os.path.join(os.path.dirname(zs.__file__), '..', '..'))
    cfg = load_zalmoxis_config(os.path.join(root, 'input', 'default.toml'))
    cfg.update(
        num_layers=50, wall_timeout=1e6, data_output_enabled=False, plotting_enabled=False
    )
    melt = load_solidus_liquidus_functions(
        cfg['layer_eos_config'],
        cfg['rock_solidus'],
        cfg['rock_liquidus'],
        cfg['mushy_zone_factor'],
    )
    misses = _stop_below_r(monkeypatch)
    r = zs.main(cfg, load_material_dictionaries(), melt, os.path.join(root, 'input'))
    P, T = (np.asarray(r[k], float) for k in ('pressure', 'temperature'))
    assert cfg['temperature_mode'] == 'adiabatic' and r['adiabat_blend'] == 1.0
    assert r['converged'] and P[-2] > 1e8
    _assert_padded(P, misses)
    assert T[-1] == cfg['surface_temperature']
