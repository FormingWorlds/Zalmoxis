"""The JAX and numpy structure solves report the same stop radius.

A solve whose pressure reaches zero before R fills its ``stop`` dict with the stop radius and
the stop pressure, zero at the terminal event. The JAX event root finder works to 1e-3 m in radius, so on a profile where
both right-hand sides agree, the two stop radii agree within 1 m at rtol 1e-10. The profile is
the PALEOS setup of ``test_jax_rhs_parity`` with T at 0.6 of its values, below the per-cell
clamp region, where numpy and JAX differ by about 1e6 Pa near the surface.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_jax_rhs_parity import _stage1b_setup  # noqa: E402

pytestmark = [pytest.mark.smoke, pytest.mark.timeout(120)]

M_EARTH = 5.972e24
RHO_G_SURFACE = 2e4  # rho g near the stop of this profile [Pa/m]


@pytest.fixture(scope='module')
def stopped_solves():
    """numpy and JAX solves of one padded profile, with their stop dicts."""
    from zalmoxis.jax_eos.wrapper import solve_structure_via_jax
    from zalmoxis.mixing import LayerMixture
    from zalmoxis.structure_model import solve_structure

    setup = _stage1b_setup()
    eos = setup['config_params']['layer_eos_config']
    t_grid, t_values = setup['T_axis_grid'], 0.6 * np.asarray(setup['T_values'])

    def temperature(r, p):
        return 1800.0 if p <= 0 else float(np.interp(np.log10(max(p, 1.0)), t_grid, t_values))

    radii = np.linspace(100.0, 6.37e6, 50)
    kwargs = dict(
        layer_mixtures={k: LayerMixture([eos[k]], [1.0]) for k in ('core', 'mantle')},
        cmb_mass=setup['cmb_mass'],
        core_mantle_mass=setup['cmb_mass'] + 0.675 * M_EARTH,
        radii=radii,
        adaptive_radial_fraction=0.98,
        relative_tolerance=1e-10,
        absolute_tolerance=1e-11,
        maximum_step=6.37e6 * 0.004,
        material_dictionaries={
            eos['core']: setup['core_mat'],
            eos['mantle']: setup['mantle_mat'],
        },
        interpolation_cache=setup['interp_cache'],
        y0=np.array([0.0, 0.0, 2.5e11]),
        solidus_func=setup['sol_func'],
        liquidus_func=setup['liq_func'],
        temperature_function=temperature,
        mushy_zone_factors=setup['mushy_zone_factors'],
    )
    stop_np, stop_jx = {}, {}
    p_np = solve_structure(**kwargs, stop=stop_np)[2]
    p_jx = solve_structure_via_jax(**kwargs, stop=stop_jx)[2]
    return radii, np.asarray(p_np), np.asarray(p_jx), stop_np, stop_jx


@pytest.mark.physics_invariant
@pytest.mark.reference_pinned
def test_the_numpy_and_jax_stop_radii_agree_within_a_metre(stopped_solves):
    """Both solves stop in the same shell and report stop radii within 1 m of each other.

    Reference: the numpy solve (cross-code check); scipy finds its event root to 4 eps.
    """
    radii, p_np, p_jx, stop_np, stop_jx = stopped_solves
    n = int(np.argmax(p_np == 0.0))
    assert p_np[-1] == 0.0 and n > 0, 'the numpy solve reaches R'
    assert np.array_equal(p_np == 0.0, p_jx == 0.0), 'the solves stop in different shells'
    both = (p_np > 0) & (p_jx > 0)
    dp = np.abs(p_np - p_jx)[both].max()
    assert dp < 0.5 * RHO_G_SURFACE, (
        f'the profiles differ by {dp:.3e} Pa, more than 0.5 m of rho g'
    )
    assert set(stop_np) == set(stop_jx) == {'r', 'p'}
    for stop in (stop_np, stop_jx):
        assert radii[n - 1] < stop['r'] <= radii[n]
    assert abs(stop_np['r'] - stop_jx['r']) <= 1.0, (
        f'{stop_np["r"]:.4f} m vs {stop_jx["r"]:.4f} m'
    )
    assert stop_np['p'] == stop_jx['p'] == 0.0
