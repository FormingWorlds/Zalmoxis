"""End-to-end parity: solve_structure_via_jax matches numpy solve_structure.

Uses the exact same call signature as zalmoxis.structure_model.solve_structure,
so the wrapper is a drop-in replacement. Compares the full output arrays
(mass, gravity, pressure) on the radii grid.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_jax_rhs_parity import _stage1b_setup  # noqa: E402


@pytest.mark.unit
def test_solve_structure_via_jax_end_to_end_parity():
    """Wrapper produces (mass, gravity, pressure) matching numpy path."""
    from zalmoxis.jax_eos.wrapper import solve_structure_via_jax
    from zalmoxis.mixing import LayerMixture
    from zalmoxis.structure_model import solve_structure

    setup = _stage1b_setup()
    cfg = setup['config_params']
    T_logP_grid = setup['T_logP_grid']
    T_values = setup['T_values']

    layer_mixtures = {
        'core': LayerMixture([cfg['layer_eos_config']['core']], [1.0]),
        'mantle': LayerMixture([cfg['layer_eos_config']['mantle']], [1.0]),
    }

    def numpy_temp(r, P):
        if P <= 0:
            return 3000.0
        return float(np.interp(np.log10(max(P, 1.0)), T_logP_grid, T_values))

    M_planet = 5.972e24
    num_layers = 50
    radii = np.linspace(100.0, 6.37e6, num_layers)
    y0 = np.array([0.0, 0.0, 3.5e11])

    material_dicts = {
        cfg['layer_eos_config']['core']: setup['core_mat'],
        cfg['layer_eos_config']['mantle']: setup['mantle_mat'],
    }

    common_kwargs = dict(
        layer_mixtures=layer_mixtures,
        cmb_mass=setup['cmb_mass'],
        core_mantle_mass=setup['cmb_mass'] + 0.675 * M_planet,
        radii=radii,
        adaptive_radial_fraction=0.98,
        relative_tolerance=1e-5,
        absolute_tolerance=1e-6,
        maximum_step=6.37e6 * 0.004,
        material_dictionaries=material_dicts,
        interpolation_cache=setup['interp_cache'],
        y0=y0,
        solidus_func=setup['sol_func'],
        liquidus_func=setup['liq_func'],
        temperature_function=numpy_temp,
        mushy_zone_factors=setup['mushy_zone_factors'],
    )

    mass_np, g_np, P_np = solve_structure(**common_kwargs)
    mass_jx, g_jx, P_jx = solve_structure_via_jax(**common_kwargs)

    def rel(a, b):
        return np.abs(a - b) / np.maximum(np.abs(a), 1e-30)

    valid = (P_np > 0) & (P_jx > 0)
    M_drift = float(rel(mass_np[valid], mass_jx[valid]).max())
    g_drift = float(rel(g_np[valid], g_jx[valid]).max())
    P_drift = float(rel(P_np[valid], P_jx[valid]).max())
    print(f"end-to-end profile max drifts: M={M_drift:.3e} g={g_drift:.3e} P={P_drift:.3e}")
    print(f"outer: M rel={rel(mass_np[-1], mass_jx[-1]):.3e}  "
          f"g rel={rel(g_np[-1], g_jx[-1]):.3e}  "
          f"P rel={rel(P_np[-1], P_jx[-1]):.3e}")

    # Solver-tolerance bar
    assert M_drift <= 1e-3, f"mass drift {M_drift:.3e} > 1e-3"
    assert g_drift <= 1e-3, f"gravity drift {g_drift:.3e} > 1e-3"
    assert P_drift <= 1e-2, f"pressure drift {P_drift:.3e} > 1e-2"
