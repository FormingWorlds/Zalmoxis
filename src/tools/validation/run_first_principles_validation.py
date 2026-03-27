"""First-principles validation of the Zalmoxis structure solver.

Generates publication-quality verification plots:
1. Uniform-density sphere: numerical vs analytic M(r), g(r), P(r)
2. Two-layer sphere: numerical vs analytic M(r), g(r)
3. Earth benchmark: radial profiles with physical annotations
4. Grid convergence: ODE error vs N_layers
5. M-R scaling: log-log plot with Seager+2007 power law
6. CMF sweep: R(CMF) curve
7. Conservation diagnostics: Gauss residual and hydrostatic residual vs radius

Usage:
    python -m src.tools.run_first_principles_validation [--outdir DIR]

All plots saved as PDF to output_files/first_principles_validation/ by default.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np

# Ensure the package is importable
sys.path.insert(0, os.path.join(os.environ.get('ZALMOXIS_ROOT', '.'), 'src'))

from zalmoxis.constants import G, earth_mass, earth_radius  # noqa: E402
from zalmoxis.mixing import LayerMixture  # noqa: E402
from zalmoxis.structure_model import solve_structure  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Helpers (shared with test_first_principles.py)
# ============================================================================


def _make_constant_density_mock(rho):
    """Return a mock calculate_mixed_density that always returns rho."""
    def _mock(pressure, temperature, mixture, *args, **kwargs):
        if pressure <= 0 or np.isnan(pressure):
            return None
        return rho
    return _mock


def _make_two_layer_density_mock(rho_core, rho_mantle, cmb_mass):
    """Return a density mock dispatching on layer via component name."""
    def _mock(pressure, temperature, mixture, *args, **kwargs):
        if pressure <= 0 or np.isnan(pressure):
            return None
        if mixture.components[0] == 'mock:core':
            return rho_core
        return rho_mantle
    return _mock


def _solve_uniform_sphere(rho, R, P_center, num_layers=300):
    """Solve structure ODEs for uniform-density sphere."""
    radii = np.linspace(0, R, num_layers)
    layer_mixtures = {'core': LayerMixture(components=['mock:uniform'], fractions=[1.0])}
    mock_fn = _make_constant_density_mock(rho)

    with patch('zalmoxis.structure_model.calculate_mixed_density', mock_fn):
        mass, gravity, pressure = solve_structure(
            layer_mixtures=layer_mixtures,
            cmb_mass=1e30, core_mantle_mass=1e30,
            radii=radii, adaptive_radial_fraction=0.98,
            relative_tolerance=1e-10, absolute_tolerance=1e-12,
            maximum_step=R / 10, material_dictionaries={},
            interpolation_cache={}, y0=[0, 0, P_center],
            solidus_func=None, liquidus_func=None,
        )
    return radii, mass, gravity, pressure


def _analytic_uniform_sphere(rho, R, P_center, radii):
    """Exact analytic profiles for uniform-density sphere."""
    M = (4.0 / 3.0) * math.pi * rho * radii**3
    g = (4.0 / 3.0) * math.pi * G * rho * radii
    P = P_center - (2.0 / 3.0) * math.pi * G * rho**2 * radii**2
    return M, g, P


def _two_layer_central_pressure(rho_core, rho_mantle, R_cmb, R_total, M_cmb):
    """Compute exact P_center for a two-layer constant-density sphere."""
    n_fine = 10000
    r_fine = np.linspace(0, R_total, n_fine)
    rho_fine = np.where(r_fine <= R_cmb, rho_core, rho_mantle)
    M_fine = np.zeros(n_fine)
    for i in range(n_fine):
        r = r_fine[i]
        if r <= R_cmb:
            M_fine[i] = (4.0 / 3.0) * math.pi * rho_core * r**3
        else:
            M_fine[i] = M_cmb + (4.0 / 3.0) * math.pi * rho_mantle * (r**3 - R_cmb**3)
    g_fine = np.zeros(n_fine)
    g_fine[1:] = G * M_fine[1:] / r_fine[1:]**2
    return float(np.trapz(rho_fine * g_fine, r_fine))


def _run_analytic_eos_solver(mass_earth, cmf=0.325, mmf=0, num_layers=200,
                             relative_tolerance=1e-8, absolute_tolerance=1e-10):
    """Run full Zalmoxis solver with Analytic EOS."""
    from zalmoxis.zalmoxis import load_material_dictionaries, main

    layer_eos = {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'}
    config_params = {
        'planet_mass': mass_earth * earth_mass,
        'core_mass_fraction': cmf,
        'mantle_mass_fraction': mmf,
        'temperature_mode': 'isothermal',
        'surface_temperature': 300, 'center_temperature': 5000,
        'temp_profile_file': '',
        'layer_eos_config': layer_eos,
        'rock_solidus': 'Stixrude14-solidus',
        'rock_liquidus': 'Stixrude14-liquidus',
        'mushy_zone_factor': 1.0,
        'mushy_zone_factors': {'PALEOS:iron': 1.0, 'PALEOS:MgSiO3': 1.0, 'PALEOS:H2O': 1.0},
        'condensed_rho_min': 322.0, 'condensed_rho_scale': 50.0, 'binodal_T_scale': 50.0,
        'num_layers': num_layers,
        'max_iterations_outer': 100, 'tolerance_outer': 3e-3,
        'max_iterations_inner': 100, 'tolerance_inner': 1e-4,
        'relative_tolerance': relative_tolerance,
        'absolute_tolerance': absolute_tolerance,
        'maximum_step': 250000, 'adaptive_radial_fraction': 0.98,
        'max_center_pressure_guess': 10e12,
        'target_surface_pressure': 101325,
        'pressure_tolerance': 1e9,
        'max_iterations_pressure': 200,
        'data_output_enabled': False, 'plotting_enabled': False,
        'verbose': False, 'iteration_profiles_enabled': False,
    }

    input_dir = os.path.join(os.environ['ZALMOXIS_ROOT'], 'input')
    return main(config_params, load_material_dictionaries(), None, input_dir)


# ============================================================================
# Plot style (paper-ready: larger labels, thicker lines)
# ============================================================================

plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 17,
    'axes.titlesize': 18,
    'legend.fontsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'lines.linewidth': 2.5,
    'lines.markersize': 9,
    'figure.titlesize': 19,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.dpi': 150,
})


# ============================================================================
# Plot functions
# ============================================================================


def plot_uniform_sphere(outdir):
    """Plot 1: Uniform-density sphere numerical vs analytic."""
    logger.info('Plot 1: Uniform-density sphere')
    rho, R, P_c = 5000.0, 6.4e6, 3.6e11
    radii, mass, gravity, pressure = _solve_uniform_sphere(rho, R, P_c, 300)
    M_ex, g_ex, P_ex = _analytic_uniform_sphere(rho, R, P_c, radii)

    r_km = radii / 1e3

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Mass
    axes[0].plot(r_km, mass, 'b-', label='Numerical', linewidth=2)
    axes[0].plot(r_km, M_ex, 'r--', label='Analytic', linewidth=2)
    axes[0].set_xlabel('Radius [km]')
    axes[0].set_ylabel('Enclosed mass [kg]')
    axes[0].set_title('(a) M(r)')
    axes[0].legend()

    # Gravity
    axes[1].plot(r_km, gravity, 'b-', label='Numerical', linewidth=2)
    axes[1].plot(r_km, g_ex, 'r--', label='Analytic', linewidth=2)
    axes[1].set_xlabel('Radius [km]')
    axes[1].set_ylabel('Gravity [m/s$^2$]')
    axes[1].set_title('(b) g(r)')
    axes[1].legend()

    # Pressure
    valid = pressure > 0
    axes[2].plot(r_km[valid], pressure[valid] / 1e9, 'b-', label='Numerical', linewidth=2)
    axes[2].plot(r_km[valid], P_ex[valid] / 1e9, 'r--', label='Analytic', linewidth=2)
    axes[2].set_xlabel('Radius [km]')
    axes[2].set_ylabel('Pressure [GPa]')
    axes[2].set_title('(c) P(r)')
    axes[2].legend()

    fig.suptitle(r'Uniform-density sphere ($\rho$ = 5000 kg/m$^3$)')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'uniform_sphere_profiles.pdf'))
    plt.close(fig)
    logger.info('  Saved uniform_sphere_profiles.pdf')


def plot_two_layer_sphere(outdir):
    """Plot 2: Two-layer sphere numerical vs analytic."""
    logger.info('Plot 2: Two-layer sphere')
    rho_c, rho_m = 13000.0, 4000.0
    M_total = earth_mass
    CMF = 0.325
    cmb_mass = CMF * M_total
    R_cmb = ((3 * cmb_mass) / (4 * math.pi * rho_c)) ** (1.0 / 3.0)
    M_mantle = M_total - cmb_mass
    R_total = (R_cmb**3 + (3 * M_mantle) / (4 * math.pi * rho_m)) ** (1.0 / 3.0)
    P_center = _two_layer_central_pressure(rho_c, rho_m, R_cmb, R_total, cmb_mass) * 1.05

    N = 400
    radii = np.linspace(0, R_total, N)
    core_mix = LayerMixture(components=['mock:core'], fractions=[1.0])
    mantle_mix = LayerMixture(components=['mock:mantle'], fractions=[1.0])
    mock_fn = _make_two_layer_density_mock(rho_c, rho_m, cmb_mass)

    with patch('zalmoxis.structure_model.calculate_mixed_density', mock_fn):
        mass, gravity, pressure = solve_structure(
            layer_mixtures={'core': core_mix, 'mantle': mantle_mix},
            cmb_mass=cmb_mass, core_mantle_mass=M_total,
            radii=radii, adaptive_radial_fraction=0.98,
            relative_tolerance=1e-10, absolute_tolerance=1e-12,
            maximum_step=R_total / 10, material_dictionaries={},
            interpolation_cache={}, y0=[0, 0, P_center],
            solidus_func=None, liquidus_func=None,
        )

    # Analytic
    M_ex = np.zeros(N)
    for i, r in enumerate(radii):
        if r <= R_cmb:
            M_ex[i] = (4.0 / 3.0) * math.pi * rho_c * r**3
        else:
            M_ex[i] = cmb_mass + (4.0 / 3.0) * math.pi * rho_m * (r**3 - R_cmb**3)

    r_km = radii / 1e3

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    axes[0].plot(r_km, mass, 'b-', label='Numerical', linewidth=2)
    axes[0].plot(r_km, M_ex, 'r--', label='Analytic', linewidth=2)
    axes[0].axvline(R_cmb / 1e3, color='gray', linestyle=':', label='CMB')
    axes[0].set_xlabel('Radius [km]')
    axes[0].set_ylabel('Enclosed mass [kg]')
    axes[0].set_title('(a) M(r)')
    axes[0].legend()

    g_gauss = np.zeros(N)
    g_gauss[1:] = G * mass[1:] / radii[1:]**2
    axes[1].plot(r_km, gravity, 'b-', label='Numerical g(r)', linewidth=2)
    axes[1].plot(r_km[1:], g_gauss[1:], 'r--', label='G M(r) / r$^2$', linewidth=2)
    axes[1].axvline(R_cmb / 1e3, color='gray', linestyle=':', label='CMB')
    axes[1].set_xlabel('Radius [km]')
    axes[1].set_ylabel('Gravity [m/s$^2$]')
    axes[1].set_title('(b) Gauss law check')
    axes[1].legend()

    fig.suptitle(
        r'Two-layer sphere ($\rho_c$ = 13000, $\rho_m$ = 4000 kg/m$^3$, CMF = 0.325)',
    )
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'two_layer_sphere.pdf'))
    plt.close(fig)
    logger.info('  Saved two_layer_sphere.pdf')


def plot_earth_benchmark(outdir):
    """Plot 3: Earth benchmark radial profiles."""
    logger.info('Plot 3: Earth benchmark (1 M_Earth, Analytic EOS)')
    results = _run_analytic_eos_solver(1.0, cmf=0.325)

    r_km = results['radii'] / 1e3
    P = results['pressure']
    valid = P > 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    axes[0, 0].plot(r_km[valid], results['density'][valid], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Radius [km]')
    axes[0, 0].set_ylabel(r'Density [kg/m$^3$]')
    axes[0, 0].set_title(r'(a) $\rho$(r)')

    axes[0, 1].plot(r_km[valid], P[valid] / 1e9, 'b-', linewidth=2)
    axes[0, 1].set_xlabel('Radius [km]')
    axes[0, 1].set_ylabel('Pressure [GPa]')
    axes[0, 1].set_title('(b) P(r)')

    axes[1, 0].plot(r_km[valid], results['gravity'][valid], 'b-', linewidth=2)
    axes[1, 0].axhline(9.81, color='gray', linestyle=':', label='9.81 m/s$^2$')
    axes[1, 0].set_xlabel('Radius [km]')
    axes[1, 0].set_ylabel('Gravity [m/s$^2$]')
    axes[1, 0].set_title('(c) g(r)')
    axes[1, 0].legend()

    axes[1, 1].plot(r_km[valid], results['mass_enclosed'][valid] / earth_mass, 'b-', linewidth=2)
    axes[1, 1].axhline(1.0, color='gray', linestyle=':', label=r'1 $M_\oplus$')
    axes[1, 1].set_xlabel('Radius [km]')
    axes[1, 1].set_ylabel(r'Enclosed mass [$M_\oplus$]')
    axes[1, 1].set_title('(d) M(r)')
    axes[1, 1].legend()

    R_planet_km = results['radii'][-1] / 1e3
    P_c_GPa = results['pressure'][0] / 1e9
    fig.suptitle(
        f'Earth benchmark: R = {R_planet_km:.0f} km '
        f'({results["radii"][-1] / earth_radius:.3f} $R_\\oplus$), '
        f'$P_c$ = {P_c_GPa:.0f} GPa',
    )
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'earth_benchmark.pdf'))
    plt.close(fig)
    logger.info(f'  Saved earth_benchmark.pdf (R = {R_planet_km:.0f} km, P_c = {P_c_GPa:.0f} GPa)')


def plot_grid_convergence(outdir):
    """Plot 4: ODE error vs grid resolution."""
    logger.info('Plot 4: Grid convergence')
    rho, R, P_c = 5000.0, 6.4e6, 3.6e11
    resolutions = [25, 50, 100, 200, 400, 800]
    mass_errors = []
    pressure_errors = []

    for n in resolutions:
        radii, mass, _, pressure = _solve_uniform_sphere(rho, R, P_c, n)
        M_ex, _, P_ex = _analytic_uniform_sphere(rho, R, P_c, radii)
        valid = pressure > 0
        mass_errors.append(np.max(np.abs(mass[1:] - M_ex[1:]) / M_ex[1:]))
        pressure_errors.append(np.max(np.abs(pressure[valid] - P_ex[valid]) / P_ex[valid]))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(resolutions, mass_errors, 'bo-', label='Mass error', linewidth=2)
    ax.loglog(resolutions, pressure_errors, 'rs-', label='Pressure error', linewidth=2)
    ax.set_xlabel('Number of radial grid points')
    ax.set_ylabel('Max relative error')
    ax.set_title('Grid convergence (uniform-density sphere, rtol=1e-10)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'grid_convergence.pdf'))
    plt.close(fig)
    logger.info('  Saved grid_convergence.pdf')


def plot_mr_scaling(outdir):
    """Plot 5: Mass-radius scaling with Seager+2007 power law."""
    logger.info('Plot 5: Mass-radius scaling')
    masses = [0.3, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
    radii_earth = []

    for m in masses:
        results = _run_analytic_eos_solver(m, cmf=0.325)
        if not results['converged']:
            logger.warning(f'  {m} M_Earth did not converge, skipping')
            continue
        radii_earth.append(results['radii'][-1] / earth_radius)

    masses_arr = np.array(masses[:len(radii_earth)])
    radii_arr = np.array(radii_earth)

    # Fit power law
    log_m = np.log10(masses_arr)
    log_r = np.log10(radii_arr)
    coeffs = np.polyfit(log_m, log_r, 1)
    alpha = coeffs[0]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.loglog(masses_arr, radii_arr, 'bo', markersize=8, label='Zalmoxis (Analytic EOS)')

    m_fit = np.logspace(np.log10(0.3), np.log10(10), 50)
    r_fit = 10**np.polyval(coeffs, np.log10(m_fit))
    ax.loglog(m_fit, r_fit, 'r--', label=f'Fit: R $\\propto$ M$^{{{alpha:.3f}}}$', linewidth=2)

    ax.set_xlabel(r'Mass [$M_\oplus$]')
    ax.set_ylabel(r'Radius [$R_\oplus$]')
    ax.set_title('Mass-radius scaling (CMF = 0.325, Seager+2007 analytic EOS)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'mr_scaling.pdf'))
    plt.close(fig)
    logger.info(f'  Saved mr_scaling.pdf (alpha = {alpha:.4f})')


def plot_cmf_sweep(outdir):
    """Plot 6: Radius vs core mass fraction."""
    logger.info('Plot 6: CMF sweep')
    cmfs = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    radii_earth = []
    valid_cmfs = []

    for cmf in cmfs:
        try:
            results = _run_analytic_eos_solver(1.0, cmf=cmf)
            if results['converged']:
                radii_earth.append(results['radii'][-1] / earth_radius)
                valid_cmfs.append(cmf)
        except Exception as e:
            logger.warning(f'  CMF={cmf} failed: {e}')

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(valid_cmfs, radii_earth, 'bo-', markersize=6, linewidth=2)
    ax.set_xlabel('Core mass fraction')
    ax.set_ylabel(r'Radius [$R_\oplus$]')
    ax.set_title(r'1 $M_\oplus$ planet: radius vs core mass fraction')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'cmf_sweep.pdf'))
    plt.close(fig)
    logger.info('  Saved cmf_sweep.pdf')


def plot_conservation_diagnostics(outdir):
    """Plot 7: Gauss residual and hydrostatic residual vs radius."""
    logger.info('Plot 7: Conservation diagnostics')
    rho, R, P_c = 5000.0, 6.4e6, 3.6e11
    N = 500
    radii, mass, gravity, pressure = _solve_uniform_sphere(rho, R, P_c, N)

    r_km = radii / 1e3
    valid = pressure > 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Gauss residual: g - GM/r^2
    idx = np.where(valid)[0]
    idx = idx[idx > 2]  # skip near-origin
    g_gauss = G * mass[idx] / radii[idx]**2
    gauss_residual = np.abs(gravity[idx] - g_gauss) / g_gauss
    axes[0].semilogy(r_km[idx], gauss_residual, 'b-', linewidth=2)
    axes[0].set_xlabel('Radius [km]')
    axes[0].set_ylabel('Relative residual |g - GM/r$^2$| / (GM/r$^2$)')
    axes[0].set_title("(a) Gauss's law residual")
    axes[0].grid(True, alpha=0.3)

    # Hydrostatic residual: dP/dr + rho*g = 0
    interior = idx[(idx > 1) & (idx < len(radii) - 1)]
    dPdr = (pressure[interior + 1] - pressure[interior - 1]) / (
        radii[interior + 1] - radii[interior - 1]
    )
    hydro_residual = dPdr + rho * gravity[interior]
    P_scale = (2.0 / 3.0) * math.pi * G * rho**2 * R
    rel_hydro = np.abs(hydro_residual) / P_scale

    axes[1].semilogy(r_km[interior], rel_hydro, 'b-', linewidth=2)
    axes[1].set_xlabel('Radius [km]')
    axes[1].set_ylabel('Relative residual |dP/dr + $\\rho$g| / P$_{\\rm scale}$')
    axes[1].set_title('(b) Hydrostatic balance residual')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(r'Conservation diagnostics (uniform sphere, $\rho$ = 5000 kg/m$^3$)')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'conservation_diagnostics.pdf'))
    plt.close(fig)
    logger.info('  Saved conservation_diagnostics.pdf')


# ============================================================================
# Main
# ============================================================================


def main():
    """Run all validation plots."""
    parser = argparse.ArgumentParser(description='Zalmoxis first-principles validation')
    parser.add_argument(
        '--outdir', default=None,
        help='Output directory for plots (default: output_files/first_principles_validation/)',
    )
    args = parser.parse_args()

    if args.outdir:
        outdir = args.outdir
    else:
        root = os.environ.get('ZALMOXIS_ROOT', '.')
        outdir = os.path.join(root, 'output_files', 'first_principles_validation')

    os.makedirs(outdir, exist_ok=True)
    logger.info(f'Output directory: {outdir}')

    # Tier 1: Pure-math ODE verification (fast, <1 s)
    plot_uniform_sphere(outdir)
    plot_two_layer_sphere(outdir)
    plot_conservation_diagnostics(outdir)
    plot_grid_convergence(outdir)

    # Tier 2: Full solver benchmarks (slower, ~3 min total)
    plot_earth_benchmark(outdir)
    plot_mr_scaling(outdir)
    plot_cmf_sweep(outdir)

    logger.info(f'All plots saved to {outdir}')


if __name__ == '__main__':
    main()
