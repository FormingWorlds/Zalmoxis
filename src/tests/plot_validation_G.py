"""Generate validation plots G6-G8 for the Zalmoxis adiabatic temperature PR.

Produces three plots:
    Plot G6 -- Mass-radius diagram: Zalmoxis (Seager2007 EOS) vs Zeng+2019
               for 1M rocky, 5M rocky, and 1M 50% H2O planets.
    Plot G7 -- T-dependent EOS convergence: density profiles vs pressure
               for 1M, 5M, 7M using WolfBower2018:MgSiO3.
    Plot G8 -- Adiabatic T(r) profiles: 1M and 5M with solidus/liquidus
               overlaid, demonstrating the new adiabatic temperature mode.

Saves all plots as PNG to src/tests/output/plots/.

Usage:
    cd /path/to/Zalmoxis
    ZALMOXIS_ROOT=$PWD python src/tests/plot_validation_G.py
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

ZALMOXIS_ROOT = os.environ.get('ZALMOXIS_ROOT')
if not ZALMOXIS_ROOT:
    # Auto-detect if running from inside the repo
    candidate = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if os.path.isfile(os.path.join(candidate, 'pyproject.toml')):
        os.environ['ZALMOXIS_ROOT'] = candidate
        ZALMOXIS_ROOT = candidate
    else:
        sys.exit('ZALMOXIS_ROOT environment variable not set')

# Ensure src/ is on the path so that both `zalmoxis` and `tools` are importable
src_dir = os.path.join(ZALMOXIS_ROOT, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from zalmoxis import zalmoxis as zal  # noqa: E402
from zalmoxis.constants import earth_mass, earth_radius  # noqa: E402
from zalmoxis.eos_functions import (  # noqa: E402
    get_solidus_liquidus_functions,
)
from zalmoxis.zalmoxis import (  # noqa: E402
    load_material_dictionaries,
    load_solidus_liquidus_functions,
)

OUTPUT_DIR = os.path.join(ZALMOXIS_ROOT, 'src', 'tests', 'output', 'plots')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Solver helper
# ---------------------------------------------------------------------------


def run_model(mass_earth, layer_eos, cmf=0.325, immf=0, temperature_mode='linear'):
    """Run Zalmoxis for a given mass and EOS configuration.

    Parameters
    ----------
    mass_earth : float
        Planet mass in Earth masses.
    layer_eos : dict
        Per-layer EOS config.
    cmf : float
        Core mass fraction.
    immf : float
        Inner mantle mass fraction (for water worlds).
    temperature_mode : str
        Temperature mode: 'linear', 'isothermal', 'adiabatic'.

    Returns
    -------
    dict
        Model results dictionary from zalmoxis.main().
    """
    default_config_path = os.path.join(ZALMOXIS_ROOT, 'input', 'default.toml')
    config_params = zal.load_zalmoxis_config(default_config_path)
    config_params['planet_mass'] = mass_earth * earth_mass
    config_params['core_mass_fraction'] = cmf
    config_params['mantle_mass_fraction'] = immf
    config_params['layer_eos_config'] = layer_eos
    config_params['temperature_mode'] = temperature_mode
    config_params['verbose'] = False
    config_params['plotting_enabled'] = False
    config_params['data_output_enabled'] = False

    melting = load_solidus_liquidus_functions(layer_eos)
    model_results = zal.main(
        config_params,
        material_dictionaries=load_material_dictionaries(),
        melting_curves_functions=melting,
        input_dir=os.path.join(ZALMOXIS_ROOT, 'input'),
    )
    return model_results


# ---------------------------------------------------------------------------
# Reference data loaders
# ---------------------------------------------------------------------------


def load_zeng_curve(filename):
    """Load a Zeng+2019 mass-radius curve.

    Parameters
    ----------
    filename : str
        Name of the file in the mass_radius_curves or comparison data directory.

    Returns
    -------
    tuple
        (masses, radii) as numpy arrays in Earth units.
    """
    # Try the main curves directory first, then the comparison data directory
    for subdir in ['mass_radius_curves', 'mass_radius_comparison_data/Zeng+2019_M-R_models']:
        path = os.path.join(ZALMOXIS_ROOT, 'data', subdir, filename)
        if os.path.isfile(path):
            break
    else:
        raise FileNotFoundError(f'Zeng curve file not found: {filename}')

    masses, radii = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            m, r = map(float, line.split())
            masses.append(m)
            radii.append(r)
    return np.array(masses), np.array(radii)


# ---------------------------------------------------------------------------
# Plot G6: Mass-radius diagram
# ---------------------------------------------------------------------------


def plot_G6():
    """Mass-radius diagram for rocky and water-world planets vs Zeng+2019."""
    print('Plot G6: Mass-radius diagram')

    # Cases: G1 = 1M rocky, G2 = 5M rocky, G3 = 1M 50% water
    cases = [
        {
            'label': '1 $M_\\oplus$ rocky',
            'mass': 1,
            'eos': {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'},
            'cmf': 0.325,
            'immf': 0,
        },
        {
            'label': '5 $M_\\oplus$ rocky',
            'mass': 5,
            'eos': {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'},
            'cmf': 0.325,
            'immf': 0,
        },
        {
            'label': '1 $M_\\oplus$ 50% H$_2$O',
            'mass': 1,
            'eos': {
                'core': 'Seager2007:iron',
                'mantle': 'Seager2007:MgSiO3',
                'ice_layer': 'Seager2007:H2O',
            },
            'cmf': 0.1625,
            'immf': 0.3375,
        },
    ]

    # Run an extended mass range for the M-R curve context
    rocky_masses = [0.5, 1, 2, 3, 5, 7, 10]
    water_masses = [0.5, 1, 2, 3, 5]

    # Collect Zalmoxis M-R points for rocky planets
    print('  Running rocky M-R sweep...')
    rocky_results = []
    for m in rocky_masses:
        print(f'    M = {m} M_earth...')
        res = run_model(
            m, {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'}, cmf=0.325
        )
        r_earth = res['radii'][-1] / earth_radius
        m_earth = res['mass_enclosed'][-1] / earth_mass
        rocky_results.append((m_earth, r_earth))
        if not res.get('converged', False):
            print('      WARNING: did not converge')

    # Collect Zalmoxis M-R points for water worlds
    print('  Running water M-R sweep...')
    water_results = []
    for m in water_masses:
        print(f'    M = {m} M_earth...')
        res = run_model(
            m,
            {
                'core': 'Seager2007:iron',
                'mantle': 'Seager2007:MgSiO3',
                'ice_layer': 'Seager2007:H2O',
            },
            cmf=0.1625,
            immf=0.3375,
        )
        r_earth = res['radii'][-1] / earth_radius
        m_earth = res['mass_enclosed'][-1] / earth_mass
        water_results.append((m_earth, r_earth))
        if not res.get('converged', False):
            print('      WARNING: did not converge')

    # Load Zeng+2019 reference curves
    zeng_rocky_m, zeng_rocky_r = load_zeng_curve('massradiusEarthlikeRocky.txt')
    zeng_water_m, zeng_water_r = load_zeng_curve('massradius_50percentH2O_300K_1mbar.txt')

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Zeng reference curves
    mask_r = (zeng_rocky_m >= 0.3) & (zeng_rocky_m <= 12)
    ax.plot(
        zeng_rocky_m[mask_r],
        zeng_rocky_r[mask_r],
        'k-',
        linewidth=2,
        alpha=0.4,
        label='Zeng+2019 Earth-like rocky',
    )
    mask_w = (zeng_water_m >= 0.3) & (zeng_water_m <= 12)
    ax.plot(
        zeng_water_m[mask_w],
        zeng_water_r[mask_w],
        'b-',
        linewidth=2,
        alpha=0.4,
        label='Zeng+2019 50% H$_2$O',
    )

    # Zalmoxis points
    rm, rr = zip(*rocky_results)
    ax.plot(
        rm,
        rr,
        'ko--',
        linewidth=1.5,
        markersize=6,
        markeredgecolor='black',
        markerfacecolor='#2ca02c',
        label='Zalmoxis rocky (Seager2007)',
        zorder=5,
    )
    wm, wr = zip(*water_results)
    ax.plot(
        wm,
        wr,
        'bs--',
        linewidth=1.5,
        markersize=6,
        markeredgecolor='black',
        markerfacecolor='#1f77b4',
        label='Zalmoxis 50% H$_2$O (Seager2007)',
        zorder=5,
    )

    # Highlight the specific G1-G3 cases
    for case in cases:
        m = case['mass']
        res = run_model(m, case['eos'], cmf=case['cmf'], immf=case['immf'])
        r_e = res['radii'][-1] / earth_radius
        m_e = res['mass_enclosed'][-1] / earth_mass
        ax.plot(
            m_e,
            r_e,
            '*',
            markersize=14,
            color='red',
            markeredgecolor='black',
            markeredgewidth=0.5,
            zorder=10,
        )
        ax.annotate(
            case['label'],
            (m_e, r_e),
            textcoords='offset points',
            xytext=(8, 8),
            fontsize=8,
            color='red',
        )

    ax.set_xlabel('Mass ($M_\\oplus$)', fontsize=13)
    ax.set_ylabel('Radius ($R_\\oplus$)', fontsize=13)
    ax.set_title('G6: Mass-Radius Diagram (Seager2007 EOS vs Zeng+2019)', fontsize=13)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0.3, 11)
    ax.set_ylim(0.6, 2.5)
    ax.tick_params(labelsize=11)

    path = os.path.join(OUTPUT_DIR, 'plot_G6_mass_radius.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


# ---------------------------------------------------------------------------
# Plot G7: T-dependent EOS convergence (density vs pressure)
# ---------------------------------------------------------------------------


def plot_G7():
    """Density profiles vs pressure for WolfBower2018:MgSiO3 at 1M, 5M, 7M."""
    print('Plot G7: T-dependent EOS convergence')

    masses = [1, 5, 7]
    colors = ['#2ca02c', '#d62728', '#1f77b4']
    eos_wb = {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'}
    eos_s07 = {'core': 'Seager2007:iron', 'mantle': 'Seager2007:MgSiO3'}

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)

    for i, (mass, color) in enumerate(zip(masses, colors)):
        ax = axes[i]
        print(f'  Running {mass} M_earth (WolfBower2018)...')
        res_wb = run_model(mass, eos_wb, cmf=0.325)
        print(f'  Running {mass} M_earth (Seager2007)...')
        res_s07 = run_model(mass, eos_s07, cmf=0.325)

        # Extract profiles
        P_wb = np.array(res_wb['pressure']) / 1e9  # GPa
        rho_wb = np.array(res_wb['density'])
        m_wb = np.array(res_wb['mass_enclosed'])
        cmb_mass_wb = res_wb['cmb_mass']
        cmb_idx_wb = np.argmax(m_wb >= cmb_mass_wb)

        P_s07 = np.array(res_s07['pressure']) / 1e9
        rho_s07 = np.array(res_s07['density'])

        # Plot density vs pressure (mantle only, skip core)
        ax.plot(
            P_wb[cmb_idx_wb:],
            rho_wb[cmb_idx_wb:],
            '-',
            color=color,
            linewidth=2,
            label='WolfBower2018 (T-dep)',
        )
        ax.plot(
            P_s07[cmb_idx_wb:],
            rho_s07[cmb_idx_wb:],
            '--',
            color='gray',
            linewidth=1.5,
            label='Seager2007 (300 K)',
        )

        # Annotate convergence
        conv_wb = res_wb.get('converged', False)
        conv_s07 = res_s07.get('converged', False)
        r_wb = res_wb['radii'][-1] / earth_radius
        r_s07 = res_s07['radii'][-1] / earth_radius

        ax.text(
            0.03,
            0.97,
            f'WB: R={r_wb:.3f} $R_\\oplus$, conv={conv_wb}\n'
            f'S07: R={r_s07:.3f} $R_\\oplus$, conv={conv_s07}',
            transform=ax.transAxes,
            fontsize=8,
            va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6),
        )

        ax.set_xlabel('Pressure (GPa)', fontsize=11)
        if i == 0:
            ax.set_ylabel('Density (kg/m$^3$)', fontsize=11)
        ax.set_title(f'{mass} $M_\\oplus$', fontsize=12)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        'G7: Mantle Density vs Pressure (WolfBower2018 T-dep vs Seager2007)',
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'plot_G7_Tdep_convergence.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


# ---------------------------------------------------------------------------
# Plot G8: Adiabatic T(r) profiles with solidus/liquidus
# ---------------------------------------------------------------------------


def plot_G8():
    """Adiabatic temperature profiles for 1M and 5M with melting curves."""
    print('Plot G8: Adiabatic T(r) profiles')

    masses = [1, 5]
    eos = {'core': 'Seager2007:iron', 'mantle': 'WolfBower2018:MgSiO3'}

    # Load melting curves
    solidus_func, liquidus_func = get_solidus_liquidus_functions()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, mass in enumerate(masses):
        ax = axes[idx]
        print(f'  Running {mass} M_earth (adiabatic)...')

        # Run with adiabatic temperature mode
        res = run_model(mass, eos, cmf=0.325, temperature_mode='adiabatic')

        # Also run with linear T for comparison
        print(f'  Running {mass} M_earth (linear T)...')
        res_lin = run_model(mass, eos, cmf=0.325, temperature_mode='linear')

        radii = np.array(res['radii'])
        pressure = np.array(res['pressure'])
        temperature = np.array(res['temperature'])
        mass_enclosed = np.array(res['mass_enclosed'])
        cmb_mass = res['cmb_mass']

        radii_lin = np.array(res_lin['radii'])
        temperature_lin = np.array(res_lin['temperature'])

        # Find CMB index
        cmb_idx = np.argmax(mass_enclosed >= cmb_mass)
        r_cmb = radii[cmb_idx]

        # Normalize radii
        r_norm = radii / earth_radius
        r_norm_lin = radii_lin / earth_radius

        # Plot adiabatic T(r)
        ax.plot(
            r_norm,
            temperature,
            'r-',
            linewidth=2,
            label='Adiabatic T(r)',
        )
        # Plot linear T(r) for comparison
        ax.plot(
            r_norm_lin,
            temperature_lin,
            'b--',
            linewidth=1.5,
            alpha=0.6,
            label='Linear T(r)',
        )

        # Overlay solidus and liquidus vs radius (mantle only)
        mantle_radii = radii[cmb_idx:]
        mantle_pressure = pressure[cmb_idx:]
        r_mantle_norm = mantle_radii / earth_radius

        T_solidus = np.array([solidus_func(p) for p in mantle_pressure])
        T_liquidus = np.array([liquidus_func(p) for p in mantle_pressure])

        # Mask out NaN values from extrapolation beyond melting curve range
        valid_sol = np.isfinite(T_solidus)
        valid_liq = np.isfinite(T_liquidus)

        ax.plot(
            r_mantle_norm[valid_sol],
            T_solidus[valid_sol],
            'k-.',
            linewidth=1.2,
            alpha=0.7,
            label='Solidus',
        )
        ax.plot(
            r_mantle_norm[valid_liq],
            T_liquidus[valid_liq],
            'k:',
            linewidth=1.2,
            alpha=0.7,
            label='Liquidus',
        )

        # Shade the mushy zone
        if np.any(valid_sol) and np.any(valid_liq):
            common = valid_sol & valid_liq
            if np.any(common):
                ax.fill_betweenx(
                    np.concatenate([T_solidus[common], T_liquidus[common][::-1]]),
                    np.concatenate([r_mantle_norm[common], r_mantle_norm[common][::-1]]),
                    alpha=0.08,
                    color='gray',
                    label='Mixed zone',
                )

        # Mark CMB
        ax.axvline(
            r_cmb / earth_radius,
            color='brown',
            linestyle=':',
            linewidth=1.2,
            alpha=0.7,
        )
        ax.text(
            r_cmb / earth_radius,
            ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1000,
            ' CMB',
            fontsize=9,
            color='brown',
            va='bottom',
        )

        # Annotate
        conv = res.get('converged', False)
        T_cmb = temperature[cmb_idx]
        T_surface = temperature[-1]
        T_max = np.max(temperature)
        ax.text(
            0.03,
            0.97,
            f'Conv: {conv}\n'
            f'$T_{{\\mathrm{{surface}}}}$ = {T_surface:.0f} K\n'
            f'$T_{{\\mathrm{{CMB}}}}$ = {T_cmb:.0f} K\n'
            f'$T_{{\\mathrm{{max}}}}$ = {T_max:.0f} K',
            transform=ax.transAxes,
            fontsize=9,
            va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6),
        )

        ax.set_xlabel('Radius ($R_\\oplus$)', fontsize=12)
        if idx == 0:
            ax.set_ylabel('Temperature (K)', fontsize=12)
        ax.set_title(f'{mass} $M_\\oplus$ (CMF = 0.325)', fontsize=12)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=10)

    fig.suptitle(
        'G8: Adiabatic Temperature Profiles (WolfBower2018, dT/dP tables)',
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, 'plot_G8_adiabatic_Tr.png')
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved: {path}')
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=' * 70)
    print('Generating Zalmoxis validation plots G6-G8')
    print(f'ZALMOXIS_ROOT = {ZALMOXIS_ROOT}')
    print(f'Output directory = {OUTPUT_DIR}')
    print('=' * 70)

    paths = []
    try:
        paths.append(plot_G6())
    except Exception as e:
        print(f'  FAILED: {e}')
        import traceback

        traceback.print_exc()

    try:
        paths.append(plot_G7())
    except Exception as e:
        print(f'  FAILED: {e}')
        import traceback

        traceback.print_exc()

    try:
        paths.append(plot_G8())
    except Exception as e:
        print(f'  FAILED: {e}')
        import traceback

        traceback.print_exc()

    print()
    print('=' * 70)
    print(f'Generated {len(paths)} plots:')
    for p in paths:
        print(f'  {p}')
    print('=' * 70)
