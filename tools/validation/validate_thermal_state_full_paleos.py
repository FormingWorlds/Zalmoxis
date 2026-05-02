"""Full PALEOS comparison for initial thermal states.

Three tiers:
1. Seager+2007 structure + constant properties (C_Fe=840, C_sil=1200, nabla_ad=0.3)
2. Seager+2007 structure + PALEOS nabla_ad + constant C_p
3. PALEOS structure + PALEOS nabla_ad + PALEOS C_p at CMB conditions

Also sweeps f_a = [0.01, 0.04, 0.10, 0.20, 0.40] for the full-PALEOS case.
"""

from __future__ import annotations

import math
import os
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from zalmoxis import get_zalmoxis_root
from zalmoxis.config import load_material_dictionaries, load_zalmoxis_config
from zalmoxis.constants import earth_mass
from zalmoxis.energetics import initial_thermal_state
from zalmoxis.solver import main as zalmoxis_main

OUTPUT_DIR = os.path.join(get_zalmoxis_root(), 'output', 'validation_thermal_state_full_paleos')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MASSES = [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
CMF = 0.32
F_D = 0.50
T_EQ = 255.0

plt.rcParams.update({
    'font.size': 14, 'axes.labelsize': 15, 'axes.titlesize': 16,
    'legend.fontsize': 11, 'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'lines.linewidth': 2.5, 'lines.markersize': 8,
    'savefig.bbox': 'tight', 'savefig.dpi': 200,
})


# ── PALEOS helpers ───────────────────────────────────────────────────

def build_paleos_cp_func(material='MgSiO3'):
    """Build a C_p(P, T) interpolator from a PALEOS unified table."""
    from scipy.interpolate import LinearNDInterpolator

    if material == 'MgSiO3':
        eos_file = os.path.join(get_zalmoxis_root(), 'data',
            'EOS_PALEOS_MgSiO3_unified', 'paleos_mgsio3_eos_table_pt.dat')
        fallback = 1200.0
    elif material == 'iron':
        eos_file = os.path.join(get_zalmoxis_root(), 'data',
            'EOS_PALEOS_iron', 'paleos_iron_eos_table_pt.dat')
        fallback = 840.0
    else:
        return None

    if not os.path.isfile(eos_file):
        return None

    data = np.genfromtxt(eos_file, usecols=range(9), comments='#')
    P, T, cp = data[:, 0], data[:, 1], data[:, 5]
    valid = (P > 0) & np.isfinite(cp) & (cp > 0) & (cp < 5000)
    if np.sum(valid) < 10:
        return None

    lp = np.log10(P[valid])
    lt = np.log10(T[valid])
    interp = LinearNDInterpolator(list(zip(lp, lt)), cp[valid])

    def func(P_Pa, T_K, _i=interp, _fb=fallback):
        if P_Pa <= 0 or T_K <= 0:
            return _fb
        v = float(_i(math.log10(P_Pa), math.log10(T_K)))
        if np.isfinite(v) and 0 < v < 5000:
            return v
        return _fb

    return func


def build_paleos_nabla_ad():
    """Build nabla_ad(P, T) from PALEOS MgSiO3 unified table."""
    from zalmoxis.eos.interpolation import load_paleos_unified_table
    eos_file = os.path.join(get_zalmoxis_root(), 'data',
        'EOS_PALEOS_MgSiO3_unified', 'paleos_mgsio3_eos_table_pt.dat')
    cache = load_paleos_unified_table(eos_file)

    def func(P_Pa, T_K):
        if P_Pa <= 0 or T_K <= 0:
            return 0.3
        lp = max(cache['logp_min'], min(math.log10(P_Pa), cache['logp_max']))
        lt = max(cache['logt_min'], min(math.log10(T_K), cache['logt_max']))
        try:
            v = float(cache['nabla_ad_interp']([[lp, lt]])[0])
            if np.isfinite(v) and v > 0:
                return v
        except Exception:
            pass
        return 0.3
    return func


def get_paleos_cp_at_conditions(P_Pa, T_K, material='MgSiO3'):
    """Query PALEOS C_p at specific (P, T) conditions."""
    if material == 'MgSiO3':
        path = os.path.join(get_zalmoxis_root(), 'data',
            'EOS_PALEOS_MgSiO3_unified', 'paleos_mgsio3_eos_table_pt.dat')
    elif material == 'iron':
        path = os.path.join(get_zalmoxis_root(), 'data',
            'EOS_PALEOS_iron', 'paleos_iron_eos_table_pt.dat')
    else:
        return None

    data = np.genfromtxt(path, usecols=range(9), comments='#')
    P, T, cp = data[:, 0], data[:, 1], data[:, 5]

    # Filter: positive P, finite cp, condensed phases only (exclude vapor/supercritical
    # where cp diverges near phase boundaries)
    valid = (P > 0) & np.isfinite(cp) & (cp > 0) & (cp < 5000)
    if np.sum(valid) == 0:
        return 1200.0 if material == 'MgSiO3' else 840.0

    lp = np.log10(P[valid])
    lt = np.log10(T[valid])
    cp_v = cp[valid]
    dist = (lp - math.log10(max(P_Pa, 1.0)))**2 + (lt - math.log10(max(T_K, 1.0)))**2
    return float(cp_v[np.argmin(dist)])


# ── Solvers ──────────────────────────────────────────────────────────

def run_seager_constant(mass_earth, f_a=0.04):
    """Tier 1: Seager structure + constant properties."""
    config = _base_config(mass_earth, {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'})
    res = _run(config)
    if res is None:
        return None
    th = initial_thermal_state(res, CMF, T_radiative_eq=T_EQ,
                               f_accretion=f_a, f_differentiation=F_D)
    return {'res': res, 'th': th, 'label': 'Seager + const'}


def run_seager_paleos_nabla(mass_earth, nabla_func, f_a=0.04):
    """Tier 2: Seager structure + PALEOS nabla_ad + constant C_p."""
    config = _base_config(mass_earth, {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'})
    res = _run(config)
    if res is None:
        return None
    th = initial_thermal_state(res, CMF, T_radiative_eq=T_EQ,
                               f_accretion=f_a, f_differentiation=F_D,
                               nabla_ad_func=nabla_func)
    return {'res': res, 'th': th, 'label': 'Seager + PALEOS ∇ad'}


def run_full_paleos(mass_earth, nabla_func, cp_fe_func, cp_sil_func, f_a=0.04):
    """Tier 3: PALEOS structure + PALEOS nabla_ad + mass-weighted PALEOS C_p."""
    config = _base_config(mass_earth, {'core': 'PALEOS:iron', 'mantle': 'PALEOS:MgSiO3'},
                          temp_mode='linear', T_surf=3000, T_center=6000)
    res = _run(config)
    if res is None:
        return None

    th = initial_thermal_state(res, CMF, T_radiative_eq=T_EQ,
                               f_accretion=f_a, f_differentiation=F_D,
                               nabla_ad_func=nabla_func,
                               cp_iron_func=cp_fe_func,
                               cp_silicate_func=cp_sil_func)
    return {'res': res, 'th': th, 'label': 'PALEOS (full)'}


def _base_config(mass_earth, eos, temp_mode='isothermal', T_surf=3000, T_center=6000):
    config_path = os.path.join(get_zalmoxis_root(), 'input', 'default.toml')
    config = load_zalmoxis_config(config_path)
    config['planet_mass'] = mass_earth * earth_mass
    config['core_mass_fraction'] = CMF
    config['layer_eos_config'] = eos
    config['temperature_mode'] = temp_mode
    config['surface_temperature'] = T_surf
    config['center_temperature'] = T_center
    config['data_output_enabled'] = False
    config['plotting_enabled'] = False
    config['verbose'] = False
    return config


def _run(config):
    mat = load_material_dictionaries()
    input_dir = os.path.join(get_zalmoxis_root(), 'input')
    res = zalmoxis_main(config, mat, None, input_dir)
    return res if res['converged'] else None


# ── Main ─────────────────────────────────────────────────────────────

def main():
    print('Building PALEOS nabla_ad and C_p interpolators...')
    nabla_func = build_paleos_nabla_ad()
    cp_fe_func = build_paleos_cp_func('iron')
    cp_sil_func = build_paleos_cp_func('MgSiO3')
    print(f'  nabla_ad: {"OK" if nabla_func else "MISSING"}')
    print(f'  C_p iron: {"OK" if cp_fe_func else "MISSING"}')
    print(f'  C_p sil:  {"OK" if cp_sil_func else "MISSING"}')

    # ── Three-tier comparison at f_a = 0.04 ──────────────────────────
    print('\n=== Three-tier comparison (f_a = 0.04) ===')
    tier1, tier2, tier3 = [], [], []

    for m in MASSES:
        print(f'\n  {m} M_earth...')
        r1 = run_seager_constant(m)
        r2 = run_seager_paleos_nabla(m, nabla_func)
        r3 = run_full_paleos(m, nabla_func, cp_fe_func, cp_sil_func)

        if r1:
            tier1.append({'mass': m, **r1['th']})
            print(f'    Tier 1 (Seager+const):     T_CMB={r1["th"]["T_cmb"]:.0f}, T_surf={r1["th"]["T_surface"]:.0f}')
        if r2:
            tier2.append({'mass': m, **r2['th']})
            print(f'    Tier 2 (Seager+PALEOS∇):   T_CMB={r2["th"]["T_cmb"]:.0f}, T_surf={r2["th"]["T_surface"]:.0f}')
        if r3:
            tier3.append({'mass': m, **r3['th']})
            C_info = f'C_Fe_avg={r3["th"].get("C_iron_avg", 0):.0f}, C_sil_avg={r3["th"].get("C_silicate_avg", 0):.0f}'
            C_info = f'C_Fe_avg={r3["th"].get("C_iron_avg", 0):.0f}, C_sil_avg={r3["th"].get("C_silicate_avg", 0):.0f}'
            print(f'    Tier 3 (full PALEOS):      T_CMB={r3["th"]["T_cmb"]:.0f}, T_surf={r3["th"]["T_surface"]:.0f} ({C_info})')

    # ── Plot: Three-tier comparison ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for data, color, marker, ls, label in [
        (tier1, '#BBBBBB', 'v', '-', r'Seager + const ($\nabla_\mathrm{ad}=0.3$, $C_p$ fixed)'),
        (tier2, '#4477AA', 'o', '--', r'Seager + PALEOS $\nabla_\mathrm{ad}$, $C_p$ fixed'),
        (tier3, '#228833', 's', '-', r'PALEOS structure + $\nabla_\mathrm{ad}$ + $C_p$'),
    ]:
        masses = [d['mass'] for d in data]
        axes[0].plot(masses, [d['T_cmb'] for d in data], f'{marker}{ls}', color=color, label=label)
        axes[1].plot(masses, [d['T_surface'] for d in data], f'{marker}{ls}', color=color, label=label)

    # Iron melting reference
    from zalmoxis.melting_curves import iron_melting_anzellini13
    if tier3:
        axes[0].plot([d['mass'] for d in tier1],
                     [iron_melting_anzellini13(d.get('P_cmb', 135e9)) for d in tier1],
                     'k--', linewidth=1.5, label='Fe melting (Anzellini)')

    axes[0].set_xlabel(r'Planet mass [$M_\oplus$]')
    axes[0].set_ylabel(r'$T_\mathrm{CMB}$ [K]')
    axes[0].set_title(r'(a) CMB temperature')
    axes[0].legend(fontsize=9)

    axes[1].axhline(1400, color='orange', ls=':', lw=1.5, label='Solidus (~1400 K)')
    axes[1].set_xlabel(r'Planet mass [$M_\oplus$]')
    axes[1].set_ylabel(r'$T_\mathrm{surf}$ [K]')
    axes[1].set_title(r'(b) Surface temperature')
    axes[1].legend(fontsize=9)

    fig.suptitle(r'Initial thermal state: three EOS tiers ($f_c=0.32$, $f_a=0.04$, $f_d=0.50$)')
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'thermal_state_three_tiers.pdf')
    fig.savefig(fname)
    fig.savefig(fname.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'\nSaved: {fname}')

    # ── f_a sweep with full PALEOS ───────────────────────────────────
    print('\n=== f_a sweep (full PALEOS) ===')
    F_A_VALUES = [0.01, 0.04, 0.10, 0.20, 0.40]
    fa_results = {}

    for f_a in F_A_VALUES:
        fa_results[f_a] = []
        for m in MASSES:
            r = run_full_paleos(m, nabla_func, cp_fe_func, cp_sil_func, f_a=f_a)
            if r:
                fa_results[f_a].append({'mass': m, **r['th']})

    colors = ['#BBBBBB', '#4477AA', '#228833', '#EE6677', '#AA3377']
    markers = ['v', 'o', 's', '^', 'D']

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for i, f_a in enumerate(F_A_VALUES):
        d = fa_results[f_a]
        masses = [x['mass'] for x in d]
        axes[0].plot(masses, [x['T_cmb'] for x in d], f'{markers[i]}-',
                     color=colors[i], label=f'$f_a = {f_a}$')
        axes[1].plot(masses, [x['T_surface'] for x in d], f'{markers[i]}-',
                     color=colors[i], label=f'$f_a = {f_a}$')

    axes[0].set_xlabel(r'Planet mass [$M_\oplus$]')
    axes[0].set_ylabel(r'$T_\mathrm{CMB}$ [K]')
    axes[0].set_title(r'(a) CMB temperature')
    axes[0].legend(fontsize=10)

    axes[1].axhline(1400, color='orange', ls=':', lw=1.5, label='Solidus')
    axes[1].set_xlabel(r'Planet mass [$M_\oplus$]')
    axes[1].set_ylabel(r'$T_\mathrm{surf}$ [K]')
    axes[1].set_title(r'(b) Surface temperature')
    axes[1].legend(fontsize=10)

    fig.suptitle(
        r'$f_a$ sensitivity: full PALEOS (structure + $\nabla_\mathrm{ad}$ + $C_p$), '
        r'$f_c=0.32$, $f_d=0.50$')
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'thermal_state_fa_full_paleos.pdf')
    fig.savefig(fname)
    fig.savefig(fname.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'Saved: {fname}')

    # ── Summary table ────────────────────────────────────────────────
    print('\n' + '=' * 110)
    print(f'{"M/ME":>5}  {"T1_CMB":>7}  {"T1_surf":>8}  {"T2_CMB":>7}  {"T2_surf":>8}  '
          f'{"T3_CMB":>7}  {"T3_surf":>8}  {"C_fe_avg":>8}  {"C_sil_avg":>9}  {"core3":>7}')
    print('-' * 110)
    for i in range(min(len(tier1), len(tier2), len(tier3))):
        t1, t2, t3 = tier1[i], tier2[i], tier3[i]
        print(f'{t1["mass"]:5.1f}  {t1["T_cmb"]:7.0f}  {t1["T_surface"]:8.0f}  '
              f'{t2["T_cmb"]:7.0f}  {t2["T_surface"]:8.0f}  '
              f'{t3["T_cmb"]:7.0f}  {t3["T_surface"]:8.0f}  '
              f'{t3.get("C_iron_avg", 0):8.0f}  {t3.get("C_silicate_avg", 0):9.0f}  '
              f'{t3["core_state"]:>7}')
    print('=' * 110)

    # ── White+Li 2025 Fig. 3 reproduction ────────────────────────────
    # Using their exact parameters (constant C_p, constant nabla_ad=0.3)
    # to reproduce their results, then compare with full PALEOS
    print('\n=== White+Li 2025 Fig. 3 reproduction ===')

    # White+Li parameters (only f_a is currently consumed by the
    # const-C_p reproduction; f_d / C_p constants are documented in
    # White+Li 2025 Table 2 but the comparison plots don't pass them).
    WL_F_A = 0.04

    # Boujibar parameters for comparison
    BJ_F_A = 0.40
    BJ_F_D = 0.40
    BJ_C_FE = 840.0
    BJ_C_SIL = 1200.0

    wl_data, bj_data, paleos_data_wl = [], [], []
    for m in MASSES:
        # White+Li reproduction (their exact parameters)
        r_wl = run_seager_constant(m, f_a=WL_F_A)
        if r_wl:
            wl_data.append({'mass': m, **r_wl['th']})

        # Boujibar reproduction (their exact parameters)
        config = _base_config(m, {'core': 'Analytic:iron', 'mantle': 'Analytic:MgSiO3'})
        res = _run(config)
        if res:
            th_bj = initial_thermal_state(res, CMF, T_radiative_eq=T_EQ,
                                          f_accretion=BJ_F_A, f_differentiation=BJ_F_D,
                                          C_iron=BJ_C_FE, C_silicate=BJ_C_SIL)
            bj_data.append({'mass': m, **th_bj})

        # Full PALEOS with White+Li f_a/f_d
        r_p = run_full_paleos(m, nabla_func, cp_fe_func, cp_sil_func, f_a=WL_F_A)
        if r_p:
            paleos_data_wl.append({'mass': m, **r_p['th']})

    # Boujibar polynomial reference
    def boujibar_poly(m):
        m = np.asarray(m, dtype=float)
        return 24.6 * m**2 + 1695.0 * m + 6041.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    masses_wl = [d['mass'] for d in wl_data]
    masses_bj = [d['mass'] for d in bj_data]
    masses_pl = [d['mass'] for d in paleos_data_wl]

    # Panel (a): T_CMB comparison
    ax = axes[0]
    ax.plot(masses_wl, [d['T_cmb'] for d in wl_data], 'o-', color='#4477AA',
            label=f'White+Li params ($f_a$={WL_F_A}, const $C_p$)')
    ax.plot(masses_bj, [d['T_cmb'] for d in bj_data], '^-', color='#EE6677',
            label=f'Boujibar params ($f_a$={BJ_F_A}, const $C_p$)')
    m_ref = np.linspace(0.5, 5, 50)
    ax.plot(m_ref, boujibar_poly(m_ref), ':', color='#EE6677', linewidth=1.5,
            label='Boujibar+2020 polynomial')
    ax.plot(masses_pl, [d['T_cmb'] for d in paleos_data_wl], 's-', color='#228833',
            label=f'Full PALEOS ($f_a$={WL_F_A}, PALEOS $C_p$+$\\nabla_{{ad}}$)')

    from zalmoxis.melting_curves import iron_melting_anzellini13
    ax.plot(m_ref, [iron_melting_anzellini13(135e9 * m**2) for m in m_ref],
            'k--', linewidth=1, alpha=0.5, label='Fe melting (rough)')

    ax.set_xlabel(r'Planet mass [$M_\oplus$]')
    ax.set_ylabel(r'$T_\mathrm{CMB}$ [K]')
    ax.set_title('(a) CMB temperature: parameter comparison')
    ax.legend(fontsize=9)
    ax.set_ylim(0, 25000)

    # Panel (b): T_surface comparison
    ax = axes[1]
    ax.plot(masses_wl, [d['T_surface'] for d in wl_data], 'o-', color='#4477AA',
            label=f'White+Li ($f_a$={WL_F_A}, const)')
    ax.plot(masses_bj, [d['T_surface'] for d in bj_data], '^-', color='#EE6677',
            label=f'Boujibar ($f_a$={BJ_F_A}, const)')
    ax.plot(masses_pl, [d['T_surface'] for d in paleos_data_wl], 's-', color='#228833',
            label=f'Full PALEOS ($f_a$={WL_F_A})')
    ax.axhline(1400, color='orange', ls=':', lw=1.5, label='Solidus')
    ax.set_xlabel(r'Planet mass [$M_\oplus$]')
    ax.set_ylabel(r'$T_\mathrm{surf}$ [K]')
    ax.set_title('(b) Surface temperature')
    ax.legend(fontsize=9)

    fig.suptitle(
        r'Comparison: White+Li 2025 vs Boujibar+2020 vs full PALEOS ($f_c=0.32$, $f_d=0.50$)',
        fontsize=14,
    )
    fig.tight_layout()
    fname = os.path.join(OUTPUT_DIR, 'thermal_state_whiteli_comparison.pdf')
    fig.savefig(fname)
    fig.savefig(fname.replace('.pdf', '.png'))
    plt.close(fig)
    print(f'\nSaved: {fname}')

    # Print comparison table
    print('\n' + '=' * 90)
    print(f'{"M/ME":>5}  {"WL_CMB":>7}  {"WL_surf":>8}  {"BJ_CMB":>7}  {"BJ_surf":>8}  '
          f'{"PAL_CMB":>8}  {"PAL_surf":>9}  {"BJ_poly":>8}')
    print('-' * 90)
    for i in range(min(len(wl_data), len(bj_data), len(paleos_data_wl))):
        w, b, p = wl_data[i], bj_data[i], paleos_data_wl[i]
        bp = boujibar_poly(w['mass'])
        print(f'{w["mass"]:5.1f}  {w["T_cmb"]:7.0f}  {w["T_surface"]:8.0f}  '
              f'{b["T_cmb"]:7.0f}  {b["T_surface"]:8.0f}  '
              f'{p["T_cmb"]:8.0f}  {p["T_surface"]:9.0f}  {bp:8.0f}')
    print('=' * 90)

    print(f'\nAll output in: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
