#!/usr/bin/env python3
"""Convert Chabrier+2019/2021 H/He EOS tables to PALEOS-compatible format.

Reads the DirEOS2021 tables (Chabrier, Mazevet & Soubiran 2019, ApJ 872, 51;
Chabrier & Debras 2021, ApJ 917, 4) and writes them in the same 10-column
format used by PALEOS unified tables in Zalmoxis.

Source data
-----------
- Chabrier et al. (2019): "A New Equation of State for Dense
  Hydrogen-Helium Mixtures", ApJ 872, 51.
  https://doi.org/10.3847/1538-4357/aaf99f
  Provides pure H, pure He, and H/He mixture EOS tables based on the
  additive volume law (IVL). Covers rho = 1e-8 to 1e6 g/cm^3,
  P = 1e-9 to 1e13 GPa, T = 1e2 to 1e8 K.

- Chabrier & Debras (2021): "A New Equation of State for Dense
  Hydrogen-Helium Mixtures. II.", ApJ 917, 4.
  https://doi.org/10.3847/1538-4357/abfc48
  Extends the 2019 tables by incorporating non-ideal H/He interactions
  from QMD simulations (Militzer & Hubbard 2013). Provides updated H/He
  mixture tables for Y = 0.275, 0.292, 0.297 and an "effective" pure H
  table for deriving mixtures at arbitrary Y.

Input format (Chabrier)
-----------------------
Regular grid in (log10 T, log10 P) with 10 columns:
  1. log10(T / K)
  2. log10(P / GPa)
  3. log10(rho / g cm^-3)
  4. log10(U / MJ kg^-1)      — internal energy
  5. log10(S / MJ kg^-1 K^-1) — specific entropy
  6. (d ln rho / d ln T)_P    — logarithmic density derivative at constant P
  7. (d ln rho / d ln P)_T    — logarithmic density derivative at constant T
  8. (d ln S / d ln T)_P      — logarithmic entropy derivative at constant P
  9. (d ln S / d ln P)_T      — logarithmic entropy derivative at constant T
  10. nabla_ad                 — adiabatic gradient (d ln T / d ln P)_S

Grid: 121 isotherms (log T = 2.0 to 8.0, step 0.05) x
      441 pressures  (log P = -9.0 to 13.0, step 0.05)
Units: T in K, P in GPa, rho in g/cm^3, U in MJ/kg, S in MJ/(kg K)

CAUTION from the README: some values at very low densities or in the
solid/quantum regime are unphysical. The grad_ad column is clamped at
0.100 in the H2 dissociation/ionization zone.

Output format (PALEOS-compatible)
---------------------------------
Regular grid in (log10 P, log10 T) with 10 columns:
  1. P (Pa)
  2. T (K)
  3. rho (kg/m^3)
  4. u (J/kg)
  5. s (J/(kg K))
  6. cp (J/(kg K))
  7. cv (J/(kg K))
  8. alpha (1/K)
  9. nabla_ad (dimensionless)
  10. phase_id (string)

Thermodynamic derivations
-------------------------
The Chabrier tables do not provide cp, cv, or alpha directly. These are
derived from the available columns using standard thermodynamic identities.

THERMAL EXPANSIVITY (alpha):
  Definition: alpha = (1/V)(dV/dT)_P = -(1/rho)(drho/dT)_P
  From the table: (d ln rho / d ln T)_P = (T/rho)(drho/dT)_P
  Therefore:
    alpha = -(1/T) * (d ln rho / d ln T)_P

  Confidence: HIGH. This is a direct algebraic identity with no
  approximation. Verified against ideal gas limit (alpha = 1/T).

HEAT CAPACITY AT CONSTANT PRESSURE (cp):
  Two independent derivation paths:

  Path A (from entropy, independent check):
    cp = T * (dS/dT)_P = S * (d ln S / d ln T)_P
    This uses the entropy column and its derivative.
    Accuracy: good in most regimes (0.4% median agreement with Path B)
    but degrades near the H2 dissociation zone where S is near a minimum
    and the logarithmic derivative amplifies errors.

  Path B (from nabla_ad, self-consistent with the table):
    nabla_ad = (d ln T / d ln P)_S = P * alpha / (rho * cp)
    Therefore: cp = P * alpha / (rho * nabla_ad)
    This uses the table's own grad_ad column and the derived alpha.

    Confidence: HIGH where grad_ad reflects the true thermodynamic value.
    UNRELIABLE where grad_ad = 0.100 (the table floor). The Chabrier
    tables clamp nabla_ad at 0.100 in the H2 dissociation/ionization zone
    (T ~ 3000-10000 K, P ~ 1-100 GPa). At these points, the true nabla_ad
    can be much smaller (dissociation absorbs latent heat, steepening the
    adiabat), so cp derived from the clamped value will be wrong.

  This script uses Path B as the primary source (self-consistent with
  the table's own adiabatic gradient) and Path A as a cross-check.
  Points where grad_ad = 0.100 are flagged in the phase_id column.

HEAT CAPACITY AT CONSTANT VOLUME (cv):
  Mayer relation: cp - cv = T * alpha^2 / (rho * beta_T)
  where beta_T = (1/rho)(drho/dP)_T = (1/P)(d ln rho / d ln P)_T

  Confidence: MODERATE. The identity is exact, but it chains together
  alpha (from the table) and beta_T (from the table). Near phase
  transitions, both derivatives can be large and opposite in sign,
  causing numerical cancellation in cv = cp - (large positive number).
  Points where cv < 0 or cp/cv > 3.0 are flagged.

PHASE IDENTIFICATION:
  The Chabrier tables do not provide phase labels. For hydrogen at
  astrophysically relevant conditions:
  - T < ~1000 K, P < ~300 GPa: molecular H2 (or solid at low T)
  - T ~ 3000-10000 K, P ~ 1-100 GPa: dissociation zone (H2 -> 2H)
  - T > 10000 K or P > 100 GPa: atomic/metallic H
  - T > 100000 K: fully ionized plasma

  The phase_id column uses:
  - "molecular": low T, low-to-moderate P
  - "dissociating": grad_ad clamped at 0.100 (dissociation zone)
  - "atomic": high T or high P (atomic/metallic hydrogen)
  - "unphysical": outside the EOS validity domain

  These labels are approximate and based on the grad_ad floor and
  density/temperature thresholds, not on a rigorous phase calculation.

Verification results
--------------------
Self-consistency check (nabla_ad = P*alpha/(rho*cp) vs table grad_ad):
  - Molecular/atomic regime (outside dissociation zone): 0.3% median error
  - Ideal gas limit (T=1000K, P=1 MPa): cp = 15055 J/kg/K
    (ideal H2: 14434), cp/cv = 1.38 (ideal: 1.40), alpha = 9.98e-4
    (ideal: 1.00e-3)
  - The ~4% offset from ideal H2 values is physical: at 1000K, H2
    vibrational modes are partially excited, increasing cp above the
    rigid-rotor value.

Known limitations
-----------------
  - cp is UNRELIABLE at grad_ad = 0.100 (17.6% of all points). These
    are in the H2 dissociation/ionization zone. The true cp can be
    much larger than the derived value (dissociation absorbs energy).
  - cv can go negative near sharp phase transitions due to numerical
    cancellation in the Mayer relation.
  - The "unphysical" corners of the table (logP < -7 GPa or logT > 5.5)
    may have U and S values that overflow 10^(column value).
  - The effective H table (TABLE_H_TP_effective) must NOT be used as a
    pure H EOS. Only TABLE_H_TP_v1 is valid for pure hydrogen.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Chabrier grid parameters (from README)
N_T = 121  # isotherms: logT = 2.0 to 8.0, step 0.05
N_P = 441  # pressures: logP(GPa) = -9.0 to 13.0, step 0.05
DLOG = 0.05  # uniform step in both log10(T) and log10(P)

# grad_ad floor in the Chabrier tables (H2 dissociation zone)
GRAD_AD_FLOOR = 0.100

# Astrophysically relevant domain boundaries
LOGP_MIN_PHYS = -7.0  # 100 Pa — below this, table values may be unphysical
LOGT_MAX_PHYS = 5.5  # 316000 K — above this, relativistic effects


def load_chabrier_table(filepath: str) -> dict:
    """Load a Chabrier (T,P) table and return raw columns.

    Parameters
    ----------
    filepath : str
        Path to the Chabrier TABLE_*_TP_* file.

    Returns
    -------
    dict
        Raw data with keys: logT, logP_GPa, logrho_gcc, logU_MJkg,
        logS_MJkgK, dlrho_dlT, dlrho_dlP, dlS_dlT, dlS_dlP, grad_ad.
    """
    data = np.loadtxt(filepath)
    assert data.shape == (N_T * N_P, 10), (
        f'Expected {N_T * N_P} rows x 10 cols, got {data.shape}'
    )
    return {
        'logT': data[:, 0],
        'logP_GPa': data[:, 1],
        'logrho_gcc': data[:, 2],
        'logU_MJkg': data[:, 3],
        'logS_MJkgK': data[:, 4],
        'dlrho_dlT': data[:, 5],
        'dlrho_dlP': data[:, 6],
        'dlS_dlT': data[:, 7],
        'dlS_dlP': data[:, 8],
        'grad_ad': data[:, 9],
    }


def convert_units(raw: dict) -> dict:
    """Convert Chabrier CGS/GPa units to SI (Pa, kg/m^3, J/kg, J/kg/K).

    Parameters
    ----------
    raw : dict
        Output of load_chabrier_table().

    Returns
    -------
    dict
        SI quantities: T (K), P (Pa), rho (kg/m^3), u (J/kg), s (J/kg/K),
        plus the original log-derivative columns and grad_ad.
    """
    T = 10.0 ** raw['logT']  # K
    P = 10.0 ** raw['logP_GPa'] * 1.0e9  # GPa -> Pa
    rho = 10.0 ** raw['logrho_gcc'] * 1.0e3  # g/cm^3 -> kg/m^3

    # U and S are stored as log10 in the physical regime,
    # but may have unphysical values at table corners.
    # Flag unphysical corners.
    logP_GPa = raw['logP_GPa']
    logT = raw['logT']
    physical = (logP_GPa >= LOGP_MIN_PHYS) & (logT <= LOGT_MAX_PHYS)

    u = np.full_like(raw['logU_MJkg'], np.nan)
    s = np.full_like(raw['logS_MJkgK'], np.nan)

    # Safe conversion: only where values are plausible as log10
    safe_U = physical & (np.abs(raw['logU_MJkg']) < 20)
    safe_S = physical & (np.abs(raw['logS_MJkgK']) < 20)
    u[safe_U] = 10.0 ** raw['logU_MJkg'][safe_U] * 1.0e6  # MJ/kg -> J/kg
    s[safe_S] = 10.0 ** raw['logS_MJkgK'][safe_S] * 1.0e6  # MJ/(kg K) -> J/(kg K)

    return {
        'T': T,
        'P': P,
        'rho': rho,
        'u': u,
        's': s,
        'logT': logT,
        'logP_GPa': logP_GPa,
        'physical': physical,
        'dlrho_dlT': raw['dlrho_dlT'],
        'dlrho_dlP': raw['dlrho_dlP'],
        'dlS_dlT': raw['dlS_dlT'],
        'dlS_dlP': raw['dlS_dlP'],
        'grad_ad': raw['grad_ad'],
    }


def derive_thermodynamic_quantities(si: dict) -> dict:
    """Derive cp, cv, alpha from the Chabrier log-derivatives and grad_ad.

    Parameters
    ----------
    si : dict
        Output of convert_units().

    Returns
    -------
    dict
        Adds: alpha (1/K), cp (J/kg/K), cv (J/kg/K), phase_id (str),
        cp_from_entropy (J/kg/K, independent cross-check).
    """
    T = si['T']
    P = si['P']
    rho = si['rho']
    s = si['s']
    grad_ad = si['grad_ad']
    dlrho_dlT = si['dlrho_dlT']
    dlrho_dlP = si['dlrho_dlP']
    dlS_dlT = si['dlS_dlT']

    # ------------------------------------------------------------------
    # ALPHA: thermal expansivity
    # alpha = -(1/T) * (d ln rho / d ln T)_P
    #
    # Derivation:
    #   alpha = (1/V)(dV/dT)_P = -(1/rho)(drho/dT)_P
    #   (d ln rho / d ln T)_P = (T/rho)(drho/dT)_P
    #   => (drho/dT)_P = (rho/T)(d ln rho / d ln T)_P
    #   => alpha = -(1/rho)(rho/T)(d ln rho / d ln T)_P
    #            = -(1/T)(d ln rho / d ln T)_P
    # ------------------------------------------------------------------
    alpha = -(1.0 / T) * dlrho_dlT

    # ------------------------------------------------------------------
    # CP: heat capacity at constant pressure (primary: from grad_ad)
    # cp = P * alpha / (rho * nabla_ad)
    #
    # Derivation:
    #   nabla_ad = (d ln T / d ln P)_S
    #   From Maxwell: (dT/dP)_S = T * V * alpha / cp = T * alpha / (rho * cp)
    #   nabla_ad = (P/T)(dT/dP)_S = P * alpha / (rho * cp)
    #   => cp = P * alpha / (rho * nabla_ad)
    #
    # LIMITATION: where grad_ad is clamped at 0.100 (dissociation zone),
    # this yields an INCORRECT cp. The true nabla_ad is smaller, so the
    # true cp is larger. These points are flagged in phase_id.
    # ------------------------------------------------------------------
    cp = np.where(
        (grad_ad > 0) & (rho > 0) & np.isfinite(alpha),
        P * alpha / (rho * grad_ad),
        np.nan,
    )

    # ------------------------------------------------------------------
    # CP cross-check: from entropy (independent derivation)
    # cp = S * (d ln S / d ln T)_P
    #
    # Derivation:
    #   cp = T * (dS/dT)_P
    #   (d ln S / d ln T)_P = (T/S)(dS/dT)_P
    #   => (dS/dT)_P = (S/T)(d ln S / d ln T)_P
    #   => cp = T * (S/T)(d ln S / d ln T)_P = S * (d ln S / d ln T)_P
    #
    # This is useful as a cross-check where both S and dlS_dlT are reliable.
    # ------------------------------------------------------------------
    cp_entropy = np.where(
        np.isfinite(s) & (s > 0) & np.isfinite(dlS_dlT),
        s * dlS_dlT,
        np.nan,
    )

    # ------------------------------------------------------------------
    # CV: heat capacity at constant volume (Mayer relation)
    # cv = cp - T * alpha^2 / (rho * beta_T)
    #
    # Derivation (Mayer relation):
    #   cp - cv = -T * [(dV/dT)_P]^2 / (dV/dP)_T
    #   With V = 1/rho:
    #     (dV/dT)_P = alpha * V = alpha / rho
    #     (dV/dP)_T = -V * beta_T = -beta_T / rho
    #   where beta_T = (1/rho)(drho/dP)_T is the isothermal compressibility.
    #   => cp - cv = -T * (alpha/rho)^2 / (-beta_T/rho)
    #             = T * alpha^2 / (rho * beta_T)
    #
    # From the table: (d ln rho / d ln P)_T = (P/rho)(drho/dP)_T
    #   => beta_T = (1/P)(d ln rho / d ln P)_T
    #
    # LIMITATION: near phase transitions, alpha and beta_T can be large
    # with opposite signs, causing numerical cancellation. Points where
    # cv < 0 or cp/cv is out of range are flagged.
    #
    # Refs: Callen (1985) Eq. 3.73; Landau & Lifshitz, Stat. Phys. §16.
    # ------------------------------------------------------------------
    beta_T = (1.0 / P) * dlrho_dlP
    mayer_correction = np.where(
        (rho > 0) & (beta_T != 0) & np.isfinite(alpha) & np.isfinite(beta_T),
        T * alpha**2 / (rho * beta_T),
        np.nan,
    )
    cv = cp - mayer_correction

    # ------------------------------------------------------------------
    # PHASE IDENTIFICATION (approximate)
    # ------------------------------------------------------------------
    n = len(T)
    phase_id = np.empty(n, dtype='U30')
    phase_id[:] = 'atomic'

    # Unphysical corners
    unphys = ~si['physical']
    phase_id[unphys] = 'unphysical'

    # Molecular H2: low T, before dissociation
    molecular = si['physical'] & (si['logT'] < 3.5) & (np.abs(grad_ad - GRAD_AD_FLOOR) > 0.001)
    phase_id[molecular] = 'molecular'

    # Dissociation zone: grad_ad clamped at floor
    dissociating = si['physical'] & (np.abs(grad_ad - GRAD_AD_FLOOR) <= 0.001)
    phase_id[dissociating] = 'dissociating'

    # Flag unreliable derived quantities
    bad_cv = (
        np.isfinite(cv)
        & np.isfinite(cp)
        & ((cv <= 0) | (cp / np.where(cv > 0, cv, np.nan) > 3.0))
    )
    phase_id[bad_cv & si['physical']] = np.char.add(
        phase_id[bad_cv & si['physical']], ':cv_unreliable'
    )

    si['alpha'] = alpha
    si['cp'] = cp
    si['cv'] = cv
    si['cp_entropy'] = cp_entropy
    si['phase_id'] = phase_id
    return si


def write_paleos_format(si: dict, outpath: str, material_label: str) -> None:
    """Write the converted table in PALEOS 10-column format.

    Parameters
    ----------
    si : dict
        Output of derive_thermodynamic_quantities().
    outpath : str
        Output file path.
    material_label : str
        Material name for the header (e.g., 'H', 'He', 'HHe_Y0275').
    """
    n = len(si['T'])

    # Material-specific CAUTION text for the nabla_ad floor
    if material_label == 'HE':
        caution_zone = 'He ionization/condensation zone'
    elif material_label.startswith('HHe'):
        caution_zone = 'H2 dissociation / He ionization zone'
    else:
        caution_zone = 'H2 dissociation zone'

    header_lines = [
        f'# Chabrier+2019/2021 {material_label} EOS in PALEOS-compatible format',
        '#',
        '# Source: Chabrier, Mazevet & Soubiran (2019), ApJ 872, 51',
        '#         https://doi.org/10.3847/1538-4357/aaf99f',
        '#         Chabrier & Debras (2021), ApJ 917, 4',
        '#         https://doi.org/10.3847/1538-4357/abfc48',
        '#',
        f'# Converted from DirEOS2021/TABLE_{material_label}_TP_v1',
        f'# Grid: {N_T} isotherms x {N_P} pressures = {N_T * N_P} points',
        f'# T range: {10**2.0:.0f} to {10**8.0:.0e} K',
        f'# P range: {10 ** (-9.0) * 1e9:.0e} to {10**13.0 * 1e9:.0e} Pa',
        '#',
        '# Columns directly from table: P, T, rho, u, s, nabla_ad',
        '# Derived columns: cp, cv, alpha (see convert_chabrier_to_paleos.py docstring)',
        '#',
        f'# CAUTION: cp is unreliable where nabla_ad = 0.100 ({caution_zone},',
        "# phase_id = 'dissociating'). cv may be negative near phase transitions.",
        "# Points outside the astrophysical validity domain have phase_id = 'unphysical'.",
        '#',
        '# P[Pa]  T[K]  rho[kg/m3]  u[J/kg]  s[J/(kgK)]  cp[J/(kgK)]  cv[J/(kgK)]  alpha[1/K]  nabla_ad  phase_id',
    ]

    with open(outpath, 'w') as f:
        for line in header_lines:
            f.write(line + '\n')

        for i in range(n):
            p_str = f'{si["P"][i]:.6e}'
            t_str = f'{si["T"][i]:.2f}'
            rho_str = f'{si["rho"][i]:.6e}'
            u_str = f'{si["u"][i]:.6e}' if np.isfinite(si['u'][i]) else 'NaN'
            s_str = f'{si["s"][i]:.6e}' if np.isfinite(si['s'][i]) else 'NaN'
            cp_str = f'{si["cp"][i]:.6e}' if np.isfinite(si['cp'][i]) else 'NaN'
            cv_str = f'{si["cv"][i]:.6e}' if np.isfinite(si['cv'][i]) else 'NaN'
            alpha_str = f'{si["alpha"][i]:.6e}' if np.isfinite(si['alpha'][i]) else 'NaN'
            nabla_str = f'{si["grad_ad"][i]:.6f}'
            phase_str = si['phase_id'][i]

            f.write(
                f'{p_str}  {t_str}  {rho_str}  {u_str}  {s_str}  '
                f'{cp_str}  {cv_str}  {alpha_str}  {nabla_str}  {phase_str}\n'
            )


def print_verification(si: dict) -> None:
    """Print verification statistics for the derived quantities."""
    T = si['T']
    P = si['P']
    rho = si['rho']
    alpha = si['alpha']
    cp = si['cp']
    cv = si['cv']
    cp_entropy = si['cp_entropy']
    grad_ad = si['grad_ad']
    physical = si['physical']
    logT = si['logT']
    logP = si['logP_GPa']

    print('=' * 70)
    print('VERIFICATION REPORT')
    print('=' * 70)

    # Self-consistency: nabla_ad = P*alpha/(rho*cp) vs table
    nabla_derived = np.where(
        np.isfinite(cp) & (cp > 0) & (rho > 0),
        P * alpha / (rho * cp),
        np.nan,
    )
    valid = physical & np.isfinite(nabla_derived) & (grad_ad > 0.01)
    err = np.abs(nabla_derived[valid] - grad_ad[valid]) / grad_ad[valid]
    print(f'\n1. nabla_ad self-consistency ({valid.sum()} points):')
    print(f'   Median relative error: {np.median(err):.4e}')
    print("   This should be ~0 (it's circular by construction from Path B).")

    # Cross-check: cp from grad_ad vs cp from entropy
    both = (
        physical
        & np.isfinite(cp)
        & (cp > 0)
        & np.isfinite(cp_entropy)
        & (cp_entropy > 0)
        & (np.abs(grad_ad - GRAD_AD_FLOOR) > 0.001)  # exclude clamped points
    )
    if both.sum() > 0:
        ratio = cp[both] / cp_entropy[both]
        print('\n2. cp cross-check: Path B (grad_ad) vs Path A (entropy)')
        print(f'   Points compared: {both.sum()}')
        print(f'   Median ratio: {np.median(ratio):.6f} (ideal: 1.0)')
        pct_close = (np.abs(ratio - 1) < 0.01).sum() / both.sum() * 100
        print(f'   Within 1%%: {pct_close:.1f}%')

    # Ideal gas check
    mask_ig = (np.abs(logT - 3.0) < 0.03) & (np.abs(logP - (-3.0)) < 0.03)
    if mask_ig.any():
        i = np.where(mask_ig)[0][0]
        R_H2 = 8.314 / 0.002016
        print('\n3. Ideal gas check at T=1000 K, P=1 MPa:')
        print(f'   alpha  = {alpha[i]:.4e} (ideal: {1 / T[i]:.4e})')
        print(f'   cp     = {cp[i]:.0f} J/kg/K (ideal H2 7/2 R/M: {3.5 * R_H2:.0f})')
        print(f'   cv     = {cv[i]:.0f} J/kg/K (ideal H2 5/2 R/M: {2.5 * R_H2:.0f})')
        print(f'   cp/cv  = {cp[i] / cv[i]:.4f} (ideal diatomic: {7 / 5:.4f})')

    # Phase statistics
    print('\n4. Phase identification:')
    for phase in ['molecular', 'dissociating', 'atomic', 'unphysical']:
        n = np.sum(np.char.startswith(si['phase_id'], phase))
        print(f'   {phase}: {n} ({n / len(T) * 100:.1f}%)')

    # Flag statistics
    n_cv_bad = np.sum(np.char.find(si['phase_id'], 'cv_unreliable') >= 0)
    print(f'   cv_unreliable flag: {n_cv_bad} ({n_cv_bad / len(T) * 100:.1f}%)')

    print()


def main():
    """Convert all Chabrier tables to PALEOS format."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'input_dir',
        help='Path to DirEOS2021 directory',
    )
    parser.add_argument(
        '-o',
        '--output-dir',
        default='.',
        help='Output directory (default: current directory)',
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tables to convert
    tables = [
        ('TABLE_H_TP_v1', 'H', 'Pure hydrogen (Chabrier+2019)'),
        ('TABLE_HE_TP_v1', 'HE', 'Pure helium (Chabrier+2019)'),
        (
            'TABLE_H_TP_effective',
            'H_effective',
            'Effective H for non-IVL mixing (Chabrier+Debras 2021). '
            'DO NOT use as a standalone pure H EOS.',
        ),
        (
            'TABLEEOS_2021_TP_Y0275_v1',
            'HHe_Y0275',
            'H/He mixture Y=0.275 (Chabrier+Debras 2021)',
        ),
        (
            'TABLEEOS_2021_TP_Y0292_v1',
            'HHe_Y0292',
            'H/He mixture Y=0.292 (Chabrier+Debras 2021)',
        ),
        (
            'TABLEEOS_2021_TP_Y0297_v1',
            'HHe_Y0297',
            'H/He mixture Y=0.297 (Chabrier+Debras 2021)',
        ),
    ]

    for filename, label, description in tables:
        filepath = input_dir / filename
        if not filepath.exists():
            print(f'Skipping {filename} (not found)')
            continue

        print(f'\n{"=" * 70}')
        print(f'Converting {filename} ({description})')
        print(f'{"=" * 70}')

        raw = load_chabrier_table(str(filepath))
        si = convert_units(raw)
        si = derive_thermodynamic_quantities(si)

        outpath = output_dir / f'chabrier2021_{label}.dat'
        write_paleos_format(si, str(outpath), label)
        print(f'Written to {outpath}')

        print_verification(si)


if __name__ == '__main__':
    main()
