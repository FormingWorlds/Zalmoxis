#!/usr/bin/env python3
"""Generate TOML config files for the Zalmoxis validation grid.

Produces ~400 runs across 9 suites, each a complete valid TOML config
based on input/default.toml with specific parameter overrides. Also
writes a manifest.csv with metadata for all runs.

Usage
-----
    python tools/validation_grid/generate_configs.py
"""

from __future__ import annotations

import csv
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import toml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_CONFIG = REPO_ROOT / 'input' / 'default.toml'
CONFIGS_DIR = SCRIPT_DIR / 'configs'


def load_default_config() -> dict:
    """Load the default TOML config as a nested dict."""
    if not DEFAULT_CONFIG.exists():
        sys.exit(f'Default config not found: {DEFAULT_CONFIG}')
    return toml.load(str(DEFAULT_CONFIG))


def make_config(
    base: dict,
    *,
    planet_mass: float,
    core_eos: str = 'PALEOS:iron',
    mantle_eos: str = 'PALEOS:MgSiO3',
    ice_layer_eos: str = '',
    temperature_mode: str = 'adiabatic',
    surface_temperature: float = 3000.0,
    center_temperature: float = 6000.0,
    core_mass_fraction: float = 0.325,
    mantle_mass_fraction: float = 0.0,
    mushy_zone_factor: float = 1.0,
    mushy_zone_factor_iron: float | None = None,
    mushy_zone_factor_MgSiO3: float | None = None,
    mushy_zone_factor_H2O: float | None = None,
    condensed_rho_min: float = 322.0,
    condensed_rho_scale: float = 50.0,
    rock_solidus: str = 'Monteux16-solidus',
    rock_liquidus: str = 'Monteux16-liquidus-A-chondritic',
    plots_enabled: bool = False,
    data_enabled: bool = True,
    verbose: bool = False,
) -> dict:
    """Build a complete config dict from defaults with specific overrides."""
    cfg = deepcopy(base)
    cfg['InputParameter']['planet_mass'] = planet_mass
    cfg['AssumptionsAndInitialGuesses']['core_mass_fraction'] = core_mass_fraction
    cfg['AssumptionsAndInitialGuesses']['mantle_mass_fraction'] = mantle_mass_fraction
    cfg['AssumptionsAndInitialGuesses']['temperature_mode'] = temperature_mode
    cfg['AssumptionsAndInitialGuesses']['surface_temperature'] = surface_temperature
    cfg['AssumptionsAndInitialGuesses']['center_temperature'] = center_temperature
    cfg['EOS']['core'] = core_eos
    cfg['EOS']['mantle'] = mantle_eos
    cfg['EOS']['ice_layer'] = ice_layer_eos
    cfg['EOS']['mushy_zone_factor'] = mushy_zone_factor
    cfg['EOS']['condensed_rho_min'] = condensed_rho_min
    cfg['EOS']['condensed_rho_scale'] = condensed_rho_scale
    cfg['EOS']['rock_solidus'] = rock_solidus
    cfg['EOS']['rock_liquidus'] = rock_liquidus
    cfg['Output']['plots_enabled'] = plots_enabled
    cfg['Output']['data_enabled'] = data_enabled
    cfg['Output']['verbose'] = verbose

    # Per-EOS mushy zone overrides: only write keys that are explicitly set
    for key in ('mushy_zone_factor_iron', 'mushy_zone_factor_MgSiO3', 'mushy_zone_factor_H2O'):
        if key in cfg['EOS']:
            del cfg['EOS'][key]

    if mushy_zone_factor_iron is not None:
        cfg['EOS']['mushy_zone_factor_iron'] = mushy_zone_factor_iron
    if mushy_zone_factor_MgSiO3 is not None:
        cfg['EOS']['mushy_zone_factor_MgSiO3'] = mushy_zone_factor_MgSiO3
    if mushy_zone_factor_H2O is not None:
        cfg['EOS']['mushy_zone_factor_H2O'] = mushy_zone_factor_H2O

    return cfg


# ---------------------------------------------------------------------------
# Manifest row helper
# ---------------------------------------------------------------------------
MANIFEST_COLUMNS = [
    'suite',
    'run_id',
    'config_path',
    'planet_mass',
    'core_eos',
    'mantle_eos',
    'ice_layer_eos',
    'temperature_mode',
    'surface_temperature',
    'center_temperature',
    'core_mass_fraction',
    'mantle_mass_fraction',
    'mushy_zone_factor',
    'mushy_zone_factor_iron',
    'mushy_zone_factor_MgSiO3',
    'mushy_zone_factor_H2O',
    'condensed_rho_min',
    'condensed_rho_scale',
    'h2o_fraction',
    'expected_convergence',
]


def _row(
    suite: str,
    run_id: str,
    config_path: str,
    *,
    planet_mass: float,
    core_eos: str,
    mantle_eos: str,
    ice_layer_eos: str = '',
    temperature_mode: str = 'adiabatic',
    surface_temperature: float = 3000.0,
    center_temperature: float = 6000.0,
    core_mass_fraction: float = 0.325,
    mantle_mass_fraction: float = 0.0,
    mushy_zone_factor: float = 1.0,
    mushy_zone_factor_iron: str = '',
    mushy_zone_factor_MgSiO3: str = '',
    mushy_zone_factor_H2O: str = '',
    condensed_rho_min: float = 322.0,
    condensed_rho_scale: float = 50.0,
    h2o_fraction: float = 0.0,
    expected_convergence: str = 'True',
) -> dict:
    return {
        'suite': suite,
        'run_id': run_id,
        'config_path': config_path,
        'planet_mass': planet_mass,
        'core_eos': core_eos,
        'mantle_eos': mantle_eos,
        'ice_layer_eos': ice_layer_eos,
        'temperature_mode': temperature_mode,
        'surface_temperature': surface_temperature,
        'center_temperature': center_temperature,
        'core_mass_fraction': core_mass_fraction,
        'mantle_mass_fraction': mantle_mass_fraction,
        'mushy_zone_factor': mushy_zone_factor,
        'mushy_zone_factor_iron': mushy_zone_factor_iron,
        'mushy_zone_factor_MgSiO3': mushy_zone_factor_MgSiO3,
        'mushy_zone_factor_H2O': mushy_zone_factor_H2O,
        'condensed_rho_min': condensed_rho_min,
        'condensed_rho_scale': condensed_rho_scale,
        'h2o_fraction': h2o_fraction,
        'expected_convergence': expected_convergence,
    }


def write_config(cfg: dict, path: Path) -> None:
    """Write a TOML config dict to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        toml.dump(cfg, f)


# ---------------------------------------------------------------------------
# Suite generators
# ---------------------------------------------------------------------------


def suite_01_mass_radius(base: dict) -> tuple[list[dict], list[dict]]:
    """Suite 1: Mass-radius baseline.

    Two temperature modes (adiabatic 3000K, isothermal 300K) across 13 masses.
    """
    suite_name = 'suite_01_mass_radius'
    suite_dir = CONFIGS_DIR / suite_name
    masses = [0.1, 0.3, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30, 50]
    configs_written = []
    manifest_rows = []
    run_num = 0

    for mass in masses:
        # Config A: adiabatic 3000K
        run_num += 1
        run_id = f'run_{run_num:03d}_M{mass}_adiabatic_3000K'
        cfg = make_config(
            base, planet_mass=mass, temperature_mode='adiabatic', surface_temperature=3000.0
        )
        path = suite_dir / f'{run_id}.toml'
        write_config(cfg, path)
        configs_written.append(cfg)
        manifest_rows.append(
            _row(
                suite_name,
                run_id,
                str(path.relative_to(SCRIPT_DIR)),
                planet_mass=mass,
                core_eos='PALEOS:iron',
                mantle_eos='PALEOS:MgSiO3',
                temperature_mode='adiabatic',
                surface_temperature=3000.0,
            )
        )

        # Config B: isothermal 300K
        run_num += 1
        run_id = f'run_{run_num:03d}_M{mass}_isothermal_300K'
        cfg = make_config(
            base, planet_mass=mass, temperature_mode='isothermal', surface_temperature=300.0
        )
        path = suite_dir / f'{run_id}.toml'
        write_config(cfg, path)
        configs_written.append(cfg)
        manifest_rows.append(
            _row(
                suite_name,
                run_id,
                str(path.relative_to(SCRIPT_DIR)),
                planet_mass=mass,
                core_eos='PALEOS:iron',
                mantle_eos='PALEOS:MgSiO3',
                temperature_mode='isothermal',
                surface_temperature=300.0,
            )
        )

    return configs_written, manifest_rows


def _mantle_eos_with_h2o(h2o_frac: float) -> str:
    """Build a mantle EOS string with optional H2O mixing fraction."""
    if h2o_frac <= 0:
        return 'PALEOS:MgSiO3'
    rock_frac = 1.0 - h2o_frac
    return f'PALEOS:MgSiO3:{rock_frac:.2f}+PALEOS:H2O:{h2o_frac:.2f}'


def suite_02_mixing(base: dict) -> tuple[list[dict], list[dict]]:
    """Suite 2: Mixing fraction sweep.

    5 masses x 7 H2O fractions x 2 surface temperatures.
    """
    suite_name = 'suite_02_mixing'
    suite_dir = CONFIGS_DIR / suite_name
    masses = [0.5, 1, 2, 5, 10]
    h2o_fractions = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    t_surfs = [2000.0, 3000.0]
    configs_written = []
    manifest_rows = []
    run_num = 0

    for mass in masses:
        for h2o_frac in h2o_fractions:
            for t_surf in t_surfs:
                run_num += 1
                h2o_pct = int(h2o_frac * 100)
                run_id = f'run_{run_num:03d}_M{mass}_H2O{h2o_pct}pct_T{int(t_surf)}K'
                mantle_eos = _mantle_eos_with_h2o(h2o_frac)
                cfg = make_config(
                    base,
                    planet_mass=mass,
                    mantle_eos=mantle_eos,
                    temperature_mode='adiabatic',
                    surface_temperature=t_surf,
                )
                path = suite_dir / f'{run_id}.toml'
                write_config(cfg, path)
                configs_written.append(cfg)
                # High H2O at high mass can fail convergence
                expected = 'True'
                if mass >= 10 and h2o_frac >= 0.15:
                    expected = 'Unknown'
                manifest_rows.append(
                    _row(
                        suite_name,
                        run_id,
                        str(path.relative_to(SCRIPT_DIR)),
                        planet_mass=mass,
                        core_eos='PALEOS:iron',
                        mantle_eos=mantle_eos,
                        temperature_mode='adiabatic',
                        surface_temperature=t_surf,
                        h2o_fraction=h2o_frac,
                        expected_convergence=expected,
                    )
                )

    return configs_written, manifest_rows


def suite_03_temperature(base: dict) -> tuple[list[dict], list[dict]]:
    """Suite 3: Surface temperature sweep.

    4 masses x 6 T_surf x 2 compositions. Isothermal for 300K, adiabatic for rest.
    """
    suite_name = 'suite_03_temperature'
    suite_dir = CONFIGS_DIR / suite_name
    masses = [0.5, 1, 5, 10]
    t_surfs = [300, 1000, 2000, 3000, 4000, 5000]
    compositions = [
        ('pure', 'PALEOS:MgSiO3', 0.0),
        ('h2o15', 'PALEOS:MgSiO3:0.85+PALEOS:H2O:0.15', 0.15),
    ]
    configs_written = []
    manifest_rows = []
    run_num = 0

    for mass in masses:
        for t_surf in t_surfs:
            t_mode = 'isothermal' if t_surf == 300 else 'adiabatic'
            for comp_label, mantle_eos, h2o_frac in compositions:
                run_num += 1
                run_id = f'run_{run_num:03d}_M{mass}_{comp_label}_T{t_surf}K_{t_mode}'
                cfg = make_config(
                    base,
                    planet_mass=mass,
                    mantle_eos=mantle_eos,
                    temperature_mode=t_mode,
                    surface_temperature=float(t_surf),
                )
                path = suite_dir / f'{run_id}.toml'
                write_config(cfg, path)
                configs_written.append(cfg)
                manifest_rows.append(
                    _row(
                        suite_name,
                        run_id,
                        str(path.relative_to(SCRIPT_DIR)),
                        planet_mass=mass,
                        core_eos='PALEOS:iron',
                        mantle_eos=mantle_eos,
                        temperature_mode=t_mode,
                        surface_temperature=float(t_surf),
                        h2o_fraction=h2o_frac,
                    )
                )

    return configs_written, manifest_rows


def suite_04_mushy_zone(base: dict) -> tuple[list[dict], list[dict]]:
    """Suite 4: Mushy zone factor sweep.

    3 masses x 4 global MZF x 2 compositions + 3 masses x 2 per-EOS configs.
    """
    suite_name = 'suite_04_mushy_zone'
    suite_dir = CONFIGS_DIR / suite_name
    masses = [1, 5, 10]
    configs_written = []
    manifest_rows = []
    run_num = 0

    # Global MZF sweep: pure mantle
    global_mzfs = [1.0, 0.9, 0.8, 0.7]
    for mass in masses:
        for mzf in global_mzfs:
            run_num += 1
            run_id = f'run_{run_num:03d}_M{mass}_pure_MZF{mzf:.1f}'
            cfg = make_config(
                base,
                planet_mass=mass,
                temperature_mode='adiabatic',
                surface_temperature=3000.0,
                mushy_zone_factor=mzf,
            )
            path = suite_dir / f'{run_id}.toml'
            write_config(cfg, path)
            configs_written.append(cfg)
            manifest_rows.append(
                _row(
                    suite_name,
                    run_id,
                    str(path.relative_to(SCRIPT_DIR)),
                    planet_mass=mass,
                    core_eos='PALEOS:iron',
                    mantle_eos='PALEOS:MgSiO3',
                    temperature_mode='adiabatic',
                    surface_temperature=3000.0,
                    mushy_zone_factor=mzf,
                )
            )

    # Global MZF sweep: 15% H2O mantle
    mantle_h2o = 'PALEOS:MgSiO3:0.85+PALEOS:H2O:0.15'
    for mass in masses:
        for mzf in global_mzfs:
            run_num += 1
            run_id = f'run_{run_num:03d}_M{mass}_h2o15_MZF{mzf:.1f}'
            cfg = make_config(
                base,
                planet_mass=mass,
                mantle_eos=mantle_h2o,
                temperature_mode='adiabatic',
                surface_temperature=3000.0,
                mushy_zone_factor=mzf,
            )
            path = suite_dir / f'{run_id}.toml'
            write_config(cfg, path)
            configs_written.append(cfg)
            manifest_rows.append(
                _row(
                    suite_name,
                    run_id,
                    str(path.relative_to(SCRIPT_DIR)),
                    planet_mass=mass,
                    core_eos='PALEOS:iron',
                    mantle_eos=mantle_h2o,
                    temperature_mode='adiabatic',
                    surface_temperature=3000.0,
                    mushy_zone_factor=mzf,
                    h2o_fraction=0.15,
                )
            )

    # Per-EOS configs
    per_eos_configs = [
        {'iron': 1.0, 'MgSiO3': 0.8, 'H2O': 1.0, 'label': 'MgSiO3_0.8'},
        {'iron': 0.9, 'MgSiO3': 0.7, 'H2O': 1.0, 'label': 'iron0.9_MgSiO3_0.7'},
    ]
    for mass in masses:
        for pec in per_eos_configs:
            run_num += 1
            run_id = f'run_{run_num:03d}_M{mass}_perEOS_{pec["label"]}'
            cfg = make_config(
                base,
                planet_mass=mass,
                temperature_mode='adiabatic',
                surface_temperature=3000.0,
                mushy_zone_factor=1.0,
                mushy_zone_factor_iron=pec['iron'],
                mushy_zone_factor_MgSiO3=pec['MgSiO3'],
                mushy_zone_factor_H2O=pec['H2O'],
            )
            path = suite_dir / f'{run_id}.toml'
            write_config(cfg, path)
            configs_written.append(cfg)
            manifest_rows.append(
                _row(
                    suite_name,
                    run_id,
                    str(path.relative_to(SCRIPT_DIR)),
                    planet_mass=mass,
                    core_eos='PALEOS:iron',
                    mantle_eos='PALEOS:MgSiO3',
                    temperature_mode='adiabatic',
                    surface_temperature=3000.0,
                    mushy_zone_factor=1.0,
                    mushy_zone_factor_iron=str(pec['iron']),
                    mushy_zone_factor_MgSiO3=str(pec['MgSiO3']),
                    mushy_zone_factor_H2O=str(pec['H2O']),
                )
            )

    return configs_written, manifest_rows


def suite_05_three_layer(base: dict) -> tuple[list[dict], list[dict]]:
    """Suite 5: Three-layer models.

    4 masses x 3 layer splits x 2 temperature modes.
    """
    suite_name = 'suite_05_three_layer'
    suite_dir = CONFIGS_DIR / suite_name
    masses = [0.5, 1, 5, 10]
    # (CMF, MMF) pairs. Ice mass fraction = 1 - CMF - MMF.
    layer_splits = [
        (0.325, 0.50, 'cmf0.325_mmf0.50'),
        (0.25, 0.25, 'cmf0.25_mmf0.25'),
        (0.10, 0.40, 'cmf0.10_mmf0.40'),
    ]
    t_modes = [
        ('isothermal', 300.0, 'iso300K'),
        ('adiabatic', 1000.0, 'adi1000K'),
    ]
    configs_written = []
    manifest_rows = []
    run_num = 0

    for mass in masses:
        for cmf, mmf, split_label in layer_splits:
            for t_mode, t_surf, t_label in t_modes:
                run_num += 1
                run_id = f'run_{run_num:03d}_M{mass}_{split_label}_{t_label}'
                cfg = make_config(
                    base,
                    planet_mass=mass,
                    core_eos='PALEOS:iron',
                    mantle_eos='PALEOS:MgSiO3',
                    ice_layer_eos='PALEOS:H2O',
                    core_mass_fraction=cmf,
                    mantle_mass_fraction=mmf,
                    temperature_mode=t_mode,
                    surface_temperature=t_surf,
                )
                path = suite_dir / f'{run_id}.toml'
                write_config(cfg, path)
                configs_written.append(cfg)
                manifest_rows.append(
                    _row(
                        suite_name,
                        run_id,
                        str(path.relative_to(SCRIPT_DIR)),
                        planet_mass=mass,
                        core_eos='PALEOS:iron',
                        mantle_eos='PALEOS:MgSiO3',
                        ice_layer_eos='PALEOS:H2O',
                        temperature_mode=t_mode,
                        surface_temperature=t_surf,
                        core_mass_fraction=cmf,
                        mantle_mass_fraction=mmf,
                    )
                )

    return configs_written, manifest_rows


def suite_06_max_mass(base: dict) -> tuple[list[dict], list[dict]]:
    """Suite 6: Maximum mass stress test.

    4 masses x 3 T_surf x 2 compositions.
    """
    suite_name = 'suite_06_max_mass'
    suite_dir = CONFIGS_DIR / suite_name
    masses = [20, 30, 40, 50]
    t_surfs = [1000, 2000, 3000]
    compositions = [
        ('pure', 'PALEOS:MgSiO3', 0.0),
        ('h2o15', 'PALEOS:MgSiO3:0.85+PALEOS:H2O:0.15', 0.15),
    ]
    configs_written = []
    manifest_rows = []
    run_num = 0

    for mass in masses:
        for t_surf in t_surfs:
            for comp_label, mantle_eos, h2o_frac in compositions:
                run_num += 1
                run_id = f'run_{run_num:03d}_M{mass}_{comp_label}_T{t_surf}K'
                cfg = make_config(
                    base,
                    planet_mass=mass,
                    mantle_eos=mantle_eos,
                    temperature_mode='adiabatic',
                    surface_temperature=float(t_surf),
                )
                path = suite_dir / f'{run_id}.toml'
                write_config(cfg, path)
                configs_written.append(cfg)
                manifest_rows.append(
                    _row(
                        suite_name,
                        run_id,
                        str(path.relative_to(SCRIPT_DIR)),
                        planet_mass=mass,
                        core_eos='PALEOS:iron',
                        mantle_eos=mantle_eos,
                        temperature_mode='adiabatic',
                        surface_temperature=float(t_surf),
                        h2o_fraction=h2o_frac,
                    )
                )

    return configs_written, manifest_rows


def suite_07_exotic(base: dict) -> tuple[list[dict], list[dict]]:
    """Suite 7: Exotic architectures.

    Physically unusual configs that should either converge or fail gracefully.
    All isothermal 300K for simplicity.
    """
    suite_name = 'suite_07_exotic'
    suite_dir = CONFIGS_DIR / suite_name
    configs_written = []
    manifest_rows = []
    run_num = 0

    def _add(
        run_id_suffix,
        mass,
        core_eos,
        mantle_eos,
        cmf=0.325,
        mmf=0.0,
        ice_layer_eos='',
        expected='Unknown',
    ):
        nonlocal run_num
        run_num += 1
        run_id = f'run_{run_num:03d}_{run_id_suffix}'
        cfg = make_config(
            base,
            planet_mass=mass,
            core_eos=core_eos,
            mantle_eos=mantle_eos,
            ice_layer_eos=ice_layer_eos,
            core_mass_fraction=cmf,
            mantle_mass_fraction=mmf,
            temperature_mode='isothermal',
            surface_temperature=300.0,
        )
        path = suite_dir / f'{run_id}.toml'
        write_config(cfg, path)
        configs_written.append(cfg)
        h2o_frac = 0.0
        if 'H2O' in mantle_eos or 'H2O' in ice_layer_eos:
            h2o_frac = -1.0  # marker for "contains H2O, fraction varies"
        manifest_rows.append(
            _row(
                suite_name,
                run_id,
                str(path.relative_to(SCRIPT_DIR)),
                planet_mass=mass,
                core_eos=core_eos,
                mantle_eos=mantle_eos,
                ice_layer_eos=ice_layer_eos,
                temperature_mode='isothermal',
                surface_temperature=300.0,
                core_mass_fraction=cmf,
                mantle_mass_fraction=mmf,
                h2o_fraction=h2o_frac,
                expected_convergence=expected,
            )
        )

    # Pure iron planet (CMF~1.0, both layers = iron)
    # cmf must be in (0, 1]; using 0.999 so the mantle gets ~0 mass.
    for mass in [0.5, 1, 5, 10]:
        _add(f'pure_iron_M{mass}', mass, 'PALEOS:iron', 'PALEOS:iron', cmf=0.999)

    # Pure iron planet with analytic EOS
    for mass in [0.5, 1, 5, 10]:
        _add(f'pure_iron_analytic_M{mass}', mass, 'Analytic:iron', 'Analytic:iron', cmf=0.999)

    # Pure rock planet (CMF=0.01)
    for mass in [0.5, 1, 5, 10]:
        _add(
            f'pure_rock_M{mass}',
            mass,
            'PALEOS:iron',
            'PALEOS:MgSiO3',
            cmf=0.01,
            expected='True',
        )

    # Pure water planet (3-layer, CMF=0.01, MMF=0.01, ice=PALEOS:H2O)
    for mass in [0.5, 1]:
        _add(
            f'pure_water_M{mass}',
            mass,
            'PALEOS:iron',
            'PALEOS:MgSiO3',
            cmf=0.01,
            mmf=0.01,
            ice_layer_eos='PALEOS:H2O',
        )

    # Swapped layers (core=MgSiO3, mantle=iron)
    for mass in [1, 5]:
        _add(f'swapped_M{mass}', mass, 'PALEOS:MgSiO3', 'PALEOS:iron', cmf=0.325)

    # Very small core (CMF=0.05)
    for mass in [1, 5]:
        _add(
            f'small_core_M{mass}',
            mass,
            'PALEOS:iron',
            'PALEOS:MgSiO3',
            cmf=0.05,
            expected='True',
        )

    # Very large core (CMF=0.9)
    for mass in [1, 5]:
        _add(
            f'large_core_M{mass}',
            mass,
            'PALEOS:iron',
            'PALEOS:MgSiO3',
            cmf=0.9,
            expected='True',
        )

    # H2O core
    for mass in [1, 5]:
        _add(f'h2o_core_M{mass}', mass, 'PALEOS:H2O', 'PALEOS:MgSiO3', cmf=0.325)

    # Mixed core (iron+MgSiO3)
    for mass in [1, 5]:
        _add(
            f'mixed_core_M{mass}',
            mass,
            'PALEOS:iron:0.70+PALEOS:MgSiO3:0.30',
            'PALEOS:MgSiO3',
            cmf=0.325,
        )

    # Pure H2O mantle (no mixing, Seager2007 isothermal)
    for mass in [0.5, 1]:
        _add(f'h2o_mantle_seager_M{mass}', mass, 'Seager2007:iron', 'Seager2007:H2O', cmf=0.325)

    # Analytic graphite + SiC planet
    for mass in [1, 5]:
        _add(f'graphite_SiC_M{mass}', mass, 'Analytic:graphite', 'Analytic:SiC', cmf=0.325)

    return configs_written, manifest_rows


def suite_08_temperature_modes(base: dict) -> tuple[list[dict], list[dict]]:
    """Suite 8: Temperature mode comparison.

    4 masses x 3 modes x 2 compositions.
    """
    suite_name = 'suite_08_temperature_modes'
    suite_dir = CONFIGS_DIR / suite_name
    masses = [0.5, 1, 5, 10]
    modes = [
        ('isothermal', 3000.0, 6000.0, 'iso3000K'),
        ('linear', 3000.0, 6000.0, 'lin3000_6000K'),
        ('adiabatic', 3000.0, 6000.0, 'adi3000K'),
    ]
    compositions = [
        ('pure', 'PALEOS:MgSiO3', 0.0),
        ('h2o15', 'PALEOS:MgSiO3:0.85+PALEOS:H2O:0.15', 0.15),
    ]
    configs_written = []
    manifest_rows = []
    run_num = 0

    for mass in masses:
        for t_mode, t_surf, t_center, mode_label in modes:
            for comp_label, mantle_eos, h2o_frac in compositions:
                run_num += 1
                run_id = f'run_{run_num:03d}_M{mass}_{comp_label}_{mode_label}'
                cfg = make_config(
                    base,
                    planet_mass=mass,
                    mantle_eos=mantle_eos,
                    temperature_mode=t_mode,
                    surface_temperature=t_surf,
                    center_temperature=t_center,
                )
                path = suite_dir / f'{run_id}.toml'
                write_config(cfg, path)
                configs_written.append(cfg)
                manifest_rows.append(
                    _row(
                        suite_name,
                        run_id,
                        str(path.relative_to(SCRIPT_DIR)),
                        planet_mass=mass,
                        core_eos='PALEOS:iron',
                        mantle_eos=mantle_eos,
                        temperature_mode=t_mode,
                        surface_temperature=t_surf,
                        center_temperature=t_center,
                        h2o_fraction=h2o_frac,
                    )
                )

    return configs_written, manifest_rows


def suite_09_legacy_eos(base: dict) -> tuple[list[dict], list[dict]]:
    """Suite 9: Legacy EOS comparison.

    Compare Seager2007, Analytic, WolfBower2018, and PALEOS across multiple masses.
    WolfBower2018 is limited to mass <= 5 M_earth (table ceiling at ~1 TPa).
    """
    suite_name = 'suite_09_legacy_eos'
    suite_dir = CONFIGS_DIR / suite_name
    masses = [0.5, 1, 5, 10]
    configs_written = []
    manifest_rows = []
    run_num = 0

    eos_configs = [
        # (core_eos, mantle_eos, t_mode, t_surf, label,
        #  rock_solidus, rock_liquidus, max_mass)
        (
            'Seager2007:iron',
            'Seager2007:MgSiO3',
            'isothermal',
            300.0,
            'seager_iso300K',
            'Monteux16-solidus',
            'Monteux16-liquidus-A-chondritic',
            999,
        ),
        (
            'Analytic:iron',
            'Analytic:MgSiO3',
            'isothermal',
            300.0,
            'analytic_iso300K',
            'Monteux16-solidus',
            'Monteux16-liquidus-A-chondritic',
            999,
        ),
        (
            'Seager2007:iron',
            'WolfBower2018:MgSiO3',
            'adiabatic',
            3000.0,
            'WB2018_adi3000K',
            'Monteux16-solidus',
            'Monteux16-liquidus-A-chondritic',
            5,
        ),
        (
            'PALEOS:iron',
            'PALEOS:MgSiO3',
            'isothermal',
            300.0,
            'paleos_iso300K',
            'Monteux16-solidus',
            'Monteux16-liquidus-A-chondritic',
            999,
        ),
        (
            'PALEOS:iron',
            'PALEOS:MgSiO3',
            'adiabatic',
            3000.0,
            'paleos_adi3000K',
            'Monteux16-solidus',
            'Monteux16-liquidus-A-chondritic',
            999,
        ),
    ]

    for mass in masses:
        for (
            core_eos,
            mantle_eos,
            t_mode,
            t_surf,
            label,
            solidus,
            liquidus,
            max_mass,
        ) in eos_configs:
            if mass > max_mass:
                continue
            run_num += 1
            run_id = f'run_{run_num:03d}_M{mass}_{label}'
            cfg = make_config(
                base,
                planet_mass=mass,
                core_eos=core_eos,
                mantle_eos=mantle_eos,
                temperature_mode=t_mode,
                surface_temperature=t_surf,
                rock_solidus=solidus,
                rock_liquidus=liquidus,
            )
            path = suite_dir / f'{run_id}.toml'
            write_config(cfg, path)
            configs_written.append(cfg)
            manifest_rows.append(
                _row(
                    suite_name,
                    run_id,
                    str(path.relative_to(SCRIPT_DIR)),
                    planet_mass=mass,
                    core_eos=core_eos,
                    mantle_eos=mantle_eos,
                    temperature_mode=t_mode,
                    surface_temperature=t_surf,
                )
            )

    return configs_written, manifest_rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    """Generate all validation grid configs and write manifest.csv."""
    base = load_default_config()

    # Clean existing configs
    if CONFIGS_DIR.exists():
        shutil.rmtree(CONFIGS_DIR)

    suite_generators = [
        ('Suite 1: Mass-radius baseline', suite_01_mass_radius),
        ('Suite 2: Mixing fraction sweep', suite_02_mixing),
        ('Suite 3: Surface temperature sweep', suite_03_temperature),
        ('Suite 4: Mushy zone factor sweep', suite_04_mushy_zone),
        ('Suite 5: Three-layer models', suite_05_three_layer),
        ('Suite 6: Maximum mass stress', suite_06_max_mass),
        ('Suite 7: Exotic architectures', suite_07_exotic),
        ('Suite 8: Temperature mode comparison', suite_08_temperature_modes),
        ('Suite 9: Legacy EOS comparison', suite_09_legacy_eos),
    ]

    all_rows = []
    suite_counts = []

    for suite_label, generator in suite_generators:
        configs, rows = generator(base)
        all_rows.extend(rows)
        suite_counts.append((suite_label, len(rows)))

    # Write manifest
    manifest_path = CONFIGS_DIR / 'manifest.csv'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary
    print('=' * 60)
    print('Zalmoxis Validation Grid: Config Generation Summary')
    print('=' * 60)
    total = 0
    for label, count in suite_counts:
        print(f'  {label:.<45} {count:>4} runs')
        total += count
    print('-' * 60)
    print(f'  {"Grand total":.<45} {total:>4} runs')
    print(f'\n  Configs written to: {CONFIGS_DIR}')
    print(f'  Manifest written to: {manifest_path}')


if __name__ == '__main__':
    main()
