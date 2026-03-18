#!/usr/bin/env python3
"""Run a single Zalmoxis validation grid config and record results.

Takes a TOML config path, runs Zalmoxis, captures output, and writes
a JSON result file alongside the profile data and log.

Usage
-----
    python tools/validation_grid/run_single.py configs/suite_01_mass_radius/run_001_M0.1_adiabatic_3000K.toml

Output directory is created under tools/validation_grid/results/ mirroring
the config path structure.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


def run_single(config_path_str: str) -> None:
    """Run Zalmoxis with a single config and capture results.

    Parameters
    ----------
    config_path_str : str
        Path to the TOML config file (absolute or relative to SCRIPT_DIR).
    """
    config_path = Path(config_path_str).resolve()
    if not config_path.exists():
        sys.exit(f'Config not found: {config_path}')

    # Derive run name and output directory from config path
    # Expected: .../configs/suite_XX_name/run_NNN_label.toml
    suite_name = config_path.parent.name
    run_name = config_path.stem

    # Use ZALMOXIS_RESULTS_DIR if set (e.g., scratch on Habrok), else local
    results_base = Path(
        os.environ.get(
            'ZALMOXIS_RESULTS_DIR',
            str(SCRIPT_DIR / 'results'),
        )
    )
    results_dir = results_base / suite_name / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to file
    log_path = results_dir / 'run.log'
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filemode='w',
        force=True,
    )
    logger = logging.getLogger(__name__)
    logger.info(f'Config: {config_path}')
    logger.info(f'Results: {results_dir}')

    # Set ZALMOXIS_ROOT so the module can find its data
    os.environ['ZALMOXIS_ROOT'] = str(REPO_ROOT)

    # Redirect Zalmoxis output_files to our results directory by
    # temporarily replacing the output path. We do this by creating a
    # local output_files directory and symlinking, then moving files after.
    local_output = results_dir / 'output_files'
    local_output.mkdir(exist_ok=True)

    result = {
        'config': str(config_path),
        'suite': suite_name,
        'run': run_name,
        'converged': False,
        'converged_pressure': False,
        'converged_density': False,
        'converged_mass': False,
        'radius_m': None,
        'radius_rearth': None,
        'mass_kg': None,
        'mass_mearth': None,
        'total_time_s': None,
        'error': None,
    }

    t_start = time.perf_counter()

    try:
        # Import Zalmoxis components
        from zalmoxis.constants import earth_mass, earth_radius
        from zalmoxis.zalmoxis import (
            load_material_dictionaries,
            load_solidus_liquidus_functions,
            load_zalmoxis_config,
            main,
        )

        # Load and validate config
        config_params = load_zalmoxis_config(temp_config_path=str(config_path))

        # Disable plotting, keep data output
        config_params['plotting_enabled'] = False
        config_params['data_output_enabled'] = True

        # Load EOS and melting curves
        layer_eos_config = config_params['layer_eos_config']
        solidus_id = config_params.get('rock_solidus', 'Stixrude14-solidus')
        liquidus_id = config_params.get('rock_liquidus', 'Stixrude14-liquidus')
        material_dicts = load_material_dictionaries()
        melting_funcs = load_solidus_liquidus_functions(
            layer_eos_config, solidus_id, liquidus_id
        )

        # Run the solver
        model_results = main(
            config_params,
            material_dictionaries=material_dicts,
            melting_curves_functions=melting_funcs,
            input_dir=str(REPO_ROOT / 'input'),
        )

        t_elapsed = time.perf_counter() - t_start

        # Extract key results
        radii = model_results['radii']
        mass_enclosed = model_results['mass_enclosed']
        result['converged'] = bool(model_results['converged'])
        result['converged_pressure'] = bool(model_results['converged_pressure'])
        result['converged_density'] = bool(model_results['converged_density'])
        result['converged_mass'] = bool(model_results['converged_mass'])
        result['radius_m'] = float(radii[-1])
        result['radius_rearth'] = float(radii[-1] / earth_radius)
        result['mass_kg'] = float(mass_enclosed[-1])
        result['mass_mearth'] = float(mass_enclosed[-1] / earth_mass)
        result['total_time_s'] = round(t_elapsed, 3)

        # Save the profile data
        import numpy as np

        output_data = np.column_stack(
            (
                radii,
                model_results['density'],
                model_results['gravity'],
                model_results['pressure'],
                model_results['temperature'],
                mass_enclosed,
            )
        )
        header = (
            'Radius (m)\tDensity (kg/m^3)\tGravity (m/s^2)\t'
            'Pressure (Pa)\tTemperature (K)\tMass Enclosed (kg)'
        )
        np.savetxt(str(results_dir / 'planet_profile.txt'), output_data, header=header)

        logger.info(
            f'Converged: {result["converged"]}, '
            f'R = {result["radius_rearth"]:.4f} R_earth, '
            f'M = {result["mass_mearth"]:.4f} M_earth, '
            f'time = {t_elapsed:.2f} s'
        )

    except Exception as exc:
        t_elapsed = time.perf_counter() - t_start
        result['total_time_s'] = round(t_elapsed, 3)
        result['error'] = f'{type(exc).__name__}: {exc}'
        tb = traceback.format_exc()
        logger.error(f'Run failed: {exc}\n{tb}')

    # Write result JSON
    result_path = results_dir / 'result.json'
    with open(result_path, 'w') as f:
        json.dump(result, f, indent=2)

    # Print summary to stdout
    status = 'CONVERGED' if result['converged'] else 'FAILED'
    if result['error']:
        status = 'ERROR'
    r_str = f'{result["radius_rearth"]:.4f}' if result['radius_rearth'] else 'N/A'
    t_str = f'{result["total_time_s"]:.2f}' if result['total_time_s'] else 'N/A'
    print(f'[{status}] {run_name}: R={r_str} R_earth, t={t_str}s')

    if result['error']:
        print(f'  Error: {result["error"]}')


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit(f'Usage: {sys.argv[0]} <config.toml>')
    run_single(sys.argv[1])
