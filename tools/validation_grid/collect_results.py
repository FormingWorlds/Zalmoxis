#!/usr/bin/env python3
"""Collect results from all validation grid runs into a summary CSV.

Walks the results/ directory, reads result.json files, merges them
with the manifest, and writes results/summary.csv. Prints a table
of convergence rates per suite.

Usage
-----
    python tools/validation_grid/collect_results.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / 'results'
CONFIGS_DIR = SCRIPT_DIR / 'configs'
MANIFEST_PATH = CONFIGS_DIR / 'manifest.csv'


def load_manifest() -> dict[str, dict]:
    """Load the manifest CSV keyed by run_id.

    Returns
    -------
    dict
        {run_id: {column: value, ...}}
    """
    if not MANIFEST_PATH.exists():
        print(f'Manifest not found: {MANIFEST_PATH}')
        print('Run generate_configs.py first.')
        return {}
    manifest = {}
    with open(MANIFEST_PATH) as f:
        reader = csv.DictReader(f)
        for row in reader:
            manifest[row['run_id']] = row
    return manifest


def collect_results() -> list[dict]:
    """Walk results/ and read all result.json files.

    Returns
    -------
    list[dict]
        One dict per run with all result fields.
    """
    results = []
    if not RESULTS_DIR.exists():
        print(f'Results directory not found: {RESULTS_DIR}')
        return results

    for result_json in sorted(RESULTS_DIR.rglob('result.json')):
        with open(result_json) as f:
            data = json.load(f)
        results.append(data)
    return results


def merge_and_write(manifest: dict[str, dict], results: list[dict]) -> Path:
    """Merge manifest metadata with run results and write summary.csv.

    Parameters
    ----------
    manifest : dict
        Manifest data keyed by run_id.
    results : list[dict]
        Result data from result.json files.

    Returns
    -------
    Path
        Path to the written summary.csv.
    """
    # Build merged rows
    summary_columns = [
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
        'converged',
        'converged_pressure',
        'converged_density',
        'converged_mass',
        'radius_m',
        'radius_rearth',
        'mass_kg',
        'mass_mearth',
        'total_time_s',
        'error',
    ]

    merged = []
    for res in results:
        run_id = res.get('run', '')
        row = {}
        # Fill from manifest
        if run_id in manifest:
            row.update(manifest[run_id])
        # Overwrite/add from results
        row['run_id'] = run_id
        row['suite'] = res.get('suite', row.get('suite', ''))
        row['converged'] = res.get('converged', False)
        row['converged_pressure'] = res.get('converged_pressure', False)
        row['converged_density'] = res.get('converged_density', False)
        row['converged_mass'] = res.get('converged_mass', False)
        row['radius_m'] = res.get('radius_m', '')
        row['radius_rearth'] = res.get('radius_rearth', '')
        row['mass_kg'] = res.get('mass_kg', '')
        row['mass_mearth'] = res.get('mass_mearth', '')
        row['total_time_s'] = res.get('total_time_s', '')
        row['error'] = res.get('error', '')
        merged.append(row)

    summary_path = RESULTS_DIR / 'summary.csv'
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=summary_columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(merged)

    return summary_path


def print_convergence_table(results: list[dict]) -> None:
    """Print a summary table of convergence rates per suite."""
    suite_stats: dict[str, dict] = {}
    for res in results:
        suite = res.get('suite', 'unknown')
        if suite not in suite_stats:
            suite_stats[suite] = {'total': 0, 'converged': 0, 'failed': 0, 'error': 0}
        suite_stats[suite]['total'] += 1
        if res.get('error'):
            suite_stats[suite]['error'] += 1
        elif res.get('converged'):
            suite_stats[suite]['converged'] += 1
        else:
            suite_stats[suite]['failed'] += 1

    print('=' * 75)
    print('Validation Grid: Convergence Summary')
    print('=' * 75)
    print(f'{"Suite":<35} {"Total":>6} {"Conv":>6} {"Fail":>6} {"Error":>6} {"Rate":>7}')
    print('-' * 75)

    totals = {'total': 0, 'converged': 0, 'failed': 0, 'error': 0}
    for suite in sorted(suite_stats.keys()):
        s = suite_stats[suite]
        rate = s['converged'] / s['total'] * 100 if s['total'] > 0 else 0
        print(
            f'{suite:<35} {s["total"]:>6} {s["converged"]:>6} {s["failed"]:>6} '
            f'{s["error"]:>6} {rate:>6.1f}%'
        )
        for k in totals:
            totals[k] += s[k]

    print('-' * 75)
    total_rate = totals['converged'] / totals['total'] * 100 if totals['total'] > 0 else 0
    print(
        f'{"TOTAL":<35} {totals["total"]:>6} {totals["converged"]:>6} '
        f'{totals["failed"]:>6} {totals["error"]:>6} {total_rate:>6.1f}%'
    )

    # Flag failed runs
    failed_runs = [r for r in results if not r.get('converged') and not r.get('error')]
    error_runs = [r for r in results if r.get('error')]

    if failed_runs:
        print(f'\nNon-converged runs ({len(failed_runs)}):')
        for r in failed_runs[:20]:
            print(f'  {r["suite"]}/{r["run"]}')
        if len(failed_runs) > 20:
            print(f'  ... and {len(failed_runs) - 20} more')

    if error_runs:
        print(f'\nRuns with errors ({len(error_runs)}):')
        for r in error_runs[:20]:
            err = r.get('error', '')[:80]
            print(f'  {r["suite"]}/{r["run"]}: {err}')
        if len(error_runs) > 20:
            print(f'  ... and {len(error_runs) - 20} more')


def main():
    """Collect results and write summary."""
    manifest = load_manifest()
    results = collect_results()

    if not results:
        print('No results found. Run the validation grid first.')
        return

    summary_path = merge_and_write(manifest, results)
    print_convergence_table(results)
    print(f'\nSummary written to: {summary_path}')


if __name__ == '__main__':
    main()
