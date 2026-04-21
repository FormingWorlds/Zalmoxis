"""Run a Zalmoxis parameter grid from a TOML specification.

Generates all combinations (Cartesian product) of sweep parameters,
runs them in parallel with multiprocessing, and collects results into
a CSV summary.

Usage
-----
    python -m src.tools.run_grid grid.toml -j 4

The grid TOML has three sections:

    [base]
    config = "input/default.toml"   # base config (relative to ZALMOXIS_ROOT)

    [sweep]
    planet_mass = [0.5, 1.0, 3.0]
    surface_temperature = [2000, 3000]

    [output]
    dir = "output_files/grid_results"
    save_profiles = true   # optional; write full radial profile per grid point

Outputs
-------
For every grid point the runner always writes:

- ``grid_summary.csv``: one row per run with the sweep parameters, the final
  planet mass and radius, convergence flags, wall time, and any error.
- ``<label>.json``: the same summary as a per-run JSON file.

When ``save_profiles = true`` in the grid TOML (default ``false``), the runner
additionally writes ``<label>.npz`` per grid point with the full radial
structure (radius, density, gravity, pressure, temperature, mass enclosed),
the CMB and core+mantle mass diagnostics, the ``converged`` flag, and the
per-layer EOS / melting-curve identifiers the solver actually used
(``core_eos``, ``mantle_eos``, ``ice_layer_eos``, ``rock_solidus_id``,
``rock_liquidus_id``). This lets downstream tools plot pressure, density,
gravity, and temperature profiles and overlay the correct reference curves
without rerunning the sweep.

Archives are written for non-converged grid points too (for debugging), but
in that case ``radii[-1]`` and ``mass_enclosed[-1]`` are the last Picard
iterate, not the planet's converged structure, and ``density`` may contain
NaN or zero in failed / padded shells. Filter on ``converged == True``
before plotting.
"""

from __future__ import annotations

import copy
import csv
import itertools
import json
import logging
import os
import tempfile
import time
from multiprocessing import Pool

import numpy as np
import toml

from src.zalmoxis.constants import earth_mass, earth_radius
from src.zalmoxis.zalmoxis import (
    load_material_dictionaries,
    load_solidus_liquidus_functions,
    load_zalmoxis_config,
    main,
)

# ZALMOXIS_ROOT is validated at import by src.zalmoxis.zalmoxis
ZALMOXIS_ROOT = os.getenv('ZALMOXIS_ROOT', '')

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter name -> (TOML section, TOML key) mapping
# ---------------------------------------------------------------------------
_PARAM_MAP: dict[str, tuple[str, str]] = {
    'planet_mass': ('InputParameter', 'planet_mass'),
    'core_mass_fraction': ('AssumptionsAndInitialGuesses', 'core_mass_fraction'),
    'mantle_mass_fraction': ('AssumptionsAndInitialGuesses', 'mantle_mass_fraction'),
    'temperature_mode': ('AssumptionsAndInitialGuesses', 'temperature_mode'),
    'surface_temperature': ('AssumptionsAndInitialGuesses', 'surface_temperature'),
    'center_temperature': ('AssumptionsAndInitialGuesses', 'center_temperature'),
    'core': ('EOS', 'core'),
    'mantle': ('EOS', 'mantle'),
    'ice_layer': ('EOS', 'ice_layer'),
    'condensed_rho_min': ('EOS', 'condensed_rho_min'),
    'condensed_rho_scale': ('EOS', 'condensed_rho_scale'),
    'binodal_T_scale': ('EOS', 'binodal_T_scale'),
    'mushy_zone_factor': ('EOS', 'mushy_zone_factor'),
    'rock_solidus': ('EOS', 'rock_solidus'),
    'rock_liquidus': ('EOS', 'rock_liquidus'),
    'num_layers': ('Calculations', 'num_layers'),
}


# ---------------------------------------------------------------------------
# Grid config loading
# ---------------------------------------------------------------------------
def load_grid_config(grid_toml_path):
    """Load a grid TOML file and return the base config path and sweep specs.

    Parameters
    ----------
    grid_toml_path : str
        Path to the grid TOML file.

    Returns
    -------
    base_config_path : str
        Absolute path to the base Zalmoxis config TOML.
    sweeps : dict[str, list]
        Mapping of sweep parameter names to their value lists.
    output_dir : str
        Absolute path to the output directory for this grid.
    save_profiles : bool
        Whether to write the full radial profile (``<label>.npz``) per grid
        point. Reads ``[output].save_profiles`` from the grid TOML, default
        ``False``.
    """
    grid = toml.load(grid_toml_path)

    # Base config path (relative to ZALMOXIS_ROOT)
    base_rel = grid['base']['config']
    base_config_path = os.path.join(ZALMOXIS_ROOT, base_rel)
    if not os.path.isfile(base_config_path):
        raise FileNotFoundError(
            f'Base config not found: {base_config_path} '
            f"(from '{base_rel}' relative to ZALMOXIS_ROOT={ZALMOXIS_ROOT})"
        )

    # Sweep parameters
    sweeps = dict(grid.get('sweep', {}))
    for name in sweeps:
        if name not in _PARAM_MAP:
            raise ValueError(
                f"Unknown sweep parameter '{name}'. "
                f'Valid parameters: {sorted(_PARAM_MAP.keys())}'
            )
        if not isinstance(sweeps[name], list):
            raise TypeError(
                f"Sweep parameter '{name}' must be a list, got {type(sweeps[name]).__name__}"
            )

    # Output directory (relative to ZALMOXIS_ROOT)
    output_section = grid.get('output', {})
    output_rel = output_section.get('dir', 'output_files/grid_results')
    output_dir = os.path.join(ZALMOXIS_ROOT, output_rel)

    # Optional: per-grid-point radial profile dump (off by default).
    # Require a real TOML bool so a stray quoted "false" string does not
    # silently evaluate as truthy.
    save_profiles = output_section.get('save_profiles', False)
    if not isinstance(save_profiles, bool):
        raise TypeError(
            f'[output].save_profiles must be a bool, got '
            f'{type(save_profiles).__name__} ({save_profiles!r})'
        )

    return base_config_path, sweeps, output_dir, save_profiles


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------
def generate_configs(base_config_path, sweeps):
    """Generate all parameter combinations from the sweep specification.

    Parameters
    ----------
    base_config_path : str
        Absolute path to the base TOML config file.
    sweeps : dict[str, list]
        Mapping of parameter names to value lists.

    Returns
    -------
    list[tuple[str, str]]
        List of (label, config_path) tuples. Each config_path is a
        temporary TOML file with the parameters for that combination.
        The label encodes the parameter values for identification.
    """
    if not sweeps:
        raise ValueError('No sweep parameters defined in the grid TOML')

    # Read the raw TOML so we can modify and re-write it
    base_toml = toml.load(base_config_path)

    # Build ordered lists for the Cartesian product
    param_names = sorted(sweeps.keys())
    param_values = [sweeps[name] for name in param_names]

    configs = []
    for combo in itertools.product(*param_values):
        # Build label string
        label_parts = []
        for name, val in zip(param_names, combo):
            label_parts.append(f'{name}={val}')
        label = '__'.join(label_parts)

        # Deep copy the base TOML and apply overrides
        modified = copy.deepcopy(base_toml)
        for name, val in zip(param_names, combo):
            section, key = _PARAM_MAP[name]
            modified[section][key] = val

        # Disable plots and verbose for grid runs (speed)
        modified['Output']['plots_enabled'] = False
        modified['Output']['verbose'] = False

        # Write to a temporary TOML file
        tmp = tempfile.NamedTemporaryFile(
            mode='w',
            suffix='.toml',
            prefix=f'zalmoxis_grid_{label}_',
            delete=False,
        )
        toml.dump(modified, tmp)
        tmp.close()

        configs.append((label, tmp.name))

    return configs


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------
def run_single(args):
    """Run a single Zalmoxis model from a config file.

    Parameters
    ----------
    args : tuple
        ``(label, config_path, output_dir, save_profiles)`` where ``label``
        identifies the run, ``config_path`` is the temporary TOML file,
        ``output_dir`` is where to write per-run results, and
        ``save_profiles`` toggles dumping the full radial profile to
        ``<label>.npz`` (arrays: ``radii``, ``density``, ``gravity``,
        ``pressure``, ``temperature``, ``mass_enclosed``; scalars:
        ``cmb_mass``, ``core_mantle_mass``, ``converged``). Archives are
        written for non-converged points too; their ``radii[-1]`` and
        ``mass_enclosed[-1]`` are the last Picard iterate, and ``density``
        can contain NaN or zero in failed / padded shells.

    Returns
    -------
    dict
        Summary with keys: label, R_earth, M_earth, converged,
        converged_pressure, converged_density, converged_mass, time_s.
        On failure, includes an 'error' key.
    """
    label, config_path, output_dir, save_profiles = args

    t0 = time.time()
    result = {
        'label': label,
        'R_earth': None,
        'M_earth': None,
        'converged': False,
        'converged_pressure': False,
        'converged_density': False,
        'converged_mass': False,
        'time_s': 0.0,
        'error': None,
    }

    try:
        config_params = load_zalmoxis_config(config_path)

        # Disable per-run text output and plotting; the grid runner collects
        # summary results programmatically and (optionally) writes profiles
        # itself as compressed .npz below.
        config_params['data_output_enabled'] = False
        config_params['plotting_enabled'] = False

        # Load EOS data and melting curves
        material_dicts = load_material_dictionaries()
        melting_curves = load_solidus_liquidus_functions(
            config_params['layer_eos_config'],
            config_params.get('rock_solidus', 'Stixrude14-solidus'),
            config_params.get('rock_liquidus', 'Stixrude14-liquidus'),
        )

        model_results = main(
            config_params,
            material_dictionaries=material_dicts,
            melting_curves_functions=melting_curves,
            input_dir=os.path.join(ZALMOXIS_ROOT, 'input'),
        )

        result['R_earth'] = model_results['radii'][-1] / earth_radius
        result['M_earth'] = model_results['mass_enclosed'][-1] / earth_mass
        result['converged'] = model_results['converged']
        result['converged_pressure'] = model_results['converged_pressure']
        result['converged_density'] = model_results['converged_density']
        result['converged_mass'] = model_results['converged_mass']

        # Save per-run JSON summary
        os.makedirs(output_dir, exist_ok=True)
        json_path = os.path.join(output_dir, f'{label}.json')
        with open(json_path, 'w') as f:
            json.dump(
                {
                    'label': label,
                    'R_earth': result['R_earth'],
                    'M_earth': result['M_earth'],
                    'converged': result['converged'],
                    'time_s': time.time() - t0,
                },
                f,
                indent=2,
            )

        # Optional: persist the full radial profile so users can plot
        # pressure, density, gravity, and temperature vs. radius (or any
        # two of them against each other) without re-running the sweep.
        # SI units throughout: m, kg/m^3, m/s^2, Pa, K, kg.
        #
        # Non-converged grid points are saved too (useful for debugging),
        # but in that case radii[-1] and mass_enclosed[-1] reflect the
        # last Picard iterate, not the planet's converged structure, and
        # density can contain NaN or zero in failed / padded shells.
        # The `converged` scalar is embedded in the archive so downstream
        # users can filter without cross-referencing grid_summary.csv.
        #
        # The EOS and melting-curve identifiers are embedded so plotting
        # tools know which solidus/liquidus the solver actually used
        # (rather than guessing from defaults).
        if save_profiles:
            layer_eos = config_params.get('layer_eos_config') or {}
            npz_path = os.path.join(output_dir, f'{label}.npz')
            # Wrap the archive write so an I/O failure (disk full,
            # permissions, etc.) does not silently contradict the
            # per-run JSON summary written below: log a warning and
            # record the failure on the result dict so it also surfaces
            # in grid_summary.csv.
            try:
                np.savez_compressed(
                    npz_path,
                    radii=model_results['radii'],
                    density=model_results['density'],
                    gravity=model_results['gravity'],
                    pressure=model_results['pressure'],
                    temperature=model_results['temperature'],
                    mass_enclosed=model_results['mass_enclosed'],
                    cmb_mass=model_results['cmb_mass'],
                    core_mantle_mass=model_results['core_mantle_mass'],
                    converged=np.bool_(model_results['converged']),
                    core_eos=np.str_(layer_eos.get('core', '')),
                    mantle_eos=np.str_(layer_eos.get('mantle', '')),
                    ice_layer_eos=np.str_(layer_eos.get('ice_layer', '')),
                    rock_solidus_id=np.str_(config_params.get('rock_solidus', '')),
                    rock_liquidus_id=np.str_(config_params.get('rock_liquidus', '')),
                )
            except OSError as profile_error:
                logger.warning(
                    "Failed to write optional profile archive %s for run '%s': %s",
                    npz_path,
                    label,
                    profile_error,
                )
                result['error'] = f'profile write failed: {profile_error}'

    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Run '{label}' failed: {e}")

    finally:
        # Clean up temp config file
        try:
            os.unlink(config_path)
        except OSError:
            pass

    result['time_s'] = time.time() - t0
    return result


# ---------------------------------------------------------------------------
# Grid runner
# ---------------------------------------------------------------------------
def run_grid(grid_toml_path, n_workers=1):
    """Run the full parameter grid and write a CSV summary.

    Per-run outputs (always written): ``grid_summary.csv`` plus a
    ``<label>.json`` file per grid point. When the grid TOML sets
    ``[output].save_profiles = true`` the runner additionally writes
    ``<label>.npz`` containing the full radial profile (radius, density,
    gravity, pressure, temperature, mass enclosed) for each grid point,
    including non-converged ones. Non-converged / failed profiles may
    contain padded or invalid shell values, so filter on
    ``converged == True`` before plotting or analysis.

    Parameters
    ----------
    grid_toml_path : str
        Path to the grid TOML specification file.
    n_workers : int
        Number of parallel workers. Default 1 (serial).

    Returns
    -------
    list[dict]
        List of result dicts, one per grid point.
    """
    t0 = time.time()

    base_config_path, sweeps, output_dir, save_profiles = load_grid_config(grid_toml_path)
    os.makedirs(output_dir, exist_ok=True)

    # Report grid size
    n_total = 1
    print('=' * 72)
    print('Zalmoxis parameter grid')
    print('=' * 72)
    print(f'  Base config:   {base_config_path}')
    print(f'  Output dir:    {output_dir}')
    print(f'  Workers:       {n_workers}')
    print(f'  Save profiles: {save_profiles}')
    print()
    print('  Sweep parameters:')
    for name in sorted(sweeps.keys()):
        vals = sweeps[name]
        n_total *= len(vals)
        print(f'    {name}: {vals}')
    print(f'\n  Total combinations: {n_total}')
    print('=' * 72)
    print()

    # Generate all configs
    configs = generate_configs(base_config_path, sweeps)
    assert len(configs) == n_total

    # Prepare arguments for run_single
    run_args = [
        (label, config_path, output_dir, save_profiles) for label, config_path in configs
    ]

    # Run in parallel
    if n_workers > 1:
        with Pool(processes=n_workers) as pool:
            results = pool.map(run_single, run_args)
    else:
        results = [run_single(a) for a in run_args]

    # Write CSV summary
    csv_path = os.path.join(output_dir, 'grid_summary.csv')
    # Extract sweep parameter names for CSV columns
    sweep_names = sorted(sweeps.keys())
    fieldnames = (
        ['label']
        + sweep_names
        + [
            'R_earth',
            'M_earth',
            'converged',
            'converged_pressure',
            'converged_density',
            'converged_mass',
            'time_s',
            'error',
        ]
    )
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = dict(r)
            # Parse label to extract sweep parameter values
            for part in r['label'].split('__'):
                if '=' in part:
                    k, v = part.split('=', 1)
                    row[k] = v
            writer.writerow({k: row.get(k, '') for k in fieldnames})

    # Print summary table
    elapsed = time.time() - t0
    n_converged = sum(1 for r in results if r['converged'])
    n_failed = sum(1 for r in results if r['error'])

    print()
    print('=' * 72)
    print('Grid results')
    print('=' * 72)
    print(f'  {"Label":<50} {"R_earth":>8} {"Conv":>5} {"Time":>6}')
    print('  ' + '-' * 70)
    for r in results:
        label_short = r['label'][:50]
        r_earth = f'{r["R_earth"]:.3f}' if r['R_earth'] is not None else 'FAIL'
        conv = 'Y' if r['converged'] else 'N'
        t_s = f'{r["time_s"]:.1f}s'
        print(f'  {label_short:<50} {r_earth:>8} {conv:>5} {t_s:>6}')

    print()
    print(f'  Converged: {n_converged}/{n_total}')
    if n_failed:
        print(f'  Errors:    {n_failed}/{n_total}')
    print(f'  Total time: {elapsed:.1f}s')
    print(f'  CSV saved:  {csv_path}')
    print('=' * 72)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    parser = argparse.ArgumentParser(description='Run Zalmoxis parameter grid')
    parser.add_argument('grid_toml', help='Path to grid TOML file')
    parser.add_argument(
        '-j',
        '--workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)',
    )
    args = parser.parse_args()

    run_grid(args.grid_toml, args.workers)
