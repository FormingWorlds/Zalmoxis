"""Shared loader for per-cell profile CSVs written by ``run_grid``.

The grid runner writes one ``<label>.csv`` per converged cell when
``[output].save_profiles = true``. The CSV body has six numeric SI
columns (``radius_m``, ``density_kg_m3``, ``gravity_m_s2``,
``pressure_Pa``, ``temperature_K``, ``mass_enclosed_kg``) plus two
string columns (``main_component``, ``phase``); the metadata (converged
flag, CMB mass, layer EOS strings, melting-curve identifiers) sits in
``# key: value`` comment lines above the header.

Three plot tools read these files (``plot_grid_profiles``,
``plot_grid_pt``, ``plot_grid_composition``); they share this loader so
the file-format contract lives in exactly one place.
"""

from __future__ import annotations

import csv
import os
from typing import Optional

import numpy as np

# CSV column name -> legacy in-memory key. The plot tools were written
# against the historical ``.npz`` archive (which used short names like
# ``radii``); preserving those keys here means the loader is a drop-in
# replacement and downstream code is unchanged.
_CSV_TO_LEGACY = {
    'radius_m': 'radii',
    'density_kg_m3': 'density',
    'gravity_m_s2': 'gravity',
    'pressure_Pa': 'pressure',
    'temperature_K': 'temperature',
    'mass_enclosed_kg': 'mass_enclosed',
}

# String body columns (added with the per-shell main-component / phase
# labelling). Loaded as object arrays rather than coerced to float.
_STR_BODY_COLUMNS = ('main_component', 'phase')

# Metadata keys that should be coerced to float when present.
_FLOAT_METADATA = ('cmb_mass', 'core_mantle_mass')

# Metadata keys that should be coerced to bool when present.
_BOOL_METADATA = ('converged',)

# Metadata keys that stay as strings.
_STR_METADATA = (
    'label',
    'core_eos',
    'mantle_eos',
    'ice_layer_eos',
    'rock_solidus_id',
    'rock_liquidus_id',
)


def load_profile(grid_dir: str, label: str) -> Optional[dict]:
    """Load one per-cell profile CSV into a dict.

    Parameters
    ----------
    grid_dir : str
        Directory containing the per-cell CSV files.
    label : str
        Cell label, e.g. ``'planet_mass=1.0'``. The loader looks for
        ``<grid_dir>/<label>.csv``.

    Returns
    -------
    dict or None
        ``None`` if the file is absent. Otherwise a dict with the six
        profile arrays under their legacy keys (``radii``, ``density``,
        ``gravity``, ``pressure``, ``temperature``, ``mass_enclosed``)
        plus any metadata fields that were present in the comment
        header. ``cmb_mass`` and ``core_mantle_mass`` are floats,
        ``converged`` is bool, EOS / curve identifiers are strings.

    Raises
    ------
    ValueError
        If the file exists but has no data rows.
    """
    path = os.path.join(grid_dir, f'{label}.csv')
    if not os.path.isfile(path):
        return None

    metadata: dict[str, str] = {}
    data_lines: list[str] = []
    with open(path) as fh:
        for line in fh:
            stripped = line.lstrip()
            if not stripped:
                continue
            if stripped.startswith('#'):
                payload = stripped[1:].lstrip()
                if ':' in payload:
                    key, _, value = payload.partition(':')
                    metadata[key.strip()] = value.strip()
                continue
            data_lines.append(line)

    if not data_lines:
        raise ValueError(f'No data rows found in profile CSV {path}')

    reader = csv.reader(data_lines)
    header = next(reader)
    columns: dict[str, list] = {name: [] for name in header}
    for row in reader:
        if not row:
            continue
        for name, value in zip(header, row):
            if name in _STR_BODY_COLUMNS:
                columns[name].append(value)
            else:
                columns[name].append(float(value))

    out: dict = {}
    for csv_name, values in columns.items():
        legacy_key = _CSV_TO_LEGACY.get(csv_name, csv_name)
        if csv_name in _STR_BODY_COLUMNS:
            out[legacy_key] = np.asarray(values, dtype=object)
        else:
            out[legacy_key] = np.asarray(values, dtype=np.float64)

    for key in _FLOAT_METADATA:
        if key in metadata:
            out[key] = float(metadata[key])
    for key in _BOOL_METADATA:
        if key in metadata:
            out[key] = metadata[key].strip().lower() in ('true', '1', 'yes')
    for key in _STR_METADATA:
        if key in metadata:
            out[key] = metadata[key]

    return out
