"""Live PALEOS tabulation — ``PALEOS-API:*`` producer side.

Generates 10-column P-T ``.dat`` files bit-format-identical to the shipped
``EOS_PALEOS_*`` Zenodo tables, but sourced at runtime from ``import paleos``
rather than from a flat-file checkout. Output is consumed unchanged by the
existing readers: ``zalmoxis.eos.interpolation.load_paleos_table`` (density +
nabla_ad path) and ``zalmoxis.eos_export.load_paleos_all_properties``
(SPIDER P-S export path).

Two entry points:

- ``generate_paleos_api_unified_table`` — one file per material, stable-phase
  dispatch (matches ``EOS_PALEOS_{iron,MgSiO3_unified,H2O}``).
- ``generate_paleos_api_2phase_mgsio3_tables`` — two files (solid + liquid)
  with metastable extensions, matching
  ``EOS_PALEOS_MgSiO3/paleos_mgsio3_tables_pt_proteus_{solid,liquid}.dat``.
  Aragog's P-S table build feeds these to
  ``eos_export.generate_spider_eos_tables`` as ``solid_eos_file`` and
  ``liquid_eos_file`` so that mushy-zone mixing uses phase-specific
  endpoints rather than interpolating across the melting curve.

Cache / registry wiring is handled by ``paleos_api_cache.py`` (A.2) and the
``PALEOS-API:*`` registry entries (A.3). This module contains pure producers
only; no cache logic lives here.

Design notes
------------
The 2-phase solid-side picker reaches into PALEOS internals
(``_phase_eos_map``, the phase-boundary functions, ``_P_HPCEN_BRG``). These
are not a committed public API. The PALEOS SHA is recorded in the output
header so any upstream rename is caught by a cache-key miss. Stage B.5 of
``zalmoxis-paleos-api-eos.md`` tracks the upstream contribution of a public
``get_mgsio3_solid_phase`` + ``generate_twophase_pt_tables`` helper; at that
point the internals access here collapses to one public call.
"""

from __future__ import annotations

import datetime as _dt
import hashlib as _hashlib
import json as _json
import logging
import math as _math
import multiprocessing as _mp
import os as _os
import subprocess as _subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# Module-global EoS instances populated inside worker processes by
# ``_worker_init``. Lets workers avoid pickling EoS instances (Wolf18 /
# MgSiO3EoS carry sympy-compiled closures that may not pickle cleanly)
# and pay the PALEOS init cost only once per worker.
_WORKER_EOS: dict = {}

# Matches the shipped Zenodo tables: 9 numeric columns written ``%.8e``
# space-delimited, followed by a string phase label.
_ROW_FMT = ' '.join(['%.8e'] * 9) + ' %s\n'
_NUMERIC_COLS = 9


@dataclass(frozen=True)
class GridSpec:
    """Log-uniform (P, T) grid specification.

    Pressure axis is log-uniform in ``[p_lo, p_hi]`` Pa with ``n_p`` points.
    Temperature axis is log-uniform in ``[t_lo, t_hi]`` K with ``n_t``
    points, matching the Zenodo shipped tables (which are log-uniform in
    both axes at 150 pts/decade).

    Attributes
    ----------
    p_lo, p_hi : float
        Pressure bounds [Pa]. ``p_lo`` must be > 0 (log spacing).
    n_p : int
        Number of pressure nodes.
    t_lo, t_hi : float
        Temperature bounds [K].
    n_t : int
        Number of temperature nodes.
    """

    p_lo: float
    p_hi: float
    n_p: int
    t_lo: float
    t_hi: float
    n_t: int

    def hash_short(self) -> str:
        """SHA-1 short digest for cache keying."""
        payload = _json.dumps(asdict(self), sort_keys=True).encode('utf-8')
        return _hashlib.sha1(payload).hexdigest()[:10]

    def axes(self):
        """Return (P_axis [Pa], T_axis [K]) as 1D numpy arrays."""
        P = np.logspace(np.log10(self.p_lo), np.log10(self.p_hi), self.n_p)
        T = np.logspace(np.log10(self.t_lo), np.log10(self.t_hi), self.n_t)
        return P, T


# Shipped Zenodo tables are 150 pts/decade (see paleos_mgsio3_tables_pt_proteus_solid.dat
# header). PROTEUS-A default is 4x denser on each axis (16x grid size) to give
# bilinear interp enough resolution to resolve phase boundaries and minimize
# coupling-loop interpolation noise. Cold-cache cost (~1 h with 16-core ProcessPool
# on Mac Studio, one-time per PALEOS SHA) is amortized across every subsequent run.
#
# Override for sanity-check runs via ``ZALMOXIS_PALEOS_API_PTS_PER_DECADE``
# (e.g. = 150 to match the shipped Zenodo grid for fast A.6 comparisons).
# Must be set before the first import of this module; ``make_grid_at_resolution``
# captures this as a default arg at def time.
DEFAULT_PTS_PER_DECADE = int(_os.environ.get('ZALMOXIS_PALEOS_API_PTS_PER_DECADE', '600'))


def _n_points_per_decade(lo: float, hi: float, pts_per_decade: int) -> int:
    """Return the number of log-uniform points spanning [lo, hi] at ``pts_per_decade``."""
    if lo <= 0 or hi <= 0:
        raise ValueError('log-uniform bounds must be positive')
    n_decades = _math.log10(hi / lo)
    return int(round(pts_per_decade * n_decades)) + 1


def make_grid_at_resolution(
    p_lo: float,
    p_hi: float,
    t_lo: float,
    t_hi: float,
    pts_per_decade: int = DEFAULT_PTS_PER_DECADE,
) -> GridSpec:
    """Build a ``GridSpec`` with a uniform log-log resolution.

    Parameters
    ----------
    p_lo, p_hi : float
        Pressure bounds [Pa], both > 0.
    t_lo, t_hi : float
        Temperature bounds [K], both > 0.
    pts_per_decade : int
        Nodes per decade on each axis. Default
        ``DEFAULT_PTS_PER_DECADE`` = 600 (4x the shipped Zenodo resolution).
    """
    return GridSpec(
        p_lo=p_lo,
        p_hi=p_hi,
        n_p=_n_points_per_decade(p_lo, p_hi, pts_per_decade),
        t_lo=t_lo,
        t_hi=t_hi,
        n_t=_n_points_per_decade(t_lo, t_hi, pts_per_decade),
    )


# Material-specific default bounds. Ranges match the shipped Zenodo tables so
# a first-order sanity check against them is possible. The 2-phase MgSiO3 grid
# uses the same bounds as the unified MgSiO3 grid (Aragog consumes the same
# (P, T) envelope from either).
def make_default_grid_iron() -> GridSpec:
    """Default Fe grid: 1e5 to 1e14 Pa, 300 to 20000 K at 600 pts/decade."""
    return make_grid_at_resolution(1e5, 1e14, 300.0, 2.0e4)


def make_default_grid_mgsio3() -> GridSpec:
    """Default MgSiO3 grid: 1e5 to 1e14 Pa, 300 to 11500 K at 600 pts/decade."""
    return make_grid_at_resolution(1e5, 1e14, 300.0, 1.15e4)


def make_default_grid_h2o() -> GridSpec:
    """Default H2O grid: 0.1 Pa to 1e14 Pa, 150 to 1e5 K at 600 pts/decade.

    Wider P range than Fe/MgSiO3 to match AQUA table coverage (extends to
    1 micro-bar). H2O is table-backed in PALEOS (``Haldemann20``) so per-point
    cost is low.
    """
    return make_grid_at_resolution(1e-1, 1e14, 150.0, 1.0e5)


def paleos_installed_sha() -> str:
    """Return the git SHA of the installed PALEOS checkout, or ``'unknown'``.

    PALEOS does not expose ``__commit_sha__`` yet (Stage B.3). We fall back
    to ``git rev-parse HEAD`` against the directory holding ``paleos``.
    If both fail, we use ``paleos.__version__`` as a coarse identifier.
    """
    try:
        import paleos as _paleos
    except ImportError:
        return 'paleos-not-installed'

    pkg_dir = Path(_paleos.__file__).resolve().parent
    repo_dir = pkg_dir.parent
    try:
        sha = _subprocess.check_output(
            ['git', '-C', str(repo_dir), 'rev-parse', 'HEAD'],
            stderr=_subprocess.DEVNULL,
            text=True,
        ).strip()
        if sha:
            return sha
    except (_subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass

    return f'version-{getattr(_paleos, "__version__", "unknown")}'


def _write_header(
    f,
    material: str,
    kind: str,
    grid: GridSpec,
    paleos_sha: str,
    n_valid: int,
    n_skipped: int,
    extra_lines=None,
):
    """Emit the commented header block. Matches shipped-table conventions."""
    timestamp = _dt.datetime.now(_dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    f.write('# ' + '=' * 78 + '\n')
    f.write(f'# PALEOS {material} EoS Lookup Table — PROTEUS (live-generated)\n')
    f.write('# ' + '=' * 78 + '\n#\n')
    f.write(f'# Kind: {kind}\n')
    f.write('# Generator: zalmoxis.eos.paleos_api\n')
    f.write(f'# PALEOS_SHA: {paleos_sha}\n')
    f.write(f'# grid_hash: {grid.hash_short()}\n')
    f.write(f'# generated: {timestamp}\n#\n')
    f.write(f'# P grid: {grid.n_p} pts log-uniform in [{grid.p_lo:.8e}, {grid.p_hi:.8e}] Pa\n')
    f.write(f'# T grid: {grid.n_t} pts log-uniform in [{grid.t_lo:.8e}, {grid.t_hi:.8e}] K\n')
    f.write(f'# Grid size: {grid.n_p} x {grid.n_t} = {grid.n_p * grid.n_t} points\n')
    f.write(f'#            ({n_valid} valid / {n_skipped} skipped due to non-convergence)\n#\n')
    f.write(
        '# Columns: P[Pa] T[K] rho[kg/m^3] u[J/kg] s[J/(kg K)] '
        'cp[J/(kg K)] cv[J/(kg K)] alpha[1/K] nabla_ad[-] phase\n'
    )
    f.write(
        '# Interpolation: bilinear in (log10 P, log10 T); '
        'NaNs mark non-converged cells, consumer uses NN fallback.\n'
    )
    if extra_lines:
        for line in extra_lines:
            f.write(f'# {line}\n')
    f.write('# ' + '=' * 78 + '\n')


_NAN_PROPS = (float('nan'),) * 7


def _resolve_n_workers(n_workers: int) -> int:
    """Map ``n_workers`` sentinel to an actual worker count.

    ``1``  -> serial (no ProcessPool).
    ``-1`` -> ``os.cpu_count()`` or 1 on exotic platforms.
    Any other positive int is passed through.
    """
    if n_workers == -1:
        return _os.cpu_count() or 1
    if n_workers < 1:
        raise ValueError(f'n_workers must be >= 1 or -1 (all); got {n_workers}')
    return n_workers


def _worker_init(material: str, h2o_table_path: str | None = None):
    """Initialize per-worker PALEOS EoS instances once per process.

    Runs inside each ``multiprocessing.Pool`` worker before it starts
    consuming tasks. Populates module-global ``_WORKER_EOS`` dict so the
    task functions look up pre-built instances instead of rebuilding on
    every call (Wolf18 in particular does sympy compilation at __init__).
    """
    if material == 'iron':
        from paleos.iron_eos import IronEoS

        _WORKER_EOS['unified'] = IronEoS()
    elif material == 'mgsio3':
        from paleos.mgsio3_eos import MgSiO3EoS, Wolf18

        _WORKER_EOS['unified'] = MgSiO3EoS()
        _WORKER_EOS['wolf18'] = Wolf18()
        _WORKER_EOS['phase_map'] = _WORKER_EOS['unified']._phase_eos_map
    elif material == 'h2o':
        from paleos.water_eos import WaterEoS

        _WORKER_EOS['unified'] = (
            WaterEoS() if h2o_table_path is None else WaterEoS(table_path=h2o_table_path)
        )
    else:
        raise ValueError(f'unknown material {material!r}')


def _worker_unified_row(args):
    """Tabulate one P-row for the unified generator inside a worker.

    ``args = (i_P, P_value, T_axis)``. Returns ``(i_P, rows_text, n_valid, n_skip)``.
    """
    i_P, P_value, T_axis = args
    eos = _WORKER_EOS['unified']
    n_valid = 0
    n_skip = 0
    out = []
    Pf = float(P_value)
    for T in T_axis:
        Tf = float(T)
        props = _evaluate_eos(eos, Pf, Tf)
        if np.isnan(props[0]):
            n_skip += 1
            phase_str = 'out-of-range'
        else:
            n_valid += 1
            try:
                phase_str = str(eos.phase(Pf, Tf))
            except (RuntimeError, ValueError):
                phase_str = 'unknown'
        out.append(_row(Pf, Tf, props, phase_str))
    return i_P, ''.join(out), n_valid, n_skip


def _worker_2phase_row(args):
    """Tabulate one P-row for the 2-phase MgSiO3 generator inside a worker.

    Returns ``(i_P, solid_rows_text, liquid_rows_text, nv_s, ns_s, nv_l, ns_l)``.
    """
    i_P, P_value, T_axis = args
    phase_map = _WORKER_EOS['phase_map']
    liquid_eos = _WORKER_EOS['wolf18']

    out_s = []
    out_l = []
    nv_s = ns_s = nv_l = ns_l = 0
    Pf = float(P_value)
    for T in T_axis:
        Tf = float(T)

        solid_phase = _get_mgsio3_solid_phase(Pf, Tf)
        solid_eos = phase_map[solid_phase]
        props_s = _evaluate_eos(solid_eos, Pf, Tf)
        if np.isnan(props_s[0]):
            ns_s += 1
            solid_label = 'out-of-range'
        else:
            nv_s += 1
            solid_label = solid_phase
        out_s.append(_row(Pf, Tf, props_s, solid_label))

        props_l = _evaluate_eos(liquid_eos, Pf, Tf)
        if np.isnan(props_l[0]):
            ns_l += 1
            liquid_label = 'out-of-range'
        else:
            nv_l += 1
            liquid_label = 'liquid'
        out_l.append(_row(Pf, Tf, props_l, liquid_label))

    return i_P, ''.join(out_s), ''.join(out_l), nv_s, ns_s, nv_l, ns_l


def _evaluate_eos(eos, P: float, T: float):
    """Call the 7 scalar property methods on a PALEOS EoS instance.

    Returns ``(rho, u, s, cp, cv, alpha, nabla_ad)`` or a 7-tuple of NaN
    on ``RuntimeError`` / ``ValueError`` (PALEOS's signal for
    brentq-failed / out-of-valid-range).

    Phase labels are handled by the caller: on top-level ``MgSiO3EoS`` /
    ``IronEoS`` instances, ``phase(P, T)`` is a method; on the polymorph
    classes (``Sokolova22``, etc.) ``phase`` is a string attribute set at
    construction. Mixing those two interfaces in a shared helper is what
    bit us in the first smoke test.
    """
    try:
        rho = float(eos.density(P, T))
        u = float(eos.specific_internal_energy(P, T))
        s = float(eos.specific_entropy(P, T))
        cp = float(eos.isobaric_heat_capacity(P, T))
        cv = float(eos.isochoric_heat_capacity(P, T))
        alpha = float(eos.thermal_expansion(P, T))
        nabla_ad = float(eos.adiabatic_gradient(P, T))
    except (RuntimeError, ValueError):
        return _NAN_PROPS
    return rho, u, s, cp, cv, alpha, nabla_ad


def _row(P: float, T: float, props, phase_str: str) -> str:
    """Format one 10-column row. ``props`` is the 7-tuple from ``_evaluate_eos``."""
    rho, u, s, cp, cv, alpha, nabla_ad = props
    return _ROW_FMT % (P, T, rho, u, s, cp, cv, alpha, nabla_ad, phase_str)


# ---------------------------------------------------------------------------
# Unified (stable-phase) generator: matches EOS_PALEOS_{iron,MgSiO3_unified,H2O}
# ---------------------------------------------------------------------------

_HUMAN_MATERIAL = {'iron': 'Fe', 'mgsio3': 'MgSiO3', 'h2o': 'H2O'}


def generate_paleos_api_unified_table(
    material: str,
    out_path: str | Path,
    grid: GridSpec,
    *,
    h2o_table_path: str | None = None,
    log_every: int = 50,
    n_workers: int = 1,
) -> dict:
    """Generate a unified PALEOS ``.dat`` for one material.

    Stable-phase dispatch: for each (P, T) the PALEOS top-level class
    (``IronEoS`` / ``MgSiO3EoS`` / ``WaterEoS``) chooses the thermodynamically
    stable phase and evaluates that phase's EoS. Phase label is PALEOS's
    intrinsic string (``'solid-hpcen'``, ``'liquid'``, ``'hcp-Fe'``, etc).

    Parameters
    ----------
    material : {'iron', 'mgsio3', 'h2o'}
        Material selector.
    out_path : path-like
        Destination ``.dat`` path. Parent directories are created.
    grid : GridSpec
        (P, T) grid. ``p_lo`` must be > 0.
    h2o_table_path : str or None
        AQUA table path. Passed to ``WaterEoS`` when ``material == 'h2o'``.
    log_every : int
        Emit a progress line every ``log_every`` pressure rows.
    n_workers : int
        Parallelism. ``1`` = serial (default, backward-compatible). ``-1`` =
        ``os.cpu_count()``. Any other positive int = that many workers.
        Workers parallelize over P-rows; each worker instantiates its own
        PALEOS EoS at init (Wolf18 sympy compile cost paid once per worker,
        not per task).

    Returns
    -------
    dict
        ``{'n_valid', 'n_skipped', 'sha'}``.
    """
    if grid.p_lo <= 0:
        raise ValueError('GridSpec.p_lo must be > 0 (log spacing)')
    if material not in _HUMAN_MATERIAL:
        raise ValueError(f"material must be one of 'iron', 'mgsio3', 'h2o'; got {material!r}")

    human_material = _HUMAN_MATERIAL[material]
    n_workers = _resolve_n_workers(n_workers)
    sha = paleos_installed_sha()
    P_axis, T_axis = grid.axes()
    T_axis_list = T_axis.tolist()  # avoid repeated numpy scalar conversions in workers

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')

    tasks = [(i, float(P), T_axis_list) for i, P in enumerate(P_axis)]

    rows_by_i: dict[int, str] = {}
    n_valid = 0
    n_skipped = 0
    n_done = 0

    if n_workers == 1:
        # Serial path: no Pool, no worker init — tabulate in-process.
        _worker_init(material, h2o_table_path=h2o_table_path)
        iterator = (_worker_unified_row(t) for t in tasks)
    else:
        ctx = _mp.get_context('spawn')
        pool = ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(material, h2o_table_path),
        )
        # chunksize kept small so progress logs stay meaningful at high cost/cell.
        chunksize = max(1, len(tasks) // (n_workers * 16))
        iterator = pool.imap_unordered(_worker_unified_row, tasks, chunksize=chunksize)

    try:
        for i_P, rows_text, nv, ns in iterator:
            rows_by_i[i_P] = rows_text
            n_valid += nv
            n_skipped += ns
            n_done += 1
            if log_every and (n_done % log_every == 0):
                logger.info(
                    'paleos_api %s unified: %d / %d P rows done (valid=%d skipped=%d)',
                    human_material,
                    n_done,
                    grid.n_p,
                    n_valid,
                    n_skipped,
                )
    finally:
        if n_workers > 1:
            pool.close()
            pool.join()

    with open(tmp_path, 'w') as f:
        _write_header(
            f,
            human_material,
            'unified (stable-phase)',
            grid,
            sha,
            n_valid=n_valid,
            n_skipped=n_skipped,
        )
        for i in range(grid.n_p):
            f.write(rows_by_i[i])
    _os.replace(tmp_path, out_path)

    logger.info(
        'paleos_api %s unified: wrote %s (valid=%d skipped=%d sha=%s n_workers=%d)',
        human_material,
        out_path,
        n_valid,
        n_skipped,
        sha[:10],
        n_workers,
    )
    return {'n_valid': n_valid, 'n_skipped': n_skipped, 'sha': sha}


# ---------------------------------------------------------------------------
# 2-phase MgSiO3 generator: matches the shipped paleos_mgsio3_tables_pt_proteus_*
# ---------------------------------------------------------------------------


def _get_mgsio3_solid_phase(P: float, T: float) -> str:
    """Pick the solid polymorph at (P, T), ignoring the melting curve.

    Direct port of the solid-side branch of
    ``paleos.mgsio3_eos.get_mgsio3_phase`` (``mgsio3_eos.py:3316-3347``) with
    the ``T >= T_melt`` early-return removed so callers get a solid-polymorph
    label even above the liquidus. This is what produces the metastable
    solid-side extension the shipped 2-phase table contains.

    Stage B.5 would move this into PALEOS as a public ``get_mgsio3_solid_phase``
    helper; this in-Zalmoxis replica is the MVP stand-in.
    """
    # Import lazily so the module stays optional-dependency clean.
    from paleos.mgsio3_eos import (
        _P_HPCEN_BRG,
        P_brg_ppv,
        P_en_hpcen,
        P_lpcen_en,
        P_lpcen_hpcen,
    )

    if P >= P_brg_ppv(T):
        return 'solid-ppv'
    if P >= _P_HPCEN_BRG:
        return 'solid-brg'
    if P >= P_en_hpcen(T):
        if P >= P_lpcen_hpcen(T):
            return 'solid-hpcen'
        return 'solid-lpcen'
    if T > 750.0 and P < P_lpcen_en(T):
        return 'solid-en'
    return 'solid-lpcen'


def generate_paleos_api_2phase_mgsio3_tables(
    out_solid: str | Path,
    out_liquid: str | Path,
    grid: GridSpec,
    *,
    log_every: int = 50,
    n_workers: int = 1,
) -> dict:
    """Generate solid and liquid MgSiO3 P-T tables with metastable extensions.

    Replaces the shipped ``paleos_mgsio3_tables_pt_proteus_solid.dat`` and
    ``paleos_mgsio3_tables_pt_proteus_liquid.dat`` — bit-format-compatible,
    consumed unchanged by ``eos_export.generate_spider_eos_tables`` via
    ``solid_eos_file`` / ``liquid_eos_file``.

    Parameters
    ----------
    out_solid, out_liquid : path-like
        Destinations for the two files. Parent directories are created.
    grid : GridSpec
        (P, T) grid; ``p_lo`` must be > 0.
    log_every : int
        Progress log cadence in pressure rows.
    n_workers : int
        Parallelism. ``1`` = serial (default, backward-compatible). ``-1`` =
        ``os.cpu_count()``. Workers parallelize over P-rows; each worker
        instantiates ``MgSiO3EoS`` + ``Wolf18`` once (Wolf18's sympy compile
        cost is paid once per worker, not per task).

    Returns
    -------
    dict
        Summary with per-side ``n_valid`` / ``n_skipped`` and the installed
        PALEOS ``sha``.

    Notes
    -----
    The liquid-side file evaluates ``paleos.mgsio3_eos.Wolf18`` on every
    grid node, including below the solidus (metastable liquid). The
    solid-side file dispatches through ``MgSiO3EoS()._phase_eos_map`` using
    ``_get_mgsio3_solid_phase`` (melting-curve test removed) so every node
    gets a solid-polymorph EoS evaluated metastably above the liquidus if
    needed.
    """
    if grid.p_lo <= 0:
        raise ValueError('GridSpec.p_lo must be > 0 (log spacing)')

    n_workers = _resolve_n_workers(n_workers)
    sha = paleos_installed_sha()
    P_axis, T_axis = grid.axes()
    T_axis_list = T_axis.tolist()

    out_solid = Path(out_solid)
    out_liquid = Path(out_liquid)
    out_solid.parent.mkdir(parents=True, exist_ok=True)
    out_liquid.parent.mkdir(parents=True, exist_ok=True)
    tmp_solid = out_solid.with_suffix(out_solid.suffix + '.tmp')
    tmp_liquid = out_liquid.with_suffix(out_liquid.suffix + '.tmp')

    tasks = [(i, float(P), T_axis_list) for i, P in enumerate(P_axis)]

    solid_by_i: dict[int, str] = {}
    liquid_by_i: dict[int, str] = {}
    n_valid_s = n_skip_s = 0
    n_valid_l = n_skip_l = 0
    n_done = 0

    if n_workers == 1:
        _worker_init('mgsio3')
        iterator = (_worker_2phase_row(t) for t in tasks)
    else:
        ctx = _mp.get_context('spawn')
        pool = ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=('mgsio3',),
        )
        chunksize = max(1, len(tasks) // (n_workers * 16))
        iterator = pool.imap_unordered(_worker_2phase_row, tasks, chunksize=chunksize)

    try:
        for i_P, s_text, l_text, nv_s, ns_s, nv_l, ns_l in iterator:
            solid_by_i[i_P] = s_text
            liquid_by_i[i_P] = l_text
            n_valid_s += nv_s
            n_skip_s += ns_s
            n_valid_l += nv_l
            n_skip_l += ns_l
            n_done += 1
            if log_every and (n_done % log_every == 0):
                logger.info(
                    'paleos_api MgSiO3 2-phase: %d / %d P rows done '
                    '(solid valid=%d skipped=%d | liquid valid=%d skipped=%d)',
                    n_done,
                    grid.n_p,
                    n_valid_s,
                    n_skip_s,
                    n_valid_l,
                    n_skip_l,
                )
    finally:
        if n_workers > 1:
            pool.close()
            pool.join()

    with open(tmp_solid, 'w') as f:
        _write_header(
            f,
            'MgSiO3',
            'solid (metastable extension above liquidus)',
            grid,
            sha,
            n_valid=n_valid_s,
            n_skipped=n_skip_s,
            extra_lines=[
                'Solid polymorph picked by _get_mgsio3_solid_phase; melting-curve test suppressed.',
                'Phase column is the polymorph label even if (P, T) is above the liquidus.',
            ],
        )
        for i in range(grid.n_p):
            f.write(solid_by_i[i])
    _os.replace(tmp_solid, out_solid)

    with open(tmp_liquid, 'w') as f:
        _write_header(
            f,
            'MgSiO3',
            'liquid (Wolf18, metastable extension below solidus)',
            grid,
            sha,
            n_valid=n_valid_l,
            n_skipped=n_skip_l,
            extra_lines=[
                'Wolf18 RTpress evaluated everywhere; phase column is always "liquid".',
            ],
        )
        for i in range(grid.n_p):
            f.write(liquid_by_i[i])
    _os.replace(tmp_liquid, out_liquid)

    logger.info(
        'paleos_api MgSiO3 2-phase: wrote %s (valid=%d/%d) and %s (valid=%d/%d) '
        'sha=%s n_workers=%d',
        out_solid,
        n_valid_s,
        grid.n_p * grid.n_t,
        out_liquid,
        n_valid_l,
        grid.n_p * grid.n_t,
        sha[:10],
        n_workers,
    )
    return {
        'solid': {'n_valid': n_valid_s, 'n_skipped': n_skip_s, 'path': str(out_solid)},
        'liquid': {'n_valid': n_valid_l, 'n_skipped': n_skip_l, 'path': str(out_liquid)},
        'sha': sha,
    }
