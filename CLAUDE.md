# Zalmoxis AI Agent Guidelines

Zalmoxis is the interior structure solver for the PROTEUS ecosystem.
When installed within PROTEUS, see also `../CLAUDE.md` for ecosystem-wide guidelines.

## Quick reference

- `python -m zalmoxis -c input/default.toml` - Run default config
- `python -m pytest -o "addopts=" -m unit` - Run unit tests (override xdist parallel default)
- `python -m pytest -o "addopts=" --cov=src/zalmoxis --cov-report=term-missing` - Coverage
- `ruff check --fix src/ tests/ tools/ && ruff format src/ tests/ tools/` - Lint and format
- `python -m tools.grids.run_grid <grid.toml> -j <workers>` - Run parameter grid
- `python -m tools.grids.plot_grid <dir_or_csv>` - Plot grid results

## Environment

- `ZALMOXIS_ROOT` is resolved lazily via `get_zalmoxis_root()` in `__init__.py`. Auto-detects from package location; set explicitly with `export ZALMOXIS_ROOT=$(pwd)` if needed.
- EOS data lives in `data/` (downloaded via `bash tools/setup/get_zalmoxis.sh`)
- Output goes to `output/` (gitignored)

## Project layout

```
Zalmoxis/
  src/zalmoxis/           # Core package (installed via pip)
    __init__.py           # get_zalmoxis_root() (lazy), __version__
    config.py             # Config parsing, validation, EOS/melting setup
    solver.py             # main() solver loop (3 nested iterations)
    output.py             # post_processing(), file output
    structure_model.py    # ODE system (dM/dr, dg/dr, dP/dr), solve_structure()
    eos/                  # EOS package, organized by family
      __init__.py         # Re-exports all public functions
      interpolation.py    # Shared grid builders, bilinear interp, table loaders
      seager.py           # Seager2007 tabulated 1D P-rho lookups
      paleos.py           # Unified PALEOS density + nabla_ad, mushy zone
      tdep.py             # T-dependent EOS, melting curves, phase routing
      dispatch.py         # calculate_density/batch (main entry points)
      temperature.py      # Adiabat computation, T profiles
      output.py           # Pressure/density profile file writing
    eos_analytic.py       # Seager+2007 analytic polytrope (6 materials)
    eos_properties.py     # Lazy EOS_REGISTRY (paths built on first access)
    mixing.py             # Multi-component mixing, LayerMixture, suppression
    melting_curves.py     # Solidus/liquidus functions
    binodal.py            # H2-MgSiO3 and H2-H2O miscibility models
    constants.py          # Physical constants (G, earth_mass, earth_radius)
  tests/                  # Test suite (at repo root)
  tools/                  # Standalone scripts (at repo root)
    setup/                # Test fixtures, data download
    validation/           # First-principles verification
    benchmarks/           # EOS benchmarks
    grids/                # Parameter sweep runners
    converters/           # EOS format conversion
    plots/                # Visualization scripts
  input/                  # TOML configs and grid specs
  data/                   # EOS tables (gitignored, ~600 MB)
  output/                 # Generated outputs (gitignored)
  docs/                   # Zensical documentation
```

## Solver architecture

Three nested loops: structure ODE (innermost) -> density Picard iteration -> mass-radius outer loop.
Main entry: `solver.py:main()`. Key modules: `eos/dispatch.py` (EOS density lookups), `mixing.py` (multi-material),
`binodal.py` (H2 phase suppression), `structure_model.py` (ODE system), `melting_curves.py`.

## Import conventions

No backward-compatibility shims. All imports are direct to the actual module:
- `from zalmoxis.solver import main`
- `from zalmoxis.config import load_zalmoxis_config, validate_config`
- `from zalmoxis.output import post_processing`
- `from zalmoxis.eos import calculate_density, load_paleos_unified_table`
- `from zalmoxis.mixing import LayerMixture`
- `from zalmoxis import get_zalmoxis_root`

## Testing

- 435 unit tests, ~40 integration, 4 slow (~480 total)
- Unit tests: `pytest -o "addopts=" -m unit` (~1.5 min)
- The `-o "addopts="` override is needed because pyproject.toml defaults include `-n auto --dist loadfile`
- Use `@pytest.mark.unit` on all new unit tests
- Use `pytest.approx` for float comparisons, never `==`
- First-principles verification: `tests/test_first_principles.py` (25 tests, analytic solutions)

## Documentation (Zensical)

- **Do not use `mkdocs serve` or `mkdocs build`.** Use `zensical serve` / `zensical build --clean`.
- Live reload: `zensical serve` watches `docs/` and `src/` automatically
- Root-level files (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `README.md`) are included via `markdown_include`
  directives in `docs/Community/`. Zensical does not detect changes to these files during live reload.
  To pick up changes: stop the server, run `zensical build --clean`, then `zensical serve`.
- Docs URL structure uses `use_directory_urls: false` (pages are `*.html`, not `*/index.html`)
- Nav defined in `mkdocs.yml`

## Code style

- `ruff` for linting and formatting (config in pyproject.toml)
- Single quotes (configured in ruff)
- `from __future__ import annotations` required in all files (enforced by ruff isort)
- Line length 96 (prefer < 92)
