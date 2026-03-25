# Zalmoxis AI Agent Guidelines

Zalmoxis is the interior structure solver for the PROTEUS ecosystem.
When installed within PROTEUS, see also `../CLAUDE.md` for ecosystem-wide guidelines.

## Quick reference

- `python -m zalmoxis -c input/default.toml` - Run default config
- `python -m pytest -o "addopts=" -m unit` - Run unit tests (override xdist parallel default)
- `python -m pytest -o "addopts=" --cov=src/zalmoxis --cov-report=term-missing` - Coverage
- `ruff check --fix src/ && ruff format src/` - Lint and format
- `python -m src.tools.run_grid <grid.toml> -j <workers>` - Run parameter grid
- `python -m src.tools.plot_grid <dir_or_csv>` - Plot grid results

## Environment

- `ZALMOXIS_ROOT` env var must be set to the repo root. Tests and imports fail without it.
- EOS data lives in `data/` (downloaded via `bash src/get_zalmoxis.sh`)
- Output goes to `output_files/` (gitignored)

## Project layout

- `src/zalmoxis/` - Main package (solver, EOS, mixing, binodal, plotting)
- `src/tests/` - All tests (unit, integration, slow markers)
- `src/tools/` - Standalone scripts (run_grid, plot_grid, setup_tests, converters)
- `input/` - TOML configs and grid specs
- `docs/` - Zensical docs (see below)

## Solver architecture

Three nested loops: structure ODE (innermost) -> density Picard iteration -> mass-radius outer loop.
Main entry: `zalmoxis.py:main()`. Key modules: `eos_functions.py` (EOS lookups), `mixing.py` (multi-material),
`binodal.py` (H2 phase suppression), `structure_model.py` (ODE system), `melting_curves.py`.

## Testing

- 425 unit tests, 41 integration, 2 slow (468 total)
- Unit tests: `pytest -o "addopts=" -m unit` (~5 min with EOS loading)
- The `-o "addopts="` override is needed because pyproject.toml defaults include `-n auto --dist loadfile`
- Use `@pytest.mark.unit` on all new unit tests
- Use `pytest.approx` for float comparisons, never `==`

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
