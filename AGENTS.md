# Zalmoxis agent instructions

Zalmoxis is the interior-structure solver for the PROTEUS ecosystem. It runs
standalone (single-planet structure for a fixed M, composition, T profile) or
inside PROTEUS (per-iteration recompute coupled to atmosphere, energetics,
and outgassing). The ecosystem-wide rules are in the shared block below. End-user
documentation is in `docs/` (Zensical site at `proteus-framework.org/Zalmoxis/`).

## Quick reference

```bash
# Run the default standalone config
python -m zalmoxis -c input/default.toml

# Tests (override pyproject.toml's xdist default)
python -m pytest -o "addopts=" -m unit
python -m pytest -o "addopts=" -m smoke
python -m pytest -o "addopts=" -m "(unit or smoke or integration) and not slow" \
    --cov=src/zalmoxis --cov-report=term-missing

# Lint and format
ruff check --fix src/ tests/ tools/ && ruff format src/ tests/ tools/

# Parameter sweeps and plotting
python -m tools.grids.run_grid <grid.toml> -j <workers>
python -m tools.grids.plot_grid <dir_or_csv>

# Documentation (Zensical, NOT raw mkdocs)
zensical serve              # live reload on docs/ and src/
zensical build --clean      # full build for CI parity
```

CI (`.github/workflows/CI.yml`) runs `ruff check`, `bash tools/validate_test_structure.sh`, `pytest -m "unit and not slow and not skip" -n auto -o "addopts="`, `zensical build --clean` and `python tools/agents/check_agents_md.py`; the nightly runs the other tiers with the coverage gate. Test rules: `tests/AGENTS.md`.

<!-- fwl-core:begin sha256=1adfa510a4f117ec -->
## PROTEUS ecosystem rules

PROTEUS couples separate module repositories into one model of the evolution of rocky planets and their atmospheres. Its config selects one module per process; the config values are the lowercase names:

- `interior_energetics.module`: `aragog`, `spider` (or the built-in `boundary`); `interior_struct.module`: `zalmoxis`, `spider`
- `atmos_clim.module`: `agni`, `janus`, both with SOCRATES spectral radiative transfer
- `outgas.module`: `calliope`, `atmodeller`; `atmos_chem.module`: `vulcan`
- `star.module`: `mors`; `escape.module`: `zephyrus`, `boreas`; `orbit.module` (tides): `obliqua`, `lovepy`
- fwl-io downloads the reference data the modules use

A module change to anything PROTEUS calls or reads (a function signature, a config key, an output column, a unit) can break the coupled model while the module's own tests pass. Search `src/proteus/` in PROTEUS for the name before you change it, and name the affected PROTEUS call sites in the pull request.

### Physics and numerics

- Units differ between and within modules: MORS and VULCAN use cgs; CALLIOPE and atmodeller work with pressures in bar; AGNI takes surface pressure in bar at setup and holds Pa inside; PROTEUS states the unit of every `hf_row` key in `GetHelpfileKeys` (`src/proteus/utils/coupler.py`: atmospheric pressures in bar, interior pressures in Pa, model time in years, orbital periods in s), and its config reference pages state each key's unit (for example stellar mass in M_sun, initial partial surface pressures in bar, stellar age in Gyr). State the unit of every physical quantity in its docstring and convert explicitly at the boundary: a unit mismatch between two modules passes the tests of both and shows only in the coupled run.
- A physics test must fail for the most plausible wrong formula: check a conservation law, a bound, a monotonicity, or a published or analytical value, and assert that the wrong result falls outside the tolerance.
- Take each physical constant from one source per repository (in Python `scipy.constants` or the module's constants file). Two retyped values of one constant differ at round-off and hide real differences between code paths.
- Do not loosen a solver tolerance, a conservation check or a clamp to make a run or a test pass. Find the cause; a loosened check also hides the next defect.

### Validation

A new or changed physics routine gets the validation cases named in its confirmed plan: an analytical law, a published figure or table, or a cross-code check. Each case is a `reference_pinned` test that runs the real code on real data, with no mocks, asserts the tolerance, and writes its light and dark plot through the `validation_plot` fixture of proteus-mpl. `pytest -m reference_pinned --validation-plots` makes all plots again. Line plots are SVG, maps are AVIF; each file is at most 60 kB, under `docs/assets/validation/`. Each case has a subsection on its page in the Validation tab, with the reference, the tolerance, the test id and the command. Data read from a published figure comes from 2 independent figlab reads that agree within the stated error, as a CSV with its provenance under `tests/validation/data/`.

### Code

- Keep an inline comment block to 2 lines or fewer, and never more than 4, because a long block drifts from the code it describes. An explanation that needs more goes in the docstring, the commit message or the pull-request description.

### Branches and pull requests

- Work on a feature branch `<initials>/<short-description>`; `main` changes only through a reviewed pull request.
- Fill every section of the repository's pull-request template, where it has one.
- Before you push, run the checks listed at the top of this `AGENTS.md` and in `tests/AGENTS.md`.

### Where knowledge goes

Rules for every contributor go in the `AGENTS.md` files, the reason for a change in its commit message and pull-request description, and the scientific validation of a module in its `docs/Validation/` pages, where the repository has them. Do not add memory, notes or decision-log files to the repository: nobody maintains them, and they go stale.
<!-- fwl-core:end -->

<!-- fwl-voice:begin sha256=b4fe24baeef8b734 -->
### Commit messages and public text

Commit messages, pull-request text, code comments, docstrings, test names, test skip reasons, parametrize ids, log strings that ship with the code, and CI job and step names describe the change and the current state of the code. They name no tool used to write the change, carry no tool-attribution trailer, use no internal plan, phase or work-group labels, and use no em or en dashes (a page range in a citation is the exception).
<!-- fwl-voice:end -->

## Environment

- `ZALMOXIS_ROOT` is resolved lazily by `get_zalmoxis_root()` in `__init__.py`.
  Auto-detects from package location; set explicitly with
  `export ZALMOXIS_ROOT=$(pwd)` only if the auto-detection picks the wrong tree.
- EOS data is fetched through fwl-io into `$FWL_DATA` (~2.3 GB); each folder of
  `data/` (gitignored) links to its dataset. Bootstrap with
  `FWL_DATA=<dir> bash tools/setup/get_zalmoxis.sh`.
- Output goes to `output/` (gitignored).
- The proteus conda env is required for development on machines that ship the
  PROTEUS toolchain (Python 3.12, numpy 2, jax, diffrax). Activate with
  `conda activate proteus`. Verify with `which python` (ends in
  `envs/proteus/bin/python`).

## Project layout

Package layout and module responsibilities: `docs/Explanations/code_architecture.md`. `src/zalmoxis/_version.py` is written by setuptools-scm.

## Solver architecture

Three nested fixed-point loops: outer mass-radius search, middle density Picard, inner structure ODE. `docs/Explanations/code_architecture.md` describes each loop, the Anderson mixing and the adiabat anchor; its JAX section names an older support envelope. The Newton tolerances and the current JAX envelope:

- `outer_solver = 'newton'` (PROTEUS schema default, `proteus.config._struct`) auto-tightens the integrator tolerances (rel=1e-9, abs=1e-10) so the central-difference dM/dR is smooth.
- **JAX path (opt-in, `use_jax = True`):** dispatches to `jax_eos/` for the
inner ODE; the outer and middle loops are unchanged. x64 mode is enabled at
module import to prevent float32 loss. Parity guarantee: `fast_bilinear_jax`
agrees with the numpy reference to rtol ≤ 1e-4. Supported on the JAX path:
2-layer configurations with a paleos_unified core and a mantle in either
representation, the unified single table (`PALEOS:MgSiO3`, the production
default; same kernel as the core with the mantle mushy zone factor, static
`mantle_is_unified` flag) or the 2-phase solid/melted sub-tables
(`PALEOS-2phase:MgSiO3`, Tdep kernel); plus the wet mantle with a
`VolatileProfile` carrying exactly one active paleos_unified volatile (the
phi-blended suppressed harmonic mean of `calculate_mixed_density`, ported
in `jax_eos/rhs.py` under the static `has_volatile` flag) on top of either
mantle. Profiles outside that envelope (Chabrier:H with its binodal
suppression, multi-volatile, `global_miscibility`, `x_interior`) and other
unsupported configs fall back to numpy automatically.

## Standalone execution

The CLI is in `__main__.py` (`get_zalmoxis_root()` → `load_zalmoxis_config()`
→ `post_processing()`) and is invoked as `python -m zalmoxis -c <config.toml>`.
The TOML is the single input; there are no behaviour-controlling flags.

The `[project.scripts]` entry in `pyproject.toml` registers `zalmoxis` as
`zalmoxis.solver:main`. This bypasses `load_zalmoxis_config()` and is useful
mainly for in-process callers; humans should prefer `python -m zalmoxis`.

Outputs (under `output/`):

- `planet_profile.txt`: six-column TSV with r, ρ, M(r), P(r), g(r), T(r).
- `calculated_planet_mass_radius.txt`: appended one-line summary per run.
- `zalmoxis.log`: per-run log (overwritten each invocation).
- Optional debug: `density_profiles.txt`, `pressure_profiles.txt` per Picard
  iteration when the Python logging level is `DEBUG`.

Recommended workflow:

1. Copy `input/default.toml` to a working location.
2. Edit `[InputParameter]` (planet_mass), `[AssumptionsAndInitialGuesses]`
   (core fraction, temperature mode), and `[EOS]` (per-layer EOS strings).
3. Run with `-c your.toml`.
4. For figures, set `[Output] plots_enabled = true`.

For sweeps: `python -m tools.grids.run_grid <grid.toml> -j <workers>`.
The grid runner produces a flat CSV and a per-cell output directory, both
plottable via `python -m tools.grids.plot_grid <dir_or_csv>`.

## PROTEUS integration

When PROTEUS drives Zalmoxis the configuration lives in PROTEUS's own TOML
schema, not in `input/default.toml`. The PROTEUS-side attrs class is
`proteus.config._struct.Zalmoxis`; defaults there override several
Zalmoxis-side defaults, most notably `outer_solver = 'newton'`.

Two call sites in PROTEUS:

- **Init solve**: `proteus.interior_struct.zalmoxis.zalmoxis_solver`,
  invoked once before the main loop.
- **Time-evolution re-solve**: `proteus.interior_energetics.wrapper.update_structure_from_interior`,
  gated by `update_interval`, `update_dphi_abs`, `update_dtmagma_frac`, and a
  stale-time ceiling.

The `hf_row` reads and writes and the mesh handover to Aragog and SPIDER are in `docs/Explanations/proteus_coupling.md`.

EOS-table generation:

- `interior_energetics.module = 'spider'`: Zalmoxis writes P-S tables via
  `generate_spider_tables` (resolution `lookup_nP × lookup_nS`).
- `interior_energetics.module = 'aragog'`: Zalmoxis writes P-T tables via
  `generate_aragog_pt_tables` (or `_2phase`). Aragog requires rectangular
  P-T grids; phase-filtering breaks `scipy.RegularGridInterpolator`.

Recommended PROTEUS-side settings (in priority order):

1. `interior_struct.module = 'zalmoxis'` + `interior_energetics.module = 'aragog'`
   for new production runs; `'spider'` for legacy parity comparisons.
2. `interior_struct.zalmoxis.update_interval`: 5e4 yr for dynamic structure
   updates during evolution; ≥1e9 to freeze the mesh after the init solve.
3. `interior_struct.zalmoxis.outer_solver = 'newton'` (PROTEUS default).
4. `interior_energetics.aragog.phi_step_cap`: 0.05 for typical evolution;
   leave at 0.0 unless mushy-zone melt-fraction oscillations show up early.
5. `planet.prevent_warming = false`. Must stay false; the clamp is
   energy-non-conserving and produces a spurious T_magma plateau when true.
6. `interior_struct.zalmoxis.mushy_zone_factor = 0.8` for paper-quality
   PALEOS runs (Stixrude+14 cryoscopic depression).
7. `interior_energetics.num_levels`: 80 for both Aragog and SPIDER.
8. `interior_struct.zalmoxis.equilibrate_init = true` (CALLIOPE+Zalmoxis
   pre-main-loop equilibration; default).

Pitfalls:

- *No `dilatation` field*: PROTEUS has no `dilatation` field; a config
  that sets `interior_struct.zalmoxis.dilatation` is rejected with
  `UnknownConfigKeyError`.
- *Resume + ULP radius drift*: a resumed run can hit single-ULP mismatch
  between saved Zalmoxis radii and Aragog's mesh bounds at
  `aragog/src/aragog/solver/entropy_solver.py:91-110`.
- *core_density echo-back*: SPIDER's `-rho_core` parameter can overwrite
  Zalmoxis's self-consistent core density; the mass-anchor check at
  `proteus/interior_energetics/wrapper.py` (rtol 3e-3) catches gross
  mismatches.

For full theory and the prioritised-settings table, see
`docs/Explanations/proteus_coupling.md` and `docs/How-to/proteus_coupling.md`.

## Import conventions

No backward-compatibility shims; imports are direct to the actual module:

- `from zalmoxis import get_zalmoxis_root`
- `from zalmoxis.config import load_zalmoxis_config, validate_config`
- `from zalmoxis.solver import main`
- `from zalmoxis.output import post_processing`
- `from zalmoxis.eos import calculate_density, load_paleos_unified_table`
- `from zalmoxis.mixing import LayerMixture`
- `from zalmoxis.binodal import h2_silicate_miscibility, h2_water_miscibility`

`from __future__ import annotations` is required at the top of every `.py`
file (enforced by ruff isort).

## Documentation (Zensical)

- The site builds with **Zensical**, not raw mkdocs. Use `zensical serve`
  and `zensical build --clean`. Raw `mkdocs serve` may fail on theme
  resolution.
- Live reload: `zensical serve` watches `docs/` and `src/` automatically.
- Root-level files (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `README.md`)
  are pulled into `docs/Community/` via `markdown_include`. Zensical does
  not detect changes to these on live-reload; stop the server and run
  `zensical build --clean` to pick them up.
- URLs use `use_directory_urls: false` (pages are `*.html`, not
  `*/index.html`).
- Nav is defined in `mkdocs.yml`. New pages need both the file and the nav
  entry.
- mkdocstrings `:::` directives auto-generate API pages from docstrings;
  add a stub under `docs/Reference/api/` for any new module.

## Code style

- `ruff` for linting and formatting (config in `pyproject.toml`).
- Single quotes (configured in ruff).
- `from __future__ import annotations` required in all files.
- Line length 96 (prefer < 92).
- Run `ruff format` before committing; the local ruff (0.12.x) and the CI
  ruff (0.15.x) sometimes disagree on formatting drift, and CI is canonical.
- NumPy-style docstrings (Parameters / Returns / Raises / Notes).
- Comments should explain WHY, not WHAT. No project-tracking labels
  (T1.x, Stage X, dates of past changes); the development history belongs
  in commit messages, not in inline comments.
