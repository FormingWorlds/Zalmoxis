# Contributing guidelines

## Development

### Setup

Install Zalmoxis with development dependencies:

```console
pip install -e ".[develop]"
```

This includes pytest, pytest-xdist (parallel test execution), ruff (linting/formatting), coverage tools, and pre-commit hooks.

### Testing

Tests are categorized by speed using pytest markers:

```console
pytest -m unit          # ~2s   -- EOS functions, no solver
pytest -m integration   # ~10min -- full solver validation
pytest -m slow          # ~30min -- composition grid sweep
pytest -m "not slow"    # everything except the slow grid sweep
pytest                  # all tests in parallel
```

Run `pytest -m unit` as a fast feedback loop during development. The full suite runs automatically on PRs.

When adding or modifying code, add or update tests in `src/tests/` to match. See the [Testing documentation](https://proteus-framework.org/Zalmoxis/How-to/testing.html) for the full guide on markers, fixtures, and test structure.

### Linting

```console
ruff check --fix src/ tests/
ruff format src/ tests/
```

### Building the documentation

The documentation is written in [markdown](https://www.markdownguide.org/basic-syntax/) and built with [Zensical](https://zensical.org/docs/get-started/), a modern static site generator compatible with mkdocs, used across the PROTEUS ecosystem.

Install the docs dependencies once:

```console
pip install -e ".[docs]"
```

To preview the documentation locally with live reload (auto-rebuilds and refreshes the browser on every file change):

```console
zensical serve
```

This serves at `http://127.0.0.1:8000` by default. Use `-a 127.0.0.1:<port>` for a different port. The `serve` command handles building internally; there is no need to run `build` first. To open the browser automatically, add `--open`.

To produce a static build without serving (e.g., for CI):

```console
zensical build --clean
```

The `--clean` flag clears the `site/` cache before building.

**Do not use `mkdocs serve` or `mkdocs build` directly.** Zensical replaces mkdocs with its own build pipeline. The raw mkdocs commands may fail on theme or icon resolution. Similarly, do not run `zensical build` before `zensical serve`, as `serve` manages its own build and a stale `site/` directory from a prior `build` can cause caching issues.

You can find the documentation source in the [docs](https://github.com/FormingWorlds/Zalmoxis/tree/main/docs) directory.
If you are adding new pages, make sure to update the listing in the [`mkdocs.yml`](https://github.com/FormingWorlds/Zalmoxis/tree/main/mkdocs.yml) under the `nav` entry.

**Live reload caveat for root-level files.** Some pages (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `README.md`) live at the repository root and are included into the docs via `markdown_include` directives in `docs/Community/`. Zensical's live reload does not detect changes to these included root-level files. If you edit one and the browser does not update, stop the server (Ctrl+C) and restart with a clean build:

```console
zensical build --clean
zensical serve
```

The `build --clean` step forces Zensical to re-resolve all `markdown_include` directives. Without it, `serve` may use a stale internal cache even if `site/` was deleted.

