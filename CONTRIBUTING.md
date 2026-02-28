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

When adding or modifying code, add or update tests in `src/tests/` to match. See the [Testing documentation](https://proteus-framework.org/Zalmoxis/testing/) for the full guide on markers, fixtures, and test structure.

### Linting

```console
ruff check --fix src/ tests/
ruff format src/ tests/
```

### Building the documentation

The documentation is written in [markdown](https://www.markdownguide.org/basic-syntax/), and uses [mkdocs](https://www.mkdocs.org/) to generate the pages.

To build the documentation for yourself in editable mode:

```console
pip install -e .[docs]
mkdocs serve
```

This will generate the pages and serve them on a local development server. Copy the displayed URL (typically `http://127.0.0.1:8000`) into your browser to view the documentation as you edit.

You can find the documentation source in the [docs](https://github.com/FormingWorlds/Zalmoxis/tree/main/docs) directory.
If you are adding new pages, make sure to update the listing in the [`mkdocs.yml`](https://github.com/FormingWorlds/Zalmoxis/tree/main/mkdocs.yml) under the `nav` entry.

