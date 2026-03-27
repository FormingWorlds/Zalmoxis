# Releasing

Zalmoxis uses [setuptools-scm](https://setuptools-scm.readthedocs.io/) for automatic versioning. The version is derived from git tags, so there are no version strings to edit manually anywhere in the codebase.

## Version scheme

Zalmoxis follows [Calendar Versioning](https://calver.org/) (CalVer) with the format `YY.MM.DD`:

- **Release versions** (from tags): `25.03.27`, `26.01.15`
- **Development versions** (between tags): `25.03.27.dev5+gabc1234` (5 commits after the `v25.03.27` tag, at commit `abc1234`)

## How to make a release

### 1. Ensure main is ready

All CI checks should pass on the `main` branch. Verify locally:

```console
pytest -m "unit or integration" -o "addopts="
ruff check src/ tests/ tools/
```

### 2. Tag the release

From the `main` branch (after merging any feature branches):

```console
git checkout main
git pull
git tag v25.03.27       # Use today's date in YY.MM.DD format
git push origin v25.03.27
```

!!! warning "Tag format"
    The tag **must** start with `v` followed by the CalVer date (e.g., `v25.03.27`). setuptools-scm strips the `v` prefix automatically.

### 3. Create a GitHub Release

```console
gh release create v25.03.27 --title "v25.03.27" --generate-notes
```

Or create the release manually at [github.com/FormingWorlds/Zalmoxis/releases/new](https://github.com/FormingWorlds/Zalmoxis/releases/new):

1. Choose the tag you just pushed (`v25.03.27`)
2. Set the title to the version (`v25.03.27`)
3. Click "Generate release notes" for an automatic changelog
4. Click "Publish release"

### 4. PyPI publication (automatic)

Publishing the GitHub Release triggers the `publish.yml` workflow, which:

1. Checks out the repository with full git history (for tag resolution)
2. Builds the package with `python -m build` (setuptools-scm writes the version from the tag)
3. Publishes to PyPI via trusted publishing (no API tokens needed)

The package appears on [pypi.org/project/fwl-zalmoxis](https://pypi.org/project/fwl-zalmoxis/) within a few minutes.

### 5. Verify

```console
pip install --upgrade fwl-zalmoxis
python -c "from zalmoxis import __version__; print(__version__)"
# Should print: 25.03.27
```

## How versioning works

setuptools-scm reads the git history to determine the version:

| Scenario | Example version |
|----------|----------------|
| Exactly on a tag (`v25.03.27`) | `25.03.27` |
| 5 commits after tag, dirty | `25.03.27.dev5+gabc1234.d20250328` |
| No tags in history | `0.0.0` (fallback) |

The version is written to `src/zalmoxis/_version.py` at install time (this file is gitignored). At runtime, `__init__.py` imports it:

```python
from zalmoxis import __version__
```

## Multiple releases per day

If you need a second release on the same day, append a patch number:

```console
git tag v25.03.27.1
```

This produces version `25.03.27.1`.

## Troubleshooting

### Version shows `0.0.0`

setuptools-scm cannot find any tags. Check:

```console
git tag --list 'v*'
git describe --tags
```

If no tags appear, you may need to fetch them:

```console
git fetch --tags
```

### CI build shows wrong version

The CI checkout must use `fetch-depth: 0` (full history). This is already configured in `CI.yml` and `publish.yml`.
