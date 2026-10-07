# Installation

!!! note "Installation within the PROTEUS framework"
    The standard way of installing Zalmoxis is within the [PROTEUS framework](https://proteus-framework.org/PROTEUS/), as described in the [PROTEUS installation guide](https://proteus-framework.org/PROTEUS/How-to/installation.html). When installed as part of PROTEUS, Zalmoxis is set up automatically alongside all other modules. The standalone instructions below are only needed if you want to use Zalmoxis independently of PROTEUS.

## Prerequisites

- **Operating system**: macOS (Intel or Apple Silicon) or Linux (Windows is not supported)
- **Python**: 3.12 (recommended; matches the PROTEUS framework requirement)
- **Conda**: [miniforge](https://github.com/conda-forge/miniforge) (macOS) or [miniconda](https://docs.anaconda.com/miniconda/) (Linux) for environment management
- **Git**: for cloning the repository
- **Disk space**: about 2.3 GB in `$FWL_DATA` for the tabulated EOS data files

## Installation steps

### Step 1: Create and activate a conda environment

```console
conda create -n zalmoxis python=3.12
conda activate zalmoxis
```

If you are already working inside a PROTEUS conda environment (`conda activate proteus`), you can skip this step and install Zalmoxis into that environment directly.

### Step 2: Clone the repository and install dependencies

```console
git clone https://github.com/FormingWorlds/Zalmoxis.git
cd Zalmoxis
pip install -e .
```

This installs Zalmoxis in editable mode, so local changes to the code are immediately reflected.

For development (includes pytest, pytest-xdist, ruff, coverage, and pre-commit):

```console
pip install -e ".[develop]"
```

The `[develop]` extras are required for running the test suite. See the [Testing](../Explanations/testing.md) page for details.

### Step 3: Set the environment variable

Zalmoxis auto-detects its root directory from the package installation path. If auto-detection fails (e.g., non-standard installation layout), set the environment variable explicitly:

```console
export ZALMOXIS_ROOT=$(pwd)
```

To make `ZALMOXIS_ROOT` available across sessions, add the above line to your shell profile file:

* For `bash` users:

```console
echo "export ZALMOXIS_ROOT=$(pwd)" >> ~/.bashrc
source ~/.bashrc
```

* For `zsh` users:

```console
echo "export ZALMOXIS_ROOT=$(pwd)" >> ~/.zshrc
source ~/.zshrc
```

### Step 4: Download EOS data

Set `FWL_DATA` to the directory for PROTEUS ecosystem data, then run the provided script to download the required equation-of-state tables and reference data:

```console
export FWL_DATA=/path/to/fwl_data
bash tools/setup/get_zalmoxis.sh
```

The script fetches the data through [fwl-io](https://github.com/FormingWorlds/fwl-io) from its Zenodo records, with their DataverseNL mirrors as the fallback, checks every file against the registry, and stores it in `$FWL_DATA`, where PROTEUS reads the same copy. Each folder of `data/` in the repository is a link to its dataset there. The script replaces a link it made (to a version of the same dataset), a dangling link, or an empty folder. A folder with files, a file, or a link to another place is kept, and the script ends with a warning that lists the command to remove each one; remove them and run the script again to use the fetched copy. The script also creates the `output/` folder for model results.

### Step 5: Run your first simulation

Run the default configuration (1 Earth-mass planet with a PALEOS iron core and MgSiO3 mantle):

```console
python -m zalmoxis -c input/default.toml
```

Output files are written to `output/`. See the [usage guide](usage.md) for an explanation of the output files and how to modify the configuration, or the [parameter grids guide](grids.md) for running parameter sweeps.

## Troubleshooting

### `ZALMOXIS_ROOT` not set

```
RuntimeError: ZALMOXIS_ROOT environment variable is not set and could not be
auto-detected. Set it explicitly: export ZALMOXIS_ROOT=/path/to/Zalmoxis
```

This error occurs when auto-detection of the repository root fails and the `ZALMOXIS_ROOT` environment variable is not defined. Set it to the root of the Zalmoxis repository:

```console
export ZALMOXIS_ROOT=/path/to/Zalmoxis
```

If you added the variable to your shell profile (Step 3) but still see the error, reload your profile (`source ~/.bashrc` or `source ~/.zshrc`) or open a new terminal session.

### Data files missing

```
FileNotFoundError: [Errno 2] No such file or directory: '.../data/EOS_Seager2007/...'
```

This error indicates that the tabulated EOS data files have not been downloaded. Run `bash tools/setup/get_zalmoxis.sh` from the Zalmoxis root directory to complete Step 4.

### A `data/` link points nowhere

Each folder in `data/` is a link into `$FWL_DATA`. After `fwl-io prune` removes a superseded version, or after `$FWL_DATA` moves, a link can point at a folder that no longer exists. The error names the cause, `Data file ... is missing: .../data/<folder> links to ..., which does not exist`: for an EOS table it is the first ERROR line in `output/zalmoxis.log` (the run itself stops with a `StructureSolveError`), and for a melting curve it is in the traceback. Run `bash tools/setup/get_zalmoxis.sh` again with `FWL_DATA` set: it fetches what is missing and replaces every link it made, every dangling link and every empty folder; a folder with files in it, or a link to another place, is kept and listed at the end with the line that removes it; remove it and run the script again.

### Import errors

```
ModuleNotFoundError: No module named 'zalmoxis'
```

Verify that you are running Python from the conda environment where Zalmoxis was installed:

```console
conda activate zalmoxis
which python  # should point to your conda env
```

If you installed Zalmoxis into the PROTEUS environment, activate that instead:

```console
conda activate proteus
```

### Convergence failures

The Brent pressure solver is robust and typically converges in 20 to 36 evaluations.
If the solver fails to converge, consider the following:

- **Bracket error** (`ValueError: f(a) and f(b) must have different signs`): The initial pressure bracket does not straddle the root. This usually means the true central pressure is outside the bracket range. Try increasing `max_center_pressure_guess` (for WolfBower2018 EOS) or check that the planet mass and composition are physically plausible.
- **WolfBower2018 mass limit**: The `WolfBower2018:MgSiO3` EOS is limited to $\leq 7\,M_\oplus$. For higher-mass planets, use `PALEOS:MgSiO3`, `RTPress100TPa:MgSiO3`, `Seager2007:MgSiO3`, or `Analytic:MgSiO3` instead.
- **Tolerance parameters**: Relax the convergence tolerance in the input configuration file. Tighter tolerances require more iterations and may not converge for extreme planetary compositions or masses.
- **Physical plausibility**: Verify that the input parameters (mass, composition fractions, core/mantle fractions) are physically plausible. Unphysical configurations (e.g., negative mass fractions, zero-thickness layers) will not converge.

!!! tip "JAX determinism for fragile runs"
    Numerically fragile coupled runs (wet 1 \(M_\oplus\) at IW+4, reduced 1 \(M_\oplus\) at IW-2) benefit from JAX 64-bit and a single-thread XLA. Set:

    ```bash
    export JAX_ENABLE_X64=1
    export XLA_FLAGS="--xla_force_host_platform_device_count=1"
    ```

    Zalmoxis enables x64 by default at module import, so the env var is mostly a safety net for unusual harnesses.
