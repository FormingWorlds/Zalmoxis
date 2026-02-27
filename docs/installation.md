# Installation

!!! note "Installation within the PROTEUS framework"
    The standard way of installing Zalmoxis is within the [PROTEUS framework](https://proteus-framework.org/PROTEUS/), as described in the [PROTEUS installation guide](https://proteus-framework.org/PROTEUS/installation.html). When installed as part of PROTEUS, Zalmoxis is set up automatically alongside all other modules. The standalone instructions below are only needed if you want to use Zalmoxis independently of PROTEUS.

## Prerequisites

- **Operating system**: macOS (Intel or Apple Silicon) or Linux (Windows is not supported)
- **Python**: 3.12 (recommended; matches the PROTEUS framework requirement)
- **Conda**: [miniforge](https://github.com/conda-forge/miniforge) (macOS) or [miniconda](https://docs.anaconda.com/miniconda/) (Linux) for environment management
- **Git**: for cloning the repository
- **Disk space**: approximately 500 MB for tabulated EOS data files

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

The `[develop]` extras are required for running the test suite. See the [Testing](testing.md) page for details.

### Step 3: Set the environment variable

Zalmoxis requires the `ZALMOXIS_ROOT` environment variable to point to the base directory:

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

Run the provided script to download the required equation-of-state tables and reference data:

```console
bash src/get_zalmoxis.sh
```

This downloads data into the `data/` directory within the Zalmoxis repository (not into `FWL_DATA`). When Zalmoxis is installed within PROTEUS, the data path is managed by the PROTEUS framework. The script also creates the `output_files/` folder for model results.

## Troubleshooting

### `ZALMOXIS_ROOT` not set

```
RuntimeError: ZALMOXIS_ROOT environment variable not set
```

This error occurs when the `ZALMOXIS_ROOT` environment variable is not defined in your current shell session. Set it to the root of the Zalmoxis repository:

```console
export ZALMOXIS_ROOT=/path/to/Zalmoxis
```

If you added the variable to your shell profile (Step 3) but still see the error, reload your profile (`source ~/.bashrc` or `source ~/.zshrc`) or open a new terminal session.

### Data files missing

```
FileNotFoundError: [Errno 2] No such file or directory: '.../data/EOS_Seager2007/...'
```

This error indicates that the tabulated EOS data files have not been downloaded. Run `bash src/get_zalmoxis.sh` from the Zalmoxis root directory to complete Step 4.

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

The Brent pressure solver is robust and typically converges in 20--36 evaluations.
If the solver fails to converge, consider the following:

- **Bracket error** (`ValueError: f(a) and f(b) must have different signs`): The initial pressure bracket does not straddle the root. This usually means the true central pressure is outside the bracket range. Try increasing `max_center_pressure_guess` (for WolfBower2018 EOS) or check that the planet mass and composition are physically plausible.
- **WolfBower2018 mass limit**: The `WolfBower2018:MgSiO3` EOS is limited to $\leq 7\,M_\oplus$. For higher-mass planets, use `Seager2007:MgSiO3` or `Analytic:MgSiO3` instead.
- **Tolerance parameters**: Relax the convergence tolerance in the input configuration file. Tighter tolerances require more iterations and may not converge for extreme planetary compositions or masses.
- **Physical plausibility**: Verify that the input parameters (mass, composition fractions, core/mantle fractions) are physically plausible. Unphysical configurations (e.g., negative mass fractions, zero-thickness layers) will not converge.
