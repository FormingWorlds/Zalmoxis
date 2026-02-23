# Installation

## Prerequisites

Zalmoxis requires the following:

- **Operating system**: macOS or Linux (Windows is not supported)
- **Python**: >= 3.12
- **Git**: for cloning the repository
- **Disk space**: approximately 500 MB for tabulated EOS data files (if using the full install)
- **Optional**: the tabulated EOS data download is only required for models that use pre-computed equation-of-state tables. If you only need the analytic Seager et al. (2007) EOS, no data download is necessary (see [Minimal install](#minimal-install-analytic-eos-only) below).

## Install options

Zalmoxis supports two installation modes, depending on whether you need the tabulated EOS data.

### Minimal install (analytic EOS only)

The minimal install requires no data download and provides access to the analytic modified polytrope EOS from Seager et al. (2007). This is sufficient for models that use `Analytic:<material>` EOS strings (e.g., `Analytic:iron`, `Analytic:MgSiO3`, `Analytic:H2O`, `Analytic:graphite`, `Analytic:SiC`, `Analytic:MgFeSiO3`). Follow Steps 1 through 3 below and skip Step 4.

### Full install (all tabulated EOS data)

The full install downloads pre-computed EOS tables (Seager 2007 tabulated, Wolf & Bower 2018, melting curves, etc.) required for models that reference tabulated EOS data. Complete all four steps below, including Step 4.

## Installation steps

### Step 1: Create a virtual environment

```console
python -m venv .venv
source .venv/bin/activate
```

### Step 2: Clone the repository and install dependencies

```console
git clone https://github.com/FormingWorlds/Zalmoxis.git
cd Zalmoxis
pip install -e .
```

This installs Zalmoxis in editable mode, so local changes to the code are immediately reflected.

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

### Step 4: Download tabulated EOS data (full install only)

Run the provided script to download required model files:

```console
bash src/get_zalmoxis.sh
```

This creates the `data/` folder for configuration and EOS table files and the `output_files/` folder for model results. Skip this step if you only intend to use the analytic EOS.

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

This error indicates that the tabulated EOS data files have not been downloaded. If your model configuration references a tabulated (non-analytic) EOS, you must complete Step 4 to download the data. Alternatively, switch your configuration to use analytic EOS strings (e.g., `Analytic:iron`, `Analytic:MgSiO3`) which do not require data files.

### Import errors

```
ModuleNotFoundError: No module named 'zalmoxis'
```

Verify that you are running Python from the virtual environment where Zalmoxis was installed:

```console
source .venv/bin/activate
which python  # should point to .venv/bin/python
```

If you installed Zalmoxis in a conda environment instead, activate that environment:

```console
conda activate <your-environment-name>
```

### Convergence failures

If the interior structure solver fails to converge, consider the following adjustments:

- **Tolerance parameters**: Relax the convergence tolerance in the input configuration file. Tighter tolerances require more iterations and may not converge for extreme planetary compositions or masses.
- **Initial guess**: Poor initial guesses for the central pressure or density can lead to divergence. Try adjusting the starting values.
- **Physical plausibility**: Verify that the input parameters (mass, composition fractions, core/mantle fractions) are physically plausible. Unphysical configurations (e.g., negative mass fractions, zero-thickness layers) will not converge.
