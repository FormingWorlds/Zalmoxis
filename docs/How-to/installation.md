# Installation

!!! note
    The standard way of installing this version of Zalmoxis is within the PROTEUS Framework, as described in the [PROTEUS installation guide](https://proteus-framework.org/PROTEUS/installation.html#9-install-submodules-as-editable). 

### Prerequisites
- **Python:** >=3.12 installed
- **pip:** available (`python -m pip --version`)
- **Git:** only needed for the developer install (`git --version`)
- **Internet access:** required to download necessary data
- *(Optional)* **Conda/Anaconda/Miniconda:** only if you want to use a conda environment

### 0. Optional: Conda/virtual environment

Create and activate a Conda environment (requires `conda` installed):
```bash
conda create -n zalmoxis python=3.12 -y
conda activate zalmoxis
```

No `conda`? create and activate a virtual environment (venv):
```bash
python -m venv .venv
source .venv/bin/activate
```

Follow these steps to install and configure Zalmoxis:

### 1. Clone the repository and install dependencies

```bash
git clone https://github.com/FormingWorlds/Zalmoxis.git
cd Zalmoxis
pip install -e .   
```

This installs Zalmoxis in editable mode, so local changes to the code are immediately reflected.

### 2. Set environment variable

Zalmoxis requires the `ZALMOXIS_ROOT` environment variable to point to the base directory:

```bash
export ZALMOXIS_ROOT=$(pwd)
```

To make `ZALMOXIS_ROOT` available across sessions, add the above line to your shell profile file:

* For `bash` users:

```bash
echo "export ZALMOXIS_ROOT=$(pwd)" >> ~/.bashrc
```

* For `zsh` users:

```bash
echo "export ZALMOXIS_ROOT=$(pwd)" >> ~/.zshrc
```

Afterwards, reload your profile with:

* For `bash` users:

```bash
source ~/.bashrc 
```

* For `zsh` users:

```bash
source ~/.zshrc
```

### 3. Download necessary input/output files

Run the provided script to download required model files:

```bash
bash src/get_zalmoxis.sh
```

This will create the `data/` folder for configuration files and the `output_files/` folder for model results.

