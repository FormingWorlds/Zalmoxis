# Installation

## Python environment

You will need to install Python (>=3.12) on your system.

## Download the framework

Follow these steps to install and configure Zalmoxis:

1. Create a virtual environment

    ```console
    python -m venv .venv
    source .venv/bin/activate
    ```

2. Clone the repository and install dependencies

    ```console
    git clone https://github.com/FormingWorlds/Zalmoxis.git
    cd Zalmoxis
    pip install -e .   
    ```
This installs Zalmoxis in editable mode, so local changes to the code are immediately reflected.

3. Download necessary input/output files

    Run the provided script to download required model files:

    ```console
    bash src/get_zalmoxis.sh
    ```
    This will create the `data/` folder for configuration files and the `output_files/` folder in `src/zalmoxis/` for model results.

3. Set environment variable

    Zalmoxis requires the `ZALMOXIS_ROOT` environment variable to point to the base directory:

    ```console
    export ZALMOXIS_ROOT=$(pwd)
    ```

    To make `ZALMOXIS_ROOT` available across sessions, add the above line to your shell profile file:

    * For `bash` users:

    ```console
    echo "export ZALMOXIS_ROOT=$(pwd)" >> ~/.bashrc
    ```

    * For `zsh` users:

    ```console
    echo "export ZALMOXIS_ROOT=$(pwd)" >> ~/.zshrc
    ```

    Afterwards, reload your profile with:

    * For `bash` users:

    ```console
    source ~/.bashrc 
    ```

    * For `zsh` users:

    ```console
    source ~/.zshrc
    ```

