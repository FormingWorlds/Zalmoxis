# Installation

## Python environment

You will need to install Python (>=3.12) on your system.

## Download the framework

1. Create a virtual environment

    ```console
    python -m venv .venv
    source .venv/bin/activate
    ```
2. Download Zalmoxis base

    ```console
    git clone https://github.com/FormingWorlds/Zalmoxis.git
    cd Zalmoxis
    pip install -e . 
    bash src/get_zalmoxis.sh  
    ```
3. Set the environment variable

    ```console
    export ZALMOXIS_ROOT=$(pwd)
    ```