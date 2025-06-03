# setup_zalmoxis.py (at root)

from src.tools.setup_utils import download_data, create_output_files

if __name__ == "__main__":
    # Download and extract data
    download_data()
    # Create output files directory
    create_output_files()
