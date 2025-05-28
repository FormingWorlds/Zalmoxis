# Exit immediately if any command fails
set -e

echo "Starting Zalmoxis data setup..."

# Add src to PYTHONPATH so imports work
export PYTHONPATH="$PYTHONPATH:$(pwd)/src"

# Run the python setup script that downloads and prepares data
python -m src.setup_zalmoxis

echo "Data setup complete!"
