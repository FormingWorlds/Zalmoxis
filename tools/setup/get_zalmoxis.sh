# Exit immediately if any command fails
set -e

if [ -z "$FWL_DATA" ]; then
    echo "FWL_DATA is not set: point it at the directory for PROTEUS ecosystem data, e.g. export FWL_DATA=\$HOME/fwl_data" >&2
    exit 1
fi

echo "Starting Zalmoxis data setup..."

# Run the python setup script that downloads and prepares data
python3 -m tools.setup.setup_zalmoxis
