#!/bin/bash
#SBATCH --job-name=zalmoxis_vgrid
#SBATCH --partition=regular
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G
#SBATCH --cpus-per-task=1
#SBATCH --output=/scratch/p311056/zalmoxis_validation/logs/%x_%A_%a.out
#SBATCH --error=/scratch/p311056/zalmoxis_validation/logs/%x_%A_%a.err

# Submit a single Zalmoxis validation grid suite as a SLURM array job.
#
# Usage:
#     sbatch tools/validation_grid/submit_suite.sh suite_01_mass_radius
#
# The script reads all .toml files in the suite's config directory,
# sorts them, and uses SLURM_ARRAY_TASK_ID to pick the correct one.
# Array size is set automatically via --array=1-N.

set -euo pipefail

# ---------------------------------------------------------------------------
# Determine paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Suite name from first argument (or from re-submission via SUITE_NAME env var)
SUITE_NAME="${1:-${SUITE_NAME:-}}"
if [ -z "${SUITE_NAME}" ]; then
    echo "Usage: sbatch submit_suite.sh <suite_name>"
    echo "Example: sbatch submit_suite.sh suite_01_mass_radius"
    exit 1
fi

CONFIGS_DIR="${SCRIPT_DIR}/configs/${SUITE_NAME}"
if [ ! -d "${CONFIGS_DIR}" ]; then
    echo "Config directory not found: ${CONFIGS_DIR}"
    echo "Run generate_configs.py first."
    exit 1
fi

# ---------------------------------------------------------------------------
# Build sorted list of configs
# ---------------------------------------------------------------------------
mapfile -t CONFIG_FILES < <(find "${CONFIGS_DIR}" -name "*.toml" | sort)
N_CONFIGS=${#CONFIG_FILES[@]}

if [ "${N_CONFIGS}" -eq 0 ]; then
    echo "No .toml files found in ${CONFIGS_DIR}"
    exit 1
fi

# ---------------------------------------------------------------------------
# Results directory on scratch
# ---------------------------------------------------------------------------
RESULTS_BASE="/scratch/p311056/zalmoxis_validation/results"

# ---------------------------------------------------------------------------
# If not inside a SLURM job, submit ourselves as an array
# ---------------------------------------------------------------------------
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    echo "Submitting array job for ${SUITE_NAME}: ${N_CONFIGS} configs"
    export SUITE_NAME
    sbatch --array="1-${N_CONFIGS}" \
           --job-name="zv_${SUITE_NAME}" \
           "${BASH_SOURCE[0]}" "${SUITE_NAME}"
    exit 0
fi

# ---------------------------------------------------------------------------
# Inside the array job: set up Habrok environment
# ---------------------------------------------------------------------------

# Load modules (needed for some dependencies)
module load GCC/12.3.0

# Initialize conda (NEVER 'source ~/.bashrc' in SLURM)
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate proteus

# Set environment variables
export ZALMOXIS_ROOT=$HOME/PROTEUS/Zalmoxis
export ZALMOXIS_RESULTS_DIR=/scratch/p311056/zalmoxis_validation/results

# ---------------------------------------------------------------------------
# Run the single config
# ---------------------------------------------------------------------------
TASK_INDEX=$((SLURM_ARRAY_TASK_ID - 1))
if [ "${TASK_INDEX}" -ge "${N_CONFIGS}" ]; then
    echo "Task index ${TASK_INDEX} exceeds number of configs ${N_CONFIGS}"
    exit 1
fi

CONFIG_FILE="${CONFIG_FILES[${TASK_INDEX}]}"
echo "Running config: ${CONFIG_FILE}"
echo "Suite: ${SUITE_NAME}, Task: ${SLURM_ARRAY_TASK_ID}/${N_CONFIGS}"

# Run
python "${SCRIPT_DIR}/run_single.py" "${CONFIG_FILE}"
