#!/bin/bash
# Submit all Zalmoxis validation grid suites to SLURM on Habrok.
#
# Prerequisites:
#     1. Run setup_habrok.sh ONCE (interactively) to prepare the environment.
#     2. Generate configs locally:
#            cd ~/PROTEUS/Zalmoxis
#            python tools/validation_grid/generate_configs.py
#     3. Then run this script:
#            bash tools/validation_grid/submit_all.sh
#
# To copy configs from your local machine to Habrok:
#     scp -r tools/validation_grid/configs/ \
#         p311056@habrok.hpc.rug.nl:~/PROTEUS/Zalmoxis/tools/validation_grid/configs/
#
# Each suite is submitted as a separate SLURM array job. All suites run
# independently (no dependencies between them).
#
# Logs:   /scratch/p311056/zalmoxis_validation/logs/
# Results: /scratch/p311056/zalmoxis_validation/results/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------
if [ ! -d "${SCRIPT_DIR}/configs" ]; then
    echo "ERROR: No configs/ directory found."
    echo "Generate configs first: python tools/validation_grid/generate_configs.py"
    exit 1
fi

# Ensure scratch directories exist
mkdir -p /scratch/p311056/zalmoxis_validation/logs
mkdir -p /scratch/p311056/zalmoxis_validation/results

# ---------------------------------------------------------------------------
# Discover suites (all subdirectories under configs/)
# ---------------------------------------------------------------------------
SUITES=()
for dir in "${SCRIPT_DIR}"/configs/suite_*/; do
    if [ -d "$dir" ]; then
        SUITES+=("$(basename "$dir")")
    fi
done

if [ ${#SUITES[@]} -eq 0 ]; then
    echo "ERROR: No suite directories found under configs/."
    exit 1
fi

echo "Submitting ${#SUITES[@]} validation grid suites..."
echo "Logs:    /scratch/p311056/zalmoxis_validation/logs/"
echo "Results: /scratch/p311056/zalmoxis_validation/results/"
echo ""

for suite in "${SUITES[@]}"; do
    echo "--- ${suite} ---"
    bash "${SCRIPT_DIR}/submit_suite.sh" "${suite}"
    echo ""
done

echo "All suites submitted. Monitor with: squeue -u p311056"
