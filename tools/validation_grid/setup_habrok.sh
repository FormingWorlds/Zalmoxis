#!/bin/bash
# One-time interactive setup for the Zalmoxis validation grid on Habrok.
#
# Run this ONCE on a Habrok login node (NOT via SLURM):
#     bash tools/validation_grid/setup_habrok.sh
#
# After running this, generate and copy configs, then submit:
#     # On your LOCAL machine:
#     cd ~/git/PROTEUS/Zalmoxis
#     python tools/validation_grid/generate_configs.py
#     scp -r tools/validation_grid/configs/ \
#         p311056@habrok.hpc.rug.nl:~/PROTEUS/Zalmoxis/tools/validation_grid/configs/
#
#     # On Habrok:
#     cd ~/PROTEUS/Zalmoxis
#     bash tools/validation_grid/submit_all.sh

set -euo pipefail

echo "=== Zalmoxis Habrok setup ==="
echo ""

# ---------------------------------------------------------------------------
# 1. Create scratch directories
# ---------------------------------------------------------------------------
echo "[1/7] Creating scratch directories..."
mkdir -p /scratch/p311056/zalmoxis_validation/logs
mkdir -p /scratch/p311056/zalmoxis_validation/results
echo "  /scratch/p311056/zalmoxis_validation/logs/"
echo "  /scratch/p311056/zalmoxis_validation/results/"

# ---------------------------------------------------------------------------
# 2. Update repository
# ---------------------------------------------------------------------------
echo "[2/7] Updating Zalmoxis repository..."
cd ~/PROTEUS/Zalmoxis
git fetch origin
git checkout tl/multi-material-mixing
git pull origin tl/multi-material-mixing
echo "  Branch: tl/multi-material-mixing (up to date)"

# ---------------------------------------------------------------------------
# 3. Activate environment
# ---------------------------------------------------------------------------
echo "[3/7] Activating conda environment..."
module load GCC/12.3.0
eval "$($HOME/miniforge3/bin/conda shell.bash hook)"
conda activate proteus

# ---------------------------------------------------------------------------
# 4. Install Zalmoxis (editable)
# ---------------------------------------------------------------------------
echo "[4/7] Installing Zalmoxis..."
pip install -e .

# ---------------------------------------------------------------------------
# 5. Download EOS data if not present
# ---------------------------------------------------------------------------
echo "[5/7] Downloading EOS data (if needed)..."
bash src/get_zalmoxis.sh

# ---------------------------------------------------------------------------
# 6. Verify installation
# ---------------------------------------------------------------------------
echo "[6/7] Verifying installation..."
python -c "import zalmoxis; print(f'  Zalmoxis version: {zalmoxis.__version__}')" 2>/dev/null \
    || python -c "import zalmoxis; print('  Zalmoxis imported successfully')"

# ---------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------
echo "[7/7] Setup complete."
echo ""
echo "Next steps:"
echo "  1. On your LOCAL machine, generate and copy configs:"
echo "       cd ~/git/PROTEUS/Zalmoxis"
echo "       python tools/validation_grid/generate_configs.py"
echo "       scp -r tools/validation_grid/configs/ \\"
echo "           p311056@habrok.hpc.rug.nl:~/PROTEUS/Zalmoxis/tools/validation_grid/configs/"
echo ""
echo "  2. On Habrok, submit all suites:"
echo "       cd ~/PROTEUS/Zalmoxis"
echo "       bash tools/validation_grid/submit_all.sh"
echo ""
echo "  3. Monitor jobs:"
echo "       squeue -u p311056"
echo ""
echo "  4. After completion, collect results:"
echo "       python tools/validation_grid/collect_results.py"
