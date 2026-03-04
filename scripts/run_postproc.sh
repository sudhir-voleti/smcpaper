#!/bin/bash
# run_postproc.sh - Run RFM-HMM simulation post-processing
# 
# Usage:
#   ./run_postproc.sh [quick|full]
#
# Args:
#   quick - Only scan inventory (fast)
#   full  - Extract all metrics from PKLs (slow, default)

set -e  # Exit on error

# Configuration
ROOT_DIR="/Users/sudhirvoleti/research related/SMC paper Feb2026/march03_simul_full"
OUT_DIR="/Users/sudhirvoleti/research related/SMC paper Feb2026/postproc_results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "RFM-HMM Post-Processing"
echo "=========================================="
echo "Script: $SCRIPT_DIR/simul_postproc.py"
echo "Input:  $ROOT_DIR"
echo "Output: $OUT_DIR"
echo ""

# Check if simul_postproc.py exists
if [ ! -f "$SCRIPT_DIR/simul_postproc.py" ]; then
    echo "ERROR: simul_postproc.py not found in $SCRIPT_DIR"
    echo "Please ensure simul_postproc.py is in the same directory as this script"
    exit 1
fi

# Parse mode
MODE="${1:-full}"

if [ "$MODE" == "quick" ]; then
    echo "Mode: QUICK (inventory only, no PKL loading)"
    QUICK_FLAG="--quick"
elif [ "$MODE" == "full" ]; then
    echo "Mode: FULL (extract all metrics - this will take time)"
    QUICK_FLAG=""
else
    echo "Usage: ./run_postproc.sh [quick|full]"
    exit 1
fi

echo ""
echo "Starting processing..."
echo ""

# Run post-processing
python3 "$SCRIPT_DIR/simul_postproc.py" \
    --root_dir "$ROOT_DIR" \
    --out_dir "$OUT_DIR" \
    $QUICK_FLAG

echo ""
echo "=========================================="
echo "Processing Complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  - $OUT_DIR/pkl_inventory.csv"
echo "  - $OUT_DIR/master_comparison.csv"
echo "  - $OUT_DIR/ablation_comparison.csv"
echo "  - $OUT_DIR/master_comparison.tex"
echo ""
echo "To view results:"
echo "  cat $OUT_DIR/master_comparison.csv"
echo ""


================================================================================
FILE 2: run_postproc.sh
================================================================================
