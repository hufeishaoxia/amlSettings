#!/bin/bash
# Quick XGBoost baseline + pairwise comparison
set -euo pipefail
cd /home/amluser/amlSettings/SLM

echo "========================================"
echo "=== Step 1: XGBoost Baseline ==="
echo "========================================"
python xgb_baseline.py data_v9 2>&1 | tee /home/amluser/amlSettings/SLM/logs/xgb_baseline_result.txt

echo ""
echo "========================================"
echo "=== Step 2: Pairwise BCE Training ==="
echo "========================================"
NGPU=8 LOSS=bce NUM_NEG=20 EPOCHS=3 BATCH=32 MBS=1 \
  DATA=data_v9 OUTPUT=output/pairwise_bce_v9 \
  EVAL_URA_ONLY=0 \
  bash run_pairwise.sh 2>&1 | tee /home/amluser/amlSettings/SLM/logs/pairwise_bce_result.txt

echo ""
echo "========================================"
echo "=== Done! ==="
echo "========================================"
