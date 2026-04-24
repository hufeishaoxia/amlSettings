#!/bin/bash
# One-shot: kill any stale runs, then launch v1 0.6B URA neg30 with CUTOFF_LEN=4096
# plus its eval watcher. Logs to logs/ with timestamps.
set -uo pipefail
cd "$(dirname "$0")"

pkill -f 'eval_v1_06b_ura_neg30' 2>/dev/null || true
pkill -f 'run_v1_06b_ura_neg30'  2>/dev/null || true
pkill -f 'torchrun.*train\.py'   2>/dev/null || true
pkill -9 -f 'train\.py'          2>/dev/null || true
sleep 5

rm -rf output/v1_Qwen3-0.6B_ura_neg30_ep3 2>/dev/null
rm -rf output/v1_Qwen3-0.6B_ura_neg30_cl4k_ep3 2>/dev/null

mkdir -p logs
TS=$(date '+%Y%m%d_%H%M%S')
TRAIN_LOG="logs/run_v1_06b_ura_neg30_cl4k_${TS}.log"
EVAL_LOG="logs/eval_v1_06b_ura_neg30_cl4k_${TS}.log"

echo "TRAIN_LOG=${TRAIN_LOG}"
echo "EVAL_LOG=${EVAL_LOG}"

CUTOFF_LEN=4096 MICRO_BATCH_SIZE=1 \
  OUTPUT_DIR=output/v1_Qwen3-0.6B_ura_neg30_cl4k_ep3 \
  WANDB_RUN_NAME=v1_Qwen3-0.6B_ura_neg30_cl4k_ep3 \
  nohup bash run_v1_06b_ura_neg30.sh > "${TRAIN_LOG}" 2>&1 < /dev/null &
echo "TRAIN_PID=$!"

MODEL_DIR=output/v1_Qwen3-0.6B_ura_neg30_cl4k_ep3 MAX_LEN=4096 \
  nohup bash eval_v1_06b_ura_neg30.sh > "${EVAL_LOG}" 2>&1 < /dev/null &
echo "EVAL_PID=$!"

disown -a 2>/dev/null || true
echo "launched"
