#!/bin/bash
# v1 (train.py) — Qwen3-0.6B on full URA traffic, sampling 30% of negatives.
# After training finishes, eval all checkpoints with eval_auc.py.
set -euo pipefail
cd "$(dirname "$0")"

export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
export DATA_PATH=${DATA_PATH:-"data"}
export TRAIN_UNTIL=${TRAIN_UNTIL:-"20260416"}
export EVAL_FROM=${EVAL_FROM:-"20260417"}
export URA_FLIGHT=${URA_FLIGHT:-"discover-rk-ura"}
export TRAIN_URA_ONLY=${TRAIN_URA_ONLY:-1}      # full URA traffic
export NEG_FRAC=${NEG_FRAC:-0.3}                 # keep 30% of negatives
export NEG_RATIO=${NEG_RATIO:-0}

export BATCH_SIZE=${BATCH_SIZE:-128}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}
export NUM_EPOCHS=${NUM_EPOCHS:-3}
export LEARNING_RATE=${LEARNING_RATE:-2e-5}
export CUTOFF_LEN=${CUTOFF_LEN:-2048}
export MAX_HISTORY=${MAX_HISTORY:-30}

export OUTPUT_DIR=${OUTPUT_DIR:-"output/v1_Qwen3-0.6B_ura_neg30_ep${NUM_EPOCHS}"}
export WANDB_PROJECT=${WANDB_PROJECT:-"pointwise_sft"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"$(basename ${OUTPUT_DIR})"}

LOG_DIR="logs"
mkdir -p "${LOG_DIR}"
TS=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="${LOG_DIR}/${WANDB_RUN_NAME}_${TS}.log"

echo "[v1-neg30] out=${OUTPUT_DIR}  neg_frac=${NEG_FRAC}  log=${LOG_FILE}"

bash run.sh 2>&1 | tee "${LOG_FILE}"
