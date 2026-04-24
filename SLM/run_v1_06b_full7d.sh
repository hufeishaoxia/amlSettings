#!/bin/bash
# v1 (train.py) — Qwen3-0.6B, FULL traffic (TRAIN_URA_ONLY=0), no neg sampling,
# adamw_torch optimizer, last 7 days of train data (20260410..20260416),
# CUTOFF_LEN=4096, 3 epochs, NO early stopping (no load_best).
set -euo pipefail
cd "$(dirname "$0")"

export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
export DATA_PATH=${DATA_PATH:-"data"}
export TRAIN_BIZDATE_MIN=${TRAIN_BIZDATE_MIN:-"20260410"}   # 7 days: 0410..0416
export TRAIN_UNTIL=${TRAIN_UNTIL:-"20260416"}
export EVAL_FROM=${EVAL_FROM:-"20260417"}
export URA_FLIGHT=${URA_FLIGHT:-"discover-rk-ura"}
export TRAIN_URA_ONLY=${TRAIN_URA_ONLY:-0}     # ALL traffic
export NEG_FRAC=${NEG_FRAC:-0}                  # keep all negs
export NEG_RATIO=${NEG_RATIO:-0}

export BATCH_SIZE=${BATCH_SIZE:-128}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}  # safe for cl=4096 + 0.6B + grad_ckpt
export NUM_EPOCHS=${NUM_EPOCHS:-3}
export LEARNING_RATE=${LEARNING_RATE:-2e-5}
export CUTOFF_LEN=${CUTOFF_LEN:-4096}
export MAX_HISTORY=${MAX_HISTORY:-30}
export OPTIM=${OPTIM:-"adamw_torch"}
export DISABLE_EARLY_STOP=${DISABLE_EARLY_STOP:-1}

export OUTPUT_DIR=${OUTPUT_DIR:-"output/v1_Qwen3-0.6B_full7d_cl4k_ep${NUM_EPOCHS}"}
export WANDB_PROJECT=${WANDB_PROJECT:-"pointwise_sft"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"$(basename ${OUTPUT_DIR})"}

bash run.sh
