#!/bin/bash
# v9 with Qwen/Qwen3-0.6B, 5 epochs — same recipe as run_v9_17b.sh but smaller model.
# data_v9 JSONL, all traffic, neg_frac=0.3, cl=4096, lr=2e-5, effective batch=128.
# 0.6B fits cl=4096 with micro_batch=2 (grad_accum=8) like the original v9 run.
set -euo pipefail
cd "$(dirname "$0")"

export MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-0.6B"}
export DATA_PATH=${DATA_PATH:-"data"}                  # placeholder; jsonl below overrides
export TRAIN_JSONL=${TRAIN_JSONL:-"data_v9/train_all.jsonl"}
export EVAL_URA_JSONL=${EVAL_URA_JSONL:-"data_v9/eval_ura.jsonl"}
export EVAL_ALL_JSONL=${EVAL_ALL_JSONL:-"data_v9/eval_all.jsonl"}

export TRAIN_UNTIL=${TRAIN_UNTIL:-"20260416"}
export EVAL_FROM=${EVAL_FROM:-"20260417"}
export URA_FLIGHT=${URA_FLIGHT:-"discover-rk-ura"}
export TRAIN_URA_ONLY=${TRAIN_URA_ONLY:-0}             # match v9: ALL traffic
export NEG_FRAC=${NEG_FRAC:-0.3}                        # match v9
export NEG_RATIO=${NEG_RATIO:-0}
export DISABLE_EARLY_STOP=${DISABLE_EARLY_STOP:-1}

export BATCH_SIZE=${BATCH_SIZE:-128}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-2}          # 0.6B fits cl=4096 with mbs=2
export NUM_EPOCHS=${NUM_EPOCHS:-5}                      # 5 epochs as requested
export LEARNING_RATE=${LEARNING_RATE:-2e-5}
export CUTOFF_LEN=${CUTOFF_LEN:-4096}
export MAX_HISTORY=${MAX_HISTORY:-30}
export OPTIM=${OPTIM:-"adamw_bnb_8bit"}                 # match v9 optim

export OUTPUT_DIR=${OUTPUT_DIR:-"output/v9_Qwen3-0.6B_all_ep${NUM_EPOCHS}"}
export WANDB_PROJECT=${WANDB_PROJECT:-"pointwise_sft"}
export WANDB_RUN_NAME=${WANDB_RUN_NAME:-"$(basename ${OUTPUT_DIR})"}

bash run.sh
