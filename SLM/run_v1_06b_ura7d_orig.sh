#!/usr/bin/env bash
# v1 original train.py (tar version loss) — 7-day URA traffic, 0.6B, 3 epochs
set -euo pipefail

export MODEL_PATH="Qwen/Qwen3-0.6B"
export DATA_PATH="data"
export TRAIN_BIZDATE_MIN="20260410"
export TRAIN_UNTIL="20260416"
export EVAL_FROM="20260417"
export TRAIN_URA_ONLY=1
export DISABLE_EARLY_STOP=1

export BATCH_SIZE=128
export MICRO_BATCH_SIZE=4
export NUM_EPOCHS=3
export CUTOFF_LEN=4096
export LR="2e-5"
export OPTIM="adamw_torch"
export SAMPLE=-1
export EVAL_SAMPLE=-1
export NEG_RATIO=0
export NEG_FRAC=0.3
export MAX_HISTORY=30

# JSONL pre-processed data
export TRAIN_JSONL="data_v8/train_ura.jsonl"
export EVAL_URA_JSONL="data_v8/eval_ura.jsonl"
export EVAL_ALL_JSONL="data_v8/eval_all.jsonl"

export WANDB_MODE=offline
export WANDB_RUN_NAME="v1orig_Qwen3-0.6B_ura7d_cl4k_ep3"
export OUTPUT_DIR="output/v1orig_Qwen3-0.6B_ura7d_cl4k_ep3"

bash run.sh
