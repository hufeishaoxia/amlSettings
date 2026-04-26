#!/bin/bash
# Reproduce v1 baseline: Qwen3-0.6B, URA only, 3 epochs, direct from parquet
# All defaults from run.sh (CUTOFF_LEN=2048, OPTIM=adamw_torch, no neg_frac)
set -e
cd /scratch/azureml/cr/j/cb7f3b2f13af4de88e98a157ca0e3eaa/exe/wd/amlSettings/SLM

# Kill any leftover training
pkill -f "train.py" 2>/dev/null || true
sleep 3

echo "===== Reproduce v1 baseline ====="
export MODEL_PATH=Qwen/Qwen3-0.6B

# Match v1 exactly: per_device=4, grad_accum=8, world=8 => effective batch=256
export BATCH_SIZE=256
export MICRO_BATCH_SIZE=4

export NUM_EPOCHS=3
export CUTOFF_LEN=2048
export LEARNING_RATE=2e-5
export MAX_HISTORY=30

# v1 defaults: URA only, no neg downsampling, no JSONL (direct parquet)
export TRAIN_URA_ONLY=1
export NEG_FRAC=0
export NEG_RATIO=0
export OPTIM=adamw_torch
export DISABLE_EARLY_STOP=1

# No JSONL — load directly from parquet like v1
export TRAIN_JSONL=""
export EVAL_URA_JSONL=""
export EVAL_ALL_JSONL=""

export OUTPUT_DIR=output/v1_repro_Qwen3-0.6B_ura_ep3

bash run.sh
