#!/bin/bash
set -e
cd /scratch/azureml/cr/j/cb7f3b2f13af4de88e98a157ca0e3eaa/exe/wd/amlSettings/SLM

# Kill any leftover training
pkill -f "train.py" 2>/dev/null || true
sleep 3

# Clean up failed output
rm -rf output/v9_Qwen3-0.6B_ura_ep3
rm -rf output/v9_Qwen3-0.6B_all_ep5

echo "===== Start training with train_all ====="
export MODEL_PATH=Qwen/Qwen3-0.6B
export BATCH_SIZE=128
export MICRO_BATCH_SIZE=2
export NUM_EPOCHS=3
export CUTOFF_LEN=4096
export LEARNING_RATE=2e-5
export NEG_FRAC=0.3
export MAX_HISTORY=30
export TRAIN_URA_ONLY=0
export DISABLE_EARLY_STOP=1
export OPTIM=adamw_torch
export TRAIN_JSONL=data_v9/train_all.jsonl
export EVAL_URA_JSONL=data_v9/eval_ura.jsonl
export EVAL_ALL_JSONL=data_v9/eval_all.jsonl
export OUTPUT_DIR=output/v9_Qwen3-0.6B_all_ep5
bash run.sh
